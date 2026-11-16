"""Out-of-sample Leave-One-Election-Out Cross-Validation & Uncertainty Calibration.

Evaluates Stage 1 (Kernel, State-Space, Ensemble), Stage 2 (Historical Multinomial-Logit + IPF),
and Stage 3 (Hierarchical Nat + CCAA + Prov Student-t Simulation) across all 5 historical
Congress elections (``2015-12``, ``2016-06``, ``2019-04``, ``2019-11``, ``2023-07`` = 260 races,
1,750 seats) at pre-election horizons ``T-15d``, ``T-6d``, and ``T-1d``.

Calibrates hierarchical Student-t shock scales and Stage 1 ensemble weights empirically
and writes ``data/outputs/backtest_summary.json``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np

from elecciones.config import (
    BACKTEST_DEFAULT_N_SIMS,
    DEFAULT_HYPERPARAMETERS,
    HISTORICAL_ELECTION_DATES,
    SIMULATION_SEED,
    ModelHyperparameters,
    get_elections_dir,
    get_metadata_dir,
    get_outputs_dir,
    get_polls_dir,
)
from elecciones.ingest.elections import load_congress_results
from elecciones.ingest.wikipedia import load_national_polls
from elecciones.models.covariance import (
    HierarchicalCovarianceConfig,
    estimate_empirical_error_hierarchy,
    run_hierarchical_monte_carlo,
)
from elecciones.models.ensemble import combine_stage1_results, run_stage1
from elecciones.models.kernel_aggregator import Stage1Result
from elecciones.models.pollster_ratings import compute_pollster_ratings
from elecciones.models.provincial import Stage2Projection, project_historical_provinces
from elecciones.schemas import (
    BacktestSummary,
    ConstituencyElectionResult,
    ModelBacktestMetrics,
    TierCalibrationBucket,
    classify_seat_tier,
    load_constituencies_catalog,
)


@dataclass(frozen=True, slots=True)
class _CaseCache:
    election_id: str
    horizon_days: int
    actual_by_cpro: dict[str, ConstituencyElectionResult]
    kernel_s1: Stage1Result
    ss_s1: Stage1Result
    kernel_s2: Stage2Projection
    ss_s2: Stage2Projection


@dataclass(frozen=True, slots=True)
class _EvalStats:
    metrics: ModelBacktestMetrics
    tier_counts: dict[str, list[int]]
    nat_vote_abs_errs: list[float]
    nat_seat_abs_errs: list[float]


def _score_simulated_projection(
    proj: Stage2Projection,
    stage1_res: Stage1Result,
    actual_by_cpro: Mapping[str, ConstituencyElectionResult],
    cov_cfg: HierarchicalCovarianceConfig,
    *,
    n_sims: int,
    seed: int,
    hyperparams: ModelHyperparameters,
) -> tuple[
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[float],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    list[int],
    dict[str, list[int]],
]:
    """Run Stage 3 MC on one historical election+horizon and return pointwise diagnostics."""
    sim = run_hierarchical_monte_carlo(
        proj,
        stage1_result=stage1_res,
        config=cov_cfg,
        n_sims=n_sims,
        seed=seed,
        hyperparams=hyperparams,
        n_workers=1,
    )
    party_ids = proj.party_ids
    n_c = len(proj.cpros)
    n_p = len(party_ids)

    # Actual provincial matrices aligned to proj.party_ids
    act_seats_mat = np.zeros((n_c, n_p), dtype=np.int32)
    act_share_mat = np.zeros((n_c, n_p), dtype=np.float64)
    act_votes_tot = np.zeros(n_p, dtype=np.float64)
    tot_valid = 0.0
    for i, cpro in enumerate(proj.cpros):
        ar = actual_by_cpro[cpro]
        val_i = max(1.0, float(ar.validos))
        tot_valid += float(ar.validos)
        for j, pid in enumerate(party_ids):
            v_ij = float(ar.party_votes.get(pid, 0))
            act_seats_mat[i, j] = int(ar.party_seats.get(pid, 0))
            act_share_mat[i, j] = v_ij / val_i
            act_votes_tot[j] += v_ij

    # 1) National vote share MAE (pp) & continuous 80%/95% vote coverage over tracked parties
    vote_errs: list[float] = []
    cov80_nat_votes: list[int] = []
    cov95_nat_votes: list[int] = []
    for pid in stage1_res.parties:
        if pid == "OTROS" or pid not in party_ids:
            continue
        j = party_ids.index(pid)
        act_share = act_votes_tot[j] / max(1.0, tot_valid)
        act_pct = act_share * 100.0
        if act_pct < 0.8:
            continue
        pred_pct = float(stage1_res.mean_shares.get(pid, 0.0)) * 100.0
        vote_errs.append(abs(pred_pct - act_pct))
        nq = np.percentile(sim.national_shares[:, j], [2.5, 10.0, 90.0, 97.5])
        cov80_nat_votes.append(1 if float(nq[1]) <= act_share <= float(nq[2]) else 0)
        cov95_nat_votes.append(1 if float(nq[0]) <= act_share <= float(nq[3]) else 0)

    # 2) Provincial vote MAE/coverage, seat MAE/coverage, and slot Brier/LogScore/Tier calibration
    briers: list[float] = []
    log_losses: list[float] = []
    prov_vote_errs: list[float] = []
    cov80_prov_votes: list[int] = []
    cov95_prov_votes: list[int] = []
    prov_seat_errs: list[float] = []
    cov80_prov: list[int] = []
    cov95_prov: list[int] = []
    tier_counts: dict[str, list[int]] = {
        "seguro": [0, 0],
        "probable": [0, 0],
        "en_juego": [0, 0],
        "remoto": [0, 0],
    }

    prov_means = sim.prov_seats.mean(axis=0)
    prov_q = np.percentile(sim.prov_seats, [2.5, 10.0, 90.0, 97.5], axis=0, method="nearest")

    for i in range(n_c):
        m_c = int(proj.seats_vec[i])
        for j in range(n_p):
            if not proj.active_mask[i, j]:
                continue
            pid = party_ids[j]
            if pid != "OTROS":
                act_sv = float(act_share_mat[i, j])
                pred_sv = float(sim.prov_share_mean[i, j])
                if act_sv >= 0.008 or pred_sv >= 0.008:
                    prov_vote_errs.append(abs(pred_sv - act_sv) * 100.0)
                    p10_v = float(sim.prov_share_p10[i, j])
                    p90_v = float(sim.prov_share_p90[i, j])
                    p25_v = float(sim.prov_share_p2_5[i, j])
                    p97_v = float(sim.prov_share_p97_5[i, j])
                    cov80_prov_votes.append(1 if p10_v <= act_sv <= p90_v else 0)
                    cov95_prov_votes.append(1 if p25_v <= act_sv <= p97_v else 0)

            act_s = int(act_seats_mat[i, j])
            pred_m = float(prov_means[i, j])
            lo95 = int(prov_q[0, i, j])
            lo80 = int(prov_q[1, i, j])
            hi80 = int(prov_q[2, i, j])
            hi95 = int(prov_q[3, i, j])

            # Only score competitive/relevant provincial slots so zero lists don't skew MAE/cov
            if act_s > 0 or hi95 > 0 or pred_m >= 0.08:
                prov_seat_errs.append(abs(pred_m - float(act_s)))
                cov80_prov.append(1 if lo80 <= act_s <= hi80 else 0)
                cov95_prov.append(1 if lo95 <= act_s <= hi95 else 0)

            # Seat-slot exceedance scoring (for slots with >= 1% chance or actually won)
            col_draws = sim.prov_seats[:, i, j]
            for k in range(1, m_c + 1):
                p_exc = float(np.mean(col_draws >= k))
                if p_exc < 0.01 and act_s < k:
                    continue
                y = 1.0 if act_s >= k else 0.0
                briers.append((p_exc - y) ** 2)
                p_clip = max(1e-4, min(1.0 - 1e-4, p_exc))
                log_losses.append(-(y * math.log(p_clip) + (1.0 - y) * math.log(1.0 - p_clip)))

                tier = classify_seat_tier(p_exc)
                if tier is not None and tier in tier_counts:
                    tier_counts[tier][0] += 1
                    if act_s >= k:
                        tier_counts[tier][1] += 1

    # 3) National seat MAE and 80%/95% coverage over competitive national parties
    act_nat_seats = act_seats_mat.sum(axis=0)
    nat_means = sim.national_seats.mean(axis=0)
    nat_q = np.percentile(sim.national_seats, [2.5, 10.0, 90.0, 97.5], axis=0, method="nearest")
    nat_seat_errs: list[float] = []
    cov80_nat: list[int] = []
    cov95_nat: list[int] = []
    for j in range(n_p):
        act_ns = int(act_nat_seats[j])
        pred_nm = float(nat_means[j])
        lo95 = int(nat_q[0, j])
        lo80 = int(nat_q[1, j])
        hi80 = int(nat_q[2, j])
        hi95 = int(nat_q[3, j])
        if act_ns >= 1 or hi95 >= 1 or pred_nm >= 0.4:
            nat_seat_errs.append(abs(pred_nm - float(act_ns)))
            cov80_nat.append(1 if lo80 <= act_ns <= hi80 else 0)
            cov95_nat.append(1 if lo95 <= act_ns <= hi95 else 0)

    return (
        briers,
        log_losses,
        vote_errs,
        prov_vote_errs,
        prov_seat_errs,
        nat_seat_errs,
        cov80_prov,
        cov95_prov,
        cov80_nat,
        cov95_nat,
        cov80_prov_votes,
        cov95_prov_votes,
        cov80_nat_votes,
        cov95_nat_votes,
        tier_counts,
    )


def _aggregate_metrics(
    briers: Sequence[float],
    log_losses: Sequence[float],
    vote_errs: Sequence[float],
    prov_vote_errs: Sequence[float],
    prov_seat_errs: Sequence[float],
    nat_seat_errs: Sequence[float],
    cov80_prov: Sequence[int],
    cov95_prov: Sequence[int],
    cov80_nat: Sequence[int],
    cov95_nat: Sequence[int],
    cov80_prov_votes: Sequence[int],
    cov95_prov_votes: Sequence[int],
    cov80_nat_votes: Sequence[int],
    cov95_nat_votes: Sequence[int],
) -> ModelBacktestMetrics:
    def _mean(xs: Sequence[float | int], default: float = 0.0) -> float:
        return float(sum(xs) / len(xs)) if xs else default

    return ModelBacktestMetrics(
        brier_score=round(_mean(briers), 6),
        log_score=round(_mean(log_losses), 6),
        vote_mae_national=round(_mean(vote_errs), 4),
        seat_mae_provincial=round(_mean(prov_seat_errs), 4),
        seat_mae_national=round(_mean(nat_seat_errs), 4),
        coverage_80_provincial_seats=round(_mean(cov80_prov, 0.8), 4),
        coverage_95_provincial_seats=round(_mean(cov95_prov, 0.95), 4),
        coverage_80_national_seats=round(_mean(cov80_nat, 0.8), 4),
        coverage_95_national_seats=round(_mean(cov95_nat, 0.95), 4),
        vote_mae_provincial=round(_mean(prov_vote_errs), 4),
        coverage_80_provincial_votes=round(_mean(cov80_prov_votes, 0.8), 4),
        coverage_95_provincial_votes=round(_mean(cov95_prov_votes, 0.95), 4),
        coverage_80_national_votes=round(_mean(cov80_nat_votes, 0.8), 4),
        coverage_95_national_votes=round(_mean(cov95_nat_votes, 0.95), 4),
    )


def _evaluate_suite(
    cases: Sequence[_CaseCache],
    model_name: str,
    cov_cfg: HierarchicalCovarianceConfig,
    hyperparams: ModelHyperparameters,
    *,
    n_sims: int,
    seed: int,
) -> tuple[
    ModelBacktestMetrics,
    dict[str, ModelBacktestMetrics],
    dict[str, TierCalibrationBucket],
    dict[str, dict[str, float]],
]:
    """Evaluate a complete model across all ``(election, horizon)`` cached cases."""
    n_metric_lists = 14
    all_lists: tuple[list[Any], ...] = tuple([] for _ in range(n_metric_lists))
    by_h: dict[int, tuple[list[Any], ...]] = {}
    by_e: dict[str, tuple[list[Any], ...]] = {}
    tier_tot: dict[str, list[int]] = {
        "seguro": [0, 0],
        "probable": [0, 0],
        "en_juego": [0, 0],
        "remoto": [0, 0],
    }

    for idx, c in enumerate(cases):
        if model_name == "kernel":
            s1, s2 = c.kernel_s1, c.kernel_s2
        elif model_name == "state_space":
            s1, s2 = c.ss_s1, c.ss_s2
        else:
            s1 = combine_stage1_results(
                c.kernel_s1,
                c.ss_s1,
                w_kernel=hyperparams.ensemble_weight_kernel,
                w_state_space=hyperparams.ensemble_weight_state_space,
            )
            # Linear convex combination on simplex preserves exact mask & sum
            mixed_mat = (
                hyperparams.ensemble_weight_kernel * c.kernel_s2.final_matrix
                + hyperparams.ensemble_weight_state_space * c.ss_s2.final_matrix
            )
            mixed_pre = (
                hyperparams.ensemble_weight_kernel * c.kernel_s2.pre_shrinkage_matrix
                + hyperparams.ensemble_weight_state_space * c.ss_s2.pre_shrinkage_matrix
            )
            s2 = replace(
                c.kernel_s2,
                stage1_targets=s1.mean_shares,
                pre_shrinkage_matrix=mixed_pre,
                final_matrix=mixed_mat,
            )

        res = _score_simulated_projection(
            s2,
            s1,
            c.actual_by_cpro,
            cov_cfg,
            n_sims=n_sims,
            seed=seed + idx * 131 + (17 if model_name == "ensemble" else 0),
            hyperparams=hyperparams,
        )
        h_lists = by_h.setdefault(c.horizon_days, tuple([] for _ in range(n_metric_lists)))
        e_lists = by_e.setdefault(c.election_id, tuple([] for _ in range(n_metric_lists)))
        for k in range(n_metric_lists):
            all_lists[k].extend(res[k])
            h_lists[k].extend(res[k])
            e_lists[k].extend(res[k])
        for tier_k, pair in res[14].items():
            tier_tot[tier_k][0] += pair[0]
            tier_tot[tier_k][1] += pair[1]

    overall = _aggregate_metrics(*all_lists)
    per_h = {f"T-{h}d": _aggregate_metrics(*lst) for h, lst in sorted(by_h.items(), reverse=True)}
    per_el: dict[str, dict[str, float]] = {}
    for eid, lst in by_e.items():
        m = _aggregate_metrics(*lst)
        per_el[eid] = {
            "brier_score": m.brier_score,
            "log_score": m.log_score,
            "vote_mae_national": m.vote_mae_national,
            "vote_mae_provincial": m.vote_mae_provincial,
            "seat_mae_provincial": m.seat_mae_provincial,
            "seat_mae_national": m.seat_mae_national,
            "coverage_80_provincial_seats": m.coverage_80_provincial_seats,
            "coverage_95_provincial_seats": m.coverage_95_provincial_seats,
            "coverage_80_national_seats": m.coverage_80_national_seats,
            "coverage_95_national_seats": m.coverage_95_national_seats,
            "coverage_80_provincial_votes": m.coverage_80_provincial_votes,
            "coverage_95_provincial_votes": m.coverage_95_provincial_votes,
            "coverage_80_national_votes": m.coverage_80_national_votes,
            "coverage_95_national_votes": m.coverage_95_national_votes,
        }

    tier_buckets: dict[str, TierCalibrationBucket] = {}
    for t_name, (tot, hits) in tier_tot.items():
        hr = float(hits) / float(tot) if tot > 0 else 0.0
        tier_buckets[t_name] = TierCalibrationBucket(
            total=int(tot),
            hits=int(hits),
            hit_rate=round(hr, 4),
        )
    return overall, per_h, tier_buckets, per_el


def _calibrate_ensemble_and_scales(
    cases: Sequence[_CaseCache],
    hyperparams: ModelHyperparameters,
    *,
    seed: int,
) -> tuple[ModelHyperparameters, HierarchicalCovarianceConfig, dict[str, float]]:
    """Empirically calibrate ensemble weights and Student-t hierarchical scales via LOOCV."""
    # Step 1: Select optimal Stage 1 ensemble blend weight from vote MSE on T-6d/T-1d
    mid_late = [c for c in cases if c.horizon_days in {6, 1}] or list(cases)

    def _vote_mse(w_k: float) -> float:
        errs: list[float] = []
        w_s = 1.0 - w_k
        for c in mid_late:
            act_map = c.actual_by_cpro
            tot_val = sum(float(r.validos) for r in act_map.values())
            for p in c.kernel_s1.parties:
                if p == "OTROS":
                    continue
                act_s = sum(float(r.party_votes.get(p, 0)) for r in act_map.values()) / max(
                    1.0, tot_val
                )
                if act_s < 0.01:
                    continue
                pred_s = w_k * c.kernel_s1.mean_shares.get(
                    p, 0.0
                ) + w_s * c.ss_s1.mean_shares.get(p, 0.0)
                errs.append((pred_s - act_s) ** 2)
        return float(sum(errs) / max(1, len(errs)))

    best_wk = hyperparams.ensemble_weight_kernel
    best_mse = float("inf")
    for cand_wk in (0.05, 0.10, 0.15, 0.20, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75):
        mse = _vote_mse(cand_wk)
        if mse < best_mse:
            best_mse = mse
            best_wk = cand_wk
    cal_hp = replace(
        hyperparams,
        ensemble_weight_kernel=round(best_wk, 2),
        ensemble_weight_state_space=round(1.0 - best_wk, 2),
    )

    # Step 2: Estimate raw empirical Nat / CCAA / Prov log-ratio residual hierarchy
    t6_cases = [c for c in cases if c.horizon_days == min(c.horizon_days for c in cases)]
    pairs: list[tuple[Stage2Projection, Mapping[str, Any]]] = []
    for c in t6_cases:
        mixed_mat = (
            cal_hp.ensemble_weight_kernel * c.kernel_s2.final_matrix
            + cal_hp.ensemble_weight_state_space * c.ss_s2.final_matrix
        )
        pairs.append((replace(c.kernel_s2, final_matrix=mixed_mat), c.actual_by_cpro))
    emp_hier = estimate_empirical_error_hierarchy(pairs)

    # Step 3: Search decoupled National vs Local Student-t scales targeting continuous coverage
    base_nat = max(0.06, min(0.14, float(emp_hier["nat_mad"]) * 0.90))
    base_ccaa = max(0.04, min(0.10, float(emp_hier["ccaa_mad"]) * 0.95))
    base_prov = max(0.035, min(0.08, float(emp_hier["prov_mad"]) * 0.95))

    best_score = float("inf")
    best_cfg = HierarchicalCovarianceConfig(
        student_t_df=5.0,
        nat_scale=round(base_nat, 4),
        ccaa_scale=round(base_ccaa, 4),
        prov_scale=round(base_prov, 4),
    )

    cal_cases = [c for c in cases if c.horizon_days in {6, 1}] or list(cases)
    for nu in (4.5, 5.5):
        for nat_mult in (0.70, 0.85, 1.00, 1.15):
            for local_mult in (0.65, 0.85, 1.05, 1.25):
                cand = HierarchicalCovarianceConfig(
                    student_t_df=nu,
                    nat_scale=round(base_nat * nat_mult, 4),
                    ccaa_scale=round(base_ccaa * local_mult, 4),
                    prov_scale=round(base_prov * local_mult, 4),
                    bloc_right_scale=round(0.025 * nat_mult, 4),
                    bloc_left_scale=round(0.025 * nat_mult, 4),
                )
                m, _, _, _ = _evaluate_suite(
                    cal_cases,
                    "ensemble",
                    cand,
                    cal_hp,
                    n_sims=140,
                    seed=seed,
                )
                # Proper scoring + continuous 80%/95% vote & national seat coverage targets
                cov_pen = (
                    3.0 * ((m.coverage_80_provincial_votes - 0.80) ** 2)
                    + 3.5 * ((m.coverage_95_provincial_votes - 0.95) ** 2)
                    + 2.0 * ((m.coverage_80_national_votes - 0.80) ** 2)
                    + 2.0 * ((m.coverage_95_national_votes - 0.95) ** 2)
                    + 1.5 * ((m.coverage_80_national_seats - 0.80) ** 2)
                    + 1.5 * ((m.coverage_95_national_seats - 0.95) ** 2)
                )
                obj = m.brier_score + 0.12 * m.log_score + cov_pen
                if obj < best_score:
                    best_score = obj
                    best_cfg = cand

    return cal_hp, best_cfg, emp_hier


def run_backtest(
    horizons: Sequence[int] = (15, 6, 1),
    *,
    n_sims: int = BACKTEST_DEFAULT_N_SIMS,
    seed: int = SIMULATION_SEED,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
    polls_dir: Path | None = None,
    elections_dir: Path | None = None,
    metadata_dir: Path | None = None,
    output_path: Path | None = None,
    write_output: bool = True,
) -> BacktestSummary:
    """Execute strict LOOCV backtest across 2015-D .. 2023-J (260 races) and save summary."""
    polls_root = polls_dir or get_polls_dir()
    elec_root = elections_dir or get_elections_dir()
    meta_root = metadata_dir or get_metadata_dir()
    consts = load_constituencies_catalog(meta_root / "constituencies_2026.json")

    hist_polls = load_national_polls(polls_root / "national_2015_2023.csv")
    sorted_horizons = sorted({int(h) for h in horizons if int(h) >= 0}, reverse=True)
    if not sorted_horizons:
        raise ValueError("horizons must contain at least one non-negative integer")

    cached_cases: list[_CaseCache] = []
    election_ids = tuple(HISTORICAL_ELECTION_DATES.keys())

    for eid in election_ids:
        edate = HISTORICAL_ELECTION_DATES[eid]
        # Strictly causal ratings: exclude held-out eid, all future elections, and 2026 polls
        causal_excluded = [e for e, d in HISTORICAL_ELECTION_DATES.items() if d >= edate]
        ratings_eid = compute_pollster_ratings(
            historical_polls=hist_polls,
            polls_2026=[],
            elections_dir=elec_root,
            polls_dir=polls_root,
            hyperparams=hyperparams,
            exclude_election_ids=causal_excluded,
        )
        epolls = [p for p in hist_polls if p.election_id == eid]
        act_rows = {
            r.cpro: r for r in load_congress_results(eid, elections_dir=elec_root)
        }

        for h in sorted_horizons:
            as_of = edate - timedelta(days=h)
            k_s1 = run_stage1(
                epolls,
                as_of,
                engine="kernel",
                election_id=eid,
                target_election_date=edate,
                ratings=ratings_eid,
                hyperparams=hyperparams,
                exclude_cis_headline=False,
                compute_cis_sensitivity=False,
            )
            ss_s1 = run_stage1(
                epolls,
                as_of,
                engine="state_space",
                election_id=eid,
                target_election_date=edate,
                ratings=ratings_eid,
                tracked_parties=tuple(p for p in k_s1.parties if p != "OTROS"),
                hyperparams=hyperparams,
                exclude_cis_headline=False,
                compute_cis_sensitivity=False,
            )
            k_s2 = project_historical_provinces(
                eid,
                k_s1.mean_shares,
                as_of,
                constituencies=consts,
                hyperparams=hyperparams,
                elections_dir=elec_root,
                metadata_dir=meta_root,
            )
            ss_s2 = project_historical_provinces(
                eid,
                ss_s1.mean_shares,
                as_of,
                constituencies=consts,
                hyperparams=hyperparams,
                elections_dir=elec_root,
                metadata_dir=meta_root,
            )
            cached_cases.append(
                _CaseCache(
                    election_id=eid,
                    horizon_days=h,
                    actual_by_cpro=act_rows,
                    kernel_s1=k_s1,
                    ss_s1=ss_s1,
                    kernel_s2=k_s2,
                    ss_s2=ss_s2,
                )
            )

    cal_hp, cal_cov, emp_hier = _calibrate_ensemble_and_scales(
        cached_cases,
        hyperparams,
        seed=seed,
    )

    k_all, k_h, _, _ = _evaluate_suite(
        cached_cases, "kernel", cal_cov, cal_hp, n_sims=n_sims, seed=seed + 101
    )
    ss_all, ss_h, _, _ = _evaluate_suite(
        cached_cases, "state_space", cal_cov, cal_hp, n_sims=n_sims, seed=seed + 202
    )
    en_all, en_h, tier_cal, per_el = _evaluate_suite(
        cached_cases, "ensemble", cal_cov, cal_hp, n_sims=n_sims, seed=seed + 303
    )

    horizons_block: dict[str, Any] = {}
    for h_key in k_h:
        horizons_block[h_key] = {
            "kernel": k_h[h_key].to_dict(),
            "state_space": ss_h[h_key].to_dict(),
            "ensemble": en_h[h_key].to_dict(),
        }

    cal_params: dict[str, Any] = {
        "ensemble_weight_kernel": float(cal_hp.ensemble_weight_kernel),
        "ensemble_weight_state_space": float(cal_hp.ensemble_weight_state_space),
        "covariance": cal_cov.to_dict(),
        "empirical_error_hierarchy": emp_hier,
        "n_sims_per_race": int(n_sims),
    }

    summary = BacktestSummary(
        elections=election_ids,
        total_constituency_races=len(election_ids) * 52,
        total_seats=len(election_ids) * 350,
        horizons=horizons_block,
        models={
            "kernel": k_all,
            "state_space": ss_all,
            "ensemble": en_all,
        },
        tier_calibration=tier_cal,
        calibrated_parameters=cal_params,
        per_election=per_el,
    )

    if write_output:
        target = output_path or (get_outputs_dir() / "backtest_summary.json")
        summary.save_json(target)

    return summary


def load_calibrated_config(
    backtest_summary_path: Path | None = None,
    *,
    base_hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
) -> tuple[ModelHyperparameters, HierarchicalCovarianceConfig]:
    """Load backtest-calibrated hyperparameters & Stage 3 scales if JSON exists."""
    target = backtest_summary_path or (get_outputs_dir() / "backtest_summary.json")
    if not target.is_file():
        return base_hyperparams, HierarchicalCovarianceConfig()

    summary = BacktestSummary.load_json(target)
    cp = summary.calibrated_parameters
    wk = float(cp.get("ensemble_weight_kernel", base_hyperparams.ensemble_weight_kernel))
    ws = float(
        cp.get("ensemble_weight_state_space", base_hyperparams.ensemble_weight_state_space)
    )
    cov_raw = cp.get("covariance", {})
    cov_cfg = HierarchicalCovarianceConfig.from_dict(cov_raw)
    hp = replace(
        base_hyperparams,
        ensemble_weight_kernel=wk,
        ensemble_weight_state_space=ws,
        student_t_df=cov_cfg.student_t_df,
    )
    return hp, cov_cfg
