"""Stage 3B: Full 52-Constituency 50,000-Draw Monte Carlo Simulator & Forecast Bundle Builder."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from elecciones.backtest.runner import load_calibrated_config
from elecciones.config import (
    BLACKOUT_CUTOFF_DATE_2026,
    DEFAULT_HYPERPARAMETERS,
    DEFAULT_N_SIMS,
    ELECTION_DATE_2026,
    MADRID_TZ,
    SIMULATION_SEED,
    get_elections_dir,
    get_metadata_dir,
    get_outputs_dir,
    get_polls_dir,
    is_in_blackout_window,
)
from elecciones.ingest.cross_check import apply_manual_poll_overlay, load_manual_overlays
from elecciones.ingest.wikipedia import load_national_polls
from elecciones.models.covariance import HierarchicalCovarianceConfig, run_hierarchical_monte_carlo
from elecciones.models.dhondt import allocate_dhondt_loreg
from elecciones.models.ensemble import run_stage1
from elecciones.models.investiture import compute_investiture_summary
from elecciones.models.pollster_ratings import compute_pollster_ratings
from elecciones.models.provincial import project_2026_provinces
from elecciones.schemas import (
    AUTONOMOUS_CITY_CPROS,
    AllocationInput,
    ConstituencyForecast,
    ForecastBundle,
    ForecastEngine,
    ForecastMetadata,
    LastSeatBattle,
    PartyConstituencyProjection,
    PartyNationalSeats,
    SeatSlot,
    VoteShareInterval,
    classify_seat_tier,
    load_constituencies_catalog,
    load_parties_catalog,
)


def _resolve_as_of_date(as_of: date | None) -> date:
    if as_of is not None:
        return as_of
    today_madrid = datetime.now(MADRID_TZ).date()
    return min(today_madrid, BLACKOUT_CUTOFF_DATE_2026)


def _ordered_quantiles_float(arr: np.ndarray) -> tuple[float, float, float, float, float, float]:
    """Compute bounded monotonic ``(mean, median, p5, p10, p90, p95)`` floats in ``[0, 1]``."""
    mean_v = min(1.0, max(0.0, round(float(np.mean(arr)), 6)))
    p5, p10, med, p90, p95 = (
        min(1.0, max(0.0, round(float(x), 6)))
        for x in np.percentile(arr, [5.0, 10.0, 50.0, 90.0, 95.0])
    )
    p10 = max(p5, p10)
    med = max(p10, med)
    p90 = max(med, p90)
    p95 = max(p90, p95)
    return mean_v, med, p5, p10, p90, p95


def _ordered_quantiles_int(arr: np.ndarray) -> tuple[float, int, int, int, int, int]:
    """Compute exact integer quantiles ``(mean, median, p5, p10, p90, p95)``."""
    mean_s = round(float(np.mean(arr)), 3)
    p5, p10, med, p90, p95 = (
        max(0, int(x))
        for x in np.percentile(arr, [5.0, 10.0, 50.0, 90.0, 95.0], method="nearest")
    )
    p10 = max(p5, p10)
    med = max(p10, med)
    p90 = max(med, p90)
    p95 = max(p90, p95)
    return mean_s, med, p5, p10, p90, p95


def run_forecast_simulation(
    as_of: date | None = None,
    *,
    engine: ForecastEngine = "ensemble",
    exclude_cis_headline: bool = False,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = SIMULATION_SEED,
    manual_polls_path: Path | None = None,
    backtest_summary_path: Path | None = None,
    output_path: Path | None = None,
    polls_dir: Path | None = None,
    elections_dir: Path | None = None,
    metadata_dir: Path | None = None,
    cov_config_override: HierarchicalCovarianceConfig | None = None,
    write_output: bool = True,
) -> ForecastBundle:
    """Execute full Stages 1–4 pipeline across 52 constituencies and save ``forecast_29n.json``."""
    eff_as_of = _resolve_as_of_date(as_of)
    polls_root = polls_dir or get_polls_dir()
    elec_root = elections_dir or get_elections_dir()
    meta_root = metadata_dir or get_metadata_dir()

    hyperparams, cal_cov = load_calibrated_config(
        backtest_summary_path,
        base_hyperparams=DEFAULT_HYPERPARAMETERS,
    )
    cov_cfg = cov_config_override or cal_cov

    consts_cat = load_constituencies_catalog(meta_root / "constituencies_2026.json")
    parties_cat = load_parties_catalog(meta_root / "parties_2026.json")
    consts_map = consts_cat.by_cpro()

    base_polls_26 = load_national_polls(polls_root / "national_2026.csv")
    overlays = load_manual_overlays(manual_polls_path or (polls_root / "manual_polls_2026.csv"))
    polls_26 = (
        apply_manual_poll_overlay(base_polls_26, overlays) if overlays else list(base_polls_26)
    )

    ratings = compute_pollster_ratings(
        polls_2026=polls_26,
        elections_dir=elec_root,
        polls_dir=polls_root,
        hyperparams=hyperparams,
        as_of=eff_as_of,
    )

    # Stage 1: National Latent Support
    s1 = run_stage1(
        polls_26,
        eff_as_of,
        engine=engine,
        election_id="2026-11",
        target_election_date=ELECTION_DATE_2026,
        ratings=ratings,
        hyperparams=hyperparams,
        exclude_cis_headline=exclude_cis_headline,
        compute_cis_sensitivity=True,
    )

    # Stage 2: Masked 52-Province Projection & Regional Poll Shrinkage
    s2 = project_2026_provinces(
        s1.mean_shares,
        eff_as_of,
        blank_nat=s1.blank_share,
        constituencies=consts_cat,
        parties=parties_cat,
        ratings=ratings,
        hyperparams=hyperparams,
        elections_dir=elec_root,
        polls_dir=polls_root,
        metadata_dir=meta_root,
    )

    # Stage 3: Hierarchical Student-t Monte Carlo & Vectorized LOREG D'Hondt
    sim = run_hierarchical_monte_carlo(
        s2,
        stage1_result=s1,
        config=cov_cfg,
        n_sims=n_sims,
        seed=seed,
        parties_catalog=parties_cat,
        hyperparams=hyperparams,
    )

    # Stage 4: Art. 99 CE 1st-ballot & 2nd-ballot Investiture Probabilities
    investiture = compute_investiture_summary(sim.national_seats, sim.party_ids)

    # Aggregate Provincial Forecasts, Tier Slots, & Deterministic Mean D'Hondt Inspection
    party_ids = sim.party_ids
    n_p = len(party_ids)
    nat_tier_counts: dict[str, dict[str, int]] = {
        pid: {"seguros": 0, "probables": 0, "en_juego": 0} for pid in party_ids
    }
    tier_totals = {"seguros": 0, "probables": 0, "en_juego": 0, "remotos": 0}
    const_forecasts: list[ConstituencyForecast] = []
    battlegrounds: list[dict[str, Any]] = []

    for i, cpro in enumerate(s2.cpros):
        con = consts_map[cpro]
        dist = s2.districts[cpro]
        m_c = con.seats_2026
        ranks = np.arange(1, m_c + 1, dtype=np.int16)

        seat_slots: list[SeatSlot] = []
        party_projs: dict[str, PartyConstituencyProjection] = {}
        dist_en_juego = 0

        for j, pid in enumerate(party_ids):
            if not s2.active_mask[i, j]:
                continue
            col_seats = sim.prov_seats[:, i, j]
            probs_k = np.mean(col_seats[:, None] >= ranks[None, :], axis=0)

            p_seg = 0
            p_prob = 0
            p_ej = 0
            for k_idx, raw_p in enumerate(probs_k, start=1):
                p_clean = min(1.0, max(0.0, round(float(raw_p), 6)))
                tier = classify_seat_tier(p_clean)
                if tier is None:
                    continue
                seat_slots.append(SeatSlot(party=pid, rank=k_idx, prob=p_clean, tier=tier))
                if tier == "seguro":
                    p_seg += 1
                    nat_tier_counts[pid]["seguros"] += 1
                    tier_totals["seguros"] += 1
                elif tier == "probable":
                    p_prob += 1
                    nat_tier_counts[pid]["probables"] += 1
                    tier_totals["probables"] += 1
                elif tier == "en_juego":
                    p_ej += 1
                    dist_en_juego += 1
                    nat_tier_counts[pid]["en_juego"] += 1
                    tier_totals["en_juego"] += 1
                elif tier == "remoto":
                    tier_totals["remotos"] += 1

            s_mean, s_med, s_p5, s_p10, s_p90, s_p95 = _ordered_quantiles_int(col_seats)
            vs_m = min(1.0, max(0.0, round(float(sim.prov_share_mean[i, j]), 6)))
            vs_10 = min(vs_m, max(0.0, round(float(sim.prov_share_p10[i, j]), 6)))
            vs_90 = min(1.0, max(vs_m, round(float(sim.prov_share_p90[i, j]), 6)))
            p_ge_1 = min(1.0, max(0.0, round(float(probs_k[0]), 6))) if probs_k.size > 0 else 0.0

            party_projs[pid] = PartyConstituencyProjection(
                vote_share_mean=vs_m,
                vote_share_p10=vs_10,
                vote_share_p90=vs_90,
                seats_mean=s_mean,
                seats_median=s_med,
                seats_p5=s_p5,
                seats_p10=s_p10,
                seats_p90=s_p90,
                seats_p95=s_p95,
                seguros=p_seg,
                probables=p_prob,
                en_juego=p_ej,
                prob_at_least_one=p_ge_1,
            )

        seat_slots.sort(key=lambda s: (-s.prob, s.party, s.rank))

        # Expected valid vote counts on deterministic final_shares -> realistic gap_abs & quotients
        exp_v = float(max(1, dist.validos_prior))
        active_votes = {
            party_ids[j]: float(s2.final_matrix[i, j]) * exp_v
            for j in range(n_p)
            if s2.active_mask[i, j] and s2.final_matrix[i, j] > 0.0
        }
        if not active_votes:
            active_votes = {"PSOE": exp_v * 0.5, "PP": exp_v * 0.5}
        alloc_det = allocate_dhondt_loreg(
            AllocationInput(
                seats=m_c,
                party_votes=active_votes,
                blank_votes=float(dist.blank_share) * exp_v,
                is_autonomous_city=(cpro in AUTONOMOUS_CITY_CPROS),
                cpro=cpro,
            ),
            include_quotients=True,
        )
        top_fallback = max(active_votes, key=lambda k: active_votes[k])
        battle = alloc_det.last_seat_battle or LastSeatBattle(
            holder=top_fallback,
            cutoff_quotient=float(alloc_det.cutoff_quotient or 0.0),
            runners_up=(),
        )
        dq = {
            k: [round(float(val), 2) for val in vals]
            for k, vals in (alloc_det.quotients or {}).items()
        }

        const_forecasts.append(
            ConstituencyForecast(
                cpro=cpro,
                name=con.name,
                codauto=con.codauto,
                seats_2026=m_c,
                turnout_mean=round(dist.turnout_mean, 5),
                blank_share=dist.blank_share,
                pre_shrinkage_ipf_shares=dist.pre_shrinkage_ipf_shares,
                final_shares=dist.final_shares,
                party_projections=party_projs,
                seat_slots=tuple(seat_slots),
                dhondt_quotients=dq,
                last_seat_battle=battle,
                pre_shrinkage_blank_share=dist.pre_shrinkage_blank_share,
            )
        )

        if battle.runners_up:
            lead_challenger = battle.runners_up[0]
            battlegrounds.append(
                {
                    "cpro": cpro,
                    "name": con.name,
                    "codauto": con.codauto,
                    "ccaa_name": con.ccaa_name,
                    "seats_2026": m_c,
                    "holder": battle.holder,
                    "challenger": lead_challenger.party,
                    "vote_gap_pp": round(float(lead_challenger.vote_gap_pp), 4),
                    "vote_gap_abs": int(lead_challenger.vote_gap_abs),
                    "cutoff_quotient": round(float(battle.cutoff_quotient), 2),
                    "en_juego_slots": int(dist_en_juego),
                    "runners_up": [r.to_dict() for r in battle.runners_up],
                }
            )

    battlegrounds.sort(key=lambda b: (b["vote_gap_pp"], -b["en_juego_slots"], b["cpro"]))

    # Build National Summary Block (sorted descending by expected seats & share)
    nat_mean_seats = sim.national_seats.mean(axis=0)
    nat_mean_shares = sim.national_shares.mean(axis=0)
    ordered_idx = sorted(
        range(n_p),
        key=lambda j: (-float(nat_mean_seats[j]), -float(nat_mean_shares[j]), party_ids[j]),
    )

    vote_shares_block: dict[str, Any] = {}
    seats_block: dict[str, Any] = {}
    for j in ordered_idx:
        pid = party_ids[j]
        vm, vmed, vp5, vp10, vp90, vp95 = _ordered_quantiles_float(sim.national_shares[:, j])
        vote_shares_block[pid] = VoteShareInterval(
            mean=vm, median=vmed, p5=vp5, p10=vp10, p90=vp90, p95=vp95
        ).to_dict()

        sm, smed, sp5, sp10, sp90, sp95 = _ordered_quantiles_int(sim.national_seats[:, j])
        t_counts = nat_tier_counts[pid]
        seats_block[pid] = PartyNationalSeats(
            mean=sm,
            median=smed,
            p5=sp5,
            p10=sp10,
            p90=sp90,
            p95=sp95,
            seguros=int(t_counts["seguros"]),
            probables=int(t_counts["probables"]),
            en_juego=int(t_counts["en_juego"]),
        ).to_dict()

    party_meta = {
        p.id: {
            "name": p.name,
            "short_name": p.short_name,
            "color": p.color,
            "bloc": p.bloc,
            "national_group": p.national_group,
        }
        for p in parties_cat.parties
    }
    constituency_meta = {
        c.cpro: {
            "short_name": c.short_name,
            "ccaa_name": c.ccaa_name,
            "seats_2023": c.seats_2023,
            "seats_2026": c.seats_2026,
            "cartogram_row": c.cartogram_row,
            "cartogram_col": c.cartogram_col,
        }
        for c in consts_cat.constituencies
    }

    national_block: dict[str, Any] = {
        "vote_shares": vote_shares_block,
        "seats": seats_block,
        "stage1_comparison": dict(s1.stage1_comparison),
        "cis_sensitivity": dict(s1.cis_sensitivity),
        "party_meta": party_meta,
        "constituency_meta": constituency_meta,
        "covariance_config": cov_cfg.to_dict(),
    }

    # Strided 200-draw Monte Carlo sample for client-side live "what-if" swing & Art. 99 calculator
    sample_target = min(int(hyperparams.sim_sample_size), n_sims)
    step = max(1, n_sims // max(1, sample_target))
    sample_indices = np.arange(0, n_sims, step, dtype=np.intp)[:sample_target]
    active_global = [j for j in ordered_idx if bool(np.any(s2.active_mask[:, j]))]
    sim_sample: list[dict[str, Any]] = []
    for s_idx in sample_indices:
        sim_sample.append(
            {
                "shares": {
                    party_ids[j]: round(float(sim.national_shares[s_idx, j]) * 100.0, 3)
                    for j in active_global
                },
                "seats": {
                    party_ids[j]: int(sim.national_seats[s_idx, j]) for j in active_global
                },
            }
        )

    metadata = ForecastMetadata(
        generated_at=datetime.now(MADRID_TZ).isoformat(timespec="seconds"),
        as_of_date=eff_as_of.isoformat(),
        election_date=ELECTION_DATE_2026.isoformat(),
        n_sims=int(n_sims),
        seed=int(seed),
        engine=engine,
        exclude_cis_headline=bool(exclude_cis_headline),
        blackout_active=bool(is_in_blackout_window()),
        simulation_runtime_ms=round(float(sim.runtime_ms), 2),
    )

    bundle = ForecastBundle(
        metadata=metadata,
        national=national_block,
        tier_totals={k: int(v) for k, v in tier_totals.items()},
        trend_series=dict(s1.trend_series),
        constituencies=tuple(const_forecasts),
        battlegrounds=battlegrounds[:25],
        investiture=investiture,
        pollster_scorecard=[r.to_dict() for r in ratings],
        sim_sample=sim_sample,
    )

    if write_output:
        target = output_path or (get_outputs_dir() / "forecast_29n.json")
        bundle.save_json(target)

    return bundle
