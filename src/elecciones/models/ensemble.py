"""Stage 1 Ensemble Combiner & Unified Runner (Stage 1A Kernel + Stage 1B State-Space)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from datetime import date
from typing import Any

import numpy as np

from elecciones.config import DEFAULT_HYPERPARAMETERS, ModelHyperparameters
from elecciones.models.kernel_aggregator import (
    DEFAULT_NATIONAL_BLANK_SHARE,
    Stage1Result,
    aggregate_kernel,
    compute_poll_weights,
    filter_polls_as_of,
)
from elecciones.models.state_space import smooth_state_space
from elecciones.schemas import ForecastEngine, PollRecord, PollsterRatingRecord


def _interpolate_series_onto_dates(
    target_dates: Sequence[str],
    src_dates: Sequence[str],
    src_vals: Sequence[float],
) -> list[float]:
    """Linearly interpolate a dated trend series onto ``target_dates``."""
    if not src_dates or not src_vals:
        return [0.0] * len(target_dates)
    tx = np.array([date.fromisoformat(d).toordinal() for d in target_dates], dtype=np.float64)
    sx = np.array([date.fromisoformat(d).toordinal() for d in src_dates], dtype=np.float64)
    sy = np.asarray(src_vals, dtype=np.float64)
    interp = np.interp(tx, sx, sy)
    return [float(x) for x in interp]


def _blend_trend_series(
    kernel_res: Stage1Result,
    ss_res: Stage1Result,
    wk: float,
    ws: float,
) -> dict[str, Any]:
    k_trend = kernel_res.trend_series
    s_trend = ss_res.trend_series
    k_dates = list(k_trend.get("dates", []))
    s_dates = list(s_trend.get("dates", []))
    if not k_dates:
        return dict(s_trend)
    if not s_dates:
        return dict(k_trend)

    k_parties = k_trend.get("parties", {})
    s_parties = s_trend.get("parties", {})
    blended_parties: dict[str, dict[str, list[float]]] = {}
    for party in k_parties:
        kp = k_parties[party]
        sp = s_parties.get(party)
        if not sp:
            blended_parties[party] = {
                "mean": list(kp["mean"]),
                "p10": list(kp["p10"]),
                "p90": list(kp["p90"]),
            }
            continue
        sp_mean = _interpolate_series_onto_dates(k_dates, s_dates, sp["mean"])
        sp_p10 = _interpolate_series_onto_dates(k_dates, s_dates, sp["p10"])
        sp_p90 = _interpolate_series_onto_dates(k_dates, s_dates, sp["p90"])
        blended_parties[party] = {
            "mean": [
                round(wk * kp["mean"][i] + ws * sp_mean[i], 5) for i in range(len(k_dates))
            ],
            "p10": [round(wk * kp["p10"][i] + ws * sp_p10[i], 5) for i in range(len(k_dates))],
            "p90": [round(wk * kp["p90"][i] + ws * sp_p90[i], 5) for i in range(len(k_dates))],
        }
    return {
        "dates": k_dates,
        "parties": blended_parties,
        "polls": k_trend.get("polls", []),
    }


def combine_stage1_results(
    kernel_res: Stage1Result,
    ss_res: Stage1Result,
    *,
    w_kernel: float,
    w_state_space: float,
) -> Stage1Result:
    """Combine Stage 1A Kernel & Stage 1B State-Space using the Law of Total Covariance."""
    w_tot = max(1e-12, float(w_kernel) + float(w_state_space))
    wk = max(0.0, float(w_kernel)) / w_tot
    ws = max(0.0, float(w_state_space)) / w_tot

    parties = kernel_res.parties
    mu_k = np.array([kernel_res.mean_shares.get(p, 0.0) for p in parties], dtype=np.float64)
    mu_s = np.array([ss_res.mean_shares.get(p, 0.0) for p in parties], dtype=np.float64)
    mu_mix = wk * mu_k + ws * mu_s

    target_sum = 1.0 - kernel_res.blank_share
    cur_sum = float(mu_mix.sum())
    if cur_sum > 0.0:
        mu_mix *= target_sum / cur_sum

    # Align SS covariance columns onto kernel_res.parties order
    s_idx = {p: i for i, p in enumerate(ss_res.parties)}
    k_n = len(parties)
    cov_s_aligned = np.zeros((k_n, k_n), dtype=np.float64)
    for a, pa in enumerate(parties):
        ia = s_idx.get(pa)
        if ia is None:
            continue
        for b, pb in enumerate(parties):
            ib = s_idx.get(pb)
            if ib is None:
                continue
            cov_s_aligned[a, b] = ss_res.cov_shares[ia, ib]

    diff = mu_k - mu_s
    cov_mix = wk * kernel_res.cov_shares + ws * cov_s_aligned + (wk * ws) * np.outer(diff, diff)
    cov_mix = 0.5 * (cov_mix + cov_mix.T)

    mean_shares = {parties[i]: float(mu_mix[i]) for i in range(k_n)}
    sd_shares = {
        parties[i]: float(math.sqrt(max(1e-12, cov_mix[i, i]))) for i in range(k_n)
    }

    comparison = {
        "kernel": kernel_res.to_summary_dict(),
        "state_space": ss_res.to_summary_dict(),
        "weights": {"kernel": round(wk, 4), "state_space": round(ws, 4)},
    }

    res = Stage1Result(
        as_of_date=kernel_res.as_of_date,
        target_election_date=kernel_res.target_election_date,
        engine="ensemble",
        parties=parties,
        mean_shares=mean_shares,
        blank_share=kernel_res.blank_share,
        sd_shares=sd_shares,
        cov_shares=cov_mix,
        house_effects=kernel_res.house_effects,
        reference_pollsters=kernel_res.reference_pollsters,
        n_polls_used=max(kernel_res.n_polls_used, ss_res.n_polls_used),
        n_effective_polls=wk * kernel_res.n_effective_polls + ws * ss_res.n_effective_polls,
        exclude_cis_headline=kernel_res.exclude_cis_headline,
        poll_diagnostics=kernel_res.poll_diagnostics,
        trend_series=_blend_trend_series(kernel_res, ss_res, wk, ws),
        stage1_comparison=comparison,
    )
    comparison["ensemble"] = res.to_summary_dict()
    return res


def _summarize_latest_raw_cis(polls: Sequence[PollRecord], as_of: date) -> dict[str, Any] | None:
    cis_polls = [p for p in polls if p.is_cis and p.fieldwork_end <= as_of]
    if not cis_polls:
        return None
    latest = max(cis_polls, key=lambda p: p.fieldwork_end)
    return {
        "pollster": latest.pollster,
        "fieldwork_start": latest.fieldwork_start.isoformat(),
        "fieldwork_end": latest.fieldwork_end.isoformat(),
        "sample_size": latest.sample_size,
        "mean_shares": {k: round(v, 6) for k, v in latest.party_shares.items()},
    }


def run_stage1(
    polls: Sequence[PollRecord],
    as_of: date,
    *,
    engine: ForecastEngine = "ensemble",
    election_id: str = "2026-11",
    target_election_date: date | None = None,
    ratings: Sequence[PollsterRatingRecord] | None = None,
    tracked_parties: Sequence[str] | None = None,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
    exclude_cis_headline: bool = False,
    reference_pollsters: Sequence[str] | None = None,
    blank_share: float = DEFAULT_NATIONAL_BLANK_SHARE,
    compute_cis_sensitivity: bool = True,
) -> Stage1Result:
    """Unified Stage 1 entrypoint executing Kernel, State-Space, or Ensemble."""
    if engine not in {"ensemble", "kernel", "state_space"}:
        raise ValueError(f"Unsupported Stage 1 engine {engine!r}")

    kernel_res = aggregate_kernel(
        polls,
        as_of,
        election_id=election_id,
        target_election_date=target_election_date,
        ratings=ratings,
        tracked_parties=tracked_parties,
        hyperparams=hyperparams,
        exclude_cis_headline=exclude_cis_headline,
        reference_pollsters=reference_pollsters,
        blank_share=blank_share,
    )

    if engine == "kernel":
        # Populate state_space/ensemble comparison in headline mode;
        # skip extra work in fast backtests
        if compute_cis_sensitivity:
            ss_res = smooth_state_space(
                polls,
                as_of,
                election_id=election_id,
                target_election_date=target_election_date,
                ratings=ratings,
                tracked_parties=tuple(p for p in kernel_res.parties if p != "OTROS"),
                hyperparams=hyperparams,
                exclude_cis_headline=exclude_cis_headline,
                reference_pollsters=reference_pollsters,
                blank_share=blank_share,
            )
            ens_res = combine_stage1_results(
                kernel_res,
                ss_res,
                w_kernel=hyperparams.ensemble_weight_kernel,
                w_state_space=hyperparams.ensemble_weight_state_space,
            )
            comp = {
                "kernel": kernel_res.to_summary_dict(),
                "state_space": ss_res.to_summary_dict(),
                "ensemble": ens_res.to_summary_dict(),
                "weights": {"kernel": 1.0, "state_space": 0.0},
            }
            chosen = replace(kernel_res, stage1_comparison=comp)
        else:
            chosen = kernel_res
    elif engine == "state_space":
        ss_res = smooth_state_space(
            polls,
            as_of,
            election_id=election_id,
            target_election_date=target_election_date,
            ratings=ratings,
            tracked_parties=tuple(p for p in kernel_res.parties if p != "OTROS"),
            hyperparams=hyperparams,
            exclude_cis_headline=exclude_cis_headline,
            reference_pollsters=reference_pollsters,
            blank_share=blank_share,
        )
        comp = {
            "kernel": kernel_res.to_summary_dict(),
            "state_space": ss_res.to_summary_dict(),
            "weights": {"kernel": 0.0, "state_space": 1.0},
        }
        if compute_cis_sensitivity:
            ens_res = combine_stage1_results(
                kernel_res,
                ss_res,
                w_kernel=hyperparams.ensemble_weight_kernel,
                w_state_space=hyperparams.ensemble_weight_state_space,
            )
            comp["ensemble"] = ens_res.to_summary_dict()
        chosen = replace(ss_res, stage1_comparison=comp)
    else:
        ss_res = smooth_state_space(
            polls,
            as_of,
            election_id=election_id,
            target_election_date=target_election_date,
            ratings=ratings,
            tracked_parties=tuple(p for p in kernel_res.parties if p != "OTROS"),
            hyperparams=hyperparams,
            exclude_cis_headline=exclude_cis_headline,
            reference_pollsters=reference_pollsters,
            blank_share=blank_share,
        )
        chosen = combine_stage1_results(
            kernel_res,
            ss_res,
            w_kernel=hyperparams.ensemble_weight_kernel,
            w_state_space=hyperparams.ensemble_weight_state_space,
        )

    if compute_cis_sensitivity:
        all_up_to = filter_polls_as_of(polls, as_of, exclude_cis_headline=False)
        has_cis = any(p.is_cis for p in all_up_to)
        if has_cis:
            alt = run_stage1(
                polls,
                as_of,
                engine=engine,
                election_id=election_id,
                target_election_date=target_election_date,
                ratings=ratings,
                tracked_parties=tracked_parties,
                hyperparams=hyperparams,
                exclude_cis_headline=not exclude_cis_headline,
                reference_pollsters=reference_pollsters,
                blank_share=blank_share,
                compute_cis_sensitivity=False,
            )
            with_cis = chosen if not exclude_cis_headline else alt
            without_cis = alt if not exclude_cis_headline else chosen
        else:
            with_cis = chosen
            without_cis = chosen

        ratings_map = {r.pollster: r for r in ratings} if ratings else {}
        if all_up_to:
            w_all, _ = compute_poll_weights(
                all_up_to,
                as_of,
                chosen.target_election_date,
                ratings_map=ratings_map,
                hyperparams=hyperparams,
                exclude_cis_headline=False,
            )
            tot_w = max(1e-12, float(w_all.sum()))
            cis_w_sum = sum(float(w_all[i]) for i, p in enumerate(all_up_to) if p.is_cis)
            cis_w_pct = 100.0 * cis_w_sum / tot_w
        else:
            cis_w_pct = 0.0

        cis_he_raw = with_cis.house_effects.get("CIS", {})
        cis_he_pp = {
            k: round(float(cis_he_raw[k]) * 100.0, 3)
            for k in ("PP", "PSOE", "VOX", "FA", "PODEMOS", "SALF")
            if k in cis_he_raw
        }

        cis_sens: dict[str, Any] = {
            "headline_excludes_cis": exclude_cis_headline,
            "with_cis": with_cis.to_summary_dict(),
            "without_cis": without_cis.to_summary_dict(),
            "cis_weight_pct": round(float(cis_w_pct), 4),
            "cis_house_effect_pp": cis_he_pp,
            "pp_diff_pp": round(
                (with_cis.mean_shares.get("PP", 0.0) - without_cis.mean_shares.get("PP", 0.0))
                * 100.0,
                3,
            ),
            "psoe_diff_pp": round(
                (with_cis.mean_shares.get("PSOE", 0.0) - without_cis.mean_shares.get("PSOE", 0.0))
                * 100.0,
                3,
            ),
            "vox_diff_pp": round(
                (with_cis.mean_shares.get("VOX", 0.0) - without_cis.mean_shares.get("VOX", 0.0))
                * 100.0,
                3,
            ),
        }
        raw_latest = _summarize_latest_raw_cis(polls, as_of)
        if raw_latest is not None:
            cis_sens["raw_cis_latest"] = raw_latest
        chosen = replace(chosen, cis_sensitivity=cis_sens)

    return chosen
