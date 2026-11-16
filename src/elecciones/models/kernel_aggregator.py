"""Stage 1A: Analytical Kernel Poll Aggregator with Wave-Overlap & Jackman House Effects.

Implements an *El País* / *FiveThirtyEight*-style weighted poll aggregator:
- Exponential recency decay over fieldwork midpoints (campaign vs. quiet half-life)
- Effective sample-size cap (``sample_size_cap_general=4000``, ``sample_size_cap_cis=2000``)
- Rolling-tracker wave-overlap discounting ``u_i = 1 / sum_j overlap(i, j)``
- Empirical-Bayes pollster quality & herding weights from ``pollster_ratings``
- Iterative Jackman (2005) zero-sum centered house effects around reference pollsters
- Combined ``FA_FV`` splitting and optional ``exclude_cis_headline`` toggle
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Final

import numpy as np

from elecciones.config import (
    DEFAULT_HYPERPARAMETERS,
    ELECTION_DATE_2026,
    HISTORICAL_ELECTION_DATES,
    ModelHyperparameters,
)
from elecciones.models.pollster_ratings import is_benchmark_or_excluded_pollster
from elecciones.schemas import (
    ForecastEngine,
    PollRecord,
    PollsterRatingRecord,
    RegionalPollRecord,
)

NATIONAL_2026_TRACKED_PARTIES: Final[tuple[str, ...]] = (
    "PP",
    "PSOE",
    "VOX",
    "FA",
    "PODEMOS",
    "COMPROMIS_FV",
    "SALF",
    "ADELANTE",
    "ERC",
    "JUNTS",
    "EH_BILDU",
    "PNV",
    "BNG",
    "CCA",
)

# Prior ratios for splitting legacy ``SUMAR`` rows only when query date predates mid-Oct 2026
_LEGACY_SUMAR_WEIGHTS: Final[dict[str, float]] = {
    "FA": 0.64,
    "COMPROMIS_FV": 0.16,
    "ADELANTE": 0.11,
    "PODEMOS": 0.09,
}

# Approximate national minor-party + blank mass expected outside 14 tracked columns
DEFAULT_NATIONAL_BLANK_SHARE: Final[float] = 0.0081
_MIN_OTHER_SHARE: Final[float] = 0.012


@dataclass(frozen=True, slots=True)
class Stage1Result:
    """Unified output contract shared by Stage 1A Kernel, 1B State-Space, and Ensemble."""

    as_of_date: date
    target_election_date: date
    engine: ForecastEngine
    parties: tuple[str, ...]
    mean_shares: dict[str, float]
    blank_share: float
    sd_shares: dict[str, float]
    cov_shares: np.ndarray
    house_effects: dict[str, dict[str, float]]
    reference_pollsters: tuple[str, ...]
    n_polls_used: int
    n_effective_polls: float
    exclude_cis_headline: bool
    poll_diagnostics: list[dict[str, Any]] = field(default_factory=list)
    trend_series: dict[str, Any] = field(default_factory=dict)
    stage1_comparison: dict[str, Any] = field(default_factory=dict)
    cis_sensitivity: dict[str, Any] = field(default_factory=dict)

    def to_summary_dict(self) -> dict[str, Any]:
        """Return compact serializable summary over national mean/CI shares."""
        p10: dict[str, float] = {}
        p90: dict[str, float] = {}
        for p in self.parties:
            m = self.mean_shares.get(p, 0.0)
            sd = self.sd_shares.get(p, 0.0)
            p10[p] = round(max(0.0, m - 1.28155 * sd), 6)
            p90[p] = round(min(1.0, m + 1.28155 * sd), 6)
        return {
            "engine": self.engine,
            "as_of_date": self.as_of_date.isoformat(),
            "target_election_date": self.target_election_date.isoformat(),
            "n_polls_used": self.n_polls_used,
            "n_effective_polls": round(self.n_effective_polls, 2),
            "exclude_cis_headline": self.exclude_cis_headline,
            "blank_share": round(self.blank_share, 6),
            "mean_shares": {k: round(v, 6) for k, v in self.mean_shares.items()},
            "sd_shares": {k: round(v, 6) for k, v in self.sd_shares.items()},
            "p10_shares": p10,
            "p90_shares": p90,
        }


def resolve_target_election_date(election_id: str) -> date:
    if election_id == "2026-11":
        return ELECTION_DATE_2026
    if election_id in HISTORICAL_ELECTION_DATES:
        return HISTORICAL_ELECTION_DATES[election_id]
    raise ValueError(f"Unknown election_id {election_id!r}")


def filter_polls_as_of(
    polls: Sequence[PollRecord],
    as_of: date,
    *,
    exclude_cis_headline: bool = False,
) -> list[PollRecord]:
    """Filter valid polls published/completed strictly up to ``as_of``."""
    out: list[PollRecord] = []
    for p in polls:
        if is_benchmark_or_excluded_pollster(p.pollster):
            continue
        if p.fieldwork_end > as_of:
            continue
        if p.pub_date is not None and p.pub_date > as_of:
            continue
        if exclude_cis_headline and p.is_cis:
            continue
        out.append(p)
    out.sort(key=lambda r: (r.fieldwork_end, r.fieldwork_start, r.pollster))
    return out


def compute_wave_overlap_penalties(polls: Sequence[PollRecord | RegionalPollRecord]) -> np.ndarray:
    """Compute within-pollster fieldwork overlap discount ``u_i = 1 / sum_j overlap(i, j)``.

    For each pollster, every pair ``(i, j)`` sharing ``d >= 1`` overlapping fieldwork days
    contributes ``d / max(len_i, len_j)`` to the overlap denominator. Duplicate identical
    windows sum to exact effective weight ``1.0`` and expanding rolling trackers are bounded
    near ~``1.0 - 1.5`` polls regardless of daily publication cadence.
    """
    n = len(polls)
    if n == 0:
        return np.zeros(0, dtype=np.float64)

    by_pollster: dict[str, list[int]] = defaultdict(list)
    for idx, p in enumerate(polls):
        by_pollster[p.pollster].append(idx)

    penalties = np.ones(n, dtype=np.float64)
    for idxs in by_pollster.values():
        if len(idxs) <= 1:
            continue
        for i in idxs:
            pi = polls[i]
            si, ei = pi.fieldwork_start, pi.fieldwork_end
            li = max(1, (ei - si).days + 1)
            denom = 0.0
            for j in idxs:
                if i == j:
                    denom += 1.0
                    continue
                pj = polls[j]
                sj, ej = pj.fieldwork_start, pj.fieldwork_end
                ov = (min(ei, ej) - max(si, sj)).days + 1
                if ov <= 0:
                    continue
                lj = max(1, (ej - sj).days + 1)
                denom += ov / float(max(li, lj))
            penalties[i] = 1.0 / max(1.0, denom)
    return penalties


def compute_poll_weights(
    polls: Sequence[PollRecord],
    as_of: date,
    target_election_date: date,
    ratings_map: dict[str, PollsterRatingRecord] | None = None,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
    *,
    exclude_cis_headline: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Stage 1 composite weights ``w_i`` and wave-overlap discounts ``u_i``."""
    n = len(polls)
    if n == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    overlap_u = compute_wave_overlap_penalties(polls)
    days_to_elec = (target_election_date - as_of).days
    half_life = (
        hyperparams.half_life_campaign_days
        if days_to_elec <= 60
        else hyperparams.half_life_quiet_days
    )
    decay_rate = math.log(2.0) / max(1.0, half_life)

    weights = np.zeros(n, dtype=np.float64)
    for i, poll in enumerate(polls):
        if exclude_cis_headline and poll.is_cis:
            weights[i] = 0.0
            continue
        age_days = max(0.0, float((as_of - poll.midpoint_date).days))
        # Trim ancient noise (> 12 half-lives) cleanly
        eff_age = min(240.0, age_days)
        w_time = math.exp(-decay_rate * eff_age)
        cap = (
            hyperparams.sample_size_cap_cis
            if poll.is_cis
            else hyperparams.sample_size_cap_general
        )
        n_eff = min(poll.sample_size or 1000, cap)
        w_sample = math.sqrt(max(100, n_eff) / 1000.0)

        rating = ratings_map.get(poll.pollster) if ratings_map else None
        if rating is not None:
            w_qual = max(0.10, rating.grade_weight) * max(0.25, rating.herding_penalty)
        else:
            w_qual = 0.45 if poll.is_cis else 1.0

        weights[i] = w_time * w_sample * w_qual * overlap_u[i]
    return weights, overlap_u


def infer_fa_fv_split(polls: Sequence[PollRecord], ref_date: date) -> float:
    """Infer contemporaneous share ``FA / (FA + COMPROMIS_FV)`` from separate polls."""
    num = 0.0
    den = 0.0
    for p in polls:
        if p.is_cis:
            continue
        if "FA" in p.party_shares and "COMPROMIS_FV" in p.party_shares:
            age = max(0, (ref_date - p.fieldwork_end).days)
            w = math.exp(-age / 30.0)
            fa = p.party_shares["FA"]
            fv = p.party_shares["COMPROMIS_FV"]
            if fa + fv > 0.0:
                num += w * fa
                den += w * (fa + fv)
    return float(num / den) if den > 0.0 else 0.825


def extract_poll_party_matrix(
    polls: Sequence[PollRecord],
    tracked_parties: Sequence[str],
    ref_date: date,
    *,
    allow_legacy_sumar_split: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract observation matrix ``Y`` and availability mask ``O`` of shape ``(N_polls, K)``.

    Missing party cells remain masked (``O[i, k] = False``) rather than coerced to ``0.0``.
    Combined ``FA_FV`` cells are split according to empirical contemporaneous ``FA / (FA+FV)``.
    """
    n_p = len(polls)
    k = len(tracked_parties)
    p_idx = {p: idx for idx, p in enumerate(tracked_parties)}
    y_mat = np.full((n_p, k), np.nan, dtype=np.float64)
    obs_mask = np.zeros((n_p, k), dtype=bool)
    fa_split = infer_fa_fv_split(polls, ref_date)

    # Check if explicit FA / FA_FV observations already exist in recent window
    has_recent_fa = any(
        ("FA" in p.party_shares or "FA_FV" in p.party_shares)
        and (ref_date - p.fieldwork_end).days <= 35
        for p in polls
    )

    for i, poll in enumerate(polls):
        for party, val in poll.party_shares.items():
            if party in p_idx:
                col = p_idx[party]
                y_mat[i, col] = val
                obs_mask[i, col] = True
            elif party == "FA_FV":
                fv_val = val
                if "FA" in p_idx and not obs_mask[i, p_idx["FA"]]:
                    y_mat[i, p_idx["FA"]] = fv_val * fa_split
                    obs_mask[i, p_idx["FA"]] = True
                if "COMPROMIS_FV" in p_idx and not obs_mask[i, p_idx["COMPROMIS_FV"]]:
                    y_mat[i, p_idx["COMPROMIS_FV"]] = fv_val * (1.0 - fa_split)
                    obs_mask[i, p_idx["COMPROMIS_FV"]] = True
            elif (
                party == "SUMAR"
                and (allow_legacy_sumar_split or not has_recent_fa)
                and "FA" in p_idx
                and not obs_mask[i, p_idx["FA"]]
            ):
                # Pre-split 2023–early Oct 2026 SUMAR fallback
                for sub_p, frac in _LEGACY_SUMAR_WEIGHTS.items():
                    if sub_p in p_idx and not obs_mask[i, p_idx[sub_p]]:
                        y_mat[i, p_idx[sub_p]] = val * frac
                        obs_mask[i, p_idx[sub_p]] = True
    return y_mat, obs_mask


def _prior_delta_share(
    rating: PollsterRatingRecord | None,
    party: str,
) -> float:
    if rating is None:
        return 0.0
    if party == "PP":
        return rating.delta_PP / 100.0
    if party == "PSOE":
        return rating.delta_PSOE / 100.0
    if party == "VOX":
        return rating.delta_VOX / 100.0
    if party in {"FA", "SUMAR", "UP"}:
        return rating.delta_FA / 100.0
    if party == "PODEMOS":
        return rating.delta_PODEMOS / 100.0
    return 0.0


def _local_trend_matrix(
    mid_ord: np.ndarray,
    weights: np.ndarray,
    adj_y: np.ndarray,
    obs_mask: np.ndarray,
    bandwidth_days: float,
) -> np.ndarray:
    """Evaluate leave-one-poll-free local weighted trend at each poll midpoint."""
    n_p, k = adj_y.shape
    dt = np.abs(mid_ord[:, None] - mid_ord[None, :])
    k_mat = np.exp(-0.5 * (dt / max(4.0, bandwidth_days)) ** 2) * weights[None, :]
    trend = np.full((n_p, k), np.nan, dtype=np.float64)
    for col in range(k):
        m = obs_mask[:, col]
        if not np.any(m):
            continue
        kw = k_mat[:, m]
        ys = adj_y[m, col]
        den = kw.sum(axis=1)
        num = kw @ ys
        valid = den > 1e-12
        trend[valid, col] = num[valid] / den[valid]
    return trend


def estimate_house_effects(
    polls: Sequence[PollRecord],
    weights: np.ndarray,
    y_mat: np.ndarray,
    obs_mask: np.ndarray,
    tracked_parties: Sequence[str],
    ratings_map: dict[str, PollsterRatingRecord] | None = None,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
    reference_pollsters: Sequence[str] | None = None,
    *,
    bandwidth_days: float = 12.0,
    n_iters: int = 10,
) -> tuple[dict[str, dict[str, float]], tuple[str, ...]]:
    """Iteratively estimate Jackman zero-sum centered house effects ``delta_{h,p}``.

    Guarantees ``sum_{h in R} delta[h][p] == 0.0`` (within machine precision) for
    every tracked party across the non-CIS A/B reference set ``R``.
    """
    n_p, k = y_mat.shape
    pollsters_present = sorted({p.pollster for i, p in enumerate(polls) if weights[i] > 0.0})
    if not pollsters_present or n_p == 0:
        return {}, ()

    if reference_pollsters is not None:
        ref_set = tuple(sorted(set(reference_pollsters) & set(pollsters_present)))
    else:
        w_by_pol: dict[str, float] = defaultdict(float)
        cis_set = {p.pollster for p in polls if p.is_cis}
        for i, p in enumerate(polls):
            w_by_pol[p.pollster] += float(weights[i])
        non_cis = [pol for pol in pollsters_present if pol not in cis_set and w_by_pol[pol] > 0.0]
        if ratings_map:
            ab_non_cis = [
                pol
                for pol in non_cis
                if ratings_map.get(pol) is not None and ratings_map[pol].grade in {"A", "B"}
            ]
        else:
            ab_non_cis = non_cis
        ref_candidates = ab_non_cis if len(ab_non_cis) >= 2 else non_cis
        if not ref_candidates:
            ref_candidates = pollsters_present
        ref_set = tuple(sorted(ref_candidates))

    delta_mat = np.zeros((len(pollsters_present), k), dtype=np.float64)
    pol_idx = {pol: idx for idx, pol in enumerate(pollsters_present)}
    ref_indices = [pol_idx[r] for r in ref_set if r in pol_idx]

    # Seed with historical prior deltas shrunk cleanly
    prior_mat = np.zeros_like(delta_mat)
    for pol, ridx in pol_idx.items():
        rating = ratings_map.get(pol) if ratings_map else None
        for col, party in enumerate(tracked_parties):
            prior_mat[ridx, col] = _prior_delta_share(rating, party)
    delta_mat[:] = prior_mat

    if ref_indices:
        ref_mean0 = delta_mat[ref_indices].mean(axis=0, keepdims=True)
        delta_mat -= ref_mean0

    row_pol_indices = np.array([pol_idx.get(p.pollster, -1) for p in polls], dtype=np.int32)
    mid_ord = np.array([p.midpoint_date.toordinal() for p in polls], dtype=np.float64)
    # Light shrinkage in effective-poll units so contemporaneous skew dominates over old priors
    tau_eff = max(0.05, hyperparams.eb_house_tau * 0.03)

    for _ in range(n_iters):
        # Compute local trend primarily anchored on non-CIS/adjusted reference observations
        adj_y = y_mat.copy()
        valid_rows = row_pol_indices >= 0
        adj_y[valid_rows] -= delta_mat[row_pol_indices[valid_rows]]
        # Reference-only trend if enough reference polls exist
        ref_w = weights.copy()
        for i, p in enumerate(polls):
            if p.pollster not in ref_set:
                ref_w[i] *= 0.25
        if ref_w.sum() <= 1e-12:
            ref_w = weights.copy()
        trend = _local_trend_matrix(mid_ord, ref_w, adj_y, obs_mask, bandwidth_days)

        new_delta = np.zeros_like(delta_mat)
        for h_i in pol_idx.values():
            p_rows = np.flatnonzero(row_pol_indices == h_i)
            for col in range(k):
                m = obs_mask[p_rows, col] & ~np.isnan(trend[p_rows, col])
                if not np.any(m):
                    new_delta[h_i, col] = prior_mat[h_i, col] * 0.5
                    continue
                rr = p_rows[m]
                ws = weights[rr]
                w_sum = float(ws.sum())
                if w_sum <= 1e-12:
                    new_delta[h_i, col] = prior_mat[h_i, col] * 0.5
                    continue
                diff_sum = float(np.sum(ws * (y_mat[rr, col] - trend[rr, col])))
                new_delta[h_i, col] = (diff_sum + tau_eff * prior_mat[h_i, col]) / (
                    w_sum + tau_eff
                )

        if ref_indices:
            new_delta -= new_delta[ref_indices].mean(axis=0, keepdims=True)
        delta_mat = 0.35 * delta_mat + 0.65 * new_delta
        if ref_indices:
            delta_mat -= delta_mat[ref_indices].mean(axis=0, keepdims=True)

    house_dict: dict[str, dict[str, float]] = {}
    for pol, ridx in pol_idx.items():
        house_dict[pol] = {
            tracked_parties[col]: float(delta_mat[ridx, col]) for col in range(k)
        }
    return house_dict, ref_set


def _infer_tracked_parties(
    election_id: str,
    polls: Sequence[PollRecord],
    as_of: date | None = None,
) -> tuple[str, ...]:
    if election_id == "2026-11":
        return NATIONAL_2026_TRACKED_PARTIES
    if not polls:
        return ()
    ref_date = as_of or max(p.fieldwork_end for p in polls)
    recent = [p for p in polls if (ref_date - p.fieldwork_end).days <= 35]
    if len(recent) < 8:
        recent = list(polls[-min(20, len(polls)) :])
    counts: dict[str, int] = defaultdict(int)
    for p in recent:
        for k in p.party_shares:
            counts[k] += 1
    min_cnt = max(2, len(recent) // 10)
    ordered = [
        p
        for p, cnt in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        if p != "OTROS" and cnt >= min_cnt
    ]
    return tuple(ordered)


def aggregate_kernel(
    polls: Sequence[PollRecord],
    as_of: date,
    *,
    election_id: str = "2026-11",
    target_election_date: date | None = None,
    ratings: Sequence[PollsterRatingRecord] | None = None,
    tracked_parties: Sequence[str] | None = None,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
    exclude_cis_headline: bool = False,
    reference_pollsters: Sequence[str] | None = None,
    blank_share: float = DEFAULT_NATIONAL_BLANK_SHARE,
    trend_days: int = 100,
) -> Stage1Result:
    """Run Stage 1A Kernel poll aggregation up to ``as_of``."""
    t_elec = target_election_date or resolve_target_election_date(election_id)
    active_polls = filter_polls_as_of(
        polls, as_of, exclude_cis_headline=exclude_cis_headline
    )
    if not active_polls:
        raise ValueError(f"No polls available <= {as_of} (exclude_cis={exclude_cis_headline})")

    raw_parties = tracked_parties or _infer_tracked_parties(election_id, active_polls, as_of)
    parties = tuple(p for p in raw_parties if p != "OTROS")
    ratings_map = {r.pollster: r for r in ratings} if ratings else {}

    weights, overlap_u = compute_poll_weights(
        active_polls,
        as_of,
        t_elec,
        ratings_map=ratings_map,
        hyperparams=hyperparams,
        exclude_cis_headline=exclude_cis_headline,
    )

    y_mat, obs_mask = extract_poll_party_matrix(active_polls, parties, as_of)
    # Trim zero-weight tail for speed in estimation while preserving recent campaign signal
    keep_idx = np.flatnonzero(weights > 1e-6 * float(weights.max()))
    est_polls = [active_polls[i] for i in keep_idx]
    est_w = weights[keep_idx]
    est_u = overlap_u[keep_idx]
    est_y = y_mat[keep_idx]
    est_obs = obs_mask[keep_idx]

    house_effects, ref_set = estimate_house_effects(
        est_polls,
        est_w,
        est_y,
        est_obs,
        parties,
        ratings_map=ratings_map,
        hyperparams=hyperparams,
        reference_pollsters=reference_pollsters,
    )

    # Subtract estimated house effects to form de-biased observations
    adj_y = est_y.copy()
    for i, poll in enumerate(est_polls):
        he = house_effects.get(poll.pollster, {})
        for col, party in enumerate(parties):
            if est_obs[i, col]:
                adj_y[i, col] = max(0.0, min(1.0, adj_y[i, col] - he.get(party, 0.0)))

    # Weighted national mean & non-sampling weighted variance at as_of
    k = len(parties)
    mean_vec = np.zeros(k, dtype=np.float64)
    var_vec = np.zeros(k, dtype=np.float64)
    n_eff_global = float((est_w.sum() ** 2) / max(1e-12, float(np.sum(est_w**2))))

    for col in range(k):
        m = est_obs[:, col] & ~np.isnan(adj_y[:, col])
        if not np.any(m):
            continue
        ws = est_w[m]
        ys = adj_y[m, col]
        w_sum = float(ws.sum())
        if w_sum <= 1e-12:
            continue
        mu = float(np.sum(ws * ys) / w_sum)
        mean_vec[col] = mu

        # Weighted empirical dispersion + theoretical SRS floor + blackout RW drift
        w_sq = float(np.sum(ws**2))
        n_eff_k = (w_sum * w_sum) / max(1e-12, w_sq)
        srs_var = max(1e-6, mu * max(0.0, 1.0 - mu) / max(400.0, n_eff_k * 900.0))
        emp_var = float(np.sum(ws * (ys - mu) ** 2) / w_sum) / max(1e-12, n_eff_k * 0.45)
        days_gap = max(0, (t_elec - as_of).days)
        # Convert ALR random-walk daily variance into linear probability variance approx: (p(1-p))^2
        logit_scale = max(1e-4, (mu * max(0.0, 1.0 - mu)) ** 2)
        rw_var = float(days_gap) * hyperparams.state_space_rw_var_per_day * logit_scale * 0.35
        var_vec[col] = max(srs_var + rw_var, 0.5 * srs_var + 0.5 * emp_var + rw_var)

    # Rescale cleanly if tracked parties exceed allowable valid candidacy share
    max_cand_share = max(0.85, 1.0 - blank_share - _MIN_OTHER_SHARE)
    tracked_sum = float(mean_vec.sum())
    if tracked_sum > max_cand_share and tracked_sum > 0.0:
        mean_vec *= max_cand_share / tracked_sum

    otros_share = max(0.0, (1.0 - blank_share) - float(mean_vec.sum()))
    full_parties = (*parties, "OTROS") if "OTROS" not in parties else parties
    mean_shares = {parties[col]: float(mean_vec[col]) for col in range(k)}
    sd_shares = {parties[col]: float(math.sqrt(var_vec[col])) for col in range(k)}
    if "OTROS" not in mean_shares:
        mean_shares["OTROS"] = otros_share
        sd_shares["OTROS"] = 0.0045

    # Diagonal-plus-weak-negative-sum covariance matrix over full_parties
    sd_arr = np.array([sd_shares[p] for p in full_parties], dtype=np.float64)
    cov_shares = np.diag(sd_arr**2)
    # Enforce subtle simplex off-diagonal structure from weighted cross-poll residuals
    if est_y.shape[0] >= 5 and k >= 2:
        obs_both = np.all(est_obs[:, : min(4, k)], axis=1)
        if np.sum(obs_both) >= 5:
            res_sub = adj_y[obs_both, :k] - mean_vec[:k][None, :]
            res_sub = np.nan_to_num(res_sub, nan=0.0)
            ws_sub = est_w[obs_both]
            ws_norm = ws_sub / max(1e-12, float(ws_sub.sum()))
            w_cov = (res_sub.T * ws_norm) @ res_sub
            for a in range(k):
                for b in range(k):
                    if a == b:
                        continue
                    denom_ab = math.sqrt(max(1e-12, float(w_cov[a, a] * w_cov[b, b])))
                    corr = max(-0.7, min(0.7, float(w_cov[a, b] / denom_ab)))
                    cov_shares[a, b] = 0.5 * corr * sd_arr[a] * sd_arr[b]

    # Build daily trend ribbon for UI / diagnostics over final trend_days up to as_of
    trend_start = as_of - timedelta(days=trend_days)
    n_grid = max(15, trend_days // 3 + 1)
    grid_dates = [
        trend_start + timedelta(days=int(round(i * trend_days / max(1, n_grid - 1))))
        for i in range(n_grid)
    ]
    if grid_dates[-1] != as_of:
        grid_dates[-1] = as_of

    est_ords = np.array([p.midpoint_date.toordinal() for p in est_polls], dtype=np.float64)
    grid_ords = np.array([d.toordinal() for d in grid_dates], dtype=np.float64)
    # Include polls inside trend window for UI plotting (use est_polls overlap weights)
    ribbon_parties = list(parties[: min(8, len(parties))])
    trend_parties: dict[str, dict[str, list[float]]] = {
        p: {"mean": [], "p10": [], "p90": []} for p in ribbon_parties
    }
    for di, g_ord in enumerate(grid_ords):
        dt = g_ord - est_ords
        # Causal up to each plotted date for realistic historical campaign trace
        causal = est_ords <= g_ord + 1.5
        k_t = np.where(causal, np.exp(-0.5 * (np.abs(dt) / 9.0) ** 2), 0.0)
        # Re-multiply sample * quality * overlap without global recency to trace campaign path
        base_q = np.array(
            [
                math.sqrt(min(p.sample_size or 1000, 4000) / 1000.0)
                * (ratings_map[p.pollster].grade_weight if p.pollster in ratings_map else 1.0)
                * est_u[i]
                * (0.45 if p.is_cis else 1.0)
                for i, p in enumerate(est_polls)
            ],
            dtype=np.float64,
        )
        w_t = k_t * base_q
        for p in ribbon_parties:
            col = parties.index(p)
            m = est_obs[:, col] & (w_t > 1e-8)
            if not np.any(m):
                prev_m = trend_parties[p]["mean"][-1] if di > 0 else mean_shares[p]
                prev_lo = trend_parties[p]["p10"][-1] if di > 0 else max(0.0, prev_m - 0.015)
                prev_hi = trend_parties[p]["p90"][-1] if di > 0 else min(1.0, prev_m + 0.015)
                trend_parties[p]["mean"].append(round(prev_m, 5))
                trend_parties[p]["p10"].append(round(prev_lo, 5))
                trend_parties[p]["p90"].append(round(prev_hi, 5))
                continue
            ws = w_t[m]
            ys = adj_y[m, col]
            mu_t = float(np.sum(ws * ys) / np.sum(ws))
            n_eff_t = float(ws.sum() ** 2 / max(1e-12, float(np.sum(ws**2))))
            disp = float(np.sum(ws * (ys - mu_t) ** 2) / ws.sum())
            se_t = math.sqrt(max(disp / max(1.0, n_eff_t), mu_t * (1.0 - mu_t) / 2500.0) + 1.2e-5)
            trend_parties[p]["mean"].append(round(mu_t, 5))
            trend_parties[p]["p10"].append(round(max(0.0, mu_t - 1.28155 * se_t), 5))
            trend_parties[p]["p90"].append(round(min(1.0, mu_t + 1.28155 * se_t), 5))

    poll_diags: list[dict[str, Any]] = []
    scatter_polls: list[dict[str, Any]] = []
    for i, poll in enumerate(est_polls):
        diag_row = {
            "pollster": poll.pollster,
            "media": poll.media,
            "fieldwork_start": poll.fieldwork_start.isoformat(),
            "fieldwork_end": poll.fieldwork_end.isoformat(),
            "midpoint_date": poll.midpoint_date.isoformat(),
            "sample_size": poll.sample_size,
            "is_cis": poll.is_cis,
            "overlap_penalty": round(float(est_u[i]), 4),
            "weight": round(float(est_w[i]), 4),
        }
        poll_diags.append(diag_row)
        if poll.midpoint_date >= trend_start:
            pts = {
                parties[col]: round(float(adj_y[i, col]), 5)
                for col in range(min(8, len(parties)))
                if est_obs[i, col]
            }
            scatter_polls.append(
                {
                    "date": poll.midpoint_date.isoformat(),
                    "pollster": poll.pollster,
                    "is_cis": poll.is_cis,
                    "shares": pts,
                }
            )

    return Stage1Result(
        as_of_date=as_of,
        target_election_date=t_elec,
        engine="kernel",
        parties=full_parties,
        mean_shares=mean_shares,
        blank_share=blank_share,
        sd_shares=sd_shares,
        cov_shares=cov_shares,
        house_effects=house_effects,
        reference_pollsters=ref_set,
        n_polls_used=len(est_polls),
        n_effective_polls=n_eff_global,
        exclude_cis_headline=exclude_cis_headline,
        poll_diagnostics=sorted(poll_diags, key=lambda d: d["weight"], reverse=True)[:80],
        trend_series={
            "dates": [d.isoformat() for d in grid_dates],
            "parties": trend_parties,
            "polls": scatter_polls,
        },
    )
