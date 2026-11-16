"""Stage 1B: Dynamic Additive Log-Ratio (ALR) State-Space Poll Smoother.

Models latent national multiparty support in Additive Log-Ratio (ALR) coordinates
relative to ``OTROS`` with daily random-walk innovation variance ``sigma_rw^2``:

- Handles partial party reporting (missing minor/regional columns) via selector
  rows ``C_i`` without zero-imputation distortion.
- Supports combined ``FA_FV`` measurements directly as ``pi_FA + pi_COMPROMIS_FV``
  in the measurement Jacobian ``H_i = C_i @ J(x_t)``.
- Propagates latent variance monotonically across gaps and across the
  Nov 24–28 LOREG Art. 69.7 poll blackout window to election day.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import date, timedelta

import numpy as np

from elecciones.config import DEFAULT_HYPERPARAMETERS, ModelHyperparameters
from elecciones.models.kernel_aggregator import (
    DEFAULT_NATIONAL_BLANK_SHARE,
    Stage1Result,
    _infer_tracked_parties,
    _non_recency_poll_weight,
    aggregate_kernel,
    compute_wave_overlap_penalties,
    filter_polls_as_of,
    resolve_target_election_date,
)
from elecciones.schemas import PollRecord, PollsterRatingRecord


def _alr_to_cand_probs(x: np.ndarray, scale: float) -> np.ndarray:
    """Convert K-vector ALR logits (wrt OTROS) into (K+1) shares summing to ``scale``."""
    mx = max(0.0, float(np.max(x)))
    ex = np.exp(x - mx)
    e0 = math.exp(-mx)
    denom = e0 + float(np.sum(ex))
    out = np.empty(x.size + 1, dtype=np.float64)
    out[:-1] = scale * (ex / denom)
    out[-1] = scale * (e0 / denom)
    return out


def _alr_jacobian_full(s_full: np.ndarray) -> np.ndarray:
    """Exact ``(K+1, K)`` Jacobian d(pi_full)/dx for ALR logits ``x in R^K``.

    ``s_full`` includes ``K`` tracked parties followed by ``OTROS`` summing to ``scale``.
    """
    k = s_full.size - 1
    scale = float(np.sum(s_full))
    p = s_full[:-1]
    inv_scale = 1.0 / max(1e-12, scale)
    j_full = np.empty((k + 1, k), dtype=np.float64)
    j_full[:-1, :] = np.diag(p) - np.outer(p, p) * inv_scale
    j_full[-1, :] = -s_full[-1] * p * inv_scale
    return j_full


def _build_time_grid(
    earliest_poll: date,
    as_of: date,
    horizon_date: date,
    poll_dates: Sequence[date],
    lookback_days: int = 180,
) -> list[date]:
    """Build compact non-uniform grid: dense daily near campaign end + all poll midpoints."""
    start = max(earliest_poll, as_of - timedelta(days=lookback_days))
    dates_set: set[date] = {start, as_of, horizon_date}
    dense_start = max(start, as_of - timedelta(days=50))
    cur = start
    while cur < dense_start:
        dates_set.add(cur)
        cur += timedelta(days=5)
    cur = dense_start
    while cur <= horizon_date:
        dates_set.add(cur)
        cur += timedelta(days=1)
    for d in poll_dates:
        if start <= d <= horizon_date:
            dates_set.add(d)
    return sorted(dates_set)


def smooth_state_space(
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
    lookback_days: int = 180,
    n_iter: int = 5,
) -> Stage1Result:
    """Run Stage 1B Iterated Extended RTS Kalman Smoother in ALR coordinates."""
    t_elec = target_election_date or resolve_target_election_date(election_id)
    horizon_date = max(as_of, t_elec)
    active_polls = filter_polls_as_of(
        polls, as_of, exclude_cis_headline=exclude_cis_headline
    )
    if not active_polls:
        raise ValueError(f"No polls available <= {as_of} (exclude_cis={exclude_cis_headline})")

    raw_parties = tracked_parties or _infer_tracked_parties(election_id, active_polls, as_of)
    parties = tuple(p for p in raw_parties if p != "OTROS")
    kernel_seed = aggregate_kernel(
        active_polls,
        as_of,
        election_id=election_id,
        target_election_date=t_elec,
        ratings=ratings,
        tracked_parties=parties,
        hyperparams=hyperparams,
        exclude_cis_headline=exclude_cis_headline,
        reference_pollsters=reference_pollsters,
        blank_share=blank_share,
    )

    k = len(parties)
    scale = 1.0 - blank_share
    ratings_map = {r.pollster: r for r in ratings} if ratings else {}
    house_effects = kernel_seed.house_effects
    p_idx = {p: idx for idx, p in enumerate(parties)}

    # Keep polls within lookback window for state-space smoothing
    start_bound = as_of - timedelta(days=lookback_days)
    ss_polls = [p for p in active_polls if p.midpoint_date >= start_bound]
    if not ss_polls:
        ss_polls = active_polls[-min(25, len(active_polls)) :]

    overlap_u = compute_wave_overlap_penalties(ss_polls)
    grid = _build_time_grid(
        ss_polls[0].midpoint_date,
        as_of,
        horizon_date,
        [p.midpoint_date for p in ss_polls],
        lookback_days=lookback_days,
    )
    grid_ord = [d.toordinal() for d in grid]
    ord_to_t = {o: idx for idx, o in enumerate(grid_ord)}
    n_t = len(grid)
    as_of_idx = ord_to_t[as_of.toordinal()]
    horizon_idx = ord_to_t[horizon_date.toordinal()]

    # Build sparse measurement specs (C_i, y_i, delta_i, var_diag_i) per poll
    polls_at_t: dict[int, list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = (
        defaultdict(list)
    )

    obs_scale_sq = hyperparams.state_space_obs_scale ** 2
    for pi, poll in enumerate(ss_polls):
        t_step = ord_to_t.get(poll.midpoint_date.toordinal())
        if t_step is None:
            continue
        cap = (
            hyperparams.sample_size_cap_cis
            if poll.is_cis
            else hyperparams.sample_size_cap_general
        )
        n_eff = float(min(poll.sample_size or 1000, cap))
        rating = ratings_map.get(poll.pollster)
        if rating is not None:
            qual = max(0.10, rating.grade_weight) * max(0.25, rating.herding_penalty)
        else:
            qual = 0.45 if poll.is_cis else 1.0
        eff_weight = max(0.02, float(overlap_u[pi]) * qual)

        he = house_effects.get(poll.pollster, {})
        c_rows: list[np.ndarray] = []
        y_vals: list[float] = []
        d_vals: list[float] = []
        var_vals: list[float] = []
        has_explicit_fa_or_fv = "FA" in poll.party_shares or "FA_FV" in poll.party_shares

        for party, raw_share in poll.party_shares.items():
            if party in p_idx:
                col = p_idx[party]
                row = np.zeros(k, dtype=np.float64)
                row[col] = 1.0
                c_rows.append(row)
                y_vals.append(raw_share)
                d_vals.append(he.get(party, 0.0))
                p_safe = max(0.004, min(0.95, raw_share))
                srs = (p_safe * (1.0 - p_safe)) / max(100.0, n_eff)
                var_vals.append((srs + obs_scale_sq * (p_safe + 0.03) ** 1.1) / eff_weight)
            elif party == "FA_FV" and "FA" in p_idx and "COMPROMIS_FV" in p_idx:
                if "FA" not in poll.party_shares and "COMPROMIS_FV" not in poll.party_shares:
                    row = np.zeros(k, dtype=np.float64)
                    row[p_idx["FA"]] = 1.0
                    row[p_idx["COMPROMIS_FV"]] = 1.0
                    c_rows.append(row)
                    y_vals.append(raw_share)
                    d_vals.append(he.get("FA", 0.0) + he.get("COMPROMIS_FV", 0.0))
                    p_safe = max(0.005, min(0.95, raw_share))
                    srs = (p_safe * (1.0 - p_safe)) / max(100.0, n_eff)
                    var_vals.append((srs + obs_scale_sq * (p_safe + 0.03) ** 1.1) / eff_weight)
            elif party == "SUMAR" and not has_explicit_fa_or_fv and "FA" in p_idx:
                row = np.zeros(k, dtype=np.float64)
                row[p_idx["FA"]] = 1.0
                include_fv = "COMPROMIS_FV" in p_idx and "COMPROMIS_FV" not in poll.party_shares
                if include_fv:
                    row[p_idx["COMPROMIS_FV"]] = 1.0
                if poll.fieldwork_end < date(2024, 1, 1):
                    c_rows.append(row)
                    y_adj = raw_share * 0.80
                    y_vals.append(y_adj)
                    d_vals.append(
                        he.get("FA", 0.0) + (he.get("COMPROMIS_FV", 0.0) if include_fv else 0.0)
                    )
                    p_safe = max(0.005, min(0.95, y_adj))
                    srs = (p_safe * (1.0 - p_safe)) / max(100.0, n_eff)
                    var_vals.append((srs + obs_scale_sq * 0.25) / (0.65 * eff_weight))
                else:
                    # 2024–Oct 2026 polls report Podemos & Adelante separately -> SUMAR = FA + FV
                    c_rows.append(row)
                    y_vals.append(raw_share)
                    d_vals.append(
                        he.get("FA", 0.0) + (he.get("COMPROMIS_FV", 0.0) if include_fv else 0.0)
                    )
                    p_safe = max(0.005, min(0.95, raw_share))
                    srs = (p_safe * (1.0 - p_safe)) / max(100.0, n_eff)
                    var_vals.append(
                        (srs + obs_scale_sq * ((p_safe + 0.03) ** 1.1 + 0.08))
                        / (0.75 * eff_weight)
                    )

        if c_rows:
            polls_at_t[t_step].append(
                (
                    np.vstack(c_rows),
                    np.asarray(y_vals, dtype=np.float64),
                    np.asarray(d_vals, dtype=np.float64),
                    np.asarray(var_vals, dtype=np.float64),
                )
            )

    # Initial ALR trajectory seeded from kernel_seed mean_shares
    otros_0 = max(1e-4, kernel_seed.mean_shares.get("OTROS", 0.02))
    x_init = np.array(
        [math.log(max(1e-4, kernel_seed.mean_shares[p]) / otros_0) for p in parties],
        dtype=np.float64,
    )
    x_nom = np.tile(x_init, (n_t, 1))

    rw_var_day = hyperparams.state_space_rw_var_per_day
    # Diffuse initial covariance around kernel mean at start_date
    p0 = np.eye(k, dtype=np.float64) * 0.25
    eye_k = np.eye(k, dtype=np.float64)

    x_filt = np.zeros((n_t, k), dtype=np.float64)
    p_filt = np.zeros((n_t, k, k), dtype=np.float64)
    x_pred = np.zeros((n_t, k), dtype=np.float64)
    p_pred = np.zeros((n_t, k, k), dtype=np.float64)

    for _it in range(n_iter):
        x_cur = x_nom[0].copy()
        p_cur = p0.copy()
        for t in range(n_t):
            if t == 0:
                xp = x_cur
                pp = p_cur
            else:
                dt = max(1.0, float(grid_ord[t] - grid_ord[t - 1]))
                xp = x_filt[t - 1]
                pp = p_filt[t - 1] + (dt * rw_var_day) * eye_k

            x_pred[t] = xp
            p_pred[t] = 0.5 * (pp + pp.T)

            # Gauss-Newton / Information update at step t using linearization x_nom[t]
            x_lin = x_nom[t]
            s_full_lin = _alr_to_cand_probs(x_lin, scale)
            j_top = _alr_jacobian_full(s_full_lin)[:-1, :]  # (K, K)
            pi_top_lin = s_full_lin[:-1]

            obs_list = polls_at_t.get(t)
            if not obs_list:
                x_filt[t] = xp
                p_filt[t] = p_pred[t]
                continue

            # Information form update: P_filt^-1 = P_pred^-1 + sum H^T R^-1 H
            pp_inv = np.linalg.inv(p_pred[t] + 1e-10 * eye_k)
            info_mat = pp_inv.copy()
            info_rhs = pp_inv @ xp
            for c_mat, y_obs, d_off, var_vec in obs_list:
                h_mat = c_mat @ j_top
                r_inv = 1.0 / np.maximum(1e-9, var_vec)
                # y_adj = C*pi(x_lin) + delta_obs + H*(x - x_lin)
                pred_y = c_mat @ pi_top_lin + d_off
                y_eff = y_obs - pred_y + h_mat @ x_lin
                hw = h_mat.T * r_inv[None, :]
                info_mat += hw @ h_mat
                info_rhs += hw @ y_eff

            pf = np.linalg.inv(0.5 * (info_mat + info_mat.T) + 1e-10 * eye_k)
            pf = 0.5 * (pf + pf.T)
            xf = pf @ info_rhs
            x_filt[t] = xf
            p_filt[t] = pf

        # Rauch-Tung-Striebel backward pass
        x_smooth = np.zeros_like(x_filt)
        p_smooth = np.zeros_like(p_filt)
        x_smooth[-1] = x_filt[-1]
        p_smooth[-1] = p_filt[-1]
        for t in range(n_t - 2, -1, -1):
            p_next_inv = np.linalg.inv(p_pred[t + 1] + 1e-10 * eye_k)
            gain = p_filt[t] @ p_next_inv
            x_smooth[t] = x_filt[t] + gain @ (x_smooth[t + 1] - x_pred[t + 1])
            ps = p_filt[t] + gain @ (p_smooth[t + 1] - p_pred[t + 1]) @ gain.T
            p_smooth[t] = 0.5 * (ps + ps.T)

        if _it < n_iter - 1:
            # Damped step for guaranteed monotonic softmax convergence across outer iterations
            x_nom = 0.25 * x_nom + 0.75 * x_smooth

    # Evaluate final state at horizon_date (incorporating blackout drift beyond as_of)
    x_eval = x_smooth[horizon_idx]
    p_eval_alr = p_smooth[horizon_idx]
    # Add post-as_of horizon drift if target_election_date > grid[-1]
    extra_days = max(0, (t_elec - grid[horizon_idx]).days)
    if extra_days > 0:
        p_eval_alr = p_eval_alr + float(extra_days) * rw_var_day * eye_k

    # Guard against tiny non-poll drift when horizon == as_of vs > as_of:
    _ = as_of_idx
    s_final = _alr_to_cand_probs(x_eval, scale)
    j_full = _alr_jacobian_full(s_final)
    cov_shares = j_full @ p_eval_alr @ j_full.T
    cov_shares = 0.5 * (cov_shares + cov_shares.T)

    full_parties = (*parties, "OTROS")
    mean_shares = {full_parties[i]: float(s_final[i]) for i in range(len(full_parties))}
    sd_shares = {
        full_parties[i]: float(math.sqrt(max(1e-12, cov_shares[i, i])))
        for i in range(len(full_parties))
    }

    # Trend series from smoothed trajectory over trailing 100 days
    trend_start = as_of - timedelta(days=100)
    ribbon_parties = list(parties[: min(8, len(parties))])
    t_indices = [i for i, d in enumerate(grid) if trend_start <= d <= as_of]
    # Subsample smoothly to <= 36 points matching kernel cadence
    if len(t_indices) > 36:
        step = max(1, len(t_indices) // 35)
        kept = t_indices[::step]
        if kept[-1] != t_indices[-1]:
            kept.append(t_indices[-1])
        t_indices = kept

    trend_parties: dict[str, dict[str, list[float]]] = {
        p: {"mean": [], "p10": [], "p90": []} for p in ribbon_parties
    }
    for ti in t_indices:
        s_t = _alr_to_cand_probs(x_smooth[ti], scale)
        j_t = _alr_jacobian_full(s_t)
        cov_t = j_t @ p_smooth[ti] @ j_t.T
        for p in ribbon_parties:
            col = parties.index(p)
            mu = float(s_t[col])
            se = math.sqrt(max(1e-10, float(cov_t[col, col])))
            trend_parties[p]["mean"].append(round(mu, 5))
            trend_parties[p]["p10"].append(round(max(0.0, mu - 1.28155 * se), 5))
            trend_parties[p]["p90"].append(round(min(1.0, mu + 1.28155 * se), 5))

    days_to_elec = (t_elec - as_of).days
    half_life = (
        hyperparams.half_life_campaign_days
        if days_to_elec <= 60
        else hyperparams.half_life_quiet_days
    )
    decay_rate = math.log(2.0) / max(1.0, half_life)
    ss_w = np.array(
        [
            math.exp(-decay_rate * min(240.0, max(0.0, float((as_of - p.midpoint_date).days))))
            * _non_recency_poll_weight(p, float(overlap_u[pi]), ratings_map, hyperparams)
            for pi, p in enumerate(ss_polls)
        ],
        dtype=np.float64,
    )
    w_sum_ss = float(ss_w.sum())
    w_sq_ss = float(np.sum(ss_w**2))
    n_eff = float((w_sum_ss * w_sum_ss) / max(1e-12, w_sq_ss)) if w_sum_ss > 0.0 else 1.0

    return Stage1Result(
        as_of_date=as_of,
        target_election_date=t_elec,
        engine="state_space",
        parties=full_parties,
        mean_shares=mean_shares,
        blank_share=blank_share,
        sd_shares=sd_shares,
        cov_shares=cov_shares,
        house_effects=house_effects,
        reference_pollsters=kernel_seed.reference_pollsters,
        n_polls_used=len(ss_polls),
        n_effective_polls=n_eff,
        exclude_cis_headline=exclude_cis_headline,
        poll_diagnostics=kernel_seed.poll_diagnostics,
        trend_series={
            "dates": [grid[ti].isoformat() for ti in t_indices],
            "parties": trend_parties,
            "polls": kernel_seed.trend_series.get("polls", []),
        },
    )
