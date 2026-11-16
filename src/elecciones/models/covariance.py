"""Stage 3A: Hierarchical National + CCAA + Provincial Student-t Error & Monte Carlo Engine.

Combines:
1. Stage 1 posterior estimation covariance (mapped cleanly onto log-ratio space
   with optional remaining horizon random-walk uncertainty propagation).
2. Shared right-bloc and left-bloc correlated electoral swings.
3. Heavy-tailed multivariate Student-t (nu ~= 5) shocks decomposed hierarchically across:
   - National systematic polling misses (19-CCAA + 52-district shared shift)
   - Autonomous Community (CCAA) correlated regional swings (19 regions)
   - Provincial idiosyncratic log-ratio shocks (restricted strictly to active M_{c,p}==1 cells)
4. Stochastic provincial turnout and blank-ballot variation.
5. Zero-allocation active-cell compressed vectorized D'Hondt execution across 50,000 draws
   completing well within the 1.5s CPU SLA.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from elecciones.config import (
    DEFAULT_HYPERPARAMETERS,
    DEFAULT_N_SIMS,
    SIMULATION_SEED,
    ModelHyperparameters,
)
from elecciones.models.kernel_aggregator import Stage1Result
from elecciones.models.provincial import Stage2Projection
from elecciones.schemas import PartiesCatalog

_RIGHT_BLOC_PARTIES: Final[frozenset[str]] = frozenset(
    {"PP", "UPN_PP", "VOX", "SALF", "NA_SUMA", "UPN"}
)
_LEFT_BLOC_PARTIES: Final[frozenset[str]] = frozenset(
    {
        "PSOE",
        "FA",
        "PODEMOS",
        "COMPROMIS_FV",
        "UPC",
        "ARA_MES",
        "ADELANTE",
        "SUMAR",
        "UP",
        "MP",
        "IU",
        "COMPROMIS",
    }
)


@dataclass(frozen=True, slots=True)
class HierarchicalCovarianceConfig:
    """Calibrated parameters for hierarchical log-ratio Student-t simulation."""

    student_t_df: float = 5.0
    nat_scale: float = 0.085
    ccaa_scale: float = 0.060
    prov_scale: float = 0.050
    bloc_right_scale: float = 0.030
    bloc_left_scale: float = 0.030
    stage1_cov_weight: float = 1.0
    blackout_rw_var_per_day: float = DEFAULT_HYPERPARAMETERS.state_space_rw_var_per_day
    turnout_nat_sd: float = 0.14
    turnout_prov_sd: float = 0.10
    blank_log_sd: float = 0.22

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> HierarchicalCovarianceConfig:
        defaults = cls()
        return cls(
            student_t_df=float(data.get("student_t_df", defaults.student_t_df)),
            nat_scale=float(data.get("nat_scale", defaults.nat_scale)),
            ccaa_scale=float(data.get("ccaa_scale", defaults.ccaa_scale)),
            prov_scale=float(data.get("prov_scale", defaults.prov_scale)),
            bloc_right_scale=float(data.get("bloc_right_scale", defaults.bloc_right_scale)),
            bloc_left_scale=float(data.get("bloc_left_scale", defaults.bloc_left_scale)),
            stage1_cov_weight=float(data.get("stage1_cov_weight", defaults.stage1_cov_weight)),
            blackout_rw_var_per_day=float(
                data.get("blackout_rw_var_per_day", defaults.blackout_rw_var_per_day)
            ),
            turnout_nat_sd=float(data.get("turnout_nat_sd", defaults.turnout_nat_sd)),
            turnout_prov_sd=float(data.get("turnout_prov_sd", defaults.turnout_prov_sd)),
            blank_log_sd=float(data.get("blank_log_sd", defaults.blank_log_sd)),
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "student_t_df": float(self.student_t_df),
            "nat_scale": float(self.nat_scale),
            "ccaa_scale": float(self.ccaa_scale),
            "prov_scale": float(self.prov_scale),
            "bloc_right_scale": float(self.bloc_right_scale),
            "bloc_left_scale": float(self.bloc_left_scale),
            "stage1_cov_weight": float(self.stage1_cov_weight),
            "blackout_rw_var_per_day": float(self.blackout_rw_var_per_day),
            "turnout_nat_sd": float(self.turnout_nat_sd),
            "turnout_prov_sd": float(self.turnout_prov_sd),
            "blank_log_sd": float(self.blank_log_sd),
        }


@dataclass(frozen=True, slots=True)
class SimulationArrays:
    """Compact simulation outputs across ``n_sims`` draws & 52 constituencies."""

    n_sims: int
    seed: int
    runtime_ms: float
    party_ids: tuple[str, ...]
    national_seats: np.ndarray
    national_shares: np.ndarray
    prov_seats: np.ndarray
    prov_share_mean: np.ndarray
    prov_share_p10: np.ndarray
    prov_share_p90: np.ndarray


def build_stage1_log_factor(
    stage1_result: Stage1Result | None,
    proj: Stage2Projection,
    *,
    extra_days: int = 0,
    cov_weight: float = 1.0,
    rw_var_per_day: float = 0.0014,
) -> np.ndarray:
    """Build low-rank PSD factor ``L`` (shape ``P x K``) for Stage 1 log-ratio uncertainty."""
    party_ids = proj.party_ids
    n_p = len(party_ids)
    cov_log = np.zeros((n_p, n_p), dtype=np.float64)

    if stage1_result is not None and cov_weight > 0.0:
        s1_parties = stage1_result.parties
        s1_idx = {p: k for k, p in enumerate(s1_parties)}
        k_s1 = len(s1_parties)
        means_s1 = np.array(
            [max(0.015, float(stage1_result.mean_shares.get(p, 0.015))) for p in s1_parties],
            dtype=np.float64,
        )
        # Delta-method Jacobian scaling: cov(log s_a, log s_b) ~= cov(s_a, s_b) / (m_a * m_b)
        inv_m = 1.0 / means_s1
        s1_log_cov = np.asarray(stage1_result.cov_shares, dtype=np.float64) * np.outer(
            inv_m, inv_m
        )
        s1_log_cov *= float(cov_weight)

        # Map Stage 2 party columns onto Stage 1 indices (UPN_PP inherits PP's national factor)
        map_to_s1: list[int | None] = []
        for pid in party_ids:
            if pid in s1_idx:
                map_to_s1.append(s1_idx[pid])
            elif pid == "UPN_PP" and "PP" in s1_idx:
                map_to_s1.append(s1_idx["PP"])
            elif pid in {"UPC", "ARA_MES"} and "FA" in s1_idx:
                map_to_s1.append(s1_idx["FA"])
            elif pid == "GEROA_BAI" and "PNV" in s1_idx:
                map_to_s1.append(s1_idx["PNV"])
            else:
                map_to_s1.append(None)

        for a in range(n_p):
            ia = map_to_s1[a]
            if ia is None or ia >= k_s1:
                continue
            for b in range(n_p):
                ib = map_to_s1[b]
                if ib is None or ib >= k_s1:
                    continue
                cov_log[a, b] = s1_log_cov[ia, ib]

    # Add modest blackout horizon growth if Stage 1 did not already include it (e.g. pure kernel)
    if (
        extra_days > 0
        and rw_var_per_day > 0.0
        and (stage1_result is None or stage1_result.engine == "kernel")
    ):
        ridge = float(extra_days) * float(rw_var_per_day) * 0.45
        cov_log[np.arange(n_p), np.arange(n_p)] += ridge

    # Symmetrize & spectral clip cleanly to avoid extreme variances on minor parties
    cov_log = 0.5 * (cov_log + cov_log.T)
    diag = np.diag(cov_log)
    max_var = 0.22**2
    exceed = np.where(diag > max_var)[0]
    if exceed.size > 0:
        scale_d = np.ones(n_p, dtype=np.float64)
        scale_d[exceed] = np.sqrt(max_var / diag[exceed])
        cov_log *= np.outer(scale_d, scale_d)

    eigvals, eigvecs = np.linalg.eigh(cov_log)
    keep = eigvals > 1e-10
    if not np.any(keep):
        return np.zeros((n_p, 1), dtype=np.float32)
    factor = eigvecs[:, keep] * np.sqrt(eigvals[keep])[None, :]
    return factor.astype(np.float32)


def estimate_empirical_error_hierarchy(
    pred_actual_pairs: Sequence[tuple[Stage2Projection, Mapping[str, Any]]],
    *,
    min_share_threshold: float = 0.025,
) -> dict[str, float]:
    """Decompose out-of-sample log-ratio errors into National + CCAA + Provincial scales."""
    nat_errs: list[float] = []
    ccaa_errs: list[float] = []
    prov_errs: list[float] = []

    for proj, act_by_cpro in pred_actual_pairs:
        p_list = proj.party_ids
        n_c = len(proj.cpros)
        n_p = len(p_list)
        err_mat = np.full((n_c, n_p), np.nan, dtype=np.float64)
        for i, cpro in enumerate(proj.cpros):
            ar = act_by_cpro[cpro]
            val = float(ar.validos)
            for j, pid in enumerate(p_list):
                if not proj.active_mask[i, j]:
                    continue
                actual_s = float(ar.party_votes.get(pid, 0)) / val
                pred_s = float(proj.final_matrix[i, j])
                if actual_s >= min_share_threshold and pred_s >= min_share_threshold:
                    err_mat[i, j] = math.log(actual_s / pred_s)

        w = proj.turnout_weights
        for j in range(n_p):
            col = err_mat[:, j]
            ok = np.isfinite(col)
            if int(ok.sum()) < 3:
                continue
            mu_n = float(np.average(col[ok], weights=w[ok]))
            nat_errs.append(mu_n)
            res = col - mu_n
            for cod in set(proj.codautos):
                idx = [
                    i for i, c in enumerate(proj.codautos) if c == cod and np.isfinite(res[i])
                ]
                if len(idx) >= 2:
                    cm = float(np.average(res[idx], weights=w[idx]))
                    ccaa_errs.append(cm)
                    for ii in idx:
                        prov_errs.append(float(res[ii] - cm))
                elif len(idx) == 1:
                    prov_errs.append(float(res[idx[0]]))

    def _safe_sd(vals: list[float], default: float) -> float:
        return float(np.std(vals, ddof=1)) if len(vals) >= 3 else default

    def _safe_mad(vals: list[float], default: float) -> float:
        return float(1.4826 * np.median(np.abs(vals))) if vals else default

    return {
        "n_nat_obs": float(len(nat_errs)),
        "n_ccaa_obs": float(len(ccaa_errs)),
        "n_prov_obs": float(len(prov_errs)),
        "nat_rmse": round(_safe_sd(nat_errs, 0.14), 5),
        "nat_mad": round(_safe_mad(nat_errs, 0.11), 5),
        "ccaa_rmse": round(_safe_sd(ccaa_errs, 0.08), 5),
        "ccaa_mad": round(_safe_mad(ccaa_errs, 0.06), 5),
        "prov_rmse": round(_safe_sd(prov_errs, 0.05), 5),
        "prov_mad": round(_safe_mad(prov_errs, 0.04), 5),
    }


def _resolve_bloc_vectors(
    party_ids: Sequence[str],
    parties_catalog: PartiesCatalog | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean indicator vectors for right-bloc and left-bloc correlated shifts."""
    n_p = len(party_ids)
    r_vec = np.zeros(n_p, dtype=np.float32)
    l_vec = np.zeros(n_p, dtype=np.float32)
    cat_map = {p.id: p for p in parties_catalog.parties} if parties_catalog is not None else {}

    for j, pid in enumerate(party_ids):
        if pid in _RIGHT_BLOC_PARTIES:
            r_vec[j] = 1.0
        elif pid in _LEFT_BLOC_PARTIES:
            l_vec[j] = 1.0
        elif pid in cat_map:
            if cat_map[pid].bloc == "right":
                r_vec[j] = 1.0
            elif cat_map[pid].bloc == "left":
                l_vec[j] = 1.0
    return r_vec, l_vec


def run_hierarchical_monte_carlo(
    proj: Stage2Projection,
    *,
    stage1_result: Stage1Result | None = None,
    config: HierarchicalCovarianceConfig | None = None,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int = SIMULATION_SEED,
    parties_catalog: PartiesCatalog | None = None,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
    n_workers: int | None = None,
    non_seat_parties: Sequence[str] = ("OTROS",),
) -> SimulationArrays:
    """Execute hierarchical Student-t Monte Carlo draws + exact LOREG D'Hondt across 52 districts.

    Guarantees:
    - Exact zero probability outside ``proj.active_mask`` (``M_{c,p} == 0 => seats == 0``)
    - Catch-all residual buckets in ``non_seat_parties`` (default ``("OTROS",)``) stay in
      valid vote shares/denominators without eligibility to clear thresholds or win seats
    - Reproducible across identical ``(seed, n_sims, config)`` calls via ``SeedSequence``
    - Completes ``50,000`` draws x ``52`` constituencies in ``< 1.5s`` on standard CPUs.
    """
    if n_sims < 1:
        raise ValueError(f"n_sims must be >= 1, got {n_sims}")
    cfg = config or HierarchicalCovarianceConfig()

    t_start = time.perf_counter()
    n_c = len(proj.cpros)
    n_p = len(proj.party_ids)
    mask = np.asarray(proj.active_mask, dtype=bool)
    act_r, act_c = np.nonzero(mask)
    n_act = int(act_r.size)

    unique_cods = tuple(dict.fromkeys(proj.codautos))
    cod_to_idx = {c: idx for idx, c in enumerate(unique_cods)}
    n_ccaa = len(unique_cods)
    c_to_ccaa = np.array([cod_to_idx[c] for c in proj.codautos], dtype=np.intp)
    act_ccaa = c_to_ccaa[act_r]

    # Compress CCAA shocks strictly to active (ccaa, party) pairs to cut RNG cost by ~60%
    ccaa_pair_lookup = np.full((n_ccaa, n_p), -1, dtype=np.intp)
    unique_pairs = np.unique(act_ccaa * n_p + act_c)
    cc_pair_r = unique_pairs // n_p
    n_ccaa_act = int(unique_pairs.size)
    ccaa_pair_lookup[cc_pair_r, unique_pairs % n_p] = np.arange(n_ccaa_act, dtype=np.intp)
    act_ccaa_pair = ccaa_pair_lookup[act_ccaa, act_c]

    active_cols = [np.flatnonzero(mask[c]) for c in range(n_c)]
    c_counts = mask.sum(axis=1)
    c_starts = np.concatenate(([0], np.cumsum(c_counts)))
    p_acts = [np.flatnonzero(act_c == j) for j in range(n_p)]
    p_rows = [act_r[pa] for pa in p_acts]
    active_p = tuple(j for j in range(n_p) if p_acts[j].size > 0)

    is_city_arr = np.asarray(proj.city_vec, dtype=bool)
    seats_vec = np.asarray(proj.seats_vec, dtype=np.int32)
    seats_a = seats_vec[act_r][:, None]
    non_seat_set = frozenset(non_seat_parties)
    non_seat_a = np.array([proj.party_ids[j] in non_seat_set for j in act_c], dtype=bool)
    thresh_a = np.where(
        non_seat_a,
        np.float32(1.01),
        np.where(is_city_arr[act_r], np.float32(0.0), np.float32(0.03)),
    )[:, None]

    base_log_a = np.log(np.maximum(1e-9, proj.final_matrix[act_r, act_c])).astype(np.float32)
    base_blank_32 = np.maximum(1e-4, proj.final_blank).astype(np.float32)
    base_censo_32 = proj.censo_vec.astype(np.float32)
    t_clip = np.clip(proj.turnout_mean_vec, 0.05, 0.95)
    base_logit_turn_32 = np.log(t_clip / (1.0 - t_clip)).astype(np.float32)

    extra_days = 0
    if stage1_result is not None:
        extra_days = max(0, (stage1_result.target_election_date - stage1_result.as_of_date).days)
    l_s1_t = build_stage1_log_factor(
        stage1_result,
        proj,
        extra_days=extra_days,
        cov_weight=cfg.stage1_cov_weight,
        rw_var_per_day=hyperparams.state_space_rw_var_per_day,
    ).T  # (K, P)

    r_vec, l_vec = _resolve_bloc_vectors(proj.party_ids, parties_catalog)
    r_shift = r_vec * np.float32(cfg.bloc_right_scale)
    l_shift = l_vec * np.float32(cfg.bloc_left_scale)

    nu = max(2.5, float(cfg.student_t_df))
    nu_32 = np.float32(nu)
    nat_scale_32 = np.float32(cfg.nat_scale)
    ccaa_scale_32 = np.float32(cfg.ccaa_scale)
    prov_scale_32 = np.float32(cfg.prov_scale)
    turn_nat_32 = np.float32(cfg.turnout_nat_sd)
    turn_prov_32 = np.float32(cfg.turnout_prov_sd)
    blank_sd_32 = np.float32(cfg.blank_log_sd)

    prov_seats_out = np.zeros((n_sims, n_c, n_p), dtype=np.int8)
    nat_seats_out = np.zeros((n_sims, n_p), dtype=np.int16)
    nat_shares_out = np.zeros((n_sims, n_p), dtype=np.float32)

    q_stride = max(1, n_sims // 4_000)
    q_indices = np.arange(0, n_sims, q_stride, dtype=np.intp)
    q_samples_a = np.zeros((q_indices.size, n_act), dtype=np.float32)

    batch_size = 5000 if n_sims >= 10_000 else max(1, min(n_sims, 2500))
    ranges = [(i, min(n_sims, i + batch_size)) for i in range(0, n_sims, batch_size)]
    n_chunks = len(ranges)

    seed_seq = np.random.SeedSequence(seed)
    child_seeds = seed_seq.spawn(n_chunks)
    chunk_share_sums = np.zeros((n_chunks, n_act), dtype=np.float64)

    def _run_chunk(task: tuple[int, int, int, np.random.SeedSequence]) -> None:
        c_idx, s_i, e_i, c_seed = task
        m = e_i - s_i
        rng = np.random.default_rng(c_seed)

        # Heavy-tailed Student-t radial multipliers: sqrt(nu / chi2_nu)
        g_nat = np.sqrt(nu_32 / rng.chisquare(nu, size=(m, 1)).astype(np.float32))
        g_cc = np.sqrt(nu_32 / rng.chisquare(nu, size=(m, n_ccaa)).astype(np.float32))
        g_pr = np.sqrt(nu_32 / rng.chisquare(nu, size=(m, n_c)).astype(np.float32))

        # Party-level national total shift (Stage 1 covariance + blocs + national t shock)
        s1_lat = rng.standard_normal((m, l_s1_t.shape[0]), dtype=np.float32)
        nat_tot = (
            (s1_lat @ l_s1_t)
            + rng.standard_normal((m, n_p), dtype=np.float32) * (nat_scale_32 * g_nat)
            + (rng.standard_normal((m, 1), dtype=np.float32) * g_nat) * r_shift[None, :]
            + (rng.standard_normal((m, 1), dtype=np.float32) * g_nat) * l_shift[None, :]
        )

        z_cc = rng.standard_normal((m, n_ccaa_act), dtype=np.float32) * ccaa_scale_32
        z_cc *= g_cc[:, cc_pair_r]
        z_pr = rng.standard_normal((m, n_act), dtype=np.float32) * prov_scale_32
        z_pr *= g_pr[:, act_r]

        eta = base_log_a[None, :] + nat_tot[:, act_c] + z_cc[:, act_ccaa_pair] + z_pr
        np.clip(eta, np.float32(-22.0), np.float32(5.0), out=eta)
        np.exp(eta, out=eta)

        # Stochastic blank ballot & active-cell simplex normalization per (sim, constituency)
        blank_jit = rng.standard_normal((m, n_c), dtype=np.float32) * blank_sd_32
        blank_mat = np.clip(
            base_blank_32[None, :] * np.exp(blank_jit), np.float32(0.002), np.float32(0.06)
        )
        rsums = np.add.reduceat(eta, c_starts[:-1], axis=1)
        scale_c = (np.float32(1.0) - blank_mat) / np.maximum(np.float32(1e-12), rsums)
        eta *= scale_c[:, act_r]

        # Exact active-cell sum for mean shares + strided slice for provincial p10/p90
        chunk_share_sums[c_idx] = eta.sum(axis=0, dtype=np.float64)
        q_start = int(np.searchsorted(q_indices, s_i, side="left"))
        q_end = int(np.searchsorted(q_indices, e_i, side="left"))
        if q_end > q_start:
            local_q = q_indices[q_start:q_end] - s_i
            q_samples_a[q_start:q_end] = eta[local_q]

        # Stochastic provincial turnout weights -> exact national valid-vote weighted shares
        t_nat = rng.standard_normal((m, 1), dtype=np.float32) * turn_nat_32
        t_pr = rng.standard_normal((m, n_c), dtype=np.float32) * turn_prov_32
        t_logit = base_logit_turn_32[None, :] + t_nat + t_pr
        turn_m = np.float32(1.0) / (np.float32(1.0) + np.exp(-t_logit))
        valid_ballots = turn_m * base_censo_32[None, :]
        w_dist = valid_ballots / valid_ballots.sum(axis=1, keepdims=True)
        for j in active_p:
            nat_shares_out[s_i:e_i, j] = np.einsum(
                "ma,ma->m", eta[:, p_acts[j]], w_dist[:, p_rows[j]]
            )

        # Contiguous active-row D'Hondt apportionment (zero 3D buffer allocations)
        if n_act == 0:
            return
        eta_t = np.ascontiguousarray(eta.T, dtype=np.float32)  # (n_act, m)
        val_t = np.add.reduceat(eta_t, c_starts[:-1], axis=0) + blank_mat.T
        q_all = np.where(eta_t >= thresh_a * val_t[act_r], eta_t, np.float32(0.0))
        q_sums = np.add.reduceat(q_all, c_starts[:-1], axis=0)
        q_safe = np.maximum(np.float32(1e-30), q_sums)
        s_all = np.maximum(
            0,
            np.floor(q_all * (seats_a / q_safe[act_r]) - np.float32(1e-4)).astype(np.int32),
        )
        s_all[:, :] = np.where(q_all > np.float32(0.0), s_all, 0)
        rem_all = seats_vec[:, None] - np.add.reduceat(s_all, c_starts[:-1], axis=0)
        cq_all = q_all / (s_all + 1)

        nat_acc = np.zeros((n_p, m), dtype=np.int16)
        row_arange = np.arange(m, dtype=np.intp)
        for c in range(n_c):
            cs, ce = int(c_starts[c]), int(c_starts[c + 1])
            if cs == ce:
                continue
            aj = active_cols[c]
            sc = int(seats_vec[c])
            if bool(is_city_arr[c]):
                city_sub = np.where(non_seat_a[cs:ce, None], np.float32(-1.0), eta_t[cs:ce])
                if not np.any(city_sub > np.float32(0.0)):
                    continue
                w = np.argmax(city_sub, axis=0)
                prov_seats_out[s_i + row_arange, c, aj[w]] = np.int8(sc)
                nat_acc[aj[w], row_arange] += np.int16(sc)
                continue
            nz_c = q_sums[c] > np.float32(0.0)
            if not np.any(nz_c):
                continue
            s = s_all[cs:ce]
            if not np.all(nz_c):
                s[:, ~nz_c] = 0
                rem = np.where(nz_c, rem_all[c], 0)
            else:
                rem = rem_all[c]
            mr = int(rem.max())
            if mr > 0:
                cq = cq_all[cs:ce]
                v_c = q_all[cs:ce]
                q1 = cq.ravel()
                v1 = v_c.ravel()
                s1_a = s.ravel()
                for st in range(mr):
                    nd = np.flatnonzero(rem > st)
                    if nd.size == 0:
                        break
                    w = np.argmax(cq[:, nd], axis=0)
                    f = w * m + nd
                    ns = s1_a[f] + 1
                    s1_a[f] = ns
                    q1[f] = v1[f] / (ns + 1)
            prov_seats_out[s_i:e_i, c, aj] = s.T.astype(np.int8)
            nat_acc[aj] += s.astype(np.int16)
        nat_seats_out[s_i:e_i] = nat_acc.T

    eff_workers = (
        max(1, int(n_workers))
        if n_workers is not None
        else (min(2, n_chunks) if n_sims >= 2500 else 1)
    )
    tasks = [(i, r[0], r[1], child_seeds[i]) for i, r in enumerate(ranges)]
    if eff_workers == 1 or n_chunks == 1:
        for t in tasks:
            _run_chunk(t)
    else:
        with ThreadPoolExecutor(max_workers=eff_workers) as pool:
            list(pool.map(_run_chunk, tasks))

    mean_a = chunk_share_sums.sum(axis=0) / float(n_sims)
    prov_share_mean = np.zeros((n_c, n_p), dtype=np.float64)
    prov_share_mean[act_r, act_c] = mean_a

    q_a = np.percentile(q_samples_a, [10.0, 90.0], axis=0)
    prov_share_p10 = np.zeros((n_c, n_p), dtype=np.float64)
    prov_share_p90 = np.zeros((n_c, n_p), dtype=np.float64)
    prov_share_p10[act_r, act_c] = np.minimum(mean_a, q_a[0].astype(np.float64))
    prov_share_p90[act_r, act_c] = np.maximum(mean_a, q_a[1].astype(np.float64))

    runtime_ms = (time.perf_counter() - t_start) * 1000.0
    return SimulationArrays(
        n_sims=int(n_sims),
        seed=int(seed),
        runtime_ms=float(runtime_ms),
        party_ids=proj.party_ids,
        national_seats=nat_seats_out,
        national_shares=nat_shares_out,
        prov_seats=prov_seats_out,
        prov_share_mean=prov_share_mean,
        prov_share_p10=prov_share_p10,
        prov_share_p90=prov_share_p90,
    )
