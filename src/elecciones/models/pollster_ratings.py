"""Historical pollster accuracy ratings, herding detection, and house-effect anchors.

Evaluates polling houses across official Spanish Congress elections
(``2015-12``, ``2016-06``, ``2019-04``, ``2019-11``, ``2023-07``) with optional
``exclude_election_ids`` support for strict leave-one-election-out backtesting,
combined with contemporaneous 2026 deviation and herding checks up to ``as_of``.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Final

import numpy as np

from elecciones.config import (
    DEFAULT_HYPERPARAMETERS,
    HISTORICAL_ELECTION_DATES,
    ModelHyperparameters,
    get_elections_dir,
    get_outputs_dir,
    get_polls_dir,
)
from elecciones.ingest.elections import load_congress_results
from elecciones.ingest.wikipedia import load_national_polls
from elecciones.schemas import PollRecord, PollsterRatingRecord

POLLSTER_RATINGS_COLUMNS: Final[tuple[str, ...]] = (
    "pollster",
    "n_elections",
    "n_polls_2026",
    "shrunk_mae",
    "grade",
    "grade_weight",
    "herding_ratio",
    "herding_penalty",
    "delta_PP",
    "delta_PSOE",
    "delta_VOX",
    "delta_FA",
    "delta_PODEMOS",
)

# Grade-level base multipliers before sample-size and overlap weighting
_GRADE_BASE_WEIGHT: Final[dict[str, float]] = {
    "A": 1.25,
    "B": 1.00,
    "C": 0.45,
}


def _compute_actual_national_pct(
    election_id: str,
    elections_dir: Path,
) -> dict[str, float]:
    """Compute official national party vote shares in ``[0, 100]`` over ``validos``."""
    results = load_congress_results(election_id, elections_dir=elections_dir)
    total_validos = sum(r.validos for r in results)
    if total_validos <= 0:
        return {}
    party_totals: dict[str, int] = defaultdict(int)
    for r in results:
        for p, v in r.party_votes.items():
            party_totals[p] += v
    shares = {p: 100.0 * v / total_validos for p, v in party_totals.items()}
    # Canonical left-of-PSOE coalition bloc across historical cycles
    if election_id == "2023-07":
        shares["LEFT"] = shares.get("SUMAR", 0.0)
    elif election_id == "2015-12":
        shares["LEFT"] = shares.get("PODEMOS", 0.0) + shares.get("IU", 0.0)
    else:
        shares["LEFT"] = shares.get("UP", 0.0)
    return shares


def _extract_historical_bloc_pcts(
    poll: PollRecord,
    election_id: str,
    actual_pct: dict[str, float],
) -> tuple[dict[str, float], dict[str, float]]:
    """Align historical poll shares to actual national blocs for error calculation."""
    p_pct = {k: v * 100.0 for k, v in poll.party_shares.items()}
    errs: dict[str, float] = {}
    for party in ("PP", "PSOE", "VOX", "CS"):
        if party in p_pct and actual_pct.get(party, 0.0) >= 2.0:
            errs[party] = p_pct[party] - actual_pct[party]

    if election_id == "2023-07" and "SUMAR" in p_pct:
        errs["LEFT"] = p_pct["SUMAR"] - actual_pct.get("LEFT", 0.0)
    elif election_id in {"2016-06", "2019-04", "2019-11"} and "UP" in p_pct:
        errs["LEFT"] = p_pct["UP"] - actual_pct.get("LEFT", 0.0)
    elif election_id == "2015-12" and "PODEMOS" in p_pct:
        # In Dec 2015 polls, compare Podemos (or Podemos+IU if IU included)
        if "IU" in p_pct:
            errs["LEFT"] = (p_pct["PODEMOS"] + p_pct["IU"]) - actual_pct.get("LEFT", 0.0)
        else:
            errs["PODEMOS"] = p_pct["PODEMOS"] - actual_pct.get("PODEMOS", 0.0)

    abs_errs = {k: abs(v) for k, v in errs.items()}
    return errs, abs_errs


def _overlap_weights_same_pollster(polls: Sequence[PollRecord]) -> list[float]:
    """Compute within-pollster fieldwork overlap weights ``u_i = 1 / sum_j overlap(i, j)``."""
    n = len(polls)
    if n == 0:
        return []
    weights: list[float] = []
    for i, pi in enumerate(polls):
        si, ei = pi.fieldwork_start, pi.fieldwork_end
        li = max(1, (ei - si).days + 1)
        overlap_sum = 0.0
        for j, pj in enumerate(polls):
            if i == j:
                overlap_sum += 1.0
                continue
            sj, ej = pj.fieldwork_start, pj.fieldwork_end
            inter = min(ei, ej) - max(si, sj)
            inter_days = inter.days + 1
            if inter_days <= 0:
                continue
            lj = max(1, (ej - sj).days + 1)
            overlap_sum += inter_days / float(max(li, lj))
        weights.append(1.0 / max(1.0, overlap_sum))
    return weights


def _estimate_contemporaneous_consensus_2026(
    polls: Sequence[PollRecord],
    reference_pollsters: set[str],
    parties: Sequence[str],
) -> tuple[list[date], dict[str, np.ndarray]]:
    """Build smoothed non-CIS reference consensus trajectory at unique 2026 dates."""
    ref_polls = [
        p
        for p in polls
        if not p.is_cis
        and p.pollster in reference_pollsters
        and p.fieldwork_end >= date(2026, 8, 1)
    ]
    if len(ref_polls) < 5:
        ref_polls = [p for p in polls if not p.is_cis]
    if not ref_polls:
        return [], {}

    ov_by_pol: dict[str, dict[int, float]] = {}
    grouped: dict[str, list[tuple[int, PollRecord]]] = defaultdict(list)
    for idx, p in enumerate(ref_polls):
        grouped[p.pollster].append((idx, p))
    for pol, items in grouped.items():
        ws = _overlap_weights_same_pollster([x[1] for x in items])
        ov_by_pol[pol] = {items[k][0]: ws[k] for k in range(len(items))}

    unique_dates = sorted({p.midpoint_date for p in polls})
    traj: dict[str, np.ndarray] = {p: np.full(len(unique_dates), np.nan) for p in parties}

    mid_days = np.array([p.midpoint_date.toordinal() for p in ref_polls], dtype=np.float64)
    ov_w = np.array(
        [ov_by_pol[p.pollster][idx] for idx, p in enumerate(ref_polls)], dtype=np.float64
    )
    n_w = np.array(
        [math.sqrt(min(p.sample_size or 1000, 4000) / 1000.0) for p in ref_polls],
        dtype=np.float64,
    )

    # Empirical FA vs COMPROMIS_FV share if FA_FV appears in polls
    fa_sum = 0.0
    fv_sum = 0.0
    for p in ref_polls:
        if "FA" in p.party_shares and "COMPROMIS_FV" in p.party_shares:
            fa_sum += p.party_shares["FA"]
            fv_sum += p.party_shares["COMPROMIS_FV"]
    fa_split = fa_sum / (fa_sum + fv_sum) if (fa_sum + fv_sum) > 0 else 0.83

    for di, d in enumerate(unique_dates):
        d_ord = float(d.toordinal())
        dt = np.abs(mid_days - d_ord)
        base_k = np.exp(-0.5 * (dt / 12.0) ** 2) * ov_w * n_w
        for party in parties:
            ys = []
            ws = []
            for pi, p in enumerate(ref_polls):
                if base_k[pi] < 1e-6:
                    continue
                if party in p.party_shares:
                    ys.append(p.party_shares[party] * 100.0)
                    ws.append(base_k[pi])
                elif party == "FA" and "FA_FV" in p.party_shares:
                    ys.append(p.party_shares["FA_FV"] * 100.0 * fa_split)
                    ws.append(base_k[pi] * 0.7)
            if ys:
                w_arr = np.asarray(ws, dtype=np.float64)
                traj[party][di] = float(np.sum(np.asarray(ys) * w_arr) / np.sum(w_arr))
    return unique_dates, traj


def _nearest_consensus_value(
    target_date: date,
    dates_grid: Sequence[date],
    vals: np.ndarray | None,
) -> float | None:
    if vals is None or not dates_grid:
        return None
    idx = min(range(len(dates_grid)), key=lambda i: abs((dates_grid[i] - target_date).days))
    val = float(vals[idx])
    return None if math.isnan(val) else val


def _extract_2026_poll_pct(
    poll: PollRecord,
    party: str,
    fa_split: float = 0.83,
) -> float | None:
    if party in poll.party_shares:
        return poll.party_shares[party] * 100.0
    if party == "FA" and "FA_FV" in poll.party_shares:
        return poll.party_shares["FA_FV"] * 100.0 * fa_split
    return None


_BENCHMARK_NON_POLLSTERS: Final[frozenset[str]] = frozenset({"2024 EP election"})
_PARTISAN_INTERNAL_POLLSTERS: Final[frozenset[str]] = frozenset(
    {"PP", "PSOE", "Vox", "Sumar", "Podemos"}
)


def is_benchmark_or_excluded_pollster(pollster: str) -> bool:
    """Return True for election benchmark summary rows embedded in Wikipedia tables."""
    return pollster.strip() in _BENCHMARK_NON_POLLSTERS


def compute_pollster_ratings(
    *,
    historical_polls: Sequence[PollRecord] | None = None,
    polls_2026: Sequence[PollRecord] | None = None,
    elections_dir: Path | None = None,
    polls_dir: Path | None = None,
    hyperparams: ModelHyperparameters = DEFAULT_HYPERPARAMETERS,
    exclude_election_ids: Sequence[str] | None = None,
    as_of: date | None = None,
    window_days: int = 21,
) -> list[PollsterRatingRecord]:
    """Compute historical accuracy, empirical-Bayes grades, herding, and house effects.

    ``shrunk_mae`` and all ``delta_*`` house-effect offsets are expressed in
    **percentage points** (``0.0..100.0`` scale) matching ``PollsterRatingRecord``.
    """
    polls_root = polls_dir or get_polls_dir()
    elec_root = elections_dir or get_elections_dir()

    if historical_polls is None:
        hist_path = polls_root / "national_2015_2023.csv"
        hist_list = load_national_polls(hist_path) if hist_path.is_file() else []
    else:
        hist_list = list(historical_polls)

    if polls_2026 is None:
        p26_path = polls_root / "national_2026.csv"
        p26_list = load_national_polls(p26_path) if p26_path.is_file() else []
    else:
        p26_list = list(polls_2026)

    p26_list = [p for p in p26_list if not is_benchmark_or_excluded_pollster(p.pollster)]
    if as_of is not None:
        p26_list = [p for p in p26_list if p.fieldwork_end <= as_of]

    excluded_set = set(exclude_election_ids or ())
    eval_elections = [eid for eid in HISTORICAL_ELECTION_DATES if eid not in excluded_set]

    # Per-pollster per-election MAE & signed bias
    per_pol_el_mae: dict[str, dict[str, float]] = defaultdict(dict)
    per_pol_el_bias: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    all_pollsters: set[str] = {p.pollster for p in p26_list}

    for eid in eval_elections:
        edate = HISTORICAL_ELECTION_DATES[eid]
        actual_pct = _compute_actual_national_pct(eid, elec_root)
        if not actual_pct:
            continue
        late_by_pol: dict[str, list[PollRecord]] = defaultdict(list)
        ext_by_pol: dict[str, list[PollRecord]] = defaultdict(list)
        for p in hist_list:
            if p.election_id != eid or is_benchmark_or_excluded_pollster(p.pollster):
                continue
            days_before = (edate - p.fieldwork_end).days
            if 0 <= days_before <= window_days:
                late_by_pol[p.pollster].append(p)
                all_pollsters.add(p.pollster)
            elif window_days < days_before <= 45:
                ext_by_pol[p.pollster].append(p)
                all_pollsters.add(p.pollster)

        for pol in set(late_by_pol) | set(ext_by_pol):
            plist = late_by_pol[pol] if late_by_pol[pol] else ext_by_pol[pol]
            plist.sort(key=lambda x: x.fieldwork_end)
            ov_ws = _overlap_weights_same_pollster(plist)
            mae_num = 0.0
            mae_den = 0.0
            bias_num: dict[str, float] = defaultdict(float)
            bias_den: dict[str, float] = defaultdict(float)
            for idx, poll in enumerate(plist):
                days_before = max(0, (edate - poll.fieldwork_end).days)
                rec_w = math.exp(-days_before / 8.0) * ov_ws[idx]
                signed_errs, abs_errs = _extract_historical_bloc_pcts(poll, eid, actual_pct)
                if not abs_errs:
                    continue
                row_mae = sum(abs_errs.values()) / len(abs_errs)
                mae_num += rec_w * row_mae
                mae_den += rec_w
                for k, v in signed_errs.items():
                    bias_num[k] += rec_w * v
                    bias_den[k] += rec_w
            if mae_den > 0.0:
                per_pol_el_mae[pol][eid] = mae_num / mae_den
                for k, num in bias_num.items():
                    if bias_den[k] > 0.0:
                        per_pol_el_bias[pol][eid][k] = num / bias_den[k]

    # Compute initial shrunk MAE per pollster from historical elections
    shrunk_mae_map: dict[str, float] = {}
    n_elec_map: dict[str, int] = {}
    hist_bias_map: dict[str, dict[str, float]] = {}
    for pol in sorted(all_pollsters):
        el_maes = per_pol_el_mae.get(pol, {})
        n_el = len(el_maes)
        n_elec_map[pol] = n_el
        emp_mae = (
            sum(el_maes.values()) / n_el if n_el > 0 else hyperparams.prior_mae_anchor
        )
        tau_r = hyperparams.eb_rating_tau
        shrunk_mae_map[pol] = (
            n_el * emp_mae + tau_r * hyperparams.prior_mae_anchor
        ) / (n_el + tau_r)

        hb: dict[str, float] = {}
        tau_h = hyperparams.eb_house_tau
        for key in ("PP", "PSOE", "VOX", "LEFT", "PODEMOS"):
            vals = [per_pol_el_bias[pol][e][key] for e in el_maes if key in per_pol_el_bias[pol][e]]
            hb[key] = sum(vals) / (len(vals) + tau_h) if vals else 0.0
        hist_bias_map[pol] = hb

    # Reference pollsters (multi-election non-partisan, non-CIS houses) for 2026 house consensus
    tracked_2026_parties = ("PP", "PSOE", "VOX", "FA", "PODEMOS")
    ref_pollsters = {
        pol
        for pol, mae in shrunk_mae_map.items()
        if pol != "CIS"
        and pol not in _PARTISAN_INTERNAL_POLLSTERS
        and n_elec_map.get(pol, 0) >= 2
        and mae <= 2.10
    }
    if not ref_pollsters:
        ref_pollsters = {
            pol
            for pol in all_pollsters
            if pol != "CIS" and pol not in _PARTISAN_INTERNAL_POLLSTERS
        }

    has_2026 = bool(p26_list)
    if has_2026:
        fa_n, fv_n = 0.0, 0.0
        for p in p26_list:
            if not p.is_cis and "FA" in p.party_shares and "COMPROMIS_FV" in p.party_shares:
                fa_n += p.party_shares["FA"]
                fv_n += p.party_shares["COMPROMIS_FV"]
        fa_split_2026 = fa_n / (fa_n + fv_n) if (fa_n + fv_n) > 0 else 0.83
        dates_grid, cons_traj = _estimate_contemporaneous_consensus_2026(
            p26_list, ref_pollsters, tracked_2026_parties
        )
    else:
        fa_split_2026 = 0.83
        dates_grid, cons_traj = [], {}

    polls_2026_by_pol: dict[str, list[PollRecord]] = defaultdict(list)
    for p in p26_list:
        polls_2026_by_pol[p.pollster].append(p)

    # First pass: raw 2026 consensus residuals + herding detection
    contemp_resids: dict[str, dict[str, float]] = {}
    herding_ratios: dict[str, float] = {}
    herding_penalties: dict[str, float] = {}
    n_polls_2026_map: dict[str, int] = {}

    for pol in sorted(all_pollsters):
        plist_26 = sorted(polls_2026_by_pol.get(pol, []), key=lambda x: x.fieldwork_end)
        n_polls_2026_map[pol] = len(plist_26)

        resids_pol: dict[str, float] = {}
        if has_2026 and plist_26 and dates_grid:
            max_26 = p26_list[-1].fieldwork_end
            ov_ws = _overlap_weights_same_pollster(plist_26)
            for party in tracked_2026_parties:
                num = 0.0
                den = 0.0
                for idx, p in enumerate(plist_26):
                    pv = _extract_2026_poll_pct(p, party, fa_split=fa_split_2026)
                    cv = _nearest_consensus_value(p.midpoint_date, dates_grid, cons_traj.get(party))
                    if pv is None or cv is None:
                        continue
                    age_days = max(0, (max_26 - p.fieldwork_end).days)
                    rw = math.exp(-age_days / 21.0) * ov_ws[idx]
                    num += rw * (pv - cv)
                    den += rw
                if den > 0.0:
                    resids_pol[party] = num / (den + 0.15)
        contemp_resids[pol] = resids_pol

        # Herding test across temporally separated independent waves (>= 4 days between ends)
        indep_waves: list[PollRecord] = []
        for p in plist_26:
            if not indep_waves or (p.fieldwork_end - indep_waves[-1].fieldwork_end).days >= 4:
                indep_waves.append(p)

        h_ratio = 1.0
        if has_2026 and len(indep_waves) >= 4 and dates_grid:
            ratios: list[float] = []
            for party in ("PP", "PSOE", "VOX"):
                devs = []
                srs_vars = []
                for p in indep_waves:
                    pv = _extract_2026_poll_pct(p, party)
                    cv = _nearest_consensus_value(
                        p.midpoint_date, dates_grid, cons_traj.get(party)
                    )
                    if pv is None or cv is None:
                        continue
                    devs.append(pv - cv)
                    prob = max(0.02, min(0.98, cv / 100.0))
                    n_eff = float(min(p.sample_size or 1000, 4000))
                    srs_vars.append((prob * (1.0 - prob) / n_eff) * 10000.0)
                if len(devs) >= 4:
                    mean_d = sum(devs) / len(devs)
                    obs_sd = math.sqrt(sum((d - mean_d) ** 2 for d in devs) / (len(devs) - 1))
                    exp_sd = math.sqrt(sum(srs_vars) / len(srs_vars))
                    if exp_sd > 1e-6:
                        ratios.append(obs_sd / exp_sd)
            if ratios:
                h_ratio = float(sum(ratios) / len(ratios))
        herding_ratios[pol] = h_ratio
        if len(indep_waves) >= 4 and h_ratio < hyperparams.herding_ratio_threshold:
            pen = max(0.35, min(1.0, h_ratio / hyperparams.herding_ratio_threshold))
        else:
            pen = 1.0
        herding_penalties[pol] = pen

    # Combine historical + 2026 house-effect deltas & enforce zero-sum on reference pollsters
    raw_deltas: dict[str, dict[str, float]] = {}
    for pol in sorted(all_pollsters):
        hb = hist_bias_map[pol]
        cr = contemp_resids[pol]
        d_map: dict[str, float] = {}
        for party in tracked_2026_parties:
            hist_key = "LEFT" if party == "FA" else party
            h_val = hb.get(hist_key, hb.get("LEFT", 0.0) if party == "PODEMOS" else 0.0)
            c_val = cr.get(party)
            if c_val is not None:
                d_map[party] = 0.20 * h_val + 0.80 * c_val if n_elec_map[pol] > 0 else c_val
            else:
                d_map[party] = h_val
        raw_deltas[pol] = d_map

    active_ref = [p for p in ref_pollsters if p in raw_deltas]
    if active_ref:
        for party in tracked_2026_parties:
            ref_mean = sum(raw_deltas[p][party] for p in active_ref) / len(active_ref)
            for pol in raw_deltas:
                raw_deltas[pol][party] -= ref_mean

    records: list[PollsterRatingRecord] = []
    for pol in sorted(all_pollsters):
        s_mae = shrunk_mae_map[pol]
        n_el = n_elec_map[pol]
        n_26 = n_polls_2026_map[pol]
        deltas = raw_deltas[pol]
        max_bias_abs = max(abs(deltas["PP"]), abs(deltas["PSOE"]), abs(deltas["VOX"]))

        eff_mae = s_mae + max(0.0, (max_bias_abs - 2.0) * 0.45)
        if pol == "CIS" or pol in _PARTISAN_INTERNAL_POLLSTERS:
            grade = "C"
            eff_mae = max(eff_mae, 2.35)
        elif n_el >= 3 and eff_mae <= 1.66 and max_bias_abs < 2.2:
            grade = "A"
        elif eff_mae > 2.02 or max_bias_abs >= 2.8:
            grade = "C"
        else:
            grade = "B"

        base_g = _GRADE_BASE_WEIGHT[grade]
        mae_scale = (hyperparams.prior_mae_anchor / max(0.9, eff_mae)) ** 1.4
        cis_factor = 0.55 if pol == "CIS" else 1.0
        g_weight = round(base_g * mae_scale * cis_factor, 4)

        records.append(
            PollsterRatingRecord(
                pollster=pol,
                n_elections=int(n_el),
                n_polls_2026=int(n_26),
                shrunk_mae=round(float(eff_mae), 4),
                grade=grade,
                grade_weight=float(g_weight),
                herding_ratio=round(float(herding_ratios[pol]), 4),
                herding_penalty=round(float(herding_penalties[pol]), 4),
                delta_PP=round(float(deltas["PP"]), 4),
                delta_PSOE=round(float(deltas["PSOE"]), 4),
                delta_VOX=round(float(deltas["VOX"]), 4),
                delta_FA=round(float(deltas["FA"]), 4),
                delta_PODEMOS=round(float(deltas["PODEMOS"]), 4),
            )
        )

    records.sort(key=lambda r: (r.grade, r.shrunk_mae, -r.n_elections, r.pollster))
    return records


def save_pollster_ratings(
    records: Sequence[PollsterRatingRecord],
    output_path: Path | None = None,
) -> Path:
    """Write ``PollsterRatingRecord`` rows to CSV (default under outputs/)."""
    target = output_path or (get_outputs_dir() / "pollster_ratings.csv")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=POLLSTER_RATINGS_COLUMNS)
        writer.writeheader()
        for r in records:
            writer.writerow(r.to_dict())
    return target


def load_pollster_ratings(csv_path: Path | None = None) -> list[PollsterRatingRecord]:
    """Load validated ``PollsterRatingRecord`` rows from CSV."""
    target = csv_path or (get_outputs_dir() / "pollster_ratings.csv")
    if not target.is_file():
        raise FileNotFoundError(f"Missing pollster ratings CSV: {target}")
    out: list[PollsterRatingRecord] = []
    with target.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            out.append(
                PollsterRatingRecord(
                    pollster=str(row["pollster"]).strip(),
                    n_elections=int(row["n_elections"]),
                    n_polls_2026=int(row["n_polls_2026"]),
                    shrunk_mae=float(row["shrunk_mae"]),
                    grade=str(row["grade"]).strip(),
                    grade_weight=float(row["grade_weight"]),
                    herding_ratio=float(row["herding_ratio"]),
                    herding_penalty=float(row["herding_penalty"]),
                    delta_PP=float(row["delta_PP"]),
                    delta_PSOE=float(row["delta_PSOE"]),
                    delta_VOX=float(row["delta_VOX"]),
                    delta_FA=float(row["delta_FA"]),
                    delta_PODEMOS=float(row["delta_PODEMOS"]),
                )
            )
    return out
