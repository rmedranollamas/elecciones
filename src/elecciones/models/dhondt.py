"""Exact LOREG Art. 163 D'Hondt seat allocator (scalar and vectorized engines).

Implements Spain's Organic Electoral Law (LOREG) Art. 163 rules:
- For the 50 provincial constituencies, lists must obtain at least ``3%`` of
  ``valid_votes`` (sum of party candidacy votes plus blank ballots ``blancos``,
  excluding null ballots) to enter the D'Hondt divisor apportionment.
- For the autonomous cities of Ceuta (``51``) and Melilla (``52``), the single
  seat is awarded by plurality (``0%`` threshold; blank ballots cannot win).
- Quotient ties are resolved per LOREG Art. 163.1.d by awarding the seat to the
  list with greater total votes, and deterministically alternating in canonical
  party order when total votes are also identical (``ties_broken`` counts ties
  occurring on awarded/cutoff seats).
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from elecciones.schemas import (
    AUTONOMOUS_CITY_CPROS,
    AllocationInput,
    AllocationResult,
    LastSeatBattle,
    QuotientEntry,
    RunnerUpGap,
    SchemaValidationError,
    ThresholdExhaustionError,
)


def _int_quotient_beats(va: int, da: int, ia: int, vb: int, db: int, ib: int) -> bool:
    """Exact cross-multiplied quotient + LOREG Art. 163.1.d tie comparison."""
    lhs = va * db
    rhs = vb * da
    if lhs != rhs:
        return lhs > rhs
    if va != vb:
        return va > vb
    return ia < ib


def _safe_int_div_float(num: int, den: int) -> float:
    """Convert exact ratio ``num / den`` to a finite IEEE-754 float (clamped at 1e308)."""
    try:
        return float(num / den)
    except OverflowError:
        return 1e308 if num > 0 else 0.0


def _float_quotient_beats(
    va: float, da: int, ia: int, vb: float, db: int, ib: int
) -> bool:
    """Floating-point quotient + LOREG Art. 163.1.d tie comparison."""
    qa = va / float(da)
    qb = vb / float(db)
    if qa != qb:
        return qa > qb
    if va != vb:
        return va > vb
    return ia < ib


def _compute_valid_votes_exact_float(
    party_vals: Sequence[float], blank_val: float
) -> tuple[float, float, list[float], float]:
    """Compute overflow-safe compensated ``valid_votes`` for floating-point inputs.

    Returns ``(valid_votes_actual, scale, scaled_party_vals, scaled_valid)`` where
    ``valid_votes_actual`` is scaled safely without overflowing near ``1e308``.
    """
    max_val = max([blank_val, *party_vals], default=0.0)
    if max_val <= 0.0:
        raise SchemaValidationError("valid_votes must be > 0")
    if max_val > 1e150 or max_val < 1e-150:
        scale = max_val
        s_party = [v / scale for v in party_vals]
        s_blank = blank_val / scale
        s_valid = math.fsum([*s_party, s_blank])
        valid_actual = s_valid * scale if not math.isinf(s_valid * scale) else max_val
        return valid_actual, scale, s_party, s_valid

    valid_exact = math.fsum([*party_vals, blank_val])
    return valid_exact, 1.0, list(party_vals), valid_exact


def allocate_dhondt_loreg(
    party_votes: Mapping[str, int | float] | AllocationInput,
    seats: int | None = None,
    blank_votes: int | float = 0,
    is_autonomous_city: bool = False,
    threshold: float | None = None,
    cpro: str | None = None,
    canonical_order: Sequence[str] | None = None,
    raise_on_exhaustion: bool = False,
    include_quotients: bool = True,
    null_votes: int | float = 0,
) -> AllocationResult:
    """Allocate seats in a single constituency strictly following LOREG Art. 163.

    Parameters
    ----------
    party_votes:
        Mapping of party identifier -> votes (``int``) or vote shares (``float``),
        or a validated :class:`AllocationInput`.
    seats:
        Constituency magnitude ``M_c`` (required unless ``party_votes`` is
        an ``AllocationInput``).
    blank_votes:
        Blank ballots (or blank vote share in ``[0, 1]``). Included directly in
        ``valid_votes = sum(party_votes) + blank_votes`` when evaluating the
        provincial 3% barrier.
    is_autonomous_city:
        If ``True`` (Ceuta/Melilla), applies ``0%`` threshold plurality rule.
    threshold:
        Optional threshold override in ``[0, 1)``. Defaults to ``0.0`` for
        autonomous cities and ``0.03`` for provinces.
    cpro:
        Optional 2-digit INE constituency code. Codes ``"51"`` and ``"52"``
        automatically enable autonomous-city rules unless overridden.
    canonical_order:
        Optional explicit tie-break fallback list when quotients and vote totals
        are strictly equal.
    raise_on_exhaustion:
        If ``True``, raises :class:`ThresholdExhaustionError` when positive
        ``valid_votes`` produce zero eligible lists above the barrier.
    include_quotients:
        If ``True``, populates ``quotients``, ``quotient_table``, and
        ``last_seat_battle``.
    null_votes:
        Optional null ballots count (validated non-negative; excluded from
        ``valid_votes`` under LOREG Art. 163).
    """
    if isinstance(null_votes, bool) or not isinstance(null_votes, (int, float)):
        raise SchemaValidationError("null_votes must be numeric")
    if math.isnan(null_votes) or math.isinf(null_votes) or null_votes < 0:
        raise SchemaValidationError(f"null_votes must be finite and >= 0, got {null_votes}")
    if isinstance(party_votes, AllocationInput):
        inp = party_votes
    else:
        if seats is None:
            raise SchemaValidationError("seats must be provided with mapping party_votes")
        city_flag = is_autonomous_city or (
            cpro is not None and str(cpro).strip().zfill(2) in AUTONOMOUS_CITY_CPROS
        )
        inp = AllocationInput(
            seats=seats,
            party_votes=dict(party_votes),
            blank_votes=blank_votes,
            is_autonomous_city=city_flag,
            cpro=cpro,
            threshold=threshold,
        )

    if canonical_order is not None:
        seen = set(canonical_order)
        if set(inp.party_votes.keys()) - seen:
            party_list = list(canonical_order) + [
                p for p in inp.party_votes if p not in seen
            ]
        else:
            party_list = [p for p in canonical_order if p in inp.party_votes]
    else:
        party_list = list(inp.party_votes.keys())

    m_seats = inp.seats
    is_city = inp.is_autonomous_city
    eff_thresh = 0.0 if is_city else float(inp.threshold if inp.threshold is not None else 0.03)
    int_mode = inp.is_integer_mode

    alloc = dict.fromkeys(party_list, 0)

    if int_mode:
        v_map_int: dict[str, int] = {p: int(inp.party_votes[p]) for p in party_list}
        b_int = int(inp.blank_votes)
        valid_int = b_int + sum(v_map_int.values())
        if valid_int <= 0:
            raise SchemaValidationError("valid_votes must be > 0")

        if is_city or abs(eff_thresh) < 1e-12:
            eligible = [p for p in party_list if v_map_int[p] > 0]
        elif abs(eff_thresh - 0.03) < 1e-12:
            eligible = [
                p
                for p in party_list
                if v_map_int[p] > 0 and v_map_int[p] * 100 >= 3 * valid_int
            ]
        else:
            eligible = [
                p
                for p in party_list
                if v_map_int[p] > 0 and (v_map_int[p] / valid_int) >= eff_thresh
            ]

        if not eligible:
            if raise_on_exhaustion:
                raise ThresholdExhaustionError(
                    f"No lists meet threshold={eff_thresh} over valid_votes={valid_int}"
                )
            return AllocationResult(
                valid_votes=valid_int,
                eligible_parties=(),
                seats=alloc,
                cutoff_quotient=None,
                ties_broken=0,
                cpro=inp.cpro,
                quotients={} if include_quotients else None,
                quotient_table=() if include_quotients else None,
                last_seat_battle=None,
            )

        idx_of = {p: i for i, p in enumerate(party_list)}
        ties_broken = 0

        if is_city:
            top_v = max(v_map_int[p] for p in eligible)
            tied_top = [p for p in eligible if v_map_int[p] == top_v]
            if len(tied_top) > 1:
                ties_broken += 1
            winner = tied_top[0]
            alloc[winner] = m_seats
            cutoff_q = _safe_int_div_float(top_v, 1)
            last_holder = winner
        else:
            cutoff_num = 0
            cutoff_den = 1
            last_holder = eligible[0]
            for _step in range(m_seats):
                best_p = eligible[0]
                best_tie_group = [best_p]
                for cand in eligible[1:]:
                    v_c = v_map_int[cand]
                    d_c = alloc[cand] + 1
                    v_b = v_map_int[best_p]
                    d_b = alloc[best_p] + 1
                    if v_c * d_b == v_b * d_c:
                        best_tie_group.append(cand)
                        if _int_quotient_beats(v_c, d_c, idx_of[cand], v_b, d_b, idx_of[best_p]):
                            best_p = cand
                    elif _int_quotient_beats(v_c, d_c, idx_of[cand], v_b, d_b, idx_of[best_p]):
                        best_p = cand
                        best_tie_group = [cand]
                if len(best_tie_group) > 1:
                    ties_broken += 1
                alloc[best_p] += 1
                cutoff_num = v_map_int[best_p]
                cutoff_den = alloc[best_p]
                last_holder = best_p
            cutoff_q = _safe_int_div_float(cutoff_num, cutoff_den)

        quotients_map: dict[str, list[float]] | None = None
        quotient_table: tuple[QuotientEntry, ...] | None = None
        last_battle: LastSeatBattle | None = None
        if include_quotients:
            quotients_map = {
                p: [
                    round(_safe_int_div_float(v_map_int[p], d), 6)
                    for d in range(1, min(m_seats, alloc[p] + 1) + 1)
                ]
                for p in eligible
            }
            all_entries: list[tuple[int, int, int, str, int, bool]] = []
            for p in eligible:
                max_d = min(m_seats, alloc[p] + 1)
                vp = v_map_int[p]
                for d in range(1, max_d + 1):
                    all_entries.append((vp, d, idx_of[p], p, d, d <= alloc[p]))
            if is_city:
                all_entries = [
                    (v_map_int[p], 1, idx_of[p], p, 1, p == last_holder) for p in eligible
                ]
            # Sort descending by exact cross-multiplied quotient (vp / d, vp, -idx)
            from functools import cmp_to_key

            def _cmp_int(
                a: tuple[int, int, int, str, int, bool],
                b: tuple[int, int, int, str, int, bool],
            ) -> int:
                if _int_quotient_beats(a[0], a[1], a[2], b[0], b[1], b[2]):
                    return -1
                return 1

            all_entries.sort(key=cmp_to_key(_cmp_int))
            top_entries = all_entries[: m_seats + min(len(eligible), 6)]
            quotient_table = tuple(
                QuotientEntry(
                    party=p,
                    divisor=d,
                    quotient=round(_safe_int_div_float(vp, d), 6),
                    won=won,
                    global_rank=rank,
                )
                for rank, (vp, d, _idx, p, _d2, won) in enumerate(top_entries, start=1)
            )
            # Runner-up gaps relative to fixed valid_votes
            runners: list[RunnerUpGap] = []
            holder_v = v_map_int[last_holder]
            holder_d = 1 if is_city else alloc[last_holder]
            min_thresh_num = 0 if is_city else (3 * valid_int + 99) // 100
            for p in party_list:
                if p == last_holder:
                    continue
                vp = v_map_int[p]
                if vp <= 0 and len(party_list) > 10:
                    continue
                next_d = 1 if is_city else (alloc[p] + 1)
                # Need V'_p / next_d > holder_v / holder_d (or >= if V'_p > holder_v)
                # Integer: req_votes * holder_d >= holder_v * next_d (+1 if req_votes <= holder_v)
                lhs = (holder_v * next_d) // holder_d
                rem = (holder_v * next_d) % holder_d
                req_q = lhs + (1 if rem > 0 or lhs <= holder_v else 0)
                req_v = max(min_thresh_num, req_q)
                gap_abs = max(0, req_v - vp)
                gap_pp = round(_safe_int_div_float(100 * gap_abs, valid_int), 6)
                runners.append(RunnerUpGap(party=p, vote_gap_pp=gap_pp, vote_gap_abs=gap_abs))
            runners.sort(key=lambda r: (r.vote_gap_abs, r.vote_gap_pp, r.party))
            last_battle = LastSeatBattle(
                holder=last_holder,
                cutoff_quotient=round(cutoff_q, 6),
                runners_up=tuple(runners[:3]),
            )

        return AllocationResult(
            valid_votes=valid_int,
            eligible_parties=tuple(eligible),
            seats=alloc,
            cutoff_quotient=cutoff_q,
            ties_broken=ties_broken,
            cpro=inp.cpro,
            quotients=quotients_map,
            quotient_table=quotient_table,
            last_seat_battle=last_battle,
        )

    # Floating-point mode
    raw_party_vals = [float(inp.party_votes[p]) for p in party_list]
    raw_blank_val = float(inp.blank_votes)
    valid_actual, scale, scaled_party_vals, scaled_valid = _compute_valid_votes_exact_float(
        raw_party_vals, raw_blank_val
    )
    v_map_f: dict[str, float] = dict(zip(party_list, scaled_party_vals, strict=True))
    raw_map_f: dict[str, float] = dict(zip(party_list, raw_party_vals, strict=True))

    if is_city or eff_thresh == 0.0:
        eligible = [p for p in party_list if v_map_f[p] > 0.0]
    else:
        eligible = [
            p
            for p in party_list
            if v_map_f[p] > 0.0 and (v_map_f[p] / scaled_valid) >= eff_thresh
        ]

    if not eligible:
        if raise_on_exhaustion:
            raise ThresholdExhaustionError(
                f"No lists meet threshold={eff_thresh} over valid_votes={valid_actual}"
            )
        return AllocationResult(
            valid_votes=valid_actual,
            eligible_parties=(),
            seats=alloc,
            cutoff_quotient=None,
            ties_broken=0,
            cpro=inp.cpro,
            quotients={} if include_quotients else None,
            quotient_table=() if include_quotients else None,
            last_seat_battle=None,
        )

    idx_of = {p: i for i, p in enumerate(party_list)}
    ties_broken = 0

    if is_city:
        top_v = max(v_map_f[p] for p in eligible)
        tied_top = [p for p in eligible if v_map_f[p] == top_v]
        if len(tied_top) > 1:
            ties_broken += 1
        winner = tied_top[0]
        alloc[winner] = m_seats
        cutoff_scaled = top_v
        last_holder = winner
    else:
        cutoff_scaled = 0.0
        last_holder = eligible[0]
        for _step in range(m_seats):
            best_p = eligible[0]
            best_q = v_map_f[best_p] / float(alloc[best_p] + 1)
            tie_count = 1
            for cand in eligible[1:]:
                v_c = v_map_f[cand]
                d_c = alloc[cand] + 1
                q_c = v_c / float(d_c)
                if q_c == best_q:
                    tie_count += 1
                    if _float_quotient_beats(
                        v_c, d_c, idx_of[cand], v_map_f[best_p], alloc[best_p] + 1, idx_of[best_p]
                    ):
                        best_p = cand
                        best_q = q_c
                elif q_c > best_q:
                    best_p = cand
                    best_q = q_c
                    tie_count = 1
            if tie_count > 1:
                ties_broken += 1
            alloc[best_p] += 1
            cutoff_scaled = best_q
            last_holder = best_p

    cutoff_out = raw_map_f[last_holder] / float(alloc[last_holder] if not is_city else 1)

    quotients_map_f: dict[str, list[float]] | None = None
    quotient_table_f: tuple[QuotientEntry, ...] | None = None
    last_battle_f: LastSeatBattle | None = None
    if include_quotients:
        quotients_map_f = {
            p: [
                raw_map_f[p] / float(d)
                for d in range(1, min(m_seats, alloc[p] + 1) + 1)
            ]
            for p in eligible
        }
        all_entries_f: list[tuple[float, float, int, str, int, bool]] = []
        if is_city:
            for p in eligible:
                all_entries_f.append(
                    (v_map_f[p], v_map_f[p], -idx_of[p], p, 1, p == last_holder)
                )
        else:
            for p in eligible:
                max_d = min(m_seats, alloc[p] + 1)
                vp_s = v_map_f[p]
                for d in range(1, max_d + 1):
                    all_entries_f.append(
                        (vp_s / float(d), vp_s, -idx_of[p], p, d, d <= alloc[p])
                    )
        all_entries_f.sort(key=lambda x: (x[0], x[1], x[2]), reverse=True)
        top_entries_f = all_entries_f[: m_seats + min(len(eligible), 6)]
        quotient_table_f = tuple(
            QuotientEntry(
                party=p,
                divisor=d,
                quotient=round(raw_map_f[p] / float(d), 8),
                won=won,
                global_rank=rank,
            )
            for rank, (_q, _vp, _ni, p, d, won) in enumerate(top_entries_f, start=1)
        )
        runners_f: list[RunnerUpGap] = []
        thresh_scaled = 0.0 if is_city else eff_thresh * scaled_valid
        for p in party_list:
            if p == last_holder:
                continue
            vp_s = v_map_f[p]
            if vp_s <= 0.0 and len(party_list) > 10:
                continue
            next_d = 1 if is_city else (alloc[p] + 1)
            req_s = max(thresh_scaled, float(next_d) * cutoff_scaled)
            gap_s = max(0.0, req_s - vp_s)
            gap_pp = round(100.0 * gap_s / scaled_valid, 6)
            gap_abs = int(math.ceil(gap_s * scale)) if valid_actual > 10.0 else 0
            runners_f.append(RunnerUpGap(party=p, vote_gap_pp=gap_pp, vote_gap_abs=gap_abs))
        runners_f.sort(key=lambda r: (r.vote_gap_pp, r.vote_gap_abs, r.party))
        last_battle_f = LastSeatBattle(
            holder=last_holder,
            cutoff_quotient=cutoff_out,
            runners_up=tuple(runners_f[:3]),
        )

    return AllocationResult(
        valid_votes=valid_actual,
        eligible_parties=tuple(eligible),
        seats=alloc,
        cutoff_quotient=cutoff_out,
        ties_broken=ties_broken,
        cpro=inp.cpro,
        quotients=quotients_map_f,
        quotient_table=quotient_table_f,
        last_seat_battle=last_battle_f,
    )


def _scalar_dhondt_row(
    vals: Sequence[int | float],
    seats: int,
    blank: int | float,
    is_city: bool,
    threshold: float,
    is_int: bool,
) -> list[int]:
    """Exact single-row fallback preserving LOREG tie-breaks and arbitrary-size ints."""
    ka = len(vals)
    out = [0] * ka
    if is_int:
        v_list = [int(x) for x in vals]
        valid = int(blank) + sum(v_list)
        if valid <= 0:
            return out
        if is_city or threshold == 0.0:
            elig = [i for i, v in enumerate(v_list) if v > 0]
        elif abs(threshold - 0.03) < 1e-12:
            elig = [i for i, v in enumerate(v_list) if v > 0 and v * 100 >= 3 * valid]
        else:
            elig = [i for i, v in enumerate(v_list) if v > 0 and (v / valid) >= threshold]
        if not elig:
            return out
        if is_city:
            top_v = max(v_list[i] for i in elig)
            best = next(i for i in elig if v_list[i] == top_v)
            out[best] = seats
            return out
        for _ in range(seats):
            best = elig[0]
            for cand in elig[1:]:
                if _int_quotient_beats(
                    v_list[cand],
                    out[cand] + 1,
                    cand,
                    v_list[best],
                    out[best] + 1,
                    best,
                ):
                    best = cand
            out[best] += 1
        return out

    f_vals = [float(x) for x in vals]
    b_val = float(blank)
    _val_act, _scale, s_vals, s_valid = _compute_valid_votes_exact_float(f_vals, b_val)
    if is_city or threshold == 0.0:
        elig = [i for i, v in enumerate(s_vals) if v > 0.0]
    else:
        elig = [i for i, v in enumerate(s_vals) if v > 0.0 and (v / s_valid) >= threshold]
    if not elig:
        return out
    if is_city:
        top_v = max(s_vals[i] for i in elig)
        best = next(i for i in elig if s_vals[i] == top_v)
        out[best] = seats
        return out
    for _ in range(seats):
        best = elig[0]
        for cand in elig[1:]:
            if _float_quotient_beats(
                s_vals[cand],
                out[cand] + 1,
                cand,
                s_vals[best],
                out[best] + 1,
                best,
            ):
                best = cand
        out[best] += 1
    return out


def allocate_packed_chunk_inplace(
    votes_chunk: np.ndarray,
    out_chunk: np.ndarray,
    seats_vec: np.ndarray,
    city_vec: np.ndarray,
    active_indices_per_c: Sequence[np.ndarray],
    recip: np.ndarray | None = None,
    valid_chunk: np.ndarray | None = None,
    *,
    blank_chunk: np.ndarray | None = None,
    threshold: float = 0.03,
    is_int_mode: bool = False,
) -> None:
    """Populate ``out_chunk`` in-place for a chunk ``(m, C, P)`` using float64/exact math.

    Mask-isolated columns outside ``active_indices_per_c[c]`` are never read
    into floating-point arithmetic, keeping ``NaN``/negative masked entries safe
    under ``np.errstate(all='raise')``.
    """
    _ = recip  # retained for explicit caller API compatibility; exact / (ns + 1) used
    out_chunk.fill(0)
    m, num_c, _num_p = votes_chunk.shape

    for c in range(num_c):
        aj = active_indices_per_c[c]
        ka = int(aj.size)
        if ka == 0:
            continue
        sc = int(seats_vec[c])
        is_city = bool(city_vec[c])

        if is_int_mode:
            sub_raw = votes_chunk[:, c, aj]
            # Exact Python int fallback if values exceed 2^50, uint64, object, or small batch
            use_exact_int = (
                sub_raw.dtype == object
                or sub_raw.dtype == np.uint64
                or m <= 64
                or int(np.max(sub_raw)) > (1 << 50)
            )
            if use_exact_int:
                for row_i in range(m):
                    vals_r = [int(x) for x in sub_raw[row_i]]
                    if valid_chunk is not None:
                        b_r = max(0, int(valid_chunk[row_i, c]) - sum(vals_r))
                    elif blank_chunk is not None:
                        b_r = int(blank_chunk[row_i, c])
                    else:
                        b_r = 0
                    res_r = _scalar_dhondt_row(vals_r, sc, b_r, is_city, threshold, True)
                    for idx_local, p_col in enumerate(aj):
                        out_chunk[row_i, c, int(p_col)] = res_r[idx_local]
                continue

            v = np.ascontiguousarray(sub_raw.T, dtype=np.float64)
            if valid_chunk is not None:
                val = np.ascontiguousarray(valid_chunk[:, c], dtype=np.float64)[None, :]
            elif blank_chunk is not None:
                val = v.sum(axis=0, keepdims=True) + np.ascontiguousarray(
                    blank_chunk[:, c], dtype=np.float64
                )[None, :]
            else:
                val = v.sum(axis=0, keepdims=True)

            if is_city or threshold == 0.0:
                q = np.where(v > 0.0, v, 0.0)
            elif abs(threshold - 0.03) < 1e-12:
                q = np.where((v > 0.0) & (v * 100.0 >= 3.0 * val), v, 0.0)
            else:
                q = np.where((v > 0.0) & (v >= threshold * val), v, 0.0)
        else:
            # Float64 active extraction (never touches inactive masked columns)
            v = np.ascontiguousarray(votes_chunk[:, c, aj].T, dtype=np.float64)  # (ka, m)
            b_row = (
                np.ascontiguousarray(blank_chunk[:, c], dtype=np.float64)
                if blank_chunk is not None
                else np.zeros(m, dtype=np.float64)
            )
            row_max = np.maximum(v.max(axis=0), b_row)
            huge_or_tiny = (row_max > 1e150) | ((row_max > 0.0) & (row_max < 1e-150))
            if np.any(huge_or_tiny):
                scale = np.where(huge_or_tiny & (row_max > 0.0), row_max, 1.0)
                v = v / scale[None, :]
                b_row = b_row / scale

            if valid_chunk is not None:
                val = np.ascontiguousarray(valid_chunk[:, c], dtype=np.float64)[None, :]
                if np.any(huge_or_tiny):
                    val = val / np.where(huge_or_tiny & (row_max > 0.0), row_max, 1.0)[None, :]
            else:
                val = v.sum(axis=0, keepdims=True) + b_row[None, :]

            if is_city or threshold == 0.0:
                q = np.where(v > 0.0, v, 0.0)
            else:
                q = np.where((v > 0.0) & (v >= threshold * val), v, 0.0)
                # Exact compensated-sum boundary refinement (e.g. 0.191 + 0.009 + 0.1 == 0.3)
                if valid_chunk is None:
                    near_thresh = np.any(
                        (v > 0.0) & (np.abs(v - threshold * val) <= 1e-12 * val), axis=0
                    )
                    if np.any(near_thresh):
                        for col_b in np.flatnonzero(near_thresh):
                            v_col = [float(x) for x in v[:, col_b]]
                            valid_fsum = math.fsum([*v_col, float(b_row[col_b])])
                            for r_i, vr in enumerate(v_col):
                                above = (
                                    vr > 0.0
                                    and valid_fsum > 0.0
                                    and (vr / valid_fsum) >= threshold
                                )
                                q[r_i, col_b] = vr if above else 0.0

        if is_city:
            w = np.argmax(q, axis=0)
            cols = np.arange(m, dtype=np.intp)
            out_chunk[cols, c, aj[w]] = np.int16(sc)
            continue

        esum = q.sum(axis=0, keepdims=True)
        nonzero_mask = esum[0] > 0.0
        if not np.any(nonzero_mask):
            continue
        esum_safe = np.where(nonzero_mask[None, :], esum, 1.0)
        # Provable Lower-Quota seed s_i <= floor(sc * p_i - 1e-9) <= true D'Hondt seats
        s = np.maximum(0, np.floor(q * (sc / esum_safe) - 1e-9).astype(np.int32))
        if not np.all(nonzero_mask):
            s[:, ~nonzero_mask] = 0
        rem = np.where(nonzero_mask, sc - s.sum(axis=0), 0)

        max_rem = int(rem.max()) if rem.size > 0 else 0
        if max_rem > 0:
            cur_q = q / (s + 1)
            q1d = cur_q.ravel()
            v1d = q.ravel()
            s1d = s.ravel()
            for step in range(max_rem):
                need = np.flatnonzero(rem > step)
                if need.size == 0:
                    break
                sub = cur_q[:, need]
                w = np.argmax(sub, axis=0)
                max_q = sub[w, np.arange(need.size)]
                if np.any((sub == max_q).sum(axis=0) > 1):
                    ties = np.flatnonzero((sub == max_q).sum(axis=0) > 1)
                    for ti in ties:
                        nc = need[ti]
                        cands = np.flatnonzero(cur_q[:, nc] == max_q[ti])
                        w[ti] = cands[int(np.argmax(q[cands, nc]))]
                f = w * m + need
                ns = s1d[f] + 1
                s1d[f] = ns
                q1d[f] = v1d[f] / (ns + 1)

        out_chunk[:, c, aj] = s.T.astype(out_chunk.dtype, copy=False)


def allocate_batch_vectorized(
    votes_or_shares: np.ndarray,
    seats: Sequence[int] | np.ndarray,
    *,
    is_autonomous_city: bool | Sequence[bool] | np.ndarray = False,
    blank_votes_or_shares: float | int | np.ndarray = 0.0,
    active_mask: np.ndarray | None = None,
    threshold: float = 0.03,
    n_workers: int | None = None,
) -> np.ndarray:
    """Vectorized batch LOREG Art. 163 D'Hondt allocation over shape ``(N, C, P)``.

    Preserves full parity with :func:`allocate_dhondt_loreg` across arbitrary
    ``int64`` / ``uint64`` / ``object`` / ``float64`` dtypes, autonomous cities,
    2D blank shares, exact quotient/vote tie-breaking, and masked ``NaN`` cells.
    """
    arr = np.asarray(votes_or_shares)
    if arr.ndim != 3:
        raise SchemaValidationError(
            f"votes_or_shares must be 3D (N, C, P), got shape {arr.shape}"
        )
    n_draws, num_c, num_p = arr.shape
    seats_vec = np.asarray(seats, dtype=np.int32)
    if seats_vec.shape != (num_c,):
        raise SchemaValidationError(f"seats shape {seats_vec.shape} != ({num_c},)")
    if np.any(seats_vec < 1):
        raise SchemaValidationError("All seats entries must be >= 1")

    if isinstance(is_autonomous_city, (bool, np.bool_)):
        city_vec = np.full(num_c, bool(is_autonomous_city), dtype=bool)
    else:
        city_vec = np.asarray(is_autonomous_city, dtype=bool)
        if city_vec.shape != (num_c,):
            raise SchemaValidationError(
                f"is_autonomous_city shape {city_vec.shape} != ({num_c},)"
            )

    out_dtype = np.int32 if int(seats_vec.max()) > 30000 else np.int16
    out = np.zeros((n_draws, num_c, num_p), dtype=out_dtype)
    if n_draws == 0:
        return out

    is_int_dtype = arr.dtype == object or np.issubdtype(arr.dtype, np.integer)

    # Normalize blank_votes_or_shares to (N, C)
    if np.ndim(blank_votes_or_shares) == 0:
        blank_mat = np.full(
            (n_draws, num_c),
            blank_votes_or_shares,
            dtype=object if arr.dtype == object else (arr.dtype if is_int_dtype else np.float64),
        )
    else:
        b_arr = np.asarray(blank_votes_or_shares)
        if b_arr.shape == (num_c,):
            blank_mat = np.broadcast_to(b_arr[None, :], (n_draws, num_c))
        elif b_arr.shape == (n_draws, num_c):
            blank_mat = b_arr
        else:
            raise SchemaValidationError(
                f"blank_votes_or_shares shape {b_arr.shape} incompatible with ({n_draws}, {num_c})"
            )

    # Row-varying 3D active_mask path (masks out per-draw NaNs cleanly)
    if active_mask is not None:
        m_arr = np.asarray(active_mask, dtype=bool)
        if m_arr.shape == (n_draws, num_c, num_p):
            masked_arr = np.where(m_arr, arr, 0 if is_int_dtype else 0.0)
            all_cols = [np.arange(num_p, dtype=np.intp) for _ in range(num_c)]
            allocate_packed_chunk_inplace(
                masked_arr,
                out,
                seats_vec,
                city_vec,
                all_cols,
                blank_chunk=blank_mat,
                threshold=threshold,
                is_int_mode=is_int_dtype,
            )
            return out
        if m_arr.shape != (num_c, num_p):
            raise SchemaValidationError(
                f"active_mask shape {m_arr.shape} != ({num_c}, {num_p}) "
                f"or ({n_draws}, {num_c}, {num_p})"
            )
        active_per_c = [np.flatnonzero(m_arr[c]) for c in range(num_c)]
    else:
        full_cols = np.arange(num_p, dtype=np.intp)
        active_per_c = [full_cols for _ in range(num_c)]

    if n_workers is None:
        eff_workers = min(2, os.cpu_count() or 1) if n_draws >= 4000 else 1
    else:
        eff_workers = max(1, int(n_workers))

    if eff_workers == 1 or n_draws < 1000:
        allocate_packed_chunk_inplace(
            arr,
            out,
            seats_vec,
            city_vec,
            active_per_c,
            blank_chunk=blank_mat,
            threshold=threshold,
            is_int_mode=is_int_dtype,
        )
        return out

    chunk_sz = max(2500, (n_draws + eff_workers * 2 - 1) // (eff_workers * 2))
    ranges = [(i, min(n_draws, i + chunk_sz)) for i in range(0, n_draws, chunk_sz)]

    def _run_range(r: tuple[int, int]) -> None:
        s_i, e_i = r
        allocate_packed_chunk_inplace(
            arr[s_i:e_i],
            out[s_i:e_i],
            seats_vec,
            city_vec,
            active_per_c,
            blank_chunk=blank_mat[s_i:e_i],
            threshold=threshold,
            is_int_mode=is_int_dtype,
        )

    with ThreadPoolExecutor(max_workers=eff_workers) as pool:
        list(pool.map(_run_range, ranges))
    return out
