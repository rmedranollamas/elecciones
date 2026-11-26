"""Stage 4: Spanish Constitution Art. 99 Investiture & Deadlock Engine.

Computes Monte Carlo probabilities across simulated 350-seat Congress chambers for:
1. First ballot (Art. 99.3 CE): absolute majority ``sum(yes) >= 176``.
2. Second ballot (Art. 99.3 CE, 48h later): simple majority ``sum(yes) > sum(no)``
   (strictly greater; abstentions excluded, ties fail).
3. Repeat-election constitutional deadlock risk (Art. 99.5 CE): fraction of
   simulated Parliaments clearing neither bloc's governable 1st/2nd-ballot paths.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np

from elecciones.schemas import (
    CONGRESS_TOTAL_SEATS,
    InvestitureSummary,
    PactProbability,
    SecondBallotScenario,
)

CONGRESS_MAJORITY_SEATS: Final[int] = (CONGRESS_TOTAL_SEATS // 2) + 1

MAJORITY_HISTOGRAM_X_PARTIES: Final[tuple[str, ...]] = ("PP", "UPN_PP", "VOX")
MAJORITY_HISTOGRAM_PSOE_ALLIES: Final[tuple[str, ...]] = (
    "PSOE",
    "FA",
    "PODEMOS",
    "COMPROMIS_FV",
    "UPC",
    "ARA_MES",
    "ADELANTE",
    "ERC",
    "EH_BILDU",
    "PNV",
    "GEROA_BAI",
    "BNG",
    "CCA",
    "TERUEL_EXISTE",
    "SORIA_YA",
)
# Stacked bottom -> top order matching editorial right-to-left coalition progression
MAJORITY_HISTOGRAM_CATEGORIES: Final[tuple[dict[str, str], ...]] = (
    {
        "key": "pp_solo",
        "label": "PP solo",
        "sublabel": "PP (+UPN) ≥176",
        "color": "#0c2c5c",
    },
    {
        "key": "pp_sin_vox",
        "label": "PP sin Vox",
        "sublabel": "con PNV y CCa",
        "color": "#7ea9d4",
    },
    {
        "key": "pp_vox",
        "label": "PP y Vox",
        "sublabel": "mayoría absoluta",
        "color": "#0055a7",
    },
    {
        "key": "en_el_aire",
        "label": "En el aire",
        "sublabel": "depende de Junts",
        "color": "#9ca3af",
    },
    {
        "key": "psoe_socios",
        "label": "Aliados de 2023",
        "sublabel": "PSOE y socios, sin Junts",
        "color": "#dc2626",
    },
)

# Canonical 1st-ballot absolute majority (>= 176) coalition definitions
DEFAULT_FIRST_BALLOT_SCENARIOS: Final[tuple[tuple[str, str, tuple[str, ...]], ...]] = (
    ("pp_solo", "PP (+UPN) mayoría absoluta", ("PP", "UPN_PP")),
    ("pp_vox", "PP + Vox (≥176)", ("PP", "UPN_PP", "VOX")),
    ("pp_vox_salf", "Bloque derecha (PP + Vox + SALF)", ("PP", "UPN_PP", "VOX", "SALF")),
    (
        "pp_vox_cca_pnv",
        "PP + Vox + CCa + PNV",
        ("PP", "UPN_PP", "VOX", "CCA", "PNV"),
    ),
    (
        "psoe_left",
        "Bloque progresista (PSOE + Izquierdas)",
        ("PSOE", "FA", "PODEMOS", "COMPROMIS_FV", "UPC", "ARA_MES", "ADELANTE"),
    ),
    (
        "psoe_left_no_junts",
        "Mayoría progresista sin Junts",
        (
            "PSOE",
            "FA",
            "PODEMOS",
            "COMPROMIS_FV",
            "UPC",
            "ARA_MES",
            "ADELANTE",
            "ERC",
            "EH_BILDU",
            "PNV",
            "GEROA_BAI",
            "BNG",
            "CCA",
            "TERUEL_EXISTE",
            "SORIA_YA",
        ),
    ),
    (
        "psoe_plurinational",
        "Mayoría plurinacional (con Junts)",
        (
            "PSOE",
            "FA",
            "PODEMOS",
            "COMPROMIS_FV",
            "UPC",
            "ARA_MES",
            "ADELANTE",
            "ERC",
            "EH_BILDU",
            "PNV",
            "GEROA_BAI",
            "BNG",
            "CCA",
            "TERUEL_EXISTE",
            "SORIA_YA",
            "JUNTS",
        ),
    ),
    ("grand_coalition", "Gran coalición (PP + PSOE)", ("PP", "UPN_PP", "PSOE")),
)

# Canonical 2nd-ballot simple majority (Sí > No) definitions with explicit abstentions
_DEFAULT_SECOND_BALLOT_SPECS: Final[tuple[dict[str, Any], ...]] = (
    {
        "key": "pp_minority_vox_abstain",
        "label": "PP (+UPN + CCa) en minoría (abstención Vox/SALF/PNV/Junts)",
        "yes": ("PP", "UPN_PP", "CCA"),
        "abstain": (
            "VOX",
            "SALF",
            "PNV",
            "GEROA_BAI",
            "JUNTS",
            "TERUEL_EXISTE",
            "SORIA_YA",
            "ALHORA",
        ),
    },
    {
        "key": "pp_vox_cca_yes_junts_abstain",
        "label": "PP + Vox + CCa (2.ª vuelta, abstención SALF/PNV/Junts)",
        "yes": ("PP", "UPN_PP", "VOX", "CCA"),
        "abstain": ("SALF", "PNV", "GEROA_BAI", "JUNTS", "TERUEL_EXISTE", "SORIA_YA", "ALHORA"),
    },
    {
        "key": "pp_vox_salf_cca_yes",
        "label": "PP + Vox + SALF + CCa (2.ª vuelta Sí > No)",
        "yes": ("PP", "UPN_PP", "VOX", "SALF", "CCA"),
        "abstain": ("PNV", "GEROA_BAI", "TERUEL_EXISTE", "SORIA_YA", "ALHORA"),
    },
    {
        "key": "psoe_left_junts_abstain",
        "label": "Gobierno progresista (abstención Junts/CCa)",
        "yes": (
            "PSOE",
            "FA",
            "PODEMOS",
            "COMPROMIS_FV",
            "UPC",
            "ARA_MES",
            "ADELANTE",
            "ERC",
            "EH_BILDU",
            "PNV",
            "GEROA_BAI",
            "BNG",
            "TERUEL_EXISTE",
            "SORIA_YA",
        ),
        "abstain": ("JUNTS", "CCA"),
    },
    {
        "key": "psoe_left_podemos_junts_abstain",
        "label": "PSOE + FA + socios (abstención Podemos/EH Bildu/Junts)",
        "yes": (
            "PSOE",
            "FA",
            "COMPROMIS_FV",
            "UPC",
            "ARA_MES",
            "ADELANTE",
            "ERC",
            "PNV",
            "GEROA_BAI",
            "BNG",
            "TERUEL_EXISTE",
            "SORIA_YA",
        ),
        "abstain": ("PODEMOS", "EH_BILDU", "JUNTS", "CCA"),
    },
    {
        "key": "psoe_plurinational_yes",
        "label": "Bloque plurinacional (2.ª vuelta Sí > No)",
        "yes": (
            "PSOE",
            "FA",
            "PODEMOS",
            "COMPROMIS_FV",
            "UPC",
            "ARA_MES",
            "ADELANTE",
            "ERC",
            "EH_BILDU",
            "PNV",
            "GEROA_BAI",
            "BNG",
            "CCA",
            "TERUEL_EXISTE",
            "SORIA_YA",
            "JUNTS",
        ),
        "abstain": ("ALHORA", "FO", "PACMA", "OTROS"),
    },
)


def _sum_seat_columns(
    seats_matrix: np.ndarray,
    p_idx: Mapping[str, int],
    parties: Sequence[str],
) -> np.ndarray:
    """Sum national seats across ``parties`` for every simulation row ``(n_sims,)``."""
    n_sims = seats_matrix.shape[0]
    cols = [p_idx[p] for p in parties if p in p_idx]
    if not cols:
        return np.zeros(n_sims, dtype=np.int32)
    return seats_matrix[:, cols].sum(axis=1, dtype=np.int32)


def evaluate_first_ballot(
    seats_matrix: np.ndarray,
    p_idx: Mapping[str, int],
    parties: Sequence[str],
    label: str,
    *,
    majority_threshold: int = CONGRESS_MAJORITY_SEATS,
) -> PactProbability:
    """Compute Art. 99.3 CE 1st-ballot absolute majority (``>= 176``) probability."""
    totals = _sum_seat_columns(seats_matrix, p_idx, parties)
    prob = float(np.mean(totals >= majority_threshold))
    q10, q50, q90 = np.percentile(totals, [10.0, 50.0, 90.0], method="nearest")
    p10_i = int(q10)
    med_i = max(p10_i, int(q50))
    p90_i = max(med_i, int(q90))
    # Deduplicate preserving order
    unique_parties = tuple(dict.fromkeys(parties))
    return PactProbability(
        label=label,
        parties=unique_parties,
        prob=round(prob, 6),
        median_seats=med_i,
        p10=p10_i,
        p90=p90_i,
    )


def evaluate_second_ballot(
    seats_matrix: np.ndarray,
    p_idx: Mapping[str, int],
    *,
    label: str,
    yes: Sequence[str],
    no: Sequence[str],
    abstain: Sequence[str],
) -> SecondBallotScenario:
    """Compute Art. 99.3 CE 2nd-ballot simple majority (strict ``Sí > No``) probability."""
    yes_u = tuple(dict.fromkeys(yes))
    abs_u = tuple(p for p in dict.fromkeys(abstain) if p not in yes_u)
    no_u = tuple(p for p in dict.fromkeys(no) if p not in yes_u and p not in abs_u)

    yes_seats = _sum_seat_columns(seats_matrix, p_idx, yes_u)
    no_seats = _sum_seat_columns(seats_matrix, p_idx, no_u)
    margin = yes_seats - no_seats
    prob = float(np.mean(margin > 0))
    med_margin = int(np.percentile(margin, 50.0, method="nearest"))
    return SecondBallotScenario(
        label=label,
        yes=yes_u,
        no=no_u,
        abstain=abs_u,
        prob=round(prob, 6),
        median_margin=med_margin,
    )


def _sum_share_columns(
    shares_matrix: np.ndarray,
    p_idx: Mapping[str, int],
    parties: Sequence[str],
) -> np.ndarray:
    """Sum national vote shares across ``parties`` for every simulation row ``(n_sims,)``."""
    n_sims = shares_matrix.shape[0]
    cols = [p_idx[p] for p in parties if p in p_idx]
    if not cols:
        return np.zeros(n_sims, dtype=np.float64)
    return shares_matrix[:, cols].sum(axis=1, dtype=np.float64)


def compute_majority_histogram(
    national_seats: np.ndarray,
    national_shares: np.ndarray,
    party_ids: Sequence[str],
    *,
    bin_width: float = 0.5,
    majority_threshold: int = CONGRESS_MAJORITY_SEATS,
) -> dict[str, Any]:
    """Compute mutually-exclusive majority breakdown binned by combined PP + Vox vote share.

    Every simulation draw is assigned to exactly one of five mutually exclusive outcomes
    (ordered bottom-to-top in the stacked histogram):
      1. ``pp_solo``: ``PP (+UPN_PP) >= 176``
      2. ``pp_sin_vox``: not ``pp_solo`` and ``PP (+UPN_PP) + PNV + CCA >= 176``
      3. ``pp_vox``: not above and ``PP (+UPN_PP) + VOX >= 176``
      4. ``psoe_socios``: not above and ``PSOE + allies (no Junts) >= 176``
      5. ``en_el_aire``: remainder (hung / depends on Junts or cross-bloc abstentions)
    """
    seats_mat = np.asarray(national_seats, dtype=np.int32)
    shares_mat = np.asarray(national_shares, dtype=np.float64)
    if seats_mat.ndim != 2 or seats_mat.shape[1] != len(party_ids):
        raise ValueError(
            f"national_seats shape {seats_mat.shape} incompatible with {len(party_ids)} parties"
        )
    if shares_mat.shape != seats_mat.shape:
        raise ValueError(
            f"national_shares shape {shares_mat.shape} must match national_seats {seats_mat.shape}"
        )
    n_sims = int(seats_mat.shape[0])
    if n_sims < 1:
        raise ValueError("national_seats must contain at least 1 simulation")
    if bin_width <= 0.0:
        raise ValueError(f"bin_width must be > 0, got {bin_width}")

    p_idx = {pid: j for j, pid in enumerate(party_ids)}

    # Combined PP (+UPN_PP in Navarre) + Vox national valid-vote share in percentage points [0, 100]
    raw_x = _sum_share_columns(shares_mat, p_idx, MAJORITY_HISTOGRAM_X_PARTIES)
    scale = 1.0 if float(np.max(raw_x)) > 1.5 else 100.0
    x_pct = np.clip(raw_x * scale, 0.0, 100.0)

    pp_solo = _sum_seat_columns(seats_mat, p_idx, ("PP", "UPN_PP")) >= majority_threshold
    pp_sin_vox = (~pp_solo) & (
        _sum_seat_columns(seats_mat, p_idx, ("PP", "UPN_PP", "PNV", "CCA")) >= majority_threshold
    )
    pp_vox = (~pp_solo) & (~pp_sin_vox) & (
        _sum_seat_columns(seats_mat, p_idx, MAJORITY_HISTOGRAM_X_PARTIES) >= majority_threshold
    )
    right_any = pp_solo | pp_sin_vox | pp_vox
    psoe_socios = (~right_any) & (
        _sum_seat_columns(seats_mat, p_idx, MAJORITY_HISTOGRAM_PSOE_ALLIES) >= majority_threshold
    )
    en_el_aire = ~(right_any | psoe_socios)

    cat_masks: dict[str, np.ndarray] = {
        "pp_solo": pp_solo,
        "pp_sin_vox": pp_sin_vox,
        "pp_vox": pp_vox,
        "en_el_aire": en_el_aire,
        "psoe_socios": psoe_socios,
    }

    # Dynamic x-axis window spanning [P0.3, P99.7] snapped to clean bin_width steps
    q_low, q_high = (float(v) for v in np.percentile(x_pct, [0.3, 99.7]))
    eff_bw = float(bin_width)
    x_min = max(0.0, math.floor(q_low / eff_bw) * eff_bw)
    x_max = min(100.0, math.ceil(q_high / eff_bw) * eff_bw)
    if x_max - x_min < eff_bw * 4.0:
        center = 0.5 * (x_min + x_max)
        x_min = max(0.0, math.floor((center - eff_bw * 2.0) / eff_bw) * eff_bw)
        x_max = min(100.0, x_min + eff_bw * 4.0)
    n_bins = max(1, int(round((x_max - x_min) / eff_bw)))
    if n_bins > 80:
        eff_bw = 1.0
        x_min = max(0.0, math.floor(q_low / eff_bw) * eff_bw)
        x_max = min(100.0, math.ceil(q_high / eff_bw) * eff_bw)
        n_bins = max(1, int(round((x_max - x_min) / eff_bw)))

    # Vectorized bin indexing (include right edge x == x_max in final bin)
    raw_idx = np.floor((x_pct - x_min) / eff_bw).astype(np.int64)
    at_top = np.isclose(x_pct, x_max, atol=1e-9)
    raw_idx = np.where(at_top, n_bins - 1, raw_idx)
    in_range = (raw_idx >= 0) & (raw_idx < n_bins)
    clipped_low = int(np.sum(raw_idx < 0))
    clipped_high = int(np.sum(raw_idx >= n_bins))
    valid_idx = raw_idx[in_range]

    counts_by_cat: dict[str, np.ndarray] = {}
    for spec in MAJORITY_HISTOGRAM_CATEGORIES:
        k = spec["key"]
        mask_in = cat_masks[k][in_range]
        counts_by_cat[k] = np.bincount(valid_idx[mask_in], minlength=n_bins)

    categories_out: list[dict[str, Any]] = []
    for spec in MAJORITY_HISTOGRAM_CATEGORIES:
        k = spec["key"]
        mask = cat_masks[k]
        cnt = int(np.sum(mask))
        prob = round(cnt / n_sims, 6)
        if cnt > 0:
            x_sub = x_pct[mask]
            c_p10, c_med, c_p90 = (
                round(float(v), 2) for v in np.percentile(x_sub, [10.0, 50.0, 90.0])
            )
        else:
            c_p10 = c_med = c_p90 = None
        categories_out.append(
            {
                "key": k,
                "label": spec["label"],
                "sublabel": spec["sublabel"],
                "color": spec["color"],
                "count": cnt,
                "prob": prob,
                "median_vote_pct": c_med,
                "p10_vote_pct": c_p10,
                "p90_vote_pct": c_p90,
            }
        )

    bins_out: list[dict[str, Any]] = []
    min_bin_for_thresh = max(15, int(n_sims * 0.0005))
    thresh_pts: list[tuple[float, float]] = []
    for b in range(n_bins):
        x0 = round(x_min + b * eff_bw, 4)
        x1 = round(x_min + (b + 1) * eff_bw, 4)
        b_counts = {
            spec["key"]: int(counts_by_cat[spec["key"]][b])
            for spec in MAJORITY_HISTOGRAM_CATEGORIES
        }
        b_total = sum(b_counts.values())
        bins_out.append(
            {
                "x0": x0,
                "x1": x1,
                "total": b_total,
                "counts": b_counts,
            }
        )
        if b_total >= min_bin_for_thresh:
            r_share = (
                b_counts["pp_solo"] + b_counts["pp_sin_vox"] + b_counts["pp_vox"]
            ) / b_total
            thresh_pts.append((0.5 * (x0 + x1), r_share))

    threshold_50_x: float | None = None
    for idx_pt in range(1, len(thresh_pts)):
        x_prev, p_prev = thresh_pts[idx_pt - 1]
        x_cur, p_cur = thresh_pts[idx_pt]
        if p_prev < 0.5 <= p_cur and (p_cur - p_prev) > 1e-9:
            frac = (0.5 - p_prev) / (p_cur - p_prev)
            threshold_50_x = round(x_prev + frac * (x_cur - x_prev), 2)
            break

    p10_x, med_x, p90_x = (round(float(v), 2) for v in np.percentile(x_pct, [10.0, 50.0, 90.0]))
    return {
        "n_sims": n_sims,
        "bin_width": round(eff_bw, 4),
        "x_min": round(x_min, 4),
        "x_max": round(x_max, 4),
        "x_label": "Voto de PP y Vox, en %",
        "x_parties": list(MAJORITY_HISTOGRAM_X_PARTIES),
        "mean_x": round(float(np.mean(x_pct)), 2),
        "median_x": med_x,
        "p10_x": p10_x,
        "p90_x": p90_x,
        "threshold_50_x": threshold_50_x,
        "clipped_low": clipped_low,
        "clipped_high": clipped_high,
        "stack_order": [spec["key"] for spec in MAJORITY_HISTOGRAM_CATEGORIES],
        "categories": categories_out,
        "bins": bins_out,
    }


def compute_investiture_summary(
    national_seats: np.ndarray,
    party_ids: Sequence[str],
    *,
    national_shares: np.ndarray | None = None,
    first_ballot_specs: Sequence[tuple[str, str, tuple[str, ...]]] | None = None,
    second_ballot_specs: Sequence[Mapping[str, Any]] | None = None,
) -> InvestitureSummary:
    """Compute complete Art. 99 1st-ballot, 2nd-ballot & repeat-election deadlock summary."""
    seats_mat = np.asarray(national_seats, dtype=np.int32)
    if seats_mat.ndim != 2 or seats_mat.shape[1] != len(party_ids):
        raise ValueError(
            f"national_seats shape {seats_mat.shape} incompatible with {len(party_ids)} parties"
        )
    if seats_mat.shape[0] < 1:
        raise ValueError("national_seats must contain at least 1 simulation")

    p_idx = {pid: j for j, pid in enumerate(party_ids)}
    all_known = list(dict.fromkeys([*party_ids]))

    fb_specs = first_ballot_specs or DEFAULT_FIRST_BALLOT_SCENARIOS
    first_ballot: dict[str, PactProbability] = {}
    for key, label, members in fb_specs:
        first_ballot[key] = evaluate_first_ballot(seats_mat, p_idx, members, label)

    sb_specs = second_ballot_specs or _DEFAULT_SECOND_BALLOT_SPECS
    second_ballot: dict[str, SecondBallotScenario] = {}
    for spec in sb_specs:
        yes_p = tuple(spec["yes"])
        abs_p = tuple(spec["abstain"])
        if "no" in spec and spec["no"] is not None:
            no_p = tuple(spec["no"])
        else:
            excluded = set(yes_p) | set(abs_p)
            no_p = tuple(p for p in all_known if p not in excluded)
        second_ballot[str(spec["key"])] = evaluate_second_ballot(
            seats_mat,
            p_idx,
            label=str(spec["label"]),
            yes=yes_p,
            no=no_p,
            abstain=abs_p,
        )

    # Deadlock (Art. 99.5 CE): neither plausible Right coalition (PP+Vox[+CCa] beating Left+Periph)
    # nor plausible Progressive/Plurinational coalition (PSOE+Left+partners beating Right even with
    # Junts abstaining) clears Congress.
    right_1 = _sum_seat_columns(seats_mat, p_idx, ("PP", "UPN_PP", "VOX")) >= 176
    right_yes_2 = _sum_seat_columns(seats_mat, p_idx, ("PP", "UPN_PP", "VOX", "CCA"))
    left_oppos = _sum_seat_columns(
        seats_mat,
        p_idx,
        (
            "PSOE",
            "FA",
            "PODEMOS",
            "COMPROMIS_FV",
            "UPC",
            "ARA_MES",
            "ADELANTE",
            "ERC",
            "EH_BILDU",
            "PNV",
            "GEROA_BAI",
            "BNG",
            "JUNTS",
        ),
    )
    right_2 = right_yes_2 > left_oppos

    left_yes_parties = (
        "PSOE",
        "FA",
        "PODEMOS",
        "COMPROMIS_FV",
        "UPC",
        "ARA_MES",
        "ADELANTE",
        "ERC",
        "EH_BILDU",
        "PNV",
        "GEROA_BAI",
        "BNG",
        "TERUEL_EXISTE",
        "SORIA_YA",
    )
    left_1 = _sum_seat_columns(seats_mat, p_idx, (*left_yes_parties, "JUNTS", "CCA")) >= 176
    left_yes_2 = _sum_seat_columns(seats_mat, p_idx, left_yes_parties)
    right_oppos = _sum_seat_columns(seats_mat, p_idx, ("PP", "UPN_PP", "VOX", "SALF"))
    left_2 = left_yes_2 > right_oppos

    governable = right_1 | right_2 | left_1 | left_2
    deadlock_prob = float(np.mean(~governable))

    majority_histogram = (
        compute_majority_histogram(seats_mat, national_shares, party_ids)
        if national_shares is not None
        else None
    )

    return InvestitureSummary(
        first_ballot=first_ballot,
        second_ballot=second_ballot,
        deadlock_prob=round(deadlock_prob, 6),
        majority_histogram=majority_histogram,
    )
