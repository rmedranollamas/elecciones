"""Stage 4: Spanish Constitution Art. 99 Investiture & Deadlock Engine.

Computes Monte Carlo probabilities across simulated 350-seat Congress chambers for:
1. First ballot (Art. 99.3 CE): absolute majority ``sum(yes) >= 176``.
2. Second ballot (Art. 99.3 CE, 48h later): simple majority ``sum(yes) > sum(no)``
   (strictly greater; abstentions excluded, ties fail).
3. Repeat-election constitutional deadlock risk (Art. 99.5 CE): fraction of
   simulated Parliaments clearing neither bloc's governable 1st/2nd-ballot paths.
"""

from __future__ import annotations

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


def compute_investiture_summary(
    national_seats: np.ndarray,
    party_ids: Sequence[str],
    *,
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

    return InvestitureSummary(
        first_ballot=first_ballot,
        second_ballot=second_ballot,
        deadlock_prob=round(deadlock_prob, 6),
    )
