"""Milestone 1 test suite: LOREG Art. 163 D'Hondt, numerical precision, schemas, and CLI."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from elecciones.cli import app
from elecciones.config import (
    CONGRESS_TOTAL_SEATS,
    SIMULATION_SEED,
    is_in_blackout_window,
)
from elecciones.models.dhondt import (
    allocate_batch_vectorized,
    allocate_dhondt_loreg,
    allocate_packed_chunk_inplace,
)
from elecciones.schemas import (
    ALL_CPROS,
    SENATE_DIRECT_SEATS,
    AllocationInput,
    AllocationResult,
    BacktestSummary,
    BatchAllocationResult,
    ConstituenciesCatalog,
    ConstituencyElectionResult,
    ConstituencyForecast,
    ForecastBundle,
    ForecastMetadata,
    InvestitureSummary,
    LastSeatBattle,
    ManualPollOverlay,
    ModelBacktestMetrics,
    PactProbability,
    PartiesCatalog,
    PartyConstituencyProjection,
    PartyNationalSeats,
    PartySpec,
    PollRecord,
    RegionalPollRecord,
    RunnerUpGap,
    SchemaValidationError,
    SeatSlot,
    SecondBallotScenario,
    ThresholdExhaustionError,
    TierCalibrationBucket,
    VoteShareInterval,
    parse_allocation_payload,
)

RUNNER = CliRunner()

CCAA_NAMES: dict[str, str] = {
    "01": "Andalucía",
    "02": "Aragón",
    "03": "Principado de Asturias",
    "04": "Illes Balears",
    "05": "Canarias",
    "06": "Cantabria",
    "07": "Castilla y León",
    "08": "Castilla-La Mancha",
    "09": "Cataluña",
    "10": "Comunitat Valenciana",
    "11": "Extremadura",
    "12": "Galicia",
    "13": "Comunidad de Madrid",
    "14": "Región de Murcia",
    "15": "Comunidad Foral de Navarra",
    "16": "País Vasco",
    "17": "La Rioja",
    "18": "Ceuta",
    "19": "Melilla",
}

# Canonical 52 constituencies (RD 806/2026: Madrid=38, Cádiz=8; Censo & Válidos from 23-J 2023)
ROWS_52: list[tuple[str, str, str, str, int, int, int, int, int, int, int]] = [
    ("01", "Araba/Álava", "Álava", "16", 4, 4, 4, 1, 5, 260214, 169107),
    ("02", "Albacete", "Albacete", "08", 4, 4, 4, 5, 5, 307988, 221191),
    ("03", "Alicante/Alacant", "Alicante", "10", 12, 12, 12, 5, 7, 1306311, 899192),
    ("04", "Almería", "Almería", "01", 6, 6, 6, 7, 6, 516114, 321242),
    ("05", "Ávila", "Ávila", "07", 3, 3, 3, 3, 3, 136669, 97956),
    ("06", "Badajoz", "Badajoz", "11", 6, 5, 5, 5, 2, 550702, 391461),
    ("07", "Illes Balears", "Illes Balears", "04", 8, 8, 8, 3, 9, 828755, 503280),
    ("08", "Barcelona", "Barcelona", "09", 32, 32, 32, 1, 9, 4243500, 2667478),
    ("09", "Burgos", "Burgos", "07", 4, 4, 4, 1, 4, 296401, 202569),
    ("10", "Cáceres", "Cáceres", "11", 4, 4, 4, 4, 2, 339491, 235851),
    ("11", "Cádiz", "Cádiz", "01", 9, 9, 8, 7, 3, 1017958, 637098),
    ("12", "Castellón/Castelló", "Castellón", "10", 5, 5, 5, 3, 7, 430830, 307930),
    ("13", "Ciudad Real", "C. Real", "08", 5, 5, 5, 5, 4, 391495, 282979),
    ("14", "Córdoba", "Córdoba", "01", 6, 6, 6, 6, 4, 645288, 448602),
    ("15", "A Coruña", "A Coruña", "12", 8, 8, 8, 0, 0, 1085673, 677702),
    ("16", "Cuenca", "Cuenca", "08", 3, 3, 3, 4, 5, 152694, 113808),
    ("17", "Girona", "Girona", "09", 6, 6, 6, 0, 9, 546370, 322988),
    ("18", "Granada", "Granada", "01", 7, 7, 7, 7, 5, 765478, 502845),
    ("19", "Guadalajara", "Guadalajara", "08", 3, 3, 3, 3, 5, 196983, 140913),
    ("20", "Gipuzkoa", "Gipuzkoa", "16", 6, 6, 6, 0, 5, 585873, 376621),
    ("21", "Huelva", "Huelva", "01", 5, 5, 5, 6, 2, 402504, 256804),
    ("22", "Huesca", "Huesca", "02", 3, 3, 3, 0, 7, 173391, 119895),
    ("23", "Jaén", "Jaén", "01", 5, 5, 5, 6, 5, 518371, 368731),
    ("24", "León", "León", "07", 4, 4, 4, 1, 2, 422917, 275240),
    ("25", "Lleida", "Lleida", "09", 4, 4, 4, 1, 8, 317225, 184539),
    ("26", "La Rioja", "La Rioja", "17", 4, 4, 4, 1, 6, 252091, 174663),
    ("27", "Lugo", "Lugo", "12", 4, 4, 4, 0, 1, 337755, 198566),
    ("28", "Madrid", "Madrid", "13", 37, 37, 38, 3, 4, 5224561, 3608673),
    ("29", "Málaga", "Málaga", "01", 11, 11, 11, 7, 4, 1239846, 788338),
    ("30", "Murcia", "Murcia", "14", 10, 10, 10, 6, 6, 1099703, 747797),
    ("31", "Navarra", "Navarra", "15", 5, 5, 5, 0, 6, 519918, 341844),
    ("32", "Ourense", "Ourense", "12", 4, 4, 4, 1, 1, 355765, 186271),
    ("33", "Asturias", "Asturias", "03", 7, 7, 7, 0, 2, 959023, 597032),
    ("34", "Palencia", "Palencia", "07", 3, 3, 3, 1, 3, 137642, 96962),
    ("35", "Las Palmas", "Las Palmas", "05", 8, 8, 8, 8, 1, 898882, 529337),
    ("36", "Pontevedra", "Pontevedra", "12", 7, 7, 7, 1, 0, 909434, 573124),
    ("37", "Salamanca", "Salamanca", "07", 4, 4, 4, 3, 2, 300506, 199110),
    ("38", "Santa Cruz de Tenerife", "S. C. Tenerife", "05", 7, 7, 7, 8, 0, 881217, 494373),
    ("39", "Cantabria", "Cantabria", "06", 5, 5, 5, 0, 3, 507818, 350150),
    ("40", "Segovia", "Segovia", "07", 3, 3, 3, 2, 4, 119454, 88613),
    ("41", "Sevilla", "Sevilla", "01", 12, 12, 12, 6, 3, 1564082, 1059358),
    ("42", "Soria", "Soria", "07", 2, 2, 2, 2, 5, 75768, 50769),
    ("43", "Tarragona", "Tarragona", "09", 6, 6, 6, 2, 8, 596689, 368024),
    ("44", "Teruel", "Teruel", "02", 3, 3, 3, 2, 7, 106352, 75913),
    ("45", "Toledo", "Toledo", "08", 6, 6, 6, 4, 4, 542567, 387200),
    ("46", "Valencia/València", "Valencia", "10", 15, 16, 16, 4, 7, 1999340, 1437161),
    ("47", "Valladolid", "Valladolid", "07", 5, 5, 5, 2, 3, 431381, 316309),
    ("48", "Bizkaia", "Bizkaia", "16", 8, 8, 8, 0, 4, 942983, 609693),
    ("49", "Zamora", "Zamora", "07", 3, 3, 3, 2, 2, 162998, 103410),
    ("50", "Zaragoza", "Zaragoza", "02", 7, 7, 7, 1, 7, 740060, 518705),
    ("51", "Ceuta", "Ceuta", "18", 1, 1, 1, 8, 7, 63310, 33321),
    ("52", "Melilla", "Melilla", "19", 1, 1, 1, 8, 8, 61118, 27349),
]

INSULAR_SENATE: dict[str, dict[str, object]] = {
    "07": {
        "direct_seats": 5,
        "max_votes_per_ballot": 2,
        "subdistricts": [
            {"id": "07-MALLORCA", "name": "Mallorca", "direct_seats": 3, "max_votes_per_ballot": 2},
            {"id": "07-MENORCA", "name": "Menorca", "direct_seats": 1, "max_votes_per_ballot": 1},
            {
                "id": "07-IBIZA_FORMENTERA",
                "name": "Ibiza-Formentera",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
        ],
    },
    "35": {
        "direct_seats": 5,
        "max_votes_per_ballot": 2,
        "subdistricts": [
            {
                "id": "35-GRAN_CANARIA",
                "name": "Gran Canaria",
                "direct_seats": 3,
                "max_votes_per_ballot": 2,
            },
            {
                "id": "35-FUERTEVENTURA",
                "name": "Fuerteventura",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
            {
                "id": "35-LANZAROTE",
                "name": "Lanzarote",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
        ],
    },
    "38": {
        "direct_seats": 6,
        "max_votes_per_ballot": 2,
        "subdistricts": [
            {"id": "38-TENERIFE", "name": "Tenerife", "direct_seats": 3, "max_votes_per_ballot": 2},
            {"id": "38-LA_PALMA", "name": "La Palma", "direct_seats": 1, "max_votes_per_ballot": 1},
            {
                "id": "38-LA_GOMERA",
                "name": "La Gomera",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
            {
                "id": "38-EL_HIERRO",
                "name": "El Hierro",
                "direct_seats": 1,
                "max_votes_per_ballot": 1,
            },
        ],
    },
}


def _build_52_catalog_dict() -> dict[str, object]:
    items = []
    for cpro, name, short_name, codauto, s19, s23, s26, row, col, censo, validos in ROWS_52:
        is_city = cpro in {"51", "52"}
        senate = (
            {"direct_seats": 2, "max_votes_per_ballot": 2, "subdistricts": []}
            if is_city
            else INSULAR_SENATE.get(
                cpro, {"direct_seats": 4, "max_votes_per_ballot": 3, "subdistricts": []}
            )
        )
        items.append(
            {
                "cpro": cpro,
                "name": name,
                "short_name": short_name,
                "codauto": codauto,
                "ccaa_name": CCAA_NAMES[codauto],
                "district_type": "autonomous_city" if is_city else "province",
                "threshold": 0.0 if is_city else 0.03,
                "seats_2019": s19,
                "seats_2023": s23,
                "seats_2026": s26,
                "cartogram_row": row,
                "cartogram_col": col,
                "censo_2023": censo,
                "validos_2023": validos,
                "senate": senate,
            }
        )
    return {
        "chamber": "congress",
        "election_date": "2026-11-29",
        "boe_source": "BOE-A-2026-20742",
        "total_seats": 350,
        "total_senate_direct_seats": 208,
        "constituencies": items,
    }


# ---------------------------------------------------------------------------
# 1. Core LOREG Art. 163 rules (blanks, nulls, threshold, ties, Ceuta/Melilla)
# ---------------------------------------------------------------------------


def test_blank_votes_included_in_valid_votes_threshold() -> None:
    # 2,990 / (95,000 + 2,990 + 2,010 = 100,000) = 2.99% < 3% even though 2,990/97,990 > 3%
    res = allocate_dhondt_loreg(
        {"PP": 95_000, "SMALL": 2_990},
        seats=35,
        blank_votes=2_010,
    )
    assert res.valid_votes == 100_000
    assert res.eligible_parties == ("PP",)
    assert res.seats == {"PP": 35, "SMALL": 0}


def test_exact_three_percent_threshold_admitted() -> None:
    res = allocate_dhondt_loreg(
        {"PP": 95_000, "SMALL": 3_000},
        seats=35,
        blank_votes=2_000,
    )
    assert res.eligible_parties == ("PP", "SMALL")
    assert res.seats == {"PP": 34, "SMALL": 1}


def test_below_three_percent_by_one_vote_excluded() -> None:
    res = allocate_dhondt_loreg(
        {"PP": 95_000, "SMALL": 2_999},
        seats=35,
        blank_votes=2_001,
    )
    assert res.seats == {"PP": 35, "SMALL": 0}


def test_null_votes_excluded_from_three_percent_denominator() -> None:
    # 3,000 / 100,000 valid = 3.00% -> eligible despite 50,000 null ballots
    res = allocate_dhondt_loreg(
        {"PP": 97_000, "SMALL": 3_000},
        seats=35,
        blank_votes=0,
        null_votes=50_000,
    )
    assert res.valid_votes == 100_000
    assert res.seats == {"PP": 34, "SMALL": 1}


def test_zero_vote_party_ignored_even_with_zero_threshold() -> None:
    res = allocate_dhondt_loreg({"PP": 1000, "ZERO": 0}, seats=3, threshold=0.0)
    assert res.eligible_parties == ("PP",)
    assert res.seats == {"PP": 3, "ZERO": 0}


def test_autonomous_city_ceuta_melilla_plurality_below_three_percent() -> None:
    # In Ceuta ("51") and Melilla ("52"), LOREG assigns 0% barrier and plurality wins
    votes = {f"P{i}": 200 + i for i in range(40)}
    res = allocate_dhondt_loreg(votes, seats=1, cpro="51", blank_votes=10_000)
    assert res.seats["P39"] == 1
    assert sum(res.seats.values()) == 1


def test_autonomous_city_melilla_explicit_flag_ignores_blanks_as_winner() -> None:
    res = allocate_dhondt_loreg(
        {"PP": 12_993, "PSOE": 11_300, "VOX": 8_700},
        seats=1,
        blank_votes=50_000,
        is_autonomous_city=True,
    )
    assert res.seats == {"PP": 1, "PSOE": 0, "VOX": 0}


def test_tie_broken_by_total_votes_loreg_art_163_1_d() -> None:
    # Step 1: A(100,000); Step 2: B(50,000) == A/2(50,000) -> A wins on total votes (100k > 50k)
    res = allocate_dhondt_loreg({"B": 50_000, "A": 100_000}, seats=2)
    assert res.seats == {"A": 2, "B": 0}
    assert res.ties_broken == 1


def test_exact_equal_votes_tie_uses_canonical_order_alternation() -> None:
    # Identical vote totals alternate in canonical_order
    res = allocate_dhondt_loreg(
        {"A": 50_000, "B": 50_000},
        seats=3,
        canonical_order=["B", "A"],
    )
    assert res.seats == {"B": 2, "A": 1}
    assert res.ties_broken == 2


def test_non_cutoff_ties_above_seat_limit_do_not_increment_ties_broken() -> None:
    # 100,000 vs 50,000 over 1 seat: Step 1 is strict winner (100k > 50k), no tie occurs
    res = allocate_dhondt_loreg({"A": 100_000, "B": 50_000}, seats=1)
    assert res.ties_broken == 0


def test_exhaustion_strict_raises_and_non_strict_returns_zero() -> None:
    # 95% blank ballots means all lists are below 3%
    non_strict = allocate_dhondt_loreg(
        {"A": 2_500, "B": 2_500},
        seats=4,
        blank_votes=95_000,
        raise_on_exhaustion=False,
    )
    assert non_strict.seats == {"A": 0, "B": 0}
    assert non_strict.cutoff_quotient is None
    assert non_strict.eligible_parties == ()

    with pytest.raises(ThresholdExhaustionError):
        allocate_dhondt_loreg(
            {"A": 2_500, "B": 2_500},
            seats=4,
            blank_votes=95_000,
            raise_on_exhaustion=True,
        )


def test_asturias_2023_historical_jec_fixture() -> None:
    # 23-J 2023 Congress in Asturias (cpro="33", 7 seats: PP 3, PSOE 2, SUMAR 1, VOX 1)
    votes_23j = {
        "PP": 212_816,
        "PSOE": 205_049,
        "VOX": 74_571,
        "SUMAR": 88_630,
        "PACMA": 3_478,
        "PCTE": 1_557,
        "FO": 3_110,
    }
    res = allocate_dhondt_loreg(votes_23j, seats=7, blank_votes=5_620, cpro="33")
    assert res.seats["PP"] == 3
    assert res.seats["PSOE"] == 2
    assert res.seats["SUMAR"] == 1
    assert res.seats["VOX"] == 1
    assert sum(res.seats.values()) == 7


def test_madrid_37_and_38_seats_exact_quotient_table() -> None:
    votes = {"PP": 1_463_183, "PSOE": 1_004_599, "SUMAR": 554_194, "VOX": 504_473}
    res37 = allocate_dhondt_loreg(votes, seats=37, blank_votes=29_800, cpro="28")
    res38 = allocate_dhondt_loreg(votes, seats=38, blank_votes=29_800, cpro="28")
    assert sum(res37.seats.values()) == 37
    assert sum(res38.seats.values()) == 38
    assert res37.quotient_table is not None and len(res37.quotient_table) > 37
    assert res37.last_seat_battle is not None


def test_canonical_order_preserves_unlisted_parties() -> None:
    res = allocate_dhondt_loreg(
        {"X": 50_000, "Y": 50_000, "Z": 60_000},
        seats=4,
        canonical_order=["Y", "X"],
    )
    assert res.seats == {"Z": 2, "Y": 1, "X": 1}


# ---------------------------------------------------------------------------
# 2. Numerical precision & D1-D6 adversarial checks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", [10**15, 10**16, 10**17, 10**18])
def test_d1_huge_int64_scales(scale: int) -> None:
    v_a, v_b = 2 * scale + 1, scale + 1
    det = allocate_dhondt_loreg({"A": v_a, "B": v_b}, seats=2)
    vec = allocate_batch_vectorized(np.array([[[v_a, v_b]]], dtype=np.int64), [2])[0, 0].tolist()
    assert [det.seats["A"], det.seats["B"]] == [1, 1] == vec


def test_d1_multiparty_sum_exceeds_int64_max() -> None:
    pv = {"P0": 5 * 10**18 + 3, "P1": 5 * 10**18 + 1}
    det = allocate_dhondt_loreg(pv, seats=3, blank_votes=2 * 10**18)
    vec = allocate_batch_vectorized(
        np.array([[[pv["P0"], pv["P1"]]]], dtype=np.int64),
        [3],
        blank_votes_or_shares=np.array([[2 * 10**18]], dtype=np.int64),
    )[0, 0].tolist()
    assert [det.seats["P0"], det.seats["P1"]] == vec


def test_d1_uint64_above_int64_max() -> None:
    u_a = (1 << 63) + 101
    u_b = u_a // 2 + 1
    det = allocate_dhondt_loreg({"A": u_a, "B": u_b}, seats=2)
    vec = allocate_batch_vectorized(np.array([[[u_a, u_b]]], dtype=np.uint64), [2])[0, 0].tolist()
    assert [det.seats["A"], det.seats["B"]] == vec == [1, 1]


def test_d1_object_arbitrary_precision_ints() -> None:
    big_a, big_b = 2 * 10**400 + 1, 10**400 + 1
    det = allocate_dhondt_loreg({"A": big_a, "B": big_b}, seats=2)
    vec = allocate_batch_vectorized(np.array([[[big_a, big_b]]], dtype=object), [2])[0, 0].tolist()
    assert [det.seats["A"], det.seats["B"]] == vec == [1, 1]


def test_d1_extreme_float_overflow_and_underflow_safe() -> None:
    with np.errstate(all="raise"):
        det_hi = allocate_dhondt_loreg({"A": 1e308, "B": 1e308}, seats=3)
        vec_hi = allocate_batch_vectorized(np.array([[[1e308, 1e308]]], dtype=np.float64), [3])[
            0, 0
        ].tolist()
        assert [det_hi.seats["A"], det_hi.seats["B"]] == vec_hi == [2, 1]

        det_asym = allocate_dhondt_loreg({"A": 0.5e308, "B": 1e308}, seats=3)
        vec_asym = allocate_batch_vectorized(
            np.array([[[0.5e308, 1e308]]], dtype=np.float64), [3]
        )[0, 0].tolist()
        assert [det_asym.seats["A"], det_asym.seats["B"]] == vec_asym == [1, 2]

        det_lo = allocate_dhondt_loreg({"A": 1e-220, "B": 2e-220}, seats=3, blank_votes=0.0)
        vec_lo = allocate_batch_vectorized(np.array([[[1e-220, 2e-220]]], dtype=np.float64), [3])[
            0, 0
        ].tolist()
        assert [det_lo.seats["A"], det_lo.seats["B"]] == vec_lo == [1, 2]


def test_d2_close_quotient_ties_and_epsilons() -> None:
    assert (
        allocate_batch_vectorized(
            np.array([[[4_000_000_000_001, 8_000_000_000_000]]], dtype=np.int64), [2]
        )[0, 0].tolist()
        == [1, 1]
    )
    assert (
        allocate_batch_vectorized(np.array([[[0.20000000000005, 0.40]]]), [2])[0, 0].tolist()
        == [1, 1]
    )


@pytest.mark.parametrize("k_way", [2, 3, 4, 5])
def test_d3_kway_equal_ties_match_scalar_across_all_seats(k_way: int) -> None:
    parties = [f"P{i}" for i in range(k_way + 1)]
    pv_dict = {f"P{i}": 50_000 for i in range(k_way)}
    pv_dict[f"P{k_way}"] = 83_000
    arr_row = np.array([[[pv_dict[p] for p in parties]]], dtype=np.int64)
    for seats in (1, 2, 3, 5, 11, 24, 38):
        det = allocate_dhondt_loreg(pv_dict, seats=seats, canonical_order=parties)
        exp = [det.seats[p] for p in parties]
        got = allocate_batch_vectorized(arr_row, [seats], n_workers=1)[0, 0].tolist()
        assert got == exp


def test_d4_2d_and_3d_active_mask_isolates_nans_under_errstate_raise() -> None:
    with np.errstate(all="raise"):
        arr_2d = np.array([[[float("nan"), 0.96, 0.035, -999.0]]])
        mask_2d = np.array([[False, True, True, False]])
        out_2d = allocate_batch_vectorized(
            arr_2d, [35], active_mask=mask_2d, blank_votes_or_shares=0.0
        )
        assert out_2d[0, 0].tolist() == [0, 34, 1, 0]

        arr_3d = np.array([[[float("nan"), 0.96, 0.035, -999.0]]])
        mask_3d = np.array([[[False, True, True, False]]])
        out_3d = allocate_batch_vectorized(
            arr_3d, [35], active_mask=mask_3d, blank_votes_or_shares=0.0
        )
        assert out_3d[0, 0].tolist() == [0, 34, 1, 0]


def test_d4_packed_chunk_inplace_valid_chunk_and_dirty_buffer_reuse() -> None:
    reused_out = np.full((1, 1, 4), 99, dtype=np.int16)
    recip = 1.0 / np.arange(1, 37, dtype=np.float64)
    allocate_packed_chunk_inplace(
        np.array([[[50_000, 96_900, 3_100, 50_000]]], dtype=np.int64),
        reused_out,
        np.array([35], dtype=np.int32),
        np.array([False], dtype=bool),
        [np.array([1, 2], dtype=np.intp)],
        recip,
        valid_chunk=np.array([[100_000]], dtype=np.int64),
        is_int_mode=True,
    )
    assert reused_out[0, 0].tolist() == [0, 34, 1, 0]


def test_d5_fixed_valid_votes_docstring_and_runner_up_gap() -> None:
    assert "valid_votes" in (allocate_dhondt_loreg.__doc__ or "")
    assert "valid_votes" in (RunnerUpGap.__doc__ or "")
    res = allocate_dhondt_loreg(
        {"PP": 40_000, "PSOE": 39_500, "VOX": 15_000},
        seats=4,
        blank_votes=1_500,
    )
    assert res.last_seat_battle is not None
    assert res.last_seat_battle.runners_up
    assert res.last_seat_battle.runners_up[0].vote_gap_abs > 0


def test_d6_compensated_float_threshold_exact_boundary() -> None:
    # 0.191 + 0.009 + 0.1 == 0.30000000000000004 naive, but exact ratio 0.009 / fsum(...) == 0.03
    res_vec_pass = allocate_batch_vectorized(
        np.array([[[0.191, 0.009]]]), [35], blank_votes_or_shares=0.1
    )[0, 0].tolist()
    res_det_pass = allocate_dhondt_loreg({"A": 0.191, "B": 0.009}, seats=35, blank_votes=0.1)
    assert res_vec_pass == [res_det_pass.seats["A"], res_det_pass.seats["B"]] == [34, 1]

    eps_below = math.nextafter(0.03, 0.0)
    res_vec_fail = allocate_batch_vectorized(
        np.array([[[0.97, eps_below]]]), [35], blank_votes_or_shares=0.0
    )[0, 0].tolist()
    res_det_fail = allocate_dhondt_loreg({"A": 0.97, "B": eps_below}, seats=35, blank_votes=0.0)
    assert res_vec_fail == [res_det_fail.seats["A"], res_det_fail.seats["B"]] == [35, 0]


def test_integer_batch_vectorized_over_64_rows_parity() -> None:
    rng = np.random.Generator(np.random.PCG64(SIMULATION_SEED + 1))
    counts = rng.integers(1_000, 120_000, size=(120, 4, 6), dtype=np.int64)
    blanks = rng.integers(500, 5_000, size=(120, 4), dtype=np.int64)
    seats = [4, 7, 1, 12]
    cities = [False, False, True, False]
    vec = allocate_batch_vectorized(
        counts, seats, is_autonomous_city=cities, blank_votes_or_shares=blanks, n_workers=2
    )
    parties = [f"P{i}" for i in range(6)]
    for m in (0, 1, 63, 64, 119):
        for c in range(4):
            pv = {parties[p]: int(counts[m, c, p]) for p in range(6)}
            det = allocate_dhondt_loreg(
                pv, seats[c], blank_votes=int(blanks[m, c]), is_autonomous_city=cities[c]
            )
            assert vec[m, c].tolist() == [det.seats[p] for p in parties]


def test_zero_draws_batch_returns_empty_shape() -> None:
    seats = [r[6] for r in ROWS_52]
    out = allocate_batch_vectorized(np.zeros((0, 52, 10), dtype=np.float64), seats)
    assert out.shape == (0, 52, 10)


def test_vectorized_invalid_shapes_raise_schema_error() -> None:
    with pytest.raises(SchemaValidationError):
        allocate_batch_vectorized(np.zeros((2, 3)), [1, 1, 1])
    with pytest.raises(SchemaValidationError):
        allocate_batch_vectorized(np.zeros((2, 3, 2)), [1, 1])
    with pytest.raises(SchemaValidationError):
        allocate_batch_vectorized(np.zeros((2, 3, 2)), [0, 1, 1])


# ---------------------------------------------------------------------------
# 3. Fuzz parity & (50,000 x 52 x 20) < 1.5s performance SLA
# ---------------------------------------------------------------------------


def test_fuzz_dirichlet_scalar_vs_vectorized_parity() -> None:
    catalog = ConstituenciesCatalog.from_dict(_build_52_catalog_dict())
    seats_vec = np.array([c.seats_2026 for c in catalog.constituencies], dtype=np.int32)
    city_vec = np.array([c.is_autonomous_city for c in catalog.constituencies], dtype=bool)

    rng = np.random.Generator(np.random.PCG64(SIMULATION_SEED + 7))
    n_draws = 150  # 150 x 52 = 7,800 continuous races with explicit 2D blanks
    parties = [f"P{i:02d}" for i in range(10)]
    raw = rng.dirichlet(
        np.array([4.0, 4.0, 2.0, 1.0, 0.32, 0.31, 0.30, 0.29, 0.15, 0.10]), size=(n_draws, 52)
    )
    blanks = rng.uniform(0.005, 0.05, size=(n_draws, 52))
    shares = raw * (1.0 - blanks[..., np.newaxis])
    vec = allocate_batch_vectorized(
        shares, seats_vec, is_autonomous_city=city_vec, blank_votes_or_shares=blanks, n_workers=2
    )
    for m in range(n_draws):
        for c in range(52):
            pv = {parties[p]: float(shares[m, c, p]) for p in range(10)}
            det = allocate_dhondt_loreg(
                pv,
                int(seats_vec[c]),
                blank_votes=float(blanks[m, c]),
                is_autonomous_city=bool(city_vec[c]),
            )
            assert vec[m, c].tolist() == [det.seats[p] for p in parties]


def test_50000_draws_52_constituencies_under_1500ms() -> None:
    catalog = ConstituenciesCatalog.from_dict(_build_52_catalog_dict())
    seats_vec = np.array([c.seats_2026 for c in catalog.constituencies], dtype=np.int32)
    city_vec = np.array([c.is_autonomous_city for c in catalog.constituencies], dtype=bool)

    rng = np.random.Generator(np.random.PCG64(SIMULATION_SEED))
    mask = np.zeros((52, 20), dtype=bool)
    mask[:, :6] = True
    for c in range(52):
        mask[c, rng.choice(np.arange(6, 20), size=3, replace=False)] = True
    raw = rng.uniform(0.01, 0.35, size=(50_000, 52, 20)) * mask[np.newaxis, :, :]
    shares = raw / raw.sum(axis=2, keepdims=True) * 0.985

    # Warmup
    _ = allocate_batch_vectorized(
        shares[:64], seats_vec, is_autonomous_city=city_vec, active_mask=mask, n_workers=2
    )
    t0 = time.perf_counter()
    out = allocate_batch_vectorized(
        shares, seats_vec, is_autonomous_city=city_vec, active_mask=mask, n_workers=2
    )
    elapsed = time.perf_counter() - t0
    assert out.shape == (50_000, 52, 20)
    assert np.all(out.sum(axis=(1, 2)) == CONGRESS_TOTAL_SEATS)
    assert np.all(out[:, ~mask] == 0)
    assert elapsed < 1.5, f"50,000-draw D'Hondt allocation took {elapsed:.3f}s >= 1.5s budget"


# ---------------------------------------------------------------------------
# 4. Schemas stdlib-only isolation, validation & config guardrails
# ---------------------------------------------------------------------------


def test_schemas_is_strictly_stdlib_only() -> None:
    code = (
        "import sys, elecciones.schemas; "
        "assert 'numpy' not in sys.modules and 'scipy' not in sys.modules "
        "and 'pandas' not in sys.modules and 'httpx' not in sys.modules"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr


def test_constituencies_catalog_52_and_senate_208() -> None:
    cat = ConstituenciesCatalog.from_dict(_build_52_catalog_dict())
    assert len(cat.constituencies) == 52
    assert cat.get("28").seats_2026 == 38
    assert cat.get("11").seats_2026 == 8
    assert sum(c.seats_2026 for c in cat.constituencies) == CONGRESS_TOTAL_SEATS
    assert sum(c.senate.direct_seats for c in cat.constituencies) == SENATE_DIRECT_SEATS
    assert len(cat.by_codauto()) == 19


def test_constituencies_duplicate_coords_and_invalid_madrid_rejected() -> None:
    d = _build_52_catalog_dict()
    bad_coords = [dict(x) for x in d["constituencies"]]  # type: ignore[index]
    bad_coords[0]["cartogram_row"] = bad_coords[1]["cartogram_row"]
    bad_coords[0]["cartogram_col"] = bad_coords[1]["cartogram_col"]
    with pytest.raises(SchemaValidationError, match="Duplicate"):
        ConstituenciesCatalog.from_dict({**d, "constituencies": bad_coords})

    bad_madrid = [dict(x) for x in d["constituencies"]]  # type: ignore[index]
    mad_idx = next(i for i, x in enumerate(bad_madrid) if x["cpro"] == "28")
    cad_idx = next(i for i, x in enumerate(bad_madrid) if x["cpro"] == "11")
    bad_madrid[mad_idx]["seats_2026"] = 37
    bad_madrid[cad_idx]["seats_2026"] = 9
    with pytest.raises(SchemaValidationError, match="38"):
        ConstituenciesCatalog.from_dict({**d, "constituencies": bad_madrid})


def test_parties_catalog_and_boe_mask_validation() -> None:
    mask_all = {c: 1 for c in ALL_CPROS}
    mask_none = {c: 0 for c in ALL_CPROS}
    p_pp = PartySpec("PP", "Partido Popular", "PP", "#0055A7", "PP", "right", mask_all)
    p_ac = PartySpec("AC", "Aliança Catalana", "AC", "#1B365D", "OTROS", "other", mask_none)
    cat = PartiesCatalog(parties=(p_pp, p_ac))
    assert cat.get("PP").contests("28") is True
    assert cat.get("AC").contests("08") is False
    assert len(cat.mask_matrix()) == 52

    with pytest.raises(SchemaValidationError, match="0 proclaimed"):
        PartiesCatalog(parties=(p_ac,))


def test_poll_csv_scale_and_missing_cell_preservation() -> None:
    rec = PollRecord.from_csv_row(
        {
            "election_id": "2026-11",
            "pollster": "40dB.",
            "media": "El País",
            "fieldwork_start": "2026-11-10",
            "fieldwork_end": "2026-11-12",
            "pub_date": "2026-11-14",
            "sample_size": "2000",
            "turnout": "69.5",
            "is_cis": "false",
            "PP": "30.8",
            "PSOE": "27.3",
            "VOX": "18.9",
            "BNG": "?",
            "ERC": "",
        }
    )
    assert abs(rec.party_shares["PP"] - 0.308) < 1e-12
    assert "BNG" not in rec.party_shares and "ERC" not in rec.party_shares
    assert rec.fieldwork_days == 3

    overlay = ManualPollOverlay.from_csv_row(
        {
            "action": "exclude",
            "election_id": "2026-11",
            "pollster": "BadPoll",
            "fieldwork_start": "2026-11-01",
            "fieldwork_end": "2026-11-02",
        }
    )
    assert overlay.poll.party_shares == {}

    reg = RegionalPollRecord.from_csv_row(
        {
            "scope_type": "province",
            "scope_code": "42",
            "pollster": "Sigma Dos",
            "fieldwork_start": "2026-11-05",
            "fieldwork_end": "2026-11-07",
            "sample_size": "600",
            "PP": "34.0",
            "SORIA_YA": "21.0",
        }
    )
    assert reg.scope_code == "42"
    assert abs(reg.party_shares["SORIA_YA"] - 0.21) < 1e-12


def test_backtest_summary_strict_int_validation() -> None:
    dummy_m = ModelBacktestMetrics(0.1, -0.2, 3.0, 0.2, 1.1, 0.8, 0.95, 0.8, 0.95)
    with pytest.raises(SchemaValidationError):
        BacktestSummary(
            elections=("2015-12", "2023-07"),
            total_constituency_races=260.0,  # type: ignore[arg-type]
            total_seats=1750,
            horizons={},
            models={"kernel": dummy_m},
            tier_calibration={"seguro_99": TierCalibrationBucket(10, 10, 1.0)},
            calibrated_parameters={},
            per_election={},
        )
    with pytest.raises(SchemaValidationError):
        TierCalibrationBucket(total=10.0, hits=9, hit_rate=0.9)  # type: ignore[arg-type]


def test_forecast_bundle_json_roundtrip(tmp_path: Path) -> None:
    cf_list: list[ConstituencyForecast] = []
    for cpro, name, _, codauto, _, _, s26, *_ in ROWS_52:
        cf_list.append(
            ConstituencyForecast(
                cpro=cpro,
                name=name,
                codauto=codauto,
                seats_2026=s26,
                turnout_mean=0.69,
                blank_share=0.01,
                pre_shrinkage_ipf_shares={"PP": 0.50, "PSOE": 0.49},
                final_shares={"PP": 0.50, "PSOE": 0.49},
                party_projections={
                    "PP": PartyConstituencyProjection(
                        0.50, 0.45, 0.55, float(s26), s26, s26, s26, s26, s26, s26, 0, 0, 1.0
                    )
                },
                seat_slots=(SeatSlot(party="PP", rank=1, prob=0.995, tier="seguro"),),
                dhondt_quotients={"PP": [0.50]},
                last_seat_battle=LastSeatBattle(
                    holder="PP",
                    cutoff_quotient=0.10,
                    runners_up=(RunnerUpGap("PSOE", 1.2, 2400),),
                ),
            )
        )
    bundle = ForecastBundle(
        metadata=ForecastMetadata(
            generated_at="2026-11-15T10:00:00+01:00",
            as_of_date="2026-11-15",
            election_date="2026-11-29",
            n_sims=50_000,
            seed=SIMULATION_SEED,
            engine="ensemble",
            exclude_cis_headline=False,
            blackout_active=False,
            simulation_runtime_ms=850.0,
        ),
        national={
            "vote_shares": {
                "PP": VoteShareInterval(0.31, 0.31, 0.28, 0.29, 0.33, 0.34).to_dict()
            },
            "seats": {
                "PP": PartyNationalSeats(135.0, 135, 118, 122, 147, 150, 80, 45, 15).to_dict()
            },
            "stage1_comparison": {"kernel": {}, "state_space": {}, "ensemble": {}},
            "cis_sensitivity": {"with_cis": {}, "without_cis": {}},
        },
        tier_totals={"seguros": 153, "probables": 150, "en_juego": 97},
        trend_series={"daily": [], "polls": []},
        constituencies=tuple(cf_list),
        battlegrounds=[],
        investiture=InvestitureSummary(
            first_ballot={
                "pp_vox": PactProbability("PP + Vox", ("PP", "VOX"), 0.91, 201, 185, 216)
            },
            second_ballot={
                "right_2nd": SecondBallotScenario(
                    "PP+Vox 2ª", ("PP", "VOX"), ("PSOE",), ("CCA",), 0.93, 28
                )
            },
            deadlock_prob=0.06,
        ),
    )
    out_file = tmp_path / "forecast_29n.json"
    bundle.save_json(out_file)
    reloaded = ForecastBundle.load_json(out_file)
    assert len(reloaded.constituencies) == 52
    assert reloaded.metadata.n_sims == 50_000


def test_allocation_input_batch_and_election_result_contracts() -> None:
    batch = parse_allocation_payload(
        {"constituencies": {"51": {"seats": 1, "party_votes": {"PP": 12000, "PSOE": 10000}}}}
    )
    assert isinstance(batch, list) and batch[0].is_autonomous_city is True

    r1 = AllocationResult(
        valid_votes=22000,
        eligible_parties=("PP", "PSOE"),
        seats={"PP": 1, "PSOE": 0},
        cutoff_quotient=12000.0,
        cpro="51",
    )
    b_res = BatchAllocationResult.from_results([r1])
    assert BatchAllocationResult.from_dict(b_res.to_dict()).national_seats["PP"] == 1

    er = ConstituencyElectionResult(
        election_id="2023-07",
        cpro="51",
        constituency="Ceuta",
        seats=1,
        censo=63310,
        votantes=33800,
        nulos=479,
        validos=33321,
        blancos=328,
        party_votes={"PP": 12993, "PSOE": 11300, "VOX": 8700},
        party_seats={"PP": 1, "PSOE": 0, "VOX": 0},
    )
    assert er.to_allocation_input().is_autonomous_city is True
    assert AllocationInput.from_dict({"seats": 1, "party_votes": {"PP": 10}}).seats == 1


def test_blackout_window_boundaries_europe_madrid() -> None:
    assert is_in_blackout_window("2026-11-23T23:59:59+01:00") is False
    assert is_in_blackout_window("2026-11-24T00:00:00+01:00") is True
    assert is_in_blackout_window("2026-11-29T19:59:59+01:00") is True
    assert is_in_blackout_window("2026-11-29T20:00:00+01:00") is False


# ---------------------------------------------------------------------------
# 5. Live CLI allocate & --help / blackout checks
# ---------------------------------------------------------------------------


def test_cli_allocate_single_district_json_and_table(tmp_path: Path) -> None:
    payload = {
        "cpro": "33",
        "seats": 7,
        "blank_votes": 5620,
        "null_votes": 7500,
        "party_votes": {"PP": 212816, "PSOE": 205049, "SUMAR": 88630, "VOX": 74571},
    }
    in_file = tmp_path / "asturias.json"
    in_file.write_text(json.dumps(payload), encoding="utf-8")

    res_json = RUNNER.invoke(app, ["allocate", "--input", str(in_file), "--json"])
    assert res_json.exit_code == 0, res_json.output
    data = json.loads(res_json.stdout)
    assert data["seats"] == {"PP": 3, "PSOE": 2, "SUMAR": 1, "VOX": 1}
    assert "quotient_table" in data

    res_tab = RUNNER.invoke(app, ["allocate", "--input", str(in_file)])
    assert res_tab.exit_code == 0
    assert "Valid votes:" in res_tab.stdout


def test_cli_allocate_batch_stdin_and_strict_failure() -> None:
    batch_payload = {
        "constituencies": [
            {"cpro": "51", "seats": 1, "party_votes": {"PP": 12000, "PSOE": 10000}},
            {
                "cpro": "42",
                "seats": 2,
                "blank_votes": 100,
                "party_votes": {"PP": 20000, "PSOE": 15000},
            },
        ]
    }
    res = RUNNER.invoke(app, ["allocate", "--json"], input=json.dumps(batch_payload))
    assert res.exit_code == 0, res.output
    data = json.loads(res.stdout)
    assert data["total_seats"] == 3
    assert data["national_seats"]["PP"] == 2

    exhaust_payload = {"seats": 4, "blank_votes": 980, "party_votes": {"A": 10, "B": 10}}
    res_strict = RUNNER.invoke(app, ["allocate", "--strict"], input=json.dumps(exhaust_payload))
    assert res_strict.exit_code == 1


def test_cli_allocate_invalid_input_exits_nonzero(tmp_path: Path) -> None:
    res_missing = RUNNER.invoke(app, ["allocate", "--input", str(tmp_path / "nope.json")])
    assert res_missing.exit_code == 1
    res_bad_json = RUNNER.invoke(app, ["allocate"], input="{not json")
    assert res_bad_json.exit_code == 1


def test_cli_all_eight_helps_have_zero_ellipsis_without_columns() -> None:
    clean_env = {k: v for k, v in os.environ.items() if k not in ("COLUMNS", "TERMINAL_WIDTH")}
    subs = (
        [],
        ["allocate"],
        ["fetch"],
        ["ratings"],
        ["backtest"],
        ["forecast"],
        ["dashboard"],
        ["run-all"],
    )
    for sub in subs:
        proc = subprocess.run(
            [sys.executable, "-m", "elecciones.cli", *sub, "--help"],
            capture_output=True,
            text=True,
            env=clean_env,
            check=False,
        )
        assert proc.returncode == 0, (sub, proc.stderr)
        assert "…" not in proc.stdout, (sub, proc.stdout)


def test_cli_blackout_guard_skips_fetch_and_run_all() -> None:
    res_fetch = RUNNER.invoke(
        app, ["fetch", "--respect-blackout", "--now", "2026-11-26T12:00:00+01:00"]
    )
    assert res_fetch.exit_code == 0
    assert "LOREG Art. 69.7" in res_fetch.stdout

    res_run = RUNNER.invoke(
        app, ["run-all", "--respect-blackout", "--now", "2026-11-26T12:00:00+01:00"]
    )
    assert res_run.exit_code == 0
    assert "LOREG Art. 69.7" in res_run.stdout

    res_override = RUNNER.invoke(
        app, ["fetch", "--no-respect-blackout", "--now", "2026-11-26T12:00:00+01:00"]
    )
    assert res_override.exit_code == 1
