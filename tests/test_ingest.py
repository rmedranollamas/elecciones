"""Tests for Milestone 2 ingestion, 1,750-seat verification, CIS #3584 IPF, and overlays."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

from typer.testing import CliRunner

from elecciones.cli import app
from elecciones.config import (
    BLACKOUT_CUTOFF_DATE_2026,
    HISTORICAL_ELECTION_DATES,
    get_elections_dir,
    get_metadata_dir,
    get_polls_dir,
    get_raw_cache_dir,
)
from elecciones.ingest import (
    apply_manual_poll_overlay,
    ensure_manual_polls_template,
    load_cis_3584_provincial,
    load_coalition_transfers,
    load_congress_results,
    load_manual_overlays,
    load_national_polls,
    load_regional_polls,
)
from elecciones.ingest.cross_check import (
    ensure_brujula_js,
    parse_brujula_polls,
    reconcile_with_brujula,
)
from elecciones.models.dhondt import allocate_dhondt_loreg
from elecciones.schemas import (
    ManualPollOverlay,
    PollRecord,
    load_constituencies_catalog,
    load_parties_catalog,
)


def test_historical_congress_1750_seats_exact_reproduction() -> None:
    elections_dir = get_elections_dir()
    total_seats_checked = 0
    for eid in HISTORICAL_ELECTION_DATES:
        rows = load_congress_results(eid, elections_dir)
        assert len(rows) == 52
        election_seats = 0
        for rec in rows:
            res = allocate_dhondt_loreg(rec.to_allocation_input())
            for party, expected_seats in rec.party_seats.items():
                assert res.seats.get(party, 0) == expected_seats, (
                    f"{eid} cpro={rec.cpro} party={party}: "
                    f"got {res.seats.get(party, 0)} != expected {expected_seats}"
                )
            assert sum(res.seats.values()) == rec.seats
            election_seats += rec.seats
        assert election_seats == 350
        total_seats_checked += election_seats
    assert total_seats_checked == 1750


def test_europarl_2024_errata_and_regional_2024_2026_snapshots() -> None:
    ep_csv = get_elections_dir() / "europarl_2024_06.csv"
    assert ep_csv.is_file()
    with ep_csv.open("r", encoding="utf-8", newline="") as fh:
        ep_rows = list(csv.DictReader(fh))
    assert len(ep_rows) == 52
    assert sum(int(r["votes_PSOE"]) for r in ep_rows) == 5_291_102
    assert sum(int(r["votes_JUNTS"]) for r in ep_rows) == 442_140
    assert sum(int(r["votes_SALF"]) for r in ep_rows) > 750_000

    reg_csv = get_elections_dir() / "regional_2024_2026.csv"
    assert reg_csv.is_file()
    with reg_csv.open("r", encoding="utf-8", newline="") as fh:
        reg_rows = list(csv.DictReader(fh))
    assert len(reg_rows) >= 35
    ccaa_codes = {r["scope_code"] for r in reg_rows if r["scope_type"] == "ccaa"}
    assert {"01", "02", "07", "09", "11", "12", "16"}.issubset(ccaa_codes)


def test_metadata_constituencies_and_boe_candidacy_mask() -> None:
    consts = load_constituencies_catalog(get_metadata_dir() / "constituencies_2026.json")
    assert consts.get("28").seats_2026 == 38
    assert consts.get("11").seats_2026 == 8
    assert sum(c.seats_2026 for c in consts.constituencies) == 350

    parties = load_parties_catalog(get_metadata_dir() / "parties_2026.json")
    pp = parties.get("PP")
    upn_pp = parties.get("UPN_PP")
    assert pp.boe_mask_2026["31"] == 0
    assert upn_pp.boe_mask_2026["31"] == 1
    assert sum(upn_pp.boe_mask_2026.values()) == 1

    fv = parties.get("COMPROMIS_FV")
    assert {c for c, v in fv.boe_mask_2026.items() if v == 1} == {"03", "12", "46"}

    fa = parties.get("FA")
    for c in ("03", "12", "46", "35", "38", "07", "44", "51", "52"):
        assert fa.boe_mask_2026[c] == 0

    for zero_party in ("AC", "CUP", "UPL"):
        assert sum(parties.get(zero_party).boe_mask_2026.values()) == 0


def test_wikipedia_poll_csv_snapshots_validated() -> None:
    hist = load_national_polls(get_polls_dir() / "national_2015_2023.csv")
    assert len(hist) >= 1800
    assert set(HISTORICAL_ELECTION_DATES).issubset({p.election_id for p in hist})
    for p in hist:
        assert p.fieldwork_end < HISTORICAL_ELECTION_DATES[p.election_id] or (
            (p.sample_size or 0) <= 50_000
        )

    polls_2026 = load_national_polls(get_polls_dir() / "national_2026.csv")
    assert len(polls_2026) >= 500
    assert max(p.fieldwork_end for p in polls_2026) <= BLACKOUT_CUTOFF_DATE_2026
    assert any(p.is_cis for p in polls_2026)
    assert all(not p.pollster.startswith("CIS (") for p in polls_2026)
    assert any("FA_FV" in p.party_shares for p in polls_2026)
    assert any("FA" in p.party_shares and "COMPROMIS_FV" in p.party_shares for p in polls_2026)

    reg = load_regional_polls(get_polls_dir() / "regional_2026.csv")
    assert len(reg) >= 60
    assert all(len(r.party_shares) >= 2 for r in reg)
    assert max(r.fieldwork_end for r in reg) <= BLACKOUT_CUTOFF_DATE_2026


def test_cis_3584_provincial_mask_ipf_and_coalition_transfers() -> None:
    cis_data = load_cis_3584_provincial(get_polls_dir() / "cis_3584_provincial.csv")
    parties = load_parties_catalog(get_metadata_dir() / "parties_2026.json")
    assert len(cis_data) == 52
    assert sum(row["n_realized"] for row in cis_data.values()) == 45_074

    for cpro, entry in cis_data.items():
        assert abs(sum(entry["raw_shares"].values()) - 1.0) < 1e-6
        assert abs(sum(entry["debias_shares"].values()) - 1.0) < 1e-6
        for pid in parties.party_ids():
            if parties.get(pid).boe_mask_2026[cpro] == 0:
                assert entry["raw_shares"][pid] == 0.0
                assert entry["debias_shares"][pid] == 0.0

    # De-biased Madrid PP exceeds raw CIS P5aR PP share
    assert cis_data["28"]["debias_shares"]["PP"] > cis_data["28"]["raw_shares"]["PP"]
    assert cis_data["31"]["debias_shares"]["UPN_PP"] > 0.20

    transfers = load_coalition_transfers(get_metadata_dir() / "coalition_transfers.json")
    assert "sumar_2023_split" in transfers
    assert "valencian_community" in transfers["sumar_2023_split"]
    assert "navarra_upn_pp_merger" in transfers
    assert transfers["unproclaimed_lists_2026"]["AC"]["proclaimed_districts"] == 0


def test_brujula_cross_check_and_manual_poll_overlay(tmp_path: Path) -> None:
    wiki_polls = load_national_polls(get_polls_dir() / "national_2026.csv")
    js_path = ensure_brujula_js(offline=True, raw_cache_dir=get_raw_cache_dir())
    brujula = parse_brujula_polls(js_path)
    report = reconcile_with_brujula(wiki_polls, brujula)
    assert report["matched_polls"] >= 65
    assert report["mean_abs_diff_pp"]["PP"] < 0.25
    assert report["mean_abs_diff_pp"]["PSOE"] < 0.25

    base_a = PollRecord(
        election_id="2026-11",
        pollster="40dB",
        media="Prisa",
        fieldwork_start=date(2026, 11, 10),
        fieldwork_end=date(2026, 11, 14),
        sample_size=2000,
        party_shares={"PP": 0.314, "PSOE": 0.299, "VOX": 0.183},
    )
    base_b = PollRecord(
        election_id="2026-11",
        pollster="GAD3",
        media="ABC",
        fieldwork_start=date(2026, 11, 11),
        fieldwork_end=date(2026, 11, 13),
        sample_size=1200,
        party_shares={"PP": 0.330, "PSOE": 0.270, "VOX": 0.180},
    )

    csv_path = tmp_path / "manual_polls_2026.csv"
    ensure_manual_polls_template(csv_path)
    assert load_manual_overlays(csv_path) == []

    overlays = [
        ManualPollOverlay(
            action="override",
            poll=PollRecord(
                election_id="2026-11",
                pollster="40dB",
                media="Prisa",
                fieldwork_start=date(2026, 11, 10),
                fieldwork_end=date(2026, 11, 14),
                sample_size=2000,
                party_shares={"PP": 0.320, "PSOE": 0.290, "VOX": 0.180},
            ),
        ),
        ManualPollOverlay(
            action="exclude",
            poll=PollRecord(
                election_id="2026-11",
                pollster="GAD3",
                fieldwork_start=date(2026, 11, 11),
                fieldwork_end=date(2026, 11, 13),
                party_shares={},
                allow_empty_shares=True,
            ),
        ),
        ManualPollOverlay(
            action="add",
            poll=PollRecord(
                election_id="2026-11",
                pollster="IMOP",
                media="El Confidencial",
                fieldwork_start=date(2026, 11, 18),
                fieldwork_end=date(2026, 11, 20),
                sample_size=1500,
                party_shares={"PP": 0.318, "PSOE": 0.272, "VOX": 0.188},
            ),
        ),
    ]
    merged = apply_manual_poll_overlay([base_a, base_b], overlays)
    assert len(merged) == 2
    assert merged[0].pollster == "40dB"
    assert abs(merged[0].party_shares["PP"] - 0.320) < 1e-12
    assert merged[1].pollster == "IMOP"


def test_cli_fetch_offline_and_blackout_guard() -> None:
    runner = CliRunner()
    res_blackout = runner.invoke(
        app,
        ["fetch", "--offline", "--now", "2026-11-25T12:00:00+01:00"],
    )
    assert res_blackout.exit_code == 0
    assert "blackout" in res_blackout.stdout.lower()

    res_offline = runner.invoke(
        app,
        ["fetch", "--offline", "--no-respect-blackout"],
    )
    assert res_offline.exit_code == 0
    assert "29-N Data Ingestion Summary" in res_offline.stdout
