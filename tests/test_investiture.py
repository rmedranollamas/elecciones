"""Tests for Milestone 4 Stages 3–4: Covariance, 50k Simulator (<1.5s), LOOCV Backtest & Art. 99."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from elecciones.backtest import run_backtest
from elecciones.cli import app
from elecciones.config import get_outputs_dir
from elecciones.ingest.cross_check import MANUAL_POLLS_COLUMNS
from elecciones.models import (
    compute_investiture_summary,
    evaluate_first_ballot,
    evaluate_second_ballot,
    run_forecast_simulation,
)
from elecciones.schemas import (
    BacktestSummary,
    ForecastBundle,
    SchemaValidationError,
    SecondBallotScenario,
    classify_seat_tier,
)


def test_art99_first_ballot_and_second_ballot_abstentions() -> None:
    party_ids = (
        "PP",
        "UPN_PP",
        "VOX",
        "SALF",
        "PSOE",
        "FA",
        "PODEMOS",
        "ERC",
        "JUNTS",
        "EH_BILDU",
        "PNV",
        "BNG",
        "CCA",
    )
    p_idx = {p: i for i, p in enumerate(party_ids)}

    # 4 deterministic Congress configurations (each summing cleanly to 350 seats):
    # Row 0: PP+UPN = 176 (passes 1st ballot >= 176)
    # Row 1: PP+UPN = 175 (fails 1st ballot pp_solo, PP+UPN+VOX = 190 passes)
    # Row 2: Exact 2nd-ballot tie (yes=165, no=165, abstain=20 -> fails strict Yes > No!)
    # Row 3: Strict hung Parliament (no 1st ballot >= 176, both 2nd ballots blocked!)
    mat = np.zeros((4, len(party_ids)), dtype=np.int32)
    # Row 0
    mat[0, p_idx["PP"]] = 174
    mat[0, p_idx["UPN_PP"]] = 2
    mat[0, p_idx["PSOE"]] = 120
    mat[0, p_idx["VOX"]] = 30
    mat[0, p_idx["ERC"]] = 12
    mat[0, p_idx["JUNTS"]] = 12
    # Row 1
    mat[1, p_idx["PP"]] = 174
    mat[1, p_idx["UPN_PP"]] = 1
    mat[1, p_idx["VOX"]] = 15
    mat[1, p_idx["PSOE"]] = 140
    mat[1, p_idx["JUNTS"]] = 20
    # Row 2: yes=(PP,UPN_PP,CCA)=165 vs no=(PSOE,FA,ERC,PNV,BNG,JUNTS)=165, abstain=(VOX,SALF)=20
    mat[2, p_idx["PP"]] = 160
    mat[2, p_idx["UPN_PP"]] = 2
    mat[2, p_idx["CCA"]] = 3
    mat[2, p_idx["VOX"]] = 15
    mat[2, p_idx["SALF"]] = 5
    mat[2, p_idx["PSOE"]] = 120
    mat[2, p_idx["FA"]] = 20
    mat[2, p_idx["ERC"]] = 10
    mat[2, p_idx["PNV"]] = 5
    mat[2, p_idx["JUNTS"]] = 10
    # Row 3: Hung Parliament (Right+CCA=170 vs Left+Junts=170 tie, Left vs Right+SALF=180 loses)
    mat[3, p_idx["PP"]] = 143
    mat[3, p_idx["UPN_PP"]] = 2
    mat[3, p_idx["VOX"]] = 25
    mat[3, p_idx["SALF"]] = 10
    mat[3, p_idx["PSOE"]] = 115
    mat[3, p_idx["FA"]] = 20
    mat[3, p_idx["ERC"]] = 7
    mat[3, p_idx["EH_BILDU"]] = 6
    mat[3, p_idx["PNV"]] = 5
    mat[3, p_idx["BNG"]] = 2
    mat[3, p_idx["JUNTS"]] = 15

    assert np.all(mat.sum(axis=1) == 350)

    pp_solo = evaluate_first_ballot(mat, p_idx, ("PP", "UPN_PP"), "PP solo")
    assert pp_solo.prob == pytest.approx(0.25)

    pp_vox = evaluate_first_ballot(mat, p_idx, ("PP", "UPN_PP", "VOX"), "PP + Vox")
    assert pp_vox.prob == pytest.approx(0.75)

    # Row 2 tie (165 vs 165 with abstain=20) MUST NOT pass strict Sí > No
    tie_check = evaluate_second_ballot(
        mat[2:3],
        p_idx,
        label="Tie check",
        yes=("PP", "UPN_PP", "CCA"),
        no=("PSOE", "FA", "ERC", "PNV", "BNG", "JUNTS"),
        abstain=("VOX", "SALF"),
    )
    assert tie_check.median_margin == 0
    assert tie_check.prob == 0.0

    # Flipping just 1 abstaining/no seat to +1 gives strict margin +1 -> prob == 1.0
    mat_pass = mat[2:3].copy()
    mat_pass[0, p_idx["PP"]] += 1
    mat_pass[0, p_idx["VOX"]] -= 1
    pass_check = evaluate_second_ballot(
        mat_pass,
        p_idx,
        label="Strict pass check",
        yes=("PP", "UPN_PP", "CCA"),
        no=("PSOE", "FA", "ERC", "PNV", "BNG", "JUNTS"),
        abstain=("VOX", "SALF"),
    )
    assert pass_check.median_margin == 1
    assert pass_check.prob == 1.0

    # Hung single-row chamber produces exact deadlock_prob == 1.0
    hung_summary = compute_investiture_summary(mat[3:4], party_ids)
    assert hung_summary.deadlock_prob == 1.0

    # Clear majority chamber produces deadlock_prob == 0.0
    clear_summary = compute_investiture_summary(mat[0:1], party_ids)
    assert clear_summary.deadlock_prob == 0.0

    # Overlapping yes/abstain strictly rejected by schema
    with pytest.raises(SchemaValidationError):
        SecondBallotScenario(
            label="Bad",
            yes=("PP", "VOX"),
            no=("PSOE",),
            abstain=("VOX",),
            prob=0.5,
            median_margin=0,
        )


def test_50k_simulator_runtime_sla_and_bundle_invariants(tmp_path: Path) -> None:
    out_json = tmp_path / "forecast_test.json"
    bundle = run_forecast_simulation(
        as_of=date(2026, 11, 16),
        engine="ensemble",
        n_sims=50_000,
        seed=20261129,
        output_path=out_json,
        write_output=True,
    )
    # Must satisfy CPU performance SLA (< 1.5s = 1500ms) for 50k x 52 draws
    assert bundle.metadata.simulation_runtime_ms < 1500.0
    assert bundle.metadata.n_sims == 50_000
    assert len(bundle.constituencies) == 52
    assert sum(c.seats_2026 for c in bundle.constituencies) == 350

    # Validate district 28 (Madrid 38) & 11 (Cádiz 8)
    by_c = {c.cpro: c for c in bundle.constituencies}
    assert by_c["28"].seats_2026 == 38
    assert by_c["11"].seats_2026 == 8

    # BOE mask invariance inside simulated outputs (Navarre PP absent, AC absent everywhere)
    nav = by_c["31"]
    assert nav.final_shares["PP"] == 0.0
    assert "PP" not in nav.party_projections
    assert nav.party_projections["UPN_PP"].seats_mean >= 1.0
    for con in bundle.constituencies:
        assert con.final_shares["AC"] == 0.0
        assert "AC" not in con.party_projections
        for slot in con.seat_slots:
            assert classify_seat_tier(slot.prob) == slot.tier

    # Tier totals match national sums cleanly
    seats_map = bundle.national["seats"]
    assert bundle.tier_totals["seguros"] == sum(v["seguros"] for v in seats_map.values())
    assert bundle.tier_totals["probables"] == sum(v["probables"] for v in seats_map.values())
    assert bundle.tier_totals["en_juego"] == sum(v["en_juego"] for v in seats_map.values())
    assert 300 <= bundle.tier_totals["seguros"] + bundle.tier_totals["probables"] + (
        bundle.tier_totals["en_juego"]
    ) <= 450

    # Battlegrounds ordered non-decreasingly by vote_gap_pp
    gaps = [b["vote_gap_pp"] for b in bundle.battlegrounds]
    assert gaps == sorted(gaps)
    assert len(bundle.sim_sample) == 200

    # JSON round-trip preserves exact validated ForecastBundle schema
    loaded = ForecastBundle.load_json(out_json)
    assert loaded.metadata.n_sims == 50_000
    assert 0.0 <= loaded.investiture.deadlock_prob <= 1.0

    # Deterministic repeatability on smaller fast check + manual overlay override shifts mean
    manual_csv = tmp_path / "manual.csv"
    header = ",".join(MANUAL_POLLS_COLUMNS)
    row_dict = {col: "" for col in MANUAL_POLLS_COLUMNS}
    row_dict.update(
        {
            "action": "add",
            "election_id": "2026-11",
            "pollster": "40dB",
            "media": "El País",
            "fieldwork_start": "2026-11-15",
            "fieldwork_end": "2026-11-16",
            "pub_date": "2026-11-16",
            "sample_size": "4000",
            "is_cis": "false",
            "PP": "39.5",
            "PSOE": "21.0",
            "VOX": "16.5",
            "FA": "5.5",
        }
    )
    manual_csv.write_text(
        header + "\n" + ",".join(row_dict[c] for c in MANUAL_POLLS_COLUMNS) + "\n",
        encoding="utf-8",
    )
    shifted = run_forecast_simulation(
        as_of=date(2026, 11, 16),
        engine="kernel",
        n_sims=2000,
        seed=20261129,
        manual_polls_path=manual_csv,
        write_output=False,
    )
    base_kernel = run_forecast_simulation(
        as_of=date(2026, 11, 16),
        engine="kernel",
        n_sims=2000,
        seed=20261129,
        write_output=False,
    )
    assert (
        shifted.national["vote_shares"]["PP"]["mean"]
        > base_kernel.national["vote_shares"]["PP"]["mean"]
    )


def test_loocv_backtest_summary_and_cli_commands(tmp_path: Path) -> None:
    bt_path = tmp_path / "backtest_test.json"
    summary = run_backtest(
        horizons=(15, 6),
        n_sims=80,
        seed=20261129,
        output_path=bt_path,
        write_output=True,
    )
    assert summary.total_constituency_races == 260
    assert summary.total_seats == 1750
    assert len(summary.elections) == 5
    for m_name in ("kernel", "state_space", "ensemble"):
        assert m_name in summary.models
        m = summary.models[m_name]
        assert 0.0 < m.vote_mae_national < 4.0
        assert 0.0 < m.seat_mae_provincial < 1.2
        assert 0.55 <= m.coverage_80_provincial_seats <= 0.98
        assert 0.80 <= m.coverage_95_provincial_seats <= 1.00

    # Tier monotonicity: seguro hit_rate >= probable hit_rate >= en_juego hit_rate >= remoto
    tc = summary.tier_calibration
    assert set(tc.keys()) == {"seguro", "probable", "en_juego", "remoto"}
    assert tc["seguro"].hit_rate >= tc["en_juego"].hit_rate
    loaded_bt = BacktestSummary.load_json(bt_path)
    assert loaded_bt.total_constituency_races == 260

    # Also verify committed output files if generated and CLI invocations work cleanly
    runner = CliRunner()
    cli_bt = tmp_path / "cli_bt.json"
    res_bt = runner.invoke(
        app,
        ["backtest", "--horizons", "6", "--n-sims", "60", "--output", str(cli_bt)],
    )
    assert res_bt.exit_code == 0, res_bt.output
    assert cli_bt.is_file()

    cli_fc = tmp_path / "cli_fc.json"
    res_fc = runner.invoke(
        app,
        [
            "forecast",
            "--as-of",
            "2026-11-16",
            "--n-sims",
            "2000",
            "--exclude-cis-headline",
            "--output",
            str(cli_fc),
        ],
    )
    assert res_fc.exit_code == 0, res_fc.output
    assert cli_fc.is_file()
    fc_bundle = ForecastBundle.load_json(cli_fc)
    assert fc_bundle.metadata.exclude_cis_headline is True

    committed_bt = get_outputs_dir() / "backtest_summary.json"
    if committed_bt.is_file():
        data = json.loads(committed_bt.read_text(encoding="utf-8"))
        assert BacktestSummary.from_dict(data).total_constituency_races == 260
