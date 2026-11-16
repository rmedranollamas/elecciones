"""Tests for Milestone 3 Stage 1 poll aggregators, ratings, and ensemble."""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from elecciones.cli import app
from elecciones.config import BLACKOUT_CUTOFF_DATE_2026, get_polls_dir
from elecciones.ingest.wikipedia import load_national_polls
from elecciones.models import (
    aggregate_kernel,
    compute_pollster_ratings,
    run_stage1,
    smooth_state_space,
)
from elecciones.models.kernel_aggregator import (
    compute_wave_overlap_penalties,
    estimate_house_effects,
    extract_poll_party_matrix,
)
from elecciones.schemas import PollRecord


def _make_poll(
    pollster: str,
    start: date,
    end: date,
    shares: dict[str, float],
    *,
    sample_size: int = 1200,
    is_cis: bool = False,
) -> PollRecord:
    return PollRecord(
        election_id="2026-11",
        pollster=pollster,
        media="Test",
        fieldwork_start=start,
        fieldwork_end=end,
        sample_size=sample_size,
        is_cis=is_cis,
        party_shares=shares,
    )


def test_wave_overlap_penalty_bounds_rolling_trackers() -> None:
    base = {"PP": 0.31, "PSOE": 0.27, "VOX": 0.18}
    # 5 duplicate identical waves from same pollster -> exact total weight 1.0
    dups = [
        _make_poll("TrackerA", date(2026, 11, 1), date(2026, 11, 5), base) for _ in range(5)
    ]
    pen_dups = compute_wave_overlap_penalties(dups)
    assert np.allclose(pen_dups, 0.2, atol=1e-12)
    assert abs(float(pen_dups.sum()) - 1.0) < 1e-12

    # 10 consecutive overlapping daily tracking waves (5-day window shifted 1 day each)
    rolling = [
        _make_poll(
            "TrackerA",
            date(2026, 11, 1 + i),
            date(2026, 11, 5 + i),
            base,
        )
        for i in range(10)
    ]
    # Plus 2 completely separated independent waves from another house
    indep = [
        _make_poll("HouseB", date(2026, 10, 1), date(2026, 10, 4), base),
        _make_poll("HouseB", date(2026, 11, 1), date(2026, 11, 4), base),
    ]
    pen = compute_wave_overlap_penalties(rolling + indep)
    rolling_sum = float(pen[:10].sum())
    assert rolling_sum < 3.6
    # Non-overlapping HouseB waves suffer zero overlap penalty
    assert np.allclose(pen[10:], 1.0, atol=1e-12)


def test_house_effects_jackman_zero_sum_centering() -> None:
    ref_truth = {"PP": 0.31, "PSOE": 0.27, "VOX": 0.19, "FA": 0.06}
    polls: list[PollRecord] = []
    # Three A/B reference pollsters around truth (zero aggregate net shift)
    offsets = {"House1": 0.01, "House2": -0.01, "House3": 0.00}
    for wk in range(3):
        d1 = date(2026, 11, 1 + 6 * wk)
        d2 = date(2026, 11, 4 + 6 * wk)
        for house, off in offsets.items():
            polls.append(
                _make_poll(
                    house,
                    d1,
                    d2,
                    {
                        "PP": ref_truth["PP"] + off,
                        "PSOE": ref_truth["PSOE"] - off,
                        "VOX": ref_truth["VOX"],
                        "FA": ref_truth["FA"],
                    },
                )
            )
    # Add CIS with systematic +5.5pp PSOE and -5.5pp PP bias
    for wk in range(3):
        d1 = date(2026, 11, 2 + 6 * wk)
        d2 = date(2026, 11, 5 + 6 * wk)
        polls.append(
            _make_poll(
                "CIS",
                d1,
                d2,
                {
                    "PP": ref_truth["PP"] - 0.055,
                    "PSOE": ref_truth["PSOE"] + 0.055,
                    "VOX": ref_truth["VOX"],
                    "FA": ref_truth["FA"],
                },
                sample_size=4000,
                is_cis=True,
            )
        )

    parties = ("PP", "PSOE", "VOX", "FA")
    y_mat, obs_mask = extract_poll_party_matrix(polls, parties, date(2026, 11, 20))
    weights = np.ones(len(polls), dtype=np.float64)
    he, ref_set = estimate_house_effects(
        polls,
        weights,
        y_mat,
        obs_mask,
        parties,
        reference_pollsters=("House1", "House2", "House3"),
        n_iters=15,
    )
    assert set(ref_set) == {"House1", "House2", "House3"}
    for party in parties:
        ref_sum = sum(he[r][party] for r in ref_set)
        assert abs(ref_sum) < 1e-9

    assert he["CIS"]["PSOE"] > 0.035
    assert he["CIS"]["PP"] < -0.035


def test_aggregate_kernel_exclude_cis_headline_toggle() -> None:
    polls = [
        _make_poll(
            "40dB",
            date(2026, 11, 15),
            date(2026, 11, 18),
            {"PP": 0.31, "PSOE": 0.27, "VOX": 0.19, "FA": 0.06},
        ),
        _make_poll(
            "GAD3",
            date(2026, 11, 16),
            date(2026, 11, 19),
            {"PP": 0.32, "PSOE": 0.26, "VOX": 0.19, "FA": 0.06},
        ),
        _make_poll(
            "CIS",
            date(2026, 11, 17),
            date(2026, 11, 20),
            {"PP": 0.21, "PSOE": 0.39, "VOX": 0.16, "FA": 0.05},
            sample_size=25000,
            is_cis=True,
        ),
    ]
    res_with = aggregate_kernel(
        polls,
        date(2026, 11, 23),
        tracked_parties=("PP", "PSOE", "VOX", "FA"),
        exclude_cis_headline=False,
    )
    res_without = aggregate_kernel(
        polls,
        date(2026, 11, 23),
        tracked_parties=("PP", "PSOE", "VOX", "FA"),
        exclude_cis_headline=True,
    )
    assert res_with.n_polls_used == 3
    assert res_without.n_polls_used == 2
    assert res_without.exclude_cis_headline is True
    assert abs(res_with.mean_shares["PSOE"] - res_without.mean_shares["PSOE"]) > 0.005

    # Real 2026 poll snapshot
    polls_26 = load_national_polls(get_polls_dir() / "national_2026.csv")
    real_res = aggregate_kernel(polls_26, BLACKOUT_CUTOFF_DATE_2026)
    tot_real = sum(real_res.mean_shares.values()) + real_res.blank_share
    assert abs(tot_real - 1.0) < 1e-6
    assert real_res.n_effective_polls > 10.0


def test_state_space_blackout_variance_growth_and_psd() -> None:
    polls_26 = load_national_polls(get_polls_dir() / "national_2026.csv")
    ratings = compute_pollster_ratings()
    as_of = BLACKOUT_CUTOFF_DATE_2026

    ss_cutoff = smooth_state_space(
        polls_26,
        as_of,
        target_election_date=as_of,
        ratings=ratings,
    )
    ss_elec = smooth_state_space(
        polls_26,
        as_of,
        target_election_date=date(2026, 11, 29),
        ratings=ratings,
    )

    # Monotonic random-walk uncertainty growth across the 6-day LOREG blackout
    for party in ("PP", "PSOE", "VOX", "FA"):
        assert ss_elec.sd_shares[party] > ss_cutoff.sd_shares[party]

    # Exact valid-vote simplex preservation
    tot_elec = sum(ss_elec.mean_shares.values()) + ss_elec.blank_share
    assert abs(tot_elec - 1.0) < 1e-9

    # Symmetric positive semi-definite covariance matrix
    assert np.allclose(ss_elec.cov_shares, ss_elec.cov_shares.T, atol=1e-12)
    eigvals = np.linalg.eigvalsh(ss_elec.cov_shares)
    assert float(eigvals.min()) >= -1e-10


def test_run_stage1_ensemble_and_cli_ratings(tmp_path: Path) -> None:
    polls_26 = load_national_polls(get_polls_dir() / "national_2026.csv")
    ratings = compute_pollster_ratings()
    res = run_stage1(
        polls_26,
        BLACKOUT_CUTOFF_DATE_2026,
        engine="ensemble",
        ratings=ratings,
        compute_cis_sensitivity=True,
    )
    assert res.engine == "ensemble"
    assert "kernel" in res.stage1_comparison
    assert "state_space" in res.stage1_comparison
    assert "ensemble" in res.stage1_comparison
    assert "with_cis" in res.cis_sensitivity
    assert "without_cis" in res.cis_sensitivity
    assert "cis_weight_pct" in res.cis_sensitivity
    assert "cis_house_effect_pp" in res.cis_sensitivity
    assert abs(sum(res.mean_shares.values()) + res.blank_share - 1.0) < 1e-9

    # Kernel and state_space runs also populate ensemble comparison cleanly for UI switching
    for alt_eng in ("kernel", "state_space"):
        alt_res = run_stage1(
            polls_26,
            BLACKOUT_CUTOFF_DATE_2026,
            engine=alt_eng,
            ratings=ratings,
            compute_cis_sensitivity=True,
        )
        assert "ensemble" in alt_res.stage1_comparison
        assert "kernel" in alt_res.stage1_comparison
        assert "state_space" in alt_res.stage1_comparison

    # Leak-free exclusion in pollster ratings removes one historical election
    ratings_no23 = compute_pollster_ratings(exclude_election_ids=["2023-07"])
    by_pol_full = {r.pollster: r for r in ratings}
    by_pol_sub = {r.pollster: r for r in ratings_no23}
    assert by_pol_full["CIS"].grade == "C"
    assert by_pol_sub["CIS"].n_elections == by_pol_full["CIS"].n_elections - 1

    # CLI execution writes valid ratings CSV
    out_csv = tmp_path / "ratings_test.csv"
    runner = CliRunner()
    cli_res = runner.invoke(app, ["ratings", "--output", str(out_csv)])
    assert cli_res.exit_code == 0
    assert out_csv.is_file()
    with out_csv.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) >= 40
    assert any(r["pollster"] == "CIS" and r["grade"] == "C" for r in rows)
