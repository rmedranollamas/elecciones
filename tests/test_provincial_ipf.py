"""Tests for Milestone 3 Stage 2 52-province IPF, coalition transfers & masks."""

from __future__ import annotations

from datetime import date

import numpy as np

from elecciones.config import BLACKOUT_CUTOFF_DATE_2026, HISTORICAL_ELECTION_DATES, get_polls_dir
from elecciones.ingest import load_coalition_transfers, load_national_polls, load_regional_polls
from elecciones.models import (
    aggregate_kernel,
    allocate_batch_vectorized,
    compute_pollster_ratings,
    project_2026_provinces,
    project_historical_provinces,
    run_stage1,
    split_sumar_2023_share,
)
from elecciones.schemas import (
    SIMPLEX_TOL,
    RegionalPollRecord,
    district_simplex_sum,
    load_parties_catalog,
)


def test_boe_2026_candidacy_mask_and_upn_pp_merger() -> None:
    polls_26 = load_national_polls(get_polls_dir() / "national_2026.csv")
    ratings = compute_pollster_ratings()
    s1 = run_stage1(
        polls_26,
        BLACKOUT_CUTOFF_DATE_2026,
        engine="ensemble",
        ratings=ratings,
        compute_cis_sensitivity=False,
    )
    proj = project_2026_provinces(
        s1.mean_shares,
        BLACKOUT_CUTOFF_DATE_2026,
        blank_nat=s1.blank_share,
        ratings=ratings,
    )
    parties = load_parties_catalog()

    for cpro, dist in proj.districts.items():
        for pid in parties.party_ids():
            m = parties.get(pid).boe_mask_2026[cpro]
            if m == 0:
                assert dist.prior_shares[pid] == 0.0
                assert dist.pre_shrinkage_ipf_shares[pid] == 0.0
                assert dist.final_shares[pid] == 0.0

        # Exact simplex invariance across all stages
        assert abs(district_simplex_sum(dist.prior_shares, dist.blank_share) - 1.0) <= SIMPLEX_TOL
        assert (
            abs(
                district_simplex_sum(dist.pre_shrinkage_ipf_shares, dist.pre_shrinkage_blank_share)
                - 1.0
            )
            <= SIMPLEX_TOL
        )
        assert abs(district_simplex_sum(dist.final_shares, dist.blank_share) - 1.0) <= SIMPLEX_TOL

    # Unproclaimed parties globally 0
    for zero_p in ("AC", "CUP", "UPL"):
        assert proj.post_shrinkage_national[zero_p] == 0.0

    # Navarre UPN+PP unified candidacy & absence of separate PP list
    nav = proj.districts["31"]
    assert nav.final_shares["PP"] == 0.0
    assert nav.final_shares["UPN_PP"] > 0.20

    # Madrid omits UPN_PP / Compromís-FV / ERC / Junts
    mad = proj.districts["28"]
    for absent in ("UPN_PP", "COMPROMIS_FV", "ERC", "JUNTS"):
        assert mad.final_shares[absent] == 0.0

    # Valencian Community routes Sumar -> Compromís-FV (FA absent)
    val = proj.districts["46"]
    assert val.final_shares["FA"] == 0.0
    assert val.final_shares["COMPROMIS_FV"] > 0.05

    # SC Tenerife seeds VOX cleanly despite 0 Congress ballots in 2023
    tfe = proj.districts["38"]
    assert tfe.prior_shares["VOX"] > 0.05
    assert tfe.final_shares["VOX"] > 0.10


def test_split_sumar_2023_share_conservation_across_zones() -> None:
    parties = load_parties_catalog()
    transfers = load_coalition_transfers()

    zone_samples = [
        ("46", "10", 0.152),  # Valencia (COMPROMIS_FV)
        ("35", "05", 0.105),  # Las Palmas (UPC)
        ("07", "04", 0.165),  # Illes Balears (ARA_MES)
        ("08", "09", 0.141),  # Barcelona (FA, PODEMOS masked out)
        ("41", "01", 0.121),  # Sevilla (FA + ADELANTE + PODEMOS)
        ("44", "02", 0.055),  # Teruel (FA absent -> PODEMOS + TERUEL_EXISTE + PSOE)
        ("51", "18", 0.025),  # Ceuta (FA absent -> PODEMOS + PSOE)
        ("52", "19", 0.030),  # Melilla (FA & PODEMOS absent -> PSOE)
        ("28", "13", 0.155),  # Madrid (general_territory)
    ]
    for cpro, codauto, sumar_share in zone_samples:
        mask_map = {pid: parties.get(pid).boe_mask_2026[cpro] for pid in parties.party_ids()}
        splits = split_sumar_2023_share(cpro, codauto, sumar_share, mask_map, transfers)
        assert abs(sum(splits.values()) - sumar_share) < 1e-12
        for pid, val in splits.items():
            assert val >= 0.0
            assert mask_map[pid] == 1


def test_ipf_marginals_and_deterministic_350_seat_allocation() -> None:
    polls_26 = load_national_polls(get_polls_dir() / "national_2026.csv")
    s1 = run_stage1(
        polls_26,
        BLACKOUT_CUTOFF_DATE_2026,
        engine="kernel",
        compute_cis_sensitivity=False,
    )
    proj = project_2026_provinces(
        s1.mean_shares,
        BLACKOUT_CUTOFF_DATE_2026,
        blank_nat=s1.blank_share,
    )
    assert proj.ipf_max_marginal_err < 1e-6

    for party, targ in proj.target_national.items():
        pre_nat = proj.pre_shrinkage_national[party]
        assert abs(pre_nat - targ) < 1e-6, f"IPF marginal mismatch for {party}: {pre_nat} vs {targ}"

    # Deterministic mean seats across all 52 districts sum strictly to 350
    seats_mean = allocate_batch_vectorized(
        proj.final_matrix[None, :, :],
        proj.seats_vec,
        is_autonomous_city=proj.city_vec,
        blank_votes_or_shares=proj.final_blank[None, :],
        active_mask=proj.active_mask,
    )[0]
    assert int(seats_mean.sum()) == 350
    assert int(seats_mean[:, proj.party_ids.index("AC")].sum()) == 0


def test_regional_poll_as_of_cutoff_prevents_future_leakage() -> None:
    polls_26 = load_national_polls(get_polls_dir() / "national_2026.csv")
    s1 = aggregate_kernel(polls_26, date(2026, 11, 16))
    base_reg = load_regional_polls(get_polls_dir() / "regional_2026.csv")

    future_poll = RegionalPollRecord(
        scope_type="province",
        scope_code="28",
        pollster="40dB",
        fieldwork_start=date(2026, 11, 19),
        fieldwork_end=date(2026, 11, 22),
        sample_size=2800,
        party_shares={"PP": 0.49, "PSOE": 0.19, "VOX": 0.16, "FA": 0.06},
    )

    p_no_future = project_2026_provinces(
        s1.mean_shares,
        date(2026, 11, 16),
        blank_nat=s1.blank_share,
        regional_polls=base_reg,
    )
    p_with_future_blocked = project_2026_provinces(
        s1.mean_shares,
        date(2026, 11, 16),
        blank_nat=s1.blank_share,
        regional_polls=[*base_reg, future_poll],
    )
    assert np.allclose(
        p_no_future.final_matrix, p_with_future_blocked.final_matrix, atol=1e-12
    )

    p_after_pub = project_2026_provinces(
        s1.mean_shares,
        date(2026, 11, 22),
        blank_nat=s1.blank_share,
        regional_polls=[*base_reg, future_poll],
    )
    assert (
        p_after_pub.districts["28"].final_shares["PP"]
        > p_with_future_blocked.districts["28"].final_shares["PP"]
    )


def test_historical_provincial_projection_all_five_cycles() -> None:
    hist_polls = load_national_polls(get_polls_dir() / "national_2015_2023.csv")
    for eid, edate in HISTORICAL_ELECTION_DATES.items():
        as_of = date.fromordinal(edate.toordinal() - 1)
        epolls = [p for p in hist_polls if p.election_id == eid]
        s1 = aggregate_kernel(epolls, as_of, election_id=eid, target_election_date=edate)
        proj = project_historical_provinces(eid, s1.mean_shares, as_of)
        assert len(proj.districts) == 52
        assert int(proj.seats_vec.sum()) == 350
        assert proj.ipf_max_marginal_err < 1e-6
        for dist in proj.districts.values():
            simplex = district_simplex_sum(dist.final_shares, dist.blank_share)
            assert abs(simplex - 1.0) <= SIMPLEX_TOL
