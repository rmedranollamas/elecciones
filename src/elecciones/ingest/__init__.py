"""Data ingestion orchestration package for historical & 29-N (2026) snapshots."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from elecciones.config import get_elections_dir, get_metadata_dir, get_polls_dir
from elecciones.ingest.cis import (
    build_cis_3584_and_transfers,
    load_cis_3584_provincial,
    load_coalition_transfers,
)
from elecciones.ingest.cross_check import (
    apply_manual_poll_overlay,
    ensure_brujula_js,
    ensure_manual_polls_template,
    load_manual_overlays,
    parse_brujula_polls,
    reconcile_with_brujula,
)
from elecciones.ingest.elections import (
    build_all_elections,
    build_constituencies_catalog,
    build_parties_catalog,
    load_congress_results,
)
from elecciones.ingest.wikipedia import (
    build_national_2015_2023_polls,
    build_national_2026_polls,
    build_regional_2026_polls,
    load_national_polls,
    load_regional_polls,
)


def fetch_all(
    *,
    since: int = 2015,
    offline: bool = False,
    refresh_live: bool = True,
    refresh_static: bool = False,
    raw_cache_dir: Path | None = None,
    metadata_dir: Path | None = None,
    elections_dir: Path | None = None,
    polls_dir: Path | None = None,
) -> dict[str, Any]:
    """Run full Milestone 2 ingestion pipeline and return summary diagnostics."""
    meta_root = metadata_dir or get_metadata_dir()
    elec_root = elections_dir or get_elections_dir()
    polls_root = polls_dir or get_polls_dir()

    const_offline = offline or (
        not refresh_static and (meta_root / "constituencies_2026.json").is_file()
    )
    parties_offline = offline or (
        not refresh_static and (meta_root / "parties_2026.json").is_file()
    )
    static_election_files = (
        "congress_2015_12.csv",
        "congress_2016_06.csv",
        "congress_2019_04.csv",
        "congress_2019_11.csv",
        "congress_2023_07.csv",
        "europarl_2024_06.csv",
        "regional_2024_2026.csv",
    )
    elec_offline = offline or (
        not refresh_static and all((elec_root / fname).is_file() for fname in static_election_files)
    )
    hist_polls_offline = offline or (
        not refresh_static and (polls_root / "national_2015_2023.csv").is_file()
    )

    constituencies = build_constituencies_catalog(
        offline=const_offline,
        metadata_dir=metadata_dir,
        raw_cache_dir=raw_cache_dir,
    )
    parties = build_parties_catalog(
        offline=parties_offline,
        metadata_dir=metadata_dir,
        raw_cache_dir=raw_cache_dir,
    )
    election_counts = build_all_elections(
        offline=elec_offline,
        raw_cache_dir=raw_cache_dir,
        elections_dir=elections_dir,
    )

    hist_polls = build_national_2015_2023_polls(
        since=since,
        offline=hist_polls_offline,
        raw_cache_dir=raw_cache_dir,
        polls_dir=polls_dir,
    )
    polls_2026 = build_national_2026_polls(
        offline=offline,
        refresh_live=refresh_live,
        raw_cache_dir=raw_cache_dir,
        polls_dir=polls_dir,
    )
    reg_polls = build_regional_2026_polls(
        offline=offline,
        refresh_live=refresh_live,
        raw_cache_dir=raw_cache_dir,
        polls_dir=polls_dir,
    )

    cis_rows, _transfers = build_cis_3584_and_transfers(
        offline=offline,
        refresh_static=refresh_static,
        raw_cache_dir=raw_cache_dir,
        polls_dir=polls_dir,
        metadata_dir=metadata_dir,
        elections_dir=elections_dir,
        national_2026_polls=polls_2026,
    )

    manual_csv = ensure_manual_polls_template(
        (polls_dir / "manual_polls_2026.csv") if polls_dir is not None else None
    )
    overlays = load_manual_overlays(manual_csv)
    polls_2026_with_manual = apply_manual_poll_overlay(polls_2026, overlays)

    brujula_report: dict[str, Any] = {}
    try:
        js_path = ensure_brujula_js(
            offline=offline,
            refresh_live=refresh_live,
            raw_cache_dir=raw_cache_dir,
        )
        brujula_polls = parse_brujula_polls(js_path)
        brujula_report = reconcile_with_brujula(polls_2026_with_manual, brujula_polls)
    except Exception as exc:
        brujula_report = {"warning": str(exc)}

    return {
        "constituencies": len(constituencies.constituencies),
        "parties": len(parties.parties),
        "elections": election_counts,
        "polls_2015_2023": len(hist_polls),
        "polls_2026": len(polls_2026),
        "manual_overlays": len(overlays),
        "polls_2026_effective": len(polls_2026_with_manual),
        "regional_polls_2026": len(reg_polls),
        "cis_3584_districts": len(cis_rows),
        "brujula_reconciliation": brujula_report,
    }


__all__ = [
    "apply_manual_poll_overlay",
    "build_all_elections",
    "build_cis_3584_and_transfers",
    "build_constituencies_catalog",
    "build_national_2015_2023_polls",
    "build_national_2026_polls",
    "build_parties_catalog",
    "build_regional_2026_polls",
    "ensure_manual_polls_template",
    "fetch_all",
    "load_cis_3584_provincial",
    "load_coalition_transfers",
    "load_congress_results",
    "load_manual_overlays",
    "load_national_polls",
    "load_regional_polls",
]
