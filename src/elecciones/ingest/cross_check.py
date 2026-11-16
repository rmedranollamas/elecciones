"""Cross-check reconciliation against Brújula 29N & manual poll overlay support.

Manages:
- Parsing and reconciling Wikipedia 2026 polls against ``brujula29n.com`` JS payloads
- Creating empty template ``data/polls/manual_polls_2026.csv`` if missing
- Loading & applying ``add`` / ``override`` / ``exclude`` ``ManualPollOverlay`` records
"""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, Final

from elecciones.config import (
    BLACKOUT_CUTOFF_DATE_2026,
    get_polls_dir,
    get_raw_cache_dir,
    make_http_client,
)
from elecciones.ingest.wikipedia import normalize_pollster_key
from elecciones.schemas import (
    ManualPollOverlay,
    PollRecord,
    SchemaValidationError,
    parse_iso_date,
)

BRUJULA_POLLS_URL: Final[str] = "https://brujula29n.com/en/polls/"

MANUAL_POLLS_COLUMNS: Final[tuple[str, ...]] = (
    "action",
    "election_id",
    "pollster",
    "media",
    "fieldwork_start",
    "fieldwork_end",
    "pub_date",
    "sample_size",
    "turnout",
    "is_cis",
    "PP",
    "PSOE",
    "VOX",
    "FA",
    "COMPROMIS_FV",
    "FA_FV",
    "PODEMOS",
    "SALF",
    "ERC",
    "JUNTS",
    "EH_BILDU",
    "PNV",
    "BNG",
    "CCA",
    "ADELANTE",
    "UPN_PP",
)

_BRUJULA_PARTY_MAP: Final[dict[str, str]] = {
    "PP": "PP",
    "PSOE": "PSOE",
    "VOX": "VOX",
    "SUMAR": "FA_FV",
    "PODEMOS": "PODEMOS",
    "SALF": "SALF",
    "ERC": "ERC",
    "JUNTS": "JUNTS",
    "BILDU": "EH_BILDU",
    "PNV": "PNV",
    "BNG": "BNG",
    "CC": "CCA",
    "UPN": "UPN",
}


def ensure_brujula_js(
    *,
    offline: bool = False,
    refresh_live: bool = True,
    raw_cache_dir: Path | None = None,
) -> Path:
    """Return local cached path for Brújula 29N data JS, fetching live when enabled."""
    cache_dir = raw_cache_dir or get_raw_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_files = sorted(cache_dir.glob("brujula29n_datos.*.js"))
    if cached_files and (offline or not refresh_live):
        return cached_files[-1]
    if offline:
        raise FileNotFoundError(
            f"Missing cached Brújula 29N JS file in {cache_dir} in offline mode."
        )

    try:
        with make_http_client(timeout=30.0) as client:
            html_resp = client.get(BRUJULA_POLLS_URL)
            html_resp.raise_for_status()
            m = re.search(r'(datos\.[a-z0-9]+\.js)', html_resp.text)
            if not m:
                if cached_files:
                    return cached_files[-1]
                raise SchemaValidationError(
                    "Could not find datos.*.js reference in Brújula HTML"
                )
            js_name = m.group(1)
            js_url = f"https://brujula29n.com/en/polls/{js_name}"
            js_resp = client.get(js_url)
            if js_resp.status_code >= 400:
                js_resp = client.get(f"https://brujula29n.com/{js_name}")
            js_resp.raise_for_status()
            target = cache_dir / f"brujula29n_{js_name}"
            target.write_bytes(js_resp.content)
            return target
    except Exception:
        if cached_files:
            return cached_files[-1]
        raise


def parse_brujula_polls(js_path: Path) -> list[PollRecord]:
    """Parse ``window.__DATA__`` Brújula 29N poll array into normalized ``PollRecord``s."""
    text = js_path.read_text(encoding="utf-8")
    brace_idx = text.find("{")
    if brace_idx < 0:
        raise SchemaValidationError(f"Invalid JS payload in {js_path}")
    payload_str = text[brace_idx:].strip().rstrip(";")
    payload = json.loads(payload_str)
    items = payload.get("encuestas", {}).get("encuestas", [])

    records: list[PollRecord] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        raw_end = item.get("campo_fin")
        if not raw_end:
            continue
        d_end = parse_iso_date(str(raw_end))
        if d_end > BLACKOUT_CUTOFF_DATE_2026:
            continue
        raw_start = item.get("campo_ini") or raw_end
        d_start = parse_iso_date(str(raw_start))
        if d_end < d_start:
            d_start = d_end

        casa = str(item.get("casa") or "").strip()
        pollster = normalize_pollster_key(casa)
        media = str(item.get("medio") or "").strip()
        n_val = item.get("n")
        sample_size = int(n_val) if isinstance(n_val, (int, float)) and n_val > 0 else None
        is_cis = pollster == "CIS"

        raw_pct = item.get("pct") or {}
        shares: dict[str, float] = {}
        for bk, pk in _BRUJULA_PARTY_MAP.items():
            val = raw_pct.get(bk)
            if isinstance(val, (int, float)) and 0.0 <= float(val) <= 100.0:
                shares[pk] = float(val) / 100.0
        if not shares:
            continue
        records.append(
            PollRecord(
                election_id="2026-11",
                pollster=pollster,
                media=media,
                fieldwork_start=d_start,
                fieldwork_end=d_end,
                pub_date=None,
                sample_size=sample_size,
                turnout=None,
                is_cis=is_cis,
                party_shares=shares,
            )
        )
    records.sort(key=lambda r: (r.fieldwork_end, r.fieldwork_start, r.pollster))
    return records


def reconcile_with_brujula(
    wikipedia_polls: Sequence[PollRecord],
    brujula_polls: Sequence[PollRecord],
) -> dict[str, Any]:
    """Audit Wikipedia 2026 polls against Brújula 29N on overlapping non-CIS polls."""
    wiki_by_exact: dict[tuple[str, date], PollRecord] = {}
    wiki_by_pollster: dict[str, list[PollRecord]] = {}
    for wp in wikipedia_polls:
        wiki_by_exact[(wp.pollster, wp.fieldwork_end)] = wp
        wiki_by_pollster.setdefault(wp.pollster, []).append(wp)

    matched = 0
    unmatched_in_wiki: list[dict[str, Any]] = []
    abs_diffs_pp: dict[str, list[float]] = {"PP": [], "PSOE": [], "VOX": []}
    flagged_diffs: list[dict[str, Any]] = []

    for bp in brujula_polls:
        match = wiki_by_exact.get((bp.pollster, bp.fieldwork_end))
        if match is None:
            # Tolerate +-1 day coding difference for overlapping waves
            cands = [
                wp
                for wp in wiki_by_pollster.get(bp.pollster, [])
                if abs((wp.fieldwork_end - bp.fieldwork_end).days) <= 1
            ]
            if cands:
                match = min(cands, key=lambda w: abs((w.fieldwork_end - bp.fieldwork_end).days))
        if match is None:
            unmatched_in_wiki.append(
                {
                    "pollster": bp.pollster,
                    "media": bp.media,
                    "fieldwork_end": bp.fieldwork_end.isoformat(),
                }
            )
            continue

        matched += 1
        row_diffs: dict[str, float] = {}
        for pid in ("PP", "PSOE", "VOX"):
            if pid in bp.party_shares and pid in match.party_shares:
                diff_pp = abs(bp.party_shares[pid] - match.party_shares[pid]) * 100.0
                abs_diffs_pp[pid].append(diff_pp)
                row_diffs[pid] = round(diff_pp, 3)
        if row_diffs and max(row_diffs.values()) >= 1.5:
            flagged_diffs.append(
                {
                    "pollster": bp.pollster,
                    "fieldwork_end": bp.fieldwork_end.isoformat(),
                    "diffs_pp": row_diffs,
                }
            )

    mean_abs = {
        pid: (round(sum(vals) / len(vals), 4) if vals else 0.0)
        for pid, vals in abs_diffs_pp.items()
    }
    max_abs = {
        pid: (round(max(vals), 4) if vals else 0.0)
        for pid, vals in abs_diffs_pp.items()
    }
    return {
        "wikipedia_total": len(wikipedia_polls),
        "brujula_total": len(brujula_polls),
        "matched_polls": matched,
        "unmatched_in_wikipedia": unmatched_in_wiki,
        "mean_abs_diff_pp": mean_abs,
        "max_abs_diff_pp": max_abs,
        "flagged_discrepancies": flagged_diffs,
    }


def ensure_manual_polls_template(manual_csv: Path | None = None) -> Path:
    """Create header-only ``data/polls/manual_polls_2026.csv`` if missing."""
    target = manual_csv or (get_polls_dir() / "manual_polls_2026.csv")
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(MANUAL_POLLS_COLUMNS)
    return target


def load_manual_overlays(manual_csv: Path | None = None) -> list[ManualPollOverlay]:
    """Load validated manual overlay rows from ``data/polls/manual_polls_2026.csv``."""
    target = manual_csv or (get_polls_dir() / "manual_polls_2026.csv")
    if not target.is_file():
        return []
    raw_lines = [
        ln
        for ln in target.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    ]
    if len(raw_lines) <= 1:
        return []
    reader = csv.DictReader(raw_lines)
    overlays: list[ManualPollOverlay] = []
    for row in reader:
        if not str(row.get("action") or "").strip():
            continue
        overlays.append(ManualPollOverlay.from_csv_row(row))
    return overlays


def _overlay_matches(base: PollRecord, ov_poll: PollRecord) -> bool:
    if base.election_id != ov_poll.election_id:
        return False
    if normalize_pollster_key(base.pollster) != normalize_pollster_key(ov_poll.pollster):
        return False
    if base.fieldwork_end != ov_poll.fieldwork_end:
        return False
    return base.fieldwork_start == ov_poll.fieldwork_start


def apply_manual_poll_overlay(
    base_polls: Sequence[PollRecord],
    overlays: Sequence[ManualPollOverlay],
) -> list[PollRecord]:
    """Apply ``add`` / ``override`` / ``exclude`` manual overlays onto base polls."""
    result: list[PollRecord] = list(base_polls)
    for ov in overlays:
        target = ov.poll
        if ov.action == "exclude":
            result = [p for p in result if not _overlay_matches(p, target)]
        elif ov.action == "override":
            replaced = False
            new_list: list[PollRecord] = []
            for p in result:
                if _overlay_matches(p, target):
                    if not replaced:
                        new_list.append(target)
                        replaced = True
                else:
                    new_list.append(p)
            if not replaced:
                new_list.append(target)
            result = new_list
        elif ov.action == "add":
            if any(_overlay_matches(p, target) for p in result):
                result = [target if _overlay_matches(p, target) else p for p in result]
            else:
                result.append(target)
    result.sort(key=lambda r: (r.election_id, r.fieldwork_end, r.fieldwork_start, r.pollster))
    return result
