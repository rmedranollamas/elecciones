"""Zero-backend static HTML/SVG editorial dashboard compiler (docs/index.html)."""

from __future__ import annotations

import csv
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from elecciones.config import (
    BLACKOUT_END_CET,
    BLACKOUT_START_CET,
    MADRID_TZ,
    get_docs_dir,
    get_metadata_dir,
    get_outputs_dir,
    get_polls_dir,
    is_in_blackout_window,
)
from elecciones.ingest.cross_check import apply_manual_poll_overlay, load_manual_overlays
from elecciones.ingest.wikipedia import load_national_polls
from elecciones.models.kernel_aggregator import summarize_latest_poll
from elecciones.schemas import (
    BacktestSummary,
    ForecastBundle,
    SchemaValidationError,
    load_constituencies_catalog,
)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
_ES_MONTHS = (
    "",
    "ene",
    "feb",
    "mar",
    "abr",
    "may",
    "jun",
    "jul",
    "ago",
    "sep",
    "oct",
    "nov",
    "dic",
)


def _format_es_date(iso_date: str | None) -> str:
    """Format an ISO YYYY-MM-DD date concisely in Spanish (e.g. '19 nov 2026')."""
    if not iso_date:
        return "—"
    try:
        d = date.fromisoformat(str(iso_date)[:10])
    except ValueError:
        return str(iso_date)
    return f"{d.day} {_ES_MONTHS[d.month]} {d.year}"


def _format_es_date_range(start_iso: str | None, end_iso: str | None) -> str:
    """Format a fieldwork date window concisely in Spanish."""
    if not end_iso and not start_iso:
        return "—"
    if not start_iso or start_iso == end_iso:
        return _format_es_date(end_iso or start_iso)
    try:
        ds = date.fromisoformat(str(start_iso)[:10])
        de = date.fromisoformat(str(end_iso)[:10])
    except ValueError:
        return _format_es_date(end_iso or start_iso)
    if ds == de:
        return f"{de.day} {_ES_MONTHS[de.month]} {de.year}"
    if ds.year == de.year and ds.month == de.month:
        return f"{ds.day}–{de.day} {_ES_MONTHS[de.month]} {de.year}"
    if ds.year == de.year:
        return f"{ds.day} {_ES_MONTHS[ds.month]}–{de.day} {_ES_MONTHS[de.month]} {de.year}"
    return f"{_format_es_date(start_iso)}–{_format_es_date(end_iso)}"


def _format_es_datetime(iso_dt: str | None) -> str:
    """Format an ISO-8601 timestamp into a crisp Spanish editorial dateline."""
    if not iso_dt:
        return "—"
    try:
        dt = datetime.fromisoformat(str(iso_dt))
    except ValueError:
        return str(iso_dt)
    tz_label = ""
    if dt.utcoffset() is not None:
        offset_min = int(dt.utcoffset().total_seconds() // 60)
        if offset_min == 60:
            tz_label = " CET"
        elif offset_min == 120:
            tz_label = " CEST"
        elif offset_min == 0:
            tz_label = " UTC"
        else:
            sign = "+" if offset_min >= 0 else "-"
            h, m = divmod(abs(offset_min), 60)
            tz_label = f" UTC{sign}{h:02d}:{m:02d}"
    return f"{dt.day} {_ES_MONTHS[dt.month]} {dt.year} · {dt.strftime('%H:%M')}{tz_label}"


def _resolve_latest_poll_info(
    bundle: ForecastBundle,
    polls_dir: Path | None = None,
) -> tuple[dict[str, Any] | None, int | None]:
    """Resolve latest active poll metadata from bundle, local poll CSVs, or trend_series."""
    latest = dict(bundle.metadata.latest_poll) if bundle.metadata.latest_poll else None
    n_polls = bundle.metadata.n_polls_2026

    if latest is None:
        polls_root = polls_dir or get_polls_dir()
        nat_csv = polls_root / "national_2026.csv"
        if nat_csv.is_file():
            try:
                base_polls = load_national_polls(nat_csv)
                manual_csv = polls_root / "manual_polls_2026.csv"
                overlays = load_manual_overlays(manual_csv) if manual_csv.is_file() else []
                eff_polls = (
                    apply_manual_poll_overlay(base_polls, overlays)
                    if overlays
                    else list(base_polls)
                )
                as_of = date.fromisoformat(bundle.metadata.as_of_date)
                latest, resolved_n = summarize_latest_poll(
                    eff_polls,
                    as_of,
                    exclude_cis_headline=bundle.metadata.exclude_cis_headline,
                )
                if n_polls is None and resolved_n > 0:
                    n_polls = resolved_n
            except (OSError, SchemaValidationError, ValueError):
                latest = None

    if latest is None:
        trend_polls = list(bundle.trend_series.get("polls", []))
        if trend_polls:
            ordered = sorted(
                trend_polls,
                key=lambda item: (str(item.get("date", "")), str(item.get("pollster", ""))),
            )
            last = ordered[-1]
            eff_date = str(last.get("date", ""))
            same_day = [p for p in ordered if str(p.get("date", "")) == eff_date]
            latest = {
                "pollster": str(last.get("pollster", "")),
                "media": "",
                "fieldwork_start": eff_date,
                "fieldwork_end": eff_date,
                "pub_date": None,
                "effective_date": eff_date,
                "sample_size": None,
                "is_cis": bool(last.get("is_cis", False)),
                "same_day_count": len(same_day),
                "same_day_others": [str(p.get("pollster", "")) for p in same_day[:-1]],
            }
            if n_polls is None:
                n_polls = len(ordered)

    return latest, n_polls


def _build_freshness_context(
    meta_dict: dict[str, Any],
) -> dict[str, Any]:
    """Create human-readable Spanish freshness labels for the masthead bar."""
    latest = meta_dict.get("latest_poll") or None
    latest_ctx: dict[str, Any] | None = None
    if isinstance(latest, dict) and latest.get("pollster"):
        fw_start = latest.get("fieldwork_start")
        fw_end = latest.get("fieldwork_end")
        eff_date = latest.get("effective_date") or latest.get("pub_date") or fw_end
        others = [str(x) for x in (latest.get("same_day_others") or []) if str(x).strip()]
        latest_ctx = {
            "pollster": str(latest["pollster"]),
            "media": str(latest.get("media") or "").strip(),
            "fieldwork_label": _format_es_date_range(fw_start, fw_end),
            "date_label": _format_es_date(eff_date),
            "effective_date": str(eff_date or ""),
            "sample_size": latest.get("sample_size"),
            "is_cis": bool(latest.get("is_cis", False)),
            "same_day_others": others,
        }

    days_left: int | None = None
    try:
        d_elec = date.fromisoformat(str(meta_dict.get("election_date", "")))
        d_as_of = date.fromisoformat(str(meta_dict.get("as_of_date", "")))
        today_madrid = datetime.now(MADRID_TZ).date()
        ref_date = max(d_as_of, today_madrid)
        days_left = (d_elec - ref_date).days
    except ValueError:
        days_left = None

    return {
        "generated_at_iso": str(meta_dict.get("generated_at", "")),
        "generated_at_human": _format_es_datetime(meta_dict.get("generated_at")),
        "as_of_human": _format_es_date(meta_dict.get("as_of_date")),
        "latest_poll": latest_ctx,
        "n_polls_2026": meta_dict.get("n_polls_2026"),
        "days_to_election": days_left,
    }


def _load_ratings_csv(
    csv_path: Path | None,
    fallback_scorecard: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Load pollster ratings CSV rows with typed numeric fields or fall back safely."""
    if csv_path is None or not csv_path.is_file():
        return list(fallback_scorecard)

    int_cols = {"n_elections", "n_polls_2026"}
    float_cols = {
        "shrunk_mae",
        "grade_weight",
        "herding_ratio",
        "herding_penalty",
        "delta_PP",
        "delta_PSOE",
        "delta_VOX",
        "delta_FA",
        "delta_PODEMOS",
    }
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            row: dict[str, Any] = {}
            for key, val in raw.items():
                if key in int_cols:
                    row[key] = int(float(val or 0))
                elif key in float_cols:
                    row[key] = round(float(val or 0.0), 4)
                else:
                    row[key] = str(val or "").strip()
            if row.get("pollster"):
                rows.append(row)
    return rows if rows else list(fallback_scorecard)


def _compact_constituency_dict(
    con_dict: dict[str, Any],
    validos_prior: int,
) -> dict[str, Any]:
    """Keep full active-party constituency data while omitting inactive zeros cleanly."""
    pp = dict(con_dict.get("party_projections", {}))
    fs_raw = dict(con_dict.get("final_shares", {}))
    active_fs = {
        k: round(float(v), 6)
        for k, v in fs_raw.items()
        if (k in pp) or float(v) > 0.0
    }
    slots = [
        s
        for s in con_dict.get("seat_slots", [])
        if float(s.get("prob", 0.0)) >= 0.05
    ]
    return {
        "cpro": con_dict["cpro"],
        "name": con_dict["name"],
        "codauto": con_dict["codauto"],
        "seats_2026": int(con_dict["seats_2026"]),
        "turnout_mean": round(float(con_dict["turnout_mean"]), 5),
        "blank_share": round(float(con_dict["blank_share"]), 6),
        "validos_prior": int(validos_prior),
        "final_shares": active_fs,
        "party_projections": pp,
        "seat_slots": slots,
        "dhondt_quotients": dict(con_dict.get("dhondt_quotients", {})),
        "last_seat_battle": dict(con_dict.get("last_seat_battle", {})),
    }


def _safe_json_for_html_script(payload: Any) -> str:
    """Serialize JSON cleanly without allowing ``</script>`` or JS line-separator breaks."""
    raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return raw.replace("</", "<\\/").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def build_dashboard_payload(
    bundle: ForecastBundle,
    *,
    backtest: BacktestSummary | None = None,
    ratings_rows: list[dict[str, Any]] | None = None,
    metadata_dir: Path | None = None,
    polls_dir: Path | None = None,
) -> dict[str, Any]:
    """Assemble the complete validated payload embedded in ``docs/index.html``."""
    meta_root = metadata_dir or get_metadata_dir()
    validos_by_cpro: dict[str, int] = {}
    const_json = meta_root / "constituencies_2026.json"
    if const_json.is_file():
        consts = load_constituencies_catalog(const_json)
        validos_by_cpro = {
            c.cpro: max(1000, int(c.validos_2023 or 100_000))
            for c in consts.constituencies
        }

    compact_consts = []
    for con in sorted(bundle.constituencies, key=lambda item: item.cpro):
        v_prior = validos_by_cpro.get(con.cpro, max(10_000, int(con.seats_2026) * 55_000))
        compact_consts.append(_compact_constituency_dict(con.to_dict(), v_prior))

    meta_dict = bundle.metadata.to_dict()
    latest_poll, n_polls_2026 = _resolve_latest_poll_info(bundle, polls_dir=polls_dir)
    if latest_poll is not None:
        meta_dict["latest_poll"] = latest_poll
    if n_polls_2026 is not None:
        meta_dict["n_polls_2026"] = int(n_polls_2026)
    # Ensure blackout flag and phase match live window cleanly regardless of snapshot build time
    now_madrid = datetime.now(MADRID_TZ)
    live_blackout = bool(is_in_blackout_window(now_madrid))
    meta_dict["live_blackout_active"] = live_blackout
    if now_madrid >= BLACKOUT_END_CET:
        blackout_phase = "post"
    elif live_blackout:
        blackout_phase = "blackout"
    else:
        blackout_phase = "pre"
    meta_dict["blackout_phase"] = blackout_phase
    meta_dict["blackout_start_cet"] = BLACKOUT_START_CET.isoformat()
    meta_dict["blackout_end_cet"] = BLACKOUT_END_CET.isoformat()

    return {
        "metadata": meta_dict,
        "national": dict(bundle.national),
        "tier_totals": dict(bundle.tier_totals),
        "trend_series": dict(bundle.trend_series),
        "constituencies": compact_consts,
        "battlegrounds": list(bundle.battlegrounds),
        "investiture": bundle.investiture.to_dict(),
        "pollster_scorecard": list(
            ratings_rows if ratings_rows is not None else bundle.pollster_scorecard
        ),
        "sim_sample": list(bundle.sim_sample),
        "backtest": backtest.to_dict() if backtest is not None else None,
        "party_order": list(bundle.national.get("seats", {}).keys()),
    }


def generate_dashboard(
    forecast_path: Path | None = None,
    *,
    backtest_path: Path | None = None,
    ratings_path: Path | None = None,
    output_path: Path | None = None,
    metadata_dir: Path | None = None,
    polls_dir: Path | None = None,
) -> Path:
    """Compile zero-backend interactive HTML editorial dashboard from forecast artifacts."""
    resolved_forecast = forecast_path or (get_outputs_dir() / "forecast_29n.json")
    if not resolved_forecast.is_file():
        raise FileNotFoundError(
            f"Forecast JSON not found at {resolved_forecast}; run `elecciones forecast` first."
        )

    bundle = ForecastBundle.load_json(resolved_forecast)

    resolved_backtest = backtest_path or (get_outputs_dir() / "backtest_summary.json")
    backtest: BacktestSummary | None = None
    if resolved_backtest.is_file():
        try:
            backtest = BacktestSummary.load_json(resolved_backtest)
        except (json.JSONDecodeError, SchemaValidationError):
            backtest = None

    resolved_ratings = ratings_path or (get_outputs_dir() / "pollster_ratings.csv")
    ratings_rows = _load_ratings_csv(resolved_ratings, bundle.pollster_scorecard)

    payload = build_dashboard_payload(
        bundle,
        backtest=backtest,
        ratings_rows=ratings_rows,
        metadata_dir=metadata_dir,
        polls_dir=polls_dir,
    )
    payload_json = _safe_json_for_html_script(payload)
    freshness = _build_freshness_context(payload["metadata"])

    seats_map = bundle.national["seats"]
    vs_map = bundle.national["vote_shares"]
    top_parties = []
    for pid, s_obj in list(seats_map.items())[:6]:
        pmeta = bundle.national.get("party_meta", {}).get(pid, {})
        top_parties.append(
            {
                "id": pid,
                "short_name": pmeta.get("short_name", pid),
                "color": pmeta.get("color", "#64748B"),
                "mean_seats": s_obj["mean"],
                "median": s_obj["median"],
                "p10": s_obj["p10"],
                "p90": s_obj["p90"],
                "vote_pct": round(float(vs_map[pid]["mean"]) * 100.0, 1),
            }
        )

    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template("index.html.j2")
    rendered = template.render(
        metadata=payload["metadata"],
        freshness=freshness,
        tier_totals=payload["tier_totals"],
        investiture=payload["investiture"],
        top_parties=top_parties,
        has_backtest=backtest is not None,
        payload_json=payload_json,
    )

    resolved_output = output_path or (get_docs_dir() / "index.html")
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_text(rendered, encoding="utf-8")
    return resolved_output
