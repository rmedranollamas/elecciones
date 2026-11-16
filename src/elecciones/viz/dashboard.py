"""Zero-backend static HTML/SVG editorial dashboard compiler (docs/index.html)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from elecciones.config import (
    get_docs_dir,
    get_metadata_dir,
    get_outputs_dir,
    is_in_blackout_window,
)
from elecciones.schemas import (
    BacktestSummary,
    ForecastBundle,
    SchemaValidationError,
    load_constituencies_catalog,
)

_TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


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
    # Ensure blackout flag matches live window as well as bundle metadata cleanly
    meta_dict["live_blackout_active"] = bool(is_in_blackout_window())

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
    )
    payload_json = _safe_json_for_html_script(payload)

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
