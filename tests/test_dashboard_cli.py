"""Tests for Milestone 5: Dashboard HTML/JS/JSON, CLI `dashboard`/`run-all`, & LOREG blackout."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from typer.testing import CliRunner

from elecciones.cli import app
from elecciones.config import BLACKOUT_BANNER, get_docs_dir, get_outputs_dir
from elecciones.viz import generate_dashboard


def test_generate_dashboard_standalone_html_and_script_safety(tmp_path: Path) -> None:
    out_html = tmp_path / "index.html"
    generated = generate_dashboard(output_path=out_html)
    assert generated == out_html
    assert out_html.is_file()

    text = out_html.read_text(encoding="utf-8")
    # Zero unrendered Jinja expressions or external script CDNs
    assert "{{" not in text
    assert "{%" not in text
    assert "https://cdn." not in text
    assert "unpkg.com" not in text

    # Editorial content, SVG hemicycle, cartogram, what-if, and backtest diagnostics
    assert "Radiografía probabilística del Congreso 29-N" in text
    assert 'id="hemicycle-svg"' in text
    assert 'id="cartogram-grid"' in text
    assert 'id="trend-svg"' in text
    assert 'id="whatif-sliders"' in text
    assert 'id="pact-builder-rows"' in text
    assert 'id="pollster-table"' in text
    assert 'id="bt-models-table"' in text

    # Embedded JSON payload passes strict parse cleanly and contains all 52 districts
    match = re.search(
        r'<script id="elecciones-data" type="application/json">(.*?)</script>',
        text,
        re.DOTALL,
    )
    assert match is not None
    payload = json.loads(match.group(1).replace("<\\/", "</"))
    assert len(payload["constituencies"]) == 52
    assert sum(c["seats_2026"] for c in payload["constituencies"]) == 350
    by_c = {c["cpro"]: c for c in payload["constituencies"]}
    assert by_c["28"]["seats_2026"] == 38
    assert by_c["11"]["seats_2026"] == 8
    assert len(payload["sim_sample"]) == 200

    # If node is present on PATH, verify inline JS syntax strictly
    node_bin = shutil.which("node")
    if node_bin is not None:
        js_match = re.search(r"<script>(.*?)</script>\s*</body>", text, re.DOTALL)
        assert js_match is not None
        js_file = tmp_path / "dash.js"
        js_file.write_text(js_match.group(1), encoding="utf-8")
        proc = subprocess.run([node_bin, "--check", str(js_file)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr


def test_dashboard_missing_optional_files_falls_back_cleanly(tmp_path: Path) -> None:
    out_html = tmp_path / "fallback_dash.html"
    generate_dashboard(
        forecast_path=get_outputs_dir() / "forecast_29n.json",
        backtest_path=tmp_path / "does_not_exist_bt.json",
        ratings_path=tmp_path / "does_not_exist_ratings.csv",
        output_path=out_html,
    )
    text = out_html.read_text(encoding="utf-8")
    assert "No se encontró" in text
    assert 'id="elecciones-data"' in text


def test_cli_dashboard_and_run_all_and_blackout_guard(tmp_path: Path, monkeypatch) -> None:
    runner = CliRunner()

    # 1. LOREG Art. 69.7 blackout guard on `run-all`
    res_blk = runner.invoke(
        app,
        ["run-all", "--offline", "--respect-blackout", "--now", "2026-11-26T12:00:00+01:00"],
    )
    assert res_blk.exit_code == 0, res_blk.output
    assert BLACKOUT_BANNER in res_blk.output

    # 2. `elecciones dashboard` CLI invocation into tmp output
    out_cli_html = tmp_path / "cli_dash.html"
    res_dash = runner.invoke(app, ["dashboard", "--output", str(out_cli_html)])
    assert res_dash.exit_code == 0, res_dash.output
    assert out_cli_html.is_file()
    assert out_cli_html.stat().st_size > 150_000

    # 3. Full isolated `run-all --offline` using ELECCIONES_REPO_ROOT copy pointing cleanly at data/
    repo_root = Path(get_docs_dir()).parent
    iso_root = tmp_path / "repo"
    iso_root.mkdir()
    os.symlink(repo_root / "pyproject.toml", iso_root / "pyproject.toml")
    shutil.copytree(repo_root / "data", iso_root / "data")
    monkeypatch.setenv("ELECCIONES_REPO_ROOT", str(iso_root))

    res_all = runner.invoke(
        app,
        [
            "run-all",
            "--offline",
            "--no-respect-blackout",
            "--n-sims",
            "500",
            "--backtest-sims",
            "40",
            "--as-of",
            "2026-11-16",
        ],
    )
    assert res_all.exit_code == 0, res_all.output
    assert "Pipeline complete" in res_all.output
    assert (iso_root / "data" / "outputs" / "pollster_ratings.csv").is_file()
    assert (iso_root / "data" / "outputs" / "backtest_summary.json").is_file()
    assert (iso_root / "data" / "outputs" / "forecast_29n.json").is_file()
    assert (iso_root / "docs" / "index.html").is_file()
