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
    assert "Mediana simulada" in text
    assert "Media simulada" not in text
    assert "Madrid — 38 escaños" in text
    assert "Madrid (28)" not in text
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

    # If node is on PATH, verify inline JS syntax and run behavior under minimal DOM harness
    node_bin = shutil.which("node")
    if node_bin is not None:
        js_match = re.search(r"<script>(.*?)</script>\s*</body>", text, re.DOTALL)
        assert js_match is not None
        js_file = tmp_path / "dash.js"
        js_file.write_text(js_match.group(1), encoding="utf-8")
        proc = subprocess.run([node_bin, "--check", str(js_file)], capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr

        harness = tmp_path / "verify_dom.js"
        harness.write_text(
            f"""
const fs = require("fs");
const vm = require("vm");
const payloadStr = fs.readFileSync({json.dumps(str(out_html))}, "utf8")
  .match(/<script id="elecciones-data" type="application\\/json">(.*?)<\\/script>/s)[1];

function makeEl(id, attrs = {{}}) {{
  const listeners = {{}};
  const children = [];
  const classes = new Set(attrs.cls || []);
  const el = {{
    id: id || "",
    _attrs: Object.assign({{}}, attrs),
    children,
    style: {{}},
    innerHTML: "",
    textContent: "",
    value: attrs.value || "",
    classList: {{
      add: c => classes.add(c),
      remove: c => classes.delete(c),
      toggle: (c, force) => {{
        if (force === undefined) {{ if (classes.has(c)) classes.delete(c); else classes.add(c); }}
        else if (force) classes.add(c);
        else classes.delete(c);
      }},
      contains: c => classes.has(c),
    }},
    getAttribute: k => el._attrs[k] !== undefined ? String(el._attrs[k]) : null,
    setAttribute: (k, v) => {{ el._attrs[k] = String(v); }},
    addEventListener: (ev, fn) => {{ (listeners[ev] = listeners[ev] || []).push(fn); }},
    dispatchEvent: (evObj) => {{
      const fns = listeners[evObj.type] || [];
      fns.forEach(fn => fn(evObj));
    }},
    click: () => el.dispatchEvent({{ type: "click" }}),
    appendChild: c => children.push(c),
    closest: () => el,
    querySelector: () => makeEl(""),
    querySelectorAll: () => [],
  }};
  return el;
}}

const els = {{}};
const getEl = id => (els[id] = els[id] || makeEl(id));
getEl("elecciones-data").textContent = payloadStr;
getEl("national-table").querySelector = () => getEl("national-table-tbody");
getEl("battlegrounds-table").querySelector = () => getEl("bg-tbody");
getEl("insp-party-table").querySelector = () => getEl("insp-tbody");
getEl("whatif-table").querySelector = () => getEl("wi-tbody");
getEl("investiture-mc-table").querySelector = () => getEl("inv-tbody");
getEl("pollster-table").querySelector = () => getEl("pol-tbody");
getEl("bt-models-table").querySelector = () => getEl("btm-tbody");
getEl("bt-elections-table").querySelector = () => getEl("bte-tbody");

const sliderInputPp = makeEl("sl-in-PP", {{ "data-pid": "PP", value: "0" }});
getEl("whatif-sliders").querySelectorAll = sel => (
  sel === "input[type='range']" ? [sliderInputPp] : []
);

const engineBtns = [
  makeEl("eb0", {{ "data-engine": "ensemble", cls: ["seg-btn", "active"] }}),
  makeEl("eb1", {{ "data-engine": "kernel", cls: ["seg-btn"] }}),
  makeEl("eb2", {{ "data-engine": "state_space", cls: ["seg-btn"] }}),
];
const cisBtns = [
  makeEl("cb0", {{ "data-cis": "with", cls: ["seg-btn", "active"] }}),
  makeEl("cb1", {{ "data-cis": "without", cls: ["seg-btn"] }}),
];

const documentStub = {{
  getElementById: id => getEl(id),
  createElement: tag => makeEl(tag),
  querySelector: sel => {{
    if (sel.startsWith("#")) {{
      const [idPart] = sel.slice(1).split(" ");
      return getEl(idPart).querySelector();
    }}
    return makeEl("");
  }},
  querySelectorAll: sel => {{
    if (sel === "#engine-toggle .seg-btn") return engineBtns;
    if (sel === "#cis-toggle .seg-btn") return cisBtns;
    return [];
  }},
}};

const ctx = {{
  document: documentStub,
  window: {{ innerWidth: 1280, innerHeight: 800 }},
  Math, Number, Object, Array, String, Set, Date, JSON, console,
}};
vm.runInNewContext(fs.readFileSync({json.dumps(str(js_file))}, "utf8"), ctx);

function parseTotalSeats(tbodyHtml) {{
  // Col 3 (0-indexed 2) in counterfactual mode is deterministic seats
  let sum = 0;
  for (const m of tbodyHtml.matchAll(/<tr>(.*?)<\\/tr>/gs)) {{
    const cells = [...m[1].matchAll(/<td[^>]*>(.*?)<\\/td>/gs)];
    const tds = cells.map(x => x[1].replace(/<[^>]+>/g, "").trim());
    if (tds.length >= 4) {{
      const val = Number(tds[2]);
      if (!Number.isNaN(val)) sum += val;
    }}
  }}
  return sum;
}}

const baseTable = getEl("national-table-tbody").innerHTML;
const baseInsp = getEl("insp-tbody").innerHTML;
const baseCartoCount = getEl("cartogram-grid").children.length;
if (baseCartoCount !== 52) throw new Error("Expected 52 cartogram tiles, got " + baseCartoCount);
if (getEl("insp-ccaa").textContent.includes("ESCENARIO SIMULADO")) {{
  throw new Error("Base inspector should not say ESCENARIO SIMULADO");
}}

engineBtns[1].click(); // Switch to kernel
const kernelTable = getEl("national-table-tbody").innerHTML;
const kernelSeats = parseTotalSeats(kernelTable);
const kernelInsp = getEl("insp-tbody").innerHTML;
if (!getEl("mode-notice").classList.contains("visible")) {{
  throw new Error("Notice missing in kernel");
}}
if (!getEl("insp-ccaa").textContent.includes("ESCENARIO SIMULADO")) {{
  throw new Error("Counterfactual kernel switch did not update inspector mode");
}}
if (kernelInsp === baseInsp) throw new Error("Kernel switch left inspector table unchanged");

engineBtns[2].click(); // Switch to state_space
const ssTable = getEl("national-table-tbody").innerHTML;
const ssSeats = parseTotalSeats(ssTable);
if (!getEl("mode-notice").classList.contains("visible")) {{
  throw new Error("Notice missing in state_space");
}}

engineBtns[0].click(); // Return to ensemble, then test What-If slider live cartogram/inspector sync
if (getEl("mode-notice").classList.contains("visible")) {{
  throw new Error("Notice should clear when returning to base ensemble");
}}
sliderInputPp.value = "4.0";
sliderInputPp.dispatchEvent({{ type: "input" }});
const sliderInsp = getEl("insp-tbody").innerHTML;
if (sliderInsp === baseInsp || !getEl("insp-ccaa").textContent.includes("ESCENARIO SIMULADO")) {{
  throw new Error("What-If slider did not update District Inspector live");
}}

if (kernelTable === baseTable) throw new Error("Kernel switcher left table unchanged");
if (ssTable === kernelTable) throw new Error("State-space switcher identical to kernel");
if (kernelSeats !== 350) throw new Error("Kernel counterfactual seats != 350: " + kernelSeats);
if (ssSeats !== 350) throw new Error("State-space counterfactual seats != 350: " + ssSeats);
if (!getEl("cis-diff-strip").innerHTML.includes("Peso efectivo CIS")) {{
  throw new Error("Missing CIS weight note");
}}
console.log("JS harness passed:", {{ kernelSeats, ssSeats }});
""",
            encoding="utf-8",
        )
        run_res = subprocess.run([node_bin, str(harness)], capture_output=True, text=True)
        assert run_res.returncode == 0, run_res.stdout + "\n" + run_res.stderr


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
    shutil.copytree(
        repo_root / "data",
        iso_root / "data",
        ignore=shutil.ignore_patterns("raw_cache"),
    )
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
