"""Typer CLI entrypoint for ``elecciones`` (LOREG Art. 163 allocator & 29-N pipeline)."""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import typer
import typer.rich_utils as rich_utils
from rich.console import Console
from rich.table import Table
from typer.core import TyperCommand, TyperGroup

from elecciones.config import (
    BLACKOUT_BANNER,
    DEFAULT_N_SIMS,
    SIMULATION_SEED,
    get_docs_dir,
    get_elections_dir,
    get_outputs_dir,
    get_polls_dir,
    is_in_blackout_window,
)
from elecciones.models.dhondt import allocate_dhondt_loreg
from elecciones.schemas import (
    AllocationInput,
    AllocationResult,
    BatchAllocationResult,
    SchemaValidationError,
    ThresholdExhaustionError,
    parse_allocation_payload,
)

# Prevent Rich/Click from truncating option help strings with "…" when COLUMNS is unset.
_HELP_WIDTH = 160
rich_utils.MAX_WIDTH = _HELP_WIDTH


class _WideHelpCommand(TyperCommand):
    """TyperCommand enforcing wide non-ellipsis help formatting."""

    def format_help(self, ctx: Any, formatter: Any) -> None:
        ctx.max_content_width = max(ctx.max_content_width or 0, _HELP_WIDTH)
        super().format_help(ctx, formatter)


class _WideHelpGroup(TyperGroup):
    """TyperGroup enforcing wide non-ellipsis help formatting across subcommands."""

    def format_help(self, ctx: Any, formatter: Any) -> None:
        ctx.max_content_width = max(ctx.max_content_width or 0, _HELP_WIDTH)
        super().format_help(ctx, formatter)


app = typer.Typer(
    name="elecciones",
    cls=_WideHelpGroup,
    no_args_is_help=True,
    add_completion=False,
    context_settings={"max_content_width": _HELP_WIDTH, "help_option_names": ["-h", "--help"]},
    help="Spain 29-N (2026) Congress statistical electoral predictor & LOREG Art. 163 toolkit.",
)


def _parse_strict_cli_iso_date(raw: str, flag_name: str) -> date:
    """Validate an ISO YYYY-MM-DD date CLI option cleanly."""
    try:
        return date.fromisoformat(raw.strip())
    except ValueError as exc:
        raise SchemaValidationError(
            f"Invalid {flag_name} date {raw!r}; expected ISO YYYY-MM-DD"
        ) from exc


def _read_json_input(input_path: Path | None) -> Any:
    """Read allocation JSON payload from file path or stdin."""
    if input_path is None or str(input_path) == "-":
        if sys.stdin.isatty():
            raise SchemaValidationError("Provide --input <path.json> or pipe JSON via stdin.")
        raw_text = sys.stdin.read()
    else:
        if not input_path.is_file():
            raise SchemaValidationError(f"Input file not found: {input_path}")
        raw_text = input_path.read_text(encoding="utf-8")

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"Invalid JSON input: {exc}") from exc


def _render_single_table(console: Console, res: AllocationResult) -> None:
    title = f"LOREG Art. 163 Allocation ({res.cpro})" if res.cpro else "LOREG Art. 163 Allocation"
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Party", style="bold")
    table.add_column("Eligible", justify="center")
    table.add_column("Seats", justify="right")
    elig_set = set(res.eligible_parties)
    ordered = sorted(res.seats.items(), key=lambda kv: (-kv[1], kv[0]))
    for party, seats in ordered:
        table.add_row(party, "yes" if party in elig_set else "no", str(seats))
    console.print(table)
    console.print(
        f"Valid votes: {res.valid_votes} | Cutoff quotient: {res.cutoff_quotient} | "
        f"Ties broken: {res.ties_broken}"
    )


def _render_batch_table(console: Console, batch: BatchAllocationResult) -> None:
    table = Table(
        title=f"National Congress Seats ({len(batch.constituencies)} districts)",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Party", style="bold")
    table.add_column("Seats", justify="right")
    for party, seats in sorted(batch.national_seats.items(), key=lambda kv: (-kv[1], kv[0])):
        if seats > 0:
            table.add_row(party, str(seats))
    console.print(table)
    console.print(
        f"Total seats allocated: {batch.total_seats} | Ties broken: {batch.total_ties_broken}"
    )


@app.command(name="allocate", cls=_WideHelpCommand)
def allocate_command(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="JSON file path or '-' for stdin."),
    ] = None,
    json_output: Annotated[
        bool,
        typer.Option("--json/--table", help="Emit machine-readable JSON or Rich table."),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict/--no-strict",
            help="Error if valid votes yield zero eligible lists.",
        ),
    ] = False,
    quotients: Annotated[
        bool,
        typer.Option("--quotients/--no-quotients", help="Include quotient tables in JSON output."),
    ] = True,
) -> None:
    """Allocate LOREG Art. 163 D'Hondt seats from a single or multi-district JSON payload."""
    try:
        raw_payload = _read_json_input(input_path)
        parsed = parse_allocation_payload(raw_payload)
        if isinstance(parsed, AllocationInput):
            raw_null = raw_payload.get("null_votes", 0) if isinstance(raw_payload, dict) else 0
            res = allocate_dhondt_loreg(
                parsed,
                raise_on_exhaustion=strict,
                include_quotients=quotients,
                null_votes=raw_null,
            )
            if json_output:
                typer.echo(json.dumps(res.to_dict(include_quotients=quotients), indent=2))
            else:
                _render_single_table(Console(), res)
            return

        results: list[AllocationResult] = []
        raw_items = (
            raw_payload.get("constituencies") if isinstance(raw_payload, dict) else None
        )
        for idx, item in enumerate(parsed):
            raw_null = 0
            if (
                isinstance(raw_items, list)
                and idx < len(raw_items)
                and isinstance(raw_items[idx], dict)
            ):
                raw_null = raw_items[idx].get("null_votes", 0)
            results.append(
                allocate_dhondt_loreg(
                    item,
                    raise_on_exhaustion=strict,
                    include_quotients=quotients,
                    null_votes=raw_null,
                )
            )
        batch = BatchAllocationResult.from_results(results)
        if json_output:
            typer.echo(json.dumps(batch.to_dict(), indent=2))
        else:
            _render_batch_table(Console(), batch)
    except (SchemaValidationError, ThresholdExhaustionError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command(name="fetch", cls=_WideHelpCommand)
def fetch_command(
    since: Annotated[
        int,
        typer.Option("--since", help="Earliest general election year to ingest."),
    ] = 2015,
    offline: Annotated[
        bool,
        typer.Option("--offline/--no-offline", help="Validate local snapshots without HTTP calls."),
    ] = False,
    respect_blackout: Annotated[
        bool,
        typer.Option(
            "--respect-blackout/--no-respect-blackout",
            help="Skip poll fetch during LOREG Art. 69.7 blackout window.",
        ),
    ] = True,
    now: Annotated[
        str | None,
        typer.Option("--now", help="Override timestamp (ISO-8601) for blackout testing."),
    ] = None,
) -> None:
    """Ingest election results, Wikipedia polls, and CIS #3584 into data/ snapshots."""
    if respect_blackout and is_in_blackout_window(now):
        typer.echo(BLACKOUT_BANNER)
        return
    try:
        from elecciones.ingest import fetch_all

        summary = fetch_all(since=since, offline=offline)
        console = Console()
        table = Table(
            title="29-N Data Ingestion Summary",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Dataset", style="bold")
        table.add_column("Records / Status", justify="right")
        table.add_row("Constituencies (2026 BOE)", str(summary["constituencies"]))
        table.add_row("Parties (2026 BOE mask)", str(summary["parties"]))
        table.add_row("Congress Elections Verified", "5 x 52 districts (1,750 seats)")
        table.add_row("Europarl 2024 + Regional", "52 + 40 rows")
        table.add_row("Historical Polls (2015–2023)", str(summary["polls_2015_2023"]))
        table.add_row("29-N National Polls", str(summary["polls_2026_effective"]))
        table.add_row("29-N Regional/Provincial Polls", str(summary["regional_polls_2026"]))
        table.add_row("CIS #3584 Provincial (IPF)", str(summary["cis_3584_districts"]))
        b_rep = summary.get("brujula_reconciliation", {})
        if "matched_polls" in b_rep:
            table.add_row(
                "Brújula 29N Cross-Check",
                f"{b_rep['matched_polls']}/{b_rep['brujula_total']} matched",
            )
        console.print(table)
    except (FileNotFoundError, SchemaValidationError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command(name="ratings", cls=_WideHelpCommand)
def ratings_command(
    polls_dir: Annotated[
        Path | None,
        typer.Option("--polls-dir", help="Directory with historical and 2026 poll CSVs."),
    ] = None,
    elections_dir: Annotated[
        Path | None,
        typer.Option("--elections-dir", help="Directory with official election result CSVs."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output CSV path for pollster ratings."),
    ] = None,
) -> None:
    """Compute historical pollster MAE, grades, herding penalty, and house-effect anchors."""
    resolved_polls = polls_dir or get_polls_dir()
    resolved_elections = elections_dir or get_elections_dir()
    resolved_output = output or (get_outputs_dir() / "pollster_ratings.csv")
    try:
        from elecciones.models.pollster_ratings import (
            compute_pollster_ratings,
            save_pollster_ratings,
        )

        ratings = compute_pollster_ratings(
            polls_dir=resolved_polls,
            elections_dir=resolved_elections,
        )
        out_path = save_pollster_ratings(ratings, output_path=resolved_output)
        console = Console()
        table = Table(
            title=f"Pollster Accuracy & House-Effect Scorecard ({len(ratings)} houses)",
            show_header=True,
            header_style="bold cyan",
        )
        table.add_column("Pollster", style="bold")
        table.add_column("Grade", justify="center")
        table.add_column("Elec.", justify="right")
        table.add_column("2026", justify="right")
        table.add_column("Shrunk MAE", justify="right")
        table.add_column("Wt", justify="right")
        table.add_column("Herd.", justify="right")
        table.add_column("dPP", justify="right")
        table.add_column("dPSOE", justify="right")
        table.add_column("dVOX", justify="right")
        for r in ratings[:20]:
            table.add_row(
                r.pollster,
                r.grade,
                str(r.n_elections),
                str(r.n_polls_2026),
                f"{r.shrunk_mae:.2f}",
                f"{r.grade_weight:.2f}",
                f"{r.herding_penalty:.2f}",
                f"{r.delta_PP:+.2f}",
                f"{r.delta_PSOE:+.2f}",
                f"{r.delta_VOX:+.2f}",
            )
        console.print(table)
        console.print(f"Saved pollster ratings -> {out_path}")
    except (FileNotFoundError, SchemaValidationError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


@app.command(name="backtest", cls=_WideHelpCommand)
def backtest_command(
    horizons: Annotated[
        str,
        typer.Option("--horizons", help="Comma-separated pre-election horizons in days."),
    ] = "15,6,1",
    n_sims: Annotated[
        int,
        typer.Option("--n-sims", help="Monte Carlo draws per race in backtest."),
    ] = 250,
    seed: Annotated[
        int,
        typer.Option("--seed", help="PRNG seed for backtest reproducibility."),
    ] = SIMULATION_SEED,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON path for backtest_summary.json."),
    ] = None,
) -> None:
    """Run leave-one-election-out calibration across 2015-D through 2023-J."""
    resolved_output = output or (get_outputs_dir() / "backtest_summary.json")
    _ = (horizons, n_sims, seed, resolved_output)
    typer.echo(
        "Error: LOOCV backtester (`elecciones.backtest.runner`) is scheduled for Milestone M4.",
        err=True,
    )
    raise typer.Exit(code=1)


@app.command(name="forecast", cls=_WideHelpCommand)
def forecast_command(
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="Cutoff date (YYYY-MM-DD) for point-in-time forecast."),
    ] = None,
    engine: Annotated[
        str,
        typer.Option("--engine", help="Stage 1 engine: ensemble, kernel, or state_space."),
    ] = "ensemble",
    exclude_cis_headline: Annotated[
        bool,
        typer.Option(
            "--exclude-cis-headline/--no-exclude-cis-headline",
            help="Exclude CIS from Stage 1 national headline aggregation.",
        ),
    ] = False,
    n_sims: Annotated[
        int,
        typer.Option("--n-sims", help="Number of Monte Carlo election simulations."),
    ] = DEFAULT_N_SIMS,
    seed: Annotated[
        int,
        typer.Option("--seed", help="PRNG seed for Monte Carlo simulation."),
    ] = SIMULATION_SEED,
    manual_polls: Annotated[
        Path | None,
        typer.Option("--manual-polls", help="Optional override path for manual_polls_2026.csv."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output JSON path for forecast_29n.json."),
    ] = None,
) -> None:
    """Run Stages 1-4 to generate forecast_29n.json across all 52 constituencies."""
    resolved_output = output or (get_outputs_dir() / "forecast_29n.json")
    if as_of is not None:
        try:
            _parse_strict_cli_iso_date(as_of, "--as-of")
        except SchemaValidationError as exc:
            typer.echo(f"Error: {exc}", err=True)
            raise typer.Exit(code=1) from exc
    _ = (as_of, engine, exclude_cis_headline, n_sims, seed, manual_polls, resolved_output)
    typer.echo(
        "Error: Forecast simulator (`elecciones.models.simulator`) is scheduled for Milestone M4.",
        err=True,
    )
    raise typer.Exit(code=1)


@app.command(name="dashboard", cls=_WideHelpCommand)
def dashboard_command(
    input_path: Annotated[
        Path | None,
        typer.Option("--input", "-i", help="Path to forecast_29n.json."),
    ] = None,
    backtest_path: Annotated[
        Path | None,
        typer.Option("--backtest", help="Path to backtest_summary.json."),
    ] = None,
    ratings_path: Annotated[
        Path | None,
        typer.Option("--ratings", help="Path to pollster_ratings.csv."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Output HTML path for standalone dashboard."),
    ] = None,
) -> None:
    """Compile zero-backend interactive HTML editorial dashboard into docs/index.html."""
    resolved_input = input_path or (get_outputs_dir() / "forecast_29n.json")
    resolved_backtest = backtest_path or (get_outputs_dir() / "backtest_summary.json")
    resolved_ratings = ratings_path or (get_outputs_dir() / "pollster_ratings.csv")
    resolved_output = output or (get_docs_dir() / "index.html")
    _ = (resolved_input, resolved_backtest, resolved_ratings, resolved_output)
    typer.echo(
        "Error: Dashboard compiler (`elecciones.viz.dashboard`) is scheduled for Milestone M5.",
        err=True,
    )
    raise typer.Exit(code=1)


@app.command(name="run-all", cls=_WideHelpCommand)
def run_all_command(
    offline: Annotated[
        bool,
        typer.Option("--offline/--no-offline", help="Use cached/committed CSV snapshots."),
    ] = False,
    respect_blackout: Annotated[
        bool,
        typer.Option(
            "--respect-blackout/--no-respect-blackout",
            help="Skip live poll fetch during LOREG Art. 69.7 blackout window.",
        ),
    ] = True,
    now: Annotated[
        str | None,
        typer.Option("--now", help="Override timestamp (ISO-8601) for blackout testing."),
    ] = None,
    as_of: Annotated[
        str | None,
        typer.Option("--as-of", help="Cutoff date (YYYY-MM-DD) for point-in-time forecast."),
    ] = None,
    engine: Annotated[
        str,
        typer.Option("--engine", help="Stage 1 engine: ensemble, kernel, or state_space."),
    ] = "ensemble",
    exclude_cis_headline: Annotated[
        bool,
        typer.Option(
            "--exclude-cis-headline/--no-exclude-cis-headline",
            help="Exclude CIS from Stage 1 national headline aggregation.",
        ),
    ] = False,
    n_sims: Annotated[
        int,
        typer.Option("--n-sims", help="Number of Monte Carlo election simulations."),
    ] = DEFAULT_N_SIMS,
    seed: Annotated[
        int,
        typer.Option("--seed", help="PRNG seed for Monte Carlo simulation."),
    ] = SIMULATION_SEED,
) -> None:
    """Execute full pipeline: fetch -> ratings -> backtest -> forecast -> dashboard."""
    if respect_blackout and is_in_blackout_window(now):
        typer.echo(BLACKOUT_BANNER)
        return
    _ = (offline, as_of, engine, exclude_cis_headline, n_sims, seed)
    typer.echo(
        "Error: End-to-end `run-all` pipeline wiring is scheduled for Milestone M5.",
        err=True,
    )
    raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
