"""Command-line interface.

Entry point for ``esback`` declared in ``pyproject.toml`` under
``[project.scripts]``. The single command ``replicate`` reads a YAML
config and reproduces the Du-Escanciano (2017) replication, writing
figures and result tables under the directories declared in the config.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Annotated, Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import typer
import yaml

from esback._logging import get_logger
from esback.backtests.battery import run_full_battery
from esback.backtests.es_baselines import StandardizedTSimulator
from esback.data.loaders import load_sp500
from esback.multi_asset import (
    AssetSpec,
    PeriodSpec,
    pivot_p_values,
    reject_rate_per_test,
    run_grid,
)
from esback.pipeline import PeriodArtifacts, run_period
from esback.reporting.tables import battery_summary, comparison_table
from esback.risk_measures import es_t as _compute_es_t
from esback.risk_measures import var_t as _compute_var_t
from esback.viz import figures, style

app = typer.Typer(
    help="esback - Backtesting Expected Shortfall",
    add_completion=False,
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Top-level callback so Typer always expects a sub-command."""


def _slug(name: str) -> str:
    """Lowercase, alphanumeric-and-underscore-only key from a period name."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s or "period"


def _write_figures_for_period(
    artifacts: PeriodArtifacts,
    figures_dir: Path,
    slug: str,
    zoom_date: str | None,
    zoom_label: str | None,
    alpha_es: float,
    alpha_var: float,
) -> None:
    plt.close("all")
    figures.plot_loss_distribution(
        artifacts.forecast.oos_returns,
        artifacts.var_t,
        artifacts.es_t,
        label=artifacts.label,
        save_to=figures_dir / f"{slug}_loss_dist.pdf",
        alpha_var=alpha_var,
        alpha_es=alpha_es,
    )
    figures.plot_violations_timeline(
        artifacts.forecast.oos_index,
        artifacts.h_t,
        artifacts.H_t,
        alpha_var=alpha_var,
        alpha_es=alpha_es,
        label=artifacts.label,
        save_to=figures_dir / f"{slug}_violations.pdf",
    )
    figures.plot_returns_var_es(
        artifacts.forecast.oos_index,
        artifacts.forecast.oos_returns,
        artifacts.var_t,
        artifacts.es_t,
        label=artifacts.label,
        zoom_date=zoom_date,
        zoom_label=zoom_label,
        save_to=figures_dir / f"{slug}_returns_var_es.pdf",
        alpha_var=alpha_var,
        alpha_es=alpha_es,
    )
    figures.plot_acf_comparison(
        artifacts.H_t,
        artifacts.h_t,
        alpha_es=alpha_es,
        alpha_var=alpha_var,
        label=artifacts.label,
        save_to=figures_dir / f"{slug}_acf.pdf",
    )
    figures.plot_pit_diagnostic(
        artifacts.u_t,
        label=artifacts.label,
        save_to=figures_dir / f"{slug}_pit.pdf",
    )
    plt.close("all")


@app.command()
def replicate(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            readable=True,
            help="Path to the replication YAML config.",
        ),
    ],
) -> None:
    """Run the Du-Escanciano replication on the periods declared in the config."""
    log = get_logger("esback.cli")
    cfg: dict[str, Any] = yaml.safe_load(config.read_text())

    style.apply_rcparams()

    asset = cfg["data"]["asset"]
    start = cfg["data"]["start"]
    end = cfg["data"]["end"]

    if asset != "sp500":
        raise typer.BadParameter(
            f"Asset '{asset}' is not supported by the replicate command; use multi-asset."
        )

    log.info("Loading %s data from %s to %s", asset, start, end)
    data = load_sp500(start=start, end=end)
    log.info("  %d observations available", len(data))

    alpha_es = float(cfg["backtest"]["alpha_es"])
    alpha_var = float(cfg["backtest"]["alpha_var"])
    m = int(cfg["backtest"]["m"])

    figures_dir = Path(cfg["output"]["figures_dir"])
    tables_dir = Path(cfg["output"]["tables_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    period_batteries: dict[str, dict] = {}
    for period_cfg in cfg["periods"]:
        log.info("Running period: %s", period_cfg["name"])
        artifacts = run_period(
            data,
            is_start=period_cfg["is_start"],
            is_end=period_cfg["is_end"],
            oos_end=period_cfg["oos_end"],
            label=period_cfg["name"],
            alpha_es=alpha_es,
            alpha_var=alpha_var,
            m=m,
        )

        slug = _slug(period_cfg["name"])
        _write_figures_for_period(
            artifacts,
            figures_dir,
            slug,
            zoom_date=period_cfg.get("figure_zoom_date"),
            zoom_label=period_cfg.get("figure_zoom_label"),
            alpha_es=alpha_es,
            alpha_var=alpha_var,
        )

        battery_summary(artifacts.battery).to_csv(tables_dir / f"{slug}_basic.csv")
        battery_summary(artifacts.robust).to_csv(tables_dir / f"{slug}_robust.csv")

        # Full battery: re-evaluate VaR and ES at the same alpha so the
        # Acerbi-Szekely / McNeil-Frey statistics live under their
        # standard convention (alpha_VaR == alpha_ES).
        var_at_alpha_var = _compute_var_t(
            artifacts.forecast.mu_oos,
            artifacts.forecast.sigma_oos,
            alpha=alpha_var,
            nu=artifacts.forecast.nu,
        )
        es_at_alpha_var = _compute_es_t(
            artifacts.forecast.mu_oos,
            artifacts.forecast.sigma_oos,
            alpha=alpha_var,
            nu=artifacts.forecast.nu,
        )
        es_simulator = StandardizedTSimulator(
            mu_oos=artifacts.forecast.mu_oos,
            sigma_oos=artifacts.forecast.sigma_oos,
            nu=artifacts.forecast.nu,
        )
        full = run_full_battery(
            u_t=artifacts.u_t,
            oos_returns=artifacts.forecast.oos_returns,
            var_t=var_at_alpha_var,
            es_t=es_at_alpha_var,
            sigma_oos=artifacts.forecast.sigma_oos,
            h_t=artifacts.h_t,
            alpha_es=alpha_es,
            alpha_var=alpha_var,
            m=m,
            fit=artifacts.fit,
            returns_all=artifacts.returns_all,
            is_end_idx=artifacts.is_end_idx,
            oos_end_idx=artifacts.oos_end_idx,
            es_simulator=es_simulator,
        )
        battery_summary(full).to_csv(tables_dir / f"{slug}_full_battery.csv")

        period_batteries[period_cfg["name"]] = artifacts.battery
        log.info(
            "  Done: %d basic, %d robust, %d full-battery results",
            len(artifacts.battery),
            len(artifacts.robust),
            len(full),
        )

    # Cross-period summary
    comparison_table(period_batteries).to_csv(tables_dir / "comparison.csv")

    # Pedagogical figures (no period dependency)
    figures.plot_numerical_example(save_to=figures_dir / "numerical_example.pdf")
    figures.plot_lambda_diagram(save_to=figures_dir / "lambda_diagram.pdf")
    figures.plot_results_table(period_batteries, save_to=figures_dir / "results_table.pdf")
    plt.close("all")

    log.info("Wrote figures to %s and tables to %s", figures_dir, tables_dir)


@app.command("multi-asset")
def multi_asset(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            readable=True,
            help="Path to the multi-asset YAML config.",
        ),
    ],
) -> None:
    """Run the unified battery on a grid of (asset, crisis-period) combinations."""
    log = get_logger("esback.cli")
    cfg: dict[str, Any] = yaml.safe_load(config.read_text())

    style.apply_rcparams()

    assets = [AssetSpec(name=a["name"], ticker=a["ticker"]) for a in cfg["assets"]]
    periods = [
        PeriodSpec(name=p["name"], oos_start=p["oos_start"], oos_end=p["oos_end"])
        for p in cfg["periods"]
    ]
    out_dir = Path(cfg["output"]["results_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running %d assets x %d periods grid", len(assets), len(periods))
    grid = run_grid(
        assets,
        periods,
        data_start=cfg["data"]["start"],
        data_end=cfg["data"]["end"],
        in_sample_years=int(cfg.get("in_sample_years", 5)),
        alpha_es=float(cfg["backtest"]["alpha_es"]),
        alpha_var=float(cfg["backtest"]["alpha_var"]),
        m=int(cfg["backtest"]["m"]),
    )

    grid.to_parquet(out_dir / "grid.parquet")
    grid.to_csv(out_dir / "grid.csv", index=False)
    log.info("Wrote %d rows to %s", len(grid), out_dir / "grid.parquet")

    if grid.empty:
        log.warning("Empty grid; nothing to plot.")
        return

    reject_rate_per_test(grid).to_csv(out_dir / "reject_rate_per_test.csv")
    plt.close("all")
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    for period in periods:
        pivot = pivot_p_values(grid, period.name)
        if pivot.empty:
            continue
        figures.plot_p_value_heatmap(
            pivot,
            title=f"{period.name}: p-values across assets",
            save_to=figures_dir / f"heatmap_{period.name}.pdf",
        )
    plt.close("all")
    log.info("Wrote heatmaps to %s", figures_dir)
