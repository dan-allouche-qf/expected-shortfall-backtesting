"""Smoke tests for the figure library.

We don't compare images byte-by-byte (overkill for P1); we just verify
that each figure builder runs end-to-end on small synthetic inputs and
produces a non-empty matplotlib :class:`Figure`.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esback.backtests.base import BacktestResult
from esback.backtests.du_escanciano import run_battery
from esback.viz import figures
from esback.viz.style import PALETTE, apply_rcparams


def _toy_oos(n: int = 200) -> dict:
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    returns = rng.standard_normal(n) * 1.5
    var_t = np.full(n, 2.0)
    es_t = np.full(n, 2.5)
    u_t = rng.uniform(size=n)
    h_t = (u_t <= 0.05).astype(float)
    H_t = (1 / 0.10) * (0.10 - u_t) * (u_t <= 0.10)
    return {
        "index": index,
        "returns": returns,
        "var_t": var_t,
        "es_t": es_t,
        "u_t": u_t,
        "h_t": h_t,
        "H_t": H_t,
    }


def test_palette_has_required_colors() -> None:
    for key in ("blue", "red", "purple", "green", "orange", "darkblue"):
        assert key in PALETTE


def test_apply_rcparams_is_idempotent() -> None:
    apply_rcparams()
    apply_rcparams()
    assert plt.rcParams["axes.titleweight"] == "bold"


def test_plot_loss_distribution_smoke() -> None:
    d = _toy_oos()
    fig = figures.plot_loss_distribution(
        d["returns"], d["var_t"], d["es_t"], "Toy", alpha_var=0.05, alpha_es=0.10
    )
    assert fig is not None
    plt.close(fig)


def test_plot_loss_distribution_legend_reflects_alpha() -> None:
    """The legend must display the alpha values actually passed in
    (Basel alpha=0.025 here), with no leftover 5% / 10% literal."""
    d = _toy_oos()
    fig = figures.plot_loss_distribution(
        d["returns"], d["var_t"], d["es_t"], "Toy", alpha_var=0.025, alpha_es=0.025
    )
    legend_texts = " | ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert "2.5%" in legend_texts, f"Expected '2.5%' in legend, got: {legend_texts}"
    assert "5%" not in legend_texts.replace("2.5%", ""), (
        f"Unexpected '5%' literal in legend: {legend_texts}"
    )
    plt.close(fig)


def test_plot_violations_timeline_smoke() -> None:
    d = _toy_oos()
    fig = figures.plot_violations_timeline(d["index"], d["h_t"], d["H_t"], 0.05, 0.10, "Toy")
    assert fig is not None
    plt.close(fig)


def test_plot_returns_var_es_with_zoom_smoke() -> None:
    d = _toy_oos()
    fig = figures.plot_returns_var_es(
        d["index"],
        d["returns"],
        d["var_t"],
        d["es_t"],
        "Toy",
        zoom_date="2020-04-01",
        zoom_label="Demo zoom",
        alpha_var=0.05,
        alpha_es=0.10,
    )
    assert fig is not None
    plt.close(fig)


def test_plot_returns_var_es_legend_reflects_alpha() -> None:
    """The legend must display the alpha values actually passed in. With the
    AS/MF convention (alpha_var = alpha_es) and Basel-flavoured alpha=0.025 / 0.05,
    both values must surface in the legend."""
    d = _toy_oos()
    fig = figures.plot_returns_var_es(
        d["index"],
        d["returns"],
        d["var_t"],
        d["es_t"],
        "Toy",
        alpha_var=0.025,
        alpha_es=0.05,
    )
    legend_texts = " | ".join(t.get_text() for t in fig.axes[0].get_legend().get_texts())
    assert "0.025" in legend_texts, f"Expected '0.025' in legend, got: {legend_texts}"
    assert "0.05" in legend_texts, f"Expected '0.05' in legend, got: {legend_texts}"
    plt.close(fig)


def test_plot_acf_comparison_smoke() -> None:
    d = _toy_oos()
    fig = figures.plot_acf_comparison(d["H_t"], d["h_t"], 0.10, 0.05, "Toy", max_lag=10)
    assert fig is not None
    plt.close(fig)


def test_plot_results_table_smoke() -> None:
    d = _toy_oos()
    battery = run_battery(d["u_t"])
    fig = figures.plot_results_table({"P1": battery, "P2": battery})
    assert fig is not None
    plt.close(fig)


def test_plot_numerical_example_smoke() -> None:
    fig = figures.plot_numerical_example()
    assert fig is not None
    plt.close(fig)


def test_plot_pit_diagnostic_smoke() -> None:
    d = _toy_oos()
    fig = figures.plot_pit_diagnostic(d["u_t"], "Toy")
    assert fig is not None
    plt.close(fig)


def test_plot_lambda_diagram_smoke() -> None:
    fig = figures.plot_lambda_diagram()
    assert fig is not None
    plt.close(fig)


def test_save_to_writes_file(tmp_path: Path) -> None:
    fig = figures.plot_numerical_example(save_to=tmp_path / "fig.png")
    assert (tmp_path / "fig.png").exists()
    plt.close(fig)


def test_save_to_creates_parent_dir(tmp_path: Path) -> None:
    out = tmp_path / "deep" / "nested" / "out.pdf"
    fig = figures.plot_numerical_example(save_to=out)
    assert out.exists()
    plt.close(fig)


def test_battery_to_results_table_with_real_BacktestResult() -> None:
    """Confirm BacktestResult flows through plot_results_table without surprises."""
    battery = {
        "U_ES": BacktestResult("U_ES", 1.5, 0.13, 0.10, "ref"),
        "C_ES(5)": BacktestResult("C_ES(5)", 4.2, 0.04, 0.10, "ref"),
    }
    fig = figures.plot_results_table({"Period A": battery})
    assert fig is not None
    plt.close(fig)


def test_plot_p_value_strip_series_smoke() -> None:
    p_values = pd.Series(
        {"U_ES": 0.001, "C_ES(5)": 0.04, "AS_Z1": 0.22, "AS_Z2": 0.0002, "McNeil_Frey": 0.28},
        name="p_value",
    )
    fig = figures.plot_p_value_strip(p_values, title="Toy")
    assert fig is not None
    plt.close(fig)


def test_plot_p_value_strip_dataframe_smoke() -> None:
    df = pd.DataFrame(
        {
            "GFC": [0.001, 0.04, 0.22, 0.0002, 0.28],
            "COVID": [0.02, 0.10, 0.10, 0.001, 0.17],
        },
        index=["U_ES", "C_ES(5)", "AS_Z1", "AS_Z2", "McNeil_Frey"],
    )
    fig = figures.plot_p_value_strip(df, title="Toy 2 scenarios")
    assert fig is not None
    plt.close(fig)


def test_plot_p_value_strip_handles_nan() -> None:
    p_values = pd.Series({"U_ES": 0.001, "AS_Z1": float("nan"), "McNeil_Frey": 0.28})
    fig = figures.plot_p_value_strip(p_values, title="Toy with NaN")
    assert fig is not None
    plt.close(fig)


def test_plot_size_vs_nominal_smoke() -> None:
    rng = np.random.default_rng(0)
    sizes = [250, 500, 1000, 2500]
    tests = ["U_ES", "Kupiec_POF", "C_ES(5)"]
    rows = [
        {"n": n, "test": t, "rejection_rate": float(rng.uniform(0.04, 0.12))}
        for t in tests
        for n in sizes
    ]
    fig = figures.plot_size_vs_nominal(pd.DataFrame(rows))
    assert fig is not None
    plt.close(fig)
