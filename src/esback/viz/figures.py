"""Figure library for the Du-Escanciano replication.

Each function returns a matplotlib :class:`Figure` and optionally writes
it to disk. Inputs are explicit data arrays so the figures can be
regenerated from any backtest outcome.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from scipy.stats import gaussian_kde

from esback.backtests.base import BacktestResult
from esback.pit import ks_test_uniform
from esback.viz.style import PALETTE


def _save(fig: Figure, save_to: Path | None) -> None:
    if save_to is not None:
        save_to.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, bbox_inches="tight")


def plot_loss_distribution(
    oos_returns: np.ndarray,
    var_t: np.ndarray,
    es_t: np.ndarray,
    label: str,
    save_to: Path | None = None,
    *,
    alpha_var: float,
    alpha_es: float,
) -> Figure:
    """Histogram of OOS losses overlaid with median VaR and ES.

    Parameters
    ----------
    alpha_var, alpha_es
        Risk levels at which ``var_t`` and ``es_t`` were computed.
        Required keyword arguments (no default) so the legend cannot
        silently misreport the convention used by the caller (e.g.
        Basel alpha=0.025 vs Du-Escanciano alpha_ES=2*alpha_VaR vs
        AS/MF alpha_ES=alpha_VaR).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    losses = -np.asarray(oos_returns)

    ax.hist(
        losses,
        bins=80,
        density=True,
        color=PALETTE["blue"],
        alpha=0.6,
        edgecolor="white",
        linewidth=0.3,
        label="Loss distribution $(-Y_t)$",
    )

    med_var = float(np.median(var_t))
    med_es = float(np.median(es_t))

    ax.axvline(
        med_var,
        color=PALETTE["red"],
        linewidth=2.5,
        linestyle="--",
        label=f"Median VaR({alpha_var:.1%}) = {med_var:.2f}%",
    )
    ax.axvline(
        med_es,
        color=PALETTE["purple"],
        linewidth=2.5,
        linestyle="-",
        label=f"Median ES({alpha_es:.1%}) = {med_es:.2f}%",
    )

    x_fill = np.linspace(med_var, losses.max() * 1.05, 200)
    kde = gaussian_kde(losses)
    ax.fill_between(x_fill, kde(x_fill), alpha=0.3, color=PALETTE["red"], label="Tail beyond VaR")

    ax.set_xlabel("Loss $(-Y_t)$ in %")
    ax.set_ylabel("Density")
    ax.set_title(f"Loss Distribution - {label}")
    ax.legend(loc="upper right")
    ax.set_xlim(-6, losses.max() * 1.1)
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_violations_timeline(
    index: pd.Index,
    h_t: np.ndarray,
    H_t: np.ndarray,
    alpha_var: float,
    alpha_es: float,
    label: str,
    save_to: Path | None = None,
) -> Figure:
    """Side-by-side: binary VaR violations on top, continuous ES severities below."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    ax1.bar(index, h_t, width=2, color=PALETTE["red"], alpha=0.8)
    ax1.set_ylabel("$h_t(\\alpha)$")
    ax1.set_title(f"VaR Violations (binary, alpha={alpha_var:.2f}) - {label}")
    ax1.set_ylim(0, 1.3)
    ax1.axhline(alpha_var, color=PALETTE["gray"], linestyle="--", linewidth=1)
    n_viol = int(h_t.sum())
    ax1.text(
        0.02,
        0.85,
        f"{n_viol} violations out of {len(h_t)}",
        transform=ax1.transAxes,
        fontsize=10,
        bbox={
            "boxstyle": "round",
            "facecolor": PALETTE["lightorange"],
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    ax2.bar(index, H_t, width=2, color=PALETTE["purple"], alpha=0.8)
    ax2.set_ylabel("$H_t(\\alpha)$")
    ax2.set_title(f"Cumulative Violations (continuous, alpha={alpha_es:.2f}) - {label}")
    ax2.set_xlabel("Date")
    ax2.axhline(alpha_es / 2, color=PALETTE["gray"], linestyle="--", linewidth=1)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_returns_var_es(
    index: pd.Index,
    returns: np.ndarray,
    var_t: np.ndarray,
    es_t: np.ndarray,
    label: str,
    zoom_date: str | None = None,
    zoom_label: str | None = None,
    save_to: Path | None = None,
    *,
    alpha_var: float,
    alpha_es: float,
) -> Figure:
    """Time series of returns with the negative VaR and ES envelopes overlaid.

    Parameters
    ----------
    alpha_var, alpha_es
        Risk levels at which ``var_t`` and ``es_t`` were computed.
        Required keyword arguments (no default) so the legend cannot
        silently misreport the convention used by the caller (e.g.
        Basel alpha=0.025 vs Du-Escanciano alpha_ES=2*alpha_VaR vs
        AS/MF alpha_ES=alpha_VaR).
    """
    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.plot(
        index, returns, color=PALETTE["darkblue"], linewidth=0.6, alpha=0.8, label="Return $Y_t$"
    )
    ax.plot(
        index,
        -var_t,
        color=PALETTE["red"],
        linewidth=1.2,
        linestyle="--",
        label=f"$-VaR_t({alpha_var:g})$",
        alpha=0.9,
    )
    ax.plot(
        index,
        -es_t,
        color=PALETTE["purple"],
        linewidth=1.2,
        linestyle="-",
        label=f"$-ES_t({alpha_es:g})$",
        alpha=0.9,
    )
    ax.axhline(0, color=PALETTE["gray"], linewidth=0.5)

    if zoom_date and zoom_label:
        target = pd.Timestamp(zoom_date)
        closest_idx = int(np.argmin(np.abs(index - target)))
        x_pt = index[closest_idx]
        y_pt = float(returns[closest_idx])
        ax.annotate(
            f"{zoom_label}\n$Y_t$ = {y_pt:.2f}%\nVaR = {var_t[closest_idx]:.2f}%\nES = {es_t[closest_idx]:.2f}%",
            xy=(x_pt, y_pt),
            xytext=(x_pt + pd.Timedelta(days=60), y_pt + 2.5),
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": PALETTE["darkblue"], "lw": 1.5},
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": PALETTE["lightorange"],
                "edgecolor": "none",
                "alpha": 0.95,
            },
        )
        ax.plot(x_pt, y_pt, marker="o", color=PALETTE["darkblue"], markersize=6)

    ax.set_xlabel("Date")
    ax.set_ylabel("Return (%)")
    ax.set_title(f"Returns vs. Risk Measures - {label}")
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    fig.tight_layout()
    _save(fig, save_to)
    return fig


def _sample_acf(x: np.ndarray, mean_val: float, max_lag: int) -> np.ndarray:
    centered = x - mean_val
    n = len(x)
    gamma_0 = float(np.mean(centered**2))
    acf = np.zeros(max_lag)
    for j in range(1, max_lag + 1):
        gamma_j = float(np.sum(centered[j:] * centered[:-j]) / n)
        acf[j - 1] = gamma_j / gamma_0 if gamma_0 > 0 else 0.0
    return acf


def plot_acf_comparison(
    H_t: np.ndarray,
    h_t: np.ndarray,
    alpha_es: float,
    alpha_var: float,
    label: str,
    max_lag: int = 12,
    save_to: Path | None = None,
) -> Figure:
    """Sample autocorrelations of cumulative versus binary violations side by side."""
    n = len(H_t)
    acf_H = _sample_acf(H_t, alpha_es / 2, max_lag)
    acf_h = _sample_acf(h_t, alpha_var, max_lag)

    ci = 1.96 / np.sqrt(n)
    lags = np.arange(1, max_lag + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    colors_H = [PALETTE["red"] if abs(r) > ci else PALETTE["blue"] for r in acf_H]
    ax1.bar(lags, acf_H, color=colors_H, edgecolor="white", width=0.6)
    ax1.axhline(ci, color=PALETTE["gray"], linestyle="--", linewidth=1)
    ax1.axhline(-ci, color=PALETTE["gray"], linestyle="--", linewidth=1)
    ax1.axhline(0, color=PALETTE["darkblue"], linewidth=0.5)
    ax1.set_xlabel("Lag")
    ax1.set_ylabel("Autocorrelation")
    ax1.set_title("ACF of $H_t$ - ES cumulative violations")
    ax1.set_xticks(lags)
    sig_H = sum(1 for r in acf_H if abs(r) > ci)
    ax1.text(
        0.02,
        0.92,
        f"{sig_H}/{max_lag} significant lags",
        transform=ax1.transAxes,
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": PALETTE["lightblue"],
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    colors_h = [PALETTE["red"] if abs(r) > ci else PALETTE["blue"] for r in acf_h]
    ax2.bar(lags, acf_h, color=colors_h, edgecolor="white", width=0.6)
    ax2.axhline(ci, color=PALETTE["gray"], linestyle="--", linewidth=1)
    ax2.axhline(-ci, color=PALETTE["gray"], linestyle="--", linewidth=1)
    ax2.axhline(0, color=PALETTE["darkblue"], linewidth=0.5)
    ax2.set_xlabel("Lag")
    ax2.set_ylabel("Autocorrelation")
    ax2.set_title("ACF of $h_t$ - VaR violations")
    ax2.set_xticks(lags)
    sig_h = sum(1 for r in acf_h if abs(r) > ci)
    ax2.text(
        0.02,
        0.92,
        f"{sig_h}/{max_lag} significant lags",
        transform=ax2.transAxes,
        fontsize=9,
        bbox={
            "boxstyle": "round",
            "facecolor": PALETTE["lightblue"],
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    fig.suptitle(f"Autocorrelation Comparison - {label}", fontsize=13, y=1.02)
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_results_table(
    period_batteries: Mapping[str, Mapping[str, BacktestResult]],
    save_to: Path | None = None,
) -> Figure:
    """Render the basic-tests result table as a matplotlib figure for slides or reports."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")

    period_names = list(period_batteries.keys())
    test_names = list(next(iter(period_batteries.values())).keys())
    col_labels = ["Test", *(["Statistic", "p-value", "Decision"] * len(period_names))]

    rows = []
    for tname in test_names:
        row: list[str] = [tname]
        for pname in period_names:
            res = period_batteries[pname][tname]
            row.extend([f"{res.statistic:.3f}", f"{res.p_value:.4f}", res.decision()])
        rows.append(row)

    table = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for j in range(len(col_labels)):
        table[0, j].set_facecolor(PALETTE["darkblue"])
        table[0, j].set_text_props(color="white", fontweight="bold")

    decision_cols = [3 + 3 * k for k in range(len(period_names))]
    for i in range(1, len(rows) + 1):
        for j in decision_cols:
            cell = table[i, j]
            if cell.get_text().get_text().strip() == "REJECT":
                cell.set_facecolor(PALETTE["lightred"])
                cell.set_text_props(color=PALETTE["red"], fontweight="bold")
            else:
                cell.set_facecolor(PALETTE["lightgreen"])

    width_per_period = 1.0 / (len(period_names) + 1)
    for k, pname in enumerate(period_names):
        ax.text(
            (k + 1) * width_per_period + width_per_period / 2,
            0.95,
            pname,
            fontsize=11,
            fontweight="bold",
            transform=ax.transAxes,
            ha="center",
            color=PALETTE["darkblue"],
        )

    ax.set_title("Du-Escanciano (2017) Backtest Results", fontsize=13, fontweight="bold", pad=30)
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_numerical_example(save_to: Path | None = None) -> Figure:
    """Pedagogical six-day example contrasting binary and cumulative violations."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    alpha = 0.10
    u_example = np.array([0.52, 0.08, 0.35, 0.03, 0.71, 0.06])
    days = [f"Day {i + 1}" for i in range(len(u_example))]

    h = (u_example <= alpha).astype(float)
    colors_h = [PALETTE["red"] if v == 1 else PALETTE["gray"] for v in h]
    ax1.bar(days, h, color=colors_h, edgecolor="white", width=0.5)
    ax1.set_title("VaR Violations $h_t$ (binary)", fontsize=11)
    ax1.set_ylim(0, 1.3)
    ax1.set_ylabel("$h_t$")
    for i, (v, u) in enumerate(zip(h, u_example, strict=True)):
        ax1.text(i, v + 0.05, f"$u_t$={u:.2f}", ha="center", fontsize=8)

    H = (1 / alpha) * (alpha - u_example) * (u_example <= alpha)
    colors_H = [PALETTE["purple"] if v > 0 else PALETTE["gray"] for v in H]
    ax2.bar(days, H, color=colors_H, edgecolor="white", width=0.5)
    ax2.set_title("Cumulative Violations $H_t$ (continuous)", fontsize=11)
    ax2.set_ylabel("$H_t$")
    for i, (v, u) in enumerate(zip(H, u_example, strict=True)):
        if v > 0:
            ax2.text(i, v + 0.02, f"$H_t$={v:.2f}\n($u_t$={u:.2f})", ha="center", fontsize=8)
        else:
            ax2.text(i, v + 0.02, f"$u_t$={u:.2f}\n(no viol.)", ha="center", fontsize=8)

    ax2.axhline(
        alpha / 2,
        color=PALETTE["gray"],
        linestyle="--",
        linewidth=1,
        label=f"$E[H_t] = \\alpha/2 = {alpha / 2:.2f}$",
    )
    ax2.legend(loc="upper right")

    fig.suptitle(f"Numerical Example (alpha={alpha}): VaR vs. Cumulative Violations", y=1.02)
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_pit_diagnostic(u_t: np.ndarray, label: str, save_to: Path | None = None) -> Figure:
    """QQ-plot and histogram of PIT residuals, with the KS-uniform p-value."""
    n = len(u_t)
    _, ks_pval = ks_test_uniform(u_t)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    u_sorted = np.sort(u_t)
    theoretical_q = np.linspace(1 / (2 * n), 1 - 1 / (2 * n), n)
    ax1.scatter(theoretical_q, u_sorted, s=8, alpha=0.5, color=PALETTE["blue"], edgecolors="none")
    ax1.plot(
        [0, 1],
        [0, 1],
        color=PALETTE["red"],
        linestyle="--",
        linewidth=1.5,
        label="45-deg line (ideal)",
    )
    ax1.set_xlabel("Uniform(0,1) Theoretical Quantiles")
    ax1.set_ylabel("Empirical PIT Quantiles ($u_t$)")
    ax1.set_title("QQ-Plot: PIT vs. Uniform(0,1)")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.legend(loc="upper left")
    ax1.text(
        0.95,
        0.05,
        f"KS p-value = {ks_pval:.4f}",
        transform=ax1.transAxes,
        fontsize=10,
        ha="right",
        bbox={
            "boxstyle": "round",
            "facecolor": PALETTE["lightblue"],
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    ax2.hist(
        u_t,
        bins=30,
        density=True,
        color=PALETTE["blue"],
        alpha=0.6,
        edgecolor="white",
        linewidth=0.3,
        label="PIT histogram",
    )
    ax2.axhline(1.0, color=PALETTE["red"], linewidth=2, linestyle="--", label="U(0,1) density")
    ax2.set_xlabel("$u_t$")
    ax2.set_ylabel("Density")
    ax2.set_title("Histogram of PIT Residuals")
    ax2.set_xlim(0, 1)
    ax2.legend(loc="upper left")
    ax2.text(
        0.98,
        0.95,
        f"n = {n}",
        transform=ax2.transAxes,
        fontsize=10,
        ha="right",
        va="top",
        bbox={
            "boxstyle": "round",
            "facecolor": PALETTE["lightblue"],
            "edgecolor": "none",
            "alpha": 0.9,
        },
    )

    fig.suptitle(f"PIT Diagnostic - {label}", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_power_curves(
    summary: pd.DataFrame,
    tests: list[str],
    *,
    save_to: Path | None = None,
) -> Figure:
    """Power curves: rejection rate vs sample size, one panel per scenario.

    ``summary`` must have columns ``scenario, n, test, rejection_rate``
    (the long-form output of
    :func:`esback.simulation.power_study.summarize_power`).
    """
    scenarios = list(summary["scenario"].unique())
    fig, axes = plt.subplots(
        1, len(scenarios), figsize=(4.5 * len(scenarios), 4), sharey=True, squeeze=False
    )
    palette_keys = ["blue", "red", "purple", "green", "orange", "darkblue"]
    for k, scenario in enumerate(scenarios):
        ax = axes[0, k]
        sub = summary[summary["scenario"] == scenario]
        for i, test in enumerate(tests):
            tsub = sub[sub["test"] == test].sort_values("n")
            if tsub.empty:
                continue
            ax.plot(
                tsub["n"],
                tsub["rejection_rate"],
                marker="o",
                label=test,
                color=PALETTE[palette_keys[i % len(palette_keys)]],
            )
        ax.axhline(0.05, color=PALETTE["gray"], linestyle="--", linewidth=1, label="5% nominal")
        ax.set_xlabel("Sample size n")
        ax.set_title(scenario)
        if k == 0:
            ax.set_ylabel("Rejection rate")
    axes[0, -1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_p_value_heatmap(
    p_values: pd.DataFrame,
    title: str,
    *,
    level: float = 0.05,
    save_to: Path | None = None,
) -> Figure:
    """Heatmap of p-values across rows (e.g. tests) and columns (e.g. assets).

    Cells with ``p < level`` are coloured red; cells failing to reject
    are coloured green. ``NaN`` cells are rendered grey. Cell colors
    interpolate between the palette's light and saturated tones so the
    intensity tracks the distance from the rejection threshold.
    """
    from matplotlib.colors import to_rgb

    n_rows, n_cols = p_values.shape
    fig, ax = plt.subplots(figsize=(max(8, 1.5 + 0.55 * n_cols), max(5, 1.0 + 0.55 * n_rows)))
    data = p_values.to_numpy(dtype=float)

    rgb_nan = np.array(to_rgb(PALETTE["lightgray"]))
    rgb_red_lo = np.array(to_rgb(PALETTE["lightred"]))
    rgb_red_hi = np.array(to_rgb(PALETTE["red"]))
    rgb_green_lo = np.array(to_rgb(PALETTE["lightgreen"]))
    rgb_green_hi = np.array(to_rgb(PALETTE["green"]))

    rgb = np.zeros((*data.shape, 3))
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                rgb[i, j] = rgb_nan
            elif v < level:
                t = min(1.0, (level - v) / level)
                rgb[i, j] = (1 - t) * rgb_red_lo + t * rgb_red_hi
            else:
                t = min(1.0, (v - level) / (1.0 - level))
                rgb[i, j] = (1 - t) * rgb_green_lo + t * rgb_green_hi

    ax.imshow(rgb, aspect="auto")
    ax.set_xticks(range(p_values.shape[1]))
    ax.set_xticklabels(p_values.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(p_values.shape[0]))
    ax.set_yticklabels(p_values.index, fontsize=10)
    ax.grid(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            if np.isnan(v):
                txt, txt_color = "n/a", PALETTE["gray"]
            elif v < 1e-3:
                txt, txt_color = f"{v:.0e}", "white"
            else:
                t = (v - level) / (1.0 - level) if v >= level else (level - v) / level
                txt = f"{v:.3f}"
                txt_color = "white" if t > 0.35 else PALETTE["darkblue"]
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=txt_color)

    ax.set_title(title)
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_lambda_diagram(save_to: Path | None = None) -> Figure:
    """Pedagogical figure illustrating when the lambda = n/T ratio matters."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    y1 = 3.0
    ax.barh(y1, 7, left=0.5, height=0.5, color=PALETTE["blue"], alpha=0.7, edgecolor="white")
    ax.barh(y1, 1.5, left=7.5, height=0.5, color=PALETTE["orange"], alpha=0.7, edgecolor="white")
    ax.text(
        4, y1, "$T = 2500$", ha="center", va="center", fontsize=13, fontweight="bold", color="white"
    )
    ax.text(
        8.25,
        y1,
        "$n = 500$",
        ha="center",
        va="center",
        fontsize=11,
        fontweight="bold",
        color="white",
    )
    ax.text(
        0.5,
        y1 + 0.45,
        "Scenario 1:",
        fontsize=12,
        fontweight="bold",
        va="bottom",
        color=PALETTE["darkblue"],
    )
    ax.text(
        9.3,
        y1,
        "$\\lambda = 0.2$",
        fontsize=13,
        fontweight="bold",
        va="center",
        color=PALETTE["green"],
    )
    ax.annotate(
        "",
        xy=(9.2, y1 - 0.15),
        xytext=(9.2, y1 - 0.6),
        arrowprops={"arrowstyle": "->", "color": PALETTE["green"], "lw": 2},
    )
    ax.text(
        9.3,
        y1 - 0.55,
        "Basic tests\nare fine",
        fontsize=11,
        va="center",
        fontweight="bold",
        color=PALETTE["green"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": PALETTE["lightgreen"], "alpha": 0.8},
    )

    y2 = 1.2
    ax.barh(y2, 3.5, left=0.5, height=0.5, color=PALETTE["blue"], alpha=0.7, edgecolor="white")
    ax.barh(y2, 3.5, left=4, height=0.5, color=PALETTE["orange"], alpha=0.7, edgecolor="white")
    ax.text(
        2.25,
        y2,
        "$T = 250$",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
    )
    ax.text(
        5.75,
        y2,
        "$n = 250$",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        color="white",
    )
    ax.text(
        0.5,
        y2 + 0.45,
        "Scenario 2:",
        fontsize=12,
        fontweight="bold",
        va="bottom",
        color=PALETTE["darkblue"],
    )
    ax.text(
        8, y2, "$\\lambda = 1.0$", fontsize=13, fontweight="bold", va="center", color=PALETTE["red"]
    )
    ax.annotate(
        "",
        xy=(8.8, y2 - 0.15),
        xytext=(8.8, y2 - 0.6),
        arrowprops={"arrowstyle": "->", "color": PALETTE["red"], "lw": 2},
    )
    ax.text(
        8.9,
        y2 - 0.55,
        "Use robust\n$MU_{ES}$, $MC_{ES}$",
        fontsize=11,
        va="center",
        fontweight="bold",
        color=PALETTE["red"],
        bbox={"boxstyle": "round,pad=0.3", "facecolor": PALETTE["lightred"], "alpha": 0.8},
    )

    blue_patch = mpatches.Patch(color=PALETTE["blue"], alpha=0.7, label="In-sample (estimation)")
    orange_patch = mpatches.Patch(
        color=PALETTE["orange"], alpha=0.7, label="Out-of-sample (evaluation)"
    )
    ax.legend(handles=[blue_patch, orange_patch], loc="upper right", fontsize=12, framealpha=0.9)

    ax.set_title(
        "Estimation Risk: When does $\\lambda = n/T$ matter?",
        fontsize=16,
        fontweight="bold",
        pad=10,
    )
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_p_value_strip(
    p_values: pd.Series | pd.DataFrame,
    title: str,
    *,
    level: float = 0.05,
    save_to: Path | None = None,
) -> Figure:
    """Strip plot of p-values on a log-x axis.

    ``p_values`` may be a Series indexed by test name, or a DataFrame
    whose rows are tests and columns are scenarios/assets — each column
    is plotted as a layer of dots. Dots left of the vertical
    ``level`` line are coloured red (rejection), the rest green.
    """
    if isinstance(p_values, pd.Series):
        df = p_values.to_frame(name="p_value")
        col_names = ["p_value"]
    else:
        df = p_values.copy()
        col_names = list(df.columns)

    tests = list(df.index)
    n_tests = len(tests)
    fig, ax = plt.subplots(figsize=(8.5, max(3.5, 0.35 * n_tests + 0.5)))

    eps = 1e-6
    for j, col in enumerate(col_names):
        for i, test in enumerate(tests):
            p = df.loc[test, col]
            if pd.isna(p):
                continue
            x = max(float(p), eps)
            color = PALETTE["red"] if x < level else PALETTE["green"]
            ax.scatter(
                x,
                i + (j - (len(col_names) - 1) / 2) * 0.08,
                s=42,
                color=color,
                edgecolors="white",
                linewidths=0.6,
                alpha=0.9,
                zorder=3,
            )

    ax.axvline(level, color=PALETTE["darkblue"], linewidth=1.0, linestyle="--", zorder=2)
    ax.text(
        level,
        n_tests - 0.4,
        " 5% level",
        color=PALETTE["darkblue"],
        fontsize=9,
        va="top",
        ha="left",
    )
    ax.set_xscale("log")
    ax.set_xlim(eps, 1.0)
    ax.set_yticks(range(n_tests))
    ax.set_yticklabels(tests)
    ax.set_xlabel("p-value (log scale)")
    ax.set_title(title)
    ax.grid(True, axis="x", linewidth=0.5, alpha=0.6)
    ax.invert_yaxis()
    fig.tight_layout()
    _save(fig, save_to)
    return fig


def plot_size_vs_nominal(
    size_summary: pd.DataFrame,
    *,
    nominal: float = 0.05,
    save_to: Path | None = None,
) -> Figure:
    """Empirical size against the nominal level with a binomial CI band.

    ``size_summary`` is the long-form summary from
    :func:`esback.simulation.power_study.summarize_power` filtered to a
    single ``H0_*`` scenario. The shaded band is the 95% binomial CI
    around the nominal level for the given number of replications,
    inferred from the ``rep`` count when the underlying replication count
    is not directly available.
    """
    df = size_summary.copy()
    tests = list(df["test"].unique())
    sizes = sorted(df["n"].unique())
    n_reps = int(df.groupby(["n", "test"]).size().median())  # rows per (n, test)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    se = float(np.sqrt(nominal * (1 - nominal) / max(n_reps, 1)))
    band_low = nominal - 1.96 * se
    band_high = nominal + 1.96 * se
    ax.axhspan(band_low, band_high, color=PALETTE["lightgray"], alpha=0.7, zorder=1)
    ax.axhline(nominal, color=PALETTE["darkblue"], linewidth=1.0, linestyle="--", zorder=2)

    palette_keys = ["blue", "orange", "red", "green", "purple", "teal", "darkblue", "gray"]
    for i, test in enumerate(tests):
        sub = df[df["test"] == test].sort_values("n")
        ax.plot(
            sub["n"],
            sub["rejection_rate"],
            marker="o",
            color=PALETTE[palette_keys[i % len(palette_keys)]],
            label=test,
            zorder=3,
        )

    ax.set_xlabel("Sample size n")
    ax.set_ylabel("Empirical rejection rate")
    ax.set_title("Empirical size under H0")
    ax.set_xticks(sizes)
    ax.set_ylim(0, max(0.15, df["rejection_rate"].max() * 1.1))
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    _save(fig, save_to)
    return fig
