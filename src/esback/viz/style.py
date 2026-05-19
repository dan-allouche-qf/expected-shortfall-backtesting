"""Plotting style: palette and matplotlib rcParams.

Modern quant style: a small, role-mapped palette and rcParams that
enforce a consistent look across all figures. Call
:func:`apply_rcparams` once at the top of a script or notebook before
any figure is drawn.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
from cycler import cycler

PALETTE: dict[str, str] = {
    # Primary series colors (saturated, web-safe)
    "blue": "#2563EB",  # primary series, GARCH, returns
    "orange": "#F59E0B",  # in-sample / OOS contrast, secondary series
    "red": "#DC2626",  # VaR, rejections, alerts
    "green": "#10B981",  # fail-to-reject, accepts
    "purple": "#7C3AED",  # ES, LSTM
    "teal": "#0EA5E9",  # tertiary, KLM / multinomial
    # Neutrals
    "darkblue": "#1E3A5F",  # axis emphasis, table headers, title color
    "gray": "#6B7280",  # supporting text, baselines
    "lightgray": "#E5E7EB",  # grid, faint fills, NaN cells
    # Semantic backgrounds (low chroma)
    "lightred": "#FEE2E2",
    "lightgreen": "#D1FAE5",
    "lightblue": "#DBEAFE",
    "lightorange": "#FFEDD5",
    "background": "#FAFAFA",
}

CATEGORICAL_CYCLE: list[str] = [
    PALETTE["blue"],
    PALETTE["orange"],
    PALETTE["red"],
    PALETTE["green"],
    PALETTE["purple"],
    PALETTE["teal"],
    PALETTE["darkblue"],
    PALETTE["gray"],
]


def apply_rcparams(dpi: int = 200, base_font: float = 11.0) -> None:
    """Apply the project-wide matplotlib settings.

    Idempotent: calling it twice has the same effect as calling it once.
    """
    plt.rcParams.update(
        {
            # Figure
            "figure.dpi": dpi,
            "figure.facecolor": "white",
            "savefig.dpi": dpi,
            "savefig.bbox": "tight",
            "savefig.facecolor": "white",
            # Axes
            "axes.facecolor": "white",
            "axes.edgecolor": "#1F2937",
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titlesize": base_font + 3,
            "axes.titleweight": "bold",
            "axes.titlepad": 12,
            "axes.titlecolor": PALETTE["darkblue"],
            "axes.labelsize": base_font + 1,
            "axes.labelcolor": "#1F2937",
            "axes.labelpad": 6,
            "axes.prop_cycle": cycler(color=CATEGORICAL_CYCLE),
            # Grid (light, horizontal-only by default)
            "axes.grid": True,
            "axes.grid.axis": "y",
            "grid.color": PALETTE["lightgray"],
            "grid.linewidth": 0.6,
            "grid.alpha": 0.7,
            # Ticks
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "xtick.labelsize": base_font,
            "ytick.labelsize": base_font,
            "xtick.major.size": 3,
            "ytick.major.size": 3,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            # Legend
            "legend.fontsize": base_font - 0.5,
            "legend.frameon": False,
            "legend.handletextpad": 0.5,
            "legend.columnspacing": 1.2,
            "legend.labelcolor": "#1F2937",
            # Font
            "font.family": "sans-serif",
            "font.sans-serif": ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans"],
            "font.size": base_font,
            "mathtext.fontset": "cm",
            # Lines
            "lines.linewidth": 1.5,
            "lines.markersize": 5,
            # Patches
            "patch.linewidth": 0.5,
            "patch.edgecolor": "white",
        }
    )
