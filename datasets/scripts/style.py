"""Shared matplotlib styling for EDA charts, using a validated light-mode
palette (see the dataviz skill) so charts drop cleanly into slides."""

import matplotlib.pyplot as plt
import seaborn as sns

SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"

BLUE = "#2a78d6"
# Sequential blue ramp, light -> dark, for ordinal/magnitude encodings.
BLUE_RAMP = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def apply_style() -> None:
    sns.set_style("white")
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": BASELINE,
            "axes.grid": True,
            "grid.color": GRIDLINE,
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "text.color": INK_PRIMARY,
            "axes.labelcolor": INK_SECONDARY,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK_PRIMARY,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "figure.dpi": 150,
            "savefig.dpi": 150,
        }
    )
