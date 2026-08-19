"""Shared plotting style and helpers for the gridworld scripts.

Used by 418_Code_1.py, td_control.py, and compare_all.py so that all three
render with the same palette and the same value/policy grid layout - makes
it easy to visually compare Dynamic Programming against TD control methods.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ---- validated categorical + sequential palette ----
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID_COLOR = "#e1e0d9"
SURFACE = "#fcfcfb"

DP_COLOR = "#c26a1d"          # categorical slot 3 (amber) - Value Iteration
PE_COLOR = "#6b4fa0"          # categorical slot 4 (violet) - Policy Evaluation
SARSA_COLOR = "#2a78d6"       # categorical slot 1 (blue)
QLEARN_COLOR = "#008300"      # categorical slot 2 (green)

SEQUENTIAL_BLUE = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

ARROWS = {0: "↑", 1: "↓", 2: "→", 3: "←"}  # up, down, right, left


def rollingMean(values, window=10):
    """Smooth a noisy per-episode series so the trend is readable."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return values
    kernel = np.ones(window) / window
    return np.convolve(values, kernel, mode="valid")


def styledAxes(ax):
    """Apply the shared grid/spine/tick styling to a matplotlib Axes."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID_COLOR, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.tick_params(colors=INK_MUTED, labelsize=9)


def plotSeries(series, title, ylabel, path, xlabel="Episode / Iteration"):
    """Styled multi-line plot. `series` is {label: (values, color)}."""
    fig, ax = plt.subplots(figsize=(9, 5), facecolor=SURFACE)
    styledAxes(ax)

    for label, (values, color) in series.items():
        ax.plot(values, color=color, linewidth=2, solid_capstyle="round", label=label)

    ax.set_title(title, color=INK_PRIMARY, fontsize=13, loc="left", pad=12)
    ax.set_xlabel(xlabel, color=INK_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=10)

    legend = ax.legend(frameon=False, loc="upper right", fontsize=9)
    for text in legend.get_texts():
        text.set_color(INK_SECONDARY)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plotValueGrids(panels, terminationStates, path, suptitle, gridSize=4):
    """Row of heatmaps of V(s) with the greedy policy arrows overlaid.

    `panels` is a list of (V, policy, name) tuples. `policy` may be None
    for panels with no well-defined greedy action (e.g. a random policy).
    """
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    worstV = min(V.min() for V, _, _ in panels)

    fig, axes = plt.subplots(1, len(panels), figsize=(4.5 * len(panels), 4.5), facecolor=SURFACE)
    if len(panels) == 1:
        axes = [axes]

    for ax, (V, policy, name) in zip(axes, panels):
        magnitude = -V  # 0 (best, near a goal) -> light ... worst -> dark
        ax.imshow(magnitude, cmap=cmap, vmin=0, vmax=-worstV)

        for i in range(gridSize):
            for j in range(gridSize):
                normVal = magnitude[i, j] / -worstV if worstV != 0 else 0
                textColor = "#ffffff" if normVal > 0.55 else INK_PRIMARY
                if [i, j] in terminationStates:
                    ax.text(j, i, "GOAL", ha="center", va="center", color=INK_PRIMARY,
                            fontsize=9, fontweight="bold")
                elif policy is None:
                    ax.text(j, i, f"{V[i, j]:.1f}", ha="center", va="center",
                            color=textColor, fontsize=10)
                else:
                    ax.text(j, i, f"{ARROWS[policy[i, j]]}\n{V[i, j]:.1f}", ha="center",
                            va="center", color=textColor, fontsize=10)

        ax.set_title(name, color=INK_PRIMARY, fontsize=12, loc="left")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(suptitle, color=INK_PRIMARY, fontsize=13, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
