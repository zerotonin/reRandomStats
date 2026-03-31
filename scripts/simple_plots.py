#!/usr/bin/env python3
"""
┌──────────────────────────────────────────────────────────────────────┐
│       monitoring_figures.py « Box + Scatter Plots »                  │
│                                                                      │
│  Generates publication-ready box-and-scatter plots for:              │
│    1. Method-type comparison  (ACO vs BTT vs STT)                    │
│    2. Pre vs post predator-control comparison                        │
│                                                                      │
│  Outputs: PNG (300 dpi) + SVG (fonttype none) in ./figures/          │
│                                                                      │
│  Author : Bart R.H. Geurten                                          │
└──────────────────────────────────────────────────────────────────────┘
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

# ┌──────────────────────────────────────────────────────────────────┐
# │                     « Configuration »                            │
# └──────────────────────────────────────────────────────────────────┘

CSV_PATH = Path("/home/geuba03p/PyProjects/reRandomStats/data/Monitoring.csv")
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

SESSIONS = [1, 2, 3, 4]

# matplotlib globals — editable SVG text, clean look
mpl.rcParams.update({
    "svg.fonttype": "none",
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})

# colour palette
PALETTE_METHOD = {"ACO": "#E69F00", "BTT": "#56B4E9", "STT": "#009E73"}
PALETTE_CTRL = {"pre_ctrl": "#7570B3", "post_ctrl": "#D95F02"}

# ┌──────────────────────────────────────────────────────────────────┐
# │                     « Data Loading »                             │
# └──────────────────────────────────────────────────────────────────┘

df = pd.read_csv(CSV_PATH)
df = df[df["session"].isin(SESSIONS)].copy()
df["total_lizard"] = df["Tukutuku"] + df["Oligosoma"]
df["ctrl_label"] = df["post_ctrl"].map({0: "pre_ctrl", 1: "post_ctrl"})


# ┌──────────────────────────────────────────────────────────────────┐
# │                     « Helper Function »                          │
# └──────────────────────────────────────────────────────────────────┘

def box_scatter(
    data: pd.DataFrame,
    group_col: str,
    value_col: str,
    order: list,
    palette: dict,
    ax: plt.Axes,
    jitter: float = 0.12,
    seed: int = 42,
) -> None:
    """Draw a box plot overlaid with jittered individual observations.

    Args:
        data:      DataFrame containing the data.
        group_col: Column name for the grouping variable.
        value_col: Column name for the response variable.
        order:     List defining the x-axis group order.
        palette:   Dict mapping group names to hex colours.
        ax:        Matplotlib Axes to draw on.
        jitter:    Half-width of the horizontal jitter.
        seed:      RNG seed for reproducible jitter.
    """
    rng = np.random.default_rng(seed)

    # ── box plot ─────────────────────────────────────────────────────
    grouped = [data.loc[data[group_col] == g, value_col].dropna().values for g in order]

    bp = ax.boxplot(
        grouped,
        positions=range(len(order)),
        widths=0.45,
        patch_artist=True,
        showfliers=False,
        zorder=2,
    )

    for patch, grp in zip(bp["boxes"], order):
        colour = palette[grp]
        patch.set_facecolor(colour)
        patch.set_alpha(0.35)
        patch.set_edgecolor(colour)
        patch.set_linewidth(1.2)

    for element in ["whiskers", "caps"]:
        for line, grp in zip(bp[element], [g for g in order for _ in range(2)]):
            line.set_color(palette[grp])
            line.set_linewidth(1.0)

    for line, grp in zip(bp["medians"], order):
        line.set_color(palette[grp])
        line.set_linewidth(2.0)

    # ── jittered scatter ─────────────────────────────────────────────
    for i, grp in enumerate(order):
        vals = data.loc[data[group_col] == grp, value_col].dropna().values
        x_jit = i + rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(
            x_jit, vals,
            c=palette[grp],
            s=12,
            alpha=0.6,
            edgecolors="none",
            zorder=3,
        )

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_ylabel(value_col)


def save_fig(fig: plt.Figure, name: str) -> None:
    """Save figure as both PNG and SVG.

    Args:
        fig:  Matplotlib Figure.
        name: Base filename (no extension).
    """
    fig.savefig(FIG_DIR / f"{name}.png", bbox_inches="tight", dpi=300)
    fig.savefig(FIG_DIR / f"{name}.svg", bbox_inches="tight")
    print(f"  Saved: {FIG_DIR / name}.png + .svg")
    plt.close(fig)


# ┌──────────────────────────────────────────────────────────────────┐
# │          « Figure 1: Method-Type Comparison »                    │
# └──────────────────────────────────────────────────────────────────┘

print("=" * 50)
print("  Generating method-type comparison figures")
print("=" * 50)

method_vars = ["Tukutuku", "Oligosoma", "total_lizard"]
method_order = ["ACO", "BTT", "STT"]

fig, axes = plt.subplots(1, 3, figsize=(9, 3.5), sharey=False)

for ax, var in zip(axes, method_vars):
    box_scatter(df, "method_type", var, method_order, PALETTE_METHOD, ax)
    ax.set_title(var)
    ax.set_xlabel("")

fig.suptitle("Detection by Method Type", fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "method_type_comparison")

# ── individual panels ────────────────────────────────────────────────
for var in method_vars:
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    box_scatter(df, "method_type", var, method_order, PALETTE_METHOD, ax)
    ax.set_title(var)
    ax.set_xlabel("Method")
    fig.tight_layout()
    save_fig(fig, f"method_type_{var}")


# ┌──────────────────────────────────────────────────────────────────┐
# │       « Figure 2: Pre vs Post Predator Control »                 │
# └──────────────────────────────────────────────────────────────────┘

print("\n" + "=" * 50)
print("  Generating pre/post control figures")
print("=" * 50)

ctrl_vars = ["Tukutuku", "Oligosoma", "total_lizard", "Rattus", "Lizard", "Nothing"]
ctrl_order = ["pre_ctrl", "post_ctrl"]

fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharey=False)

for ax, var in zip(axes.flat, ctrl_vars):
    sub = df[["ctrl_label", var]].dropna(subset=[var])
    box_scatter(sub, "ctrl_label", var, ctrl_order, PALETTE_CTRL, ax)
    ax.set_title(var)
    ax.set_xlabel("")

fig.suptitle("Detection: Pre vs Post Predator Control", fontweight="bold", y=1.02)
fig.tight_layout()
save_fig(fig, "pre_post_ctrl_comparison")

# ── individual panels ────────────────────────────────────────────────
for var in ctrl_vars:
    sub = df[["ctrl_label", var]].dropna(subset=[var])
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    box_scatter(sub, "ctrl_label", var, ctrl_order, PALETTE_CTRL, ax)
    ax.set_title(var)
    ax.set_xlabel("")
    fig.tight_layout()
    save_fig(fig, f"pre_post_{var}")


# ┌──────────────────────────────────────────────────────────────────┐
# │     « Figure 3: Pre/Post Control — Per Method Type »             │
# └──────────────────────────────────────────────────────────────────┘

print("\n" + "=" * 50)
print("  Generating per-method pre/post figures")
print("=" * 50)

per_method_vars = ["Tukutuku", "Oligosoma", "total_lizard", "Rattus"]

for var in per_method_vars:
    methods = sorted(df["method_type"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(3.5 * len(methods), 3.5), sharey=True)

    for ax, method in zip(axes, methods):
        sub = df.loc[df["method_type"] == method, ["ctrl_label", var]].dropna(subset=[var])
        if sub["ctrl_label"].nunique() < 2:
            ax.set_visible(False)
            continue
        box_scatter(sub, "ctrl_label", var, ctrl_order, PALETTE_CTRL, ax)
        ax.set_title(method)
        ax.set_xlabel("")

    fig.suptitle(f"{var}: Pre vs Post by Method", fontweight="bold", y=1.02)
    fig.tight_layout()
    save_fig(fig, f"pre_post_by_method_{var}")

print("\n  All figures saved to:", FIG_DIR.resolve())