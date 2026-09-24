"""
ePic Unified Publication-Quality Styling System
===============================================
Enforces a consistent, modern scientific dark theme across all ePic
experiments, diagnostics, animated movies, and interactive notebooks.
"""

from typing import Optional
import os
import matplotlib.pyplot as plt

# ePic Color Tokens
EPIC_COLORS = {
    "bg_dark": "#080c14",        # Figure background (Deep Cosmic Obsidian)
    "bg_axes": "#0d1322",        # Axes background (Twilight Navy)
    "text": "#f1f5f9",           # Primary text (Crisp Slate)
    "text_muted": "#94a3b8",     # Muted text / annotations
    "grid": "#1e293b",           # Grid line color
    "spine": "#334155",          # Axes border color
    "legend_bg": "#131b2e",      # Legend background
    # Curated plasma palette
    "cyan": "#38bdf8",           # Primary electron / wave mode
    "crimson": "#f43f5e",        # Secondary beam / ions / field peak
    "emerald": "#34d399",        # Kinetic energy / velocity / potential
    "gold": "#fbbf24",           # Theory fit / benchmark lines
    "purple": "#a855f7",         # Resonant trapping / distribution
    "blue": "#60a5fa",           # Outflow / guide fields
    "white": "#ffffff",          # Reference lines
}


def apply_epic_style():
    """Apply the unified ePic deep-space theme to Matplotlib global rcParams."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        # Figure and axes backgrounds
        "figure.facecolor": EPIC_COLORS["bg_dark"],
        "figure.edgecolor": "none",
        "figure.dpi": 140,
        "axes.facecolor": EPIC_COLORS["bg_axes"],
        "axes.edgecolor": EPIC_COLORS["spine"],
        "axes.linewidth": 1.2,
        "axes.labelcolor": EPIC_COLORS["text"],
        "axes.labelsize": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": EPIC_COLORS["text"],
        "axes.grid": True,

        # Grid styling
        "grid.color": EPIC_COLORS["grid"],
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "grid.alpha": 0.5,

        # Ticks styling
        "xtick.color": EPIC_COLORS["text_muted"],
        "ytick.color": EPIC_COLORS["text_muted"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.direction": "out",
        "ytick.direction": "out",

        # Fonts and text
        "font.family": "sans-serif",
        "text.color": EPIC_COLORS["text"],

        # Legend styling
        "legend.facecolor": EPIC_COLORS["legend_bg"],
        "legend.edgecolor": EPIC_COLORS["spine"],
        "legend.fontsize": 10,
        "legend.framealpha": 0.85,
        "legend.labelcolor": EPIC_COLORS["text"],

        # Lines and scatter
        "lines.linewidth": 2.0,
        "lines.antialiased": True,

        # Colorbar
        "image.cmap": "inferno",
    })


def format_epic_figure(
    fig,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    top_margin: float = 0.94,
):
    """Add unified titles and ensure consistent figure padding."""
    if title:
        full_title = title
        if subtitle:
            full_title = f"{title}\n{subtitle}"
        fig.suptitle(
            full_title,
            fontsize=15,
            fontweight="bold",
            color=EPIC_COLORS["text"],
            y=0.98,
        )
    fig.subplots_adjust(top=top_margin)


def save_epic_plot(fig, output_path: str):
    """Save plot with strict transparent-free dark background consistency."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(
        output_path,
        facecolor=EPIC_COLORS["bg_dark"],
        edgecolor="none",
        bbox_inches="tight",
        dpi=140,
    )
    plt.close(fig)
    print(f"  [SAVED - EPIC STYLE] {output_path}")
