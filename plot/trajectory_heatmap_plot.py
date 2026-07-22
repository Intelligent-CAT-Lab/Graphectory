#!/usr/bin/env python3
"""
Trajectory metric heatmap with dispersion annotations.

- Keep original single 8-column layout (SA 4 models, OH 4 models).
- Top Resolution(%) band (one cell per column).
- In the main heatmap, each agent-model cell is vertically expanded into two half-cells:
  left half = Resolved, right half = Unresolved.
- Annotate each half with two lines: "<central>\\n<dispersion>".
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ----------------------- Configuration -----------------------

AGENTS = ["SWE-agent", "OpenHands"]
MODELS = [
    "deepseek/deepseek-chat",
    "openrouter/deepseek/deepseek-r1-0528",
    "openrouter/mistralai/devstral-small",
    "openrouter/anthropic/claude-sonnet-4",
]

AGENT_ABBR = {"SWE-agent": "SA", "OpenHands": "OH"}
MODEL_ABBR = {
    "deepseek/deepseek-chat": "DSK-V3",
    "openrouter/deepseek/deepseek-r1-0528": "DSK-R1",
    "openrouter/mistralai/devstral-small": "Dev",
    "openrouter/anthropic/claude-sonnet-4": "CLD-4",
}

# Fixed order: SA's 4 models, then OH's 4 models
X_ORDER = [
    ("SWE-agent", "deepseek/deepseek-chat"),
    ("SWE-agent", "openrouter/deepseek/deepseek-r1-0528"),
    ("SWE-agent", "openrouter/mistralai/devstral-small"),
    ("SWE-agent", "openrouter/anthropic/claude-sonnet-4"),
    ("OpenHands", "deepseek/deepseek-chat"),
    ("OpenHands", "openrouter/deepseek/deepseek-r1-0528"),
    ("OpenHands", "openrouter/mistralai/devstral-small"),
    ("OpenHands", "openrouter/anthropic/claude-sonnet-4"),
]

# Resolution (% and n) per (agent, model) – for top band
RESOLUTION = {
    ("SWE-agent", "deepseek/deepseek-chat"):              {"n": 191, "pct": 38.2},
    ("SWE-agent", "openrouter/deepseek/deepseek-r1-0528"): {"n": 196, "pct": 39.2},
    ("SWE-agent", "openrouter/mistralai/devstral-small"): {"n": 202, "pct": 40.4},
    ("SWE-agent", "openrouter/anthropic/claude-sonnet-4"): {"n": 338, "pct": 67.6},
    ("OpenHands", "deepseek/deepseek-chat"):              {"n": 176, "pct": 35.2},
    ("OpenHands", "openrouter/deepseek/deepseek-r1-0528"): {"n": 204,  "pct": 40.8},
    ("OpenHands", "openrouter/mistralai/devstral-small"): {"n": 253, "pct": 50.6},
    ("OpenHands", "openrouter/anthropic/claude-sonnet-4"): {"n": 355, "pct": 71.0},
}

# Metrics to summarize (raw)
REQUIRED_COLUMNS = [
    "node_count",
    "exec_edge_count",
    "loop_count",
    "avg_loop_length",
    "hier_edge_count",
    "max_view_span",
]

ROW_LABELS = [
    "NodeCount", "TempEdgeCount", "LoopCount", "AvgLoopLength", 
    "StructEdgeCount", "StructuralBreadth",
]
KEY_MAP = {
    "NodeCount": "node_count",
    "TempEdgeCount": "exec_edge_count",
    "LoopCount": "loop_count",
    "AvgLoopLength": "avg_loop_length",
    "StructEdgeCount": "hier_edge_count",
    "StructuralBreadth": "max_view_span",
}

# ----------------------- Path Configuration -----------------------

def find_csv_path(data_dir: Path, agent: str, model: str) -> Path:
    """
    Locate trajectory_metrics.csv for a given agent and model.

    Expected structure: {data_dir}/{agent}/analysis/{model_dir}/trajectory_metrics.csv

    Args:
        data_dir: Base data directory
        agent: Agent name (e.g., "SWE-agent", "OpenHands")
        model: Full model identifier

    Returns:
        Path to the trajectory_metrics.csv file
    """
    # Map full model names to directory names
    model_dir_map = {
        "deepseek/deepseek-chat": "deepseek-v3",
        "openrouter/deepseek/deepseek-r1-0528": "deepseek-r1-0528",
        "openrouter/mistralai/devstral-small": "devstral-small",
        "openrouter/anthropic/claude-sonnet-4": "claude-sonnet-4",
    }

    model_dir = model_dir_map.get(model)
    if not model_dir:
        raise ValueError(f"Unknown model: {model}")

    csv_file = data_dir / agent / "analysis" / model_dir / "trajectory_metrics.csv"
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found at {csv_file}")

    return csv_file

# ----------------------- Stats -----------------------

def _col_stats(series: pd.Series) -> Dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return {"mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan}
    q1, q3 = np.percentile(s, [25, 75])
    return {
        "mean": float(np.mean(s)),
        "std":  float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
        "median": float(np.median(s)),
        "iqr": float(q3 - q1),
    }

def _stats_for_df(df_sub: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Compute statistics for all required columns in a DataFrame subset."""
    s: Dict[str, Dict[str, float]] = {}
    for m in REQUIRED_COLUMNS:
        s[m] = _col_stats(df_sub[m]) if m in df_sub.columns else {
            "mean": np.nan, "std": np.nan, "median": np.nan, "iqr": np.nan
        }
    return s

def build_mats_expanded(
    data_dir: Path,
    central_key: Literal["mean", "median"] = "median",
    disp_key: Literal["std", "iqr"] = "iqr",
):
    """
    Build two 3D arrays (rows x cols x 2) for resolved/unresolved central and dispersion.

    Args:
        data_dir: Base data directory containing agent analysis results
        central_key: Which central tendency measure to use ("mean" or "median")
        disp_key: Which dispersion measure to use ("std" or "iqr")

    Returns:
        Tuple of (central, disper, res_band_vals, res_band_ann, x_labels)
        - central: (R x C x 2) array of central values
        - disper: (R x C x 2) array of dispersion values
        - res_band_vals: (C,) array of resolution percentages
        - res_band_ann: List of formatted resolution strings
        - x_labels: List of model abbreviations
    """
    R = len(ROW_LABELS)
    C = len(X_ORDER)

    central = np.full((R, C, 2), np.nan, dtype=float)
    disper  = np.full((R, C, 2), np.nan, dtype=float)

    for j, (agent, model) in enumerate(X_ORDER):
        csv_file = find_csv_path(data_dir, agent, model)
        df = pd.read_csv(csv_file)
        res_series = (
            df["resolution"].astype(str).str.strip().str.lower()
            if "resolution" in df.columns else pd.Series([""] * len(df))
        )
        for side, tag in enumerate(["resolved", "unresolved"]):
            stats = _stats_for_df(df[res_series == tag])
            for i, rl in enumerate(ROW_LABELS):
                k = KEY_MAP[rl]
                central[i, j, side] = stats[k][central_key]
                disper[i,  j, side] = stats[k][disp_key]

    res_band_vals = np.array([RESOLUTION[(a, m)]["pct"] for (a, m) in X_ORDER], dtype=float)
    res_band_ann  = [f'{RESOLUTION[(a, m)]["pct"]:.1f}%' for (a, m) in X_ORDER]
    x_labels      = [MODEL_ABBR[m] for (_, m) in X_ORDER]
    return central, disper, res_band_vals, res_band_ann, x_labels

# ----------------------- Plotting -----------------------

def _adaptive_text_color(val: float, vmin: float, vmax: float) -> str:
    if not np.isfinite(val) or vmax <= vmin:
        return "#1e1e1e"
    norm = (val - vmin) / (vmax - vmin)
    return "#f9f9f9" if norm > 0.60 else "#1e1e1e"

def plot_expanded_cells(
    central: np.ndarray,  # (R,C,2)
    disper: np.ndarray,   # (R,C,2)
    res_band_vals: np.ndarray,  # (C,)
    res_band_ann: List[str],
    x_labels: List[str],
    out_path: Path,
):
    """
    Create and save the expanded heatmap visualization.

    Args:
        central: (R,C,2) array of central tendency values
        disper: (R,C,2) array of dispersion values
        res_band_vals: (C,) array of resolution percentages
        res_band_ann: List of formatted resolution annotations
        x_labels: List of model labels for x-axis
        out_path: Output file path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Typography / styling (keep original feel)
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#bbbbbb"
    plt.rcParams["axes.linewidth"] = 0.6

    # Colormaps: green for Resolved, pink for Unresolved
    green_cmap = LinearSegmentedColormap.from_list(
        "green_palette",
        ["#e8f5e9", "#c8e6c9", "#a5d6a7", "#81c784", "#66bb6a", "#4caf50"],
        N=256,
    )
    pink_cmap = LinearSegmentedColormap.from_list(
        "pink_palette",
        ["#fde8ef", "#f9d1e1", "#f5b9d2", "#f1a0c3", "#ec87b3", "#e66da3"],
        N=256,
    )
    res_cmap = LinearSegmentedColormap.from_list(
        "sky_blue",
        ["#e0f2ff", "#b3deff", "#87ceeb", "#5bb7e5", "#3ba1d9"],
        N=256,
    )

    R, C, _ = central.shape

    # Global vmin/vmax across BOTH halves to keep color scale consistent
    all_vals = central.reshape(-1)
    vmin = float(np.nanmin(all_vals)) if np.any(np.isfinite(all_vals)) else 0.0
    vmax = float(np.nanmax(all_vals)) if np.any(np.isfinite(all_vals)) else 1.0
    if vmin == vmax:
        vmin -= 0.5; vmax += 0.5
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm_green = ScalarMappable(norm=norm, cmap=green_cmap)
    sm_pink = ScalarMappable(norm=norm, cmap=pink_cmap)

    # Figure with top band + main grid
    fig = plt.figure(figsize=(10.6, 6.9), constrained_layout=True)
    gs = fig.add_gridspec(nrows=2, ncols=1, height_ratios=[0.10, 1.0])

    # ----- Top resolution band -----
    ax_res = fig.add_subplot(gs[0, 0])
    ax_res.imshow(res_band_vals[None, :], aspect="auto", cmap=res_cmap, vmin=0, vmax=100)
    ax_res.set_yticks([0]); ax_res.set_yticklabels(["Resolution (%)"], fontsize=12)
    ax_res.set_xticks([]); ax_res.tick_params(bottom=False, top=False, labelbottom=False, labeltop=False)
    for j in range(C):
        ax_res.text(
            j, 0, res_band_ann[j], ha="center", va="center",
            fontsize=13, fontweight="bold",
            color=_adaptive_text_color(res_band_vals[j], 0.0, 100.0),
        )
    for sp in ax_res.spines.values():
        sp.set_visible(False)

    # ----- Main expanded heatmap -----
    ax = fig.add_subplot(gs[1, 0])

    # Draw each cell as two vertical half-rectangles
    for i in range(R):
        for j in range(C):
            # resolved (left half) - green
            val_r = central[i, j, 0]
            color_r = sm_green.to_rgba(val_r) if np.isfinite(val_r) else (1,1,1,1)
            rect_r = Rectangle((j - 0.5, i - 0.5), 0.5, 1.0, facecolor=color_r, edgecolor='none')
            ax.add_patch(rect_r)
            # unresolved (right half) - pink
            val_u = central[i, j, 1]
            color_u = sm_pink.to_rgba(val_u) if np.isfinite(val_u) else (1,1,1,1)
            rect_u = Rectangle((j, i - 0.5), 0.5, 1.0, facecolor=color_u, edgecolor='none')
            ax.add_patch(rect_u)

    # Annotations: render median (larger) and IQR (slightly smaller) as two lines
    dy = 0.14  # vertical offset within a cell (in data coords)
    for i in range(R):
        for j in range(C):
            # left half (Resolved)
            val_r  = central[i, j, 0]
            disp_r = disper[i, j, 0]
            txtc_r = _adaptive_text_color(val_r, vmin, vmax)
            ax.text(j - 0.25, i - dy, f"{val_r:.1f}",
                    ha="center", va="center",
                    fontsize=12.0, fontweight="bold", color=txtc_r)
            ax.text(j - 0.25, i + dy, f"{disp_r:.1f}",
                    ha="center", va="center",
                    fontsize=11.4, fontweight="semibold", color=txtc_r)

            # right half (Unresolved)
            val_u  = central[i, j, 1]
            disp_u = disper[i, j, 1]
            txtc_u = _adaptive_text_color(val_u, vmin, vmax)
            ax.text(j + 0.25, i - dy, f"{val_u:.1f}",
                    ha="center", va="center",
                    fontsize=12.0, fontweight="bold", color=txtc_u)
            ax.text(j + 0.25, i + dy, f"{disp_u:.1f}",
                    ha="center", va="center",
                    fontsize=11.4, fontweight="semibold", color=txtc_u)

    # Axes ticks/labels
    ax.set_xlim(-0.5, C - 0.5); ax.set_ylim(R - 0.5, -0.5)
    ax.set_xticks(np.arange(C)); ax.set_xticklabels(x_labels, fontsize=13)
    # --- Add per-pair sublabels: R (left half) and U (right half) ---
    minor_positions = np.repeat(np.arange(C), 2) + np.tile([-0.25, 0.25], C)
    ax.set_xticks(minor_positions, minor=True)
    ax.set_xticklabels(["R", "U"] * C, fontsize=11, minor=True)

    # Space the two label rows so they don't overlap; hide tick marks
    ax.tick_params(axis="x", which="major", pad=18, length=0)
    ax.tick_params(axis="x", which="minor", pad=2,  length=0)

    ax.set_yticks(np.arange(R)); ax.set_yticklabels(ROW_LABELS, fontsize=15)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    # Add vertical lines to separate different models (between column 3 and 4)
    ax.axvline(x=3.5, color='#666666', linewidth=1.5, linestyle='-', zorder=10)

    # Colorbar (metrics) - use green colormap as reference
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.6%", pad="2%")
    cbar = fig.colorbar(sm_green, cax=cax)
    cbar.ax.set_ylabel("Value (median / IQR)", rotation=270, labelpad=12, fontsize=14)
    cbar.outline.set_linewidth(0.6)

    # Colorbar for resolution band
    divider_res = make_axes_locatable(ax_res)
    cax_res = divider_res.append_axes("right", size="2.6%", pad="2%")
    sm_res = ScalarMappable(norm=Normalize(vmin=0, vmax=100), cmap=res_cmap)
    cbar_res = fig.colorbar(sm_res, cax=cax_res)
    cbar_res.ax.set_ylabel("Rate (%)", rotation=270, labelpad=12, fontsize=14)
    cbar_res.outline.set_linewidth(0.6)

    # Optional agent group labels (subtle, above columns)
    pos0 = ax.get_position()
    # centers for columns 0..3 and 4..7 in figure coordinates
    # compute midpoints using axis transform:
    def data_to_fig_x(x):
        return ax.transData.transform((x, 0))[0] / fig.dpi / fig.get_size_inches()[0]
    left_mid  = (data_to_fig_x(-0.5) + data_to_fig_x(3.5)) / 2.0
    right_mid = (data_to_fig_x(3.5) + data_to_fig_x(7.5)) / 2.0
    top_y = ax_res.get_position().y1 + 0.01
    # fig.text(left_mid,  top_y, "SWE-Agent",  ha="center", va="bottom", fontsize=12)
    # fig.text(right_mid, top_y, "OpenHands", ha="center", va="bottom", fontsize=12)
    fig.savefig(out_path, format="png", bbox_inches="tight")
    plt.close(fig)

# ----------------------- Main -----------------------

def main():
    """CLI entry point for generating trajectory heatmap visualization."""
    parser = argparse.ArgumentParser(
        description="Generate trajectory metrics heatmap comparing agents and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate heatmap using default data and figures directories
  python plot/trajectory_heatmap_plot.py

  # Specify custom data directory
  python trajectory_heatmap_plot.py --data-dir ./my_data

  # Specify custom output path
  python trajectory_heatmap_plot.py --output figures/custom_heatmap.png

  # Use mean/std instead of median/IQR
  python trajectory_heatmap_plot.py --central mean --dispersion std
        """
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory containing trajectory analysis data (default: data)"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(os.environ.get("OUTPUT_DIR", "figures")) / "median_iqr_trajectory_heatmap.png",
        help="Output file path (default: OUTPUT_DIR/median_iqr_trajectory_heatmap.png or figures/...)"
    )

    parser.add_argument(
        "--central",
        choices=["mean", "median"],
        default="median",
        help="Central tendency measure to use (default: median)"
    )

    parser.add_argument(
        "--dispersion",
        choices=["std", "iqr"],
        default="iqr",
        help="Dispersion measure to use (default: iqr)"
    )

    args = parser.parse_args()

    # Build data matrices
    central, disper, res_vals, res_ann, x_labels = build_mats_expanded(
        data_dir=args.data_dir,
        central_key=args.central,
        disp_key=args.dispersion
    )

    # Generate and save plot
    plot_expanded_cells(central, disper, res_vals, res_ann, x_labels, args.output)
    print(f"[OK] Heatmap saved to {args.output}")

if __name__ == "__main__":
    main()
