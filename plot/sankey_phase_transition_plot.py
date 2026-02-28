#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib Sankey (Alluvial) — First 10 transitions per agent–model pair
Clean labels: only phase letters in nodes; no counts/percent labels.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path as FilePath
from typing import Dict, List, Tuple, Iterable, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches

# ----------------------- Configuration -----------------------

AGENTS = ["SWE-agent", "OpenHands"]
DISPLAY_MODELS = [
    "deepseek/deepseek-chat",                # DSK-V3
    "openrouter/deepseek/deepseek-r1-0528", # DSK-R1
    "openrouter/mistralai/devstral-small",   # Dev
    "openrouter/anthropic/claude-sonnet-4", # CLD-4
]

AGENT_ABBR = {"SWE-agent": "SA", "OpenHands": "OH"}
MODEL_ABBR = {
    "deepseek/deepseek-chat": "DSK-V3",
    "openrouter/deepseek/deepseek-r1-0528": "DSK-R1",
    "openrouter/mistralai/devstral-small": "Dev",
    "openrouter/anthropic/claude-sonnet-4": "CLD-4",
}

PHASE_ABBR = {"localization": "L", "patch": "P", "validation": "V"}
PHASES = ("L", "P", "V", "T")                    # include termination
PHASE_ORDER = {p: i for i, p in enumerate(PHASES)}

PASTEL = {
    "L": (0.62, 0.52, 0.95, 0.7),   # vivid lilac / purple
    "P": (0.98, 0.78, 0.25, 0.7),   # rich amber / gold
    "V": (0.55, 0.82, 0.55, 0.7),   # saturated soft green
    "T": (0.65, 0.65, 0.65, 0.7),   # darker neutral gray
}

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42

# Layout / styling
FIG_W = 26.0
FIG_H = 12.0
LEFT_MARGIN, RIGHT_MARGIN = 0.06, 0.00
TOP_MARGIN, BOTTOM_MARGIN = 0.08, 0.08
COL_SPACING = 1.95          # wider spacing to avoid overlaps
NODE_GAP = 0.07             # a bit more vertical breathing room
NODE_MIN_HEIGHT = 0.015
LABEL_SIZE = 20
TITLE_SIZE = 24
RIGHT_MARGIN = 0.0

# Link visibility controls
MIN_LINK_SHARE_TO_DRAW = 0.01     # hide links < 2% of that column's traffic

# ----------------------- NEW: title formatting helper -----------------------

def format_title(agent_name: str, model_abbr: str) -> str:
    """
    Render 'SWE-agent_{DSK-V3}' with the model as a math subscript.
    Hyphen in model abbr becomes \text{-} for proper LaTeX hyphen.
    Example: SWE-agent$_{\\mathrm{DSK\\text{-}V3}}$
    """
    safe_model = model_abbr.replace("-", r"\text{-}")
    return rf"{agent_name}$_{{\mathbf{{{safe_model}}}}}$"

# ----------------------- Paths -----------------------

def find_graph_root(data_dir: FilePath, agent: str, model: str) -> FilePath:
    """
    Locate graph directory for a given agent and model.

    Expected structure: {data_dir}/{agent}/graphs/{model_dir}/

    Args:
        data_dir: Base data directory
        agent: Agent name (e.g., "SWE-agent", "OpenHands")
        model: Full model identifier

    Returns:
        Path to the graphs directory
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

    graph_dir = data_dir / agent / "graphs" / model_dir
    if not graph_dir.exists():
        raise FileNotFoundError(f"Graph directory not found at {graph_dir}")

    return graph_dir

# ----------------------- Core utilities -----------------------

def _iter_nodes(graph_json: dict) -> Iterable[dict]:
    if isinstance(graph_json, dict):
        if "nodes" in graph_json and isinstance(graph_json["nodes"], list):
            yield from graph_json["nodes"]
        elif "graph" in graph_json and isinstance(graph_json["graph"], dict) and "nodes" in graph_json["graph"]:
            yield from graph_json["graph"]["nodes"]

class SequenceExtractor:
    """Extract phase sequences from a graph JSON (merged & filtered)."""

    @staticmethod
    def extract_phase_sequence(graph_json: dict) -> List[str]:
        step_phase: List[Tuple[int, Optional[str]]] = []
        for node in _iter_nodes(graph_json):
            step_indices = node.get("step_indices") or []
            phases = node.get("phases") or node.get("phase")

            if isinstance(phases, list):
                if len(phases) == len(step_indices):
                    for idx, phase in zip(step_indices, phases):
                        step_phase.append((idx, phase))
            else:
                phase = phases
                for idx in step_indices:
                    step_phase.append((idx, phase))

        if not step_phase:
            return []

        step_phase.sort(key=lambda x: x[0])

        seq: List[str] = []
        prev = None
        for _, ph in step_phase:
            if not ph or str(ph).lower() == "general":
                continue
            abbr = PHASE_ABBR.get(str(ph).lower())
            if not abbr:
                continue
            if abbr != prev:
                seq.append(abbr)
                prev = abbr
        
        if seq:
            seq.append("T")
        return seq

def list_graph_files(root: FilePath) -> List[FilePath]:
    """Recursively find all JSON files in the graph directory."""
    paths: List[FilePath] = []
    for json_file in root.rglob("*.json"):
        paths.append(json_file)
    return paths

def load_sequences_for_pair(data_dir: FilePath, agent: str, model: str) -> List[List[str]]:
    """Load phase sequences from all graphs for a given agent-model pair."""
    root = find_graph_root(data_dir, agent, model)
    sequences: List[List[str]] = []
    for fp in list_graph_files(root):
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
            seq = SequenceExtractor.extract_phase_sequence(data)
            if len(seq) >= 2:
                sequences.append(seq)
        except Exception:
            continue
    return sequences

# ----------------------- Aggregation to links -----------------------

def build_link_counts(
    sequences: List[List[str]],
    max_transitions: int = 10,
) -> Tuple[Dict[Tuple[str, int], int], Dict[Tuple[str, int, str, int], int], int, Dict[int, int]]:
    """
    Returns:
      node_volume[(phase, t)] = total volume at node (sum of incident link values)
      link_counts[(a, t, b, t+1)] = count
      max_t = last column index present
      col_total[t] = total transitions at iteration t (sum over all a->b at t)
    """
    node_volume: Dict[Tuple[str, int], int] = defaultdict(int)
    link_counts: Dict[Tuple[str, int, str, int], int] = defaultdict(int)
    col_total: Dict[int, int] = defaultdict(int)
    max_t = 0

    for seq in sequences:
        if len(seq) <= 1:
            continue
        usable = min(len(seq) - 1, max_transitions)
        for t in range(usable):
            a, b = seq[t], seq[t + 1]
            if a not in PHASES or b not in PHASES:
                continue
            link_counts[(a, t, b, t + 1)] += 1
            node_volume[(a, t)] += 1
            node_volume[(b, t + 1)] += 1
            col_total[t] += 1
            max_t = max(max_t, t + 1)

    return node_volume, link_counts, max_t, col_total

# ----------------------- Layout & drawing helpers -----------------------

def layout_columns(
    node_volume: Dict[Tuple[str, int], int],
    max_t: int,
) -> Dict[Tuple[str, int], Tuple[float, float]]:
    node_spans: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for t in range(max_t + 1):
        phases_here = [p for p in PHASES if (p, t) in node_volume]
        if not phases_here:
            continue
        vols = np.array([node_volume[(p, t)] for p in phases_here], dtype=float)
        total = vols.sum()
        gaps_total = NODE_GAP * (len(phases_here) - 1)
        usable = max(1e-6, 1.0 - gaps_total)
        heights = usable * (vols / total) if total > 0 else np.full_like(vols, usable / len(vols))
        y = 0.0
        for p, h in zip(phases_here, heights):
            h2 = max(h, NODE_MIN_HEIGHT)
            node_spans[(p, t)] = (y, min(1.0, y + h2))
            y = y + h2 + NODE_GAP
    return node_spans

def flow_slices_from_spans(
    node_spans: Dict[Tuple[str, int], Tuple[float, float]],
    link_counts: Dict[Tuple[str, int, str, int], int],
) -> Dict[Tuple[str, int, str, int], Tuple[Tuple[float, float], Tuple[float, float]]]:
    by_source: Dict[Tuple[str, int], List[Tuple[Tuple[str, int, str, int], int]]] = defaultdict(list)
    by_target: Dict[Tuple[str, int], List[Tuple[Tuple[str, int, str, int], int]]] = defaultdict(list)
    for key, v in link_counts.items():
        a, ta, b, tb = key
        by_source[(a, ta)].append((key, v))
        by_target[(b, tb)].append((key, v))

    link_slices: Dict[Tuple[str, int, str, int], Tuple[Tuple[float, float], Tuple[float, float]]] = {}

    node_total_src = {k: sum(v for _, v in vals) for k, vals in by_source.items()}
    node_total_tgt = {k: sum(v for _, v in vals) for k, vals in by_target.items()}

    # source stacking
    for node, items in by_source.items():
        if node not in node_spans:
            continue
        y0, y1 = node_spans[node]
        H = max(1e-9, y1 - y0)
        total = max(1, node_total_src[node])
        off = 0.0
        items.sort(key=lambda kv: (PHASE_ORDER.get(kv[0][2], 99), -kv[1]))
        for (a, ta, b, tb), v in items:
            h = H * (v / total)
            link_slices[(a, ta, b, tb)] = [(y0 + off, y0 + off + h), (0.0, 0.0)]
            off += h

    # target stacking
    for node, items in by_target.items():
        if node not in node_spans:
            continue
        y0, y1 = node_spans[node]
        H = max(1e-9, y1 - y0)
        total = max(1, node_total_tgt[node])
        off = 0.0
        items.sort(key=lambda kv: (PHASE_ORDER.get(kv[0][0], 99), -kv[1]))
        for (a, ta, b, tb), v in items:
            y_pair = link_slices.get((a, ta, b, tb))
            if y_pair is None:
                continue
            h = H * (v / total)
            link_slices[(a, ta, b, tb)] = (y_pair[0], (y0 + off, y0 + off + h))
            off += h

    return link_slices

def bezier_band(x0, x1, y0a, y1a, y0b, y1b, curvature=0.35) -> Path:
    cx0 = x0 + curvature * (x1 - x0)
    cx1 = x1 - curvature * (x1 - x0)
    verts = [
        (x0, y0a),
        (cx0, y0a),
        (cx1, y0b),
        (x1, y0b),
        (x1, y1b),
        (cx1, y1b),
        (cx0, y1a),
        (x0, y1a),
        (x0, y0a),
    ]
    codes = [
        Path.MOVETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.LINETO,
        Path.CURVE4, Path.CURVE4, Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)

def scale_rgba(color_rgba: Tuple[float, float, float, float], alpha_scale: float) -> Tuple[float, float, float, float]:
    r, g, b, a = color_rgba
    a2 = max(0.05, min(0.95, a * alpha_scale))
    return (r, g, b, a2)

# ----------------------- Drawing -----------------------

def draw_sankey_matplotlib(
    title: str,
    node_volume: Dict[Tuple[str, int], int],
    link_counts: Dict[Tuple[str, int, str, int], int],
    max_t: int,
    col_total: Dict[int, int],
    out_path: FilePath,
):
    """Draw and save a standalone Sankey diagram for a single agent-model pair."""
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), constrained_layout=False)
    x_last = LEFT_MARGIN + max_t * COL_SPACING
    ax.set_xlim(LEFT_MARGIN - 0.14, x_last + 0.14)  # 0.14 ≈ half node width + tiny buffer
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = {t: LEFT_MARGIN + t * COL_SPACING for t in range(max_t + 1)}

    # Layout
    node_spans = layout_columns(node_volume, max_t)
    link_slices = flow_slices_from_spans(node_spans, link_counts)

    # Column guides & shared iteration labels
    for t in range(max_t + 1):
        x = x_positions[t]
        ax.plot([x, x], [0, 1], color=(0, 0, 0, 0.05), linewidth=1.0, zorder=0)
        ax.text(x, -BOTTOM_MARGIN/2, f"{t}", ha="center", va="top",
                fontsize=LABEL_SIZE, color=(0, 0, 0, 0.8), transform=ax.transData)

    # Draw links: color by source phase; alpha scales with column share
    for (a, ta, b, tb), v in sorted(link_counts.items(), key=lambda kv: (kv[0][1], PHASE_ORDER.get(kv[0][0], 99), -kv[1])):
        total = max(1, col_total.get(ta, 1))
        share = v / total
        if share < MIN_LINK_SHARE_TO_DRAW:
            continue
        src_span, tgt_span = link_slices[(a, ta, b, tb)]
        x0, x1 = x_positions[ta], x_positions[tb]
        path = bezier_band(x0, x1, src_span[0], src_span[1], tgt_span[0], tgt_span[1], curvature=0.35)
        alpha_scale = 0.25 + 0.75 * np.sqrt(share)
        face = scale_rgba(PASTEL.get(a, (0.8, 0.8, 0.8, 0.9)), alpha_scale)
        ax.add_patch(PathPatch(path, facecolor=face, edgecolor="none", linewidth=0.0, zorder=1))

    # Node blocks + phase letters only
    for (p, t), (y0, y1) in node_spans.items():
        x = x_positions[t]
        ax.add_patch(plt.Rectangle((x - 0.12, y0), 0.24, y1 - y0,
                                   facecolor=PASTEL.get(p, (0.8, 0.8, 0.8, 0.9)),
                                   edgecolor=(0, 0, 0, 0.12),
                                   linewidth=0.6, zorder=2))
        ax.text(x, (y0 + y1) / 2, f"{p}",
                ha="center", va="center", fontsize=LABEL_SIZE,
                color=(0, 0, 0, 0.9), zorder=3)

    # Title
    ax.text(0.5, 1 - TOP_MARGIN/2,
            f"{title}",
            ha="center", va="top", transform=fig.transFigure,
            fontsize=TITLE_SIZE, weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

def draw_transition_midaxis(mid_ax, shared_max_t: int):
    """A thin middle axis with a double-headed arrow and 0..K labels."""
    mid_ax.set_ylim(0, 1)
    mid_ax.axis("off")

    # Match x-lims used in the sankey axes
    x_last = LEFT_MARGIN + shared_max_t * COL_SPACING
    mid_ax.set_xlim(LEFT_MARGIN - 0.14, x_last + 0.14)

    # Arrow across full usable span
    mid_ax.annotate(
        "",
        xy=(x_last + 0.08, 0.5),
        xytext=(LEFT_MARGIN - 0.08, 0.5),
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color=(0, 0, 0, 0.7))
    )

    # Ticks: 0..K aligned to time columns
    for t in range(shared_max_t + 1):
        x = LEFT_MARGIN + t * COL_SPACING
        mid_ax.plot([x, x], [0.40, 0.60], color=(0, 0, 0, 0.7), lw=1.0)
        mid_ax.text(x, 0.15, f"{t}", ha="center", va="center",
                    fontsize=LABEL_SIZE, color=(0, 0, 0, 0.85))

    mid_ax.text(
        (LEFT_MARGIN + shared_max_t * COL_SPACING)/2, 0.85,
        "Transition index", ha="center", va="center",
        fontsize=LABEL_SIZE, color=(0, 0, 0, 0.8)
    )


def draw_sankey_on_ax(
    ax,
    title: str,
    node_volume: Dict[Tuple[str, int], int],
    link_counts: Dict[Tuple[str, int, str, int], int],
    max_t: int,
    col_total: Dict[int, int],
    force_max_t: Optional[int] = None,
):
    """Draw a single sankey into an existing Axes (no iteration labels here)."""
    use_max_t = force_max_t if force_max_t is not None else max_t

    # Consistent x-lims across top/bottom for this column
    x_last = LEFT_MARGIN + use_max_t * COL_SPACING
    ax.set_xlim(LEFT_MARGIN - 0.14, x_last + 0.14)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = {t: LEFT_MARGIN + t * COL_SPACING for t in range(use_max_t + 1)}

    # Layout
    node_spans = layout_columns(node_volume, max_t)
    link_slices = flow_slices_from_spans(node_spans, link_counts)

    # Faint column guides (no numbers)
    for t in range(use_max_t + 1):
        x = x_positions[t]
        ax.plot([x, x], [0, 1], color=(0, 0, 0, 0.05), linewidth=1.0, zorder=0)

    # Links
    for (a, ta, b, tb), v in sorted(
        link_counts.items(), key=lambda kv: (kv[0][1], PHASE_ORDER.get(kv[0][0], 99), -kv[1])
    ):
        total = max(1, col_total.get(ta, 1))
        share = v / total
        if share < MIN_LINK_SHARE_TO_DRAW:
            continue
        src_span, tgt_span = link_slices[(a, ta, b, tb)]
        x0, x1 = x_positions[ta], x_positions[tb]
        path = bezier_band(x0, x1, src_span[0], src_span[1], tgt_span[0], tgt_span[1], curvature=0.35)
        alpha_scale = 0.25 + 0.75 * np.sqrt(share)
        face = scale_rgba(PASTEL.get(a, (0.8, 0.8, 0.8, 0.9)), alpha_scale)
        ax.add_patch(PathPatch(path, facecolor=face, edgecolor="none", linewidth=0.0, zorder=1))

    # Nodes + phase letters
    for (p, t), (y0, y1) in node_spans.items():
        x = LEFT_MARGIN + t * COL_SPACING
        ax.add_patch(plt.Rectangle((x - 0.12, y0), 0.24, y1 - y0,
                                   facecolor=PASTEL.get(p, (0.8, 0.8, 0.8, 0.9)),
                                   edgecolor=(0, 0, 0, 0.12),
                                   linewidth=0.6, zorder=2))
        ax.text(x, (y0 + y1) / 2, f"{p}",
                ha="center", va="center", fontsize=LABEL_SIZE,
                color=(0, 0, 0, 0.9), zorder=3)

    # Panel title
    ax.text(0.5, 1.02, title, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=LABEL_SIZE+2, weight="bold")

# ----------------------- Main driver -----------------------

def main():
    """CLI entry point for generating Sankey phase transition diagrams."""
    parser = argparse.ArgumentParser(
        description="Generate Sankey phase transition diagrams comparing agents and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate Sankey diagram using default data and figures directories
  python plot/sankey_phase_transition_plot.py

  # Specify custom data directory
  python sankey_phase_transition_plot.py --data-dir ./my_data

  # Specify custom output path
  python sankey_phase_transition_plot.py --output figures/custom_sankey.png
        """
    )

    parser.add_argument(
        "--data-dir",
        type=FilePath,
        default=FilePath("data"),
        help="Base directory containing graph data (default: data)"
    )

    parser.add_argument(
        "--output",
        type=FilePath,
        default=FilePath("figures/sankey_grid.png"),
        help="Output file path (default: figures/sankey_grid.png)"
    )

    args = parser.parse_args()

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=False)

    # 3 rows: top (SA), middle (arrow/ticks), bottom (OH)
    gs = GridSpec(
        3, 4, figure=fig,
        height_ratios=[1.0, 0.10, 1.0],
        wspace=0.18,
        hspace=0.28
    )

    # Precompute data & shared max_t per column (model)
    shared_max_t_by_col = {}
    data_by_cell = {}
    for col, model in enumerate(DISPLAY_MODELS):
        # SA
        seq_sa = load_sequences_for_pair(args.data_dir, "SWE-agent", model)
        nv_sa, lc_sa, mt_sa, ct_sa = build_link_counts(seq_sa, max_transitions=10)
        data_by_cell[(0, col)] = (nv_sa, lc_sa, mt_sa, ct_sa)

        # OH
        seq_oh = load_sequences_for_pair(args.data_dir, "OpenHands", model)
        nv_oh, lc_oh, mt_oh, ct_oh = build_link_counts(seq_oh, max_transitions=10)
        data_by_cell[(2, col)] = (nv_oh, lc_oh, mt_oh, ct_oh)

        shared_max_t_by_col[col] = max(mt_sa, mt_oh)

    # Draw grid
    for col, model in enumerate(DISPLAY_MODELS):
        # top (SWE-agent)
        ax_top = fig.add_subplot(gs[0, col])
        nv, lc, mt, ct = data_by_cell[(0, col)]
        agent_name_top = "SWE-agent"
        model_name = MODEL_ABBR[model]
        if lc:
            draw_sankey_on_ax(
                ax=ax_top,
                title=format_title(agent_name_top, model_name),
                node_volume=nv,
                link_counts=lc,
                max_t=mt,
                col_total=ct,
                force_max_t=shared_max_t_by_col[col],
            )
        else:
            ax_top.axis("off")
            ax_top.text(0.5, 0.5,
                        f"{format_title(agent_name_top, model_name)}\n(no data)",
                        ha="center", va="center", fontsize=LABEL_SIZE+2)

        # middle arrow/ticks (shared for column)
        ax_mid = fig.add_subplot(gs[1, col])
        draw_transition_midaxis(ax_mid, shared_max_t_by_col[col])

        # bottom (OpenHands)
        ax_bot = fig.add_subplot(gs[2, col])
        nv, lc, mt, ct = data_by_cell[(2, col)]
        agent_name_bot = "OpenHands"
        if lc:
            draw_sankey_on_ax(
                ax=ax_bot,
                title=format_title(agent_name_bot, model_name),
                node_volume=nv,
                link_counts=lc,
                max_t=mt,
                col_total=ct,
                force_max_t=shared_max_t_by_col[col],
            )
        else:
            ax_bot.axis("off")
            ax_bot.text(0.5, 0.5,
                        f"{format_title(agent_name_bot, model_name)}\n(no data)",
                        ha="center", va="center", fontsize=LABEL_SIZE+2)

    # Shared legend (phase colors)
    handles = [
        mpatches.Patch(color=PASTEL["L"], label="L (Localization)"),
        mpatches.Patch(color=PASTEL["P"], label="P (Patching)"),
        mpatches.Patch(color=PASTEL["V"], label="V (Validation)"),
        mpatches.Patch(color=PASTEL["T"], label="T (Termination)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=LABEL_SIZE,
        bbox_to_anchor=(0.5, 0.98),
        borderaxespad=0.2
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.output, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Sankey diagram saved to {args.output}")


if __name__ == "__main__":
    main()
