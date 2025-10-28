#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Matplotlib Sankey (Alluvial) - Language transition visualization

Visualizes transitions between language alphabets (L_reproduce, L_navigate, P, V_newly_generated_test, V_regression_test)
across the first 10 transitions per agent-model pair.

Input:
  - lang_path: Path to languatory.json file (optional)
  - agent: Agent name (optional, used with model for default path)
  - model: Model name (optional, used with agent for default path)
  - Default: data/{agent}/langs/{model}/languatory.json

Output directory structure (defaults to figures/):
  1. Multi-mode (default): Scans all agents/models, outputs one 2*4 grid of sankey transition plots
     Output: {output_dir}/lang_sankey/all_sankey.pdf

  2. Single-mode (--lang-path or --agent + --model): Requires specific data
     Output: {output_dir}/lang_sankey/{agent}_{model}_sankey.pdf
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec


# ----------------------- Configuration -----------------------

AGENTS = ["SWE-agent", "OpenHands"]
DISPLAY_MODELS = [
    "deepseek-v3",
    "deepseek-r1-0528",
    "devstral-small",
    "claude-sonnet-4",
]

MODEL_ABBR = {
    "deepseek-v3": "DSK-V3",
    "deepseek-r1-0528": "DSK-R1",
    "devstral-small": "Dev",
    "claude-sonnet-4": "CLD-4",
}

# Language roles (alphabets) - add T (termination) to given alphabets
ALPHABETS = (
    "L_reproduce",
    "L_navigate",
    "P",
    "V_newly_generated_test",
    "V_regression_test",
    "T"
)
ALPHABET_ORDER = {p: i for i, p in enumerate(ALPHABETS)}

# Abbreviated labels for display (used in nodes and legend)
ALPHABET_ABBR = {
    "L_reproduce": "L_repr",
    "L_navigate": "L_nav",
    "P": "P",
    "V_newly_generated_test": "V_new",
    "V_regression_test": "V_reg",
    "T": "T",
}

# Color scheme: roles in same family have similar colors but are clearly distinguishable
# L family: Purple/Lavender tones
# P family: Orange/Yellow tones
# V family: Green/Teal tones
# T: Gray (neutral)
PASTEL = {
    "L_reproduce": (0.85, 0.70, 0.95, 1.0),          # Light purple/lavender
    "L_navigate": (0.60, 0.50, 0.85, 1.0),           # Deeper purple (clearly different)
    "P": (1.00, 0.80, 0.40, 1.0),                     # Warm orange/yellow
    "V_newly_generated_test": (0.60, 0.90, 0.70, 1.0), # Light green/mint
    "V_regression_test": (0.40, 0.75, 0.60, 1.0),    # Deeper teal/green (clearly different)
    "T": (0.75, 0.75, 0.75, 1.0),                     # Neutral gray
}

# Layout / styling
FIG_W = 32.0                # Larger width for better readability
FIG_H = 16.0                # Taller height
LEFT_MARGIN = 0.12
RIGHT_MARGIN = 0.00
TOP_MARGIN = 0.08
BOTTOM_MARGIN = 0.08
COL_SPACING = 2.8           # Much wider spacing between transitions
NODE_GAP = 0.12             # More vertical breathing room
NODE_MIN_HEIGHT = 0.025
NODE_WIDTH = 0.18           # Wider nodes for labels
LABEL_SIZE = 15             # Smaller font to fit better
TITLE_SIZE = 18             # Panel titles

# Link visibility controls
MIN_LINK_SHARE_TO_DRAW = 0.0    # hide links with less than this fraction of column total


# ----------------------- Data Loading -----------------------

def load_languatory(lang_path: Path) -> List[List[str]]:
    """
    Load languatory.json and extract sequences.

    Args:
        lang_path: Path to languatory.json file

    Returns:
        List of sequences where each sequence is a list of alphabet symbols
        (ignoring the run_length suffix after underscore)
    """
    try:
        with open(lang_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[ERROR] Failed to load {lang_path}: {e}")
        return []

    sequences = []

    for entry in data:
        if not isinstance(entry, dict):
            continue

        languatory = entry.get("languatory", [])
        if not languatory:
            continue

        # Extract alphabet symbols, ignoring run_length (the _N suffix)
        sequence = []
        for item in languatory:
            if not isinstance(item, str):
                continue
            # Split by underscore and ignore the last part (run_length)
            parts = item.rsplit("_", 1)
            alphabet = parts[0] if parts else item

            # Only keep valid alphabets
            if alphabet in ALPHABETS:
                sequence.append(alphabet)

        if sequence:
            # Add termination marker
            sequence.append("T")
            sequences.append(sequence)

    return sequences


# ----------------------- Title Formatting -----------------------

def format_title(agent_name: str, model_abbr: str) -> str:
    """
    Render 'SWE-agent_{DSK-V3}' with the model as a math subscript.
    Example: SWE-agent$_{\\mathbf{DSK-V3}}$
    """
    return rf"{agent_name}$_{{\mathbf{{{model_abbr}}}}}$"


# ----------------------- Transition Aggregation -----------------------

def build_link_counts(
    sequences: List[List[str]],
    max_transitions: int = 10,
) -> Tuple[Dict[Tuple[str, int], int], Dict[Tuple[str, int, str, int], int], int, Dict[int, int]]:
    """
    Build transition counts from sequences.

    Args:
        sequences: List of alphabet sequences
        max_transitions: Maximum number of transitions to consider

    Returns:
        node_volume[(alphabet, t)] = total volume at node (sum of incident link values)
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
            if a not in ALPHABETS or b not in ALPHABETS:
                continue
            link_counts[(a, t, b, t + 1)] += 1
            node_volume[(a, t)] += 1
            node_volume[(b, t + 1)] += 1
            col_total[t] += 1
            max_t = max(max_t, t + 1)

    return node_volume, link_counts, max_t, col_total


# ----------------------- Layout & Drawing Helpers -----------------------

def layout_columns(
    node_volume: Dict[Tuple[str, int], int],
    max_t: int,
) -> Dict[Tuple[str, int], Tuple[float, float]]:
    """Calculate vertical positions for each node."""
    node_spans: Dict[Tuple[str, int], Tuple[float, float]] = {}
    for t in range(max_t + 1):
        alphabets_here = [p for p in ALPHABETS if (p, t) in node_volume]
        if not alphabets_here:
            continue
        vols = np.array([node_volume[(p, t)] for p in alphabets_here], dtype=float)
        total = vols.sum()
        gaps_total = NODE_GAP * (len(alphabets_here) - 1)
        usable = max(1e-6, 1.0 - gaps_total)
        heights = usable * (vols / total) if total > 0 else np.full_like(vols, usable / len(vols))
        y = 0.0
        for p, h in zip(alphabets_here, heights):
            h2 = max(h, NODE_MIN_HEIGHT)
            node_spans[(p, t)] = (y, min(1.0, y + h2))
            y = y + h2 + NODE_GAP
    return node_spans


def flow_slices_from_spans(
    node_spans: Dict[Tuple[str, int], Tuple[float, float]],
    link_counts: Dict[Tuple[str, int, str, int], int],
) -> Dict[Tuple[str, int, str, int], Tuple[Tuple[float, float], Tuple[float, float]]]:
    """Calculate slice positions for each link."""
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
        items.sort(key=lambda kv: (ALPHABET_ORDER.get(kv[0][2], 99), -kv[1]))
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
        items.sort(key=lambda kv: (ALPHABET_ORDER.get(kv[0][0], 99), -kv[1]))
        for (a, ta, b, tb), v in items:
            y_pair = link_slices.get((a, ta, b, tb))
            if y_pair is None:
                continue
            h = H * (v / total)
            link_slices[(a, ta, b, tb)] = (y_pair[0], (y0 + off, y0 + off + h))
            off += h

    return link_slices


def bezier_band(x0, x1, y0a, y1a, y0b, y1b, curvature=0.35) -> MplPath:
    """Create a bezier band path for flow visualization."""
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
        MplPath.MOVETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.LINETO,
        MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
        MplPath.CLOSEPOLY,
    ]
    return MplPath(verts, codes)


def scale_rgba(color_rgba: Tuple[float, float, float, float], alpha_scale: float) -> Tuple[float, float, float, float]:
    """Scale the alpha channel of an RGBA color."""
    r, g, b, a = color_rgba
    a2 = max(0.05, min(0.95, a * alpha_scale))
    return (r, g, b, a2)


# ----------------------- Drawing Functions -----------------------

def draw_transition_midaxis(mid_ax, shared_max_t: int):
    """Draw a thin middle axis with a double-headed arrow and 0..K labels."""
    mid_ax.set_ylim(0, 1)
    mid_ax.axis("off")

    # Match x-lims used in the sankey axes
    x_margin = NODE_WIDTH / 2 + 0.02
    x_last = LEFT_MARGIN + shared_max_t * COL_SPACING
    mid_ax.set_xlim(LEFT_MARGIN - x_margin, x_last + x_margin)

    # Arrow across full usable span
    mid_ax.annotate(
        "",
        xy=(x_last + x_margin * 0.6, 0.5),
        xytext=(LEFT_MARGIN - x_margin * 0.6, 0.5),
        arrowprops=dict(arrowstyle="-|>", lw=1.6, color=(0, 0, 0, 0.65))
    )

    # Ticks: 0..K aligned to time columns
    for t in range(shared_max_t + 1):
        x = LEFT_MARGIN + t * COL_SPACING
        mid_ax.plot([x, x], [0.35, 0.65], color=(0, 0, 0, 0.65), lw=1.2)
        mid_ax.text(x, 0.10, f"{t}", ha="center", va="center",
                    fontsize=LABEL_SIZE + 1, color=(0, 0, 0, 0.8), weight="normal")

    mid_ax.text(
        (LEFT_MARGIN + shared_max_t * COL_SPACING)/2, 0.88,
        "Transition index", ha="center", va="center",
        fontsize=LABEL_SIZE + 2, color=(0, 0, 0, 0.75), weight="normal"
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
    x_margin = NODE_WIDTH / 2 + 0.02
    x_last = LEFT_MARGIN + use_max_t * COL_SPACING
    ax.set_xlim(LEFT_MARGIN - x_margin, x_last + x_margin)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = {t: LEFT_MARGIN + t * COL_SPACING for t in range(use_max_t + 1)}

    # Layout
    node_spans = layout_columns(node_volume, max_t)
    link_slices = flow_slices_from_spans(node_spans, link_counts)

    # Faint column guides (no numbers)
    for t in range(use_max_t + 1):
        x = x_positions[t]
        ax.plot([x, x], [0, 1], color=(0, 0, 0, 0.04), linewidth=0.8, zorder=0)

    # Links
    for (a, ta, b, tb), v in sorted(
        link_counts.items(), key=lambda kv: (kv[0][1], ALPHABET_ORDER.get(kv[0][0], 99), -kv[1])
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

    # Nodes + alphabet letters
    for (p, t), (y0, y1) in node_spans.items():
        x = LEFT_MARGIN + t * COL_SPACING
        half_width = NODE_WIDTH / 2
        ax.add_patch(plt.Rectangle((x - half_width, y0), NODE_WIDTH, y1 - y0,
                                   facecolor=PASTEL.get(p, (0.8, 0.8, 0.8, 0.9)),
                                   edgecolor=(0, 0, 0, 0.15),
                                   linewidth=0.5, zorder=2))
        # Use abbreviated labels for display
        display_label = ALPHABET_ABBR.get(p, p)
        ax.text(x, (y0 + y1) / 2, f"{display_label}",
                ha="center", va="center", fontsize=LABEL_SIZE,
                color=(0, 0, 0, 0.85), weight="normal", zorder=3)

    # Panel title
    ax.text(0.5, 1.03, title, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=TITLE_SIZE, weight="bold")


def draw_single_sankey(
    title: str,
    node_volume: Dict[Tuple[str, int], int],
    link_counts: Dict[Tuple[str, int, str, int], int],
    max_t: int,
    col_total: Dict[int, int],
    out_path: Path,
):
    """Draw a standalone sankey diagram for a single agent/model."""
    fig, ax = plt.subplots(figsize=(FIG_W * 0.85, FIG_H * 0.65), constrained_layout=False)
    x_margin = NODE_WIDTH / 2 + 0.02
    x_last = LEFT_MARGIN + max_t * COL_SPACING
    ax.set_xlim(LEFT_MARGIN - x_margin, x_last + x_margin)
    ax.set_ylim(0, 1)
    ax.axis("off")

    x_positions = {t: LEFT_MARGIN + t * COL_SPACING for t in range(max_t + 1)}

    # Layout
    node_spans = layout_columns(node_volume, max_t)
    link_slices = flow_slices_from_spans(node_spans, link_counts)

    # Column guides & shared iteration labels
    for t in range(max_t + 1):
        x = x_positions[t]
        ax.plot([x, x], [0, 1], color=(0, 0, 0, 0.04), linewidth=0.8, zorder=0)
        ax.text(x, -BOTTOM_MARGIN/2, f"{t}", ha="center", va="top",
                fontsize=LABEL_SIZE + 3, color=(0, 0, 0, 0.75), transform=ax.transData)

    # Draw links: color by source alphabet; alpha scales with column share
    for (a, ta, b, tb), v in sorted(link_counts.items(), key=lambda kv: (kv[0][1], ALPHABET_ORDER.get(kv[0][0], 99), -kv[1])):
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

    # Node blocks + alphabet letters
    for (p, t), (y0, y1) in node_spans.items():
        x = x_positions[t]
        half_width = NODE_WIDTH / 2
        ax.add_patch(plt.Rectangle((x - half_width, y0), NODE_WIDTH, y1 - y0,
                                   facecolor=PASTEL.get(p, (0.8, 0.8, 0.8, 0.9)),
                                   edgecolor=(0, 0, 0, 0.15),
                                   linewidth=0.5, zorder=2))
        display_label = ALPHABET_ABBR.get(p, p)
        ax.text(x, (y0 + y1) / 2, f"{display_label}",
                ha="center", va="center", fontsize=LABEL_SIZE + 2,
                color=(0, 0, 0, 0.85), weight="normal", zorder=3)

    # Title
    ax.text(0.5, 1 - TOP_MARGIN/2.5,
            f"{title}",
            ha="center", va="top", transform=fig.transFigure,
            fontsize=TITLE_SIZE + 4, weight="bold")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


# ----------------------- Main Processing -----------------------

def generate_sankey_multi(
    base_data_dir: Path,
    output_dir: Path
) -> None:
    """
    Generate unified Sankey diagram grid for all agents and models.

    Output: {output_dir}/lang_sankey/all_sankey.pdf
    """
    fig = plt.figure(figsize=(FIG_W, FIG_H), constrained_layout=False)

    # 3 rows: top (SA), middle (arrow/ticks), bottom (OH)
    gs = GridSpec(
        3, 4, figure=fig,
        height_ratios=[1.0, 0.14, 1.0],
        left=0.02, right=0.98,
        top=0.94, bottom=0.04,
        wspace=0.08,
        hspace=0.20
    )

    # Precompute data & shared max_t per column (model)
    shared_max_t_by_col = {}
    data_by_cell = {}

    for col, model in enumerate(DISPLAY_MODELS):
        # SWE-agent
        lang_path_sa = base_data_dir / "SWE-agent" / "langs" / model / "languatory.json"
        if lang_path_sa.exists():
            seq_sa = load_languatory(lang_path_sa)
            nv_sa, lc_sa, mt_sa, ct_sa = build_link_counts(seq_sa, max_transitions=10)
            data_by_cell[(0, col)] = (nv_sa, lc_sa, mt_sa, ct_sa)
            print(f"[INFO] Loaded SWE-agent/{model}: {len(seq_sa)} sequences")
        else:
            data_by_cell[(0, col)] = ({}, {}, 0, {})
            print(f"[WARN] Not found: {lang_path_sa}")

        # OpenHands
        lang_path_oh = base_data_dir / "OpenHands" / "langs" / model / "languatory.json"
        if lang_path_oh.exists():
            seq_oh = load_languatory(lang_path_oh)
            nv_oh, lc_oh, mt_oh, ct_oh = build_link_counts(seq_oh, max_transitions=10)
            data_by_cell[(2, col)] = (nv_oh, lc_oh, mt_oh, ct_oh)
            print(f"[INFO] Loaded OpenHands/{model}: {len(seq_oh)} sequences")
        else:
            data_by_cell[(2, col)] = ({}, {}, 0, {})
            print(f"[WARN] Not found: {lang_path_oh}")

        # Shared max_t
        mt_sa = data_by_cell[(0, col)][2]
        mt_oh = data_by_cell[(2, col)][2]
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

    # Shared legend (alphabet colors with full labels)
    handles = [
        mpatches.Patch(facecolor=PASTEL[role], label=role,
                      edgecolor=(0, 0, 0, 0.25), linewidth=0.5)
        for role in ALPHABETS
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(ALPHABETS),
        frameon=False,
        fontsize=LABEL_SIZE + 1,
        bbox_to_anchor=(0.5, 0.99),
        borderaxespad=0,
        columnspacing=1.0,
        handlelength=1.3,
        handleheight=1.0
    )

    out_path = output_dir / "lang_sankey" / "all_sankey.pdf"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Wrote {out_path}")


def generate_sankey_single(
    lang_path: Path,
    output_dir: Path,
    agent: str,
    model: str
) -> None:
    """
    Generate Sankey diagram for a single agent/model.

    Output: {output_dir}/lang_sankey/{agent}_{model}_sankey.pdf
    """
    print(f"[INFO] Loading {lang_path}...")
    sequences = load_languatory(lang_path)

    if not sequences:
        print(f"[ERROR] No valid sequences found in {lang_path}")
        return

    print(f"[INFO] Loaded {len(sequences)} sequences")

    node_volume, link_counts, max_t, col_total = build_link_counts(sequences, max_transitions=10)

    model_abbr = MODEL_ABBR.get(model, model)
    title = format_title(agent, model_abbr)

    out_path = output_dir / "lang_sankey" / f"{agent}_{model}_sankey.pdf"

    draw_single_sankey(
        title=title,
        node_volume=node_volume,
        link_counts=link_counts,
        max_t=max_t,
        col_total=col_total,
        out_path=out_path
    )

    print(f"[OK] Wrote {out_path}")


# ----------------------- Main Entry Point -----------------------

def main(
    lang_path: Optional[str] = None,
    agent: Optional[str] = None,
    model: Optional[str] = None,
    output_dir: Optional[str] = None
) -> None:
    """
    Main entry point.

    Args:
        lang_path: Path to languatory.json file (overrides agent/model)
        agent: Agent name (used with model to build default path)
        model: Model name (used with agent to build default path)
        output_dir: Output directory (defaults to data/ or figures/)
    """
    # Determine mode
    is_single_mode = lang_path is not None or (agent is not None and model is not None)

    # Resolve output directory
    if output_dir is None:
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        output_path = project_root / "figures"
    else:
        output_path = Path(output_dir)

    if is_single_mode:
        # Single mode
        if lang_path:
            lang_file = Path(lang_path)
        elif agent and model:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            lang_file = project_root / "data" / agent / "langs" / model / "languatory.json"
        else:
            print("[ERROR] Single mode requires either --lang-path or both --agent and --model")
            return

        if not lang_file.exists():
            print(f"[ERROR] File not found: {lang_file}")
            return

        # Infer agent/model from path if not provided
        if not agent or not model:
            parts = lang_file.parts
            try:
                langs_idx = parts.index("langs")
                if langs_idx >= 1:
                    agent = agent or parts[langs_idx - 1]
                    model = model or parts[langs_idx + 1]
            except (ValueError, IndexError):
                agent = agent or "Unknown"
                model = model or "Unknown"

        print(f"[INFO] Mode: Single ({agent}/{model})")
        print(f"[INFO] Lang file: {lang_file}")
        print(f"[INFO] Output directory: {output_path}")
        print()

        generate_sankey_single(
            lang_path=lang_file,
            output_dir=output_path,
            agent=agent,
            model=model
        )
    else:
        # Multi mode
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        base_data_dir = project_root / "data"

        if not base_data_dir.exists():
            print(f"[ERROR] Data directory not found: {base_data_dir}")
            return

        print(f"[INFO] Mode: Multi-agent/model")
        print(f"[INFO] Data directory: {base_data_dir}")
        print(f"[INFO] Output directory: {output_path}")
        print()

        generate_sankey_multi(
            base_data_dir=base_data_dir,
            output_dir=output_path
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Sankey diagrams for language transition visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Multi-mode: Scan all agents/models (default)
  # Output: figures/lang_sankey/all_sankey.pdf
  python sankey_lang_plot.py

  # Single-mode: Specific languatory.json file
  # Output: figures/lang_sankey/SWE-agent_deepseek-v3_sankey.pdf
  python sankey_lang_plot.py --lang-path data/SWE-agent/langs/deepseek-v3/languatory.json

  # Single-mode: Use agent and model (default path)
  # Output: results/lang_sankey/OpenHands_deepseek-v3_sankey.pdf
  python sankey_lang_plot.py --agent OpenHands --model deepseek-v3 --output-dir ./results

  # Single-mode: Test with sample data
  python sankey_lang_plot.py --agent SWE-agent --model deepseek-v3 \\
                             --lang-path data/samples/SWE-agent/langs/deepseek-v3/languatory.json
        """
    )
    parser.add_argument(
        "--lang-path",
        type=str,
        help="Path to languatory.json file (overrides --agent and --model)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Agent name (used with --model to build default path)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (used with --agent to build default path)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: figures/)"
    )

    args = parser.parse_args()

    main(
        lang_path=args.lang_path,
        agent=args.agent,
        model=args.model,
        output_dir=args.output_dir
    )
