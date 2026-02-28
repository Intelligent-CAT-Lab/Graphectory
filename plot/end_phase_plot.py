#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Start/End PHASE donut plots (inner=Resolved, outer=Unresolved), 2×4 grid:
- Row 1 = SWE-agent + {models}, Row 2 = OpenHands + {models}
- end_phases_donuts.png   — LAST  non-general phase
- Exactly 3 canonical phases: localization (L), patch (P), validation (V)

Modified:
  - Labels are now drawn ON the rings (both inner and outer); no leader lines.
  - Ring widths are equal and larger.
  - Subplot titles now use full agent name with model abbreviation as math subscript.
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path as FilePath
from collections import Counter
from typing import Dict, Iterable, Tuple, List, Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ------------ Config ------------
AGENTS = ["SWE-agent", "OpenHands"]
MODELS = [
    "deepseek/deepseek-chat",
    "openrouter/deepseek/deepseek-r1-0528",
    "openrouter/mistralai/devstral-small",
    "openrouter/anthropic/claude-sonnet-4",
]
AGENT_ABBR = {"SWE-agent": "SA", "OpenHands": "OH"}  # kept for compatibility; not used in final titles
MODEL_ABBR = {
    "deepseek/deepseek-chat": "DSK-V3",
    "openrouter/mistralai/devstral-small": "Dev",
    "openrouter/deepseek/deepseek-r1-0528": "DSK-R1",
    "openrouter/anthropic/claude-sonnet-4": "CLD-4",
}

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype']  = 42
plt.rcParams["axes.titlelocation"] = "center"

# ------------ Paths ------------
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

# ------------ Phase Canonicalization ------------
PHASE_ORDER = ["localization", "patch", "validation"]
ABBR = {"localization": "L", "patch": "P", "validation": "V"}
DISPLAY = {"localization": "Localization", "patch": "Patching", "validation": "Validation"}

def canon_phase(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s2 = str(s).strip().lower()
    return s2

# ------------ JSON utilities ------------
def _iter_json_files(root: FilePath) -> Iterable[FilePath]:
    """Recursively find all JSON files in the graph directory."""
    for json_file in root.rglob("*.json"):
        if any(skip in json_file.parts for skip in {"analysis", ".git", "__pycache__"}):
            continue
        yield json_file

def _iter_nodes(graph_json: dict) -> Iterable[dict]:
    nodes = None
    if isinstance(graph_json, dict):
        g = graph_json.get("graph", {})
        if isinstance(g, dict) and "nodes" in g:
            nodes = g["nodes"]
        elif "nodes" in graph_json:
            nodes = graph_json["nodes"]
    if isinstance(nodes, dict):
        for _, nd in nodes.items():
            if isinstance(nd, dict):
                yield nd
    elif isinstance(nodes, list):
        for nd in nodes:
            if isinstance(nd, dict):
                yield nd

def extract_step_phase_triples(graph_json: dict) -> List[Tuple[int, Optional[str]]]:
    triples: List[Tuple[int, Optional[str]]] = []
    for node in _iter_nodes(graph_json):
        raw_steps = node.get("step_indices") or []
        phases_field = node.get("phases", node.get("phase", None))

        # normalize steps
        steps: List[int] = []
        if isinstance(raw_steps, int):
            steps = [raw_steps]
        elif isinstance(raw_steps, (list, tuple)):
            for it in raw_steps:
                try:
                    steps.append(int(it))
                except Exception:
                    pass
        if not steps:
            continue

        if isinstance(phases_field, list):
            if len(phases_field) != len(steps):
                continue
            for idx, ph in zip(steps, phases_field):
                triples.append((idx, canon_phase(ph)))
        else:
            ph = canon_phase(phases_field)
            for idx in steps:
                triples.append((idx, ph))
    triples.sort(key=lambda t: t[0])
    return triples

def first_non_general_phase(graph_json: dict) -> Optional[str]:
    for _, ph in extract_step_phase_triples(graph_json):
        if ph and ph != "general":
            return ph if ph in PHASE_ORDER else None
    return None

def last_non_general_phase(graph_json: dict) -> Optional[str]:
    last = None
    for _, ph in extract_step_phase_triples(graph_json):
        if ph and ph != "general":
            if ph in PHASE_ORDER:
                last = ph
    return last

# ------------ Aggregation ------------
def build_phase_counters(data_dir: FilePath, agent: str, model: str, which: str = "start") -> Tuple[Counter, Counter]:
    """
    Build phase counters (resolved/unresolved) for a given agent-model pair.

    Args:
        data_dir: Base data directory
        agent: Agent name
        model: Model identifier
        which: "start" for first phase, "end" for last phase

    Returns:
        Tuple of (resolved_counter, unresolved_counter)
    """
    root = find_graph_root(data_dir, agent, model)
    res = Counter()
    unres = Counter()
    chooser = first_non_general_phase if which == "start" else last_non_general_phase

    for fpath in _iter_json_files(root):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        ph = chooser(data)
        if not ph:
            continue

        resolution = str(
            data.get("graph", {}).get("resolution_status", data.get("resolution_status", "unknown"))
        ).strip().lower()

        if resolution.startswith("res"):
            res[ph] += 1
        elif resolution.startswith("unres"):
            unres[ph] += 1

    return res, unres

# ------------ Colors ------------
PASTEL_COLORS = {
    "localization": "#B6AEF2",  # bright lilac (L)
    "patch":        "#F2C24A",  # bright warm yellow (P)
    "validation":   "#A9DA84",  # bright light green (V)
}

# ------------ Label helpers ------------
def _draw_labels_on_ring(ax, wedges, sizes, abbrs, inner_r, outer_r, pct_thresh: float, color="#0d1b24"):
    """
    Place labels directly on the ring (no leader lines), for slices whose
    percentage >= pct_thresh.
    """
    total = float(sum(sizes)) if sum(sizes) > 0 else 1.0
    r_text = (inner_r + outer_r) * 0.5
    for w, sz, txt in zip(wedges, sizes, abbrs):
        if sz <= 0:
            continue
        pct = 100.0 * sz / total
        if pct >= pct_thresh:
            theta = 0.5 * (w.theta1 + w.theta2)
            ang = np.deg2rad(theta)
            ax.text(
                r_text * np.cos(ang),
                r_text * np.sin(ang),
                txt,
                ha="center", va="center",
                fontsize=16.0, color=color,
                bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9),
            )

# ------------ Title helper (subscript model) ------------
def format_title(agent_name: str, model_abbr: str) -> str:
    # e.g. "SWE-agent$_{\mathrm{DSK\text{-}V3}}$"
    safe_model = model_abbr.replace("-", r"\text{-}")
    return rf"{agent_name}$_{{\mathrm{{{safe_model}}}}}$"

# ------------ Plotting ------------
def plot_phase_donuts(per_unit: Dict[Tuple[str, str], Tuple[Counter, Counter]],
                      fig_title: str,
                      output_path: FilePath) -> None:
    labels = PHASE_ORDER
    colors = [PASTEL_COLORS[k] for k in labels]

    # Grid & sizing (same rows/cols)
    ncols, nrows = len(MODELS), 2
    fig_w, fig_h = 14.6, 7.0
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    plt.subplots_adjust(wspace=0.001, hspace=0.001)

    for ax in axes.ravel():
        ax.set_anchor("C")
        ax.margins(x=0.0, y=0.0)

    # --- RING GEOMETRY (equal width, thick like screenshot) ---
    OUTER_R = 1.30        # outer radius of unresolved ring
    RING_W  = 0.40        # thickness of each ring
    INNER_R = OUTER_R - (RING_W)  # inner ring sits clearly inside
    # Explanation:
    #   OUTER ring: radius OUTER_R, width RING_W
    #   INNER ring: radius INNER_R, width RING_W
    # both rings use same width RING_W

    # label thresholds (% of that ring)
    PCT_THRESH_INNER = 5.0   # draw label on ring if >= this %
    PCT_THRESH_OUTER = 5.0

    def draw(ax, ctr_res: Counter, ctr_unr: Counter, agent_name: str, model_name: str):
        sizes_res = [ctr_res.get(k, 0) for k in labels]
        sizes_unr = [ctr_unr.get(k, 0) for k in labels]
        tot_res, tot_unr = sum(sizes_res), sum(sizes_unr)

        if tot_res == 0 and tot_unr == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", fontsize=13.0)
            ax.set_axis_off()
            return

        # OUTER ring = unresolved
        outer_wedges = ax.pie(
            sizes_unr, radius=OUTER_R, startangle=90,
            wedgeprops=dict(width=RING_W, edgecolor="white", linewidth=0.8),
            colors=colors,
        )[0]

        # INNER ring = resolved
        inner_wedges = ax.pie(
            sizes_res, radius=INNER_R, startangle=90,
            wedgeprops=dict(width=RING_W, edgecolor="white", linewidth=0.8),
            colors=colors,
        )[0]

        ax.set(aspect="equal")

        # Subplot title with subscript model
        ax.set_title(
            format_title(agent_name, model_name),
            fontsize=18.5,
            pad=0.6,
            # fontweight="bold",
        )

        # Center text (resolved count)
        ax.text(
            0, 0,
            f"Res:{tot_res}",
            ha="center", va="center",
            fontsize=17.5, color="#1d232a"
        )

        abbrs = [ABBR[k] for k in labels]

        # Label directly ON INNER ring
        _draw_labels_on_ring(
            ax,
            inner_wedges,
            sizes_res,
            abbrs,
            INNER_R - RING_W,
            INNER_R,
            pct_thresh=PCT_THRESH_INNER,
            color="#0d1b24",
        )

        # Label directly ON OUTER ring
        _draw_labels_on_ring(
            ax,
            outer_wedges,
            sizes_unr,
            abbrs,
            OUTER_R - RING_W,
            OUTER_R,
            pct_thresh=PCT_THRESH_OUTER,
            color="#0d1b24",
        )

    # Row 0: SWE-agent, Row 1: OpenHands
    for ci, model in enumerate(MODELS):
        ctrs = per_unit.get(("SWE-agent", model), (Counter(), Counter()))
        draw(
            axes[0, ci],
            ctrs[0],
            ctrs[1],
            "SWE-agent",
            MODEL_ABBR[model],
        )
    for ci, model in enumerate(MODELS):
        ctrs = per_unit.get(("OpenHands", model), (Counter(), Counter()))
        draw(
            axes[1, ci],
            ctrs[0],
            ctrs[1],
            "OpenHands",
            MODEL_ABBR[model],
        )

    # Caption & legend
    fig.text(
        0.5, 0.968,
        "Inner = Resolved • Outer = Unresolved",
        ha="center", va="center",
        fontsize=18, color="#2a3340"
    )

    handles = [
        plt.Line2D(
            [0],[0],
            marker="o", linestyle="",
            markerfacecolor=PASTEL_COLORS[k],
            markeredgecolor="none",
            markersize=8.4,
            label=DISPLAY[k]
        )
        for k in labels
    ]

    fig.tight_layout(rect=[0.02, 0.12, 0.99, 0.96])
    fig.legend(
        handles=handles,
        loc="lower center", bbox_to_anchor=(0.5, 0.058),
        ncol=3, frameon=False, fontsize=14.0,
        handlelength=1.0, columnspacing=1.0, labelspacing=0.28,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"[OK] Saved: {output_path}")

# ------------ Driver ------------
def main() -> None:
    """CLI entry point for generating end phase donut diagrams."""
    parser = argparse.ArgumentParser(
        description="Generate end phase donut diagrams comparing agents and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate end phase donuts using default data and figures directories
  python plot/end_phase_plot.py

  # Specify custom data directory
  python end_phase_plot.py --data-dir ./my_data

  # Specify custom output path
  python end_phase_plot.py --output figures/custom_end_phases.png
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
        default=FilePath("figures/end_phases_donuts.png"),
        help="Output file path (default: figures/end_phases_donuts.png)"
    )

    args = parser.parse_args()

    # Build per-unit counters for END phases
    per_unit_end: Dict[Tuple[str, str], Tuple[Counter, Counter]] = {}
    for agent in AGENTS:
        for model in MODELS:
            try:
                per_unit_end[(agent, model)] = build_phase_counters(
                    args.data_dir, agent, model, which="end"
                )
            except (FileNotFoundError, ValueError) as e:
                print(f"[WARN] Skipping ({agent}, {model}): {e}")
                continue

    plot_phase_donuts(per_unit_end, fig_title="End Phases", output_path=args.output)

if __name__ == "__main__":
    main()
