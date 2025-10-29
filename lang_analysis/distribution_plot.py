#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Distribution of languatory alphabets among resolved/unresolved across agents and models.

Visualizes the distribution of language alphabets (L_reproduce, L_navigate, P,
V_newly_generated_test, V_regression_test) for resolved vs. unresolved instances.

Input modes:
  1. Multi-mode (default): Scans all agents/models
  2. Agent-specific mode (--agent): Processes all default models for given agent
  3. Single-mode (--lang-path or --agent + --model): Processes specific file

Output directory structure (defaults to figures/):
  - Multi-mode: {output_dir}/lang_distribution/all_pairs_distribution.pdf
  - Agent-specific: {output_dir}/lang_distribution/{agent}_distribution.pdf
  - Single-mode: {output_dir}/lang_distribution/{agent}_{model}_distribution.pdf
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


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

# Language alphabets (without termination)
ALPHABETS = [
    "L_reproduce",
    "L_navigate",
    "P",
    "V_newly_generated_test",
    "V_regression_test",
]

# Abbreviated labels for display
ALPHABET_ABBR = {
    "L_reproduce": "L_repr",
    "L_navigate": "L_nav",
    "P": "P",
    "V_newly_generated_test": "V_new",
    "V_regression_test": "V_reg",
}

# Color scheme
COLOR_RESOLVED = "#66BB6A"      # Green
COLOR_UNRESOLVED = "#EF5350"    # Red


# ----------------------- Data Loading -----------------------

def load_languatory(lang_path: Path) -> List[Tuple[str, List[str]]]:
    """
    Load languatory.json and extract sequences with resolution status.

    Args:
        lang_path: Path to languatory.json file

    Returns:
        List of (resolution_status, sequence) tuples where sequence is a list
        of alphabet symbols (ignoring run_length suffix after underscore)
    """
    try:
        with open(lang_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[ERROR] Failed to load {lang_path}: {e}")
        return []

    results = []

    for entry in data:
        if not isinstance(entry, dict):
            continue

        resolution_status = str(entry.get("resolution_status", "")).strip().lower()
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
            results.append((resolution_status, sequence))

    return results


def build_alphabet_counters(
    lang_path: Path,
) -> Tuple[Counter, Counter, int, int]:
    """
    Build counters for alphabet presence distribution.

    For each sequence, detects whether each alphabet exists (presence/absence).
    Counts the number of instances where each alphabet appears.

    Args:
        lang_path: Path to languatory.json file

    Returns:
        resolved_counter: Count of resolved instances containing each alphabet
        unresolved_counter: Count of unresolved instances containing each alphabet
        total_resolved: Total count of resolved instances
        total_unresolved: Total count of unresolved instances
    """
    resolved_counter = Counter()
    unresolved_counter = Counter()
    total_resolved = 0
    total_unresolved = 0

    sequences = load_languatory(lang_path)

    for resolution_status, sequence in sequences:
        # Get unique alphabets in this sequence (presence/absence detection)
        unique_alphabets = set(sequence)

        # Update counters based on resolution status
        if resolution_status.startswith("res"):
            total_resolved += 1
            for alphabet in unique_alphabets:
                resolved_counter[alphabet] += 1
        elif resolution_status.startswith("unres"):
            total_unresolved += 1
            for alphabet in unique_alphabets:
                unresolved_counter[alphabet] += 1

    return resolved_counter, unresolved_counter, total_resolved, total_unresolved


# ----------------------- Plotting Functions -----------------------

def format_title(agent_name: str, model_abbr: str) -> str:
    """
    Render 'Agent_{Model}' with the model as a math subscript.
    Example: SWE-agent$_{\mathrm{DSK-V3}}$
    """
    safe_model = model_abbr.replace("-", r"\text{-}")
    return rf"{agent_name}$_{{\mathrm{{{safe_model}}}}}$"


def draw_distribution_bars(
    ax,
    resolved_counter: Counter,
    unresolved_counter: Counter,
    total_resolved: int,
    total_unresolved: int,
    agent_name: str,
    model_name: str
):
    """
    Draw grouped bar chart showing alphabet distribution.

    Args:
        ax: Matplotlib axes
        resolved_counter: Count of resolved instances containing each alphabet
        unresolved_counter: Count of unresolved instances containing each alphabet
        total_resolved: Total number of resolved instances
        total_unresolved: Total number of unresolved instances
        agent_name: Agent name for title
        model_name: Model abbreviation for title
    """
    if total_resolved == 0 and total_unresolved == 0:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                fontsize=12.0, transform=ax.transAxes)
        ax.set_axis_off()
        return

    # Calculate percentages
    resolved_pcts = [100.0 * resolved_counter.get(a, 0) / total_resolved if total_resolved > 0 else 0
                     for a in ALPHABETS]
    unresolved_pcts = [100.0 * unresolved_counter.get(a, 0) / total_unresolved if total_unresolved > 0 else 0
                       for a in ALPHABETS]

    x = np.arange(len(ALPHABETS))
    width = 0.35

    # Draw bars
    bars1 = ax.bar(x - width/2, resolved_pcts, width, label='Resolved',
                   color=COLOR_RESOLVED, edgecolor='white', linewidth=0.5)
    bars2 = ax.bar(x + width/2, unresolved_pcts, width, label='Unresolved',
                   color=COLOR_UNRESOLVED, edgecolor='white', linewidth=0.5)

    # Styling
    ax.set_ylabel('Percentage (%)', fontsize=11.0)
    ax.set_title(format_title(agent_name, model_name), fontsize=13.0, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels([ALPHABET_ABBR[a] for a in ALPHABETS], fontsize=10.0)
    ax.set_ylim(0, 105)
    ax.tick_params(axis='y', labelsize=9.5)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)

    # Add value labels on bars (only if > 5%)
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 5:
                ax.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                        f'{height:.0f}%',
                        ha='center', va='bottom', fontsize=8.0, color='#333')


def plot_distribution_grid(
    data_by_cell: Dict[Tuple[int, int], Tuple[Counter, Counter, int, int]],
    agents_to_process: List[str],
    output_path: Path
):
    """
    Plot a grid of bar charts for agents and models.

    Args:
        data_by_cell: Dictionary mapping (row, col) to (resolved_counter,
                      unresolved_counter, total_resolved, total_unresolved)
        agents_to_process: List of agent names
        output_path: Output file path
    """
    nrows = len(agents_to_process)
    ncols = len(DISPLAY_MODELS)

    fig_w, fig_h = 16.0, 4.0 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    plt.subplots_adjust(wspace=0.25, hspace=0.35, left=0.06, right=0.98, top=0.95, bottom=0.05)

    # Draw grid
    for row, agent in enumerate(agents_to_process):
        for col, model in enumerate(DISPLAY_MODELS):
            ax = axes[row, col]
            resolved_ctr, unresolved_ctr, total_res, total_unres = data_by_cell.get(
                (row, col),
                (Counter(), Counter(), 0, 0)
            )

            model_abbr = MODEL_ABBR[model]
            draw_distribution_bars(ax, resolved_ctr, unresolved_ctr, total_res, total_unres, agent, model_abbr)

            # Add legend only to first subplot
            if row == 0 and col == 0:
                ax.legend(loc='upper left', fontsize=10.0, framealpha=0.9)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] Wrote {output_path}")


# ----------------------- Main Processing -----------------------

def generate_distribution_multi(
    base_data_dir: Path,
    output_dir: Path,
    target_agent: Optional[str] = None
) -> None:
    """
    Generate unified distribution bar chart grid for agents and models.

    Args:
        base_data_dir: Base data directory
        output_dir: Output directory
        target_agent: If specified, only process this agent; otherwise process all agents

    Output:
        - All agents: {output_dir}/lang_distribution/all_pairs_distribution.pdf
        - Single agent: {output_dir}/lang_distribution/{agent}_distribution.pdf
    """
    # Determine which agents to process
    agents_to_process = [target_agent] if target_agent else AGENTS

    # Precompute data per cell
    data_by_cell = {}

    for row, agent in enumerate(agents_to_process):
        for col, model in enumerate(DISPLAY_MODELS):
            lang_path = base_data_dir / agent / "langs" / model / "languatory.json"

            if lang_path.exists():
                resolved_ctr, unresolved_ctr, total_res, total_unres = (
                    build_alphabet_counters(lang_path)
                )
                data_by_cell[(row, col)] = (
                    resolved_ctr, unresolved_ctr, total_res, total_unres
                )
                print(
                    f"[INFO] Loaded {agent}/{model}: "
                    f"{total_res} resolved, {total_unres} unresolved"
                )
            else:
                data_by_cell[(row, col)] = (Counter(), Counter(), 0, 0)
                print(f"[WARN] Not found: {lang_path}")

    # Determine output filename
    if target_agent:
        out_filename = f"{target_agent}_distribution.pdf"
    else:
        out_filename = "all_pairs_distribution.pdf"

    output_path = output_dir / "lang_distribution" / out_filename

    plot_distribution_grid(data_by_cell, agents_to_process, output_path)


def generate_distribution_single(
    lang_path: Path,
    output_dir: Path,
    agent: str,
    model: str
) -> None:
    """
    Generate distribution bar chart for a single agent/model.

    Output: {output_dir}/lang_distribution/{agent}_{model}_distribution.pdf
    """
    print(f"[INFO] Loading {lang_path}...")

    resolved_ctr, unresolved_ctr, total_res, total_unres = (
        build_alphabet_counters(lang_path)
    )

    if total_res == 0 and total_unres == 0:
        print(f"[ERROR] No valid data found in {lang_path}")
        return

    print(
        f"[INFO] Loaded: {total_res} resolved, {total_unres} unresolved instances"
    )

    # Create single plot
    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    model_abbr = MODEL_ABBR.get(model, model)
    draw_distribution_bars(ax, resolved_ctr, unresolved_ctr, total_res, total_unres, agent, model_abbr)
    ax.legend(loc='upper left', fontsize=11.0, framealpha=0.9)

    plt.tight_layout(pad=1.5)

    output_path = output_dir / "lang_distribution" / f"{agent}_{model}_distribution.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"[OK] Wrote {output_path}")


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
        agent: Agent name (if model given: single mode; else: agent-specific mode)
        model: Model name (used with agent for single mode)
        output_dir: Output directory (defaults to figures/)
    """
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    output_path = Path(output_dir) if output_dir else project_root / "figures"

    # Determine mode based on parameters
    if lang_path:
        # Explicit file mode
        lang_file = Path(lang_path)
        if not lang_file.exists():
            print(f"[ERROR] File not found: {lang_file}")
            return

        # Infer agent/model from path
        parts = lang_file.parts
        try:
            langs_idx = parts.index("langs")
            agent = agent or (parts[langs_idx - 1] if langs_idx >= 1 else "Unknown")
            model = model or (parts[langs_idx + 1] if langs_idx + 1 < len(parts) else "Unknown")
        except (ValueError, IndexError):
            agent = agent or "Unknown"
            model = model or "Unknown"

        print(f"[INFO] Mode: Single file ({agent}/{model})")
        print(f"[INFO] Lang file: {lang_file}")
        print(f"[INFO] Output directory: {output_path}\n")

        generate_distribution_single(lang_file, output_path, agent, model)

    elif agent and model:
        # Single agent/model mode
        base_data_dir = project_root / "data"
        lang_file = base_data_dir / agent / "langs" / model / "languatory.json"

        if not lang_file.exists():
            print(f"[ERROR] File not found: {lang_file}")
            return

        print(f"[INFO] Mode: Single agent/model ({agent}/{model})")
        print(f"[INFO] Lang file: {lang_file}")
        print(f"[INFO] Output directory: {output_path}\n")

        generate_distribution_single(lang_file, output_path, agent, model)

    elif agent:
        # Agent-specific multi-model mode
        base_data_dir = project_root / "data"

        if not (base_data_dir / agent).exists():
            print(f"[ERROR] Agent directory not found: {base_data_dir / agent}")
            return

        print(f"[INFO] Mode: Agent-specific multi-model ({agent})")
        print(f"[INFO] Data directory: {base_data_dir}")
        print(f"[INFO] Output directory: {output_path}\n")

        generate_distribution_multi(base_data_dir, output_path, target_agent=agent)

    else:
        # Full multi-agent/model mode
        base_data_dir = project_root / "data"

        if not base_data_dir.exists():
            print(f"[ERROR] Data directory not found: {base_data_dir}")
            return

        print(f"[INFO] Mode: Full multi-agent/model")
        print(f"[INFO] Data directory: {base_data_dir}")
        print(f"[INFO] Output directory: {output_path}\n")

        generate_distribution_multi(base_data_dir, output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate distribution bar charts for language alphabet analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Full multi-mode: All agents/models (default)
  #    Output: figures/lang_distribution/all_pairs_distribution.pdf
  python distribution_plot.py

  # 2. Agent-specific mode: All models for one agent
  #    Output: figures/lang_distribution/OpenHands_distribution.pdf
  python distribution_plot.py --agent OpenHands

  # 3. Single agent/model mode
  #    Output: figures/lang_distribution/OpenHands_deepseek-v3_distribution.pdf
  python distribution_plot.py --agent OpenHands --model deepseek-v3

  # 4. Direct file mode with custom output
  #    Output: results/lang_distribution/SWE-agent_deepseek-v3_distribution.pdf
  python distribution_plot.py --lang-path data/SWE-agent/langs/deepseek-v3/languatory.json \\
                             --output-dir ./results
        """
    )
    parser.add_argument(
        "--lang-path",
        type=str,
        help="Path to languatory.json file (single file mode)"
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Agent name (alone: all models for agent; with --model: single mode)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (requires --agent for single mode)"
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
