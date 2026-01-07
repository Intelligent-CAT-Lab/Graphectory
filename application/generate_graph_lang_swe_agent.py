#!/usr/bin/env python3
"""
Process SWE-agent trajectories to generate graphs and language sequences.

This script:
1. Discovers trajectories in: trajectories/{config}/exp-{run_id}/{model}/
2. Finds matching reports: reports/{config}/{model}.{run_id}.json
3. Generates graphs directly to: graphs/{config}/exp-{run_id}/{model}/{instance_id}/
4. Generates langs directly to: langs/{config}/exp-{run_id}/{model}/phases.json

Usage:
    python application/generate_graph_lang_swe_agent.py --config oscillation --run_id 1
    python application/generate_graph_lang_swe_agent.py --config default --run_id 1 --model deepseek-chat-v3-0324
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from graph_construction.commandParser import CommandParser
from graph_construction.buildGraph import build_graph_from_sa_trajectory
from lang_construction.extractSeq import extract_node_sequence
from lang_construction.buildPhases import build_phase_sequence_rle


# ==================== Configuration ====================
# Model abbreviations for command-line
MODEL_ABBR_MAP = {
    "deepseek-r1-0528": "dsk-r1",
    "deepseek-chat-v3-0324": "dsk-v3",
    "devstral-small": "dev",
    "claude-sonnet-4": "cld-4"
}

# OpenRouter format for report matching
OPENROUTER_MAP = {
    "deepseek-r1-0528": "openrouter--deepseek--deepseek-r1-0528",
    "deepseek-chat-v3-0324": "openrouter--deepseek--deepseek-chat-v3-0324",
    "devstral-small": "openrouter--mistralai--devstral-small",
    "claude-sonnet-4": "openrouter--anthropic--claude-sonnet-4"
}

PHASE_ABBR = {
    'localization': 'L',
    'patch': 'P',
    'validation': 'V',
}


# ==================== Path Discovery ====================
def find_report_file(reports_dir: Path, config: str, model: str, run_id: int) -> Optional[Path]:
    """Find the evaluation report for a given config, model, and run_id."""
    openrouter_model = OPENROUTER_MAP.get(model, model)

    # Pattern: {config}__{openrouter_model}__*.{run_id}.json
    pattern = f"{config}/{model}.{run_id}.json"
    # print(pattern)

    for report_file in reports_dir.glob(pattern):
        return report_file

    return None


def load_trajectories(traj_dir: Path) -> List[Dict[str, Any]]:
    """Load SWE-agent trajectories from a model directory."""
    trajectories = []

    for instance_dir in sorted(traj_dir.iterdir()):
        if not instance_dir.is_dir():
            continue

        instance_id = instance_dir.name
        traj_file = instance_dir / f"{instance_id}.traj"

        if not traj_file.exists():
            print(f"  [WARN] Missing .traj file for {instance_id}")
            continue

        try:
            with open(traj_file, 'r') as f:
                traj_data = json.load(f)
            trajectories.append({
                "instance_id": instance_id,
                "traj_data": traj_data
            })
        except json.JSONDecodeError as e:
            print(f"  [ERROR] Failed to parse {traj_file}: {e}")
            continue

    return trajectories


def setup_parser() -> CommandParser:
    """Setup CommandParser with SWE-agent tool configurations."""
    parser = CommandParser()

    tool_configs = [
        "data/SWE-agent/tools/edit_anthropic/config.yaml",
        "data/SWE-agent/tools/review_on_submit_m/config.yaml",
        "data/SWE-agent/tools/registry/config.yaml",
    ]

    parser.load_tool_yaml_files(tool_configs)
    return parser


# ==================== Graph Generation ====================
def generate_graph(instance_id: str, traj_data: Dict[str, Any], parser: CommandParser,
                   output_dir: Path, eval_report_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Generate a single graph and return paths to JSON and PDF."""
    try:
        json_path, pdf_path = build_graph_from_sa_trajectory(
            traj_data=traj_data,
            parser=parser,
            instance_id=instance_id,
            output_dir=str(output_dir),
            eval_report_path=str(eval_report_path)
        )
        return json_path, pdf_path
    except Exception as e:
        print(f"  [ERROR] Failed to generate graph for {instance_id}: {e}")
        return None, None


# ==================== Language Generation ====================
def extract_phases_from_graph(graph_json_path: Path) -> Optional[List[str]]:
    """Extract phase sequence from a graph JSON file."""
    try:
        with open(graph_json_path, 'r') as f:
            graph_json = json.load(f)

        # Extract metadata
        instance_id = graph_json.get("graph", {}).get("instance_name")
        resolution_status = graph_json.get("graph", {}).get("resolution_status")
        debug_difficulty = graph_json.get("graph", {}).get("debug_difficulty")

        if not all([instance_id, resolution_status, debug_difficulty]):
            return None

        # Extract node sequence
        step_nodes = extract_node_sequence(graph_json)
        if not step_nodes:
            return None

        # Build RLE phase sequence
        phases_full, run_lengths = build_phase_sequence_rle(step_nodes)
        if not phases_full:
            return None

        # Convert to abbreviations and format
        phase_abbrs = [PHASE_ABBR.get(p.lower(), p) for p in phases_full]
        phases = [f"{phase}_{length}" for phase, length in zip(phase_abbrs, run_lengths)]

        return {
            "instance_id": instance_id,
            "resolution_status": resolution_status,
            "debug_difficulty": debug_difficulty,
            "phases": phases
        }

    except Exception as e:
        print(f"  [ERROR] Failed to extract phases from {graph_json_path}: {e}")
        return None


# ==================== Main Processing ====================
def process_model(swe_agent_dir: Path, config: str, run_id: int, model: str) -> bool:
    """Process a single model configuration."""
    print(f"\n{'='*70}")
    print(f"Processing: {config}/exp-{run_id}/{model}")
    print(f"{'='*70}")

    # Setup paths
    traj_dir = swe_agent_dir / "trajectories" / config / f"exp-{run_id}" / model
    reports_dir = swe_agent_dir / "reports"
    graphs_output_dir = swe_agent_dir / "graphs" / config / f"exp-{run_id}" / model
    langs_output_dir = swe_agent_dir / "langs" / config / f"exp-{run_id}" / model

    # Validate trajectory directory
    if not traj_dir.exists():
        print(f"[ERROR] Trajectory directory not found: {traj_dir}")
        return False

    # Find report file
    report_file = find_report_file(reports_dir, config, model, run_id) or ""
    if not report_file:
        print(f"[WARNING] Report file not found for {config}/{model}/run-{run_id}")
        print(f"  Searched in: {reports_dir}")
    

    print(f"Trajectories: {traj_dir}")
    # print(f"Report: {report_file.name}")
    print(f"Graphs output: {graphs_output_dir}")
    print(f"Langs output: {langs_output_dir}")

    # Create output directories
    graphs_output_dir.mkdir(parents=True, exist_ok=True)
    langs_output_dir.mkdir(parents=True, exist_ok=True)

    # Load trajectories
    print("\nLoading trajectories...")
    trajectories = load_trajectories(traj_dir)
    if not trajectories:
        print("[ERROR] No trajectories found")
        return False
    print(f"Loaded {len(trajectories)} trajectories")

    # Setup parser
    print("Setting up parser...")
    parser = setup_parser()

    # Generate graphs
    print(f"\n{'='*70}")
    print("Generating Graphs")
    print(f"{'='*70}")

    success_count = 0
    graph_paths = []

    for i, traj in enumerate(trajectories, 1):
        instance_id = traj["instance_id"]
        print(f"[{i}/{len(trajectories)}] Processing {instance_id}...")

        # Note: build_graph_from_sa_trajectory creates {output_dir}/{instance_id}/ structure
        # So we pass graphs_output_dir directly, not graphs_output_dir/instance_id
        json_path, pdf_path = generate_graph(
            instance_id=instance_id,
            traj_data=traj["traj_data"],
            parser=parser,
            output_dir=graphs_output_dir,
            eval_report_path=report_file
        )

        if json_path:
            print(f"  ✓ Generated: {instance_id}.json, {instance_id}.pdf")
            graph_paths.append(Path(json_path))
            success_count += 1
        else:
            print(f"  ✗ Failed to generate graph")

    print(f"\nGraphs: {success_count}/{len(trajectories)} successful")

    # Generate language sequences
    print(f"\n{'='*70}")
    print("Generating Language Sequences (Phases)")
    print(f"{'='*70}")

    phases_data = []
    for graph_path in graph_paths:
        instance_id = graph_path.parent.name
        print(f"  Extracting phases from {instance_id}...")

        phases = extract_phases_from_graph(graph_path)
        if phases:
            phases_data.append(phases)
            print(f"    ✓ Extracted {len(phases['phases'])} phases")
        else:
            print(f"    ✗ Failed to extract phases")

    # Save phases to JSON
    if phases_data:
        phases_file = langs_output_dir / "phases.json"
        with open(phases_file, 'w') as f:
            json.dump(phases_data, f, indent=2)
        print(f"\n✓ Saved phases for {len(phases_data)} instances to: {phases_file}")
    else:
        print("\n✗ No phases extracted")

    print(f"\n{'='*70}")
    print(f"Completed: {config}/exp-{run_id}/{model}")
    print(f"  Graphs: {success_count}/{len(trajectories)}")
    print(f"  Langs: {len(phases_data)}/{len(trajectories)}")
    print(f"{'='*70}")

    return success_count > 0


def main():
    parser = argparse.ArgumentParser(
        description="Process SWE-agent trajectories to generate graphs and language sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        default="default",
        help="Configuration name (default: default)"
    )

    parser.add_argument(
        "--run_id",
        type=int,
        required=True,
        help="Run ID (corresponds to exp-N)"
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to process (e.g., deepseek-chat-v3-0324). If not specified, process all models."
    )

    args = parser.parse_args()

    # Get SWE-agent directory
    script_dir = Path(__file__).parent
    swe_agent_dir = script_dir / "SWE-agent"

    if not swe_agent_dir.exists():
        print(f"[ERROR] SWE-agent directory not found: {swe_agent_dir}")
        sys.exit(1)

    # Determine which models to process
    traj_base = swe_agent_dir / "trajectories" / args.config / f"exp-{args.run_id}"

    if not traj_base.exists():
        print(f"[ERROR] Trajectory directory not found: {traj_base}")
        sys.exit(1)

    if args.model:
        models = [args.model]
    else:
        # Auto-discover all models
        models = [d.name for d in traj_base.iterdir() if d.is_dir()]

    if not models:
        print("[ERROR] No models found to process")
        sys.exit(1)

    print(f"Processing config: {args.config}, run_id: {args.run_id}")
    print(f"Models: {', '.join(models)}")

    # Process each model
    success = True
    for model in models:
        if not process_model(swe_agent_dir, args.config, args.run_id, model):
            success = False

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
