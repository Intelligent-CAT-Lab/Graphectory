#!/usr/bin/env python3
"""
Sample Flawed Trajectories

Identifies instances with plan violations in phase sequences:
1. Starts with "P" (patch before localization)
2. No "V" included (no validation after patches)

Scans data/{agent}/langs/{model}/phases.json and generates CSV reports
under stats/flawed_trajs/

Usage:
    python application/sample_flawed_trajs.py
    python application/sample_flawed_trajs.py --data_dir data
    python application/sample_flawed_trajs.py --agent SWE-agent --model claude-sonnet-4
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# Default configurations
DEFAULT_AGENTS = ["SWE-agent", "OpenHands"]
DEFAULT_MODELS = [
    "claude-sonnet-4",
    "deepseek-r1-0528",
    "deepseek-v3",
    "devstral-small"
]

# GitHub repository base URL
GITHUB_BASE_URL = "https://github.com/Intelligent-CAT-Lab/Graphectory/blob/application/data"


# ==================== Data Classes ====================
@dataclass
class FlawedInstance:
    """Represents an instance with a plan violation."""
    agent: str
    model: str
    instance_id: str
    resolution_status: str
    debug_difficulty: str
    violation_type: str  # "starts_with_P" or "no_V"
    phases: str  # String representation of phase sequence
    link_to_graphectory: str  # GitHub link to PDF


# ==================== URL Generation ====================
def generate_graphectory_link(agent: str, model: str, instance_id: str) -> str:
    """Generate GitHub link to the graphectory PDF.

    Example:
        https://github.com/Intelligent-CAT-Lab/Graphectory/blob/application/data/OpenHands/graphs/claude-sonnet-4/astropy__astropy-12907/astropy__astropy-12907.pdf
    """
    return f"{GITHUB_BASE_URL}/{agent}/graphs/{model}/{instance_id}/{instance_id}.pdf"


# ==================== Phase Analysis ====================
def extract_phase_abbr(phase_str: str) -> str:
    """Extract phase abbreviation from phase string (e.g., 'P_2' -> 'P')."""
    return phase_str.split('_')[0] if '_' in phase_str else phase_str


def starts_with_patch(phases: List[str]) -> bool:
    """Check if phase sequence starts with 'P' (patch)."""
    if not phases:
        return False
    first_phase = extract_phase_abbr(phases[0])
    return first_phase == 'P'


def has_no_validation(phases: List[str]) -> bool:
    """Check if phase sequence contains no 'V' (validation)."""
    return not any(extract_phase_abbr(p) == 'V' for p in phases)


# ==================== File I/O ====================
def load_phases_json(json_path: Path) -> Optional[List[Dict[str, Any]]]:
    """Load phases JSON file."""
    if not json_path.exists():
        return None

    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Warning: Could not load {json_path}: {e}", file=sys.stderr)
        return None


def discover_phases_files(data_dir: Path, agents: List[str], models: List[str]) -> List[tuple]:
    """Discover all phases.json files for given agents and models.

    Returns:
        List of (agent, model, phases_path) tuples
    """
    phases_files = []

    for agent in agents:
        for model in models:
            phases_path = data_dir / agent / "langs" / model / "phases.json"
            if phases_path.exists():
                phases_files.append((agent, model, phases_path))

    return phases_files


# ==================== Analysis ====================
def analyze_phases(
    agent: str,
    model: str,
    phases_data: List[Dict[str, Any]]
) -> tuple[List[FlawedInstance], List[FlawedInstance]]:
    """Analyze phase sequences and identify plan violations.

    Returns:
        (starts_with_P_instances, no_V_instances)
    """
    starts_with_P = []
    no_V = []

    for entry in phases_data:
        instance_id = entry.get("instance_id", "")
        resolution_status = entry.get("resolution_status", "")
        debug_difficulty = entry.get("debug_difficulty", "")
        phases = entry.get("phases", [])

        if not instance_id or not phases:
            continue

        phases_str = ", ".join(phases)
        graphectory_link = generate_graphectory_link(agent, model, instance_id)

        # Check violation type 1: starts with P
        if starts_with_patch(phases):
            starts_with_P.append(FlawedInstance(
                agent=agent,
                model=model,
                instance_id=instance_id,
                resolution_status=resolution_status,
                debug_difficulty=debug_difficulty,
                violation_type="starts_with_P",
                phases=phases_str,
                link_to_graphectory=graphectory_link
            ))

        # Check violation type 2: no V included
        if has_no_validation(phases):
            no_V.append(FlawedInstance(
                agent=agent,
                model=model,
                instance_id=instance_id,
                resolution_status=resolution_status,
                debug_difficulty=debug_difficulty,
                violation_type="no_V",
                phases=phases_str,
                link_to_graphectory=graphectory_link
            ))

    return starts_with_P, no_V


# ==================== CSV Output ====================
def save_to_csv(instances: List[FlawedInstance], output_path: Path, violation_type: str):
    """Save flawed instances to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not instances:
        print(f"  No instances found for violation type: {violation_type}")
        return

    fieldnames = [
        'agent',
        'model',
        'resolution_status',
        'debug_difficulty',
        'instance_id',
        'phases',
        'link_to_graphectory'
    ]

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for instance in sorted(instances, key=lambda x: (x.agent, x.model, x.resolution_status, x.debug_difficulty, x.instance_id)):
            writer.writerow({
                'agent': instance.agent,
                'model': instance.model,
                'resolution_status': instance.resolution_status,
                'debug_difficulty': instance.debug_difficulty,
                'instance_id': instance.instance_id,
                'phases': instance.phases,
                'link_to_graphectory': instance.link_to_graphectory
            })

    print(f"  Saved {len(instances)} instances to {output_path}")


# ==================== Main Processing ====================
def process_all(
    data_dir: str = "data",
    output_dir: str = "stats/flawed_trajs",
    agents: Optional[List[str]] = None,
    models: Optional[List[str]] = None
):
    """Process all phases files and generate CSV reports."""
    agents = agents or DEFAULT_AGENTS
    models = models or DEFAULT_MODELS

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Discover phases files
    phases_files = discover_phases_files(data_path, agents, models)

    if not phases_files:
        print("No phases.json files found.")
        return

    print(f"Found {len(phases_files)} phases.json files")
    print("=" * 60)

    # Collect all flawed instances
    all_starts_with_P = []
    all_no_V = []

    for agent, model, phases_path in phases_files:
        print(f"\nProcessing: {agent} / {model}")

        phases_data = load_phases_json(phases_path)
        if not phases_data:
            print(f"  Skipping: could not load {phases_path}")
            continue

        starts_with_P, no_V = analyze_phases(agent, model, phases_data)

        print(f"  Total instances: {len(phases_data)}")
        print(f"  Starts with P: {len(starts_with_P)}")
        print(f"  No V included: {len(no_V)}")

        all_starts_with_P.extend(starts_with_P)
        all_no_V.extend(no_V)

    # Save results to CSV
    print(f"\n{'='*60}")
    print("Saving results...")
    print(f"{'='*60}")

    save_to_csv(
        all_starts_with_P,
        output_path / "starts_with_P.csv",
        "starts_with_P"
    )

    save_to_csv(
        all_no_V,
        output_path / "no_V.csv",
        "no_V"
    )

    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total instances starting with P: {len(all_starts_with_P)}")
    print(f"Total instances with no V: {len(all_no_V)}")
    print(f"\nOutput directory: {output_path}")


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(
        description="Identify and sample flawed trajectories from phase sequences",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all agents and models (default)
  python application/sample_flawed_trajs.py

  # Custom data directory
  python application/sample_flawed_trajs.py --data_dir data

  # Specific agent
  python application/sample_flawed_trajs.py --agent SWE-agent

  # Specific model
  python application/sample_flawed_trajs.py --model claude-sonnet-4

  # Custom output directory
  python application/sample_flawed_trajs.py --output_dir stats/violations
        """
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory containing phases.json files (default: data)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="stats/flawed_trajs",
        help="Output directory for CSV files (default: stats/flawed_trajs)"
    )

    parser.add_argument(
        "--agent",
        type=str,
        choices=DEFAULT_AGENTS,
        help="Process specific agent only"
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=DEFAULT_MODELS,
        help="Process specific model only"
    )

    args = parser.parse_args()

    # Determine which agents/models to process
    agents = [args.agent] if args.agent else DEFAULT_AGENTS
    models = [args.model] if args.model else DEFAULT_MODELS

    print("Flawed Trajectory Analysis")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Agents: {', '.join(agents)}")
    print(f"Models: {', '.join(models)}")

    process_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        agents=agents,
        models=models
    )


if __name__ == "__main__":
    main()
