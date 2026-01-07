#!/usr/bin/env python3
"""
Sample Flawed Trajectories

Identifies instances with plan violations in phase sequences:
1. Start with "P" (patch before localization)
2. No "V" included (no validation after patches)

Identifies instances with oscillations in trajectories.

Phase violation analysis (start_with_P, no_V):
- Supports both SWE-agent and OpenHands
- Scans data/{agent}/langs/{model}/phases.json

Oscillation detection (--detect-oscillations):
- Currently supports SWE-agent only
- Scans data/SWE-agent/trajectories/{model_config}/{instance_id}/*.traj

Generates CSV reports under stats/flawed_trajs/{start_with_P, no_V, oscillation}.csv.

Usage:
    python application/sample_flawed_trajs.py
    python application/sample_flawed_trajs.py --data_dir data
    python application/sample_flawed_trajs.py --agent SWE-agent --model claude-sonnet-4
    python application/sample_flawed_trajs.py --detect-oscillations
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, asdict

# Import oscillation detection from plan_monitor
from plan_monitor.monitor import StatefulPhaseMonitor
from plan_monitor.simulator.swe_extractor import ActionExtractor

# Default configurations
DEFAULT_AGENTS = ["SWE-agent", "OpenHands"]
DEFAULT_MODELS = [
    "deepseek-r1-0528",
    "deepseek-v3",
    "devstral-small",
    "claude-sonnet-4"
]

# Model name mapping: maps display names to trajectory directory patterns
MODEL_TRAJECTORY_PATTERNS = {
    "deepseek-v3": "deepseek-chat",  # deepseek-v3 uses deepseek-chat in trajectory dirs
    "deepseek-r1-0528": "deepseek-r1-0528",
    "claude-sonnet-4": "claude-sonnet-4",
    "devstral-small": "devstral-small"
}

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
    violation_type: str  # "start_with_P" or "no_V"
    phases: str  # String representation of phase sequence
    link_to_graphectory: str  # GitHub link to PDF


@dataclass
class OscillationInstance:
    """Represents an instance with oscillation patterns."""
    agent: str
    model: str
    instance_id: str
    resolution_status: str
    debug_difficulty: str
    phases: str
    link_to_graphectory: str
    # Oscillation types detected (True/False)
    self_loop: bool
    two_node_cycle: bool
    multi_node_cycle: bool
    loop_family: bool
    max_repeats_rule: bool


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


def start_with_patch(phases: List[str]) -> bool:
    """Check if phase sequence start with 'P' (patch)."""
    if not phases:
        return False
    first_phase = extract_phase_abbr(phases[0])
    return first_phase == 'P'


def has_no_validation(phases: List[str]) -> bool:
    """Check if phase sequence contains no 'V' (validation)."""
    return not any(extract_phase_abbr(p) == 'V' for p in phases)


# ==================== Oscillation Detection ====================
def detect_oscillations(trajectory_path: Path) -> Dict[str, bool]:
    """Detect oscillation patterns in a trajectory file.

    Args:
        trajectory_path: Path to .traj file

    Returns:
        Dictionary with oscillation types as keys and boolean values
    """
    oscillation_types = {
        'self_loop': False,
        'two_node_cycle': False,
        'multi_node_cycle': False,
        'loop_family': False,
        'max_repeats_rule': False
    }

    try:
        # Initialize monitor with rules enabled
        monitor = StatefulPhaseMonitor(enable_rules=True)
        extractor = ActionExtractor(str(trajectory_path))

        # Process trajectory and collect rule triggers
        for event, thought, observation in extractor.extract_actions():
            result = monitor.on_step(event, thought=thought, observation=observation)

            if result and result.rule_matches:
                for match in result.rule_matches:
                    # Map rule_id to oscillation type
                    if match.rule_id == "oscillation_self_loop":
                        oscillation_types['self_loop'] = True
                    elif match.rule_id == "oscillation_two_node":
                        oscillation_types['two_node_cycle'] = True
                    elif match.rule_id == "oscillation_multi_node":
                        oscillation_types['multi_node_cycle'] = True
                    elif match.rule_id == "oscillation_loop_family":
                        oscillation_types['loop_family'] = True
                    elif match.rule_id == "oscillation_max_repeats":
                        oscillation_types['max_repeats_rule'] = True

        return oscillation_types

    except Exception as e:
        print(f"  Warning: Failed to detect oscillations in {trajectory_path.name}: {e}", file=sys.stderr)
        return oscillation_types


def discover_trajectory_files(data_dir: Path, agents: List[str], models: List[str]) -> List[tuple]:
    """Discover all trajectory files for oscillation detection (SWE-agent only).

    Note: OpenHands oscillation detection is not yet supported.
    Phase violation analysis (start_with_P, no_V) works for both agents via phases.json.

    Returns:
        List of (agent, model_name, trajectory_path, instance_id) tuples
    """
    trajectory_files = []

    for agent in agents:
        if agent != "SWE-agent":  # Oscillation detection only supports SWE-agent for now
            continue

        traj_base = data_dir / agent / "trajectories"
        if not traj_base.exists():
            continue

        # Find all .traj files
        for traj_file in traj_base.rglob("*.traj"):
            # Extract model and instance_id from path structure
            # Path: data/SWE-agent/trajectories/{model_config}/{instance_id}/{instance_id}.traj
            parts = traj_file.parts
            if len(parts) >= 3:
                instance_id = traj_file.stem
                model_config = parts[-3]  # e.g., "anthropic_filemap__deepseek--deepseek-chat__..."

                # Match model using trajectory patterns
                model_name = None
                for model in models:
                    pattern = MODEL_TRAJECTORY_PATTERNS.get(model, model)
                    # Check if pattern appears in model_config (case-insensitive, flexible matching)
                    if pattern.replace("-", "").lower() in model_config.replace("-", "").lower():
                        model_name = model
                        break

                if model_name:
                    trajectory_files.append((agent, model_name, traj_file, instance_id))

    return trajectory_files


def get_instance_metadata(data_dir: Path, agent: str, model: str, instance_id: str) -> tuple[str, str, str]:
    """Get instance metadata from phases.json if available.

    Returns:
        (resolution_status, debug_difficulty, phases_str)
    """
    phases_path = data_dir / agent / "langs" / model / "phases.json"

    if not phases_path.exists():
        return "", "", ""

    try:
        with open(phases_path, 'r') as f:
            phases_data = json.load(f)

        for entry in phases_data:
            if entry.get("instance_id") == instance_id:
                resolution_status = entry.get("resolution_status", "")
                debug_difficulty = entry.get("debug_difficulty", "")
                phases = entry.get("phases", [])
                phases_str = ", ".join(phases)
                return resolution_status, debug_difficulty, phases_str

    except Exception:
        pass

    return "", "", ""


def analyze_oscillations(
    data_dir: Path,
    agents: List[str],
    models: List[str]
) -> List[OscillationInstance]:
    """Analyze trajectories for oscillation patterns.

    Returns:
        List of OscillationInstance objects
    """
    oscillation_instances = []

    # Discover trajectory files
    traj_files = discover_trajectory_files(data_dir, agents, models)

    print(f"Found {len(traj_files)} trajectory files to analyze")

    for agent, model, traj_path, instance_id in traj_files:
        print(f"  Analyzing: {agent} / {model} / {instance_id}")

        # Detect oscillations
        osc_types = detect_oscillations(traj_path)

        # Check if any oscillation was detected
        if any(osc_types.values()):
            # Get metadata
            resolution_status, debug_difficulty, phases_str = get_instance_metadata(
                data_dir, agent, model, instance_id
            )

            graphectory_link = generate_graphectory_link(agent, model, instance_id)

            oscillation_instances.append(OscillationInstance(
                agent=agent,
                model=model,
                instance_id=instance_id,
                resolution_status=resolution_status,
                debug_difficulty=debug_difficulty,
                phases=phases_str,
                link_to_graphectory=graphectory_link,
                self_loop=osc_types['self_loop'],
                two_node_cycle=osc_types['two_node_cycle'],
                multi_node_cycle=osc_types['multi_node_cycle'],
                loop_family=osc_types['loop_family'],
                max_repeats_rule=osc_types['max_repeats_rule']
            ))

            print(f"    ✓ Oscillations detected: {', '.join([k for k, v in osc_types.items() if v])}")

    return oscillation_instances


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
        (start_with_P_instances, no_V_instances)
    """
    start_with_P = []
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

        # Check violation type 1: start with P
        if start_with_patch(phases):
            start_with_P.append(FlawedInstance(
                agent=agent,
                model=model,
                instance_id=instance_id,
                resolution_status=resolution_status,
                debug_difficulty=debug_difficulty,
                violation_type="start_with_P",
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

    return start_with_P, no_V


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


def save_oscillations_to_csv(instances: List[OscillationInstance], output_path: Path):
    """Save oscillation instances to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not instances:
        print(f"  No oscillation instances found")
        return

    fieldnames = [
        'agent',
        'model',
        'resolution_status',
        'debug_difficulty',
        'instance_id',
        'phases',
        'link_to_graphectory',
        'self_loop',
        'two_node_cycle',
        'multi_node_cycle',
        'loop_family',
        'max_repeats_rule'
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
                'link_to_graphectory': instance.link_to_graphectory,
                'self_loop': instance.self_loop,
                'two_node_cycle': instance.two_node_cycle,
                'multi_node_cycle': instance.multi_node_cycle,
                'loop_family': instance.loop_family,
                'max_repeats_rule': instance.max_repeats_rule
            })

    print(f"  Saved {len(instances)} oscillation instances to {output_path}")


# ==================== Main Processing ====================
def process_all(
    data_dir: str = "data",
    output_dir: str = "stats/flawed_trajs",
    agents: Optional[List[str]] = None,
    models: Optional[List[str]] = None,
    detect_oscillations_flag: bool = False
):
    """Process all phases files and generate CSV reports."""
    agents = agents or DEFAULT_AGENTS
    models = models or DEFAULT_MODELS

    data_path = Path(data_dir)
    output_path = Path(output_dir)

    # Process phase-based violations (start_with_P and no_V)
    if not detect_oscillations_flag:
        # Discover phases files
        phases_files = discover_phases_files(data_path, agents, models)

        if not phases_files:
            print("No phases.json files found.")
            return

        print(f"Found {len(phases_files)} phases.json files")
        print("=" * 60)

        # Collect all flawed instances
        all_start_with_P = []
        all_no_V = []

        for agent, model, phases_path in phases_files:
            print(f"\nProcessing: {agent} / {model}")

            phases_data = load_phases_json(phases_path)
            if not phases_data:
                print(f"  Skipping: could not load {phases_path}")
                continue

            start_with_P, no_V = analyze_phases(agent, model, phases_data)

            print(f"  Total instances: {len(phases_data)}")
            print(f"  Start with P: {len(start_with_P)}")
            print(f"  No V included: {len(no_V)}")

            all_start_with_P.extend(start_with_P)
            all_no_V.extend(no_V)

        # Save results to CSV
        print(f"\n{'='*60}")
        print("Saving results...")
        print(f"{'='*60}")

        save_to_csv(
            all_start_with_P,
            output_path / "start_with_P.csv",
            "start_with_P"
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
        print(f"Total instances starting with P: {len(all_start_with_P)}")
        print(f"Total instances with no V: {len(all_no_V)}")
        print(f"\nOutput directory: {output_path}")

    # Process oscillation detection
    else:
        print("=" * 60)
        print("Analyzing Oscillations in Trajectories")
        print("=" * 60)

        oscillation_instances = analyze_oscillations(data_path, agents, models)

        # Save results
        print(f"\n{'='*60}")
        print("Saving results...")
        print(f"{'='*60}")

        save_oscillations_to_csv(
            oscillation_instances,
            output_path / "oscillation.csv"
        )

        # Summary
        print(f"\n{'='*60}")
        print("Summary")
        print(f"{'='*60}")
        print(f"Total instances with oscillations: {len(oscillation_instances)}")
        if oscillation_instances:
            print(f"  - Self-loop: {sum(1 for i in oscillation_instances if i.self_loop)}")
            print(f"  - Two-node cycle: {sum(1 for i in oscillation_instances if i.two_node_cycle)}")
            print(f"  - Multi-node cycle: {sum(1 for i in oscillation_instances if i.multi_node_cycle)}")
            print(f"  - Loop family: {sum(1 for i in oscillation_instances if i.loop_family)}")
            print(f"  - Max repeats: {sum(1 for i in oscillation_instances if i.max_repeats_rule)}")
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

  # Detect oscillations in trajectories
  python application/sample_flawed_trajs.py --detect-oscillations
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

    parser.add_argument(
        "--detect-oscillations",
        action="store_true",
        help="Detect oscillations in trajectories (only for SWE-agent)"
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
    print(f"Detect oscillations: {args.detect_oscillations}")

    process_all(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        agents=agents,
        models=models,
        detect_oscillations_flag=args.detect_oscillations
    )


if __name__ == "__main__":
    main()
