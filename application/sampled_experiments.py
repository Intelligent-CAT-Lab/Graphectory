#!/usr/bin/env python3
"""
Sample Experiment Results

Generates a comprehensive CSV comparing experiment results across multiple runs.
Processes instances from any CSV in stats/flawed_trajs/ (e.g., starts_with_P.csv, oscillation.csv).

Reads from:
- stats/flawed_trajs/{config}.csv (defines which instances to include, provides run 0 data)
- application/SWE-agent/reports/ (run 1+ resolution data)
- application/SWE-agent/langs/ (run 1+ phases data)

Output CSV structure:
    agent, model,
    debug_difficulty, instance_id,
    resolution vanilla run 0, (resolution vanilla run 1, resolution vanilla run 2, ...)
    phases vanilla 0, (phases vanilla 1, phases vanilla 2, ...)
    link_to_graphectory vanilla 0, (link_to_graphectory vanilla 1, link_to_graphectory vanilla 2, ...)
    resolution after run 1, (resolution after run 2, ...)
    phases after 1, (phases after 2, ...)
    link_to_graphectory after 1, (link_to_graphectory after 2, ...)

Usage:
    python application/sampled_experiments.py --config starts_with_P
    python application/sampled_experiments.py --config oscillation
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass, field


# ==================== Configuration ====================
DEFAULT_CONFIG = "default"
GITHUB_BASE_URL = "https://github.com/Intelligent-CAT-Lab/Graphectory/blob/application/application/SWE-agent"

# OpenRouter format for report matching
OPENROUTER_MAP = {
    "deepseek-r1-0528": "openrouter--deepseek--deepseek-r1-0528",
    "deepseek-chat-v3-0324": "openrouter--deepseek--deepseek-chat-v3-0324",
    "deepseek-v3": "openrouter--deepseek--deepseek-chat-v3-0324",
    "devstral-small": "openrouter--mistralai--devstral-small",
    "claude-sonnet-4": "openrouter--anthropic--claude-sonnet-4"
}

# Model name aliases (maps trajectory folder names to CSV model names)
# Used to match run 1+ data with run 0 instances
MODEL_ALIASES = {
    "deepseek-chat-v3-0324": "deepseek-v3",  # Run 1 uses deepseek-chat-v3-0324, run 0 uses deepseek-v3
}


# ==================== Data Classes ====================
@dataclass
class ExperimentData:
    """Experiment data for a single instance across multiple runs and configs."""
    agent: str
    model: str
    instance_id: str
    debug_difficulty: str = ""

    # Resolution status for each vanilla run (0, 1, 2, ...)
    resolution_vanilla: Dict[int, str] = field(default_factory=dict)

    # Phases for each vanilla run
    phases_vanilla: Dict[int, str] = field(default_factory=dict)

    # Links for each vanilla run
    links_vanilla: Dict[int, str] = field(default_factory=dict)

    # Resolution status for experimental config (e.g., starts_with_P)
    resolution_experimental: Dict[int, str] = field(default_factory=dict)

    # Phases for experimental config
    phases_experimental: Dict[int, str] = field(default_factory=dict)

    # Links for experimental config
    links_experimental: Dict[int, str] = field(default_factory=dict)


# ==================== Path and URL Generation ====================
def generate_graphectory_link(config: str, run_id: int, model: str, instance_id: str, is_experimental: bool = False, vanilla_config: str = "default") -> str:
    """Generate GitHub link to the graphectory PDF.

    Example for run 0 (from data/SWE-agent/graphs/):
        https://github.com/Intelligent-CAT-Lab/Graphectory/blob/application/data/SWE-agent/graphs/deepseek-r1-0528/django__django-14672/django__django-14672.pdf

    Example for vanilla run 1+ (from application/SWE-agent/graphs/default/{config}/):
        https://github.com/Intelligent-CAT-Lab/Graphectory/blob/application/application/SWE-agent/graphs/default/start_with_P/exp-1/deepseek-chat-v3-0324/astropy__astropy-7336/astropy__astropy-7336.pdf

    Example for experimental run 1+ (from application/SWE-agent/graphs/{config}/):
        https://github.com/Intelligent-CAT-Lab/Graphectory/blob/application/application/SWE-agent/graphs/oscillation/exp-1/deepseek-chat-v3-0324/astropy__astropy-7336/astropy__astropy-7336.pdf
    """
    if run_id == 0:
        # Run 0 is from data/SWE-agent/graphs/ directory
        return f"https://github.com/Intelligent-CAT-Lab/Graphectory/blob/application/data/SWE-agent/graphs/{model}/{instance_id}/{instance_id}.pdf"
    else:
        # Run 1+ is from application/SWE-agent/graphs/
        if is_experimental:
            # Experimental: graphs/{config}/exp-N/
            return f"{GITHUB_BASE_URL}/graphs/{config}/exp-{run_id}/{model}/{instance_id}/{instance_id}.pdf"
        else:
            # Vanilla: graphs/{vanilla_config}/{config}/exp-N/
            return f"{GITHUB_BASE_URL}/graphs/{vanilla_config}/{config}/exp-{run_id}/{model}/{instance_id}/{instance_id}.pdf"


def find_report_file(reports_dir: Path, config_path: str, model: str, run_id: int) -> Optional[Path]:
    """Find the evaluation report for a given config, model, and run_id.

    Args:
        reports_dir: Base reports directory
        config_path: Full path under reports dir (e.g., "default/start_with_P" or "oscillation")
        model: Model name
        run_id: Run ID
    """
    openrouter_model = OPENROUTER_MAP.get(model, model)

    # Pattern: {config_path}/{model}.{run_id}.json
    pattern = f"{config_path}/{model}.{run_id}.json"

    matches = list(reports_dir.glob(pattern))
    return matches[0] if matches else None


def load_phases_from_langs(langs_dir: Path, config_path: str, run_id: int, model: str) -> Optional[Dict[str, str]]:
    """Load phases data from langs directory.

    Args:
        langs_dir: Base langs directory
        config_path: Full path under langs dir (e.g., "default/start_with_P" or "oscillation")
        run_id: Run ID
        model: Model name

    Returns:
        Dict mapping instance_id to phases string
    """
    phases_file = langs_dir / config_path / f"exp-{run_id}" / model / "phases.json"

    if not phases_file.exists():
        return None

    try:
        with open(phases_file, 'r') as f:
            phases_data = json.load(f)

        # Convert to dict: instance_id -> phases_str
        result = {}
        for entry in phases_data:
            instance_id = entry.get("instance_id")
            phases = entry.get("phases", [])
            if instance_id and phases:
                result[instance_id] = ", ".join(phases)

        return result
    except Exception as e:
        print(f"  Warning: Could not load {phases_file}: {e}", file=sys.stderr)
        return None


def load_resolution_from_report(report_file: Path) -> Dict[str, str]:
    """Load resolution status for each instance from a report file.

    Returns:
        Dict mapping instance_id to resolution status
    """
    try:
        with open(report_file, 'r') as f:
            report_data = json.load(f)

        result = {}

        # Get resolved instances
        for instance_id in report_data.get("resolved_ids", []):
            result[instance_id] = "resolved"

        # Get unresolved instances
        for instance_id in report_data.get("unresolved_ids", []):
            result[instance_id] = "unresolved"

        # Get error instances
        for instance_id in report_data.get("error_ids", []):
            result[instance_id] = "unsubmitted"

        # Get incomplete instances
        for instance_id in report_data.get("incomplete_ids", []):
            result[instance_id] = "unsubmitted"

        return result
    except Exception as e:
        print(f"  Warning: Could not load {report_file}: {e}", file=sys.stderr)
        return {}


# ==================== Data Loading ====================
def load_sampled_instances(csv_file: Path) -> Dict[tuple, ExperimentData]:
    """Load instances from starts_with_P.csv.
    
    Only these instances will be included in the output.

    Returns:
        Dict mapping (agent, model, instance_id) to ExperimentData
    """
    experiments = {}

    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}", file=sys.stderr)
        return experiments

    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            agent = row['agent']
            model = row['model']
            instance_id = row['instance_id']

            key = (agent, model, instance_id)

            exp = ExperimentData(
                agent=agent,
                model=model,
                instance_id=instance_id,
                debug_difficulty=row.get('debug_difficulty', ''),
            )

            # Run 0 data from the CSV
            exp.resolution_vanilla[0] = row.get('resolution_status', '')
            exp.phases_vanilla[0] = row.get('phases', '')
            
            # Generate link for run 0 (from data/ directory)
            # The link in CSV is like: https://.../data/SWE-agent/graphs/deepseek-r1-0528/...
            # We need to extract the actual model name from it or use the provided link
            original_link = row.get('link_to_graphectory', '')
            if original_link:
                exp.links_vanilla[0] = original_link
            else:
                # Fallback: generate from agent/model
                exp.links_vanilla[0] = generate_graphectory_link(DEFAULT_CONFIG, 0, agent + "/graphs/" + model, instance_id)

            experiments[key] = exp

    return experiments


def discover_available_runs(trajs_dir: Path) -> list[int]:
    """Discover available run IDs in a trajectories directory.

    Args:
        trajs_dir: Path to trajectories directory (e.g., .../trajectories/default/oscillation/)

    Returns:
        List of run IDs sorted in ascending order
    """
    if not trajs_dir.exists():
        return []

    run_ids = []
    for exp_dir in trajs_dir.glob("exp-*"):
        if exp_dir.is_dir():
            try:
                run_id = int(exp_dir.name.split('-')[1])
                run_ids.append(run_id)
            except (ValueError, IndexError):
                continue

    return sorted(run_ids)


def update_with_swe_agent_runs(
    experiments: Dict[tuple, ExperimentData],
    swe_agent_dir: Path,
    config: str,
    vanilla_config: str,
    is_experimental: bool = False
):
    """Update experiments dict with SWE-agent run data (run 1+).

    Only updates instances that already exist in experiments dict.

    Args:
        experiments: Dictionary of experiment data
        swe_agent_dir: Path to SWE-agent directory
        config: Configuration name for data storage (e.g., "oscillation", "start_with_P")
        vanilla_config: Configuration name for vanilla runs (typically "default")
        is_experimental: If True, store in experimental fields instead of vanilla fields
    """
    reports_dir = swe_agent_dir / "reports"
    langs_dir = swe_agent_dir / "langs"

    # Determine the paths for trajectories, reports, and langs
    if is_experimental:
        # Experimental runs: stored directly under {config}/
        trajs_dir = swe_agent_dir / "trajectories" / config
        reports_config_path = config
        langs_config_path = config
    else:
        # Vanilla runs: stored under {vanilla_config}/{config}/
        trajs_dir = swe_agent_dir / "trajectories" / vanilla_config / config
        reports_config_path = f"{vanilla_config}/{config}"
        langs_config_path = f"{vanilla_config}/{config}"

    if not trajs_dir.exists():
        print(f"  Warning: Trajectories directory not found: {trajs_dir}")
        return

    # Auto-discover available runs
    available_runs = discover_available_runs(trajs_dir)
    if not available_runs:
        print(f"  No runs found in {trajs_dir}")
        return

    print(f"  Found runs: {available_runs}")

    # Process each discovered run
    for run_id in available_runs:
        exp_dir = trajs_dir / f"exp-{run_id}"
        print(f"  Processing run {run_id}...")

        # Process each model in this run
        for model_dir in exp_dir.iterdir():
            if not model_dir.is_dir():
                continue

            model = model_dir.name
            agent = "SWE-agent"

            # Find report for this run
            report_file = find_report_file(reports_dir, reports_config_path, model, run_id)
            if not report_file:
                print(f"    Warning: No report found for {reports_config_path}/{model}/run-{run_id}")
                continue

            # Load resolution statuses
            resolutions = load_resolution_from_report(report_file)

            # Load phases
            phases_dict = load_phases_from_langs(langs_dir, langs_config_path, run_id, model)

            # Update experiment data for instances that are in our sampled set
            num_updated = 0
            for instance_id in resolutions.keys():
                # Check if this model has an alias (e.g., deepseek-chat-v3-0324 -> deepseek-v3)
                model_for_lookup = MODEL_ALIASES.get(model, model)
                key = (agent, model_for_lookup, instance_id)

                # Only update if this instance is in our sampled set
                if key not in experiments:
                    continue

                exp = experiments[key]

                # Store in appropriate fields based on config type
                if is_experimental:
                    exp.resolution_experimental[run_id] = resolutions[instance_id]
                    if phases_dict and instance_id in phases_dict:
                        exp.phases_experimental[run_id] = phases_dict[instance_id]
                    exp.links_experimental[run_id] = generate_graphectory_link(
                        config, run_id, model, instance_id, is_experimental=True
                    )
                else:
                    exp.resolution_vanilla[run_id] = resolutions[instance_id]
                    if phases_dict and instance_id in phases_dict:
                        exp.phases_vanilla[run_id] = phases_dict[instance_id]
                    exp.links_vanilla[run_id] = generate_graphectory_link(
                        config, run_id, model, instance_id, is_experimental=False, vanilla_config=vanilla_config
                    )

                num_updated += 1

            if num_updated > 0:
                print(f"    Updated {num_updated} instances for {model}")


# ==================== CSV Output ====================
def determine_max_runs(experiments: Dict[tuple, ExperimentData]) -> tuple[int, int]:
    """Determine the maximum run IDs for vanilla and experimental configs.

    Returns:
        Tuple of (max_vanilla_run, max_experimental_run)
    """
    max_vanilla = 0
    max_experimental = 0

    for exp in experiments.values():
        if exp.resolution_vanilla:
            max_vanilla = max(max_vanilla, max(exp.resolution_vanilla.keys()))
        if exp.resolution_experimental:
            max_experimental = max(max_experimental, max(exp.resolution_experimental.keys()))

    return max_vanilla, max_experimental


def save_to_csv(experiments: Dict[tuple, ExperimentData], output_path: Path, experimental_config: Optional[str] = None):
    """Save experiment data to CSV file.

    Output structure:
        agent, model,
        debug_difficulty, instance_id,
        resolution vanilla run 0, (resolution vanilla run 1, resolution vanilla run 2, ...)
        phases vanilla 0, (phases vanilla 1, phases vanilla 2, ...)
        link_to_graphectory vanilla 0, (link_to_graphectory vanilla 1, link_to_graphectory vanilla 2, ...)
        resolution after run 1, (resolution after run 2, ...)
        phases after 1, (phases after 2, ...)
        link_to_graphectory after 1, (link_to_graphectory after 2, ...)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine max runs from actual data
    max_vanilla_run, max_experimental_run = determine_max_runs(experiments)

    # Build fieldnames dynamically
    fieldnames = ['agent', 'model', 'debug_difficulty', 'instance_id']

    # Resolution vanilla run columns (0, 1, 2, ...)
    for run_id in range(max_vanilla_run + 1):
        fieldnames.append(f'resolution_vanilla_{run_id}')

    # Phases vanilla columns (0, 1, 2, ...)
    for run_id in range(max_vanilla_run + 1):
        fieldnames.append(f'phases_vanilla_{run_id}')

    # Link to graphectory vanilla columns (0, 1, 2, ...)
    for run_id in range(max_vanilla_run + 1):
        fieldnames.append(f'link_graphectory_vanilla_{run_id}')

    # Resolution experimental columns if experimental config is provided (starting from run 1)
    if experimental_config and max_experimental_run > 0:
        for run_id in range(1, max_experimental_run + 1):
            fieldnames.append(f'resolution_after_{run_id}')

    # Phases experimental columns (starting from run 1)
    if experimental_config and max_experimental_run > 0:
        for run_id in range(1, max_experimental_run + 1):
            fieldnames.append(f'phases_after_{run_id}')

    # Link to graphectory experimental columns (starting from run 1)
    if experimental_config and max_experimental_run > 0:
        for run_id in range(1, max_experimental_run + 1):
            fieldnames.append(f'link_graphectory_after_{run_id}')

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        # Sort by agent, model, instance_id
        for key in sorted(experiments.keys()):
            exp = experiments[key]

            row = {
                'agent': exp.agent,
                'model': exp.model,
                'debug_difficulty': exp.debug_difficulty,
                'instance_id': exp.instance_id,
            }

            # Add resolution vanilla columns
            for run_id in range(max_vanilla_run + 1):
                row[f'resolution_vanilla_{run_id}'] = exp.resolution_vanilla.get(run_id, '')

            # Add phases vanilla columns
            for run_id in range(max_vanilla_run + 1):
                row[f'phases_vanilla_{run_id}'] = exp.phases_vanilla.get(run_id, '')

            # Add link vanilla columns
            for run_id in range(max_vanilla_run + 1):
                row[f'link_graphectory_vanilla_{run_id}'] = exp.links_vanilla.get(run_id, '')

            # Add resolution experimental columns (starting from run 1)
            if experimental_config and max_experimental_run > 0:
                for run_id in range(1, max_experimental_run + 1):
                    row[f'resolution_after_{run_id}'] = exp.resolution_experimental.get(run_id, '')

            # Add phases experimental columns (starting from run 1)
            if experimental_config and max_experimental_run > 0:
                for run_id in range(1, max_experimental_run + 1):
                    row[f'phases_after_{run_id}'] = exp.phases_experimental.get(run_id, '')

            # Add link experimental columns (starting from run 1)
            if experimental_config and max_experimental_run > 0:
                for run_id in range(1, max_experimental_run + 1):
                    row[f'link_graphectory_after_{run_id}'] = exp.links_experimental.get(run_id, '')

            writer.writerow(row)

    print(f"\n✓ Saved {len(experiments)} instances to {output_path}")
    print(f"  Vanilla runs: 0-{max_vanilla_run}")
    if experimental_config and max_experimental_run > 0:
        print(f"  Experimental runs: 1-{max_experimental_run}")


# ==================== Main Processing ====================
def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive experiment results CSV for sampled instances",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration name (e.g., 'starts_with_P', 'oscillation'). This determines which CSV to load from stats/flawed_trajs/"
    )

    parser.add_argument(
        "--vanilla_config",
        type=str,
        default="default",
        help="Vanilla configuration name for run 1+ data (default: 'default'). Vanilla runs are stored in trajectories/{vanilla_config}/{config}/"
    )

    parser.add_argument(
        "--swe_agent_dir",
        type=str,
        default="application/SWE-agent",
        help="Path to SWE-agent directory (default: application/SWE-agent)"
    )

    parser.add_argument(
        "--experimental_config",
        type=str,
        default=None,
        help="Experimental configuration name for 'after' runs (optional). If set, loads data from trajectories/{config}/. If not set or empty, no experimental runs are processed."
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file (default: application/stats/{config}.csv)"
    )

    args = parser.parse_args()

    # Normalize experimental config (empty string or None means disabled)
    experimental_config = args.experimental_config.strip() if args.experimental_config else None

    # Default output path if not provided
    if args.output is None:
        args.output = f"application/stats/{args.config}.csv"

    # Default sampled CSV path
    sampled_csv_path = f"stats/flawed_trajs/{args.config}.csv"

    print("=" * 70)
    print("Sampled Experiment Results")
    print("=" * 70)
    print(f"Config: {args.config}")
    print(f"Vanilla config: {args.vanilla_config}")
    print(f"Experimental config: {experimental_config or 'None'}")
    print(f"Sampled CSV: {sampled_csv_path}")
    print(f"SWE-agent dir: {args.swe_agent_dir}")
    print(f"Output: {args.output}")
    print("=" * 70)

    # Load sampled instances from CSV
    print("\nLoading sampled instances...")
    sampled_csv = Path(sampled_csv_path)
    experiments = load_sampled_instances(sampled_csv)
    print(f"Loaded {len(experiments)} sampled instances")

    # Update with SWE-agent vanilla run data (run 1+)
    # Vanilla runs are stored in trajectories/{vanilla_config}/{config}/exp-N/
    print(f"\nLoading vanilla run data from '{args.vanilla_config}/{args.config}'...")
    swe_agent_dir = Path(args.swe_agent_dir)
    update_with_swe_agent_runs(
        experiments,
        swe_agent_dir,
        args.config,
        args.vanilla_config,
        is_experimental=False
    )

    # Update with SWE-agent experimental run data if config is provided
    # Experimental runs are stored directly in trajectories/{config}/exp-N/
    # Also in reports/{config}/ and langs/{config}/
    if experimental_config:
        print(f"\nLoading experimental 'after' run data from '{experimental_config}'...")
        update_with_swe_agent_runs(
            experiments,
            swe_agent_dir,
            experimental_config,
            args.vanilla_config,  # Not used for experimental, but pass it anyway
            is_experimental=True
        )
    else:
        # If no experimental config provided, check if {config} exists directly (for after runs)
        experimental_trajs_dir = swe_agent_dir / "trajectories" / args.config
        if experimental_trajs_dir.exists():
            print(f"\nAuto-detected experimental 'after' run data in '{args.config}'...")
            experimental_config = args.config  # Set it for CSV output
            update_with_swe_agent_runs(
                experiments,
                swe_agent_dir,
                args.config,
                args.vanilla_config,
                is_experimental=True
            )

    # Save to CSV
    print("\nSaving results...")
    output_path = Path(args.output)
    save_to_csv(experiments, output_path, experimental_config)

    print("\n" + "=" * 70)
    print("✓ Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
