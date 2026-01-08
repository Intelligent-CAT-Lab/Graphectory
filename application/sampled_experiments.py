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

# Add necessary paths for imports
_SCRIPT_DIR = Path(__file__).parent
_GRAPH_ANALYSIS_PATH = _SCRIPT_DIR.parent / "graph_analysis"
_PLAN_MONITOR_PATH = _SCRIPT_DIR / "plan_monitor"

if str(_GRAPH_ANALYSIS_PATH) not in sys.path:
    sys.path.insert(0, str(_GRAPH_ANALYSIS_PATH))
if str(_PLAN_MONITOR_PATH) not in sys.path:
    sys.path.insert(0, str(_PLAN_MONITOR_PATH))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))


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

    # Oscillation detection for vanilla runs (True if any oscillation pattern detected)
    oscillation_vanilla: Dict[int, bool] = field(default_factory=dict)

    # Step count for vanilla runs (number of trajectory steps)
    step_vanilla: Dict[int, int] = field(default_factory=dict)

    # Plan compliance for vanilla runs (True if follows intended plan)
    plan_compliance_vanilla: Dict[int, bool] = field(default_factory=dict)

    # Links for each vanilla run
    links_vanilla: Dict[int, str] = field(default_factory=dict)

    # Resolution status for experimental config (e.g., starts_with_P)
    resolution_experimental: Dict[int, str] = field(default_factory=dict)

    # Phases for experimental config
    phases_experimental: Dict[int, str] = field(default_factory=dict)

    # Oscillation detection for experimental runs
    oscillation_experimental: Dict[int, bool] = field(default_factory=dict)

    # Step count for experimental runs
    step_experimental: Dict[int, int] = field(default_factory=dict)

    # Plan compliance for experimental runs
    plan_compliance_experimental: Dict[int, bool] = field(default_factory=dict)

    # Monitor triggered during experimental runs (True if monitor intervention occurred)
    monitor_triggered_experimental: Dict[int, bool] = field(default_factory=dict)

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


def compute_step_count_from_graph(graph_file: Path, fallback_phases: Optional[str] = None) -> int:
    """Compute step count from a graph JSON file.

    Args:
        graph_file: Path to graph JSON file
        fallback_phases: Optional phases string to count if graph loading fails

    Returns:
        Step count (from graph if available, otherwise from phase count, or 0)
    """
    if not graph_file.exists():
        # Fallback: count phases
        return len([p.strip() for p in fallback_phases.split(',') if p.strip()]) if fallback_phases else 0

    try:
        from analyzer import TrajectoryGraphAnalyzer

        with open(graph_file, 'r') as f:
            graph_data = json.load(f)

        analyzer = TrajectoryGraphAnalyzer(graph_data)
        return analyzer.get_step_count()

    except Exception:
        # Fallback: count phases
        return len([p.strip() for p in fallback_phases.split(',') if p.strip()]) if fallback_phases else 0


def check_monitor_triggered(traj_file: Path) -> bool:
    """Check if monitor was triggered in a trajectory file.

    Args:
        traj_file: Path to trajectory file

    Returns:
        True if any message with message_type="monitor" exists in the last query
    """
    if not traj_file.exists():
        return False

    try:
        with open(traj_file, 'r') as f:
            data = json.load(f)

        # Get the last trajectory item
        trajectory = data.get('trajectory', [])
        if not trajectory:
            return False

        last_traj = trajectory[-1]
        query = last_traj.get('query', [])

        # Check if any query item has message_type="monitor"
        for item in query:
            if isinstance(item, dict) and item.get('message_type') == 'monitor':
                return True

        return False

    except Exception:
        return False


def compute_oscillation_fields_for_instance(
    swe_agent_dir: Path,
    config_path: str,
    run_id: int,
    model: str,
    instance_id: str,
    phases_str: Optional[str]
) -> tuple[bool, int, Optional[bool], bool]:
    """Compute oscillation-specific fields for an instance.

    Args:
        swe_agent_dir: Path to SWE-agent directory
        config_path: Config path (e.g., "default/start_with_P" or "oscillation")
        run_id: Run ID
        model: Model name
        instance_id: Instance ID
        phases_str: Phases string from langs (e.g., "L_1, P_1, V_1")

    Returns:
        Tuple of (has_oscillation, step_count, plan_compliance, monitor_triggered)
    """
    # Compute step count from graph
    graphs_dir = swe_agent_dir / "graphs" / config_path / f"exp-{run_id}" / model / instance_id
    graph_file = graphs_dir / f"{instance_id}.json"
    step_count = compute_step_count_from_graph(graph_file, phases_str)

    # Detect oscillation by running monitor on trajectory
    has_oscillation = False
    trajs_dir = swe_agent_dir / "trajectories" / config_path / f"exp-{run_id}" / model
    traj_file = trajs_dir / instance_id / f"{instance_id}.traj"

    if traj_file.exists():
        try:
            from plan_monitor.monitor import StatefulPhaseMonitor
            from plan_monitor.simulator.swe_extractor import ActionExtractor

            # Run monitor on trajectory
            monitor = StatefulPhaseMonitor(enable_rules=True)
            extractor = ActionExtractor(str(traj_file))

            for event, thought, observation in extractor.extract_actions():
                result = monitor.on_step(event, thought=thought, observation=observation)

                if result and result.rule_matches:
                    for match in result.rule_matches:
                        # Check if any oscillation rule triggered
                        if match.rule_id in ["oscillation_self_loop", "oscillation_two_node",
                                            "oscillation_multi_node", "oscillation_loop_family",
                                            "oscillation_max_repeats"]:
                            has_oscillation = True
                            break

                if has_oscillation:
                    break

        except Exception as e:
            # Silently use default has_oscillation
            pass

    # Check plan compliance
    plan_compliance = check_plan_compliance_from_phases(phases_str) if phases_str else None

    # Check if monitor was triggered
    monitor_triggered = check_monitor_triggered(traj_file)

    return has_oscillation, step_count, plan_compliance, monitor_triggered


def check_plan_compliance_from_phases(phases_str: str) -> bool:
    """Check if the phases follow the intended L->P->V plan.

    Uses the plan_monitor's logic to track first appearance of each phase category
    and check if they follow the L->P->V order.

    Args:
        phases_str: Comma-separated phases (e.g., "L_1, P_1, V_1, L_2, P_2, V_2")

    Returns:
        True if plan-compliant (follows L->P->V with first appearances), False otherwise
    """
    if not phases_str:
        return False

    # Parse phases
    phases = [p.strip() for p in phases_str.split(',') if p.strip()]
    if not phases:
        return False

    # Track first appearance of each phase category (using prefix logic from plan_monitor)
    seen_prefixes = set()
    first_appearances = []

    for phase in phases:
        if '_' in phase:
            prefix = phase.split('_')[0]
            if prefix in ['L', 'P', 'V'] and prefix not in seen_prefixes:
                seen_prefixes.add(prefix)
                first_appearances.append(prefix)

    if not first_appearances:
        return False

    # Check if first appearances follow L->P->V order
    # Expected order: L must appear first, then P, then V
    if len(first_appearances) < 3:
        # Incomplete plan (missing phases)
        return False

    # Find indices of first L, P, V
    l_idx = first_appearances.index('L') if 'L' in first_appearances else -1
    p_idx = first_appearances.index('P') if 'P' in first_appearances else -1
    v_idx = first_appearances.index('V') if 'V' in first_appearances else -1

    # Check if order is L < P < V
    if l_idx == -1 or p_idx == -1 or v_idx == -1:
        return False

    return l_idx < p_idx < v_idx


# ==================== Data Loading ====================
def load_sampled_instances(csv_file: Path, config: str) -> Dict[tuple, ExperimentData]:
    """Load instances from CSV file.

    Only these instances will be included in the output.

    Args:
        csv_file: Path to the CSV file
        config: Configuration name (e.g., "oscillation", "starts_with_P")

    Returns:
        Dict mapping (agent, model, instance_id) to ExperimentData
    """
    experiments = {}

    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_file}", file=sys.stderr)
        return experiments

    is_oscillation_config = (config == "oscillation")

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

            # Load oscillation-specific fields if this is the oscillation config
            if is_oscillation_config:
                # Oscillation is True if any of the oscillation patterns are True
                self_loop = row.get('self_loop', '').lower() == 'true'
                two_node_cycle = row.get('two_node_cycle', '').lower() == 'true'
                multi_node_cycle = row.get('multi_node_cycle', '').lower() == 'true'
                loop_family = row.get('loop_family', '').lower() == 'true'

                exp.oscillation_vanilla[0] = any([self_loop, two_node_cycle, multi_node_cycle, loop_family])

                # Step count: load from graph file in data/SWE-agent/graphs/{model}/{instance_id}/
                graph_file = Path(f"data/SWE-agent/graphs/{model}/{instance_id}/{instance_id}.json")
                exp.step_vanilla[0] = compute_step_count_from_graph(graph_file, exp.phases_vanilla[0])

                # Plan compliance: compute from phases
                exp.plan_compliance_vanilla[0] = check_plan_compliance_from_phases(exp.phases_vanilla[0])

            # Generate link for run 0 (from data/ directory)
            # The link in CSV is like: https://.../data/SWE-agent/graphs/deepseek-r1-0528/...
            # We need to extract the actual model name from it or use the provided link
            original_link = row.get('link_to_graphectory', '')
            if original_link:
                exp.links_vanilla[0] = original_link
            else:
                # Fallback: generate from agent/model
                exp.links_vanilla[0] = generate_graphectory_link("default", 0, agent + "/graphs/" + model, instance_id)

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

                # Get phases for this instance
                phases_str = phases_dict.get(instance_id) if phases_dict else None

                # Compute oscillation fields if config is "oscillation"
                has_oscillation = False
                step_count = 0
                plan_compliance = None
                monitor_triggered = False

                if config == "oscillation":
                    has_oscillation, step_count, plan_compliance, monitor_triggered = compute_oscillation_fields_for_instance(
                        swe_agent_dir, reports_config_path, run_id, model, instance_id, phases_str
                    )

                # Store in appropriate fields based on config type
                if is_experimental:
                    exp.resolution_experimental[run_id] = resolutions[instance_id]
                    if phases_str:
                        exp.phases_experimental[run_id] = phases_str
                    if config == "oscillation":
                        exp.oscillation_experimental[run_id] = has_oscillation
                        exp.step_experimental[run_id] = step_count
                        exp.plan_compliance_experimental[run_id] = plan_compliance
                        exp.monitor_triggered_experimental[run_id] = monitor_triggered
                    exp.links_experimental[run_id] = generate_graphectory_link(
                        config, run_id, model, instance_id, is_experimental=True
                    )
                else:
                    exp.resolution_vanilla[run_id] = resolutions[instance_id]
                    if phases_str:
                        exp.phases_vanilla[run_id] = phases_str
                    if config == "oscillation":
                        exp.oscillation_vanilla[run_id] = has_oscillation
                        exp.step_vanilla[run_id] = step_count
                        exp.plan_compliance_vanilla[run_id] = plan_compliance
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


def save_to_csv(experiments: Dict[tuple, ExperimentData], output_path: Path, config: str, experimental_config: Optional[str] = None):
    """Save experiment data to CSV file.

    Output structure:
        agent, model,
        debug_difficulty, instance_id,
        resolution vanilla run 0, (resolution vanilla run 1, resolution vanilla run 2, ...)
        phases vanilla 0, (phases vanilla 1, phases vanilla 2, ...)
        [oscillation_vanilla_0, step_vanilla_0, plan_compliance_vanilla_0, ...] (if config == "oscillation")
        link_to_graphectory vanilla 0, (link_to_graphectory vanilla 1, link_to_graphectory vanilla 2, ...)
        resolution after run 1, (resolution after run 2, ...)
        phases after 1, (phases after 2, ...)
        [oscillation_after_1, step_after_1, plan_compliance_after_1, ...] (if config == "oscillation")
        link_to_graphectory after 1, (link_to_graphectory after 2, ...)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Determine max runs from actual data
    max_vanilla_run, max_experimental_run = determine_max_runs(experiments)

    is_oscillation_config = (config == "oscillation")

    # Build fieldnames dynamically
    fieldnames = ['agent', 'model', 'debug_difficulty', 'instance_id']

    # Resolution vanilla run columns (0, 1, 2, ...)
    for run_id in range(max_vanilla_run + 1):
        fieldnames.append(f'resolution_vanilla_{run_id}')

    # Phases vanilla columns (0, 1, 2, ...)
    for run_id in range(max_vanilla_run + 1):
        fieldnames.append(f'phases_vanilla_{run_id}')

    # Oscillation fields for vanilla runs (only if config == "oscillation")
    if is_oscillation_config:
        for run_id in range(max_vanilla_run + 1):
            fieldnames.append(f'oscillation_vanilla_{run_id}')
        for run_id in range(max_vanilla_run + 1):
            fieldnames.append(f'step_vanilla_{run_id}')
        for run_id in range(max_vanilla_run + 1):
            fieldnames.append(f'plan_compliance_vanilla_{run_id}')

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

    # Oscillation fields for experimental runs (only if config == "oscillation")
    if is_oscillation_config and experimental_config and max_experimental_run > 0:
        for run_id in range(1, max_experimental_run + 1):
            fieldnames.append(f'oscillation_after_{run_id}')
        for run_id in range(1, max_experimental_run + 1):
            fieldnames.append(f'step_after_{run_id}')
        for run_id in range(1, max_experimental_run + 1):
            fieldnames.append(f'plan_compliance_after_{run_id}')
        for run_id in range(1, max_experimental_run + 1):
            fieldnames.append(f'monitor_triggered_after_{run_id}')

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

            # Add oscillation fields for vanilla runs
            if is_oscillation_config:
                for run_id in range(max_vanilla_run + 1):
                    osc_val = exp.oscillation_vanilla.get(run_id)
                    row[f'oscillation_vanilla_{run_id}'] = str(osc_val) if osc_val is not None else ''
                for run_id in range(max_vanilla_run + 1):
                    step_val = exp.step_vanilla.get(run_id)
                    row[f'step_vanilla_{run_id}'] = str(step_val) if step_val is not None else ''
                for run_id in range(max_vanilla_run + 1):
                    plan_val = exp.plan_compliance_vanilla.get(run_id)
                    row[f'plan_compliance_vanilla_{run_id}'] = str(plan_val) if plan_val is not None else ''

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

            # Add oscillation fields for experimental runs
            if is_oscillation_config and experimental_config and max_experimental_run > 0:
                for run_id in range(1, max_experimental_run + 1):
                    osc_val = exp.oscillation_experimental.get(run_id)
                    row[f'oscillation_after_{run_id}'] = str(osc_val) if osc_val is not None else ''
                for run_id in range(1, max_experimental_run + 1):
                    step_val = exp.step_experimental.get(run_id)
                    row[f'step_after_{run_id}'] = str(step_val) if step_val is not None else ''
                for run_id in range(1, max_experimental_run + 1):
                    plan_val = exp.plan_compliance_experimental.get(run_id)
                    row[f'plan_compliance_after_{run_id}'] = str(plan_val) if plan_val is not None else ''
                for run_id in range(1, max_experimental_run + 1):
                    monitor_val = exp.monitor_triggered_experimental.get(run_id)
                    row[f'monitor_triggered_after_{run_id}'] = str(monitor_val) if monitor_val is not None else ''

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
    experiments = load_sampled_instances(sampled_csv, args.config)
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
    save_to_csv(experiments, output_path, args.config, experimental_config)

    print("\n" + "=" * 70)
    print("✓ Complete")
    print("=" * 70)


if __name__ == "__main__":
    main()
