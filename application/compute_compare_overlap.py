"""
Compare the performance of agents with and without the plan monitor integrated.
Take the stats/{config}.csv as input, compare the performance before (vanilla) and after the monitor is added.
Compute the resolution rate, average number of steps, oscillation rate, plan compliance rate, and issue rate (true if either oscillation is True or plan compliance is False) per <agent,model> pair before and after.

Compute the reproducible overlap for vanilla run:
for the available vainilla_{run_id}s, get the instances that at least two runs have issues (oscillation or plan violation), mark as resolved only if all runs resolved it, get the avg. steps for this instance across the corresponding vanillar_{run_id}.
For these reproducible overlap instances, get the performance after the monitor is added (after_{run_id}), and compute the same metrics.
print out the summary table.
including the vanilla_reproducible and after metrics in the overall summary.

Sample usage:
    python application/compute_compare_overlap.py --config oscillation
"""

import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple


def load_experiment_data(csv_path: Path) -> List[Dict]:
    """Load experiment data from CSV file.

    Args:
        csv_path: Path to the CSV file

    Returns:
        List of experiment rows as dictionaries
    """
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append(row)
    return data


def compute_metrics(data: List[Dict], config: str) -> Dict[Tuple[str, str], Dict]:
    """Compute performance metrics per (agent, model) pair, including reproducible overlap.

    Args:
        data: List of experiment rows
        config: Config name (e.g., "oscillation")

    Returns:
        Dict mapping (agent, model) to metrics dict with 'vanilla', 'after', and 'vanilla_reproducible' keys
    """
    is_oscillation = (config == "oscillation")

    # Discover available vanilla and after run_ids from the first row
    vanilla_run_ids = set()
    after_run_ids = set()
    if data:
        for key in data[0].keys():
            if key.startswith('resolution_vanilla_'):
                parts = key.split('_')
                if len(parts) >= 3:
                    vanilla_run_ids.add(parts[2])  # run_id
            elif key.startswith('resolution_after_'):
                parts = key.split('_')
                if len(parts) >= 3:
                    after_run_ids.add(parts[2])  # run_id

    # Group by (agent, model)
    # Create metrics structure dynamically for all vanilla runs
    def create_metrics_dict():
        m = {
            'after': {'resolved': 0, 'total': 0, 'steps': [], 'oscillations': 0, 'plan_violations': 0, 'issues': 0},
            'vanilla_reproducible': {'resolved': 0, 'total': 0, 'steps': [], 'oscillations': 0, 'plan_violations': 0, 'issues': 0},
            'after_reproducible': {'resolved': 0, 'total': 0, 'steps': [], 'oscillations': 0, 'plan_violations': 0, 'issues': 0}
        }
        # Add metrics for each vanilla run
        for run_id in sorted(vanilla_run_ids):
            m[f'vanilla_{run_id}'] = {'resolved': 0, 'total': 0, 'steps': [], 'oscillations': 0, 'plan_violations': 0, 'issues': 0}
        return m

    metrics = defaultdict(create_metrics_dict)

    for row in data:
        agent = row['agent']
        model = row['model']
        key = (agent, model)

        # Process all vanilla runs
        for run_id in vanilla_run_ids:
            res_col = f"resolution_vanilla_{run_id}"
            if row.get(res_col):
                vanilla_key = f'vanilla_{run_id}'
                metrics[key][vanilla_key]['total'] += 1
                if row[res_col] == 'resolved':
                    metrics[key][vanilla_key]['resolved'] += 1

                if is_oscillation:
                    step_col = f"step_vanilla_{run_id}"
                    if row.get(step_col) and row[step_col].isdigit():
                        metrics[key][vanilla_key]['steps'].append(int(row[step_col]))

                    has_oscillation = row.get(f"oscillation_vanilla_{run_id}") == 'True'
                    has_plan_violation = row.get(f"plan_compliance_vanilla_{run_id}") == 'False'

                    if has_oscillation:
                        metrics[key][vanilla_key]['oscillations'] += 1

                    if has_plan_violation:
                        metrics[key][vanilla_key]['plan_violations'] += 1

                    if has_oscillation or has_plan_violation:
                        metrics[key][vanilla_key]['issues'] += 1

        # After run 1 metrics (baseline)
        if row.get('resolution_after_1'):
            metrics[key]['after']['total'] += 1
            if row['resolution_after_1'] == 'resolved':
                metrics[key]['after']['resolved'] += 1

        if is_oscillation:
            if row.get('step_after_1') and row['step_after_1'].isdigit():
                metrics[key]['after']['steps'].append(int(row['step_after_1']))

            has_oscillation = row.get('oscillation_after_1') == 'True'
            has_plan_violation = row.get('plan_compliance_after_1') == 'False'

            if has_oscillation:
                metrics[key]['after']['oscillations'] += 1

            if has_plan_violation:
                metrics[key]['after']['plan_violations'] += 1

            if has_oscillation or has_plan_violation:
                metrics[key]['after']['issues'] += 1

        # Compute reproducible overlap instances
        # Check if at least 2 vanilla runs have issues
        vanilla_runs_data = {}  # Store data per run_id

        for run_id in sorted(vanilla_run_ids):
            osc_col = f"oscillation_vanilla_{run_id}"
            plan_col = f"plan_compliance_vanilla_{run_id}"
            res_col = f"resolution_vanilla_{run_id}"
            step_col = f"step_vanilla_{run_id}"

            # Check if data exists for this run (resolution must exist)
            if row.get(res_col):
                has_osc = row.get(osc_col) == 'True'
                has_plan_viol = row.get(plan_col) == 'False'
                has_issue = has_osc or has_plan_viol

                vanilla_runs_data[run_id] = {
                    'has_issue': has_issue,
                    'resolved': row.get(res_col) == 'resolved',
                    'step': int(row[step_col]) if row.get(step_col) and row[step_col].isdigit() else None,
                    'has_osc': has_osc,
                    'has_plan_viol': has_plan_viol
                }

        # Get runs with issues
        runs_with_issues = [run_id for run_id, data in vanilla_runs_data.items() if data['has_issue']]

        # If at least 2 vanilla runs have issues, this is a reproducible instance
        if len(runs_with_issues) >= 2:
            metrics[key]['vanilla_reproducible']['total'] += 1

            # Extract data only from runs with issues
            repro_resolved = [vanilla_runs_data[run_id]['resolved'] for run_id in runs_with_issues]
            repro_steps = [vanilla_runs_data[run_id]['step'] for run_id in runs_with_issues if vanilla_runs_data[run_id]['step'] is not None]
            repro_has_osc = [vanilla_runs_data[run_id]['has_osc'] for run_id in runs_with_issues]
            repro_has_plan_viol = [vanilla_runs_data[run_id]['has_plan_viol'] for run_id in runs_with_issues]

            # Mark as resolved only if ALL reproducible runs resolved it
            if repro_resolved and all(repro_resolved):
                metrics[key]['vanilla_reproducible']['resolved'] += 1

            # Average steps across reproducible runs only
            if repro_steps:
                avg_step = sum(repro_steps) / len(repro_steps)
                metrics[key]['vanilla_reproducible']['steps'].append(avg_step)

            # Count oscillations and plan violations across reproducible runs (OR logic - any run has it)
            if is_oscillation:
                if any(repro_has_osc):
                    metrics[key]['vanilla_reproducible']['oscillations'] += 1

                if any(repro_has_plan_viol):
                    metrics[key]['vanilla_reproducible']['plan_violations'] += 1

                # Issue if any reproducible run has oscillation or any has plan violation
                if any(repro_has_osc) or any(repro_has_plan_viol):
                    metrics[key]['vanilla_reproducible']['issues'] += 1

            # Get after performance for this reproducible instance
            # Use the first available after run
            for run_id in sorted(after_run_ids):
                res_col = f"resolution_after_{run_id}"
                step_col = f"step_after_{run_id}"
                osc_col = f"oscillation_after_{run_id}"
                plan_col = f"plan_compliance_after_{run_id}"

                if row.get(res_col):
                    metrics[key]['after_reproducible']['total'] += 1
                    if row[res_col] == 'resolved':
                        metrics[key]['after_reproducible']['resolved'] += 1

                    if is_oscillation:
                        if row.get(step_col) and row[step_col].isdigit():
                            metrics[key]['after_reproducible']['steps'].append(int(row[step_col]))

                        has_osc = row.get(osc_col) == 'True'
                        has_plan_viol = row.get(plan_col) == 'False'

                        if has_osc:
                            metrics[key]['after_reproducible']['oscillations'] += 1

                        if has_plan_viol:
                            metrics[key]['after_reproducible']['plan_violations'] += 1

                        if has_osc or has_plan_viol:
                            metrics[key]['after_reproducible']['issues'] += 1

                    break  # Only use the first available after run

    return metrics


def print_phase_metrics(metrics: Dict[Tuple[str, str], Dict], config: str, phase_keys: list, title: str):
    """Print a table for specific phase metrics.

    Args:
        metrics: Computed metrics per (agent, model) pair
        config: Config name
        phase_keys: List of phase keys to display (e.g., ['vanilla', 'after'] or ['vanilla_reproducible', 'after_reproducible'])
        title: Title for this table section
    """
    is_oscillation = (config == "oscillation")

    print("\n" + "="*120)
    print(f"{title}")
    print("="*120)

    # Table header
    header = ["Agent", "Model", "Phase", "Resolution Rate", "Total Instances"]
    if is_oscillation:
        header.extend(["Avg Steps", "Oscillation Rate", "Plan Violation Rate", "Issue Rate"])

    # Print header
    print()
    col_widths = [15, 25, 10, 18, 18]
    if is_oscillation:
        col_widths.extend([12, 18, 22, 15])

    header_str = ""
    for i, col in enumerate(header):
        header_str += col.ljust(col_widths[i])
    print(header_str)
    print("-" * sum(col_widths))

    # Generate phase labels dynamically
    phase_labels = {
        'after': 'After',
        'vanilla_reproducible': 'V-Repro',
        'after_reproducible': 'A-Repro'
    }
    # Add labels for all vanilla runs
    for phase_key in phase_keys:
        if phase_key.startswith('vanilla_') and phase_key not in phase_labels:
            run_id = phase_key.replace('vanilla_', '')
            phase_labels[phase_key] = f'V-{run_id}'

    def compute_phase_rates(phase_data):
        """Helper to compute rates for a phase."""
        total = phase_data['total']
        if total == 0:
            return None
        return {
            'total': total,
            'res_rate': (phase_data['resolved'] / total * 100),
            'avg_steps': sum(phase_data['steps']) / len(phase_data['steps']) if phase_data['steps'] else 0,
            'osc_rate': (phase_data['oscillations'] / total * 100),
            'viol_rate': (phase_data['plan_violations'] / total * 100),
            'issue_rate': (phase_data['issues'] / total * 100)
        }

    # Sort by agent, model for consistent output
    for (agent, model) in sorted(metrics.keys()):
        m = metrics[(agent, model)]

        # Compute metrics for all requested phases
        phase_metrics = {}
        for phase_key in phase_keys:
            phase_metrics[phase_key] = compute_phase_rates(m[phase_key])

        # Skip if first phase has no data
        if phase_metrics[phase_keys[0]] is None:
            continue

        # Print each phase row
        for idx, phase_key in enumerate(phase_keys):
            pm = phase_metrics[phase_key]
            if pm is None:
                continue

            row_data = [
                agent if idx == 0 else "",
                model if idx == 0 else "",
                phase_labels[phase_key],
                f"{pm['res_rate']:.1f}%",
                str(pm['total'])
            ]
            if is_oscillation:
                row_data.extend([
                    f"{pm['avg_steps']:.1f}",
                    f"{pm['osc_rate']:.1f}%",
                    f"{pm['viol_rate']:.1f}%",
                    f"{pm['issue_rate']:.1f}%"
                ])

            print("".join(val.ljust(col_widths[i]) for i, val in enumerate(row_data)))

        # Don't print delta rows for multiple vanilla runs table
        # Only print delta for 2-phase comparisons (e.g., vanilla_reproducible vs after_reproducible)
        if len(phase_keys) == 2 and phase_metrics[phase_keys[1]] is not None:
            pm1 = phase_metrics[phase_keys[0]]
            pm2 = phase_metrics[phase_keys[1]]

            row_data = [
                "",
                "",
                "Δ",
                f"{pm2['res_rate'] - pm1['res_rate']:+.1f}%",
                ""
            ]
            if is_oscillation:
                row_data.extend([
                    f"{pm2['avg_steps'] - pm1['avg_steps']:+.1f}",
                    f"{pm2['osc_rate'] - pm1['osc_rate']:+.1f}%",
                    f"{pm2['viol_rate'] - pm1['viol_rate']:+.1f}%",
                    f"{pm2['issue_rate'] - pm1['issue_rate']:+.1f}%"
                ])

            print("".join(val.ljust(col_widths[i]) for i, val in enumerate(row_data)))

        print()

    print("="*120)


def print_overall_summary(metrics: Dict[Tuple[str, str], Dict], config: str, phase_keys: list, title: str):
    """Print overall summary for specific phases.

    Args:
        metrics: Computed metrics per (agent, model) pair
        config: Config name
        phase_keys: List of phase keys to summarize (e.g., ['vanilla', 'after'])
        title: Title for this summary section
    """
    is_oscillation = (config == "oscillation")

    print(f"\n{title}")
    print("-"*120)

    # Generate phase labels dynamically
    phase_labels = {
        'after': 'After Monitor',
        'vanilla_reproducible': 'Vanilla Reproducible (instances with issues in ≥2 runs)',
        'after_reproducible': 'After Monitor (on reproducible instances)'
    }
    # Add labels for all vanilla runs
    for phase_key in phase_keys:
        if phase_key.startswith('vanilla_') and phase_key not in phase_labels:
            run_id = phase_key.replace('vanilla_', '')
            phase_labels[phase_key] = f'Vanilla Run {run_id}'

    # Compute aggregated metrics for each phase
    phase_aggregates = {}
    for phase_key in phase_keys:
        total_resolved = sum(m[phase_key]['resolved'] for m in metrics.values())
        total_instances = sum(m[phase_key]['total'] for m in metrics.values())

        if total_instances == 0:
            phase_aggregates[phase_key] = None
            continue

        agg = {
            'total': total_instances,
            'resolved': total_resolved,
            'res_rate': (total_resolved / total_instances * 100)
        }

        if is_oscillation:
            all_steps = [step for m in metrics.values() for step in m[phase_key]['steps']]
            total_osc = sum(m[phase_key]['oscillations'] for m in metrics.values())
            total_viol = sum(m[phase_key]['plan_violations'] for m in metrics.values())
            total_issues = sum(m[phase_key]['issues'] for m in metrics.values())

            agg.update({
                'avg_steps': sum(all_steps) / len(all_steps) if all_steps else 0,
                'osc_rate': (total_osc / total_instances * 100),
                'viol_rate': (total_viol / total_instances * 100),
                'issue_rate': (total_issues / total_instances * 100)
            })

        phase_aggregates[phase_key] = agg

    # Print each phase summary
    for phase_key in phase_keys:
        agg = phase_aggregates[phase_key]
        if agg is None:
            continue

        print(f"\n{phase_labels[phase_key]}:")
        print(f"  Total instances: {agg['total']}")
        print(f"  Resolved: {agg['resolved']}")
        print(f"  Resolution rate: {agg['res_rate']:.1f}%")

        if is_oscillation:
            print(f"  Average steps: {agg['avg_steps']:.1f}")
            print(f"  Oscillation rate: {agg['osc_rate']:.1f}%")
            print(f"  Plan violation rate: {agg['viol_rate']:.1f}%")
            print(f"  Issue rate: {agg['issue_rate']:.1f}%")

    # Print improvement only for 2-phase comparisons
    if len(phase_keys) == 2:
        agg1 = phase_aggregates[phase_keys[0]]
        agg2 = phase_aggregates[phase_keys[1]]

        if agg1 is not None and agg2 is not None:
            print(f"\nOverall Improvement:")
            print(f"  Resolution rate: {agg2['res_rate'] - agg1['res_rate']:+.1f}%")

            if is_oscillation:
                print(f"  Average steps: {agg2['avg_steps'] - agg1['avg_steps']:+.1f}")
                print(f"  Oscillation rate: {agg2['osc_rate'] - agg1['osc_rate']:+.1f}%")
                print(f"  Plan violation rate: {agg2['viol_rate'] - agg1['viol_rate']:+.1f}%")
                print(f"  Issue rate: {agg2['issue_rate'] - agg1['issue_rate']:+.1f}%")


def print_summary_table(metrics: Dict[Tuple[str, str], Dict], config: str):
    """Print summary tables for all instances and reproducible instances."""

    # Get all vanilla run keys from the first metric entry
    if metrics:
        first_key = next(iter(metrics.keys()))
        vanilla_keys = sorted([k for k in metrics[first_key].keys() if k.startswith('vanilla_') and k != 'vanilla_reproducible'])
    else:
        vanilla_keys = []

    # Table 1: All instances (all vanilla runs + after)
    all_phases = vanilla_keys + ['after']
    print_phase_metrics(metrics, config, all_phases,
                       f"Performance Comparison: All Instances (Config: {config})")

    print_overall_summary(metrics, config, all_phases,
                         "Overall Summary - All Instances")

    # Table 2: Reproducible instances (vanilla_reproducible vs after_reproducible)
    print_phase_metrics(metrics, config, ['vanilla_reproducible', 'after_reproducible'],
                       f"Performance Comparison: Reproducible Instances (Config: {config})")

    print_overall_summary(metrics, config, ['vanilla_reproducible', 'after_reproducible'],
                         "Overall Summary - Reproducible Instances")

    print("="*120 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare agent performance with and without plan monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Configuration name (e.g., 'oscillation'). Loads from application/stats/{config}.csv"
    )

    args = parser.parse_args()

    # Load CSV data
    csv_path = Path(f"application/stats/{args.config}.csv")
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        print(f"Please run sampled_experiments.py first to generate the data.")
        return 1

    print(f"\nLoading data from {csv_path}...")
    data = load_experiment_data(csv_path)
    print(f"Loaded {len(data)} instances")

    # Compute metrics
    metrics = compute_metrics(data, args.config)

    # Print summary table
    print_summary_table(metrics, args.config)

    return 0


if __name__ == "__main__":
    exit(main())
