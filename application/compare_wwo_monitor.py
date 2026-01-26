"""
Compare the performance of agents with and without the plan monitor integrated.
Take the stats/{config}.csv as input, compare the performance before (vanilla) and after the monitor is added.
Compute the resolution rate, average number of steps, oscillation rate, and plan compliance rate per <agent,model> pair before and after.
print out a summary table.

Sample usage:
    python application/compare_wwo_monitor.py --config oscillation
    python application/compare_wwo_monitor.py --config oscillation --with_rerun
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


def compute_metrics(data: List[Dict], config: str, with_rerun: bool = False) -> Dict[Tuple[str, str], Dict]:
    """Compute performance metrics per (agent, model) pair.

    Args:
        data: List of experiment rows
        config: Config name (e.g., "oscillation")
        with_rerun: If True, also include metrics for all available run_ids

    Returns:
        Dict mapping (agent, model) to metrics dict with 'vanilla' and 'after' keys (and run-specific keys if with_rerun)
    """
    is_oscillation = (config == "oscillation")

    # Discover available run_ids from the first row
    run_ids = set()
    if with_rerun and data:
        for key in data[0].keys():
            if key.startswith('resolution_'):
                parts = key.split('_')
                if len(parts) >= 3:
                    run_ids.add((parts[1], parts[2]))  # (phase, run_id)

    # Group by (agent, model)
    if with_rerun:
        metrics = defaultdict(lambda: {
            'vanilla': {'resolved': 0, 'total': 0, 'steps': [], 'oscillations': 0, 'plan_compliant': 0, 'issues': 0},
            'after': {'resolved': 0, 'total': 0, 'steps': [], 'oscillations': 0, 'plan_compliant': 0, 'issues': 0},
            'runs': defaultdict(lambda: {'resolved': 0, 'total': 0, 'steps': [], 'oscillations': 0, 'plan_compliant': 0, 'issues': 0})
        })
    else:
        metrics = defaultdict(lambda: {
            'vanilla': {'resolved': 0, 'total': 0, 'steps': [], 'oscillations': 0, 'plan_compliant': 0, 'issues': 0},
            'after': {'resolved': 0, 'total': 0, 'steps': [], 'oscillations': 0, 'plan_compliant': 0, 'issues': 0}
        })

    for row in data:
        agent = row['agent']
        model = row['model']
        key = (agent, model)

        # Vanilla run 0 metrics
        if row.get('resolution_vanilla_0'):
            metrics[key]['vanilla']['total'] += 1
            if row['resolution_vanilla_0'] == 'resolved':
                metrics[key]['vanilla']['resolved'] += 1

        if is_oscillation:
            if row.get('step_vanilla_0') and row['step_vanilla_0'].isdigit():
                metrics[key]['vanilla']['steps'].append(int(row['step_vanilla_0']))

            has_oscillation = row.get('oscillation_vanilla_0') == 'True'
            has_plan_violation = row.get('plan_compliance_vanilla_0') == 'False'

            if has_oscillation:
                metrics[key]['vanilla']['oscillations'] += 1

            if row.get('plan_compliance_vanilla_0') == 'True':
                metrics[key]['vanilla']['plan_compliant'] += 1

            # Count instances with oscillation OR plan violation
            if has_oscillation or has_plan_violation:
                metrics[key]['vanilla']['issues'] += 1

        # After run 1 metrics (experimental)
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

            if row.get('plan_compliance_after_1') == 'True':
                metrics[key]['after']['plan_compliant'] += 1

            # Count instances with oscillation OR plan violation
            if has_oscillation or has_plan_violation:
                metrics[key]['after']['issues'] += 1

        # Process all other runs if with_rerun is enabled
        if with_rerun:
            for phase, run_id in run_ids:
                run_key = f"{phase}_{run_id}"
                res_col = f"resolution_{phase}_{run_id}"

                if row.get(res_col):
                    metrics[key]['runs'][run_key]['total'] += 1
                    if row[res_col] == 'resolved':
                        metrics[key]['runs'][run_key]['resolved'] += 1

                if is_oscillation:
                    step_col = f"step_{phase}_{run_id}"
                    if row.get(step_col) and row[step_col].isdigit():
                        metrics[key]['runs'][run_key]['steps'].append(int(row[step_col]))

                    osc_col = f"oscillation_{phase}_{run_id}"
                    plan_col = f"plan_compliance_{phase}_{run_id}"

                    has_oscillation = row.get(osc_col) == 'True'
                    has_plan_violation = row.get(plan_col) == 'False'

                    if has_oscillation:
                        metrics[key]['runs'][run_key]['oscillations'] += 1

                    if row.get(plan_col) == 'True':
                        metrics[key]['runs'][run_key]['plan_compliant'] += 1

                    # Count instances with oscillation OR plan violation
                    if has_oscillation or has_plan_violation:
                        metrics[key]['runs'][run_key]['issues'] += 1

    return metrics


def print_summary_table(metrics: Dict[Tuple[str, str], Dict], config: str, with_rerun: bool = False):
    """Print a formatted summary table comparing vanilla vs after monitor.

    Args:
        metrics: Computed metrics per (agent, model) pair
        config: Config name
        with_rerun: If True, also print metrics for all available run_ids
    """
    is_oscillation = (config == "oscillation")

    print("\n" + "="*120)
    print(f"Performance Comparison: With vs Without Monitor (Config: {config})")
    print("="*120)

    # Table header
    header = ["Agent", "Model", "Phase", "Resolution Rate", "Total Instances"]
    if is_oscillation:
        header.extend(["Avg Steps", "Oscillation Rate", "Plan Compliance Rate", "Issue Rate"])

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

    # Sort by agent, model for consistent output
    for (agent, model) in sorted(metrics.keys()):
        m = metrics[(agent, model)]

        # Vanilla metrics
        vanilla = m['vanilla']
        vanilla_total = vanilla['total']
        vanilla_res_rate = (vanilla['resolved'] / vanilla_total * 100) if vanilla_total > 0 else 0
        vanilla_avg_steps = sum(vanilla['steps']) / len(vanilla['steps']) if vanilla['steps'] else 0
        vanilla_osc_rate = (vanilla['oscillations'] / vanilla_total * 100) if vanilla_total > 0 else 0
        vanilla_plan_rate = (vanilla['plan_compliant'] / vanilla_total * 100) if vanilla_total > 0 else 0
        vanilla_issue_rate = (vanilla['issues'] / vanilla_total * 100) if vanilla_total > 0 else 0

        # After metrics
        after = m['after']
        after_total = after['total']
        after_res_rate = (after['resolved'] / after_total * 100) if after_total > 0 else 0
        after_avg_steps = sum(after['steps']) / len(after['steps']) if after['steps'] else 0
        after_osc_rate = (after['oscillations'] / after_total * 100) if after_total > 0 else 0
        after_plan_rate = (after['plan_compliant'] / after_total * 100) if after_total > 0 else 0
        after_issue_rate = (after['issues'] / after_total * 100) if after_total > 0 else 0

        # Print vanilla row
        row_data = [
            agent,
            model,
            "Before",
            f"{vanilla_res_rate:.1f}%",
            str(vanilla_total)
        ]
        if is_oscillation:
            row_data.extend([
                f"{vanilla_avg_steps:.1f}",
                f"{vanilla_osc_rate:.1f}%",
                f"{vanilla_plan_rate:.1f}%",
                f"{vanilla_issue_rate:.1f}%"
            ])

        row_str = ""
        for i, val in enumerate(row_data):
            row_str += val.ljust(col_widths[i])
        print(row_str)

        # Print after row
        row_data = [
            "",
            "",
            "After",
            f"{after_res_rate:.1f}%",
            str(after_total)
        ]
        if is_oscillation:
            row_data.extend([
                f"{after_avg_steps:.1f}",
                f"{after_osc_rate:.1f}%",
                f"{after_plan_rate:.1f}%",
                f"{after_issue_rate:.1f}%"
            ])

        row_str = ""
        for i, val in enumerate(row_data):
            row_str += val.ljust(col_widths[i])
        print(row_str)

        # Print delta row
        if after_total > 0:
            delta_res = after_res_rate - vanilla_res_rate
            delta_steps = after_avg_steps - vanilla_avg_steps
            delta_osc = after_osc_rate - vanilla_osc_rate
            delta_plan = after_plan_rate - vanilla_plan_rate
            delta_issue = after_issue_rate - vanilla_issue_rate

            row_data = [
                "",
                "",
                "Δ",
                f"{delta_res:+.1f}%",
                ""
            ]
            if is_oscillation:
                row_data.extend([
                    f"{delta_steps:+.1f}",
                    f"{delta_osc:+.1f}%",
                    f"{delta_plan:+.1f}%",
                    f"{delta_issue:+.1f}%"
                ])

            row_str = ""
            for i, val in enumerate(row_data):
                row_str += val.ljust(col_widths[i])
            print(row_str)

        print()

    print("="*120)

    # Print additional run data if with_rerun is enabled
    if with_rerun:
        print("\n" + "="*120)
        print("All Available Run Data")
        print("="*120)

        # Collect all unique run_keys
        all_run_keys = set()
        for m in metrics.values():
            if 'runs' in m:
                all_run_keys.update(m['runs'].keys())

        sorted_run_keys = sorted(all_run_keys)

        for (agent, model) in sorted(metrics.keys()):
            m = metrics[(agent, model)]
            if 'runs' not in m:
                continue

            print(f"\n{agent} - {model}:")
            print("-" * 120)

            # Table header for this agent-model pair
            header = ["Run ID", "Resolution Rate", "Total", "Avg Steps", "Oscillation Rate", "Plan Compliance", "Issue Rate"]
            col_widths = [20, 18, 10, 12, 18, 18, 15]

            header_str = "  "
            for i, col in enumerate(header):
                header_str += col.ljust(col_widths[i])
            print(header_str)

            # Print each run
            for run_key in sorted_run_keys:
                if run_key in m['runs'] and m['runs'][run_key]['total'] > 0:
                    run_data = m['runs'][run_key]
                    total = run_data['total']
                    res_rate = (run_data['resolved'] / total * 100) if total > 0 else 0
                    avg_steps = sum(run_data['steps']) / len(run_data['steps']) if run_data['steps'] else 0
                    osc_rate = (run_data['oscillations'] / total * 100) if total > 0 else 0
                    plan_rate = (run_data['plan_compliant'] / total * 100) if total > 0 else 0
                    issue_rate = (run_data['issues'] / total * 100) if total > 0 else 0

                    row_data = [
                        run_key,
                        f"{res_rate:.1f}%",
                        str(total),
                        f"{avg_steps:.1f}" if is_oscillation else "N/A",
                        f"{osc_rate:.1f}%" if is_oscillation else "N/A",
                        f"{plan_rate:.1f}%" if is_oscillation else "N/A",
                        f"{issue_rate:.1f}%" if is_oscillation else "N/A"
                    ]

                    row_str = "  "
                    for i, val in enumerate(row_data):
                        row_str += val.ljust(col_widths[i])
                    print(row_str)

        print("\n" + "="*120)

    # Overall summary
    print("\n" + "="*120)
    print("Overall Summary")
    print("="*120)

    total_vanilla_resolved = sum(m['vanilla']['resolved'] for m in metrics.values())
    total_vanilla_instances = sum(m['vanilla']['total'] for m in metrics.values())
    total_after_resolved = sum(m['after']['resolved'] for m in metrics.values())
    total_after_instances = sum(m['after']['total'] for m in metrics.values())

    vanilla_overall_rate = (total_vanilla_resolved / total_vanilla_instances * 100) if total_vanilla_instances > 0 else 0
    after_overall_rate = (total_after_resolved / total_after_instances * 100) if total_after_instances > 0 else 0

    print(f"\nBefore Monitor:")
    print(f"  Total instances: {total_vanilla_instances}")
    print(f"  Resolved: {total_vanilla_resolved}")
    print(f"  Resolution rate: {vanilla_overall_rate:.1f}%")

    if is_oscillation:
        all_vanilla_steps = [step for m in metrics.values() for step in m['vanilla']['steps']]
        total_vanilla_osc = sum(m['vanilla']['oscillations'] for m in metrics.values())
        total_vanilla_plan = sum(m['vanilla']['plan_compliant'] for m in metrics.values())
        total_vanilla_issues = sum(m['vanilla']['issues'] for m in metrics.values())

        vanilla_avg_steps = sum(all_vanilla_steps) / len(all_vanilla_steps) if all_vanilla_steps else 0
        vanilla_osc_rate = (total_vanilla_osc / total_vanilla_instances * 100) if total_vanilla_instances > 0 else 0
        vanilla_plan_rate = (total_vanilla_plan / total_vanilla_instances * 100) if total_vanilla_instances > 0 else 0
        vanilla_issue_rate = (total_vanilla_issues / total_vanilla_instances * 100) if total_vanilla_instances > 0 else 0

        print(f"  Average steps: {vanilla_avg_steps:.1f}")
        print(f"  Oscillation rate: {vanilla_osc_rate:.1f}%")
        print(f"  Plan compliance rate: {vanilla_plan_rate:.1f}%")
        print(f"  Issue rate: {vanilla_issue_rate:.1f}%")

    print(f"\nAfter Monitor:")
    print(f"  Total instances: {total_after_instances}")
    print(f"  Resolved: {total_after_resolved}")
    print(f"  Resolution rate: {after_overall_rate:.1f}%")

    if is_oscillation:
        all_after_steps = [step for m in metrics.values() for step in m['after']['steps']]
        total_after_osc = sum(m['after']['oscillations'] for m in metrics.values())
        total_after_plan = sum(m['after']['plan_compliant'] for m in metrics.values())
        total_after_issues = sum(m['after']['issues'] for m in metrics.values())

        after_avg_steps = sum(all_after_steps) / len(all_after_steps) if all_after_steps else 0
        after_osc_rate = (total_after_osc / total_after_instances * 100) if total_after_instances > 0 else 0
        after_plan_rate = (total_after_plan / total_after_instances * 100) if total_after_instances > 0 else 0
        after_issue_rate = (total_after_issues / total_after_instances * 100) if total_after_instances > 0 else 0

        print(f"  Average steps: {after_avg_steps:.1f}")
        print(f"  Oscillation rate: {after_osc_rate:.1f}%")
        print(f"  Plan compliance rate: {after_plan_rate:.1f}%")
        print(f"  Issue rate: {after_issue_rate:.1f}%")

        print(f"\nOverall Improvement:")
        print(f"  Resolution rate: {after_overall_rate - vanilla_overall_rate:+.1f}%")
        print(f"  Average steps: {after_avg_steps - vanilla_avg_steps:+.1f}")
        print(f"  Oscillation rate: {after_osc_rate - vanilla_osc_rate:+.1f}%")
        print(f"  Plan compliance rate: {after_plan_rate - vanilla_plan_rate:+.1f}%")
        print(f"  Issue rate: {after_issue_rate - vanilla_issue_rate:+.1f}%")
    else:
        print(f"\nOverall Improvement:")
        print(f"  Resolution rate: {after_overall_rate - vanilla_overall_rate:+.1f}%")

    print("\n" + "="*120 + "\n")


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

    parser.add_argument(
        "--with_rerun",
        action="store_true",
        help="Include all available run_ids in the analysis (e.g., vanilla_1, after_2, etc.)"
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
    metrics = compute_metrics(data, args.config, args.with_rerun)

    # Print summary table
    print_summary_table(metrics, args.config, args.with_rerun)

    return 0


if __name__ == "__main__":
    exit(main())
