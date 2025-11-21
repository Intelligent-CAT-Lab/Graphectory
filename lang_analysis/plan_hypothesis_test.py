"""
Hypothesis test for whether a given model/agent's trajectory language follows
the intended plan structure significantly more often than random chance.

1. Extract phase skeleton from languatory (run length - agnostic).
2. Check if it hits the given order (L_nav → L_repr → P → V_new_test: default set up for SWE-agent human intended plan, can be customized. Whether and where V_regression_test exist do not matter) by the first appearance.
3. Mark that trajectory as 1 or 0 (product of the binary value of the compliance).
4. Across all tasks for a given model/agent, compute the confidence interval.

Currently only supports single-agent trajectories (only SWE-agent for now).

output_dir defaults to stats/
Two modes:
- multi model: run test and record results for the given agent with multiple models
outputs: {output_dir}/plan_hypothesis_test/{agent}_test.txt
- single model: run test and record results for the given agent with the given model
outputs: {output_dir}/plan_hypothesis_test/{agent}_{model}_test.txt
"""

from __future__ import annotations
import json
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from scipy import stats


@dataclass
class PlanTestConfig:
    """Configuration for plan hypothesis test."""
    agent: str = "SWE-agent"
    models: Optional[List[str]] = None
    data_dir: str = "data"
    output_dir: str = "stats"
    confidence_level: float = 0.95
    expected_order: Tuple[str, ...] = ("L_navigate", "L_reproduce", "P", "V_newly_generated_test")


def extract_phase_prefix(lang_item: str) -> str:
    """
    Extract phase prefix from language item (e.g., 'L_navigate_2' -> 'L_navigate').

    Args:
        lang_item: Language item from trajectory (e.g., 'L_navigate_2', 'P_1')

    Returns:
        Phase prefix without run-length suffix
    """
    # Split by underscore and remove the last numeric part
    parts = lang_item.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return lang_item


def extract_first_appearances(languatory: List[str]) -> Dict[str, int]:
    """
    Extract first appearance indices for each unique phase, accounting for run lengths.

    Args:
        languatory: List of language items with run-length suffixes

    Returns:
        Dictionary mapping phase prefix to its cumulative index (sum of previous run lengths)

    Example:
        ["L_navigate_2", "L_reproduce_2", "L_navigate_1", "P_1"]
        -> {"L_navigate": 0, "L_reproduce": 2, "P": 5}
    """
    first_appearances = {}
    cumulative_idx = 0

    for item in languatory:
        phase = extract_phase_prefix(item)

        # Extract run length from suffix
        parts = item.rsplit('_', 1)
        run_length = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 1

        if phase not in first_appearances:
            first_appearances[phase] = cumulative_idx

        cumulative_idx += run_length

    return first_appearances


def check_trajectory_compliance(
    languatory: List[str],
    expected_order: Tuple[str, ...]
) -> bool:
    """
    Check if a trajectory follows the expected phase order by first appearance.

    The trajectory is compliant if the first appearance indices of the expected
    phases follow a strictly increasing order.

    Args:
        languatory: Trajectory language items
        expected_order: Expected phase order (e.g., ("L_navigate", "L_reproduce", "P", "V_newly_generated_test"))

    Returns:
        True if the first appearances of expected phases occur in the specified order

    Example:
        languatory = ["L_navigate_2", "L_reproduce_2", "L_navigate_1", "P_1",
                      "V_regression_test_3", "V_newly_generated_test_1"]
        expected_order = ("L_navigate", "L_reproduce", "P", "V_newly_generated_test")

        First appearances: L_navigate=0, L_reproduce=2, P=5, V_newly_generated_test=9
        Indices: [0, 2, 5, 9] - strictly increasing, so returns True
    """
    first_appearances = extract_first_appearances(languatory)

    # Check if all expected phases appear in the trajectory
    for phase in expected_order:
        if phase not in first_appearances:
            return False

    # Check if the first appearances are in strictly increasing order
    indices = [first_appearances[phase] for phase in expected_order]
    return all(indices[i] < indices[i + 1] for i in range(len(indices) - 1))


def calculate_confidence_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate proportion and Wilson score confidence interval.

    Args:
        successes: Number of compliant trajectories
        total: Total number of trajectories
        confidence_level: Confidence level (default 0.95 for 95% CI)

    Returns:
        Tuple of (proportion, lower_bound, upper_bound)
    """
    if total == 0:
        return 0.0, 0.0, 0.0

    p = successes / total

    # Wilson score interval
    # z-score for confidence level (1.96 for 95%, 2.576 for 99%)
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence_level, 1.96)

    denominator = 1 + (z * z) / total
    center = (p + (z * z) / (2 * total)) / denominator
    margin = (z * math.sqrt((p * (1 - p) / total) + (z * z) / (4 * total * total))) / denominator

    lower = max(0.0, center - margin)
    upper = min(1.0, center + margin)

    return p, lower, upper


def compute_resolution_compliance_correlation(
    resolution_stats: Dict[str, Tuple[int, int, List[str], List[str]]]
) -> Dict[str, float]:
    """
    Compute statistical correlation between resolution status and compliance.

    Uses Chi-square test for independence and Phi coefficient for effect size.

    Args:
        resolution_stats: Dict mapping resolution status to (compliant_count, total_count, compliant_ids, non_compliant_ids)

    Returns:
        Dict containing:
            - chi2: Chi-square statistic
            - p_value: P-value for the test
            - phi: Phi coefficient (effect size)
            - cramers_v: Cramer's V (alternative effect size measure)
    """
    # Extract counts for 2x2 contingency table
    # Rows: resolved, unresolved
    # Cols: compliant, non-compliant
    resolved_compliant = resolution_stats.get("resolved", [0, 0, [], []])[0]
    resolved_total = resolution_stats.get("resolved", [0, 0, [], []])[1]
    resolved_non_compliant = resolved_total - resolved_compliant

    unresolved_compliant = resolution_stats.get("unresolved", [0, 0, [], []])[0]
    unresolved_total = resolution_stats.get("unresolved", [0, 0, [], []])[1]
    unresolved_non_compliant = unresolved_total - unresolved_compliant

    # Contingency table
    contingency_table = [
        [resolved_compliant, resolved_non_compliant],
        [unresolved_compliant, unresolved_non_compliant]
    ]

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    # Phi coefficient (effect size for 2x2 table)
    n = resolved_total + unresolved_total
    phi = math.sqrt(chi2 / n) if n > 0 else 0.0

    # Cramer's V (same as Phi for 2x2 table)
    cramers_v = phi

    return {
        "chi2": chi2,
        "p_value": p_value,
        "phi": phi,
        "cramers_v": cramers_v,
        "n": n
    }


def test_model(
    agent: str,
    model: str,
    data_dir: str,
    expected_order: Tuple[str, ...]
) -> Tuple[int, int, List[str], List[str], Dict[str, Tuple[int, int, List[str], List[str]]]]:
    """
    Test a single model's trajectories for plan compliance.

    Args:
        agent: Agent name (e.g., "SWE-agent")
        model: Model name (e.g., "claude-sonnet-4")
        data_dir: Root data directory
        expected_order: Expected phase order

    Returns:
        Tuple of (compliant_count, total_count, compliant_ids, non_compliant_ids, resolution_stats)
        where resolution_stats maps "resolved"/"unresolved" to (compliant_count, total_count, compliant_ids, non_compliant_ids)
    """
    lang_file = Path(data_dir) / agent / "langs" / model / "languatory.json"

    if not lang_file.exists():
        raise FileNotFoundError(f"Languatory file not found: {lang_file}")

    with open(lang_file, 'r') as f:
        data = json.load(f)

    compliant_count = 0
    compliant_ids = []
    non_compliant_ids = []

    # Track stats by resolution_status (resolved/unresolved only)
    resolution_stats = {
        "resolved": [0, 0, [], []],  # [compliant_count, total_count, compliant_ids, non_compliant_ids]
        "unresolved": [0, 0, [], []]
    }

    for entry in data:
        instance_id = entry.get("instance_id", "unknown")
        languatory = entry.get("languatory", [])
        resolution_status = entry.get("resolution_status", "")

        # Only process resolved/unresolved
        if resolution_status not in ["resolved", "unresolved"]:
            continue

        is_compliant = check_trajectory_compliance(languatory, expected_order)

        if is_compliant:
            compliant_count += 1
            compliant_ids.append(instance_id)
            resolution_stats[resolution_status][0] += 1
            resolution_stats[resolution_status][2].append(instance_id)
        else:
            non_compliant_ids.append(instance_id)
            resolution_stats[resolution_status][3].append(instance_id)

        resolution_stats[resolution_status][1] += 1

    total_count = len(data)
    return compliant_count, total_count, compliant_ids, non_compliant_ids, resolution_stats


def format_results(
    model: str,
    compliant: int,
    total: int,
    proportion: float,
    lower: float,
    upper: float,
    confidence_level: float,
    compliant_ids: List[str],
    non_compliant_ids: List[str],
    expected_order: Tuple[str, ...],
    resolution_stats: Dict[str, Tuple[int, int, List[str], List[str]]] = None
) -> str:
    """
    Format test results as a readable string.

    Args:
        model: Model name
        compliant: Number of compliant trajectories
        total: Total trajectories
        proportion: Compliance proportion
        lower: Lower bound of CI
        upper: Upper bound of CI
        confidence_level: Confidence level
        compliant_ids: List of compliant instance IDs
        non_compliant_ids: List of non-compliant instance IDs
        expected_order: Expected phase order
        resolution_stats: Optional dict mapping resolution status to stats

    Returns:
        Formatted result string
    """
    result = []
    result.append("=" * 80)
    result.append(f"Plan Hypothesis Test: {model}")
    result.append("=" * 80)
    result.append("")
    result.append(f"Expected Phase Order (by first appearance): {' → '.join(expected_order)}")
    result.append("")
    # result.append(f"Total Trajectories: {total}")
    # result.append(f"Compliant Trajectories: {compliant}")
    # result.append(f"Non-Compliant Trajectories: {total - compliant}")
    # result.append("")
    # result.append(f"Compliance Rate: {proportion:.2%} ({compliant}/{total})")
    # result.append(f"{int(confidence_level * 100)}% Confidence Interval: [{lower:.2%}, {upper:.2%}]")
    # result.append("")

    # Add breakdown by resolution status
    if resolution_stats:
        result.append("-" * 80)
        result.append("Breakdown by Resolution Status:")
        result.append("-" * 80)
        for status in ["resolved", "unresolved"]:
            if status in resolution_stats:
                res_compliant, res_total, res_compliant_ids, res_non_compliant_ids = resolution_stats[status]
                if res_total > 0:
                    res_proportion, res_lower, res_upper = calculate_confidence_interval(
                        res_compliant, res_total, confidence_level
                    )
                    result.append(f"\n{status.capitalize()}:")
                    result.append(f"  Total: {res_total}")
                    result.append(f"  Compliant: {res_compliant}")
                    result.append(f"  Non-Compliant: {res_total - res_compliant}")
                    result.append(f"  Compliance Rate: {res_proportion:.2%} ({res_compliant}/{res_total})")
                    result.append(f"  {int(confidence_level * 100)}% CI: [{res_lower:.2%}, {res_upper:.2%}]")
        result.append("")

        # Add correlation statistics
        correlation = compute_resolution_compliance_correlation(resolution_stats)
        result.append("-" * 80)
        result.append("Statistical Association (Resolution Status ↔ Compliance):")
        result.append("-" * 80)
        result.append(f"Chi-square test: χ² = {correlation['chi2']:.4f}, p = {correlation['p_value']:.4f}")

        # Interpret p-value
        if correlation['p_value'] < 0.001:
            significance = "highly significant (p < 0.001)"
        elif correlation['p_value'] < 0.01:
            significance = "very significant (p < 0.01)"
        elif correlation['p_value'] < 0.05:
            significance = "significant (p < 0.05)"
        else:
            significance = "not significant (p ≥ 0.05)"
        result.append(f"Significance: {significance}")

        result.append(f"Phi coefficient: φ = {correlation['phi']:.4f}")

        # Interpret effect size
        phi_abs = abs(correlation['phi'])
        if phi_abs < 0.1:
            effect_size = "negligible"
        elif phi_abs < 0.3:
            effect_size = "small"
        elif phi_abs < 0.5:
            effect_size = "medium"
        else:
            effect_size = "large"
        result.append(f"Effect size: {effect_size}")

        result.append(f"Sample size: n = {correlation['n']}")
        result.append("")

    result.append("-" * 80)
    result.append(f"Compliant Instances ({len(compliant_ids)}):")
    result.append("-" * 80)
    for instance_id in compliant_ids:
        result.append(f"  - {instance_id}")
    result.append("")
    result.append("-" * 80)
    result.append(f"Non-Compliant Instances ({len(non_compliant_ids)}):")
    result.append("-" * 80)
    for instance_id in non_compliant_ids:
        result.append(f"  - {instance_id}")
    result.append("")
    result.append("=" * 80)

    return "\n".join(result)


def run_single_model_test(config: PlanTestConfig, model: str) -> None:
    """
    Run hypothesis test for a single model and save results.

    Args:
        config: Test configuration
        model: Model name to test
    """
    print(f"Testing model: {model}")

    compliant, total, compliant_ids, non_compliant_ids, resolution_stats = test_model(
        config.agent,
        model,
        config.data_dir,
        config.expected_order
    )

    proportion, lower, upper = calculate_confidence_interval(
        compliant,
        total,
        config.confidence_level
    )

    # Format results
    results_text = format_results(
        model,
        compliant,
        total,
        proportion,
        lower,
        upper,
        config.confidence_level,
        compliant_ids,
        non_compliant_ids,
        config.expected_order,
        resolution_stats
    )

    # Print to console
    print(results_text)

    # Save to file
    output_path = Path(config.output_dir) / "plan_hypothesis_test" / f"{config.agent}_{model}_test.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write(results_text)

    print(f"\nResults saved to: {output_path}")


def run_multi_model_test(config: PlanTestConfig) -> None:
    """
    Run hypothesis test for multiple models and save aggregated results.

    Args:
        config: Test configuration with models list
    """
    if not config.models:
        raise ValueError("No models specified for multi-model test")

    all_results = []

    for model in config.models:
        print(f"\n{'=' * 80}")
        print(f"Testing model: {model}")
        print(f"{'=' * 80}\n")

        try:
            compliant, total, compliant_ids, non_compliant_ids, resolution_stats = test_model(
                config.agent,
                model,
                config.data_dir,
                config.expected_order
            )

            proportion, lower, upper = calculate_confidence_interval(
                compliant,
                total,
                config.confidence_level
            )

            results_text = format_results(
                model,
                compliant,
                total,
                proportion,
                lower,
                upper,
                config.confidence_level,
                compliant_ids,
                non_compliant_ids,
                config.expected_order,
                resolution_stats
            )

            all_results.append(results_text)
            print(results_text)

        except Exception as e:
            error_msg = f"Error testing model {model}: {str(e)}"
            print(error_msg)
            all_results.append(error_msg)

    # Save aggregated results
    output_path = Path(config.output_dir) / "plan_hypothesis_test" / f"{config.agent}_test.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        f.write("\n\n".join(all_results))

    print(f"\n\nAggregated results saved to: {output_path}")


def main():
    """Main entry point for plan hypothesis testing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Test whether agent trajectories follow expected plan structure"
    )
    parser.add_argument(
        "--agent",
        type=str,
        default="SWE-agent",
        help="Agent name (default: SWE-agent)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Single model to test (e.g., claude-sonnet-4)"
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        help="Multiple models to test"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="Root data directory (default: data)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="stats",
        help="Output directory (default: stats)"
    )
    parser.add_argument(
        "--confidence-level",
        type=float,
        default=0.95,
        choices=[0.90, 0.95, 0.99],
        help="Confidence level for interval (default: 0.95)"
    )
    parser.add_argument(
        "--expected-order",
        type=str,
        nargs="+",
        default=["L_navigate", "L_reproduce", "P", "V_newly_generated_test"],
        help="Expected phase order (default: L_navigate L_reproduce P V_newly_generated_test)"
    )

    args = parser.parse_args()

    config = PlanTestConfig(
        agent=args.agent,
        models=args.models,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        confidence_level=args.confidence_level,
        expected_order=tuple(args.expected_order)
    )

    if args.model:
        # Single model test
        run_single_model_test(config, args.model)
    elif args.models:
        # Multi-model test
        run_multi_model_test(config)
    else:
        # Auto-detect all models in data directory
        langs_dir = Path(args.data_dir) / args.agent / "langs"
        if langs_dir.exists():
            models = [d.name for d in langs_dir.iterdir() if d.is_dir()]
            if models:
                print(f"Auto-detected models: {', '.join(models)}")
                config.models = models
                run_multi_model_test(config)
            else:
                print("No models found. Please specify --model or --models.")
        else:
            print(f"Directory not found: {langs_dir}")
            print("Please specify --model or --models.")


if __name__ == "__main__":
    main()
