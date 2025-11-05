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
    Extract first appearance indices for each unique phase.

    Args:
        languatory: List of language items with run-length suffixes

    Returns:
        Dictionary mapping phase prefix to its first appearance index

    Example:
        ["L_navigate_2", "L_reproduce_2", "L_navigate_1", "P_1"]
        -> {"L_navigate": 0, "L_reproduce": 1, "P": 3}
    """
    first_appearances = {}

    for idx, item in enumerate(languatory):
        phase = extract_phase_prefix(item)
        if phase not in first_appearances:
            first_appearances[phase] = idx

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

        First appearances: L_navigate=0, L_reproduce=1, P=3, V_newly_generated_test=5
        Indices: [0, 1, 3, 5] - strictly increasing, so returns True
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


def test_model(
    agent: str,
    model: str,
    data_dir: str,
    expected_order: Tuple[str, ...]
) -> Tuple[int, int, List[str], List[str]]:
    """
    Test a single model's trajectories for plan compliance.

    Args:
        agent: Agent name (e.g., "SWE-agent")
        model: Model name (e.g., "claude-sonnet-4")
        data_dir: Root data directory
        expected_order: Expected phase order

    Returns:
        Tuple of (compliant_count, total_count, compliant_ids, non_compliant_ids)
    """
    lang_file = Path(data_dir) / agent / "langs" / model / "languatory.json"

    if not lang_file.exists():
        raise FileNotFoundError(f"Languatory file not found: {lang_file}")

    with open(lang_file, 'r') as f:
        data = json.load(f)

    compliant_count = 0
    compliant_ids = []
    non_compliant_ids = []

    for entry in data:
        instance_id = entry.get("instance_id", "unknown")
        languatory = entry.get("languatory", [])

        if check_trajectory_compliance(languatory, expected_order):
            compliant_count += 1
            compliant_ids.append(instance_id)
        else:
            non_compliant_ids.append(instance_id)

    total_count = len(data)
    return compliant_count, total_count, compliant_ids, non_compliant_ids


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
    expected_order: Tuple[str, ...]
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
    result.append(f"Total Trajectories: {total}")
    result.append(f"Compliant Trajectories: {compliant}")
    result.append(f"Non-Compliant Trajectories: {total - compliant}")
    result.append("")
    result.append(f"Compliance Rate: {proportion:.2%} ({compliant}/{total})")
    result.append(f"{int(confidence_level * 100)}% Confidence Interval: [{lower:.2%}, {upper:.2%}]")
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

    compliant, total, compliant_ids, non_compliant_ids = test_model(
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
        config.expected_order
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
            compliant, total, compliant_ids, non_compliant_ids = test_model(
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
                config.expected_order
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
