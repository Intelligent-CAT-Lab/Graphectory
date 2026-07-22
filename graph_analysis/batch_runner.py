"""Batch runner for analyzing trajectory graphs across multiple models and agents."""

import argparse
import sys
from pathlib import Path

from graph_analysis.processor import TrajectoryAnalysisProcessor


DEFAULT_AGENTS = ["SWE-agent", "OpenHands"]
DEFAULT_MODELS = [
    "claude-sonnet-4",
    "deepseek-r1-0528",
    "deepseek-v3",
    "devstral-small"
]


class BatchAnalysisRunner:
    """Orchestrates batch analysis across multiple agents and models."""

    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)

    def discover_models(self, agent: str) -> list[Path]:
        """Discover all model directories for a given agent."""
        agent_graphs_dir = self.data_dir / agent / "graphs"

        if not agent_graphs_dir.exists():
            print(f"Warning: Agent graphs directory does not exist: {agent_graphs_dir}")
            return []

        model_dirs = [d for d in agent_graphs_dir.iterdir() if d.is_dir()]
        return sorted(model_dirs)

    def process_single(self, agent: str, model_graphs_dir: Path) -> Path:
        """Process a single agent-model combination."""
        model_name = model_graphs_dir.name
        output_dir = self.output_dir / agent / "analysis" / model_name

        print(f"\n{'='*60}")
        print(f"Processing {agent} - {model_name}")
        print(f"{'='*60}")

        processor = TrajectoryAnalysisProcessor(
            graphs_dir=model_graphs_dir,
            output_dir=output_dir
        )

        return processor.run()

    def run(
        self,
        agents: list[str] | None = None,
        models: list[str] | None = None
    ) -> dict[str, list[Path]]:
        """
        Run analysis for specified agents and models.

        Args:
            agents: List of agent names to process (None = all discovered)
            models: List of model names to process (None = all discovered)

        Returns:
            Dictionary mapping agent names to list of output files
        """
        agents = agents or DEFAULT_AGENTS
        results = {}

        for agent in agents:
            agent_results = []
            discovered_models = self.discover_models(agent)

            if not discovered_models:
                print(f"No models found for {agent}, skipping...")
                continue

            # Filter by specified models if provided
            if models:
                discovered_models = [
                    d for d in discovered_models
                    if any(model in d.name for model in models)
                ]

            if not discovered_models:
                print(f"No matching models found for {agent}, skipping...")
                continue

            print(f"\nProcessing {len(discovered_models)} model(s) for {agent}")

            for model_dir in discovered_models:
                try:
                    output_file = self.process_single(agent, model_dir)
                    agent_results.append(output_file)
                except Exception as e:
                    print(f"Error processing {agent}/{model_dir.name}: {e}")
                    continue

            results[agent] = agent_results

        return results


def main():
    """CLI entry point for batch analysis."""
    parser = argparse.ArgumentParser(
        description="Batch analyze trajectory graphs for multiple agents and models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Default Configuration:
  Agents: {', '.join(DEFAULT_AGENTS)}
  Models: {', '.join(DEFAULT_MODELS)}

Output Structure:
  {{output_dir}}/{{agent}}/analysis/{{model}}/trajectory_metrics.csv

Examples:
  # Analyze all default agents and models
  python -m graph_analysis.batch_runner

  # Analyze specific agent(s)
  python -m graph_analysis.batch_runner --agents SWE-agent

  # Analyze specific model(s) across all agents
  python -m graph_analysis.batch_runner --models claude-sonnet-4 deepseek-v3

  # Analyze specific agent and model combination
  python -m graph_analysis.batch_runner --agents OpenHands --models deepseek-r1-0528

  # Custom directories
  python -m graph_analysis.batch_runner --data-dir data/samples --output-dir data/samples
        """
    )

    parser.add_argument(
        "--agents",
        nargs="+",
        default=None,
        help=f"Agent(s) to analyze (default: {' '.join(DEFAULT_AGENTS)})"
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        help=f"Model(s) to analyze (default: {' '.join(DEFAULT_MODELS)})"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Base directory containing data (default: data)"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Base directory for output files (default: data)"
    )

    args = parser.parse_args()

    try:
        runner = BatchAnalysisRunner(
            data_dir=args.data_dir,
            output_dir=args.output_dir
        )

        results = runner.run(
            agents=args.agents,
            models=args.models
        )

        # Summary
        print(f"\n{'='*60}")
        print("Batch analysis complete!")
        print(f"{'='*60}")

        total_processed = 0
        for agent, output_files in results.items():
            count = len(output_files)
            total_processed += count
            print(f"{agent}: {count} model(s) processed")

        print(f"\nTotal: {total_processed} model(s) processed")
        print(f"{'='*60}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
