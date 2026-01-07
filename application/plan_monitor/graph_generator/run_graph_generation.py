#!/usr/bin/env python3
"""
Generate Graphectory for agent trajectories in a post-hoc manner for analysis.

Reads trajectory files, extracts actions step-by-step, build the Graphectory.
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Optional

from plan_monitor.simulator.mini_extractor import ActionExtractor
from plan_monitor.monitor import StatefulPhaseMonitor


class GraphGenerator:
    """
    Generates Graphectory from trajectory files without simulation.

    Processes trajectories to build and save graphs showing phase transitions
    and action sequences.
    """

    def __init__(self, trajectory_path: Path, graph_output_dir: Optional[str] = None):
        """
        Initialize the graph generator.

        Args:
            trajectory_path: Path to trajectory JSON file
            graph_output_dir: Output directory for graphs (default: derived from traj path)
        """
        self.trajectory_path = trajectory_path
        # Enable rules to ensure graph_builder is initialized
        self.monitor = StatefulPhaseMonitor(enable_rules=True)
        self.extractor = ActionExtractor(trajectory_path)

        # Determine graph output directory
        if graph_output_dir:
            self.graph_output_dir = graph_output_dir
        else:
            # Convert outputs path to graphs path: outputs/X/Y/Z -> graphs/X/Y/Z
            traj_parts = self.trajectory_path.parts
            if 'outputs' in traj_parts:
                outputs_idx = traj_parts.index('outputs')
                # Replace 'outputs' with 'graphs'
                self.graph_output_dir = str(Path(*traj_parts[:outputs_idx]) / 'graphs' / Path(*traj_parts[outputs_idx+1:-1]))
            else:
                self.graph_output_dir = 'graphs'

    def generate(self) -> Optional[str]:
        """
        Generate the graph from the trajectory.

        Returns:
            Path to saved graph JSON file, or None if failed
        """
        instance_id = self.extractor.get_instance_id()

        # Process all actions to build the graph
        step_count = 0
        for event, thought, observation in self.extractor.extract_actions():
            self.monitor.on_step(event, thought=thought, observation=observation)
            step_count += 1

        # Save the graph
        graph_path = self.monitor.save_graph(self.graph_output_dir, instance_id)

        if not graph_path:
            raise RuntimeError(f"Failed to save graph for {instance_id} (processed {step_count} steps)")

        return graph_path


def process_single_file(trajectory_path: Path, graph_output_dir: Optional[str] = None, quiet: bool = False) -> bool:
    """
    Process a single trajectory file.

    Args:
        trajectory_path: Path to trajectory file
        graph_output_dir: Optional custom output directory
        quiet: Suppress output messages

    Returns:
        True if successful, False otherwise
    """
    try:
        generator = GraphGenerator(trajectory_path, graph_output_dir)
        graph_path = generator.generate()

        if graph_path:
            if not quiet:
                print(f"✓ {trajectory_path.name} -> {graph_path}")
            return True
        else:
            if not quiet:
                print(f"✗ {trajectory_path.name} (failed to save graph)")
            return False
    except Exception as e:
        if not quiet:
            print(f"✗ {trajectory_path.name} (error: {e})")
        return False


def process_directory(input_path: Path, graph_output_dir: Optional[str] = None, quiet: bool = False) -> tuple[int, int]:
    """
    Process all trajectory files in a directory recursively.

    Args:
        input_path: Directory containing trajectory files
        graph_output_dir: Optional custom output directory
        quiet: Suppress output messages

    Returns:
        Tuple of (success_count, total_count)
    """
    # Find all .traj.json files
    traj_files = sorted(input_path.rglob("*.traj.json"))

    if not traj_files:
        print(f"No trajectory files found in {input_path}")
        return 0, 0

    total_count = len(traj_files)
    if not quiet:
        print(f"Processing {total_count} trajectory files")
        print(f"{'='*80}")

    success_count = 0
    failed_files = []

    for i, traj_file in enumerate(traj_files, 1):
        # For batch processing, derive output dir per file unless override specified
        output_dir = graph_output_dir if graph_output_dir else None

        try:
            if process_single_file(traj_file, output_dir, quiet):
                success_count += 1
            else:
                failed_files.append(str(traj_file.name))
        except KeyboardInterrupt:
            print(f"\n\nInterrupted at {i}/{total_count}")
            print(f"Completed: {success_count} successful")
            raise

        # Progress indicator for large batches
        if not quiet and total_count > 20 and i % 50 == 0:
            print(f"[{i}/{total_count}] {success_count} successful, {i - success_count} failed")

    return success_count, total_count


def main():
    """CLI entry point for graph generation."""
    parser = argparse.ArgumentParser(
        description="Generate Graphectory from trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single trajectory file
  python run_graph_generation.py outputs/instance1/trajectory.traj.json

  # Process all trajectories in a directory
  python plan_monitor/graph_generator/run_graph_generation.py mini-swe-agent/outputs/swebench/gpt-5-mini

  # Custom output directory
  python run_graph_generation.py outputs/ --output-dir custom_graphs/

  # Quiet mode (only show summary)
  python run_graph_generation.py outputs/ --quiet
        """
    )

    parser.add_argument(
        'input_path',
        type=str,
        help='Path to trajectory file or directory containing trajectories'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Custom output directory (default: auto-detect from input path)'
    )

    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Quiet mode - only show summary'
    )

    args = parser.parse_args()

    # Validate input path exists
    input_path = Path(args.input_path)
    if not input_path.exists():
        print(f"Error: Input path not found: {input_path}")
        return 1

    # Process file or directory
    try:
        if input_path.is_file():
            # Single file processing
            if not args.quiet:
                print(f"Processing: {input_path}")
                print(f"{'='*80}")

            success = process_single_file(input_path, args.output_dir, args.quiet)

            if not args.quiet:
                print(f"{'='*80}")
                print(f"Result: {'Success' if success else 'Failed'}")

            return 0 if success else 1

        elif input_path.is_dir():
            # Batch directory processing
            success_count, total_count = process_directory(input_path, args.output_dir, args.quiet)

            if not args.quiet or total_count > 0:
                print(f"{'='*80}")
                print(f"Completed: {success_count}/{total_count} graphs generated successfully")
                if success_count < total_count:
                    print(f"Failed: {total_count - success_count} trajectories")

            return 0 if success_count == total_count else 1

        else:
            print(f"Error: Invalid input path type: {input_path}")
            return 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
