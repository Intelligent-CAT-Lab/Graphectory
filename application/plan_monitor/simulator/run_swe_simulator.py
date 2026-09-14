#!/usr/bin/env python3
"""
SWE-Agent Simulator for testing the Phase Monitor.

Reads trajectory files, extracts actions step-by-step, feeds them to the monitor,
and prints messages when phase transitions occur.
"""

from __future__ import annotations
import time
import argparse
from pathlib import Path
from typing import Optional

from plan_monitor.monitor import StatefulPhaseMonitor
from plan_monitor.simulator.swe_extractor import ActionExtractor


class MonitorSimulator:
    """
    Simulates agent execution by replaying trajectory files through the monitor.

    The simulator:
    1. Reads actions from a trajectory file
    2. Feeds each action to the monitor
    3. Prints messages when phase transitions occur
    4. Optionally adds delays to simulate real-time execution
    """

    def __init__(
        self,
        trajectory_path: str | Path,
        delay: float = 0.05,
        verbose: bool = False,
        enable_rules: bool = True,
        save_graph: bool = False,
        graph_output_dir: Optional[str] = None
    ):
        """
        Initialize the simulator.

        Args:
            trajectory_path: Path to trajectory JSON file
            delay: Delay in seconds between steps (default: 0.0)
            verbose: Print all steps, not just transitions (default: False)
            enable_rules: Enable rule engine for guidance (default: True)
            save_graph: Whether to save graph after simulation (default: False)
            graph_output_dir: Output directory for graphs (default: derived from traj path)
        """
        self.trajectory_path = Path(trajectory_path)
        self.delay = delay
        self.verbose = verbose
        self.enable_rules = enable_rules
        self.save_graph = save_graph
        self.monitor = StatefulPhaseMonitor(enable_rules=enable_rules)
        self.extractor = ActionExtractor(trajectory_path)

        # Determine graph output directory
        if graph_output_dir:
            self.graph_output_dir = graph_output_dir
        else:
            # Convert outputs path to graphs path: SWE-agent/trajectories/X/Y/Z -> SWE-agent/graphs/X/Y/Z
            traj_parts = self.trajectory_path.parts
            if 'trajectories' in traj_parts:
                outputs_idx = traj_parts.index('trajectories')
                # Find SWE-agent in the path
                if 'SWE-agent' in traj_parts:
                    mini_idx = traj_parts.index('SWE-agent')
                    # Construct: SWE-agent/graphs/...
                    self.graph_output_dir = str(Path(*traj_parts[:mini_idx+1]) / 'graphs' / Path(*traj_parts[outputs_idx+1:-1]))
                else:
                    # Fallback: just replace outputs with graphs
                    self.graph_output_dir = str(Path(*(['graphs'] + list(traj_parts[outputs_idx+1:-1]))))
            else:
                self.graph_output_dir = 'graphs'

    def run(self):
        """
        Run the simulation.

        Processes all actions from the trajectory and prints phase transitions.
        """
        instance_id = self.extractor.get_instance_id()
        message_count = self.extractor.get_message_count()

        print(f"{'='*80}")
        print(f"Phase Monitor Simulation")
        print(f"{'='*80}")
        print(f"Instance ID: {instance_id}")
        print(f"Total messages: {message_count}")
        print(f"Rules enabled: {self.enable_rules}")
        print(f"{'='*80}\n")

        total_transitions = 0
        category_transitions = 0
        within_category_transitions = 0
        rule_trigger_count = 0
        dwell_trigger_count = 0
        oscillation_trigger_count = 0
        total_steps = 0

        # Get plan compliance rule from monitor (if enabled)
        plan_rule = None
        if self.enable_rules and self.monitor.rule_engine:
            plan_rule = self.monitor.rule_engine.rules.get("plan_compliance")

        for event, thought, observation in self.extractor.extract_actions():
            total_steps += 1

            # Track phase before processing (for verbose output)
            phase_before = self.monitor.get_current_phase()

            # Track phase history length before processing compound commands
            history_len_before = len(self.monitor.get_phase_history()) if self.verbose else 0

            if self.verbose:
                print(f"\n[Step {event.step_index}]")
                if thought:
                    print(f"Thought: {thought[:100]}{'...' if len(thought) > 100 else ''}")
                print(f"Command: {event.command[:100]}{'...' if len(event.command) > 100 else ''}")
                # if observation:
                #     print(f"Observation: {observation[:100]}{'...' if len(observation) > 100 else ''}")

            # Process the action through the monitor
            result = self.monitor.check_step_pre_emptively(event, thought=thought)

            # Print phase information in verbose mode
            if self.verbose:
                # Get phase from role_history since check_step_pre_emptively updates role_history
                history_after = self.monitor.get_phase_history()

                # Check if this was a compound command by comparing history lengths
                new_phases = history_after[history_len_before:]

                if len(new_phases) > 1:
                    # Compound command with multiple phases
                    print(f"Phases: {' → '.join(new_phases)}")
                elif len(new_phases) == 1:
                    # Single phase added
                    phase_after = new_phases[0]
                    print(f"Phase: {phase_after}")
                elif phase_before:
                    # No new phases added, stay in current phase
                    print(f"Phase: {phase_before}")

            if result:
                # Count phase transitions
                if result.phase_changed:
                    total_transitions += 1
                    if result.category_changed:
                        category_transitions += 1
                    else:
                        within_category_transitions += 1

                # Count rule triggers
                if result.rule_matches:
                    rule_trigger_count += len(result.rule_matches)
                    for match in result.rule_matches:
                        if 'dwell' in match.rule_id:
                            dwell_trigger_count += 1
                        elif 'oscillation' in match.rule_id:
                            oscillation_trigger_count += 1

                # Print result (transitions and/or rule triggers)
                self._print_result(result, event.step_index)

            # Simulate delay
            if self.delay > 0:
                time.sleep(self.delay)

        # Print summary
        print(f"\n{'='*80}")
        print(f"Simulation Complete")
        print(f"{'='*80}")
        print(f"Total steps: {total_steps}")
        print(f"Phase transitions: {total_transitions} (category: {category_transitions}, within: {within_category_transitions})")

        if self.enable_rules:
            transition_shift_count = rule_trigger_count - dwell_trigger_count - oscillation_trigger_count
            print(f"Rules triggered: {rule_trigger_count} (transition/shift: {transition_shift_count}, dwell: {dwell_trigger_count}, oscillation: {oscillation_trigger_count})")

        print(f"Unique phases: {', '.join(self.monitor.get_unique_phases())}")
        print(f"{'='*80}\n")

        # Save graph if requested
        if self.save_graph:
            graph_path = self.monitor.save_graph(self.graph_output_dir, instance_id)
            if graph_path:
                print(f"Graph saved to: {graph_path}\n")

    def _print_result(self, result, step_index: int):
        """
        Print rule messages when triggered.

        Args:
            result: MonitorResult with rule information
            step_index: Current step number
        """
        # Only print when rules are triggered
        all_messages = result.get_all_messages()
        if all_messages:
            print(f"\n[Step {step_index}] Rule triggered:")
            for msg in all_messages:
                print(f"  {msg}")


def main():
    """CLI entry point for the simulator."""
    parser = argparse.ArgumentParser(
        description="Simulate phase monitoring with trajectory files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run simulation on a trajectory file
  python application/plan_monitor/simulator/run_swe_simulator.py data/SWE-agent/trajectories/anthropic_filemap__openrouter--mistralai--devstral-small__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/sympy__sympy-19783/sympy__sympy-19783.traj
  python application/ data/SWE-agent/trajectories/anthropic_filemap__openrouter--deepseek--deepseek-r1-0528__t-0.00__p-1.00__c-2.00___swe_bench_verified_test/django__django-16667/django__django-16667.traj
  python application/plan_monitor/simulator/run_swe_simulator.py application/SWE-agent/trajectories/monitor/exp-1/devstral-small/sphinx-doc__sphinx-9711/sphinx-doc__sphinx-9711.traj
  
  # Run with verbose output
  python run_swe_simulator.py trajectory.json --verbose

  # Run with delay to simulate real-time
  python run_swe_simulator.py trajectory.json --delay 0.5
        """
    )

    parser.add_argument(
        'trajectory',
        type=str,
        help='Path to trajectory JSON file'
    )

    parser.add_argument(
        '--delay',
        type=float,
        default=0.0,
        help='Delay in seconds between steps (default: 0.0)'
    )

    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print all steps, not just rule triggers'
    )

    parser.add_argument(
        '--no-rules',
        action='store_true',
        help='Disable rule engine'
    )

    parser.add_argument(
        '--save-graph',
        action='store_true',
        help='Save graph after simulation'
    )

    parser.add_argument(
        '--graph-output-dir',
        type=str,
        default=None,
        help='Output directory for graphs (default: derived from trajectory path)'
    )

    args = parser.parse_args()

    # Validate trajectory file exists
    trajectory_path = Path(args.trajectory)
    if not trajectory_path.exists():
        print(f"Error: Trajectory file not found: {trajectory_path}")
        return 1

    # Run simulation
    try:
        simulator = MonitorSimulator(
            trajectory_path=trajectory_path,
            delay=args.delay,
            verbose=args.verbose,
            enable_rules=not args.no_rules,
            save_graph=args.save_graph,
            graph_output_dir=args.graph_output_dir
        )
        simulator.run()
        return 0
    except Exception as e:
        print(f"Error running simulation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
