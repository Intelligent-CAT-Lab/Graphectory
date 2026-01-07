"""
Stateful Phase Monitor Implementation.

This module provides a plug-and-play monitor that tracks phase transitions
across agent actions. It maintains state across steps and detects both
coarse-grained (L -> P -> V) and fine-grained phase changes.

The monitor integrates a rule engine that provides:
- Phase transition detection (L -> P -> V)
- Strategy shift detection (e.g., V -> L, P -> L)
- Dwell time monitoring (detecting stuck agents)
"""

from __future__ import annotations
from typing import Optional, Set
from pathlib import Path
from plan_monitor.phases import ActionEvent, MonitorResult, Phase
from plan_monitor.commandParser import CommandParser
from plan_monitor.mapLang import get_action_role
from plan_monitor.rules import RuleEngine
from plan_monitor.buildGraph import GraphBuilder, build_online_graph_from_trajectory, check_command_outcome


class StatefulPhaseMonitor:
    """
    Plug-and-play phase monitor with state tracking and rule engine.

    Detects phase transitions (both category-level and exact phase changes)
    and returns messages when transitions occur. Maintains state for:
    - Current phase
    - Previous roles (for context in mapLang)
    - Created test files (for validation phase detection)
    - Dynamic test suites (for ephemeral validation code)

    Integrates a rule engine that provides:
    - Phase transition guidance (L -> P -> V)
    - Strategy shift detection (e.g., V -> L, P -> L)
    - Dwell time monitoring and intervention

    Usage:
        monitor = StatefulPhaseMonitor()
        for action_event in action_stream:
            result = monitor.on_step(action_event)
            if result:
                # Print all messages (phase change + rule matches)
                for msg in result.get_all_messages():
                    print(msg)
    """

    def __init__(
        self,
        parser: Optional[CommandParser] = None,
        enable_rules: bool = True
    ):
        """
        Initialize the monitor.

        Args:
            parser: Optional CommandParser instance. If not provided, creates default.
            enable_rules: Whether to enable rule engine (default: True)
        """
        # Rule engine integration (create before parser to access config)
        self.enable_rules = enable_rules
        self.rule_engine = RuleEngine() if enable_rules else None

        # Initialize parser with tool configs from rule engine
        if parser is None:
            parser = CommandParser()
            if self.rule_engine:
                # Load tool configs from rule engine's config
                tool_configs = self.rule_engine.config.get("meta_data", {}).get("agent", {}).get("SWE-agent", {}).get("tool_configs", [])
                if tool_configs:
                    # Resolve paths relative to the plan_monitor directory
                    from pathlib import Path
                    base_path = Path(__file__).parent.parent
                    resolved_paths = [str(base_path / config) for config in tool_configs]
                    parser.load_tool_yaml_files(resolved_paths)

        self.parser = parser
        self.current_phase: Optional[Phase] = None
        self.previous_phase: Optional[Phase] = None
        self.role_history: list[str] = []
        self.created_tests: Set[str] = set()
        self.created_dynamic_suites: Set[str] = set()

        # Graph building (built when rules enabled for oscillation detection)
        self.graph_builder = GraphBuilder() if enable_rules else None
        self.step_counter = 0

    def check_step_pre_emptively(
        self,
        event: ActionEvent,
        thought: str = ""
    ) -> Optional[MonitorResult]:
        """
        Check if rules would trigger, updating state only if no blocking rules fire.

        Temporarily updates graph and role_history to detect patterns. Reverts if blocking, commits if not.

        Args:
            event: ActionEvent containing step information
            thought: Optional thought/reasoning text from this step

        Returns:
            MonitorResult with rule_matches if rules trigger, None otherwise
        """
        import copy

        parsed_commands = self.parser.parse(event.command)
        if not parsed_commands:
            return None

        # Backup state
        graph_backup = copy.deepcopy(self.graph_builder.G) if self.graph_builder else None
        prev_node_backup = self.graph_builder.previous_node if self.graph_builder else None
        node_sig_backup = copy.deepcopy(self.graph_builder.node_signature_to_key) if self.graph_builder else None
        role_history_len = len(self.role_history)
        step_counter_backup = self.step_counter

        # Temporarily build graph for this step
        if self.graph_builder:
            build_online_graph_from_trajectory(
                builder=self.graph_builder,
                step_idx=self.step_counter,
                thought=thought,
                action=event.command,
                observation="",
                parser=self.parser
            )
            self.step_counter += 1

        # Process commands and temporarily update role_history
        all_rule_matches = []
        for cmd_info in parsed_commands:
            tool = cmd_info.get("tool")
            subcommand = cmd_info.get("subcommand")
            command = cmd_info.get("command")
            args = cmd_info.get("args", [])
            flags = cmd_info.get("flags", {})

            role = get_action_role(
                tool=tool, subcommand=subcommand, command=command, args=args, flags=flags,
                prev_roles=self.role_history, created_tests=self.created_tests,
                created_dynamic_suites=self.created_dynamic_suites
            )

            # Check plan compliance rule even for "general" roles to detect ending flags
            # (e.g., "submit" command that ends execution)
            if role == "general" and self.enable_rules and self.rule_engine:
                plan_rule = self.rule_engine.rules.get("plan_compliance")
                if plan_rule and plan_rule.enabled:
                    # Check plan compliance for ending flags only (current_phase=None)
                    plan_match = plan_rule.check(
                        current_phase=None,
                        previous_phase=self.current_phase,
                        step_index=event.step_index,
                        command=event.command
                    )
                    if plan_match:
                        all_rule_matches.append(plan_match)
                # Skip further processing for general roles
                continue

            self.role_history.append(role)

            # Check rules for non-general roles
            if self.enable_rules and self.rule_engine:
                rule_matches = self.rule_engine.evaluate(
                    current_phase=Phase(role), previous_phase=self.current_phase,
                    graph=self.graph_builder.G if self.graph_builder else None,
                    step_index=event.step_index, command=event.command, outcome=None
                )
                if rule_matches:
                    all_rule_matches.extend(rule_matches)

        # Check if blocking
        should_block = any(hasattr(m, 'block_execution') and m.block_execution for m in all_rule_matches)

        if should_block:
            # Rollback all changes
            del self.role_history[role_history_len:]
            self.step_counter = step_counter_backup
            if self.graph_builder and graph_backup:
                self.graph_builder.G = graph_backup
                self.graph_builder.previous_node = prev_node_backup
                self.graph_builder.node_signature_to_key = node_sig_backup

            return MonitorResult(
                current_phase=self.current_phase, phase_changed=False, category_changed=False,
                previous_phase=self.previous_phase, rule_matches=all_rule_matches
            )
        else:
            # State already updated, return result
            if all_rule_matches:
                return MonitorResult(
                    current_phase=self.current_phase, phase_changed=False, category_changed=False,
                    previous_phase=self.previous_phase, rule_matches=all_rule_matches
                )
            return None

    def on_step(
        self,
        event: ActionEvent,
        thought: str = "",
        observation: str = ""
    ) -> Optional[MonitorResult]:
        """
        Process a single action event and detect phase transitions.

        Args:
            event: ActionEvent containing step information
            thought: Optional thought/reasoning text from this step
            observation: Optional observation/output from this step

        Returns:
            MonitorResult if a phase change occurred, None otherwise
        """
        # Build graph online if graph builder exists
        if self.graph_builder:
            build_online_graph_from_trajectory(
                builder=self.graph_builder,
                step_idx=self.step_counter,
                thought=thought,
                action=event.command,
                observation=observation,
                parser=self.parser
            )
            self.step_counter += 1

        # Parse the command to extract structured information
        parsed_commands = self.parser.parse(event.command)

        if not parsed_commands:
            # Unable to parse, skip this step
            return None

        # Process ALL parsed commands (commands can be compound with &&, ||, ;)
        # We must process all sub-commands to build the complete languatory/graph
        all_rule_matches = []
        last_significant_result = None

        for cmd_info in parsed_commands:
            result = self._process_command(event, cmd_info, observation)
            if result:
                # Collect rule matches from this sub-command
                if result.rule_matches:
                    all_rule_matches.extend(result.rule_matches)

                # Track if phase actually changed
                if result.phase_changed:
                    last_significant_result = result

        # Return aggregated result if any transitions or rules triggered
        if last_significant_result:
            # Attach all collected rule matches to the final result
            last_significant_result.rule_matches = all_rule_matches
            return last_significant_result

        # No phase changes, but might have rule triggers
        if all_rule_matches:
            # Create a result with current state and all rule matches
            return MonitorResult(
                current_phase=self.current_phase,
                phase_changed=False,
                category_changed=False,
                previous_phase=self.previous_phase,
                rule_matches=all_rule_matches
            )

        return None

    def _process_command(
        self,
        event: ActionEvent,
        cmd_info: dict,
        observation: str = ""
    ) -> Optional[MonitorResult]:
        """
        Process a single parsed command and detect phase transitions.

        Args:
            event: The original ActionEvent
            cmd_info: Parsed command information from CommandParser
            observation: Observation/output from the command

        Returns:
            MonitorResult if phase changed, None otherwise
        """
        # Extract command components
        tool = cmd_info.get("tool")
        subcommand = cmd_info.get("subcommand")
        command = cmd_info.get("command")
        args = cmd_info.get("args", [])
        flags = cmd_info.get("flags", {})

        # Compute outcome for this command
        outcome = check_command_outcome(command, observation, tool, subcommand, args)

        # Get the action role using mapLang
        role = get_action_role(
            tool=tool,
            subcommand=subcommand,
            command=command,
            args=args,
            flags=flags,
            prev_roles=self.role_history,
            created_tests=self.created_tests,
            created_dynamic_suites=self.created_dynamic_suites
        )

        # Check plan compliance rule even for "general" roles to detect ending flags
        # (e.g., "submit" command that ends execution)
        if role == "general":
            if self.enable_rules and self.rule_engine:
                plan_rule = self.rule_engine.rules.get("plan_compliance")
                if plan_rule and plan_rule.enabled:
                    # Check plan compliance for ending flags only (current_phase=None)
                    plan_match = plan_rule.check(
                        current_phase=None,
                        previous_phase=self.current_phase,
                        step_index=event.step_index,
                        command=event.command
                    )
                    if plan_match:
                        # Return result with plan compliance violation
                        return MonitorResult(
                            current_phase=self.current_phase,
                            phase_changed=False,
                            category_changed=False,
                            previous_phase=self.previous_phase,
                            rule_matches=[plan_match]
                        )
            # Skip further processing for general roles
            return None

        # Track role in history
        self.role_history.append(role)

        # Convert role string to Phase object
        new_phase = Phase(role)

        # Detect phase change
        if self.current_phase is None:
            # First phase - just initialize, don't report as a change
            self.current_phase = new_phase
            return None

        # Check if phase changed
        if new_phase != self.current_phase:
            # Detect category change (L -> P -> V)
            category_changed = not new_phase.in_same_group(self.current_phase)

            # Update state
            self.previous_phase = self.current_phase
            self.current_phase = new_phase

            # Evaluate rules if enabled
            rule_matches = []
            if self.enable_rules and self.rule_engine:
                # Pass graph for oscillation detection
                graph = self.graph_builder.G if self.graph_builder else None

                rule_matches = self.rule_engine.evaluate(
                    current_phase=new_phase,
                    previous_phase=self.previous_phase,
                    graph=graph,
                    step_index=event.step_index,
                    command=event.command,
                    outcome=outcome
                )

            return MonitorResult(
                current_phase=new_phase,
                phase_changed=True,
                category_changed=category_changed,
                previous_phase=self.previous_phase,
                rule_matches=rule_matches
            )

        # No phase change, but still check dwell rules, oscillations, and plan compliance
        if self.enable_rules and self.rule_engine:
            # Pass graph for oscillation detection
            graph = self.graph_builder.G if self.graph_builder else None

            rule_matches = self.rule_engine.evaluate(
                current_phase=new_phase,
                previous_phase=None,  # No transition
                graph=graph,
                step_index=event.step_index,
                command=event.command,
                outcome=outcome
            )
            if rule_matches:
                # Return result with only dwell rule matches
                return MonitorResult(
                    current_phase=new_phase,
                    phase_changed=False,
                    category_changed=False,
                    previous_phase=self.previous_phase,
                    rule_matches=rule_matches
                )

        # No change and no rules triggered
        return None

    def reset(self):
        """Reset the monitor state to initial conditions."""
        self.current_phase = None
        self.previous_phase = None
        self.role_history.clear()
        self.created_tests.clear()
        self.created_dynamic_suites.clear()

        # Reset rule engine (includes trigger history)
        if self.rule_engine:
            self.rule_engine.reset()
            self.rule_engine.trigger_history.clear()

        # Reset graph builder
        if self.graph_builder:
            self.graph_builder = GraphBuilder()
            self.step_counter = 0

    def get_current_phase(self) -> Optional[Phase]:
        """Get the current phase."""
        return self.current_phase

    def get_phase_history(self) -> list[str]:
        """Get the full history of roles (including duplicates)."""
        return self.role_history.copy()

    def get_unique_phases(self) -> list[str]:
        """Get unique phases in order of first appearance."""
        seen = set()
        unique = []
        for role in self.role_history:
            if role not in seen:
                seen.add(role)
                unique.append(role)
        return unique

    def save_graph(self, output_dir: str, instance_id: str) -> Optional[str]:
        """
        Save the built graph to disk.

        Args:
            output_dir: Base output directory for saving graphs
            instance_id: Instance identifier (e.g., 'astropy__astropy-12907')

        Returns:
            Path to saved JSON file, or None if graph builder is not available
        """
        if not self.graph_builder:
            return None

        return self.graph_builder.finalize_and_save(output_dir, instance_id)

    def get_graph(self):
        """
        Get the current graph (without saving).

        Returns:
            NetworkX MultiDiGraph or None if graph builder is not available
        """
        if not self.graph_builder:
            return None

        return self.graph_builder.G
