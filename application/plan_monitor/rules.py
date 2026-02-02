"""
Rule Engine for Phase Transition and Trajectory Monitoring.

This module provides a flexible, extensible rule-based system for detecting
and responding to patterns in agent trajectories. The system supports:

1. Languatory-based rules (phase transitions, strategy shifts, dwell times)
2. Graphectory-based rules (oscillation detection)
3. Plan compliance rules (intended workflow enforcement)
4. Rule selection, combination, and configuration via JSON

Design principles:
- Clean separation between rule logic and rule messages
- Abstract base classes for extensibility
- Rule registry pattern for dynamic rule loading
- Support for both languatory and graphectory inputs
"""

from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Set, Dict, Any
from plan_monitor.phases import Phase


@dataclass
class RuleMatch:
    """Result of a rule evaluation."""
    rule_id: str
    message: str
    triggered: bool = True
    block_execution: bool = False  # If True, prevents action execution and history addition
    metadata: Dict[str, Any] = field(default_factory=dict)


class Rule(ABC):
    """Abstract base class for all rules."""

    def __init__(self, rule_id: str, message: str):
        self.rule_id = rule_id
        self.message = message
        self.enabled = True

    @abstractmethod
    def check(self, **kwargs) -> Optional[RuleMatch]:
        """Check if the rule is triggered."""
        pass

    @abstractmethod
    def reset(self):
        """Reset rule state."""
        pass


class PhaseTransitionRule(Rule):
    """Rule for coarse-grained phase transitions (L -> P -> V)."""

    def __init__(self, rule_id: str, message: str, from_category: str, to_category: str, threshold: Optional[int] = None):
        super().__init__(rule_id, message)
        self.from_category = from_category
        self.to_category = to_category
        self.threshold = threshold
        self.trigger_count: int = 0

    def check(self, from_phase: Phase = None, to_phase: Phase = None, **kwargs) -> Optional[RuleMatch]:
        if not self.enabled or from_phase is None or to_phase is None:
            return None

        # Only trigger for category changes
        if from_phase.in_same_group(to_phase):
            return None

        # Check if this is the transition we're watching for
        if from_phase.prefix != self.from_category or to_phase.prefix != self.to_category:
            return None

        # Check threshold: if set, stop triggering after reaching it
        if self.threshold is not None and self.trigger_count >= self.threshold:
            return None

        self.trigger_count += 1
        return RuleMatch(
            rule_id=self.rule_id,
            message=self.message,
            metadata={"transition": f"{self.from_category}->{self.to_category}", "type": "forward"}
        )

    def reset(self):
        self.trigger_count = 0


class StrategyShiftRule(Rule):
    """Rule for detecting strategy shifts and backtracks."""

    def __init__(self, rule_id: str, message: str, from_category: str, to_category: str, shift_type: str, threshold: Optional[int] = None):
        super().__init__(rule_id, message)
        self.from_category = from_category
        self.to_category = to_category
        self.shift_type = shift_type  # 'shortcut', 'rework', 'backtrack'
        self.threshold = threshold
        self.trigger_count: int = 0

    def check(self, from_phase: Phase = None, to_phase: Phase = None, **kwargs) -> Optional[RuleMatch]:
        if not self.enabled or from_phase is None or to_phase is None:
            return None

        # Check if this is the shift we're watching for
        if from_phase.prefix != self.from_category or to_phase.prefix != self.to_category:
            return None

        # Check threshold: if set, stop triggering after reaching it; if None, always trigger
        if self.threshold is not None and self.trigger_count >= self.threshold:
            return None

        self.trigger_count += 1
        return RuleMatch(
            rule_id=self.rule_id,
            message=self.message,
            metadata={"transition": f"{self.from_category}->{self.to_category}", "type": self.shift_type}
        )

    def reset(self):
        self.trigger_count = 0


class DwellTimeRule(Rule):
    """Rule for detecting abnormally long phase dwells."""

    def __init__(self, rule_id: str, message: str, threshold: int,
                 phase_exact: Optional[str] = None, phase_category: Optional[str] = None,
                 block_execution: bool = False):
        super().__init__(rule_id, message)
        self.threshold = threshold
        self.phase_exact = phase_exact  # Exact phase match (e.g., 'L_navigate')
        self.phase_category = phase_category  # Category match (e.g., 'P', 'V')
        self.block_execution = block_execution

        # State tracking
        self.current_phase: Optional[Phase] = None
        self.dwell_count: int = 0
        self.triggered_dwells: Set[tuple[str, int]] = set()

    def check(self, current_phase: Phase = None, **kwargs) -> Optional[RuleMatch]:
        if not self.enabled or current_phase is None:
            return None

        # Check if phase changed
        if self.current_phase is None or current_phase != self.current_phase:
            self.current_phase = current_phase
            self.dwell_count = 1
            return None

        # Same phase - increment dwell count
        self.dwell_count += 1

        # Check if this rule applies to current phase
        if not self._matches_phase(current_phase):
            return None

        # Check threshold
        if self.dwell_count < self.threshold:
            return None

        # # Check if already triggered for this phase at this threshold
        # trigger_key = (current_phase.value, self.threshold)
        # if trigger_key in self.triggered_dwells:
        #     return None

        # self.triggered_dwells.add(trigger_key)
        return RuleMatch(
            rule_id=self.rule_id,
            message=self.message,
            block_execution=self.block_execution,
            metadata={
                "phase": current_phase.value,
                "threshold": self.threshold,
                "dwell_count": self.dwell_count
            }
        )

    def _matches_phase(self, phase: Phase) -> bool:
        """Check if this rule applies to the given phase."""
        if self.phase_exact and phase.value == self.phase_exact:
            return True
        if self.phase_category and phase.prefix == self.phase_category:
            return True
        return False

    def reset(self):
        self.current_phase = None
        self.dwell_count = 0
        self.triggered_dwells.clear()


class OscillationDetector:
    """
    Efficient oscillation detector analyzing graph topology.

    The graph is built externally during execution. This detector analyzes
    execution edges to identify oscillation patterns.

    Key optimization: Only analyze when current node creates a back edge.
    """

    def __init__(self, block_execution: bool = False):
        self.triggered_patterns: Dict[str, int] = {}  # Track warned patterns with last trigger step
        self.pattern_counts: Dict[str, int] = {}  # Track how many times each pattern triggered
        self.block_execution = block_execution

    def detect(self, graph, thresholds: Dict[str, Any]) -> Optional[RuleMatch]:
        """
        Detect oscillations by analyzing graph execution path.

        Args:
            graph: NetworkX MultiDiGraph with nodes containing step_indices
            thresholds: Oscillation thresholds from config

        Returns:
            RuleMatch if oscillation detected, None otherwise
        """
        if graph is None or len(graph.nodes()) == 0:
            return None

        # Extract execution path and current step from graph
        exec_path, current_step = self._extract_exec_path(graph)
        if len(exec_path) < 2:
            return None

        # Heuristic: Only check when back edge exists (last node appears earlier in path)
        if exec_path[-1] not in exec_path[:-1]:
            return None

        # Extract a window of recent nodes to analyze for patterns
        # Use last 30 nodes to capture multiple iterations of cycles
        window_size = min(30, len(exec_path))
        loop_segment = exec_path[-window_size:]

        # Check patterns: self-loop → two-node → multi-node → loop-family
        max_repeats = thresholds.get("max_repeats", 5)
        max_repeats_message = thresholds.get("max_repeats_rule", {}).get("message", "If you cannot make further progress, stop and exit or submit your patch.")

        # 1. Self-loop: [A, A, ...]
        if self._is_self_loop(loop_segment, thresholds["self_loop"]["threshold"]):
            pattern_id = f"self:{exec_path[-1]}"
            last_trigger = self.triggered_patterns.get(pattern_id, -float('inf'))
            if current_step - last_trigger >= 0:  # Always allow, rate_limit handles gap
                self.triggered_patterns[pattern_id] = current_step
                self.pattern_counts[pattern_id] = self.pattern_counts.get(pattern_id, 0) + 1

                # Check if max_repeats reached
                if self.pattern_counts[pattern_id] >= max_repeats:
                    return RuleMatch(
                        rule_id="oscillation_max_repeats",
                        message=max_repeats_message,
                        block_execution=self.block_execution,
                        metadata={"type": "self_loop", "node": exec_path[-1], "repeat_count": self.pattern_counts[pattern_id]}
                    )

                return RuleMatch(
                    rule_id="oscillation_self_loop",
                    message=thresholds["self_loop"]["message"],
                    block_execution=self.block_execution,
                    metadata={"type": "self_loop", "node": exec_path[-1]}
                )

        # 2. Two-node cycle: [A, B, A, B, ...]
        two_cycle = self._detect_two_node_cycle(loop_segment, thresholds["two_node_cycle"]["threshold"])
        if two_cycle:
            pattern_id = f"two:{'-'.join(sorted(two_cycle))}"
            last_trigger = self.triggered_patterns.get(pattern_id, -float('inf'))
            if current_step - last_trigger >= 0:  # Always allow, rate_limit handles gap
                self.triggered_patterns[pattern_id] = current_step
                self.pattern_counts[pattern_id] = self.pattern_counts.get(pattern_id, 0) + 1

                # Check if max_repeats reached
                if self.pattern_counts[pattern_id] >= max_repeats:
                    return RuleMatch(
                        rule_id="oscillation_max_repeats",
                        message=max_repeats_message,
                        block_execution=self.block_execution,
                        metadata={"type": "two_node_cycle", "nodes": two_cycle, "repeat_count": self.pattern_counts[pattern_id]}
                    )

                return RuleMatch(
                    rule_id="oscillation_two_node",
                    message=thresholds["two_node_cycle"]["message"],
                    block_execution=self.block_execution,
                    metadata={"type": "two_node_cycle", "nodes": two_cycle}
                )

        # 3. Multi-node cycle: [A, B, C, A, B, C, ...]
        multi_cycle = self._detect_multi_node_cycle(loop_segment, thresholds["multi_node_cycle"]["threshold"])
        if multi_cycle:
            pattern_id = f"multi:{'-'.join(multi_cycle)}"
            last_trigger = self.triggered_patterns.get(pattern_id, -float('inf'))
            if current_step - last_trigger >= 0:  # Always allow, rate_limit handles gap
                self.triggered_patterns[pattern_id] = current_step
                self.pattern_counts[pattern_id] = self.pattern_counts.get(pattern_id, 0) + 1

                # Check if max_repeats reached
                if self.pattern_counts[pattern_id] >= max_repeats:
                    return RuleMatch(
                        rule_id="oscillation_max_repeats",
                        message=max_repeats_message,
                        block_execution=self.block_execution,
                        metadata={"type": "multi_node_cycle", "pattern": multi_cycle, "repeat_count": self.pattern_counts[pattern_id]}
                    )

                return RuleMatch(
                    rule_id="oscillation_multi_node",
                    message=thresholds["multi_node_cycle"]["message"],
                    block_execution=self.block_execution,
                    metadata={"type": "multi_node_cycle", "pattern": multi_cycle}
                )

        # 4. Loop family: Switching between different loops
        loop_family = self._detect_loop_family(
            exec_path,
            thresholds["loop_family"]["threshold_outer"],
            thresholds["loop_family"]["threshold_inner"]
        )
        if loop_family:
            pattern_id = f"family:{hash(frozenset(str(l) for l in loop_family))}"
            last_trigger = self.triggered_patterns.get(pattern_id, -float('inf'))
            if current_step - last_trigger >= 0:  # Always allow, rate_limit handles gap
                self.triggered_patterns[pattern_id] = current_step
                self.pattern_counts[pattern_id] = self.pattern_counts.get(pattern_id, 0) + 1

                # Check if max_repeats reached
                if self.pattern_counts[pattern_id] >= max_repeats:
                    return RuleMatch(
                        rule_id="oscillation_max_repeats",
                        message=max_repeats_message,
                        block_execution=self.block_execution,
                        metadata={"type": "loop_family", "repeat_count": self.pattern_counts[pattern_id]}
                    )

                return RuleMatch(
                    rule_id="oscillation_loop_family",
                    message=thresholds["loop_family"]["message"],
                    block_execution=self.block_execution,
                    metadata={"type": "loop_family"}
                )

        return None

    def _extract_exec_path(self, graph) -> tuple[list[str], int]:
        """
        Extract execution path by collecting all (node, step_idx) pairs and sorting.

        Each node tracks step_indices in its properties. We flatten these to get
        the full execution sequence.

        Args:
            graph: NetworkX MultiDiGraph with nodes containing step_indices

        Returns:
            Tuple of (list of node keys in execution order, current step index)
        """
        # Collect all (step_idx, node_key) pairs
        step_node_pairs = []
        for node_key, data in graph.nodes(data=True):
            step_indices = data.get('step_indices', [])
            for step_idx in step_indices:
                step_node_pairs.append((step_idx, node_key))

        # Sort by step index
        step_node_pairs.sort(key=lambda x: x[0])
        node_keys = [node_key for _, node_key in step_node_pairs]
        current_step = step_node_pairs[-1][0] if step_node_pairs else 0
        return node_keys, current_step

    def _is_self_loop(self, segment: list[str], threshold: int) -> bool:
        """
        Check if the last N consecutive nodes are identical (self-loop).

        A self-loop means the agent is repeating the exact same action.
        We look at the end of the segment for consecutive repetitions.
        """
        if len(segment) < threshold:
            return False

        # Check if the last 'threshold' nodes are all the same
        tail = segment[-threshold:]
        return len(set(tail)) == 1

    def _detect_two_node_cycle(self, segment: list[str], threshold: int) -> Optional[tuple[str, str]]:
        """
        Detect A-B-A-B alternating pattern with >= threshold complete cycles.

        A complete cycle is (A,B) or (B,A). Threshold=2 means A-B-A-B (4 nodes minimum).
        Looks for the longest alternating subsequence at the end of the segment.
        """
        if len(segment) < 2:
            return None

        # Try to find the longest alternating pattern at the end
        # Work backwards to find where alternation starts
        for start in range(max(0, len(segment) - 20), len(segment) - 1):
            subseg = segment[start:]
            if len(subseg) < threshold * 2:
                continue

            unique = list(dict.fromkeys(subseg))
            if len(unique) != 2:
                continue

            a, b = unique

            # Check if this is a valid alternating pattern
            # Count consecutive alternating pairs from the start
            is_alternating = True
            pairs_count = 0
            for i in range(len(subseg) - 1):
                if (subseg[i], subseg[i+1]) in [(a, b), (b, a)]:
                    pairs_count += 1
                else:
                    is_alternating = False
                    break

            if is_alternating and pairs_count >= threshold:
                return (a, b)

        return None

    def _detect_multi_node_cycle(self, segment: list[str], threshold: int) -> Optional[tuple[str, ...]]:
        """
        Detect repeating pattern of 3+ nodes: A-B-C-A-B-C repeated >= threshold times.
        Searches for patterns at the end of the segment.
        """
        if len(segment) < 6:  # Minimum: 3-node pattern repeated twice
            return None

        # Try to find repeating patterns at the end, working backwards
        # Try cycle lengths from 3 to 10 (reasonable upper bound for patterns)
        max_cycle_len = min(10, len(segment) // threshold)

        for cycle_len in range(3, max_cycle_len + 1):
            # Try different starting positions near the end
            for start in range(max(0, len(segment) - cycle_len * threshold - 5), len(segment) - cycle_len * threshold + 1):
                subseg = segment[start:]
                if len(subseg) < cycle_len * threshold:
                    continue

                pattern = tuple(subseg[:cycle_len])

                # Count consecutive repetitions from the start
                reps = 0
                for i in range(0, len(subseg), cycle_len):
                    if tuple(subseg[i:i+cycle_len]) == pattern:
                        reps += 1
                    else:
                        break

                if reps >= threshold:
                    return pattern

        return None

    def _detect_loop_family(self, exec_path: list[str], threshold_outer: int, threshold_inner: int) -> Optional[list]:
        """
        Detect repeating alternation pattern between distinct loops.

        Loop family: A sequence of distinct loops that repeats threshold_outer times.
        Example with threshold_outer=2:
        - Loops found: [Loop1, Loop2, Loop1, Loop2] → pattern [Loop1, Loop2] repeats 2x ✓
        - Loops found: [Loop1, Loop2, Loop3, Loop1, Loop2, Loop3] → pattern [L1, L2, L3] repeats 2x ✓
        - Loops found: [Loop1, Loop2, Loop1] → pattern only 1.5x ✗

        Args:
            exec_path: Full execution path
            threshold_outer: Number of times the alternation pattern must repeat
            threshold_inner: Minimum size for each loop (unused, kept for compatibility)

        Returns:
            List of loops in the repeating pattern if detected
        """
        if len(exec_path) < 6:
            return None

        # Extract loops from recent history
        recent = exec_path[-min(30, len(exec_path)):]
        loops_sequence = []

        i = 0
        while i < len(recent) - 1:
            start_node = recent[i]
            for j in range(i + 1, min(i + 15, len(recent))):
                if recent[j] == start_node:
                    loops_sequence.append(tuple(recent[i:j]))
                    i = j
                    break
            else:
                i += 1

        if len(loops_sequence) < threshold_outer * 2:  # Need at least 2 loops repeated threshold_outer times
            return None

        # Try to find a repeating pattern of distinct loops
        # Pattern lengths from 2 to len(loops_sequence) // threshold_outer
        for pattern_len in range(2, len(loops_sequence) // threshold_outer + 1):
            pattern = loops_sequence[:pattern_len]

            # Check if all loops in pattern are distinct
            if len(set(pattern)) != len(pattern):
                continue

            # Count how many times this pattern repeats consecutively
            reps = 0
            pos = 0
            while pos + pattern_len <= len(loops_sequence):
                if loops_sequence[pos:pos + pattern_len] == pattern:
                    reps += 1
                    pos += pattern_len
                else:
                    break

            if reps >= threshold_outer:
                return pattern

        return None

    def reset(self):
        """Reset triggered patterns tracking."""
        self.triggered_patterns.clear()
        self.pattern_counts.clear()


class GraphectoryRule(Rule):
    """Base class for graph-based rules."""

    def check(self, graph=None, **kwargs) -> Optional[RuleMatch]:
        return None

    def reset(self):
        pass


class OscillationRule(GraphectoryRule):
    """Detects oscillation patterns in graph trajectories."""

    def __init__(self, config: Dict[str, Any], block_execution: bool = False):
        super().__init__("oscillation", "")
        self.config = config
        self.detector = OscillationDetector(block_execution=block_execution)

    def check(self, graph=None, **kwargs) -> Optional[RuleMatch]:
        if not self.enabled or graph is None:
            return None

        return self.detector.detect(graph, self.config)

    def reset(self):
        self.detector.reset()


class PlanComplianceRule(Rule):
    """Rule for detecting deviations from intended execution plan."""

    def __init__(self, rule_id: str, message: str, intended_plan: list[str],
                 ending_flags: Optional[list[str]] = None,
                 phase_names: Optional[Dict[str, str]] = None,
                 block_execution: bool = False,
                 threshold: int = 3):
        """
        Initialize plan compliance rule.

        Args:
            rule_id: Unique identifier for this rule
            message: Message template for violations
            intended_plan: Expected sequence of phase categories (e.g., ["L_navigate", "L_reproduce", "P", "V"])
            ending_flags: Optional list of ending indicators (e.g., ["echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT", "submit"])
            phase_names: Optional mapping of phase codes to human-readable names for messages
            block_execution: Whether this rule should block execution and drop the current step
            threshold: Maximum number of times this rule can trigger (default: 3)
        """
        super().__init__(rule_id, message)
        self.intended_plan = intended_plan
        self.ending_flags = ending_flags or []
        self.phase_names = phase_names or {}
        self.block_execution = block_execution
        self.threshold = threshold

        # Track first appearance of each phase category
        self.seen_phases: Dict[str, int] = {}  # phase_category -> step_index of first appearance
        self.plan_index = 0  # Current position in intended plan
        self.has_ended = False
        self.violation_triggered = False
        self.violation_count = 0  # Number of times violations have been triggered

    def _get_phase_name(self, phase_code: str) -> str:
        """Get human-readable name for a phase code."""
        return self.phase_names.get(phase_code, phase_code)

    def check(self, current_phase: Phase = None, previous_phase: Phase = None,
              step_index: int = None, command: str = None, **kwargs) -> Optional[RuleMatch]:
        """
        Check if execution deviates from intended plan.

        Args:
            current_phase: Current phase
            previous_phase: Previous phase
            step_index: Current step index
            command: Raw command string (for ending flag detection)
            **kwargs: Additional context

        Returns:
            RuleMatch if plan violation detected, None otherwise
        """
        if not self.enabled or self.violation_triggered:
            return None

        # Check threshold: stop triggering after reaching max violations
        if self.violation_count >= self.threshold:
            return None

        # Check for ending flags (can be checked even without phase transitions)
        if self.ending_flags and command and not self.has_ended:
            for ending_flag in self.ending_flags:
                if ending_flag in command:
                    self.has_ended = True
                    # Check if we completed the plan before ending
                    if self.plan_index < len(self.intended_plan):
                        self.violation_triggered = True
                        self.violation_count += 1
                        skipped_phases = self.intended_plan[self.plan_index:]
                        skipped_names = [self._get_phase_name(p) for p in skipped_phases]
                        return RuleMatch(
                            rule_id=self.rule_id,
                            message=self.message.format(
                                violation="ended execution",
                                expected=f"complete phases: {', '.join(skipped_names)}",
                                skipped=', '.join(skipped_names)
                            ),
                            block_execution=self.block_execution,
                            metadata={
                                "type": "premature_ending",
                                "skipped_phases": skipped_phases,
                                "current_index": self.plan_index
                            }
                        )
                    # Completed plan successfully, no violation
                    return None

        # Need current phase for plan checking
        if current_phase is None:
            return None

        # Optimization: only check when phase changes (avoid checking same phase repeatedly)
        if previous_phase is not None and current_phase == previous_phase:
            return None

        # Get the current phase category (exact match or prefix)
        current_category = current_phase.value  # Use full phase name first

        # Get role_history to distinguish between successful execution vs blocked attempts
        role_history = kwargs.get('role_history', [])

        # Only check plan compliance on FIRST appearance of a phase category in role_history
        # (i.e., first SUCCESSFUL execution, not first attempt)
        if current_category in self.seen_phases:
            # Phase was already attempted
            if current_category in role_history:
                # Phase was successfully executed before (in role_history), allow revisit
                return None
            else:
                # Phase was attempted but blocked (in seen_phases but not in role_history)
                # Agent is retrying the same phase
                # Don't mark it in seen_phases again, and skip reporting (already triggered once)
                return None

        # Check if current phase matches expected position in plan
        if self.plan_index >= len(self.intended_plan):
            # Already completed the plan, no more violations to detect
            return None

        expected_phase = self.intended_plan[self.plan_index]

        # Check if we've already seen a phase that matches the expected phase
        # This handles cases like: expected="V", seen=["V_regression_test"]
        for seen_phase in self.seen_phases:
            # Check if seen phase matches expected phase (exact or prefix match)
            if seen_phase == expected_phase or seen_phase.startswith(expected_phase + '_'):
                # Already seen a phase matching the expected, no need to check again
                return None

        # Mark this as first appearance
        self.seen_phases[current_category] = step_index or 0

        # Check if current phase matches expected (exact match or prefix match)
        if current_category == expected_phase or current_category.startswith(expected_phase + '_'):
            # On track - advance to next expected phase
            self.plan_index += 1
            return None

        # Check if current phase appears later in the plan (skip detected)
        found_index = -1

        for i in range(self.plan_index, len(self.intended_plan)):
            if self.intended_plan[i] == current_category or \
               (current_phase.prefix and self.intended_plan[i] == current_phase.prefix):
                found_index = i
                break

        if found_index > self.plan_index:
            # Found later in plan - phases were skipped
            skipped_phases = self.intended_plan[self.plan_index:found_index]
            skipped_names = [self._get_phase_name(p) for p in skipped_phases]
            current_name = self._get_phase_name(current_category)
            self.plan_index = found_index + 1
            self.violation_triggered = True
            self.violation_count += 1

            return RuleMatch(
                rule_id=self.rule_id,
                message=self.message.format(
                    violation=f"skipped from expected phase to {current_name}",
                    expected=f"complete {', '.join(skipped_names)} first",
                    skipped=', '.join(skipped_names)
                ),
                block_execution=self.block_execution,
                metadata={
                    "type": "phase_skip",
                    "current": current_category,
                    "skipped_phases": skipped_phases
                }
            )

        # Current phase not in plan - this is acceptable (optional phases allowed)
        return None

    def reset(self):
        """Reset rule state."""
        self.seen_phases.clear()
        self.plan_index = 0
        self.has_ended = False
        self.violation_triggered = False
        self.violation_count = 0


class RuleEngine:
    """
    Unified rule engine with flexible rule management.

    Supports:
    - Loading rules from JSON configuration
    - Dynamic rule selection and combination
    - Both languatory and graphectory inputs
    - Rule enable/disable at runtime
    - Rate limiting to prevent excessive rule triggering
    - Plan compliance checking
    """

    def __init__(self, config_path: Optional[str] = None, rule_set: Optional[Set[str]] = None):
        """
        Initialize the rule engine.

        Args:
            config_path: Path to rules.json configuration file
            rule_set: Set of rule IDs to enable (None = all rules)
        """
        self.rules: Dict[str, Rule] = {}
        self.config: Dict[str, Any] = {}

        # Rate limiting state
        self.rate_limit_enabled: bool = False
        self.min_gap: int = 5
        self.trigger_history: list[int] = []  # List of step indices where rules triggered

        # Load configuration
        if config_path is None:
            config_path = Path(__file__).parent / "config" / "default_rules.json"
        self._load_config(config_path)

        # Initialize rules from config
        self._initialize_rules(rule_set)

    def _load_config(self, config_path: Path):
        """Load rule configuration from JSON file."""
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        # Load rate limiting configuration
        rate_limit_config = self.config.get("rate_limit", {})
        self.rate_limit_enabled = rate_limit_config.get("enabled", False)
        self.min_gap = rate_limit_config.get("min_gap", 5)

        # Load trigger_on_success_only configuration
        self.trigger_on_success_only = self.config.get("trigger_on_success_only", False)

    def _initialize_rules(self, rule_set: Optional[Set[str]] = None):
        """
        Initialize rules from configuration.

        Only rules present in the config file will be initialized.
        This allows config-driven rule selection (e.g., lang_only or graph_only).
        """
        # Get the rules section from config (supports both old and new structure)
        rules_config = self.config.get("rules", self.config)

        # Phase transition rules - only initialize if present in config
        transitions = rules_config.get("phase_transitions", {})
        if "transition_L_P" in transitions:
            self.rules["transition_L_P"] = PhaseTransitionRule(
                "transition_L_P",
                transitions["transition_L_P"]["message"],
                "L", "P",
                threshold=transitions["transition_L_P"]["threshold"]
            )
        if "transition_P_V" in transitions:
            self.rules["transition_P_V"] = PhaseTransitionRule(
                "transition_P_V",
                transitions["transition_P_V"]["message"],
                "P", "V",
                threshold=transitions["transition_P_V"]["threshold"]
            )

        # Strategy shift rules - only initialize if present in config
        shifts = rules_config.get("strategy_shifts", {})
        if "shift_L_V" in shifts:
            self.rules["shift_L_V"] = StrategyShiftRule(
                "shift_L_V",
                shifts["shift_L_V"]["message"],
                "L", "V", "shortcut",
                threshold=shifts["shift_L_V"]["threshold"]
            )
        if "shift_V_L" in shifts:
            self.rules["shift_V_L"] = StrategyShiftRule(
                "shift_V_L",
                shifts["shift_V_L"]["message"],
                "V", "L", "rework",
                threshold=shifts["shift_V_L"]["threshold"]
            )
        if "shift_V_P" in shifts:
            self.rules["shift_V_P"] = StrategyShiftRule(
                "shift_V_P",
                shifts["shift_V_P"]["message"],
                "V", "P", "backtrack",
                threshold=shifts["shift_V_P"]["threshold"]
            )
        if "shift_P_L" in shifts:
            self.rules["shift_P_L"] = StrategyShiftRule(
                "shift_P_L",
                shifts["shift_P_L"]["message"],
                "P", "L", "backtrack",
                threshold=shifts["shift_P_L"]["threshold"]
            )

        # Dwell time rules - only initialize if present in config
        dwells = rules_config.get("dwell_times", {})
        dwell_block_execution = dwells.get("block_execution", False)
        if "dwell_L_navigate" in dwells:
            self.rules["dwell_L_navigate"] = DwellTimeRule(
                "dwell_L_navigate",
                dwells["dwell_L_navigate"]["message"],
                dwells["dwell_L_navigate"]["threshold"],
                phase_exact="L_navigate",
                block_execution=dwell_block_execution
            )
        if "dwell_L_reproduce" in dwells:
            self.rules["dwell_L_reproduce"] = DwellTimeRule(
                "dwell_L_reproduce",
                dwells["dwell_L_reproduce"]["message"],
                dwells["dwell_L_reproduce"]["threshold"],
                phase_exact="L_reproduce",
                block_execution=dwell_block_execution
            )
        if "dwell_P" in dwells:
            self.rules["dwell_P"] = DwellTimeRule(
                "dwell_P",
                dwells["dwell_P"]["message"],
                dwells["dwell_P"]["threshold"],
                phase_category="P",
                block_execution=dwell_block_execution
            )
        if "dwell_V" in dwells:
            self.rules["dwell_V"] = DwellTimeRule(
                "dwell_V",
                dwells["dwell_V"]["message"],
                dwells["dwell_V"]["threshold"],
                phase_category="V",
                block_execution=dwell_block_execution
            )

        # Oscillation rule (graphectory-based) - only initialize if present in config
        oscillations = rules_config.get("oscillations", {})
        if oscillations:
            osc_block_execution = oscillations.get("block_execution", False)
            self.rules["oscillation"] = OscillationRule(oscillations, block_execution=osc_block_execution)

        # Plan compliance rule - only initialize if present in config
        plan_compliance = rules_config.get("plan_compliance", {})
        if plan_compliance:
            intended_plan = plan_compliance.get("intended_plan", [])
            ending_flags = plan_compliance.get("ending_flags", [])
            phase_names = plan_compliance.get("phase_names", {})
            message = plan_compliance.get("message",
                "Plan compliance violation: {violation}. Expected to {expected}. Skipped phases: {skipped}")
            compliance_block_execution = plan_compliance.get("block_execution", False)
            threshold = plan_compliance.get("threshold", 3)

            if intended_plan:
                self.rules["plan_compliance"] = PlanComplianceRule(
                    rule_id="plan_compliance",
                    message=message,
                    intended_plan=intended_plan,
                    ending_flags=ending_flags,
                    phase_names=phase_names,
                    block_execution=compliance_block_execution,
                    threshold=threshold
                )

        # Apply rule selection if specified
        if rule_set is not None:
            for rule_id, rule in self.rules.items():
                rule.enabled = rule_id in rule_set

    def evaluate(self, current_phase: Phase = None, previous_phase: Phase = None,
                 graph = None, step_index: int = None, command: str = None, outcome: str = None, **kwargs) -> list[RuleMatch]:
        """
        Evaluate all enabled rules for the current state.

        Args:
            current_phase: Current phase (for languatory rules)
            previous_phase: Previous phase (for transition rules)
            graph: Graph trajectory (for graphectory rules)
            step_index: Current step index (for rate limiting)
            command: Raw command string (for plan compliance ending flags)
            outcome: Outcome of the current action ("success", "failure", or None)
            **kwargs: Additional context

        Returns:
            List of triggered rule matches
        """
        # Check if trigger_on_success_only is enabled and outcome is not "success"
        if self.trigger_on_success_only and outcome != "success":
            return []

        # Check rate limit: if rate-limited, still update rule states but don't return matches
        is_rate_limited = False
        if self.rate_limit_enabled and step_index is not None:
            is_rate_limited = self._is_rate_limited(step_index)

        matches = []

        # Evaluate all enabled rules
        # Always call rules to update their internal state, even when rate-limited
        for rule in self.rules.values():
            if not rule.enabled:
                continue

            match = rule.check(
                current_phase=current_phase,
                previous_phase=previous_phase,
                from_phase=previous_phase,
                to_phase=current_phase,
                graph=graph,
                step_index=step_index,
                command=command,
                **kwargs
            )
            # Only add matches if not rate-limited (suppresses message output while maintaining state)
            if match and not is_rate_limited:
                matches.append(match)

        # Record this step if any rules triggered (and not rate-limited)
        if matches and step_index is not None:
            self._record_trigger(step_index)

        return matches

    def _is_rate_limited(self, step_index: int) -> bool:
        """
        Check if rate limiting should block rule evaluation at this step.

        Args:
            step_index: Current step index

        Returns:
            True if rate limited (should skip evaluation), False otherwise
        """
        if not self.trigger_history:
            return False

        # Check if any trigger occurred in the previous min_gap steps
        recent_triggers = [idx for idx in self.trigger_history if step_index - self.min_gap < idx < step_index]
        return len(recent_triggers) > 0

    def _record_trigger(self, step_index: int):
        """
        Record that a rule triggered at this step.

        Args:
            step_index: Step index where rule triggered
        """
        self.trigger_history.append(step_index)
        # Keep only recent history to avoid unbounded growth
        # Keep last 50 triggers to support min_gap up to 50
        if len(self.trigger_history) > 50:
            self.trigger_history = self.trigger_history[-50:]

    def enable_rule(self, rule_id: str):
        """Enable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = True

    def disable_rule(self, rule_id: str):
        """Disable a specific rule."""
        if rule_id in self.rules:
            self.rules[rule_id].enabled = False

    def enable_rules(self, rule_ids: Set[str]):
        """Enable only the specified rules, disable all others."""
        for rule_id, rule in self.rules.items():
            rule.enabled = rule_id in rule_ids

    def get_enabled_rules(self) -> Set[str]:
        """Get set of currently enabled rule IDs."""
        return {rule_id for rule_id, rule in self.rules.items() if rule.enabled}

    def reset(self):
        """Reset all rule states."""
        for rule in self.rules.values():
            rule.reset()
