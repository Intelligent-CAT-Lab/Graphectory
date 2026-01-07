from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional


class Phase(str):
    """
    Generic phase representation with hierarchical structure.

    Phase naming convention uses underscore-separated prefixes:
    - L_* : Localization (e.g., L_reproduce, L_navigate)
    - P   : Patch (e.g., P, P_refactor - though P alone is most common)
    - V_* : Validation (e.g., V_newly_generated_test, V_regression_test)
    - *   : General fallback (e.g., general)

    The design supports:
    1. Coarse-grained grouping by prefix (L, P, V)
    2. Fine-grained differentiation within groups (e.g., V_newly_generated vs V_regression)
    3. Extension to new phase names without code changes
    """

    def __init__(self, value: str):
        self._value = value
        # Parse prefix and suffix for efficient categorization
        # Check for single-letter prefix followed by underscore
        if value.startswith('L_'):
            self._prefix = 'L'
            self._suffix = value[2:]
        elif value.startswith('V_'):
            self._prefix = 'V'
            self._suffix = value[2:]
        elif value.startswith('P_'):
            self._prefix = 'P'
            self._suffix = value[2:]
        elif value == 'P':
            self._prefix = 'P'
            self._suffix = ''
        else:
            self._prefix = ''
            self._suffix = value

    def __str__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"Phase({self._value!r})"

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Phase):
            return self._value == other._value
        if isinstance(other, str):
            return self._value == other
        return False

    def __hash__(self) -> int:
        return hash(self._value)

    @property
    def value(self) -> str:
        """The full phase name."""
        return self._value

    @property
    def prefix(self) -> str:
        """The phase category prefix (L, P, V, or empty)."""
        return self._prefix

    @property
    def suffix(self) -> str:
        """The phase-specific suffix (e.g., 'reproduce', 'newly_generated_test')."""
        return self._suffix

    def is_localization(self) -> bool:
        """Check if this is a localization phase (L_*)."""
        return self._prefix == 'L'

    def is_validation(self) -> bool:
        """Check if this is a validation phase (V_*)."""
        return self._prefix == 'V'

    def is_patch(self) -> bool:
        """Check if this is a patch phase (P or P_*)."""
        return self._prefix == 'P'

    def in_same_group(self, other: "Phase") -> bool:
        """Check if two phases belong to the same category (same prefix)."""
        return self._prefix == other._prefix and self._prefix != ''


@dataclass
class ActionEvent:
    """
    Minimal info about a single agent step.
    This is all the monitor needs; it does NOT depend on minisweagent types.
    """
    step_index: int
    command: str                                    # the bash command string
    cwd: Optional[str] = None                       # optional: current working dir
    last_output: Optional[dict[str, str]] = None    # env.execute result from previous step
    extra: dict[str, Any] = field(default_factory=dict)  # free-form metadata


@dataclass
class MonitorResult:
    """
    What the monitor returns on each step.

    Utility data structure containing phase transition information and rule matches.
    Messages are generated only by the rule engine, not by this class.

    Attributes:
        current_phase: Current phase
        phase_changed: True when phase string changes (exact match)
        category_changed: True when phase category changes (L -> P -> V)
        previous_phase: Previous phase (None if first phase)
        rule_matches: List of RuleMatch objects from the rule engine
    """
    current_phase: Phase
    phase_changed: bool
    category_changed: bool = False
    previous_phase: Optional[Phase] = None
    rule_matches: list = field(default_factory=list)

    def get_all_messages(self) -> list[str]:
        """
        Get all messages from triggered rules, combined into a single message.

        If all messages are identical, returns one message.
        Otherwise, concatenates them sequentially.

        Returns:
            List containing a single combined message, or empty list if no messages
        """
        messages = []
        for match in self.rule_matches:
            if hasattr(match, 'message') and match.message:
                messages.append(match.message)

        if not messages:
            return []

        # Remove duplicates while preserving order
        unique_messages = []
        seen = set()
        for msg in messages:
            if msg not in seen:
                seen.add(msg)
                unique_messages.append(msg)

        # If only one unique message, return it as-is
        if len(unique_messages) == 1:
            return [unique_messages[0]]

        # Concatenate multiple unique messages with newlines
        combined = "\n".join(unique_messages)
        return [combined]

    def should_block_and_refine(self) -> bool:
        """
        Check if any triggered rules require BLOCKING execution and calling refiner.

        This is for rules (plan_compliance, dwell_times, oscillations) with block_execution=True
        that should prevent the current action from being executed or added to history.

        Other rules (phase_transitions, strategy_shifts) still trigger refiner but allow
        execution to proceed normally (post-hoc refinement).

        Returns:
            True if any rule_match has block_execution=True, False otherwise
        """
        for match in self.rule_matches:
            if hasattr(match, 'block_execution') and match.block_execution:
                return True
        return False
