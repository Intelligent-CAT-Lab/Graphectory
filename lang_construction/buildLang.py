from __future__ import annotations
from typing import List, Tuple, Optional
from lang_construction.mapLang import get_action_role

def build_lang_sequence_rle(step_nodes: List[Tuple[int, dict]]) -> Tuple[List[str], List[int]]:
    """
    Build run-length encoded role sequence from extracted node sequence.

    Args:
        step_nodes: List of (step_index, node) tuples from extract_node_sequence

    Returns:
        roles: run-collapsed roles e.g. ['L_navigate','L_reproduce', 'P', 'V_newly_generated_test', 'V_regression'...]
        lens:   streak length per run              [  3,  2,  3, 1, 2 ...]
    """
    roles: List[str] = []
    lens: List[int] = []
    prev: Optional[str] = None
    created_tests = set()

    for _, node in step_nodes:
        tool = node.get('tool')
        command = node.get('command')
        subcommand = node.get('subcommand')
        args = node.get('args')
        role = get_action_role(tool, subcommand, command, args,
                               prev_roles=roles, created_tests=created_tests)

        # Skip general or empty
        if not role or role == 'general':
            continue

        # Run-length encoding
        if role == prev:
            lens[-1] += 1
        else:
            roles.append(role)
            lens.append(1)
            prev = role

    return roles, lens


def build_lang_sequence(step_nodes: List[Tuple[int, dict]]) -> List[str]:
    """
    Build full role sequence from extracted node sequence (no run-length encoding).

    Args:
        step_nodes: List of (step_index, node) tuples from extract_node_sequence

    Returns:
        List of role for each step
    """
    roles: List[str] = []
    created_tests = set()

    for _, node in step_nodes:
        tool = node.get('tool')
        command = node.get('command')
        subcommand = node.get('subcommand')
        args = node.get('args')
        role = get_action_role(tool, subcommand, command, args,
                               prev_roles=roles, created_tests=created_tests)
        # Skip general role or empty
        if not role or role == 'general':
            continue

        roles.append(role)

    return roles
