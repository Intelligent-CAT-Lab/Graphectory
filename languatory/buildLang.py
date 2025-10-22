from __future__ import annotations
from typing import List, Tuple, Optional

ROLE_ABBR = {
    'L_reproduce': 'L_reproduce',
    'L_navigate': 'L_navigate',
    'patch': 'P',
    'V_newly_generated_test': 'V_newly_generated_test', 
    'V_regression_test': 'V_regression_test',
}

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

    for _, node in step_nodes:
        # role = node.get('phase') # to be replaced with get_role function

        # Skip general or empty
        if not role or role == 'general':
            continue

        # Get abbreviation
        abbr = ROLE_ABBR.get(str(role).lower())
        if not abbr:
            continue

        # Run-length encoding
        if abbr == prev:
            lens[-1] += 1
        else:
            roles.append(abbr)
            lens.append(1)
            prev = abbr

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

    for _, node in step_nodes:
        # role = node.get('phase') # to be replaced with get_role function

        # Skip general role or empty
        if not role or role == 'general':
            continue

        # Get abbreviation
        abbr = ROLE_ABBR.get(str(role).lower())
        if abbr:
            roles.append(abbr)

    return roles
