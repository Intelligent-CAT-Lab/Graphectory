from __future__ import annotations
from typing import List, Tuple, Optional

# Phase abbreviations
PHASE_ABBR = {
    'localization': 'L',
    'patch': 'P',
    'validation': 'V',
}


def build_phase_sequence_rle(step_nodes: List[Tuple[int, dict]]) -> Tuple[List[str], List[int]]:
    """
    Build run-length encoded phase sequence from extracted node sequence.

    Args:
        step_nodes: List of (step_index, node) tuples from extract_node_sequence

    Returns:
        phases: run-collapsed phases (L/P/V) e.g. ['L','P','V',...]
        lens:   streak length per run              [  3,  2,  3, ...]
    """
    phases: List[str] = []
    lens: List[int] = []
    prev: Optional[str] = None

    for _, node in step_nodes:
        phase = node.get('phase')

        # Skip general phase or empty
        if not phase or phase == 'general':
            continue

        # Get abbreviation
        abbr = PHASE_ABBR.get(str(phase).lower())
        if not abbr:
            continue

        # Run-length encoding
        if abbr == prev:
            lens[-1] += 1
        else:
            phases.append(abbr)
            lens.append(1)
            prev = abbr

    return phases, lens


def build_phase_sequence(step_nodes: List[Tuple[int, dict]]) -> List[str]:
    """
    Build full phase sequence from extracted node sequence (no run-length encoding).

    Args:
        step_nodes: List of (step_index, node) tuples from extract_node_sequence

    Returns:
        List of phase abbreviations (L/P/V) for each step
    """
    phases: List[str] = []

    for _, node in step_nodes:
        phase = node.get('phase')

        # Skip general phase or empty
        if not phase or phase == 'general':
            continue

        # Get abbreviation
        abbr = PHASE_ABBR.get(str(phase).lower())
        if abbr:
            phases.append(abbr)

    return phases
