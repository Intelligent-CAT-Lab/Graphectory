from __future__ import annotations
from typing import List, Tuple, Iterable


def _iter_nodes(graph_json: dict) -> Iterable[dict]:
    """Iterate over nodes in the graph JSON structure."""
    if isinstance(graph_json, dict):
        if "nodes" in graph_json and isinstance(graph_json["nodes"], list):
            yield from graph_json["nodes"]
        elif "graph" in graph_json and isinstance(graph_json["graph"], dict) and "nodes" in graph_json["graph"]:
            yield from graph_json["graph"]["nodes"]


def extract_node_sequence(graph_json: dict) -> List[Tuple[int, dict]]:
    """
    Extract a flattened list of nodes sorted by step_indices.
    Each node appears once for each step_index it belongs to, with the corresponding phase.
    Returns clean nodes with only: label, args, flags, phase, step_index.

    Args:
        graph_json: The graph JSON structure

    Returns:
        List of (step_index, clean_node) tuples sorted by step_index
    """
    step_nodes = []

    for node in _iter_nodes(graph_json):
        step_indices = node.get("step_indices") or []
        phases = node.get("phases") or node.get("phase")

        # Extract base fields
        tool = node.get("tool")
        command = node.get("command")
        subcommand = node.get("subcommand")
        args = node.get("args")
        flags = node.get("flags")

        # Handle matching step_indices with phases
        if isinstance(phases, list) and len(phases) == len(step_indices):
            # Each step_index has a corresponding phase
            for step_idx, phase in zip(step_indices, phases):
                clean_node = {
                    "tool": tool,
                    "command": command,
                    "subcommand": subcommand,
                    "args": args,
                    "flags": flags,
                    "phase": phase,
                    "step_index": step_idx
                }
                step_nodes.append((step_idx, clean_node))
        else:
            # Same phase for all step_indices
            for step_idx in step_indices:
                clean_node = {
                    "tool": tool,
                    "command": command,
                    "subcommand": subcommand,
                    "args": args,
                    "flags": flags,
                    "phase": phases,
                    "step_index": step_idx
                }
                step_nodes.append((step_idx, clean_node))

    # Sort by step_index (chronological order)
    step_nodes.sort(key=lambda x: x[0])
    return step_nodes
