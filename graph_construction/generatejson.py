#!/usr/bin/env python3
"""
Graph Generation Script for Agent Trajectories (JSON only)

This script generates trajectory graphs (JSON) from agent execution traces.
Supports SWE-agent and OpenHands trajectories across multiple models.

Usage:
    python generatejson.py --agent sa --model dsk-v3 --trajs path_to_your_trajectory_folder --eval_report path_to_your_report.json --output_dir data/samples
    python generatejson.py --agent oh --model dsk-v3 --trajs path_to_your_output.jsonl --eval_report path_to_your_report.json --output_dir data/samples

Output Structure:
    {output_dir}/SWE-agent/graphs/{model}/{instance_id}/{instance_id}.json
    {output_dir}/OpenHands/graphs/{model}/{instance_id}/{instance_id}.json
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import networkx as nx
from datasets import load_dataset
from networkx.readwrite import json_graph

from commandParser import CommandParser
from mapPhase import get_phase


# ==================== Configuration ====================
SUPPORTED_AGENTS = {"sa", "oh"}
SUPPORTED_MODELS = {"dsk-v3", "dsk-r1", "dev", "cld-4"}

MODEL_NAMES = {
    "dsk-v3": "deepseek-v3",
    "dsk-r1": "deepseek-r1-0528",
    "dev": "devstral-small",
    "cld-4": "claude-sonnet-4",
}

AGENT_NAMES = {
    "sa": "SWE-agent",
    "oh": "OpenHands",
}

# -------------------- Data lookups --------------------
swe_bench_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
difficulty_lookup = {row["instance_id"]: row["difficulty"] for row in swe_bench_ds}


# ==================== Data Classes ====================
@dataclass
class ProcessingResult:
    """Result of processing a single trajectory."""
    instance_id: str
    status: str           # "success" or "error"
    json_path: Optional[str] = None
    error: Optional[str] = None


# ==================== Helpers ====================
def hash_node_signature(label, args, flags):
    normalized = json.dumps({"label": label, "args": args, "flags": flags}, sort_keys=True)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def check_edit_status(tool, subcommand, args, observation):
    def check_str_edit_status(obs):
        if not obs:
            return None
        if "has been edited." in obs:
            return "success"
        if "did not appear verbatim" in obs:
            return "failure: not found"
        if "Multiple occurrences of old_str" in obs:
            return "failure: multiple occurrences"
        if "old_str" in obs and "is the same as new_str" in obs:
            return "failure: no change"
        return "failure: unknown"

    if tool == "str_replace_editor" and subcommand in {"str_replace"}:
        return check_str_edit_status(observation)
    return None


def determine_resolution_status(instance_id: str, eval_report_path: str) -> str:
    """Determine resolution status from eval report given an instance ID."""
    if not os.path.isfile(eval_report_path):
        return "N/A"
    with open(eval_report_path, "r") as f:
        report = json.load(f)
    if instance_id in report.get("resolved_ids", []):
        return "resolved"
    elif instance_id in report.get("unresolved_ids", []):
        return "unresolved"
    return "unsubmitted"


# ==================== Graph Builder ====================
class GraphBuilder:
    """Utility class for managing graph construction operations."""

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.node_signature_to_key = {}
        self.localization_nodes = []
        self.prev_phases = set()
        self.previous_node = None

    def add_or_update_node(self, node_label, args, flags, phase, step_idx,
                           tool=None, command=None, subcommand=None):
        node_signature = hash_node_signature(node_label, args, flags)

        if node_signature in self.node_signature_to_key:
            node_key = self.node_signature_to_key[node_signature]
            self.G.nodes[node_key]["step_indices"].append(step_idx)
            if "phases" not in self.G.nodes[node_key]:
                self.G.nodes[node_key]["phases"] = []
            self.G.nodes[node_key]["phases"].append(phase)
        else:
            node_key = f"{len(self.G.nodes)}:{node_label}"
            self.G.add_node(
                node_key,
                label=node_label,
                args=args,
                flags=flags,
                phases=[phase],
                step_indices=[step_idx],
                tool=tool,
                command=command,
                subcommand=subcommand,
            )
            self.node_signature_to_key[node_signature] = node_key

            if tool == "str_replace_editor" and subcommand == "view":
                self.localization_nodes.append(node_key)

        return node_key

    def add_execution_edge(self, node_key, step_idx):
        if self.previous_node:
            self.G.add_edge(self.previous_node, node_key, label=str(step_idx), type="exec")

    def update_previous_node(self, node_key):
        self.previous_node = node_key

    def add_phase(self, phase):
        self.prev_phases.add(phase)

    def finalize_and_save(self, output_dir, instance_id, eval_report_path):
        """Build hierarchical edges, add metadata, and save JSON graph.

        Returns:
            json_path: path to the saved JSON file
        """
        build_hierarchical_edges(self.G, self.localization_nodes)

        resolution_status = determine_resolution_status(instance_id, eval_report_path)
        self.G.graph["resolution_status"] = resolution_status
        self.G.graph["instance_name"] = instance_id
        self.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")

        instance_dir = os.path.join(output_dir, instance_id)
        os.makedirs(instance_dir, exist_ok=True)

        json_path = os.path.join(instance_dir, f"{instance_id}.json")

        with open(json_path, "w") as f:
            json.dump(json_graph.node_link_data(self.G, edges="edges"), f, indent=2)

        return json_path


# ==================== Hierarchical Edges ====================
def build_hierarchical_edges(G: nx.MultiDiGraph, localization_nodes):
    """Add 'hier' edges with transitive reduction: only immediate parent relationships.

    Avoids redundancy: if A→B and B→C exist, A→C is not added.
    """
    path_nodes = []
    range_nodes_by_path = defaultdict(list)

    for node in localization_nodes:
        data = G.nodes[node]
        path = data.get("args", {}).get("path")
        view_range = data.get("args", {}).get("view_range")

        if path:
            path_obj = Path(path)
            if view_range is None:
                path_nodes.append((node, path_obj))
            elif (
                isinstance(view_range, (list, tuple))
                and len(view_range) == 2
                and all(isinstance(x, int) for x in view_range)
            ):
                range_nodes_by_path[str(path_obj)].append((node, view_range))
            else:
                print(f"[WARN] Skipping invalid view_range for node {node}: {view_range}")

    # Path hierarchy: connect only to closest parent
    for child_node, child_path in path_nodes:
        best_parent_node = None
        best_parent_path = None
        for parent_node, parent_path in path_nodes:
            if parent_node == child_node:
                continue
            if len(parent_path.parts) < len(child_path.parts) and \
                    child_path.parts[:len(parent_path.parts)] == parent_path.parts:
                if best_parent_path is None or len(parent_path.parts) > len(best_parent_path.parts):
                    best_parent_node = parent_node
                    best_parent_path = parent_path
        if best_parent_node:
            G.add_edge(best_parent_node, child_node, type="hier")

    path_to_node = {str(p): n for n, p in path_nodes}

    for path_str, range_nodes in range_nodes_by_path.items():
        is_nested = {n: False for n, _ in range_nodes}

        # Range nesting: connect only immediate outer→inner, not all ancestors
        for i, (node_i, r_i) in enumerate(range_nodes):
            for j, (node_j, r_j) in enumerate(range_nodes):
                if i == j:
                    continue
                try:
                    a1, a2 = r_i
                    b1, b2 = r_j
                    # node_j nested inside node_i
                    if b1 >= a1 and b2 <= a2:
                        # Check if immediate (no intermediate range between i and j)
                        is_immediate = True
                        for k, (node_k, r_k) in enumerate(range_nodes):
                            if k == i or k == j:
                                continue
                            c1, c2 = r_k
                            # node_k is between node_i and node_j if both conditions hold
                            if (c1 >= a1 and c2 <= a2 and b1 >= c1 and b2 <= c2):
                                is_immediate = False
                                break
                        if is_immediate:
                            G.add_edge(node_i, node_j, type="hier")
                            is_nested[node_j] = True
                except Exception as e:
                    print(f"[WARN] Failed to unpack ranges for nesting check: {r_i}, {r_j} ({e})")

        # Link outermost ranges to path node or closest ancestor
        path_node = path_to_node.get(path_str)
        if path_node:
            for node, _ in range_nodes:
                if not is_nested[node]:
                    G.add_edge(path_node, node, type="hier")
        else:
            path_parts = Path(path_str).parts
            best_ancestor_node = None
            best_ancestor_depth = -1
            for pn, pp in path_nodes:
                if len(pp.parts) < len(path_parts) and path_parts[:len(pp.parts)] == pp.parts:
                    if len(pp.parts) > best_ancestor_depth:
                        best_ancestor_node = pn
                        best_ancestor_depth = len(pp.parts)
            for node, _ in range_nodes:
                if not is_nested[node] and best_ancestor_node:
                    G.add_edge(best_ancestor_node, node, type="hier")


# ==================== Graph Build Functions ====================
def build_graph_from_sa_trajectory(traj_data, parser: CommandParser, instance_id, output_dir, eval_report_path):
    """Build graph from SWE-agent trajectory data and save JSON.

    Args:
        traj_data: SWE-agent trajectory dictionary containing 'trajectory' key
        parser: CommandParser instance for parsing action strings
        instance_id: Instance identifier (e.g., 'django__django-12345')
        output_dir: Base output directory for saving graphs
        eval_report_path: Path to evaluation report JSON file

    Returns:
        json_path: path to the saved JSON file

    Output Structure:
        {output_dir}/{instance_id}/{instance_id}.json
    """
    builder = GraphBuilder()
    trajectory = traj_data.get("trajectory", [])

    for step_idx, step in enumerate(trajectory):
        action_str = step.get("action", "")

        if action_str.strip() == "":
            node_key = builder.add_or_update_node(
                node_label="think", args={}, flags={}, phase="general",
                step_idx=step_idx, tool=None, command=None, subcommand=None,
            )
            builder.add_execution_edge(node_key, step_idx)
            builder.update_previous_node(node_key)
            builder.add_phase("general")
            continue

        parsed_commands = parser.parse(action_str)
        if not parsed_commands:
            continue

        for parsed in parsed_commands:
            tool = parsed.get("tool", "").strip() if parsed.get("tool") else ""
            subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
            command = parsed.get("command", "").strip() if parsed.get("command") else ""
            args = parsed.get("args", {})
            flags = parsed.get("flags", {})

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, builder.prev_phases)

            edit_status = check_edit_status(tool, subcommand, args, step.get("observation", ""))
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_key = builder.add_or_update_node(
                node_label=node_label, args=args, flags=flags, phase=phase,
                step_idx=step_idx, tool=tool, command=command, subcommand=subcommand,
            )
            builder.add_execution_edge(node_key, step_idx)
            builder.update_previous_node(node_key)
            builder.add_phase(phase)

    return builder.finalize_and_save(output_dir, instance_id, eval_report_path)


def build_graph_from_oh_trajectory(traj_data, parser: CommandParser, instance_id, output_dir, eval_report_path):
    """Build graph from OpenHands trajectory data and save JSON.

    Args:
        traj_data: OpenHands trajectory dictionary containing 'history' key
        parser: CommandParser instance for parsing action strings
        instance_id: Instance identifier (e.g., 'django__django-12345')
        output_dir: Base output directory for saving graphs
        eval_report_path: Path to evaluation report JSON file

    Returns:
        json_path: path to the saved JSON file

    Output Structure:
        {output_dir}/{instance_id}/{instance_id}.json
    """
    builder = GraphBuilder()
    step_idx = 0

    for step in traj_data.get("history", []):
        action = step.get("observation") if step.get("observation") else None
        if action in ("system", "message") or action is None:
            continue

        action_str = action or ""

        tool_calls = step.get("tool_call_metadata", {}).get("model_response", {}).get("choices", [])
        if not tool_calls and "tool_call_metadata" in step:
            tool_calls = [step["tool_call_metadata"]]

        parsed_commands = []
        for call in tool_calls:
            function_call = None
            if isinstance(call, dict):
                if "function" in call:
                    function_call = call["function"]
                elif "message" in call and "tool_calls" in call["message"]:
                    for tc in call["message"]["tool_calls"]:
                        if "function" in tc:
                            function_call = tc["function"]

            if not function_call:
                continue

            tool_name = function_call.get("name")
            args_raw = function_call.get("arguments", "{}")

            try:
                args_loaded = json.loads(args_raw)
            except json.JSONDecodeError:
                args_loaded = {}

            if tool_name == "execute_bash":
                cmd = args_loaded.get("command", "").strip()
                parsed_commands = parser.parse(cmd)
                if not parsed_commands:
                    continue
            else:
                subcommand = args_loaded.pop("command", None)
                parsed_commands = [{"tool": tool_name, "subcommand": subcommand, "args": args_loaded}]

        if not parsed_commands:
            continue

        for parsed in parsed_commands:
            tool = parsed.get("tool", "").strip()

            if tool == "think":
                node_key = builder.add_or_update_node(
                    node_label="think", args={}, flags={}, phase="general",
                    step_idx=step_idx, tool=None, command=None, subcommand=None,
                )
                builder.add_execution_edge(node_key, step_idx)
                builder.update_previous_node(node_key)
                builder.add_phase("general")
                continue

            subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
            command = parsed.get("command", "").strip() if parsed.get("command") else ""
            args = parsed.get("args", {})
            flags = parsed.get("flags", {})

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, builder.prev_phases)

            edit_status = check_edit_status(tool, subcommand, args, step.get("content", ""))
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_key = builder.add_or_update_node(
                node_label=node_label, args=args, flags=flags, phase=phase,
                step_idx=step_idx, tool=tool, command=command, subcommand=subcommand,
            )
            builder.add_execution_edge(node_key, step_idx)
            builder.update_previous_node(node_key)
            builder.add_phase(phase)

        step_idx += 1

    return builder.finalize_and_save(output_dir, instance_id, eval_report_path)


# ==================== Path Management ====================
def get_graph_output_dir(base_output_dir: str, agent: str, model: str) -> Path:
    """Construct the graph output directory path."""
    return Path(base_output_dir) / AGENT_NAMES[agent] / "graphs" / MODEL_NAMES[model]


# ==================== Trajectory Loaders ====================
class TrajectoryLoader:
    """Loaders for SWE-agent and OpenHands trajectories."""

    @staticmethod
    def load_sa_trajectories(trajs_path: Path) -> List[Dict[str, Any]]:
        """Load SWE-agent trajectories from directory structure.

        Directory structure:
            trajs_path/
                ├── instance-1/
                │   ├── instance-1.traj
                │   └── ...
                └── ...
        """
        trajectories = []
        if not trajs_path.is_dir():
            raise ValueError(f"SA trajectories path must be a directory: {trajs_path}")

        for instance_dir in sorted(trajs_path.iterdir()):
            if not instance_dir.is_dir():
                continue
            instance_id = instance_dir.name
            traj_file = instance_dir / f"{instance_id}.traj"

            if not traj_file.exists():
                print(f"[WARN] Missing .traj file for {instance_id}, skipping")
                continue

            try:
                with open(traj_file, "r") as f:
                    traj_data = json.load(f)
                trajectories.append({"instance_id": instance_id, "traj_data": traj_data})
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse {traj_file}: {e}")

        return trajectories

    @staticmethod
    def load_oh_trajectories(trajs_path: Path) -> List[Dict[str, Any]]:
        """Load OpenHands trajectories from output.jsonl file."""
        trajectories = []
        if not trajs_path.is_file():
            raise ValueError(f"OH trajectories path must be a file: {trajs_path}")

        with open(trajs_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    traj_data = json.loads(line)
                    instance_id = traj_data.get("instance_id")
                    if not instance_id:
                        print(f"[WARN] Line {line_num}: Missing instance_id, skipping")
                        continue
                    trajectories.append({"instance_id": instance_id, "traj_data": traj_data})
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_num}: Failed to parse JSON: {e}")

        return trajectories


# ==================== Graph Processor ====================
class GraphProcessor:
    """Process trajectories and generate JSON graphs."""

    def __init__(self, agent: str, parser: CommandParser, eval_report_path: str, output_dir: Path):
        self.agent = agent
        self.parser = parser
        self.eval_report_path = eval_report_path
        self.output_dir = output_dir

    def process_trajectory(self, instance_id: str, traj_data: Dict[str, Any]) -> ProcessingResult:
        """Process a single trajectory and generate JSON graph."""
        try:
            if self.agent == "sa":
                json_path = build_graph_from_sa_trajectory(
                    traj_data=traj_data,
                    parser=self.parser,
                    instance_id=instance_id,
                    output_dir=str(self.output_dir),
                    eval_report_path=self.eval_report_path,
                )
            elif self.agent == "oh":
                json_path = build_graph_from_oh_trajectory(
                    traj_data=traj_data,
                    parser=self.parser,
                    instance_id=instance_id,
                    output_dir=str(self.output_dir),
                    eval_report_path=self.eval_report_path,
                )
            else:
                raise ValueError(f"Unsupported agent: {self.agent}")

            return ProcessingResult(instance_id=instance_id, status="success", json_path=json_path)

        except Exception as e:
            return ProcessingResult(instance_id=instance_id, status="error", error=str(e))


# ==================== Batch Processing ====================
def setup_parser_for_agent(agent: str) -> CommandParser:
    """Setup CommandParser with appropriate tool configurations."""
    parser = CommandParser()
    tool_configs = []
    if agent == "sa":
        tool_configs = [
            "data/SWE-agent/tools/edit_anthropic/config.yaml",
            "data/SWE-agent/tools/review_on_submit_m/config.yaml",
            "data/SWE-agent/tools/registry/config.yaml",
        ]
    if tool_configs:
        parser.load_tool_yaml_files(tool_configs)
    return parser


def process_batch(
    trajectories: List[Dict[str, Any]],
    processor: GraphProcessor,
    max_workers: int = 8,
) -> Dict[str, List]:
    """Process trajectories in parallel."""
    results: Dict[str, List] = {"success": [], "failed": []}
    total = len(trajectories)

    print(f"\n{'='*70}")
    print(f"Processing {total} trajectories with {max_workers} workers...")
    print(f"{'='*70}\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance = {
            executor.submit(processor.process_trajectory, traj["instance_id"], traj["traj_data"]): traj["instance_id"]
            for traj in trajectories
        }
        completed = 0
        for future in as_completed(future_to_instance):
            result = future.result()
            completed += 1
            if result.status == "success":
                results["success"].append(result)
                print(f"[{completed}/{total}] ✓ {result.instance_id}")
            else:
                results["failed"].append(result)
                print(f"[{completed}/{total}] ✗ {result.instance_id}: {result.error}")

    return results


def print_summary(results: Dict[str, List], agent: str, model: str, output_dir: Path):
    """Print processing summary."""
    success_count = len(results["success"])
    failed_count = len(results["failed"])
    total = success_count + failed_count

    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Agent:        {AGENT_NAMES[agent]}")
    print(f"Model:        {MODEL_NAMES[model]}")
    print(f"Output:       {output_dir}")
    print(f"Succeeded:    {success_count}/{total}")
    print(f"{'='*70}\n")

    if failed_count > 0:
        print("Failed instances:")
        for result in results["failed"][:10]:
            print(f"  - {result.instance_id}: {result.error}")
        if failed_count > 10:
            print(f"  ... and {failed_count - 10} more")
        print()


# ==================== Entry Point ====================
def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectory graphs (JSON only) for agent executions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SWE-agent with DeepSeek-V3
  python %(prog)s --agent sa --model dsk-v3 --trajs sa_trajectories --eval_report report.json --output_dir output

  # OpenHands with Claude Sonnet 4
  python %(prog)s --agent oh --model cld-4 --trajs output.jsonl --eval_report report.json --output_dir output

Output Structure:
  {output_dir}/SWE-agent/graphs/deepseek-v3/{instance_id}/{instance_id}.json
  {output_dir}/OpenHands/graphs/claude-sonnet-4/{instance_id}/{instance_id}.json

Supported agents: sa (SWE-agent), oh (OpenHands)
Supported models: dsk-v3 (deepseek-v3), dsk-r1 (deepseek-r1-0528), dev (devstral-small), cld-4 (claude-sonnet-4)
        """,
    )

    parser.add_argument("--agent", type=str, required=True, choices=list(SUPPORTED_AGENTS),
                        help="Agent type: sa (SWE-agent) or oh (OpenHands)")
    parser.add_argument("--model", type=str, required=True, choices=list(SUPPORTED_MODELS),
                        help="Model type: dsk-v3, dsk-r1, dev, or cld-4")
    parser.add_argument("--trajs", type=str, required=True,
                        help="Path to trajectories (directory for SA, output.jsonl for OH)")
    parser.add_argument("--eval_report", type=str, required=True,
                        help="Path to evaluation report JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output directory")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")

    args = parser.parse_args()

    trajs_path = Path(args.trajs)
    eval_report_path = Path(args.eval_report)

    if not trajs_path.exists():
        print(f"[ERROR] Trajectories path does not exist: {trajs_path}")
        sys.exit(1)
    if not eval_report_path.exists():
        print(f"[ERROR] Evaluation report does not exist: {eval_report_path}")
        sys.exit(1)

    graph_output_dir = get_graph_output_dir(args.output_dir, args.agent, args.model)
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"Agent:        {AGENT_NAMES[args.agent]}")
    print(f"Model:        {MODEL_NAMES[args.model]}")
    print(f"Trajectories: {trajs_path}")
    print(f"Eval Report:  {eval_report_path}")
    print(f"Graph Output: {graph_output_dir}")
    print(f"Workers:      {args.workers}")
    print(f"{'='*70}\n")

    print("Loading trajectories...")
    try:
        if args.agent == "sa":
            trajectories = TrajectoryLoader.load_sa_trajectories(trajs_path)
        else:
            trajectories = TrajectoryLoader.load_oh_trajectories(trajs_path)
    except Exception as e:
        print(f"[ERROR] Failed to load trajectories: {e}")
        sys.exit(1)

    if not trajectories:
        print("[ERROR] No trajectories found")
        sys.exit(1)

    print(f"Loaded {len(trajectories)} trajectories\n")

    cmd_parser = setup_parser_for_agent(args.agent)

    processor = GraphProcessor(
        agent=args.agent,
        parser=cmd_parser,
        eval_report_path=str(eval_report_path),
        output_dir=graph_output_dir,
    )

    results = process_batch(trajectories, processor, max_workers=args.workers)
    print_summary(results, args.agent, args.model, graph_output_dir)

    if results["failed"]:
        sys.exit(1)
    print("✓ All trajectories processed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()