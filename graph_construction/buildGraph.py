"""
Graph Construction Module

Builds trajectory graphs from agent execution traces (SWE-agent and OpenHands).
"""

import json
import os
import re
import hashlib
import networkx as nx
from pathlib import Path
from networkx.readwrite import json_graph
from collections import defaultdict
# Import the refactored visualizer
from visualizer import GraphVisualizer

# Optional datasets import for difficulty lookup
try:
    from datasets import load_dataset
    swe_bench_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    difficulty_lookup = {row["instance_id"]: row["difficulty"] for row in swe_bench_ds}
except ImportError:
    # Fallback if datasets is not available
    difficulty_lookup = {}


FONT_FAMILY = os.environ.get("GRAPH_FONT", "DejaVu Sans, Arial, sans-serif")

# ── Thought-length helpers ──────────────────────────────────────────────────

def compute_thought_length_raw(thought: str) -> int:
    """Raw character count of thought text."""
    return len(thought or "")


def compute_thought_length_clean(thought: str) -> int:
    """Character count excluding text inside quotes/backticks.

    Strips content inside:
      - "..."  (double quotes)
      - '...'  (single quotes)
      - `...`  (single backtick)
      - ```...``` (triple backtick)
    """
    if not thought:
        return 0
    s = re.sub(r'```.*?```', '', thought, flags=re.DOTALL)
    s = re.sub(r'`[^`]*`', '', s)
    s = re.sub(r'"[^"]*"', '', s)
    s = re.sub(r"'[^']*'", '', s)
    return len(s)


# ── Outcome detection helper ────────────────────────────────────────────────

def detect_observation_outcome(observation: str) -> str:
    """Return 'success', 'failure', or 'neutral' based on observation content."""
    if not observation:
        return "neutral"

    obs_lower = observation.lower()

    failure_signs = [
        "traceback (most recent call last)",
        "error:",
        "exception:",
        "failed",
        "failure",
        "assertion",
        "syntaxerror",
        "nameerror",
        "typeerror",
    ]
    if any(sign in obs_lower for sign in failure_signs):
        return "failure"

    success_signs = [
        "success",
        "passed",
        "ok",
        "has been edited",
        "created successfully",
    ]
    if any(sign in obs_lower for sign in success_signs):
        return "success"

    return "neutral"


# -------------------- Helpers --------------------
def hash_node_signature(label, args, flags):
    """Create unique hash for node signature."""
    normalized = json.dumps({"label": label, "args": args, "flags": flags}, sort_keys=True)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()


def check_edit_status(tool, subcommand, args, observation):
    """Check if an edit operation succeeded or failed."""
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
    with open(eval_report_path, 'r') as f:
        report = json.load(f)
    if instance_id in report.get("resolved_ids", []):
        return "resolved"
    elif instance_id in report.get("unresolved_ids", []):
        return "unresolved"
    return "unsubmitted"


# -------------------- Graph Builder Class --------------------
class GraphBuilder:
    """Utility class for managing graph construction operations.

    This class encapsulates all shared graph construction logic for building
    trajectory graphs from agent execution traces.
    """

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.node_signature_to_key = {}
        self.localization_nodes = []
        self.prev_phases = set()
        self.previous_node = None
        self.thought_history = []  # Track (node_key, thought_text) pairs

    def add_or_update_node(self, node_label, args, flags, phase, step_idx,
                          tool=None, command=None, subcommand=None, thought_length=0, has_cd=False):
        """Add a new node or update existing node with a new occurrence.

        Args:
            node_label: Display label for the node
            args: Command arguments dictionary
            flags: Command flags dictionary
            phase: Phase classification (localization/patch/validation/general)
            step_idx: Step index in trajectory
            tool: Tool name (if applicable)
            command: Command name (if applicable)
            subcommand: Subcommand name (if applicable)
            thought_length: Length of thought text for this step
            has_cd: Whether this node had a cd command stripped

        Returns:
            node_key: The key of the added or updated node
        """
        node_signature = hash_node_signature(node_label, args, flags)

        if node_signature in self.node_signature_to_key:
            # Update existing node
            node_key = self.node_signature_to_key[node_signature]
            self.G.nodes[node_key]["step_indices"].append(step_idx)
            self.G.nodes[node_key]["thought_lengths"].append(thought_length)
            if "phases" not in self.G.nodes[node_key]:
                self.G.nodes[node_key]["phases"] = []
            self.G.nodes[node_key]["phases"].append(phase)
            # Update has_cd if this occurrence has cd
            if has_cd:
                self.G.nodes[node_key]["has_cd"] = True
        else:
            # Add new node
            node_key = f"{len(self.G.nodes)}:{node_label}"
            self.G.add_node(
                node_key,
                label=node_label,
                args=args,
                flags=flags,
                phases=[phase],
                step_indices=[step_idx],
                thought_lengths=[thought_length],
                tool=tool,
                command=command,
                subcommand=subcommand,
                has_cd=has_cd
            )
            self.node_signature_to_key[node_signature] = node_key

            # Track localization nodes
            if tool == "str_replace_editor" and subcommand == "view":
                self.localization_nodes.append(node_key)

        return node_key

    def add_execution_edge(self, node_key, step_idx, is_first_in_step=False,
                           thought_length_raw: int = 0,
                           thought_length_clean: int = 0):
        """Add execution edge from previous node to current node.

        Args:
            node_key: Target node key
            step_idx: Step index for edge label
            is_first_in_step: Whether this is the first edge in this trajectory step
            thought_length_raw: Raw character count of the thought for this step
            thought_length_clean: Character count with quoted text stripped
        """
        if self.previous_node:
            self.G.add_edge(
                self.previous_node,
                node_key,
                label=str(step_idx),
                type="exec",
                is_first_in_step=is_first_in_step,
                thought_length_raw=thought_length_raw,
                thought_length_clean=thought_length_clean,
            )

    def update_previous_node(self, node_key):
        """Update the previous node pointer.

        Args:
            node_key: Node to set as previous
        """
        self.previous_node = node_key

    def add_phase(self, phase):
        """Add phase to the set of previous phases.

        Args:
            phase: Phase to add
        """
        self.prev_phases.add(phase)

    def track_thought(self, node_key, thought_text):
        """Track thought text for a node and detect substring relationships.
        
        Args:
            node_key: The node to associate with this thought
            thought_text: The thought text content
        """
        # Only track non-empty thoughts
        if not thought_text or not thought_text.strip():
            return
        
        # Check if this thought is a substring continuation of the previous thought
        if self.thought_history:
            prev_node_key, prev_thought = self.thought_history[-1]
            
            # Check if previous thought is a substring of current thought
            # (indicating the current thought extends the previous one)
            if prev_thought and thought_text.startswith(prev_thought):
                # Add a "thought" edge to show this relationship
                self.G.add_edge(prev_node_key, node_key, type="thought", label="")
        
        # Add to history
        self.thought_history.append((node_key, thought_text))

    def finalize_and_save(self, output_dir, instance_id, eval_report_path, template_dir=None, metadata_comment=""):
        """Build hierarchical edges, add metadata, and save graph.

        Args:
            output_dir: Base output directory
            instance_id: Instance identifier
            eval_report_path: Path to evaluation report
            template_dir: Optional path to template directory for visualizer
            metadata_comment: Optional comment about model/plan

        Returns:
            tuple: (json_path, html_path) paths to saved files
        """
        build_hierarchical_edges(self.G, self.localization_nodes)

        resolution_status = determine_resolution_status(instance_id, eval_report_path)
        self.G.graph["resolution_status"] = resolution_status
        self.G.graph["instance_name"] = instance_id
        self.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")

        # Construct output paths: output_dir/{instance_id}/{instance_id}.{json,html}
        instance_dir = os.path.join(output_dir, instance_id)
        os.makedirs(instance_dir, exist_ok=True)

        json_path = os.path.join(instance_dir, f"{instance_id}.json")
        html_path = os.path.join(instance_dir, f"{instance_id}.html")

        # Save JSON
        with open(json_path, "w") as f:
            json.dump(json_graph.node_link_data(self.G, edges="edges"), f, indent=2)

        # Save HTML using refactored visualizer
        GraphVisualizer.draw_with_timeout(
            self.G, 
            html_path, 
            timeout_sec=60,
            template_dir=template_dir,
            metadata_comment=metadata_comment
        )

        return json_path, html_path


# -------------------- Build graph --------------------
def build_graph_from_sa_trajectory(traj_data, parser, instance_id, output_dir, eval_report_path, template_dir=None, metadata_comment=""):
    """Build graph from SWE-agent trajectory data.

    Args:
        traj_data: SWE-agent trajectory dictionary containing 'trajectory' key
        parser: CommandParser instance for parsing action strings
        instance_id: Instance identifier (e.g., 'django__django-12345')
        output_dir: Base output directory for saving graphs
        eval_report_path: Path to evaluation report JSON file
        template_dir: Optional path to template directory for visualizer
        metadata_comment: Optional comment about model/plan

    Returns:
        tuple: (json_path, html_path) paths to the saved graph files

    Output Structure:
        {output_dir}/{instance_id}/{instance_id}.json
        {output_dir}/{instance_id}/{instance_id}.html
    """
    from mapPhase import get_phase
    
    builder = GraphBuilder()
    trajectory = traj_data.get("trajectory", [])

    for step_idx, step in enumerate(trajectory):
        action_str = step.get("action", "")
        thought = step.get("thought", "") or ""
        observation = step.get("observation", "") or ""

        thought_len_raw   = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)

        # Handle explicit "think" steps (blank action)
        if action_str.strip() == "":
            node_key = builder.add_or_update_node(
                node_label="think",
                args={"thought_len": thought_len_raw},
                flags={},
                phase="general",
                step_idx=step_idx,
                tool=None,
                command=None,
                subcommand=None,
                thought_length=thought_len_raw
            )
            builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            builder.G.nodes[node_key]["observation_length"]  = len(observation)
            builder.G.nodes[node_key]["observation_outcome"] = detect_observation_outcome(observation)
            builder.add_execution_edge(node_key, step_idx,
                                       is_first_in_step=True,
                                       thought_length_raw=thought_len_raw,
                                       thought_length_clean=thought_len_clean)
            builder.update_previous_node(node_key)
            builder.add_phase("general")
            continue

        # Parse actionable commands
        parsed_commands = parser.parse(action_str)
        if not parsed_commands:
            continue

        # Filter out cd commands if there are other commands in the same step
        # and mark remaining nodes as having cd prefix
        has_cd = False
        filtered_commands = []
        
        if len(parsed_commands) > 1:
            first_cmd = parsed_commands[0]
            if first_cmd.get("command", "").strip().lower() == "cd":
                has_cd = True
                filtered_commands = parsed_commands[1:]
            else:
                filtered_commands = parsed_commands
        else:
            filtered_commands = parsed_commands
        
        is_first_in_step = True
        node_keys_in_step = []
        
        for parsed in filtered_commands:
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

            edit_status = check_edit_status(tool, subcommand, args, observation)
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_key = builder.add_or_update_node(
                node_label=node_label,
                args=args,
                flags=flags,
                phase=phase,
                step_idx=step_idx,
                tool=tool,
                command=command,
                subcommand=subcommand,
                thought_length=thought_len_raw,
                has_cd=has_cd
            )
            builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            node_keys_in_step.append(node_key)

            builder.add_execution_edge(
                node_key, step_idx,
                is_first_in_step=is_first_in_step,
                thought_length_raw=thought_len_raw if is_first_in_step else 0,
                thought_length_clean=thought_len_clean if is_first_in_step else 0,
            )
            builder.update_previous_node(node_key)
            builder.add_phase(phase)
            is_first_in_step = False

        # Mark last node of this step with observation info
        if node_keys_in_step:
            last_node = node_keys_in_step[-1]
            builder.G.nodes[last_node]["observation_length"]  = len(observation)
            builder.G.nodes[last_node]["observation_outcome"] = detect_observation_outcome(observation)

    return builder.finalize_and_save(output_dir, instance_id, eval_report_path, template_dir, metadata_comment)


def build_graph_from_oh_trajectory(traj_data, parser, instance_id, output_dir, eval_report_path, template_dir=None, metadata_comment=""):
    """Build graph from OpenHands trajectory data.

    Args:
        traj_data: OpenHands trajectory dictionary containing 'history' key
        parser: CommandParser instance for parsing action strings
        instance_id: Instance identifier (e.g., 'django__django-12345')
        output_dir: Base output directory for saving graphs
        eval_report_path: Path to evaluation report JSON file
        template_dir: Optional path to template directory for visualizer
        metadata_comment: Optional comment about model/plan

    Returns:
        tuple: (json_path, html_path) paths to the saved graph files

    Output Structure:
        {output_dir}/{instance_id}/{instance_id}.json
        {output_dir}/{instance_id}/{instance_id}.html
    """
    from mapPhase import get_phase
    
    builder = GraphBuilder()
    step_idx = 0

    for step in traj_data.get("history", []):
        action = step.get("observation") if step.get("observation") else None
        if action in ("system", "message") or action is None:
            continue

        # Use action text only as a fallback when command string is empty
        action_str = action or ""
        thought = step.get("content", "") or ""

        thought_len_raw   = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)

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
                subcommand = args_loaded.pop("command", None)  # remove 'command' key from args
                parsed_commands = [{
                    "tool": tool_name,
                    "subcommand": subcommand,
                    "args": args_loaded,
                }]

        if not parsed_commands:
            continue

        # Filter out cd commands if there are other commands in the same step
        has_cd = False
        filtered_commands = []
        
        if len(parsed_commands) > 1:
            first_cmd = parsed_commands[0]
            if first_cmd.get("command", "").strip().lower() == "cd":
                has_cd = True
                filtered_commands = parsed_commands[1:]
            else:
                filtered_commands = parsed_commands
        else:
            filtered_commands = parsed_commands
        
        is_first_in_step = True
        node_keys_in_step = []

        for parsed in filtered_commands:
            tool = parsed.get("tool", "").strip()
            
            # ---- THINK NODES ----
            if tool == "think":
                node_key = builder.add_or_update_node(
                    node_label="think",
                    args={"thought_len": thought_len_raw},
                    flags={},
                    phase="general",
                    step_idx=step_idx,
                    tool=None,
                    command=None,
                    subcommand=None,
                    thought_length=thought_len_raw
                )
                builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
                builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
                node_keys_in_step.append(node_key)
                builder.add_execution_edge(node_key, step_idx,
                                           is_first_in_step=is_first_in_step,
                                           thought_length_raw=thought_len_raw if is_first_in_step else 0,
                                           thought_length_clean=thought_len_clean if is_first_in_step else 0)
                builder.update_previous_node(node_key)
                builder.add_phase("general")
                is_first_in_step = False
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

            observation = step.get("content", "") or ""
            edit_status = check_edit_status(tool, subcommand, args, observation)
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_key = builder.add_or_update_node(
                node_label=node_label,
                args=args,
                flags=flags,
                phase=phase,
                step_idx=step_idx,
                tool=tool,
                command=command,
                subcommand=subcommand,
                thought_length=thought_len_raw,
                has_cd=has_cd
            )
            builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            node_keys_in_step.append(node_key)
            builder.add_execution_edge(node_key, step_idx,
                                       is_first_in_step=is_first_in_step,
                                       thought_length_raw=thought_len_raw if is_first_in_step else 0,
                                       thought_length_clean=thought_len_clean if is_first_in_step else 0)
            builder.update_previous_node(node_key)
            builder.add_phase(phase)
            is_first_in_step = False

        # Mark last node of this step with observation info
        if node_keys_in_step:
            last_node = node_keys_in_step[-1]
            obs_text = step.get("content", "") or ""
            builder.G.nodes[last_node]["observation_length"]  = len(obs_text)
            builder.G.nodes[last_node]["observation_outcome"] = detect_observation_outcome(obs_text)

        step_idx += 1

    return builder.finalize_and_save(output_dir, instance_id, eval_report_path, template_dir, metadata_comment)


def build_hierarchical_edges(G: nx.MultiDiGraph, localization_nodes):
    """Add 'hier' edges between str_replace_editor view nodes based on file-path
    containment and view-range nesting.

    Hierarchy rules
    ---------------
    1. Directory containment: if node A views a directory (or file) that is a
       prefix of the path viewed by node B, add A → B.
    2. Range nesting within the same file: if node A views a range [a1, a2] and
       node B views [b1, b2] with b1 >= a1 and b2 <= a2, add A → B.
    3. Whole-file view → ranged view of the same file: if node A has no range
       and node B views a range of the same file, add A → B.
    """
    path_nodes: list[tuple[str, list | None]] = []  # (node_key, view_range_or_None)

    for node_key in localization_nodes:
        data = G.nodes.get(node_key, {})
        args = data.get("args", {}) or {}
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not path:
            continue
        vr = args.get("view_range")
        if isinstance(vr, (list, tuple)) and len(vr) == 2:
            try:
                vr = [int(vr[0]), int(vr[1])]
            except (TypeError, ValueError):
                vr = None
        else:
            vr = None
        path_nodes.append((node_key, str(path), vr))

    added: set[tuple] = set()

    def _add(src, dst):
        if src != dst and (src, dst) not in added:
            G.add_edge(src, dst, type="hier", label="")
            added.add((src, dst))

    # Group by normalised path for range comparisons
    by_path: dict[str, list] = defaultdict(list)
    for node_key, path, vr in path_nodes:
        by_path[path].append((node_key, vr))

    for path, entries in by_path.items():
        whole  = [(nk, vr) for nk, vr in entries if vr is None]
        ranged = [(nk, vr) for nk, vr in entries if vr is not None]

        # Whole-file → ranged views of same file
        for w_nk, _ in whole:
            for r_nk, _ in ranged:
                _add(w_nk, r_nk)

        # Range nesting: outer range → inner range
        for i, (nk_a, vr_a) in enumerate(ranged):
            for j, (nk_b, vr_b) in enumerate(ranged):
                if i == j:
                    continue
                if vr_b[0] >= vr_a[0] and vr_b[1] <= vr_a[1]:
                    _add(nk_a, nk_b)

    # Directory/path prefix containment across different paths
    path_list = list(by_path.keys())
    for path_a in path_list:
        for path_b in path_list:
            if path_a == path_b:
                continue
            parts_a = [p for p in path_a.replace("\\", "/").split("/") if p]
            parts_b = [p for p in path_b.replace("\\", "/").split("/") if p]
            if (len(parts_a) < len(parts_b) and
                    parts_b[:len(parts_a)] == parts_a):
                # path_a is a parent dir of path_b
                for nk_a, _ in by_path[path_a]:
                    for nk_b, _ in by_path[path_b]:
                        _add(nk_a, nk_b)