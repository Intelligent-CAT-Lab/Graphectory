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

        # Process all commands including cd; mark subsequent nodes with has_cd
        is_first_in_step = True
        node_keys_in_step = []
        saw_cd = False

        for parsed in parsed_commands:
            tool = parsed.get("tool", "").strip() if parsed.get("tool") else ""
            subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
            command = parsed.get("command", "").strip() if parsed.get("command") else ""
            args = parsed.get("args", {})
            flags = parsed.get("flags", {})

            # Check if this is a cd command
            is_cd = command.lower() == "cd"
            if is_cd:
                saw_cd = True

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, builder.prev_phases, flags)

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
                has_cd=(saw_cd and not is_cd)
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

        # Process all commands including cd; mark subsequent nodes with has_cd
        is_first_in_step = True
        node_keys_in_step = []
        saw_cd = False

        for parsed in parsed_commands:
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

            # Check if this is a cd command
            is_cd = command.lower() == "cd"
            if is_cd:
                saw_cd = True

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, builder.prev_phases, flags)

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
                has_cd=(saw_cd and not is_cd)
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


def build_graph_from_msa_trajectory(traj_data, parser, instance_id, output_dir, eval_report_path, template_dir=None, metadata_comment=""):
    """Build graph from mini-swe-agent trajectory data.

    Mini-swe-agent format: messages = [system, user, assistant_resp, tool_result, ...]
    - Assistant response contains: thought (message) + actions (function_calls)
    - Tool result contains: observation (extra.raw_output or output string)

    Args:
        traj_data: Mini-swe-agent trajectory dictionary containing 'messages' key
        parser: CommandParser instance for parsing action strings
        instance_id: Instance identifier (e.g., 'astropy__astropy-12907')
        output_dir: Base output directory for saving graphs
        eval_report_path: Path to evaluation report JSON file
        template_dir: Optional path to template directory for visualizer
        metadata_comment: Optional comment about model/plan

    Returns:
        tuple: (json_path, html_path) paths to the saved graph files
    """
    from mapPhase import get_phase

    builder = GraphBuilder()
    messages = traj_data.get("messages", [])
    step_idx = 0

    # Process messages in pairs: assistant response (i) + tool result (i+1)
    i = 2  # Skip system and user messages
    while i < len(messages):
        msg = messages[i]

        # Skip if not an assistant response with output
        if not msg.get("output") or not isinstance(msg.get("output"), list):
            i += 1
            continue

        # Extract thought from message content
        thought = ""
        for item in msg.get("output", []):
            if isinstance(item, dict) and item.get("type") == "message":
                content = item.get("content", [])
                if content and isinstance(content, list):
                    thought = content[0].get("text", "")
                    break

        thought_len_raw = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)

        # Extract actions from function calls
        actions = []
        for item in msg.get("output", []):
            if isinstance(item, dict) and item.get("type") == "function_call":
                try:
                    args_json = json.loads(item.get("arguments", "{}"))
                    command = args_json.get("command", "")
                    if command:
                        actions.append(command)
                except json.JSONDecodeError:
                    continue

        # Get observation from next message
        observation = ""
        if i + 1 < len(messages):
            next_msg = messages[i + 1]
            # observation can be in 'output' (as string) or 'extra.raw_output'
            if isinstance(next_msg.get("output"), str):
                observation = next_msg["output"]
            else:
                observation = next_msg.get("extra", {}).get("raw_output", "")

        # Process each action
        if actions:
            for action_str in actions:
                if not action_str.strip():
                    continue

                parsed_commands = parser.parse(action_str)
                if not parsed_commands:
                    # Create generic node for unparsed commands
                    parsed_commands = [{
                        "tool": None,
                        "subcommand": None,
                        "command": action_str.split()[0] if action_str.split() else "bash",
                        "args": {"_raw": action_str},
                        "flags": {}
                    }]

                # Process all commands including cd
                is_first_in_step = True
                node_keys_in_step = []
                saw_cd = False

                for parsed in parsed_commands:
                    tool = parsed.get("tool", "").strip() if parsed.get("tool") else ""
                    subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
                    command = parsed.get("command", "").strip() if parsed.get("command") else ""
                    args = parsed.get("args", {})
                    flags = parsed.get("flags", {})

                    # Check if this is a cd command
                    is_cd = command.lower() == "cd"
                    if is_cd:
                        saw_cd = True

                    if tool:
                        node_label = f"{tool}: {subcommand}" if subcommand else tool
                    else:
                        node_label = command.strip() or action_str.strip()

                    phase = get_phase(tool, subcommand, command, args, builder.prev_phases, flags)

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
                        has_cd=(saw_cd and not is_cd)
                    )
                    builder.G.nodes[node_key]["thought_len_raw"] = thought_len_raw
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

                # Mark last node with observation info
                if node_keys_in_step:
                    last_node = node_keys_in_step[-1]
                    builder.G.nodes[last_node]["observation_length"] = len(observation)
                    builder.G.nodes[last_node]["observation_outcome"] = detect_observation_outcome(observation)

                step_idx += 1

        # Skip to next assistant response
        i += 2

    return builder.finalize_and_save(output_dir, instance_id, eval_report_path, template_dir, metadata_comment)


def build_hierarchical_edges(G: nx.MultiDiGraph, localization_nodes):
    """Add 'hier' edges between str_replace_editor view nodes based on file-path
    containment and view-range nesting, with transitive reduction.

    Hierarchy rules (with transitive reduction)
    --------------------------------------------
    1. Directory containment: each node connects only to its closest parent,
       not all ancestors (avoiding A→B, B→C, A→C redundancy).
    2. Range nesting: each range connects only to its immediate outer range.
    3. Whole-file → ranged views: outermost ranges connect to path nodes.
    """
    path_nodes = []  # [(node_id, Path_object)]
    range_nodes_by_path = defaultdict(list)  # path_str -> [(node_id, [start, end])]

    for node in localization_nodes:
        data = G.nodes.get(node, {})
        args = data.get("args", {}) or {}
        if not isinstance(args, dict):
            continue
        path = args.get("path")
        if not path:
            continue

        view_range = args.get("view_range")
        if view_range is None:
            path_nodes.append((node, Path(path)))
        elif (isinstance(view_range, (list, tuple)) and
              len(view_range) == 2 and
              all(isinstance(x, int) for x in view_range)):
            range_nodes_by_path[str(Path(path))].append((node, view_range))

    # --- 1) Path hierarchy by folder containment (closest parent only) ---
    for child_node, child_path in path_nodes:
        best_parent_node = None
        best_parent_path = None
        for parent_node, parent_path in path_nodes:
            if parent_node == child_node:
                continue
            # Check if parent_path is a prefix of child_path
            if (len(parent_path.parts) < len(child_path.parts) and
                child_path.parts[:len(parent_path.parts)] == parent_path.parts):
                # Keep only the closest (deepest) parent
                if best_parent_path is None or len(parent_path.parts) > len(best_parent_path.parts):
                    best_parent_node = parent_node
                    best_parent_path = parent_path
        if best_parent_node:
            G.add_edge(best_parent_node, child_node, type="hier", label="")

    # --- 2) Range nodes: handle nesting + link outermost to path nodes ---
    path_to_node = {str(p): n for n, p in path_nodes}

    for path_str, range_nodes in range_nodes_by_path.items():
        is_nested = {n: False for n, _ in range_nodes}

        # Detect nesting and mark inner ranges, connecting only immediate parent→child
        for i, (node_i, r_i) in enumerate(range_nodes):
            for j, (node_j, r_j) in enumerate(range_nodes):
                if i == j:
                    continue
                a1, a2 = r_i
                b1, b2 = r_j
                # node_j is nested inside node_i
                if b1 >= a1 and b2 <= a2:
                    # Check if there's no intermediate range between i and j
                    is_immediate = True
                    for k, (node_k, r_k) in enumerate(range_nodes):
                        if k == i or k == j:
                            continue
                        c1, c2 = r_k
                        # node_k is between node_i and node_j if:
                        # c is inside i AND j is inside c
                        if (c1 >= a1 and c2 <= a2 and b1 >= c1 and b2 <= c2):
                            is_immediate = False
                            break
                    if is_immediate:
                        G.add_edge(node_i, node_j, type="hier", label="")
                        is_nested[node_j] = True

        # Link outermost ranges to path node (or closest ancestor)
        path_node = path_to_node.get(path_str)
        if path_node:
            for node, _ in range_nodes:
                if not is_nested[node]:
                    G.add_edge(path_node, node, type="hier", label="")
        else:
            # No exact path node → find nearest ancestor
            path_parts = Path(path_str).parts
            best_ancestor_node = None
            best_ancestor_depth = -1
            for pn, pp in path_nodes:
                if (len(pp.parts) < len(path_parts) and
                    path_parts[:len(pp.parts)] == pp.parts):
                    if len(pp.parts) > best_ancestor_depth:
                        best_ancestor_node = pn
                        best_ancestor_depth = len(pp.parts)
            for node, _ in range_nodes:
                if not is_nested[node] and best_ancestor_node:
                    G.add_edge(best_ancestor_node, node, type="hier", label="")