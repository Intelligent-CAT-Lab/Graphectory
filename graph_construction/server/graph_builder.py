"""
server/graph_builder.py

Responsible for:
  - Scanning the trajectories directory for available instances
  - Loading individual .traj files
  - Building a NetworkX graph from a trajectory (with optional cd filtering)
"""

import json
import re
import sys
from pathlib import Path

# Ensure parent directory is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from buildGraph import (
    GraphBuilder as _GraphBuilderBase,
    determine_resolution_status,
    check_edit_status,
    compute_thought_length_raw,
    compute_thought_length_clean,
    detect_observation_outcome,
    build_hierarchical_edges,
)

import networkx as nx


# ── Test-outcome helpers ─────────────────────────────────────────────────────

TEST_COMMANDS = {"python", "python2", "python3", "pytest", "unittest", "nosetests", "tox"}
RE_PYTEST_FAIL  = re.compile(r"\b(\d+)\s+failed\b",  re.IGNORECASE)
RE_PYTEST_ERROR = re.compile(r"\b(\d+)\s+errors?\b", re.IGNORECASE)
RE_PYTEST_PASS  = re.compile(r"\b(\d+)\s+passed\b",  re.IGNORECASE)
EXCEPTION_SIGNS = ["Traceback (most recent call last):"]


def check_command_outcome(command: str, observation: str,
                          tool: str = None, subcommand: str = None,
                          args: dict = None) -> str | None:
    """Return 'success', 'failure', or None for a command + its observation."""
    obs = observation or ""

    # Edit-status from str_replace_editor takes priority
    if tool and subcommand:
        edit_status = check_edit_status(tool, subcommand, args or {}, observation)
        if edit_status and str(edit_status).startswith("failure"):
            return "failure"
        if edit_status == "success":
            return "success"

    for sig in EXCEPTION_SIGNS:
        if sig in obs:
            return "failure"

    if RE_PYTEST_FAIL.search(obs) or RE_PYTEST_ERROR.search(obs):
        return "failure"
    if RE_PYTEST_PASS.search(obs):
        return "success"
    if "FAILURES" in obs or "ERRORS" in obs or "INTERNALERROR" in obs:
        return "failure"

    return None


# ── Extended GraphBuilder ────────────────────────────────────────────────────

class GraphBuilder(_GraphBuilderBase):
    """Extends the base GraphBuilder – no overrides needed; inherits everything."""
    pass


# ── Directory scanning ──────────────────────────────────────────────────────

def scan_trajectories(graphs_dir: Path,
                      eval_report_path: str | None = None,
                      agent_type: str = "sa") -> list[dict]:
    """Return a sorted list of trajectory metadata dicts.

    Each dict has: instance_id, status, difficulty, step_count.

    For SWE-agent (agent_type='sa'), graphs_dir is a directory tree of .traj files.
    For OpenHands (agent_type='oh'), graphs_dir is a path to an output.jsonl file.
    """
    resolved_set:   set[str] = set()
    unresolved_set: set[str] = set()
    if eval_report_path:
        try:
            with open(eval_report_path) as f:
                report = json.load(f)
            resolved_set   = set(report.get("resolved_ids",   []))
            unresolved_set = set(report.get("unresolved_ids", []))
        except Exception:
            pass

    results = []

    if agent_type == "oh":
        # OpenHands: graphs_dir is actually the output.jsonl file
        jsonl_path = graphs_dir
        if not jsonl_path.is_file():
            return results
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                instance_id = entry.get("instance_id")
                if not instance_id:
                    continue

                if instance_id in resolved_set:
                    status = "resolved"
                elif instance_id in unresolved_set:
                    status = "unresolved"
                else:
                    status = "unsubmitted"

                # Count action steps (non-system, non-message observations)
                step_count = sum(
                    1 for s in entry.get("history", [])
                    if s.get("observation") not in ("system", "message", None)
                )

                results.append({
                    "instance_id": instance_id,
                    "status":      status,
                    "difficulty":  "unknown",
                    "step_count":  step_count,
                })

        results.sort(key=lambda x: x["instance_id"])
        return results

    # SWE-agent: directory tree of .traj files
    for traj_file in sorted(graphs_dir.rglob("*.traj")):
        instance_id = traj_file.stem

        if instance_id in resolved_set:
            status = "resolved"
        elif instance_id in unresolved_set:
            status = "unresolved"
        else:
            status = "unsubmitted"
            json_file = traj_file.with_suffix(".json")
            if json_file.exists():
                try:
                    with open(json_file) as f:
                        meta = json.load(f)
                    s = meta.get("graph", {}).get("resolution_status", "")
                    if s in ("resolved", "unresolved", "unsubmitted"):
                        status = s
                except Exception:
                    pass

        difficulty = "unknown"
        json_file = traj_file.with_suffix(".json")
        if json_file.exists():
            try:
                with open(json_file) as f:
                    meta = json.load(f)
                difficulty = meta.get("graph", {}).get("debug_difficulty", "unknown")
            except Exception:
                pass

        step_count = 0
        try:
            with open(traj_file) as f:
                traj = json.load(f)
            step_count = len(traj.get("trajectory", []))
        except Exception:
            pass

        results.append({
            "instance_id": instance_id,
            "status":      status,
            "difficulty":  difficulty,
            "step_count":  step_count,
        })

    return results


# ── Trajectory loading ──────────────────────────────────────────────────────

def load_trajectory(graphs_dir: Path, instance_id: str,
                    agent_type: str = "sa") -> dict:
    """Load and return raw trajectory data for *instance_id*.

    For SWE-agent, searches for a matching .traj file under graphs_dir.
    For OpenHands, scans the output.jsonl file for the matching instance.

    Raises FileNotFoundError if the trajectory cannot be found.
    """
    if agent_type == "oh":
        jsonl_path = graphs_dir
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"OpenHands output.jsonl not found: {jsonl_path}"
            )
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("instance_id") == instance_id:
                    return entry
        raise FileNotFoundError(
            f"No entry for '{instance_id}' found in {jsonl_path}"
        )

    # SWE-agent: search for .traj file
    for traj_file in graphs_dir.rglob(f"{instance_id}.traj"):
        with open(traj_file) as f:
            return json.load(f)

    raise FileNotFoundError(
        f"No .traj file found for '{instance_id}' under {graphs_dir}"
    )


def _find_instance_config(graphs_dir: Path, instance_id: str) -> Path | None:
    """Locate the config YAML for a given instance.

    Expected location: {graphs_dir}/{instance_id}/{instance_id}.config.yaml
    Falls back to a recursive search within graphs_dir if not found at the
    canonical location.
    """
    # Canonical path (matches the observed folder structure)
    canonical = graphs_dir / instance_id / f"{instance_id}.config.yaml"
    if canonical.exists():
        return canonical

    # Fallback: recursive glob (handles unexpected nesting depths)
    for match in graphs_dir.rglob(f"{instance_id}.config.yaml"):
        return match

    return None


def _make_parser_for_instance(base_parser, graphs_dir: Path, instance_id: str):
    """Return a CommandParser loaded with the instance's tool config.

    Creates a fresh CommandParser and copies the base parser's tool_map as a
    starting point, then overlays the instance-specific config YAML on top.
    This avoids mutating the shared base parser between requests.
    """
    import copy
    from commandParser import CommandParser

    parser = CommandParser()
    # Start from whatever the base already knows (may be empty)
    parser.tool_map = copy.deepcopy(base_parser.tool_map)

    config_path = _find_instance_config(graphs_dir, instance_id)
    if config_path:
        parser.load_tool_yaml_files([str(config_path)])
        print(f"  [config] Loaded {config_path.name}")
    else:
        print(f"  [config] No config YAML found for '{instance_id}' – using base parser")

    return parser




def _accumulate_step_data(node_data: dict, step_idx: int,
                           thought: str, action: str, observation: str) -> None:
    """Append the full text of this step visit to the node's step_data list.

    Each entry is a dict with step_idx, thought, action, observation so the
    detail sidebar can display them verbatim and let users page between visits.
    """
    if "step_data" not in node_data:
        node_data["step_data"] = []
    node_data["step_data"].append({
        "step_idx":    step_idx,
        "thought":     thought or "",
        "action":      action  or "",
        "observation": observation or "",
    })


def _accumulate_observation(node_data: dict, observation: str) -> None:
    """Append the observation length for this step visit to the node's running list.

    Also maintains the scalar ``observation_length`` / ``observation_outcome``
    fields (set to the most-recent value) so older rendering code keeps working.
    """
    length  = len(observation)
    outcome = detect_observation_outcome(observation)

    if "observation_lengths" not in node_data:
        node_data["observation_lengths"] = []
    node_data["observation_lengths"].append(length)

    # Scalar fields: keep the latest value (renderer uses last step's outcome)
    node_data["observation_length"]  = length
    node_data["observation_outcome"] = outcome


# ── Thought-continuation helper ─────────────────────────────────────────────

def _mark_thought_continuation(
    G,
    src_node: str | None,
    dst_node: str,
    prev_thought: str,
    curr_thought: str,
) -> None:
    """Mark the most-recently-added exec edge src→dst as a thought continuation.

    A continuation is detected when prev_thought is non-empty and is either
    equal to curr_thought or is a substring of it (the model reused / extended
    its previous reasoning verbatim).  Only the edge whose endpoints match
    (src_node, dst_node) is updated; all other edges between the same pair are
    left untouched.
    """
    if not src_node or not prev_thought or not curr_thought:
        return
    if prev_thought not in curr_thought:
        return
    # Walk the most-recently-added parallel edge between src→dst
    edges = G.get_edge_data(src_node, dst_node)
    if not edges:
        return
    # MultiDiGraph stores edges as {0: data, 1: data, …}; use the last key
    last_key = max(edges.keys())
    if edges[last_key].get("type") == "exec":
        edges[last_key]["is_thought_continuation"] = True


# ── OpenHands graph construction ────────────────────────────────────────────

def _build_graph_oh(traj_data: dict, instance_id: str,
                    eval_report_path: str, cmd_parser,
                    filter_cd: bool = True,
                    unique_think: bool = True):
    """Build a NetworkX MultiDiGraph from an OpenHands trajectory entry.

    OpenHands trajectories store tool calls inside ``history`` entries.
    Each history step has an ``observation`` field that names the event type,
    plus a ``tool_call_metadata`` dict that contains the model's function call.
    Steps with observation in {"system", "message"} are skipped.
    The ``thought`` for each step is extracted from the model response content.
    """
    try:
        from mapPhase import get_phase
    except ImportError:
        def get_phase(*_args, **_kwargs):
            return "general"

    builder          = GraphBuilder()
    prev_phases_list: list[str] = []
    prev_thought:    str = ""
    prev_step_first_node: str | None = None
    step_idx = 0

    for step in traj_data.get("history", []):
        obs_type = step.get("observation")
        if obs_type in ("system", "message") or obs_type is None:
            continue

        # Extract thought from the model response (assistant message content)
        thought = ""
        tool_call_meta = step.get("tool_call_metadata", {})
        model_response = tool_call_meta.get("model_response", {})
        choices = model_response.get("choices", [])
        if choices:
            msg = choices[0].get("message", {})
            content = msg.get("content") or ""
            if isinstance(content, str):
                thought = content
            elif isinstance(content, list):
                # content blocks: pull out text blocks
                thought = "\n".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )

        observation = step.get("content", "") or ""

        thought_len_raw   = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)

        # Gather tool calls from the metadata (same logic as generatejson.py)
        tool_calls = model_response.get("choices", [])
        if not tool_calls and tool_call_meta:
            tool_calls = [tool_call_meta]

        parsed_commands = []
        for call in tool_calls:
            function_call = None
            if isinstance(call, dict):
                if "function" in call:
                    function_call = call["function"]
                else:
                    msg_obj = call.get("message", {})
                    for tc in (msg_obj.get("tool_calls") or []):
                        if "function" in tc:
                            function_call = tc["function"]
                            break

            if not function_call:
                continue

            tool_name = function_call.get("name", "")
            args_raw  = function_call.get("arguments", "{}")
            try:
                args_loaded = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
            except (json.JSONDecodeError, TypeError):
                args_loaded = {}

            if tool_name == "execute_bash":
                cmd_str = args_loaded.get("command", "").strip()
                cmds = cmd_parser.parse(cmd_str) if cmd_str else []
                if cmds:
                    parsed_commands.extend(cmds)
            else:
                subcommand = args_loaded.pop("command", None)
                parsed_commands.append({
                    "tool":       tool_name,
                    "subcommand": subcommand,
                    "args":       args_loaded,
                    "flags":      {},
                    "command":    "",
                })

        if not parsed_commands:
            step_idx += 1
            continue

        # Optional cd filtering
        has_cd = False
        if filter_cd and len(parsed_commands) > 1:
            first = parsed_commands[0]
            if (first.get("command") or "").strip().lower() == "cd":
                has_cd          = True
                parsed_commands = parsed_commands[1:]

        is_first_in_step  = True
        node_keys_in_step = []
        step_first_node: str | None = None

        for parsed in parsed_commands:
            tool       = (parsed.get("tool")       or "").strip()
            subcommand = (parsed.get("subcommand") or "").strip()
            command    = (parsed.get("command")    or "").strip()
            args       = parsed.get("args",  {})
            flags      = parsed.get("flags", {})

            if tool == "think":
                # When unique_think is on, use the thought text in the node
                # signature so each distinct thought gets its own node.
                # When off, all think steps collapse into one node (old behaviour).
                think_args = {"_thought": thought} if unique_think else {}
                node_key = builder.add_or_update_node(
                    node_label     = "think",
                    args           = think_args,
                    flags          = {},
                    phase          = "general",
                    step_idx       = step_idx,
                    tool           = None,
                    command        = None,
                    subcommand     = None,
                    thought_length = thought_len_raw,
                    has_cd         = False,
                )
                builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
                builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
                _accumulate_observation(builder.G.nodes[node_key], observation)
                _accumulate_step_data(builder.G.nodes[node_key], step_idx,
                                      thought, "", observation)
                builder.add_execution_edge(
                    node_key, step_idx,
                    is_first_in_step=is_first_in_step,
                    thought_length_raw=thought_len_raw if is_first_in_step else 0,
                    thought_length_clean=thought_len_clean if is_first_in_step else 0,
                )
                if is_first_in_step:
                    _mark_thought_continuation(
                        builder.G, prev_step_first_node, node_key,
                        prev_thought, thought,
                    )
                    if step_first_node is None:
                        step_first_node = node_key
                builder.update_previous_node(node_key)
                prev_phases_list.append("general")
                builder.prev_phases.add("general")
                node_keys_in_step.append(node_key)
                is_first_in_step = False
                continue

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command or obs_type or "action"

            phase = get_phase(tool, subcommand, command, args, prev_phases_list)

            outcome = check_command_outcome(
                command=command, observation=observation,
                tool=tool, subcommand=subcommand,
                args=args if isinstance(args, dict) else {},
            )
            edit_status = check_edit_status(tool, subcommand, args, observation)
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status
            if outcome and isinstance(args, dict):
                args.setdefault("command_outcome", outcome)

            node_key = builder.add_or_update_node(
                node_label     = node_label,
                args           = args,
                flags          = flags,
                phase          = phase,
                step_idx       = step_idx,
                tool           = tool,
                command        = command,
                subcommand     = subcommand,
                thought_length = thought_len_raw,
                has_cd         = has_cd,
            )

            builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            _accumulate_step_data(builder.G.nodes[node_key], step_idx,
                                  thought, f"{tool} {subcommand}".strip(), observation)

            node_keys_in_step.append(node_key)
            if step_first_node is None:
                step_first_node = node_key

            builder.add_execution_edge(
                node_key, step_idx,
                is_first_in_step=is_first_in_step,
                thought_length_raw=thought_len_raw if is_first_in_step else 0,
                thought_length_clean=thought_len_clean if is_first_in_step else 0,
            )

            if is_first_in_step:
                _mark_thought_continuation(
                    builder.G, prev_step_first_node, node_key,
                    prev_thought, thought,
                )

            builder.update_previous_node(node_key)
            prev_phases_list.append(phase)
            builder.prev_phases.add(phase)
            is_first_in_step = False

        if node_keys_in_step:
            last_node = node_keys_in_step[-1]
            _accumulate_observation(builder.G.nodes[last_node], observation)

        prev_thought = thought
        prev_step_first_node = step_first_node
        step_idx += 1

    # Post-processing
    build_hierarchical_edges(builder.G, builder.localization_nodes)

    resolution_status = determine_resolution_status(instance_id, eval_report_path)
    builder.G.graph["resolution_status"] = resolution_status
    builder.G.graph["instance_name"]     = instance_id

    try:
        from buildGraph import difficulty_lookup
        builder.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")
    except Exception:
        builder.G.graph["debug_difficulty"] = "unknown"

    return builder.G


# ── Graph construction ──────────────────────────────────────────────────────

def build_graph(traj_data: dict, instance_id: str,
                eval_report_path: str, cmd_parser,
                graphs_dir: Path | None = None,
                filter_cd: bool = True,
                agent_type: str = "sa",
                unique_think: bool = True):
    """Build and return a NetworkX MultiDiGraph from *traj_data*.

    The instance's tool config YAML is auto-discovered from:
        {graphs_dir}/{instance_id}/{instance_id}.config.yaml

    and loaded into a fresh per-request CommandParser so that the shared
    base parser (cmd_parser) is never mutated between concurrent requests.

    Args:
        traj_data:        Raw trajectory dict (from .traj JSON file).
        instance_id:      Instance identifier, e.g. 'astropy__astropy-7166'.
        eval_report_path: Path to the evaluation report JSON.
        cmd_parser:       Base CommandParser instance (tool_map may be empty).
        graphs_dir:       Root directory containing per-instance sub-folders.
                          Required for config YAML discovery; if None the base
                          parser is used as-is.
        filter_cd:        Strip leading ``cd`` commands and mark nodes with ▲.

    Raises:
        ValueError: if cmd_parser is None.
    """
    if cmd_parser is None:
        # commandParser was unavailable at startup.  Use a minimal no-op shim
        # so the server stays functional — actions will be stored as raw strings
        # rather than structured tool/command records.
        class _NullParser:
            def parse(self, action_str):
                cmd = action_str.strip().split()[0] if action_str.strip() else "action"
                return [{"tool": "", "subcommand": "", "command": cmd,
                         "args": {"_raw": action_str.strip()}, "flags": {}}]
        cmd_parser = _NullParser()

    # Dispatch to agent-specific builder
    if agent_type == "oh":
        return _build_graph_oh(traj_data, instance_id, eval_report_path,
                               cmd_parser, filter_cd, unique_think=unique_think)

    # Build a per-instance parser loaded with this trajectory's config YAML
    if graphs_dir is not None:
        instance_parser = _make_parser_for_instance(cmd_parser, graphs_dir, instance_id)
    else:
        instance_parser = cmd_parser

    try:
        from mapPhase import get_phase
    except ImportError:
        def get_phase(*_args, **_kwargs):
            return "general"

    builder    = GraphBuilder()
    trajectory = traj_data.get("trajectory", [])
    prev_phases_list: list[str] = []

    # For thought-continuation detection: track the thought text of each step
    # and the first node_key produced by that step.
    prev_thought: str = ""           # thought text of the previous step
    prev_step_first_node: str | None = None  # first node key of the previous step

    for step_idx, step in enumerate(trajectory):
        action_str  = step.get("action", "")
        thought     = step.get("thought", "") or ""
        observation = step.get("observation", "") or ""

        # Compute both thought lengths (for user-controlled switch)
        thought_len_raw   = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)

        # ── Pure-think steps (blank action) ────────────────────────────
        if not action_str.strip():
            # When unique_think is on, include thought text in the signature
            # so each distinct thought gets its own node instead of collapsing.
            think_args = {"_thought": thought} if unique_think else {"thought_len": thought_len_raw}
            node_key = builder.add_or_update_node(
                node_label         = "think",
                args               = think_args,
                flags              = {},
                phase              = "general",
                step_idx           = step_idx,
                tool               = None,
                command            = None,
                subcommand         = None,
                thought_length     = thought_len_raw,
                has_cd             = False,
            )
            builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            _accumulate_observation(builder.G.nodes[node_key], observation)
            _accumulate_step_data(builder.G.nodes[node_key], step_idx,
                                  thought, action_str, observation)

            builder.add_execution_edge(
                node_key, step_idx,
                is_first_in_step=True,
                thought_length_raw=thought_len_raw,
                thought_length_clean=thought_len_clean,
            )
            # Mark edge as thought-continuation if applicable
            _mark_thought_continuation(
                builder.G, prev_step_first_node, node_key,
                prev_thought, thought,
            )

            builder.update_previous_node(node_key)
            prev_phases_list.append("general")
            builder.prev_phases.add("general")
            prev_thought = thought
            prev_step_first_node = node_key
            continue

        # ── Parse action string ────────────────────────────────────────
        parsed_commands = instance_parser.parse(action_str)

        if not parsed_commands:
            continue

        # ── Optional cd filtering ──────────────────────────────────────
        has_cd = False
        if filter_cd and len(parsed_commands) > 1:
            first = parsed_commands[0]
            if (first.get("command") or "").strip().lower() == "cd":
                has_cd          = True
                parsed_commands = parsed_commands[1:]

        # ── Create nodes / edges ───────────────────────────────────────
        is_first_in_step  = True
        node_keys_in_step = []
        step_first_node: str | None = None

        for parsed in parsed_commands:
            tool       = (parsed.get("tool")       or "").strip()
            subcommand = (parsed.get("subcommand") or "").strip()
            command    = (parsed.get("command")    or "").strip()
            args       = parsed.get("args",  {})
            flags      = parsed.get("flags", {})

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, prev_phases_list)

            outcome = check_command_outcome(
                command=command, observation=observation,
                tool=tool, subcommand=subcommand,
                args=args if isinstance(args, dict) else {},
            )
            edit_status = check_edit_status(tool, subcommand, args, observation)
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status
            if outcome and isinstance(args, dict):
                args.setdefault("command_outcome", outcome)

            node_key = builder.add_or_update_node(
                node_label     = node_label,
                args           = args,
                flags          = flags,
                phase          = phase,
                step_idx       = step_idx,
                tool           = tool,
                command        = command,
                subcommand     = subcommand,
                thought_length = thought_len_raw,
                has_cd         = has_cd,
            )

            # Store both thought lengths on node
            builder.G.nodes[node_key]["thought_len_raw"]   = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            _accumulate_step_data(builder.G.nodes[node_key], step_idx,
                                  thought, action_str, observation)

            node_keys_in_step.append(node_key)
            if step_first_node is None:
                step_first_node = node_key

            # First edge in each step carries thought; subsequent intra-step edges carry 0
            builder.add_execution_edge(
                node_key, step_idx,
                is_first_in_step=is_first_in_step,
                thought_length_raw=thought_len_raw if is_first_in_step else 0,
                thought_length_clean=thought_len_clean if is_first_in_step else 0,
            )

            # Mark the first edge of this step as thought-continuation if applicable
            if is_first_in_step:
                _mark_thought_continuation(
                    builder.G, prev_step_first_node, node_key,
                    prev_thought, thought,
                )

            builder.update_previous_node(node_key)
            prev_phases_list.append(phase)
            builder.prev_phases.add(phase)

            is_first_in_step = False

        # ── Mark last node of this step with observation info ─────────
        if node_keys_in_step:
            last_node = node_keys_in_step[-1]
            _accumulate_observation(builder.G.nodes[last_node], observation)

        prev_thought = thought
        prev_step_first_node = step_first_node

    # ── Post-processing ────────────────────────────────────────────────
    build_hierarchical_edges(builder.G, builder.localization_nodes)

    resolution_status = determine_resolution_status(instance_id, eval_report_path)
    builder.G.graph["resolution_status"] = resolution_status
    builder.G.graph["instance_name"]     = instance_id

    try:
        from buildGraph import difficulty_lookup
        builder.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")
    except Exception:
        builder.G.graph["debug_difficulty"] = "unknown"

    return builder.G