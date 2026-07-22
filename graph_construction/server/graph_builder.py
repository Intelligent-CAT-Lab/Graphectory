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
    GraphBuilder,
    determine_resolution_status,
    check_edit_status,
    compute_thought_length_raw,
    compute_thought_length_clean,
    detect_observation_outcome,
    build_hierarchical_edges,
)

# ── Test-outcome helpers ─────────────────────────────────────────────────────

RE_PYTEST_FAIL  = re.compile(r"\b(\d+)\s+failed\b",  re.IGNORECASE)
RE_PYTEST_ERROR = re.compile(r"\b(\d+)\s+errors?\b", re.IGNORECASE)
RE_PYTEST_PASS  = re.compile(r"\b(\d+)\s+passed\b",  re.IGNORECASE)
EXCEPTION_SIGNS = ["Traceback (most recent call last):"]


def check_command_outcome(observation: str, tool: str = None, subcommand: str = None,
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

    # A successful view can contain arbitrary source code, including exception
    # names or test-result strings.  Only its explicit editor validation errors
    # should affect the rendered outcome.
    if tool == "str_replace_editor" and subcommand == "view":
        return None

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


# --- Agent-format detection -------------------------------------------------

KIMI_TRAJECTORY_FORMAT = "kimi-code-wire-1"


def _read_jsonl(path: Path):
    """Yield valid JSON objects from *path*, tolerating a partial final line."""
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def _is_kimi_wire_file(path: Path) -> bool:
    """Return whether *path* looks like a current Kimi Code wire stream."""
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        return False
    try:
        for index, record in enumerate(_read_jsonl(path)):
            record_type = record.get("type")
            if record_type in {
                "config.update",
                "turn.prompt",
                "context.append_loop_event",
                "context.append_message",
            }:
                return True
            if record_type == "metadata" and "protocol_version" in record:
                return True
            if index >= 20:
                break
    except OSError:
        return False
    return False


def _kimi_wire_files(source: Path) -> list[Path]:
    """Find main-agent Kimi Code wire streams below a file or directory."""
    if source.is_file():
        return [source] if _is_kimi_wire_file(source) else []

    wires = []
    for path in source.rglob("wire.jsonl"):
        # Current Kimi Code stores each session's primary stream here. Ignore
        # agents/<subagentId>/wire.jsonl so one session remains one trajectory.
        if path.parent.name == "main" and path.parent.parent.name == "agents":
            wires.append(path)
    return sorted(wires)


def _kimi_session_dir(wire_path: Path) -> Path:
    if wire_path.parent.name == "main" and wire_path.parent.parent.name == "agents":
        return wire_path.parent.parent.parent
    return wire_path.parent


def _load_kimi_state(wire_path: Path) -> dict:
    state_path = _kimi_session_dir(wire_path) / "state.json"
    if not state_path.is_file():
        return {}
    try:
        with open(state_path, encoding="utf-8", errors="replace") as stream:
            value = json.load(stream)
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def detect_agent_type(source: Path) -> str:
    """Infer the trajectory format represented by *source*."""
    if source.is_file():
        if _is_kimi_wire_file(source):
            return "kimi"
        if source.suffix.lower() == ".jsonl":
            return "oh"
        return "sa"

    if any(source.rglob("agents/main/wire.jsonl")):
        return "kimi"
    if any(source.rglob("*.traj.json")):
        return "msa"
    return "sa"


# ── Directory scanning ──────────────────────────────────────────────────────

def scan_trajectories(graphs_dir: Path,
                      eval_report_path: str | None = None,
                      agent_type: str = "sa") -> list[dict]:
    """Return a sorted list of trajectory metadata dicts.

    Each dict has: instance_id, status, difficulty, step_count.

    For SWE-agent (agent_type='sa'), graphs_dir is a directory tree of .traj files.
    For OpenHands (agent_type='oh'), graphs_dir is a path to an output.jsonl file.
    For mini-swe-agent (agent_type='msa'), graphs_dir is a directory tree of .traj.json files.
    For Kimi Code (agent_type='kimi'), graphs_dir contains session wire.jsonl files.
    """
    resolved_set:   set[str] = set()
    unresolved_set: set[str] = set()
    if eval_report_path:
        try:
            with open(eval_report_path, encoding="utf-8", errors="replace") as f:
                report = json.load(f)
            resolved_set   = set(report.get("resolved_ids",   []))
            unresolved_set = set(report.get("unresolved_ids", []))
        except Exception:
            pass

    results = []

    if agent_type == "kimi":
        for wire_path in _kimi_wire_files(graphs_dir):
            session_dir = _kimi_session_dir(wire_path)
            instance_id = session_dir.name or wire_path.stem
            state = _load_kimi_state(wire_path)
            title = str(state.get("title") or "").strip()
            step_count = 0
            try:
                for record in _read_jsonl(wire_path):
                    if record.get("type") != "context.append_loop_event":
                        continue
                    event = record.get("event") or {}
                    if event.get("type") == "step.begin":
                        step_count += 1
            except OSError:
                continue

            results.append({
                "instance_id": instance_id,
                "display_name": title or instance_id,
                "status": "none",
                "difficulty": "unknown",
                "step_count": step_count,
            })

        results.sort(key=lambda item: (item["display_name"].lower(), item["instance_id"]))
        return results

    if agent_type == "oh":
        # OpenHands: graphs_dir is actually the output.jsonl file
        jsonl_path = graphs_dir
        if not jsonl_path.is_file():
            return results
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
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
                elif eval_report_path:
                    status = "unsubmitted"
                else:
                    status = "none"

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

    if agent_type == "msa":
        # mini-swe-agent: directory tree of .traj.json files
        # Each instance lives at: graphs_dir/{instance_id}/{instance_id}.traj.json
        for traj_file in sorted(graphs_dir.rglob("*.traj.json")):
            instance_id = traj_file.name[: -len(".traj.json")]

            if instance_id in resolved_set:
                status = "resolved"
            elif instance_id in unresolved_set:
                status = "unresolved"
            elif not eval_report_path:
                status = "none"
            else:
                status = "unsubmitted"

            step_count = 0
            try:
                with open(traj_file, encoding="utf-8", errors="replace") as f:
                    traj = json.load(f)
                if traj.get("trajectory_format") == "mini-swe-agent-1":
                    # v1.0: plain text messages — count assistant turns with content
                    step_count = sum(
                        1 for m in traj.get("messages", [])
                        if m.get("role") == "assistant"
                        and isinstance(m.get("content"), str)
                        and m["content"].strip()
                    )
                else:
                    # Default: count assistant-response messages (those with an 'output' list)
                    step_count = sum(
                        1 for m in traj.get("messages", [])
                        if isinstance(m.get("output"), list)
                    )
            except Exception:
                pass

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
        elif not eval_report_path:
            status = "none"
        else:
            status = "unsubmitted"
            json_file = traj_file.with_suffix(".json")
            if json_file.exists():
                try:
                    with open(json_file, encoding="utf-8", errors="replace") as f:
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
                with open(json_file, encoding="utf-8", errors="replace") as f:
                    meta = json.load(f)
                difficulty = meta.get("graph", {}).get("debug_difficulty", "unknown")
            except Exception:
                pass

        step_count = 0
        try:
            with open(traj_file, encoding="utf-8", errors="replace") as f:
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
    For mini-swe-agent, searches for a matching .traj.json file under graphs_dir.
    For Kimi Code, loads the main agent's wire.jsonl event stream.

    Raises FileNotFoundError if the trajectory cannot be found.
    """
    if agent_type == "kimi":
        for wire_path in _kimi_wire_files(graphs_dir):
            session_dir = _kimi_session_dir(wire_path)
            if (session_dir.name or wire_path.stem) != instance_id:
                continue
            return {
                "trajectory_format": KIMI_TRAJECTORY_FORMAT,
                "wire_path": str(wire_path),
                "state": _load_kimi_state(wire_path),
                "records": list(_read_jsonl(wire_path)),
            }
        raise FileNotFoundError(
            f"No Kimi Code session '{instance_id}' found under {graphs_dir}"
        )

    if agent_type == "oh":
        jsonl_path = graphs_dir
        if not jsonl_path.is_file():
            raise FileNotFoundError(
                f"OpenHands output.jsonl not found: {jsonl_path}"
            )
        with open(jsonl_path, encoding="utf-8", errors="replace") as f:
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

    if agent_type == "msa":
        # Canonical path: {graphs_dir}/{instance_id}/{instance_id}.traj.json
        canonical = graphs_dir / instance_id / f"{instance_id}.traj.json"
        if canonical.exists():
            with open(canonical, encoding="utf-8", errors="replace") as f:
                return json.load(f)
        # Fallback: recursive glob
        for traj_file in graphs_dir.rglob(f"{instance_id}.traj.json"):
            with open(traj_file, encoding="utf-8", errors="replace") as f:
                return json.load(f)
        raise FileNotFoundError(
            f"No .traj.json file found for '{instance_id}' under {graphs_dir}"
        )

    # SWE-agent: search for .traj file
    for traj_file in graphs_dir.rglob(f"{instance_id}.traj"):
        with open(traj_file, encoding="utf-8", errors="replace") as f:
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
    outcome = detect_observation_outcome(
        observation,
        tool=node_data.get("tool"),
        subcommand=node_data.get("subcommand"),
        args=node_data.get("args"),
    )

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


# ── Shell no-op filter ───────────────────────────────────────────────────────

def _is_shell_noop(parsed: dict) -> bool:
    """Return True when *parsed* represents a bare shell boolean constant.

    The bash commands ``true`` and ``false`` are used purely for short-circuit
    evaluation (e.g. ``|| true`` to suppress a non-zero exit code).  They carry
    no semantic meaning as agent actions and must NOT become graph nodes.

    The check is intentionally narrow: we only suppress the command when
      • there is no tool name  (it's a raw shell command, not a known tool)
      • the command token is exactly "true" or "false"
      • there are no positional args and no flags
    If the agent ever uses ``true`` as a genuine argument to another command it
    will appear as part of that command's args dict, not as a standalone parsed
    entry, so this guard will not fire.
    """
    if parsed.get("tool"):
        return False  # has a tool name → never a bare shell no-op
    command = (parsed.get("command") or "").strip().lower()
    if command not in ("true", "false"):
        return False
    # Only suppress when there are genuinely no args and no flags
    args  = parsed.get("args",  [])
    flags = parsed.get("flags", {})
    args_empty  = (not args)  or (isinstance(args,  (list, dict)) and len(args)  == 0)
    flags_empty = (not flags) or (isinstance(flags, dict)         and len(flags) == 0)
    return args_empty and flags_empty


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
        action_str_parts = []   # collect per-call action text for the sidebar
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
                # Action shown in sidebar is the raw bash command
                action_str_parts.append(cmd_str if cmd_str else tool_name)
                cmds = cmd_parser.parse(cmd_str) if cmd_str else []
                if cmds:
                    parsed_commands.extend(cmds)
                else:
                    parsed_commands.append({
                        "tool":       "",
                        "subcommand": "",
                        "args":       {"_raw": cmd_str} if cmd_str else {},
                        "flags":      {},
                        "command":    cmd_str or tool_name,
                    })
            else:
                # Capture the raw args string before any mutation for faithful sidebar display
                raw_for_display = args_raw if isinstance(args_raw, str) else json.dumps(args_raw)
                subcommand = args_loaded.pop("command", None)
                # Action shown in sidebar: tool [subcommand] then the verbatim args JSON,
                # so the user sees exactly what the model called (path, file_text, old_str, etc.)
                header = tool_name + (" " + subcommand if subcommand else "")
                action_str_parts.append(f"{header}\n{raw_for_display}")
                parsed_commands.append({
                    "tool":       tool_name,
                    "subcommand": subcommand,
                    "args":       args_loaded,
                    "flags":      {},
                    "command":    "",
                })

        # Full action text shown verbatim in the sidebar
        action_str = "\n".join(action_str_parts)

        if not parsed_commands:
            # No tool calls found — still create a node so thought/observation
            # are visible in the sidebar rather than being silently dropped.
            parsed_commands = [{
                "tool":       "",
                "subcommand": "",
                "args":       {},
                "flags":      {},
                "command":    "",
            }]

        # Optional cd filtering
        has_cd = False
        if filter_cd and len(parsed_commands) > 1:
            first = parsed_commands[0]
            if (first.get("command") or "").strip().lower() == "cd":
                has_cd          = True
                parsed_commands = parsed_commands[1:]

       
        parsed_commands = [p for p in parsed_commands if not _is_shell_noop(p)]

        if not parsed_commands:
            # Every command in this step was a noop — skip the step entirely
            # so no orphaned state is left behind.
            step_idx += 1
            continue

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
                think_args = {"_thought": thought, "_action": action_str} if unique_think else {}
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
                                      thought, action_str, observation)
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

            phase = get_phase(tool, subcommand, command, args, prev_phases_list, flags)

            outcome = check_command_outcome(
                observation=observation,
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
                                  thought, action_str, observation)

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

    resolution_status = determine_resolution_status(instance_id, eval_report_path) \
        if eval_report_path else "none"
    builder.G.graph["resolution_status"] = resolution_status
    builder.G.graph["instance_name"]     = instance_id

    try:
        from buildGraph import difficulty_lookup
        builder.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")
    except Exception:
        builder.G.graph["debug_difficulty"] = "unknown"

    return builder.G


# ── mini-swe-agent v1.0 graph construction ──────────────────────────────────

def _kimi_output_text(value) -> str:
    """Flatten Kimi tool output/content blocks into sidebar-friendly text."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for part in value:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                text = part.get("text") or part.get("think")
                if isinstance(text, str):
                    chunks.append(text)
        return "\n".join(chunks)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def iter_kimi_steps(traj_data: dict) -> list[dict]:
    """Fold Kimi Code wire records into ordered model steps and tool calls."""
    ordered_steps: list[dict] = []
    steps_by_uuid: dict[str, dict] = {}
    calls_by_id: dict[str, dict] = {}
    current_step: dict | None = None

    for record in traj_data.get("records", []):
        if record.get("type") != "context.append_loop_event":
            continue
        event = record.get("event") or {}
        event_type = event.get("type")

        if event_type == "step.begin":
            step = {
                "source_step": event.get("step"),
                "uuid": str(event.get("uuid") or len(ordered_steps)),
                "thought_parts": [],
                "calls": [],
            }
            ordered_steps.append(step)
            steps_by_uuid[step["uuid"]] = step
            current_step = step
            continue

        step_uuid = str(event.get("stepUuid") or "")
        step = steps_by_uuid.get(step_uuid) or current_step

        if event_type == "content.part" and step is not None:
            part = event.get("part") or {}
            part_type = str(part.get("type") or "")
            if part_type == "think":
                text = part.get("think") or part.get("text") or ""
            elif part_type == "text":
                text = part.get("text") or ""
            else:
                text = ""
            if isinstance(text, str) and text:
                step["thought_parts"].append(text)
            continue

        if event_type == "tool.call" and step is not None:
            args = event.get("args")
            if not isinstance(args, dict):
                args = {"_raw": args} if args is not None else {}
            call = {
                "id": str(event.get("toolCallId") or event.get("uuid") or ""),
                "name": str(event.get("name") or "tool"),
                "args": args,
                "description": str(event.get("description") or ""),
                "observation": "",
                "is_error": None,
            }
            step["calls"].append(call)
            if call["id"]:
                calls_by_id[call["id"]] = call
            continue

        if event_type == "tool.result":
            call_id = str(event.get("toolCallId") or "")
            call = calls_by_id.get(call_id)
            if call is None:
                continue
            result = event.get("result") or {}
            output = _kimi_output_text(result.get("output"))
            message = result.get("message")
            if isinstance(message, str) and message and message not in output:
                output = f"{output}\n{message}".strip()
            call["observation"] = output
            error_value = result.get("isError", result.get("is_error"))
            if isinstance(error_value, bool):
                call["is_error"] = error_value

    for step in ordered_steps:
        step["thought"] = "\n".join(step.pop("thought_parts", []))
    return ordered_steps


def kimi_tool_phase(tool_name: str, args: dict, prev_phases: list[str]) -> str:
    """Map Kimi Code's native tools onto the Graphectory phase taxonomy."""
    try:
        from mapPhase import get_phase
    except ImportError:
        return "general"

    name = (tool_name or "").lower()
    phase_args = dict(args or {})
    for key in ("path", "file", "file_path", "filename"):
        value = phase_args.get(key)
        if isinstance(value, str) and value and not value.startswith(("/", "./", "../", "~")):
            # mapPhase's test-directory hint is slash-delimited. Preserve the
            # original node arguments while making relative tests/... paths
            # classify the same way as absolute /tests/... paths.
            phase_args[key] = f"./{value}"
    if name in {"read", "readfile", "readmediafile"}:
        return get_phase("str_replace_editor", "view", "", phase_args, prev_phases, {})
    if name in {"write", "writefile"}:
        return get_phase("str_replace_editor", "create", "", phase_args, prev_phases, {})
    if name in {"edit", "strreplace", "str_replace"}:
        return get_phase("str_replace_editor", "str_replace", "", phase_args, prev_phases, {})
    if name == "grep":
        return get_phase("", "", "grep", phase_args, prev_phases, {})
    if name == "glob":
        return get_phase("", "", "find", phase_args, prev_phases, {})
    return "general"


def _build_graph_kimi(traj_data: dict, instance_id: str,
                      eval_report_path: str, cmd_parser,
                      filter_cd: bool = True,
                      unique_think: bool = True):
    """Build a graph from a current Kimi Code agents/main/wire.jsonl stream."""
    try:
        from mapPhase import get_phase
    except ImportError:
        def get_phase(*_args, **_kwargs):
            return "general"

    builder = GraphBuilder()
    prev_phases_list: list[str] = []
    prev_thought = ""
    prev_step_first_node: str | None = None

    for step_idx, step in enumerate(iter_kimi_steps(traj_data)):
        thought = step.get("thought", "") or ""
        calls = step.get("calls", [])
        thought_len_raw = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)

        action_parts = []
        observation_parts = []
        action_entries = []

        for call in calls:
            tool_name = str(call.get("name") or "tool")
            args = dict(call.get("args") or {})
            observation = str(call.get("observation") or "")
            is_error = call.get("is_error")

            if tool_name.lower() in {"bash", "shell", "execute_bash"}:
                command_text = str(args.get("command") or "").strip()
                action_parts.append(command_text or tool_name)
                parsed_commands = cmd_parser.parse(command_text) if command_text else []
                if not parsed_commands:
                    parsed_commands = [{
                        "tool": "",
                        "subcommand": "",
                        "args": {"_raw": command_text} if command_text else args,
                        "flags": {},
                        "command": command_text or tool_name,
                    }]

                has_cd = False
                if filter_cd and len(parsed_commands) > 1:
                    first = parsed_commands[0]
                    if (first.get("command") or "").strip().lower() == "cd":
                        has_cd = True
                        parsed_commands = parsed_commands[1:]

                for parsed in parsed_commands:
                    if _is_shell_noop(parsed):
                        continue
                    action_entries.append({
                        "parsed": parsed,
                        "native_tool": tool_name,
                        "observation": observation,
                        "is_error": is_error,
                        "has_cd": has_cd,
                        "phase": None,
                    })
            else:
                action_parts.append(
                    f"{tool_name}\n{json.dumps(args, ensure_ascii=False, indent=2, default=str)}"
                )
                action_entries.append({
                    "parsed": {
                        "tool": tool_name,
                        "subcommand": "",
                        "args": args,
                        "flags": {},
                        "command": "",
                    },
                    "native_tool": tool_name,
                    "observation": observation,
                    "is_error": is_error,
                    "has_cd": False,
                    "phase": "kimi",
                })

            if observation:
                observation_parts.append(f"{tool_name}: {observation}")

        action_str = "\n\n".join(action_parts)
        step_observation = "\n\n".join(observation_parts)

        if not action_entries:
            if not thought:
                continue
            think_args = {"_thought": thought} if unique_think else {}
            node_key = builder.add_or_update_node(
                node_label="think", args=think_args, flags={}, phase="general",
                step_idx=step_idx, tool=None, command=None, subcommand=None,
                thought_length=thought_len_raw, has_cd=False,
            )
            builder.G.nodes[node_key]["thought_len_raw"] = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            _accumulate_step_data(
                builder.G.nodes[node_key], step_idx, thought, "", step_observation,
            )
            builder.add_execution_edge(
                node_key, step_idx, is_first_in_step=True,
                thought_length_raw=thought_len_raw,
                thought_length_clean=thought_len_clean,
            )
            _mark_thought_continuation(
                builder.G, prev_step_first_node, node_key, prev_thought, thought,
            )
            builder.update_previous_node(node_key)
            prev_phases_list.append("general")
            builder.prev_phases.add("general")
            prev_step_first_node = node_key
            prev_thought = thought
            continue

        is_first_in_step = True
        step_first_node: str | None = None

        for entry in action_entries:
            parsed = entry["parsed"]
            tool = str(parsed.get("tool") or "").strip()
            subcommand = str(parsed.get("subcommand") or "").strip()
            command = str(parsed.get("command") or "").strip()
            raw_args = parsed.get("args", {})
            args = dict(raw_args) if isinstance(raw_args, dict) else raw_args
            flags = parsed.get("flags") or {}
            observation = entry["observation"]

            if entry["phase"] == "kimi":
                phase = kimi_tool_phase(entry["native_tool"], args, prev_phases_list)
            else:
                phase = entry["phase"] or get_phase(
                    tool, subcommand, command, args, prev_phases_list, flags,
                )
            node_label = f"{tool}: {subcommand}" if tool and subcommand else (tool or command or entry["native_tool"])

            explicit_error = entry["is_error"]
            if explicit_error is True:
                outcome = "failure"
            elif explicit_error is False and entry["native_tool"].lower() in {
                "write", "writefile", "edit", "strreplace", "str_replace"
            }:
                outcome = "success"
            else:
                outcome = check_command_outcome(
                    observation=observation,
                    tool=tool,
                    subcommand=subcommand,
                    args=args if isinstance(args, dict) else {},
                )
            if outcome and isinstance(args, dict):
                args.setdefault("command_outcome", outcome)

            node_key = builder.add_or_update_node(
                node_label=node_label, args=args, flags=flags, phase=phase,
                step_idx=step_idx, tool=tool, command=command,
                subcommand=subcommand, thought_length=thought_len_raw,
                has_cd=entry["has_cd"],
            )
            builder.G.nodes[node_key]["thought_len_raw"] = thought_len_raw
            builder.G.nodes[node_key]["thought_len_clean"] = thought_len_clean
            if outcome:
                builder.G.nodes[node_key]["command_outcome"] = outcome
            _accumulate_observation(builder.G.nodes[node_key], observation)
            _accumulate_step_data(
                builder.G.nodes[node_key], step_idx, thought, action_str,
                step_observation,
            )
            builder.add_execution_edge(
                node_key, step_idx, is_first_in_step=is_first_in_step,
                thought_length_raw=thought_len_raw if is_first_in_step else 0,
                thought_length_clean=thought_len_clean if is_first_in_step else 0,
            )
            if is_first_in_step:
                _mark_thought_continuation(
                    builder.G, prev_step_first_node, node_key, prev_thought, thought,
                )
                step_first_node = node_key

            builder.update_previous_node(node_key)
            prev_phases_list.append(phase)
            builder.prev_phases.add(phase)
            is_first_in_step = False

        prev_step_first_node = step_first_node
        prev_thought = thought

    build_hierarchical_edges(builder.G, builder.localization_nodes)
    resolution_status = determine_resolution_status(instance_id, eval_report_path) \
        if eval_report_path else "none"
    builder.G.graph["resolution_status"] = resolution_status
    builder.G.graph["debug_difficulty"] = "unknown"
    builder.G.graph["trajectory_format"] = KIMI_TRAJECTORY_FORMAT
    title = str((traj_data.get("state") or {}).get("title") or "").strip()
    builder.G.graph["instance_name"] = title or instance_id
    builder.G.graph["session_id"] = instance_id
    if title:
        builder.G.graph["session_title"] = title
    return builder.G


def _build_graph_msa_v1(traj_data: dict, instance_id: str,
                        eval_report_path: str, cmd_parser,
                        filter_cd: bool = True,
                        unique_think: bool = True):
    """Build a NetworkX MultiDiGraph from a mini-swe-agent v1.0 trajectory.

    v1.0 format: ``messages`` list of plain role/content pairs:
      - messages[0]  : system prompt (skipped)
      - messages[1]  : initial user problem (skipped)
      - messages[2i] : assistant — content is a plain string:
                         "THOUGHT: <reasoning>\\n\\n```bash\\n<command>\\n```"
      - messages[2i+1]: user — content is:
                         "<returncode>N</returncode>\\n<o>\\n<output>\\n</o>"

    This differs from the default MSA format which uses structured ``output``
    lists with typed blocks (message / function_call).
    """
    try:
        from mapPhase import get_phase
    except ImportError:
        def get_phase(*_args, **_kwargs):
            return "general"

    builder = GraphBuilder()
    prev_phases_list: list[str] = []
    prev_thought: str = ""
    prev_step_first_node: str | None = None
    step_idx = 0

    messages = traj_data.get("messages", [])
    i = 2  # skip system (0) and initial user problem (1)

    while i < len(messages):
        msg = messages[i]

        # Only process assistant turns
        if msg.get("role") != "assistant":
            i += 1
            continue

        content = msg.get("content", "")
        if not isinstance(content, str) or not content.strip():
            i += 2
            continue

        # ── Extract thought from THOUGHT: section ─────────────────────
        thought = ""
        thought_match = re.search(
            r'THOUGHT:\s*(.*?)(?=\n\n```|\n```|$)', content, re.DOTALL
        )
        if thought_match:
            thought = thought_match.group(1).strip()

        thought_len_raw   = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)

        # ── Extract bash command ───────────────────────────────────────
        action_str = ""
        bash_match = re.search(r'```bash\s*(.*?)```', content, re.DOTALL)
        if bash_match:
            action_str = bash_match.group(1).strip()

        # ── Extract observation from the following user message ────────
        observation = ""
        if i + 1 < len(messages):
            next_msg = messages[i + 1]
            if next_msg.get("role") == "user":
                next_content = next_msg.get("content", "")
                output_match = re.search(r'<output>(.*?)</output>', next_content, re.DOTALL)
                if output_match:
                    observation = output_match.group(1).strip()

        # ── Handle steps with no bash command as pure think ───────────
        if not action_str:
            think_args = ({"_thought": thought} if unique_think
                          else {"thought_len": thought_len_raw})
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
                is_first_in_step=True,
                thought_length_raw=thought_len_raw,
                thought_length_clean=thought_len_clean,
            )
            _mark_thought_continuation(
                builder.G, prev_step_first_node, node_key, prev_thought, thought,
            )
            builder.update_previous_node(node_key)
            prev_phases_list.append("general")
            builder.prev_phases.add("general")
            prev_thought = thought
            prev_step_first_node = node_key
            step_idx += 1
            i += 2
            continue

        # ── Parse action string ────────────────────────────────────────
        parsed_commands = cmd_parser.parse(action_str)
        if not parsed_commands:
            parsed_commands = [{
                "tool":       "",
                "subcommand": "",
                "command":    action_str.split()[0] if action_str.split() else "bash",
                "args":       {"_raw": action_str},
                "flags":      {},
            }]

        # Optional cd filtering
        has_cd = False
        if filter_cd and len(parsed_commands) > 1:
            first = parsed_commands[0]
            if (first.get("command") or "").strip().lower() == "cd":
                has_cd          = True
                parsed_commands = parsed_commands[1:]

        parsed_commands = [p for p in parsed_commands if not _is_shell_noop(p)]
        if not parsed_commands:
            step_idx += 1
            i += 2
            continue

        is_first_in_step  = True
        node_keys_in_step: list[str] = []
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
                node_label = command or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, prev_phases_list, flags)

            outcome = check_command_outcome(
                observation=observation,
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
                                  thought, action_str, observation)

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
                    builder.G, prev_step_first_node, node_key, prev_thought, thought,
                )

            builder.update_previous_node(node_key)
            prev_phases_list.append(phase)
            builder.prev_phases.add(phase)
            is_first_in_step = False

        if node_keys_in_step:
            _accumulate_observation(builder.G.nodes[node_keys_in_step[-1]], observation)

        prev_thought = thought
        prev_step_first_node = step_first_node
        step_idx += 1
        i += 2  # advance past this assistant message and the following user response

    # ── Post-processing ────────────────────────────────────────────────
    build_hierarchical_edges(builder.G, builder.localization_nodes)

    resolution_status = determine_resolution_status(instance_id, eval_report_path) \
        if eval_report_path else "none"
    builder.G.graph["resolution_status"] = resolution_status
    builder.G.graph["instance_name"]     = instance_id

    try:
        from buildGraph import difficulty_lookup
        builder.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")
    except Exception:
        builder.G.graph["debug_difficulty"] = "unknown"

    return builder.G


# ── mini-swe-agent graph construction ───────────────────────────────────────

def _build_graph_msa(traj_data: dict, instance_id: str,
                     eval_report_path: str, cmd_parser,
                     filter_cd: bool = True,
                     unique_think: bool = True):
    """Build a NetworkX MultiDiGraph from a mini-swe-agent trajectory.

    mini-swe-agent format: ``messages`` list where:
      - messages[0]  : system prompt (skipped)
      - messages[1]  : initial user message (skipped)
      - messages[i]  : assistant response — ``output`` is a list of blocks:
                           {type: "message", content: [{text: "..."}]}  → thought
                           {type: "function_call", arguments: "..."}    → action
      - messages[i+1]: tool result — observation in ``extra.raw_output``
                        or the ``output`` string field

    Processing mirrors _build_graph_oh but adapted to the MSA message schema.
    """
    try:
        from mapPhase import get_phase
    except ImportError:
        def get_phase(*_args, **_kwargs):
            return "general"

    # Detect mini-swe-agent v1.0 text-based format and dispatch immediately.
    # v1.0 uses plain role/content string messages instead of structured output
    # blocks, identified by the top-level "trajectory_format" field.
    if traj_data.get("trajectory_format") == "mini-swe-agent-1":
        return _build_graph_msa_v1(
            traj_data, instance_id, eval_report_path, cmd_parser,
            filter_cd=filter_cd, unique_think=unique_think,
        )

    builder = GraphBuilder()
    prev_phases_list: list[str] = []
    prev_thought: str = ""
    prev_step_first_node: str | None = None
    step_idx = 0

    messages = traj_data.get("messages", [])
    i = 2  # skip system (0) and initial user (1) messages

    while i < len(messages):
        msg = messages[i]

        # Only process assistant-response messages that carry an output list
        if not isinstance(msg.get("output"), list):
            i += 1
            continue

        output_blocks = msg["output"]

        # ── Extract thought ────────────────────────────────────────────
        thought = ""
        for block in output_blocks:
            if isinstance(block, dict) and block.get("type") == "message":
                content = block.get("content", [])
                if isinstance(content, list) and content:
                    thought = content[0].get("text", "") if isinstance(content[0], dict) else ""
                elif isinstance(content, str):
                    thought = content
                break

        thought_len_raw   = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)

        # ── Extract observation from the following tool-result message ─
        observation = ""
        if i + 1 < len(messages):
            next_msg = messages[i + 1]
            if isinstance(next_msg.get("output"), str):
                observation = next_msg["output"]
            else:
                observation = next_msg.get("extra", {}).get("raw_output", "")

        # ── Extract actions from function_call blocks ──────────────────
        raw_actions: list[str] = []
        for block in output_blocks:
            if isinstance(block, dict) and block.get("type") == "function_call":
                try:
                    args_json = json.loads(block.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args_json = {}
                cmd_str = args_json.get("command", "")
                if cmd_str:
                    raw_actions.append(cmd_str)

        # If there are no function-call actions this step is a pure-think step
        if not raw_actions:
            think_args = {"_thought": thought} if unique_think else {"thought_len": thought_len_raw}
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
                is_first_in_step=True,
                thought_length_raw=thought_len_raw,
                thought_length_clean=thought_len_clean,
            )
            _mark_thought_continuation(
                builder.G, prev_step_first_node, node_key, prev_thought, thought,
            )
            builder.update_previous_node(node_key)
            prev_phases_list.append("general")
            builder.prev_phases.add("general")
            prev_thought = thought
            prev_step_first_node = node_key
            step_idx += 1
            i += 2
            continue

        # ── Parse each action string and build nodes ───────────────────
        is_first_in_step  = True
        node_keys_in_step: list[str] = []
        step_first_node: str | None = None

        for action_str in raw_actions:
            parsed_commands = cmd_parser.parse(action_str)
            if not parsed_commands:
                parsed_commands = [{
                    "tool":       "",
                    "subcommand": "",
                    "command":    action_str.split()[0] if action_str.split() else "bash",
                    "args":       {"_raw": action_str},
                    "flags":      {},
                }]

            # Optional cd filtering
            has_cd = False
            if filter_cd and len(parsed_commands) > 1:
                first = parsed_commands[0]
                if (first.get("command") or "").strip().lower() == "cd":
                    has_cd          = True
                    parsed_commands = parsed_commands[1:]

            parsed_commands = [p for p in parsed_commands if not _is_shell_noop(p)]
            # If every command in this action was a noop, skip it entirely.
            if not parsed_commands:
                continue

            for parsed in parsed_commands:
                tool       = (parsed.get("tool")       or "").strip()
                subcommand = (parsed.get("subcommand") or "").strip()
                command    = (parsed.get("command")    or "").strip()
                args       = parsed.get("args",  {})
                flags      = parsed.get("flags", {})

                if tool:
                    node_label = f"{tool}: {subcommand}" if subcommand else tool
                else:
                    node_label = command or action_str.strip()

                phase = get_phase(tool, subcommand, command, args, prev_phases_list, flags)

                outcome = check_command_outcome(
                    observation=observation,
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
                                      thought, action_str, observation)

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
                        builder.G, prev_step_first_node, node_key, prev_thought, thought,
                    )

                builder.update_previous_node(node_key)
                prev_phases_list.append(phase)
                builder.prev_phases.add(phase)
                is_first_in_step = False

        # Mark the last node with observation info
        if node_keys_in_step:
            _accumulate_observation(builder.G.nodes[node_keys_in_step[-1]], observation)

        prev_thought = thought
        prev_step_first_node = step_first_node
        step_idx += 1
        i += 2  # advance past this assistant message and its tool-result reply

    # ── Post-processing ────────────────────────────────────────────────
    build_hierarchical_edges(builder.G, builder.localization_nodes)

    resolution_status = determine_resolution_status(instance_id, eval_report_path) \
        if eval_report_path else "none"
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
        raise ValueError(
            "cmd_parser must be a CommandParser instance. "
            "Pass a configured CommandParser from live_graph_server.setup_cmd_parser()."
        )

    # Dispatch to agent-specific builder
    if agent_type == "kimi":
        return _build_graph_kimi(traj_data, instance_id, eval_report_path,
                                 cmd_parser, filter_cd, unique_think=unique_think)

    if agent_type == "oh":
        return _build_graph_oh(traj_data, instance_id, eval_report_path,
                               cmd_parser, filter_cd, unique_think=unique_think)

    if agent_type == "msa":
        return _build_graph_msa(traj_data, instance_id, eval_report_path,
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

       
        parsed_commands = [p for p in parsed_commands if not _is_shell_noop(p)]
        if not parsed_commands:
            continue

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

            phase = get_phase(tool, subcommand, command, args, prev_phases_list, flags)

            outcome = check_command_outcome(
                observation=observation,
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

    resolution_status = determine_resolution_status(instance_id, eval_report_path) \
        if eval_report_path else "none"
    builder.G.graph["resolution_status"] = resolution_status
    builder.G.graph["instance_name"]     = instance_id

    try:
        from buildGraph import difficulty_lookup
        builder.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")
    except Exception:
        builder.G.graph["debug_difficulty"] = "unknown"

    return builder.G
