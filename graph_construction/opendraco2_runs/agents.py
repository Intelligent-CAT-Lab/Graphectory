"""
opendraco2_runs/agents.py

One Graphectory trajectory per agent of an OpenDraco2 run (and of a public
OpenDraco run, which ``opendraco_runs`` converts into the same ``RunLog``).

Each agent's activations, in ``local_turn`` order, become a step stream in the
Claude Code *wire* shape (``context.append_loop_event`` records), which the
upstream ``_build_graph_claude`` already turns into a graph. Reusing it means an
OpenDraco agent's graph gets the same node merging, phases, outcomes, sidebar
data and metrics as every other format, with no fork of the builder:

* a step opens at each ``model_call`` whose purpose is ``work`` or ``tool_turn``
  (the record is written when the call completes and before anything it caused);
* ``thinking`` and every ``message`` but the turn's last become the step's thought;
* ``tool_call`` + its ``tool_result`` become one action and its observation. Call
  ids come from the model and repeat across turns (``call_0``), so pairing happens
  inside one activation only and the ids handed to the builder are generated;
* the turn's last ``message`` is its final answer, a ``final_answer`` action as
  the Codex builder emits;
* the deliveries that caused a turn are prepended to its first thought as
  ``from <agent>: <value>`` ("from the parent run" for a fork's missing causes).

Tool names already line up with the Claude Code mapping (``bash`` through
commandParser and the shell rules, ``read`` → view, ``write`` → create, unknown
tools → the generic rule). The one exception is ``web_search``, which the generic
rule calls ``general``; it is relabelled ``localization`` after the build, which
is safe because only a prior ``patch`` changes how later actions classify.
"""

import re
from typing import Any, Optional

from server.graph_builder import CLAUDE_CODE_TRAJECTORY_FORMAT, build_graph

from opendraco2_runs.log import (
    MESSAGE_TYPES,
    MODEL_CALL,
    TOOL_CALL,
    TOOL_RESULT,
    Activation,
    RunLog,
)

# Model-call purposes that are the agent doing work (and so open a step). Routing
# and summary calls are bookkeeping: routing is shown on the run view instead.
STEP_PURPOSES = {"work", "tool_turn"}

# opendraco/tools/base.py: tool_failure()
_TOOL_FAILURE_PREFIX = "tool failure:"
# opendraco/tools/shell.py: _RESULT_TEMPLATE / _TIMEOUT_TEMPLATE
_EXIT_CODE_RE = re.compile(r"\Aexit code (-?\d+)\nstdout:\n")
_TIMEOUT_RE = re.compile(r"\Athe command did not finish within [\d.]+ seconds and was stopped")

# Tools whose phase the generic rule gets wrong for this framework.
PHASE_OVERRIDES = {"web_search": "localization"}


def result_is_error(tool_name: str, result: str) -> Optional[bool]:
    """Failure as the result text records it; None where the text does not say."""
    if result.startswith(_TOOL_FAILURE_PREFIX):
        return True
    if tool_name == "bash":
        if _TIMEOUT_RE.match(result):
            return True
        match = _EXIT_CODE_RE.match(result)
        if match:
            return int(match.group(1)) != 0
    return None


def _cause_lines(run: RunLog, act: Activation) -> list[str]:
    lines = []
    for uuid in act.new_causes:
        delivery = run.deliveries.get(uuid)
        if delivery is None:
            lines.append("from the parent run" if run.forked_from else "from a delivery not in this log")
        else:
            lines.append(f"from {delivery.get('from_agent') or '?'}: {delivery.get('value') or ''}")
    return lines


def activation_steps(run: RunLog, act: Activation) -> list[dict]:
    """The steps of one activation: ``{"thought": str, "calls": [call...]}``.

    OpenDraco streams a response as ``thinking``/``message`` chunks (a few
    characters each), then writes the call's ``model_call`` record, then the
    ``tool_call``s that response asked for. So a work/tool_turn ``model_call``
    closes one step: the chunks before it are its thought, the calls after it
    its actions. The last response that asked for no tool is the turn's answer.
    """
    steps: list[dict] = []
    # Streamed text since the last step closed, as [kind, text] runs: consecutive
    # chunks of one kind are one piece of text.
    buffer: list[list[str]] = []
    current: Optional[dict] = None
    open_calls: dict[str, list[dict]] = {}

    def close_step() -> dict:
        nonlocal current
        current = {"parts": [tuple(part) for part in buffer], "calls": []}
        buffer.clear()
        steps.append(current)
        return current

    for record in act.records:
        rtype = record["type"]
        if rtype in MESSAGE_TYPES:
            text = record.get("content")
            if isinstance(text, str) and text:
                if buffer and buffer[-1][0] == rtype:
                    buffer[-1][1] += text
                else:
                    buffer.append([rtype, text])
        elif rtype == MODEL_CALL:
            if record.get("purpose") in STEP_PURPOSES:
                close_step()
        elif rtype == TOOL_CALL:
            call = record.get("call") if isinstance(record.get("call"), dict) else {}
            args = call.get("args")
            entry = {
                "name": str(call.get("name") or "tool"),
                "args": args if isinstance(args, dict) else {"_raw": args},
                "observation": "",
                "is_error": None,
            }
            step = current if current is not None and not buffer else close_step()
            step["calls"].append(entry)
            open_calls.setdefault(str(call.get("id") or ""), []).append(entry)
        elif rtype == TOOL_RESULT:
            waiting = open_calls.get(str(record.get("call_id") or ""))
            if waiting:
                entry = waiting.pop(0)
                result = record.get("result")
                entry["observation"] = result if isinstance(result, str) else str(result)
                # An explicit flag (set by converted logs) wins over reading the text.
                flag = record.get("is_error")
                entry["is_error"] = flag if isinstance(flag, bool) else result_is_error(
                    entry["name"], entry["observation"])
        elif rtype == "error":
            buffer.append(["error", f"error: {record.get('message') or ''}"])

    if buffer:  # text with no closing model_call: a turn cut off, or an older log
        close_step()

    # The turn's answer: the message text of the last response that called no tool.
    for step in reversed(steps):
        if step["calls"]:
            break
        answer = "".join(text for kind, text in step["parts"] if kind == "message").strip()
        if answer:
            step["parts"] = [(k, t) for k, t in step["parts"] if k != "message"]
            step["calls"].append({"name": "final_answer", "args": {"text": answer},
                                  "observation": "", "is_error": None})
            break

    context = _cause_lines(run, act)
    if context:
        if not steps:
            steps.append({"parts": [], "calls": []})
        steps[0]["parts"] = [("context", line) for line in context] + steps[0]["parts"]

    out = []
    for step in steps:
        thought = "\n\n".join(text.strip() for _, text in step["parts"] if text.strip())
        if thought or step["calls"]:
            out.append({"thought": thought, "calls": step["calls"]})
    return out


def agent_trajectory(run: RunLog, agent_id: str, step_supersteps: Optional[list] = None) -> dict:
    """The agent's turns as a Claude Code wire trajectory for ``build_graph``.

    *step_supersteps*, when given, receives the superstep of every emitted step, in
    order; the builder numbers steps the same way, so index = ``step_idx``.
    """
    records: list[dict] = []
    step_no = 0
    call_no = 0
    for act in run.agent_activations(agent_id):
        for step in activation_steps(run, act):
            if step_supersteps is not None:
                step_supersteps.append(act.superstep)
            step_uuid = f"step-{step_no}"
            step_no += 1
            records.append(_event({"type": "step.begin", "uuid": step_uuid, "step": step_no}))
            if step["thought"]:
                records.append(_event({"type": "content.part", "stepUuid": step_uuid,
                                       "part": {"type": "text", "text": step["thought"]}}))
            for call in step["calls"]:
                call_id = f"call-{call_no}"
                call_no += 1
                records.append(_event({"type": "tool.call", "stepUuid": step_uuid,
                                       "toolCallId": call_id, "name": call["name"],
                                       "args": call["args"]}))
                result: dict[str, Any] = {"output": call["observation"]}
                if call["is_error"] is not None:
                    result["isError"] = call["is_error"]
                records.append(_event({"type": "tool.result", "toolCallId": call_id,
                                       "result": result}))
    return {
        "trajectory_format": CLAUDE_CODE_TRAJECTORY_FORMAT,
        "state": {"title": agent_id, "custom": {"sourceFramework": "OpenDraco"}},
        "records": records,
    }


def _event(event: dict) -> dict:
    return {"type": "context.append_loop_event", "event": event}


def _first_commit_superstep(run: RunLog) -> Optional[int]:
    """The earliest superstep in which any agent committed to the workspace."""
    steps = [a.superstep for a in run.activations if a.commit and a.status != "abandoned"]
    return min(steps) if steps else None


def _apply_run_history(G, run: RunLog, step_supersteps: list) -> int:
    """Reclassify visits that come after another agent's edit; return how many changed.

    Upstream classifies each trajectory from its own history only, so a tester
    agent running the tests another agent's edit needs would read as
    localization. A superstep ends only once every activation in it has
    committed (OpenDraco's ``reader.commit_before`` relies on the same barrier),
    so a commit in an earlier superstep happened before this step. ``get_phase``
    only ever asks whether a patch came earlier, so recomputing with a history of
    ``["patch"]`` gives exactly the phase a single shared history would.
    """
    first = _first_commit_superstep(run)
    if first is None:
        return 0
    try:
        from mapPhase import get_phase
    except ImportError:
        return 0
    from server.graph_builder import claude_tool_phase

    changed = 0
    for _, data in G.nodes(data=True):
        phases, indices = data.get("phases"), data.get("step_indices")
        if not isinstance(phases, list) or not isinstance(indices, list) or data.get("label") == "think":
            continue
        args = data.get("args")
        args = {k: v for k, v in args.items() if k != "command_outcome"} if isinstance(args, dict) else args
        tool, sub = str(data.get("tool") or ""), str(data.get("subcommand") or "")
        command, flags = str(data.get("command") or ""), data.get("flags") or {}
        for i, step_idx in enumerate(indices):
            if i >= len(phases) or not isinstance(step_idx, int) or step_idx >= len(step_supersteps):
                continue
            if step_supersteps[step_idx] <= first or phases[i] == "patch":
                continue
            if data.get("opendraco_native"):
                new = claude_tool_phase(tool, args, ["patch"])
            else:
                new = get_phase(tool, sub, command, args, ["patch"], flags, framework="claude")
            if new != phases[i]:
                phases[i] = new
                changed += 1
    return changed


def _alias_phase(kind: str, args, prev: list) -> str:
    """The phase of a tool another framework names differently, by its kind."""
    from server.graph_builder import claude_tool_phase
    if kind in {"read", "write"}:
        return claude_tool_phase(kind, args if isinstance(args, dict) else {}, prev)
    if kind == "search":
        return claude_tool_phase("grep", args if isinstance(args, dict) else {}, prev)
    if kind == "test":
        return "validation" if "patch" in prev else "localization"
    return kind  # "localization" | "general" | "plan" | "patch" taken as given


def _reclassify_with_aliases(G, aliases: dict, run: RunLog, step_supersteps: list,
                             run_phases: bool) -> int:
    """Recompute every visit's phase in step order with one shared patch history.

    Used when tools carry framework-specific names (``tool_aliases``): an aliased
    write is a patch, which changes how every later visit classifies, so the
    whole agent is replayed rather than only the aliased nodes. With
    *run_phases*, a workspace commit in an earlier superstep also counts.
    """
    try:
        from mapPhase import get_phase
    except ImportError:
        return 0
    from server.graph_builder import claude_tool_phase

    first = _first_commit_superstep(run) if run_phases else None
    visits = []
    for key, data in G.nodes(data=True):
        phases, indices = data.get("phases"), data.get("step_indices")
        if not isinstance(phases, list) or not isinstance(indices, list):
            continue
        prefix = str(key).split(":", 1)[0]
        order = int(prefix) if prefix.isdigit() else 0
        for i, step_idx in enumerate(indices):
            if i < len(phases) and isinstance(step_idx, int):
                visits.append((step_idx, order, i, data))
    visits.sort(key=lambda v: (v[0], v[1], v[2]))

    has_patch, changed = False, 0
    for step_idx, _, i, data in visits:
        seeded = first is not None and step_idx < len(step_supersteps) and step_supersteps[step_idx] > first
        prev = ["patch"] if (has_patch or seeded) else []
        phases = data["phases"]
        if data.get("label") == "think":
            new = phases[i]
        else:
            args = data.get("args")
            args = {k: v for k, v in args.items() if k != "command_outcome"} if isinstance(args, dict) else args
            tool = str(data.get("tool") or "")
            if data.get("opendraco_native") and tool in aliases:
                new = _alias_phase(aliases[tool], args, prev)
            elif data.get("opendraco_native"):
                new = claude_tool_phase(tool, args if isinstance(args, dict) else {}, prev)
            else:
                new = get_phase(tool, str(data.get("subcommand") or ""), str(data.get("command") or ""),
                                args, prev, data.get("flags") or {}, framework="claude")
        if new != phases[i]:
            phases[i] = new
            changed += 1
        has_patch = has_patch or new == "patch"
    return changed


def build_agent_graph(run: RunLog, agent_id: str, cmd_parser, *, filter_cd: bool = False,
                      unique_think: bool = True, run_phases: bool = True,
                      tool_aliases: Optional[dict] = None):
    """The agent's Graphectory graph (a networkx MultiDiGraph).

    *run_phases* (the default) lets a workspace commit by any agent in an earlier
    superstep count as a prior patch for this agent's phases; False keeps the
    strict per-trajectory classification every other format uses.

    *tool_aliases* maps a framework's own tool names to a kind (``read``,
    ``write``, ``search``, ``test``, or a phase) for frameworks whose tools the
    Claude Code mapping does not recognise by name.
    """
    step_supersteps: list = []
    traj = agent_trajectory(run, agent_id, step_supersteps)
    G = build_graph(
        traj, agent_id, None, cmd_parser,
        graphs_dir=None, filter_cd=filter_cd, agent_type="claude",
        unique_think=unique_think,
    )
    native = {r["event"]["name"] for r in traj["records"]
              if r["event"].get("type") == "tool.call" and r["event"].get("name") != "bash"}
    for _, data in G.nodes(data=True):
        # A non-shell tool call becomes a node whose tool is the call's name and
        # whose command is empty; a parsed bash command always has a command.
        data["opendraco_native"] = str(data.get("tool") or "") in native and not data.get("command")
    G.graph["phase_scope"] = "run" if run_phases else "agent"
    if tool_aliases:
        G.graph["phases_reclassified"] = _reclassify_with_aliases(
            G, tool_aliases, run, step_supersteps, run_phases)
    else:
        G.graph["phases_from_run_history"] = (
            _apply_run_history(G, run, step_supersteps) if run_phases else 0
        )
    for _, data in G.nodes(data=True):
        data.pop("opendraco_native", None)
        override = PHASE_OVERRIDES.get(str(data.get("tool") or ""))
        if override and isinstance(data.get("phases"), list):
            data["phases"] = [override if p == "general" else p for p in data["phases"]]
    G.graph["agent_id"] = agent_id
    G.graph["run_id"] = run.run_id
    return G
