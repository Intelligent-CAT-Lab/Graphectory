"""
opendraco_runs/events.py

Turn a public-OpenDraco inference event log into an OpenDraco2-shaped ``RunLog``.

The public OpenDraco (github.com/OpenDraco/OpenDraco) writes each inference run
to ``opendraco/logs/inference_logs/<stem>.ndjson``: the same events its web UI
streams (``api/routers/inference.py``), one JSON object per line. One file holds
one or more SWE-bench instances run in sequence; each instance is one
multi-agent run. Rather than a second analysis stack, the events of the chosen
instance are rewritten as OpenDraco2 records and read by
``opendraco2_runs.log.parse_run_log``, so per-agent graphs, the run view and
pattern detection are shared:

==================  ===========================================================
event               becomes
==================  ===========================================================
``run_meta``        ``run_started`` (topology built from the agents and hand-offs)
``thinking_chunk``  ``thinking``, opening the agent's turn if none is open
``response``        ``message`` + ``model_call``: one model response = one step
``tool_call``       ``tool_call`` + ``tool_result`` (truncated previews only)
``agent_input``     deliveries from START for the inputs that are not agents
``agent_event``     ``activation_finished``: the agent's graph node completed
``handoff``         ``delivery`` from the finished turn to the next turn of ``to``
``agent_tokens``    the agent's token totals, on its last turn
``error``           ``error``
``instance_done``   ``run_finished`` (``completed``), with the final patch
==================  ===========================================================

What the log does not record is derived or left out, and ``RunLog.meta`` says so:
supersteps come from hand-off causality (START = 0); a turn that called a
write-kind tool gets an *inferred* ``workspace_commit`` naming the paths it wrote,
so run-aware phases and ping-pong detection work; durations and per-turn tokens
are unknown.
"""

import json
import re
from collections import defaultdict
from typing import Optional

from opendraco2_runs.log import RunLog, iter_lines, parse_run_log
from opendraco_runs.tools import (
    command_of,
    parse_args_preview,
    path_of,
    result_is_error,
    tool_kind,
)

START = "start"
END_NAMES = {"end", "__end__", "END"}
_DIFF_PATH_RE = re.compile(r"^diff --git a/(\S+) b/", re.MULTILINE)


class EventLogError(ValueError):
    """The body is not a usable event log, or names an instance it lacks (400)."""


def looks_like_opendraco(body: bytes) -> bool:
    """True when the first parsed record is the event log's ``run_meta`` header."""
    for record in iter_lines(body):
        if record is None:
            continue
        return record.get("type") == "run_meta" and "instance_ids" in record and "config_name" in record
    return False


def list_instances(events: list[dict]) -> list[str]:
    seen: list[str] = []
    for event in events:
        if event.get("type") == "instance_start" and isinstance(event.get("instance_id"), str):
            if event["instance_id"] not in seen:
                seen.append(event["instance_id"])
    if not seen:
        meta = next((e for e in events if e.get("type") == "run_meta"), {})
        seen = [i for i in meta.get("instance_ids") or [] if isinstance(i, str)]
    return seen


def _segment(events: list[dict], instance_id: Optional[str]) -> tuple[list[dict], list[dict]]:
    """(events of *instance_id*, run-level events after the last instance)."""
    if instance_id is None:
        return events, []
    inside, out, tail = False, [], []
    for event in events:
        etype = event.get("type")
        if etype == "instance_start":
            inside = event.get("instance_id") == instance_id
            if inside:
                out.append(event)
            continue
        if inside:
            out.append(event)
            if etype == "instance_done":
                inside = False
        elif etype in {"done", "cancelled", "error"} and out:
            tail.append(event)
    return out, tail


class _Writer:
    """Builds OpenDraco2 records with the envelope its parser expects."""

    def __init__(self, run_id: str, timestamp: str):
        self.run_id, self.timestamp = run_id, timestamp
        self.records: list[dict] = []
        self.counter = 0

    def uuid(self, prefix: str) -> str:
        self.counter += 1
        return f"{prefix}-{self.counter:05d}"

    def add(self, rtype: str, *, parent=None, agent=None, turn=None, uuid=None, **fields) -> dict:
        record = {"type": rtype, "uuid": uuid or self.uuid(rtype.split("_")[0]), "parent_uuid": parent,
                  "run_id": self.run_id, "global_seq": len(self.records) + 1, "agent_id": agent,
                  "local_turn": turn, "timestamp": fields.pop("timestamp", None) or self.timestamp}
        record.update(fields)
        self.records.append(record)
        return record


def parse_event_log(body: bytes, instance_id: Optional[str] = None) -> tuple[RunLog, list[str]]:
    """The chosen instance (default: the first) as a RunLog, and every instance id."""
    events = [e for e in iter_lines(body) if e is not None]
    meta = next((e for e in events if e.get("type") == "run_meta"), None)
    if meta is None:
        raise EventLogError("No OpenDraco event log found: expected a first run_meta record.")
    instances = list_instances(events)
    if instance_id is not None and instance_id not in instances:
        raise EventLogError(f"Unknown instance '{instance_id}'; this log has: {', '.join(instances) or 'none'}.")
    chosen = instance_id or (instances[0] if instances else None)
    has_markers = any(e.get("type") == "instance_start" for e in events)
    segment, tail = _segment(events, chosen if has_markers else None)

    run_id = f"{meta.get('run_id') or 'run'}/{chosen}" if chosen else str(meta.get("run_id") or "run")
    w = _Writer(run_id, str(meta.get("ts") or ""))
    started = w.add("run_started", topology={"nodes": [], "edges": []}, agents={}, topology_hash="",
                    framework_version="OpenDraco (inference event log)", forked_from=None)

    agents: list[str] = []
    open_act: dict[str, dict] = {}
    last_act: dict[str, dict] = {}
    turns: dict[str, int] = defaultdict(int)
    pending: dict[str, list[str]] = defaultdict(list)
    superstep_of: dict[str, int] = {}
    delivery_parent: dict[str, Optional[str]] = {}
    responses: dict[str, int] = defaultdict(int)
    wrote: dict[str, list[str]] = defaultdict(list)
    finished_of: dict[str, dict] = {}
    edges: dict[str, dict] = {}
    status, final_patch = None, ""

    def note_agent(name: str):
        if name and name not in agents and name not in END_NAMES and name != START:
            agents.append(name)

    def open_turn(agent: str, timestamp=None) -> dict:
        if agent in open_act:
            return open_act[agent]
        note_agent(agent)
        causes = pending.pop(agent, [])
        producers = [superstep_of.get(delivery_parent.get(c) or "", 0) for c in causes]
        turns[agent] += 1
        act = w.add("activation_started", agent=agent, turn=turns[agent], caused_by=list(causes),
                    superstep=(max(producers) + 1) if producers else 1, timestamp=timestamp)
        superstep_of[act["uuid"]] = act["superstep"]
        open_act[agent] = act
        return act

    def in_turn(agent: str, rtype: str, **fields) -> dict:
        act = open_turn(agent)
        return w.add(rtype, parent=act["uuid"], agent=agent, turn=act["local_turn"], **fields)

    def deliver(frm: str, to: str, value: str, parent: Optional[str], timestamp=None) -> str:
        edge_id = f"{frm}->{to}"
        edges.setdefault(edge_id, {"id": edge_id, "source": frm, "target": to, "slot": "input"})
        d = w.add("delivery", parent=parent or started["uuid"], from_agent=frm, to_agent=to,
                  edge_id=edge_id, slot="input", policy="append", value=value, timestamp=timestamp)
        delivery_parent[d["uuid"]] = parent
        if to not in END_NAMES:
            pending[to].append(d["uuid"])
        return d["uuid"]

    for event in segment:
        etype = event.get("type")
        agent = str(event.get("agent") or "")
        if etype == "thinking_chunk" and agent:
            if isinstance(event.get("chunk"), str) and event["chunk"]:
                in_turn(agent, "thinking", content=event["chunk"])
        elif etype == "response" and agent:
            act = open_turn(agent)
            text = event.get("content")
            if isinstance(text, str) and text:
                in_turn(agent, "message", content=text)
            responses[act["uuid"]] += 1
            in_turn(agent, "model_call", purpose="work" if responses[act["uuid"]] == 1 else "tool_turn",
                    started_at=w.timestamp, ended_at=w.timestamp,
                    usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0})
        elif etype == "tool_call" and agent:
            act = open_turn(agent)
            name = str(event.get("tool") or "tool")
            args = parse_args_preview(event.get("args_preview"))
            kind = tool_kind(name)
            if kind == "shell":
                call_name, call_args = "bash", {"command": command_of(args) or str(event.get("args_preview") or "")}
            else:
                call_name, call_args = name, args
            if kind == "write" and path_of(args):
                wrote[act["uuid"]].append(path_of(args))
            call_id = w.uuid("call")
            in_turn(agent, "tool_call", call={"name": call_name, "args": call_args, "id": call_id})
            result = str(event.get("result_preview") or "")
            # The preview has no exit code or failure marker of OpenDraco2's
            # kind, so the verdict travels as an explicit flag.
            in_turn(agent, "tool_result", call_id=call_id, name=call_name, result=result,
                    is_error=result_is_error(result))
        elif etype == "agent_input" and agent:
            act = open_turn(agent)
            inputs = event.get("inputs") if isinstance(event.get("inputs"), dict) else {}
            for pred, value in inputs.items():
                if pred in agents and pred != agent:
                    continue  # an agent predecessor arrives as its own hand-off
                uuid = deliver(START, agent, value if isinstance(value, str) else json.dumps(value, default=str),
                               None)
                pending[agent].remove(uuid)
                if uuid not in act["caused_by"]:
                    act["caused_by"].append(uuid)
        elif etype == "agent_event" and agent:
            act = open_turn(agent)
            finished = w.add("activation_finished", parent=act["uuid"], agent=agent, turn=act["local_turn"],
                             usage={"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
                             duration_s=None, stop_reason=None)
            finished_of[agent] = finished
            if wrote.get(act["uuid"]):
                w.add("workspace_commit", parent=act["uuid"], agent=agent, turn=act["local_turn"],
                      activation_uuid=act["uuid"], sha="", inferred=True,
                      changed_paths=sorted(set(wrote[act["uuid"]])))
            last_act[agent] = act
            del open_act[agent]
        elif etype == "handoff":
            frm, to = str(event.get("from") or ""), str(event.get("to") or "")
            if not frm or not to:
                continue
            note_agent(frm)
            note_agent(to)
            source = last_act.get(frm) or open_act.get(frm)
            value = event.get("preview") or event.get("summary") or ""
            deliver(frm, to, value if isinstance(value, str) else json.dumps(value, default=str),
                    source["uuid"] if source else None, timestamp=event.get("timestamp"))
        elif etype == "agent_tokens" and agent in finished_of:
            finished_of[agent]["usage"] = {
                "input_tokens": int(event.get("input") or 0), "output_tokens": int(event.get("output") or 0),
                "total_tokens": int(event.get("total") or 0)}
        elif etype == "error":
            w.add("error", parent=started["uuid"], message=str(event.get("message") or ""))
            status = status or "failed"
        elif etype == "instance_done":
            final_patch = str(event.get("patch") or "")
            status = "completed"
    for event in tail:
        if event.get("type") == "cancelled":
            status = "stopped"
        elif event.get("type") == "error" and status is None:
            status = "failed"

    nodes = [{"id": START, "kind": "start"}] + [{"id": a, "kind": "agent"} for a in agents]
    if any(e["target"] in END_NAMES for e in edges.values()):
        end = next(e["target"] for e in edges.values() if e["target"] in END_NAMES)
        nodes.append({"id": end, "kind": "end"})
    started["topology"] = {"nodes": nodes, "edges": list(edges.values())}
    started["agents"] = {a: {"id": a} for a in agents}
    if status:
        w.add("run_finished", parent=started["uuid"], status=status, error_uuid=None)

    body2 = "\n".join(json.dumps(r, default=str) for r in w.records).encode()
    run = parse_run_log(body2)
    run.meta = {
        "source_format": "opendraco-event-log",
        "config_name": meta.get("config_name"),
        "instance_id": chosen,
        "instances": instances,
        "final_patch_paths": sorted(set(_DIFF_PATH_RE.findall(final_patch))),
        "derived": {
            "supersteps": "from hand-off causality",
            "workspace_commits": "inferred from write-kind tool calls",
            "tool_arguments": "previews, each value cut to 40 characters",
            "tool_results": "previews of at most 80 characters",
            "per_turn_tokens": False,
            "durations": False,
        },
    }
    return run, instances
