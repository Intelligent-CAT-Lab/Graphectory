"""
opendraco_runs/log.py

Parse an OpenDraco ``run.jsonl`` into a ``RunLog``: activations with the records
that belong to them, deliveries, commits and route decisions.

The schema is OpenDraco's ``backend/src/opendraco/runlog/records.py``; this reader
mirrors ``runlog/reader.py``'s tolerance rather than its strictness:

* a line that is not a JSON object naming a ``type`` is skipped, so a log whose
  writer was killed mid-line (a run still in progress, or a crash) still reads;
* records are grouped by ``parent_uuid`` (the opening ``activation_started``),
  never by line adjacency, because agents of one superstep run in parallel and
  their records interleave;
* a ``run_resumed`` abandons every activation still open, and the activation that
  later reruns it is linked back with ``rerun_of``;
* a fork's first activations name deliveries recorded in the parent run's log,
  which this file does not hold; those causes are kept as ``missing_causes``.

Everything here is data. Nothing read from the log is evaluated, and the text
fields stay plain strings for the renderers to escape.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

RUN_STARTED = "run_started"
RUN_FINISHED = "run_finished"
RUN_RESUMED = "run_resumed"
DELIVERY = "delivery"
ACTIVATION_STARTED = "activation_started"
ACTIVATION_FINISHED = "activation_finished"
WORKSPACE_COMMIT = "workspace_commit"
ROUTE_DECISION = "route_decision"
MODEL_CALL = "model_call"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
ERROR = "error"
THINKING = "thinking"
MESSAGE = "message"
MESSAGE_TYPES = {THINKING, MESSAGE}

# Status reported for a log with no closing ``run_finished`` (or one reopened by a
# later ``run_resumed``). Not an OpenDraco RunStatus: the run may still be going.
IN_PROGRESS = "in_progress"

# Largest single line read. The whole body is already capped by the endpoint; this
# keeps one pathological line from being parsed at all.
MAX_LINE_BYTES = 16 * 1024 * 1024


def _str(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _int(value: Any) -> Optional[int]:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def usage_tokens(usage: Any) -> dict[str, int]:
    """``providers.base.Usage`` as plain ints, zero where absent or malformed."""
    usage = usage if isinstance(usage, dict) else {}
    out = {key: _int(usage.get(key)) or 0
           for key in ("input_tokens", "output_tokens", "total_tokens")}
    if not out["total_tokens"]:
        out["total_tokens"] = out["input_tokens"] + out["output_tokens"]
    return out


@dataclass
class Activation:
    """One agent turn: its opening record and every record that hangs off it."""

    uuid: str
    agent_id: str
    local_turn: Optional[int]
    superstep: int
    caused_by: list[str]
    seq: int
    timestamp: str
    records: list[dict] = field(default_factory=list)
    finished: Optional[dict] = None
    commit: Optional[dict] = None
    # "finished" | "abandoned" (cut off by a pause, rerun later) | "running"
    status: str = "running"
    rerun_of: Optional[str] = None
    # Under the append delivery policy caused_by lists every delivery the agent has
    # ever received; new_causes are the ones this turn is the first to see.
    new_causes: list[str] = field(default_factory=list)
    # new_causes this log does not hold (a fork's parent-run deliveries).
    missing_causes: list[str] = field(default_factory=list)

    @property
    def usage(self) -> dict[str, int]:
        if self.finished is not None:
            return usage_tokens(self.finished.get("usage"))
        # An unfinished turn has no total yet; sum what its model calls recorded.
        total = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        for record in self.records:
            if record["type"] == MODEL_CALL:
                for key, value in usage_tokens(record.get("usage")).items():
                    total[key] += value
        return total

    @property
    def duration_s(self) -> Optional[float]:
        if self.finished is None:
            return None
        value = self.finished.get("duration_s")
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None

    @property
    def stop_reason(self) -> Optional[str]:
        return _str(self.finished.get("stop_reason")) or None if self.finished else None

    def count(self, record_type: str) -> int:
        return sum(1 for record in self.records if record["type"] == record_type)


@dataclass
class RunLog:
    run_id: str = ""
    started: Optional[dict] = None
    finished: Optional[dict] = None
    status: str = IN_PROGRESS
    resumes: list[dict] = field(default_factory=list)
    forked_from: Optional[dict] = None
    deliveries: dict[str, dict] = field(default_factory=dict)
    activations: list[Activation] = field(default_factory=list)
    commits: list[dict] = field(default_factory=list)
    route_decisions: list[dict] = field(default_factory=list)
    errors: list[dict] = field(default_factory=list)
    # Records whose parent is neither an activation nor the run's opening record.
    orphans: list[dict] = field(default_factory=list)
    skipped_lines: int = 0
    # The last non-empty line was not a record: a half-written tail.
    truncated: bool = False
    by_uuid: dict[str, Activation] = field(default_factory=dict, repr=False)

    def activation(self, uuid: str) -> Optional[Activation]:
        return self.by_uuid.get(uuid)

    @property
    def agent_ids(self) -> list[str]:
        """Agents in topology order, then any that only appear in activations."""
        ordered: list[str] = []
        topology = (self.started or {}).get("topology")
        for node in (topology or {}).get("nodes", []) if isinstance(topology, dict) else []:
            if isinstance(node, dict) and node.get("kind", "agent") == "agent":
                node_id = _str(node.get("id"))
                if node_id and node_id not in ordered:
                    ordered.append(node_id)
        for agent_id in (self.started or {}).get("agents", {}) or {}:
            if isinstance(agent_id, str) and agent_id not in ordered:
                ordered.append(agent_id)
        for activation in self.activations:
            if activation.agent_id not in ordered:
                ordered.append(activation.agent_id)
        return ordered

    def agent_activations(self, agent_id: str, *, include_abandoned: bool = False) -> list[Activation]:
        acts = [a for a in self.activations if a.agent_id == agent_id
                and (include_abandoned or a.status != "abandoned")]
        return sorted(acts, key=lambda a: (a.local_turn if a.local_turn is not None else 1 << 30, a.seq))

    def delivery_superstep(self, delivery: dict) -> Optional[int]:
        """The superstep of the activation that produced *delivery*; None for START's."""
        producer = self.activation(_str(delivery.get("parent_uuid")))
        return producer.superstep if producer is not None else None


def iter_lines(body: bytes) -> Iterable[Optional[dict]]:
    """Yield each non-empty line as a dict record, or None where it is not one."""
    for raw in body.split(b"\n"):
        if not raw.strip():
            continue
        if len(raw) > MAX_LINE_BYTES:
            yield None
            continue
        try:
            value = json.loads(raw.decode("utf-8", errors="replace"))
        except (json.JSONDecodeError, ValueError):
            yield None
            continue
        yield value if isinstance(value, dict) and isinstance(value.get("type"), str) else None


def looks_like_opendraco(body: bytes) -> bool:
    """True when the first parsed record is OpenDraco's ``run_started``."""
    for record in iter_lines(body):
        if record is None:
            continue
        return (record.get("type") == RUN_STARTED
                and "framework_version" in record and "topology" in record)
    return False


def parse_run_log(body: bytes) -> RunLog:
    log = RunLog()
    records: list[dict] = []
    last_was_bad = False
    for index, record in enumerate(iter_lines(body)):
        if record is None:
            log.skipped_lines += 1
            last_was_bad = True
            continue
        last_was_bad = False
        record["_line"] = index
        records.append(record)
    log.truncated = last_was_bad

    # Write order is global_seq order; sort on it where present so a log
    # concatenated or re-serialised out of order still reads causally.
    records.sort(key=lambda r: (_int(r.get("global_seq")) or 0, r["_line"]))

    by_uuid = log.by_uuid
    open_uuids: list[str] = []
    abandoned: list[Activation] = []
    rerun_claimed: set[str] = set()
    run_started_uuid = ""
    pending: list[dict] = []

    for record in records:
        rtype = record["type"]
        seq = _int(record.get("global_seq")) or record["_line"]
        if rtype == RUN_STARTED and log.started is None:
            log.started = record
            log.run_id = _str(record.get("run_id"))
            run_started_uuid = _str(record.get("uuid"))
            fork = record.get("forked_from")
            log.forked_from = fork if isinstance(fork, dict) else None
        elif rtype == RUN_FINISHED:
            log.finished = record
        elif rtype == RUN_RESUMED:
            log.resumes.append(record)
            log.finished = None  # the resume reopens a paused log
            for uuid in open_uuids:
                act = by_uuid[uuid]
                act.status = "abandoned"
                abandoned.append(act)
            open_uuids.clear()
        elif rtype == DELIVERY:
            uuid = _str(record.get("uuid"))
            if uuid:
                log.deliveries[uuid] = record
        elif rtype == ACTIVATION_STARTED:
            uuid = _str(record.get("uuid"))
            agent_id = _str(record.get("agent_id"))
            if not uuid or not agent_id:
                log.orphans.append(record)
                continue
            caused_by = [c for c in record.get("caused_by") or [] if isinstance(c, str)]
            act = Activation(
                uuid=uuid, agent_id=agent_id, local_turn=_int(record.get("local_turn")),
                superstep=_int(record.get("superstep")) or 0, caused_by=caused_by,
                seq=seq, timestamp=_str(record.get("timestamp")),
            )
            # The rerun of an abandoned attempt: same agent, same superstep, opened
            # after the resume that abandoned it.
            for old in abandoned:
                if old.agent_id == agent_id and old.superstep == act.superstep \
                        and old.uuid not in rerun_claimed:
                    act.rerun_of = old.uuid
                    rerun_claimed.add(old.uuid)
                    break
            by_uuid[uuid] = act
            log.activations.append(act)
            open_uuids.append(uuid)
        elif rtype == ACTIVATION_FINISHED:
            # Closed inline, not in the attach pass below: a later run_resumed must
            # only abandon the turns that were still open when it was written.
            owner = by_uuid.get(_str(record.get("parent_uuid")))
            if owner is None:
                log.orphans.append(record)
            elif owner.status != "abandoned":
                owner.finished = record
                owner.status = "finished"
                if owner.uuid in open_uuids:
                    open_uuids.remove(owner.uuid)
        elif rtype == WORKSPACE_COMMIT:
            log.commits.append(record)
            pending.append(record)
        else:
            pending.append(record)

        if not log.run_id and _str(record.get("run_id")):
            log.run_id = _str(record.get("run_id"))

    # Attach in-activation records now that every activation is known (a record can
    # only follow its opening, but attaching late keeps one code path).
    for record in pending:
        rtype = record["type"]
        if rtype == WORKSPACE_COMMIT:
            owner = by_uuid.get(_str(record.get("activation_uuid"))) or by_uuid.get(_str(record.get("parent_uuid")))
            if owner is not None:
                owner.commit = record
            continue
        owner = by_uuid.get(_str(record.get("parent_uuid")))
        if owner is None:
            if rtype == ERROR:
                log.errors.append(record)
            elif _str(record.get("parent_uuid")) != run_started_uuid:
                log.orphans.append(record)
            continue
        owner.records.append(record)
        if rtype == ROUTE_DECISION:
            log.route_decisions.append(record)
        elif rtype == ERROR:
            log.errors.append(record)

    for act in log.activations:
        act.records.sort(key=lambda r: (_int(r.get("global_seq")) or 0, r["_line"]))

    # An abandoned attempt does not count as having seen its causes, so the rerun
    # that replaces it is shown receiving them.
    for agent_id in {a.agent_id for a in log.activations}:
        seen: set[str] = set()
        for act in log.agent_activations(agent_id, include_abandoned=True):
            act.new_causes = [c for c in act.caused_by if c not in seen]
            act.missing_causes = [c for c in act.new_causes if c not in log.deliveries]
            if act.status != "abandoned":
                seen.update(act.caused_by)

    if log.finished is not None:
        log.status = _str(log.finished.get("status")) or IN_PROGRESS
    return log
