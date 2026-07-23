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
from functools import lru_cache
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
KIMI_SWE_TOGETHER_FORMAT = "kimi-code-swe-together-sharegpt-1"
CLAUDE_CODE_TRAJECTORY_FORMAT = "claude-code-wire-1"
CODEX_TRAJECTORY_FORMAT = "codex-rollout-jsonl-1"

_KIMI_SWE_TOGETHER_FILENAME = "kimicode_swetogether_r3_sharegpt.json"
_KIMI_SWE_TOGETHER_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(?P<name>[A-Za-z_][\w.-]*)\s*\("
    r"(?P<args>.*?)\)\s*</tool_call>",
    re.DOTALL,
)
_KIMI_SWE_TOGETHER_CALL_INDEX_RE = re.compile(r":call-(\d+)$")

_CODEX_CALL_TYPES = {
    "function_call",
    "custom_tool_call",
    "tool_search_call",
    "web_search_call",
    "image_generation_call",
}
_CODEX_OUTPUT_TYPES = {
    "function_call_output",
    "custom_tool_call_output",
    "tool_search_output",
}


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
    """Return whether *path* looks like a compatible Code wire stream."""
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


def _is_codex_rollout_file(path: Path) -> bool:
    """Return whether *path* looks like a persisted Codex rollout stream."""
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        return False

    saw_session_meta = False
    saw_codex_event = False
    try:
        for index, record in enumerate(_read_jsonl(path)):
            record_type = record.get("type")
            payload = record.get("payload")
            if record_type == "session_meta" and isinstance(payload, dict):
                saw_session_meta = bool(
                    payload.get("id") or payload.get("session_id")
                ) and bool(payload.get("cwd") or payload.get("originator"))
            elif record_type in {"turn_context", "response_item", "event_msg"}:
                saw_codex_event = True
            if saw_session_meta and saw_codex_event:
                return True
            if index >= 40:
                break
    except OSError:
        return False
    return False


def _codex_rollout_files(source: Path) -> list[Path]:
    """Find Codex rollout JSONL files below a file or directory."""
    if source.is_file():
        return [source] if _is_codex_rollout_file(source) else []
    return sorted(
        path for path in source.rglob("*.jsonl")
        if _is_codex_rollout_file(path)
    )


def _codex_content_text(content) -> str:
    """Flatten Codex message/reasoning content blocks into visible text."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""

    chunks = []
    for block in content:
        if isinstance(block, str):
            chunks.append(block)
        elif isinstance(block, dict):
            text = block.get("text") or block.get("input_text") \
                or block.get("output_text")
            if isinstance(text, str):
                chunks.append(text)
    return "\n".join(chunk for chunk in chunks if chunk)


@lru_cache(maxsize=2048)
def _read_codex_rollout_metadata(path_text: str, size: int, mtime_ns: int) -> dict:
    """Read lightweight list metadata without retaining a rollout in memory."""
    del size, mtime_ns  # Values participate in cache invalidation.
    path = Path(path_text)
    metadata: dict = {}
    title = ""
    first_user_message = ""
    model = ""
    step_count = 0

    for record in _read_jsonl(path):
        record_type = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue

        if record_type == "session_meta" and not metadata:
            metadata = payload
        elif record_type == "turn_context" and not model:
            model = str(payload.get("model") or "")
        elif record_type == "response_item" and payload.get("type") in _CODEX_CALL_TYPES:
            step_count += 1
        elif record_type == "event_msg":
            event_type = payload.get("type")
            if event_type == "thread_name_updated":
                title = str(payload.get("thread_name") or title).strip()
            elif event_type == "user_message" and not first_user_message:
                first_user_message = str(payload.get("message") or "").strip()

    instance_id = str(
        metadata.get("id") or metadata.get("session_id") or path.stem
    )
    if not title and first_user_message:
        title = re.sub(r"\s+", " ", first_user_message).strip()
    if len(title) > 96:
        title = f"{title[:93].rstrip()}..."

    cwd = str(metadata.get("cwd") or "")
    fallback_name = Path(cwd).name if cwd else path.stem
    return {
        "instance_id": instance_id,
        "display_name": title or fallback_name or instance_id,
        "model": model,
        "cwd": cwd,
        "step_count": step_count,
        "source_path": str(path),
        "metadata": metadata,
    }


def _codex_rollout_metadata(path: Path) -> dict:
    """Return cached metadata, invalidated when the rollout file changes."""
    stat = path.stat()
    return dict(_read_codex_rollout_metadata(
        str(path.resolve()), stat.st_size, stat.st_mtime_ns,
    ))


def _kimi_wire_files(source: Path) -> list[Path]:
    """Find main-agent compatible Code wire streams below a file or directory."""
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


def _wire_source_framework(wire_path: Path) -> str:
    """Return a declared original framework for converted wire sessions."""
    custom = _load_kimi_state(wire_path).get("custom")
    if not isinstance(custom, dict):
        return ""
    return str(custom.get("sourceFramework") or "").strip().lower()


def _claude_code_wire_files(source: Path) -> list[Path]:
    """Find converted Claude Code sessions that use the compatible wire schema."""
    return [
        path for path in _kimi_wire_files(source)
        if _wire_source_framework(path) == "claude code"
    ]


def _native_kimi_wire_files(source: Path) -> list[Path]:
    """Find Kimi Code sessions, excluding sources explicitly marked as Claude Code."""
    return [
        path for path in _kimi_wire_files(source)
        if _wire_source_framework(path) != "claude code"
    ]


def _kimi_swe_together_path(source: Path) -> Path | None:
    """Locate the published Kimi Code SWE-Together ShareGPT export."""
    if source.is_file():
        return source if source.name == _KIMI_SWE_TOGETHER_FILENAME else None
    candidate = source / _KIMI_SWE_TOGETHER_FILENAME
    return candidate if candidate.is_file() else None


def _is_kimi_swe_together_canonical_file(path: Path) -> bool:
    """Return whether a JSONL file is a SWE-Together canonical request trace."""
    if not path.is_file() or path.suffix.lower() != ".jsonl":
        return False
    try:
        for record in _read_jsonl(path):
            return record.get("schema") == "swe-together-agentic-trace-v2"
    except OSError:
        return False
    return False


def _load_kimi_swe_together_canonical_records(source: Path) -> list[dict]:
    """Load valid per-call rows from a canonical SWE-Together JSONL trace."""
    if not _is_kimi_swe_together_canonical_file(source):
        return []
    return [
        record for record in _read_jsonl(source)
        if record.get("schema") == "swe-together-agentic-trace-v2"
    ]


def _kimi_swe_together_call_index(record: dict) -> int:
    match = _KIMI_SWE_TOGETHER_CALL_INDEX_RE.search(str(record.get("id") or ""))
    return int(match.group(1)) if match else 0


def _load_kimi_swe_together_rows(source: Path) -> list[dict]:
    """Load valid conversation rows from the SWE-Together export."""
    path = _kimi_swe_together_path(source)
    if path is None:
        return []
    try:
        with open(path, encoding="utf-8", errors="replace") as stream:
            rows = json.load(stream)
    except (OSError, json.JSONDecodeError):
        return []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _kimi_swe_together_session_id(row: dict) -> str:
    metadata = row.get("metadata")
    if isinstance(metadata, dict):
        return str(metadata.get("session_id") or "").strip()
    return ""


def detect_agent_type(source: Path) -> str:
    """Infer the trajectory format represented by *source*."""
    if _is_kimi_swe_together_canonical_file(source):
        return "kimi_swe_together"
    if _kimi_swe_together_path(source):
        return "kimi_swe_together"
    if source.is_file():
        if _is_codex_rollout_file(source):
            return "codex"
        if _is_kimi_wire_file(source):
            return "claude" if _wire_source_framework(source) == "claude code" else "kimi"
        if source.suffix.lower() == ".jsonl":
            return "oh"
        return "sa"

    if _claude_code_wire_files(source):
        return "claude"
    if _native_kimi_wire_files(source):
        return "kimi"
    for jsonl_path in source.rglob("*.jsonl"):
        if _is_codex_rollout_file(jsonl_path):
            return "codex"
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
    For Kimi Code (agent_type='kimi') or Claude Code (agent_type='claude'),
    graphs_dir contains session wire.jsonl files.
    For Kimi Code SWE-Together (agent_type='kimi_swe_together'), graphs_dir
    contains the published ShareGPT request-trace export.
    For Codex (agent_type='codex'), graphs_dir contains rollout JSONL files.
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

    if agent_type == "kimi_swe_together":
        canonical_records = _load_kimi_swe_together_canonical_records(graphs_dir)
        if canonical_records:
            records_by_session: dict[str, list[dict]] = {}
            for record in canonical_records:
                session_id = str(record.get("session_id") or "").strip()
                if session_id:
                    records_by_session.setdefault(session_id, []).append(record)
            for instance_id, records in records_by_session.items():
                records.sort(key=_kimi_swe_together_call_index)
                reconstructed_steps = _iter_kimi_swe_together_canonical_steps(records)
                if not reconstructed_steps:
                    continue
                results.append({
                    "instance_id": instance_id,
                    "display_name": f"SWE-Together {instance_id}",
                    "status": "none",
                    "difficulty": "unknown",
                    "step_count": len(reconstructed_steps),
                    "model": str(records[0].get("model") or "Kimi K2.6"),
                    "llm_calls": len(records),
                })
            results.sort(key=lambda item: (-item["step_count"], item["instance_id"]))
            return results

        for row in _load_kimi_swe_together_rows(graphs_dir):
            instance_id = _kimi_swe_together_session_id(row)
            if not instance_id:
                continue
            conversations = row.get("conversations")
            step_count = sum(
                1 for turn in conversations if isinstance(turn, dict)
                and turn.get("from") == "gpt"
            ) if isinstance(conversations, list) else 0
            metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
            results.append({
                "instance_id": instance_id,
                "display_name": f"SWE-Together {instance_id}",
                "status": "none",
                "difficulty": "unknown",
                "step_count": step_count,
                "model": "Kimi K2.6",
                "llm_calls": metadata.get("llm_calls"),
            })

        results.sort(key=lambda item: item["instance_id"])
        return results

    if agent_type == "codex":
        for rollout_path in _codex_rollout_files(graphs_dir):
            try:
                metadata = _codex_rollout_metadata(rollout_path)
            except OSError:
                continue
            instance_id = metadata["instance_id"]
            if instance_id in resolved_set:
                status = "resolved"
            elif instance_id in unresolved_set:
                status = "unresolved"
            elif eval_report_path:
                status = "unsubmitted"
            else:
                status = "none"
            results.append({
                "instance_id": instance_id,
                "display_name": metadata["display_name"],
                "status": status,
                "difficulty": "unknown",
                "step_count": metadata["step_count"],
                "model": metadata["model"],
            })

        results.sort(key=lambda item: (item["display_name"].lower(), item["instance_id"]))
        return results

    if agent_type in {"kimi", "claude"}:
        wire_files = (
            _claude_code_wire_files(graphs_dir)
            if agent_type == "claude" else _native_kimi_wire_files(graphs_dir)
        )
        for wire_path in wire_files:
            session_dir = _kimi_session_dir(wire_path)
            instance_id = session_dir.name or wire_path.stem
            state = _load_kimi_state(wire_path)
            custom = state.get("custom")
            model = str(custom.get("model") or "") if isinstance(custom, dict) else ""
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
                "model": model,
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
    For Kimi Code or Claude Code, loads the main agent's wire.jsonl event stream.
    For Kimi Code SWE-Together, loads one published ShareGPT conversation.
    For Codex, loads one persisted rollout JSONL event stream.

    Raises FileNotFoundError if the trajectory cannot be found.
    """
    if agent_type == "kimi_swe_together":
        canonical_records = _load_kimi_swe_together_canonical_records(graphs_dir)
        if canonical_records:
            records = [
                record for record in canonical_records
                if str(record.get("session_id") or "") == instance_id
            ]
            if records:
                return {
                    "trajectory_format": KIMI_SWE_TOGETHER_FORMAT,
                    "source_path": str(graphs_dir),
                    "metadata": {
                        "session_id": instance_id,
                        "model": records[0].get("model") or "Kimi K2.6",
                    },
                    "canonical_records": records,
                }

        for row in _load_kimi_swe_together_rows(graphs_dir):
            if _kimi_swe_together_session_id(row) != instance_id:
                continue
            return {
                "trajectory_format": KIMI_SWE_TOGETHER_FORMAT,
                "source_path": str(_kimi_swe_together_path(graphs_dir)),
                "metadata": row.get("metadata") or {},
                "conversations": row.get("conversations") or [],
            }
        raise FileNotFoundError(
            f"No Kimi Code SWE-Together session '{instance_id}' found under {graphs_dir}"
        )

    if agent_type == "codex":
        for rollout_path in _codex_rollout_files(graphs_dir):
            metadata = _codex_rollout_metadata(rollout_path)
            if metadata["instance_id"] != instance_id:
                continue
            return {
                "trajectory_format": CODEX_TRAJECTORY_FORMAT,
                "source_path": str(rollout_path),
                "metadata": metadata,
                "records": list(_read_jsonl(rollout_path)),
            }
        raise FileNotFoundError(
            f"No Codex session '{instance_id}' found under {graphs_dir}"
        )

    if agent_type in {"kimi", "claude"}:
        wire_files = (
            _claude_code_wire_files(graphs_dir)
            if agent_type == "claude" else _native_kimi_wire_files(graphs_dir)
        )
        for wire_path in wire_files:
            session_dir = _kimi_session_dir(wire_path)
            if (session_dir.name or wire_path.stem) != instance_id:
                continue
            return {
                "trajectory_format": (
                    CLAUDE_CODE_TRAJECTORY_FORMAT
                    if agent_type == "claude" else KIMI_TRAJECTORY_FORMAT
                ),
                "wire_path": str(wire_path),
                "state": _load_kimi_state(wire_path),
                "records": list(_read_jsonl(wire_path)),
            }
        raise FileNotFoundError(
            f"No {agent_type.title()} Code session '{instance_id}' found under {graphs_dir}"
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

# --- Codex rollout graph construction --------------------------------------

_CODEX_SHELL_TOOLS = {
    "shell_command", "exec_command", "local_shell", "local_shell_call",
}
_CODEX_PATCH_TOOLS = {"apply_patch", "patch"}
_CODEX_READ_TOOL_HINTS = (
    "read", "view", "find", "search", "list", "open", "grep", "fetch",
)
_CODEX_EDIT_TOOL_HINTS = ("write", "edit", "create", "replace", "patch")
_CODEX_POST_PATCH_VALIDATION_COMMANDS = {
    "curl", "sleep", "pgrep", "pidof", "ps", "lsof", "ss", "netstat",
    "nc", "ncat", "telnet", "uvicorn", "gunicorn", "node", "deno", "bun",
    "java", "pkill", "kill", "killall", "wait", "timeout",
}
_CODEX_SHELL_EDIT_COMMANDS = {
    "mkdir", "rmdir", "cp", "mv", "rm", "ln", "install", "truncate",
}
_CODEX_JS_COMMAND_RE = re.compile(
    r"\bcommand\s*:\s*(?:"
    r'"(?P<double>(?:\\.|[^"\\])*)"|'
    r"'(?P<single>(?:\\.|[^'\\])*)'|"
    r"`(?P<template>(?:\\.|[^`\\])*)`"
    r")",
    re.DOTALL,
)
_CODEX_PATCH_PATH_RE = re.compile(
    r"^\*\*\*\s+(?:Update|Add|Delete)\s+File:\s*(.+?)\s*$",
    re.MULTILINE,
)
_CODEX_DIFF_PATH_RE = re.compile(r"^\+\+\+\s+b/(.+?)\s*$", re.MULTILINE)


def _codex_json_args(value) -> dict:
    """Normalize Codex function arguments or custom-tool input to a dict."""
    if isinstance(value, dict):
        return dict(value)
    if value is None:
        return {}
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
        return {"_raw": value}
    return {"_raw": value}


def _codex_output_text(value) -> str:
    """Flatten Codex tool output while preserving structured error details."""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        chunks = []
        for item in value:
            if isinstance(item, str):
                chunks.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("output_text") \
                    or item.get("content")
                chunks.append(text if isinstance(text, str) else json.dumps(
                    item, ensure_ascii=False, default=str,
                ))
        return "\n".join(chunks)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def _codex_reasoning_text(payload: dict) -> str:
    """Return only reasoning text explicitly surfaced in the rollout."""
    chunks = []
    for key in ("summary", "content"):
        text = _codex_content_text(payload.get(key))
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def _codex_call_from_payload(payload: dict, fallback_id: int) -> dict:
    item_type = str(payload.get("type") or "")
    name = str(payload.get("name") or "")
    if item_type == "web_search_call":
        name = name or "web_search"
        raw_args = payload.get("action")
    elif item_type == "image_generation_call":
        name = name or "image_generation"
        raw_args = {
            key: payload.get(key) for key in ("revised_prompt", "status")
            if payload.get(key) is not None
        }
    else:
        raw_args = payload.get("arguments", payload.get("input"))

    call_id = str(payload.get("call_id") or payload.get("id") or f"call-{fallback_id}")
    return {
        "id": call_id,
        "name": name or item_type or "tool",
        "args": _codex_json_args(raw_args),
        "raw_input": raw_args if isinstance(raw_args, str) else "",
        "observation": "",
        "is_error": None,
        "exit_code": None,
        "item_type": item_type,
    }


def _codex_append_output(call: dict, value) -> None:
    output = _codex_output_text(value)
    if output and output not in call["observation"]:
        call["observation"] = "\n".join(
            part for part in (call["observation"], output) if part
        )


def iter_codex_steps(traj_data: dict) -> list[dict]:
    """Reconstruct visible Codex actions from a rollout JSONL stream.

    Codex does not expose private chain-of-thought. The ``thought`` field here
    contains only persisted reasoning summaries and assistant commentary that
    appeared immediately before a tool-call group.
    """
    ordered_steps: list[dict] = []
    calls_by_id: dict[str, tuple[dict, dict]] = {}
    pending_context: list[str] = []
    current_step: dict | None = None

    def start_step() -> dict:
        nonlocal current_step, pending_context
        current_step = {
            "thought": "\n\n".join(part for part in pending_context if part),
            "calls": [],
            "received_output": False,
        }
        pending_context = []
        ordered_steps.append(current_step)
        return current_step

    for record in traj_data.get("records", []):
        record_type = record.get("type")
        payload = record.get("payload")
        if not isinstance(payload, dict):
            continue
        item_type = str(payload.get("type") or "")

        if record_type == "response_item" and item_type == "reasoning":
            text = _codex_reasoning_text(payload)
            if text:
                pending_context.append(text)
            continue

        if record_type == "response_item" and item_type == "message" \
                and payload.get("role") == "assistant":
            message_text = _codex_content_text(payload.get("content"))
            if not message_text:
                continue
            if payload.get("phase") == "final_answer":
                step = start_step()
                call = {
                    "id": f"final-{len(ordered_steps)}",
                    "name": "final_answer",
                    "args": {"text": message_text},
                    "raw_input": message_text,
                    "observation": "",
                    "is_error": None,
                    "exit_code": None,
                    "item_type": "message",
                }
                step["calls"].append(call)
                current_step = None
            else:
                pending_context.append(message_text)
            continue

        if record_type == "response_item" and item_type in _CODEX_CALL_TYPES:
            if current_step is None or current_step["received_output"]:
                start_step()
            call = _codex_call_from_payload(payload, len(calls_by_id))
            current_step["calls"].append(call)
            calls_by_id[call["id"]] = (call, current_step)
            continue

        if record_type == "response_item" and item_type in _CODEX_OUTPUT_TYPES:
            call_id = str(payload.get("call_id") or "")
            match = calls_by_id.get(call_id)
            if match:
                call, step = match
                _codex_append_output(call, payload.get("output", payload))
                step["received_output"] = True
            continue

        if record_type != "event_msg":
            continue

        call_id = str(payload.get("call_id") or payload.get("callId") or "")
        match = calls_by_id.get(call_id)
        if not match:
            continue
        call, step = match

        if item_type == "exec_command_end":
            exit_code = payload.get("exit_code")
            if isinstance(exit_code, int):
                call["exit_code"] = exit_code
                call["is_error"] = exit_code != 0
            _codex_append_output(
                call,
                payload.get("formatted_output") or payload.get("aggregated_output")
                or payload.get("stdout") or payload.get("stderr"),
            )
            step["received_output"] = True
        elif item_type == "patch_apply_end":
            success = payload.get("success")
            if isinstance(success, bool):
                call["is_error"] = not success
            _codex_append_output(call, payload.get("stdout") or payload.get("stderr"))
            step["received_output"] = True
        elif item_type in {"mcp_tool_call_end", "dynamic_tool_call_response", "web_search_end"}:
            if isinstance(payload.get("success"), bool):
                call["is_error"] = not payload["success"]
            if payload.get("error"):
                call["is_error"] = True
            _codex_append_output(call, payload.get("result") or payload.get("error"))
            step["received_output"] = True

    for step in ordered_steps:
        step.pop("received_output", None)
    return [step for step in ordered_steps if step.get("calls")]


def _codex_patch_paths(text: str) -> list[str]:
    paths = _CODEX_PATCH_PATH_RE.findall(text or "")
    paths.extend(_CODEX_DIFF_PATH_RE.findall(text or ""))
    return list(dict.fromkeys(path.strip() for path in paths if path.strip()))


def _decode_codex_js_string(match: re.Match) -> str:
    value = match.group("double")
    if value is not None:
        try:
            return json.loads(f'"{value}"')
        except json.JSONDecodeError:
            return value
    value = match.group("single")
    if value is not None:
        return value.replace("\\'", "'").replace("\\n", "\n").replace("\\\\", "\\")
    value = match.group("template") or ""
    return value.replace("\\`", "`").replace("\\n", "\n").replace("\\\\", "\\")


def _codex_nested_exec_commands(source: str) -> list[str]:
    """Extract literal shell commands from modern tools.exec JavaScript."""
    commands = []
    marker_re = re.compile(r"tools\.(?:shell_command|exec_command)\s*\(")
    for marker in marker_re.finditer(source or ""):
        match = _CODEX_JS_COMMAND_RE.search(source, marker.end(), marker.end() + 20000)
        if match:
            commands.append(_decode_codex_js_string(match))
    return commands


def _codex_split_shell_commands(source: str) -> list[str]:
    """Split a shell script while keeping each heredoc body with its command."""
    commands: list[str] = []
    heredoc_lines: list[str] = []
    delimiter = ""
    strip_tabs = False

    def split_operators(text: str) -> list[str]:
        parts: list[str] = []
        start = 0
        quote = ""
        escaped = False
        index = 0
        while index < len(text):
            char = text[index]
            if escaped:
                escaped = False
                index += 1
                continue
            if char == "\\" and quote != "'":
                escaped = True
                index += 1
                continue
            if quote:
                if char == quote:
                    quote = ""
                index += 1
                continue
            if char in {"'", '"', "`"}:
                quote = char
                index += 1
                continue
            operator_length = 2 if text[index:index + 2] in {"&&", "||"} else 0
            if char == ";" or operator_length:
                part = text[start:index].strip()
                if part:
                    parts.append(part)
                index += operator_length or 1
                start = index
                continue
            index += 1
        remainder = text[start:].strip()
        if remainder:
            parts.append(remainder)
        return parts

    heredoc_re = re.compile(r"<<(?P<strip>-)?\s*(['\"]?)(?P<name>[A-Za-z_]\w*)\2")
    for raw_line in (source or "").splitlines(keepends=True):
        if delimiter:
            heredoc_lines.append(raw_line)
            candidate = raw_line.rstrip("\r\n")
            if strip_tabs:
                candidate = candidate.lstrip("\t")
            if candidate == delimiter:
                command = "".join(heredoc_lines).strip()
                if command:
                    commands.append(command)
                heredoc_lines = []
                delimiter = ""
                strip_tabs = False
            continue

        for part in split_operators(raw_line.rstrip("\r\n")):
            match = heredoc_re.search(part)
            if match:
                heredoc_lines = [f"{part}\n"]
                delimiter = match.group("name")
                strip_tabs = bool(match.group("strip"))
            else:
                commands.append(part)

    if heredoc_lines:
        commands.append("".join(heredoc_lines).strip())
    return commands


def _codex_recover_edit_command(command_text: str) -> dict | None:
    """Recover common edit commands whose nested quoting defeats ``shlex``."""
    match = re.match(r"^\s*(?P<command>sed|perl)\s+(?P<body>.+?)\s+"
                     r"(?P<path>(?:[/~.]|[A-Za-z]:[\\/])\S+)\s*$",
                     command_text or "", re.DOTALL | re.IGNORECASE)
    if not match:
        return None

    command = match.group("command").lower()
    body = match.group("body").strip()
    path = match.group("path")
    flags: dict[str, object] = {}
    remaining = body
    flag_match = re.match(r"-(?P<flags>[A-Za-z]+)(?:\s+|$)", remaining)
    if flag_match:
        for flag in flag_match.group("flags"):
            flags[flag] = True
        remaining = remaining[flag_match.end():].strip()

    args = [remaining, path] if remaining else [path]
    return {"command": command, "args": args, "flags": flags}


def _codex_parse_shell_commands(command_text: str, cmd_parser) -> list[dict]:
    """Parse atomic Codex shell actions with tolerant edit-command recovery."""
    # Let the Bash-aware parser see the complete source first. This preserves
    # quoted multiline programs such as ``node -e "..."`` whose internal
    # newlines and semicolons are not shell command boundaries.
    whole_parse = cmd_parser.parse(command_text) if command_text else []
    if whole_parse and all(
        parsed.get("command") != "complex_command" for parsed in whole_parse
    ):
        return whole_parse

    parsed_commands: list[dict] = []
    for command in _codex_split_shell_commands(command_text) or [command_text]:
        parsed = cmd_parser.parse(command) if command else []
        if len(parsed) == 1 and parsed[0].get("command") == "complex_command":
            recovered = _codex_recover_edit_command(command)
            if recovered:
                parsed = [recovered]
        if not parsed:
            parsed = [{
                "tool": "", "subcommand": "", "flags": {},
                "args": {"_raw": command}, "command": command,
            }]
        parsed_commands.extend(parsed)
    return parsed_commands


def codex_tool_phase(tool_name: str, args: dict, prev_phases: list[str]) -> str:
    """Map non-shell Codex tools onto the Graphectory phase taxonomy."""
    name = (tool_name or "").lower()
    if name in _CODEX_PATCH_TOOLS or any(hint in name for hint in _CODEX_EDIT_TOOL_HINTS):
        return "patch"
    if name == "web_search" or any(hint in name for hint in _CODEX_READ_TOOL_HINTS):
        return "localization"
    return "general"


def codex_shell_phase(parsed: dict, prev_phases: list[str]) -> str:
    """Classify Bash or PowerShell commands commonly emitted by Codex."""
    command = str(parsed.get("command") or "").strip().lower().replace("\\", "/")
    command = command.rsplit("/", 1)[-1]
    args = parsed.get("args", {})
    if isinstance(args, dict):
        arg_text = " ".join(str(value) for value in args.values())
    elif isinstance(args, (list, tuple)):
        arg_text = " ".join(str(value) for value in args)
    else:
        arg_text = str(args or "")
    flags = parsed.get("flags", {})
    if isinstance(flags, dict):
        flag_text = " ".join(
            f"{name} {value}" for name, value in flags.items()
            if value not in (False, None)
        )
    else:
        flag_text = str(flags or "")
    full_text = f"{command} {flag_text} {arg_text}".lower()
    has_patch = "patch" in prev_phases

    # Codex often invokes apply_patch through exec_command. Treat that shell
    # wrapper as an edit before the generic command mapper can label it general.
    if command == "apply_patch":
        return "patch"

    # A Node one-liner can itself be the edit. Check for filesystem mutation
    # before treating ordinary Node invocations as post-patch runtime probes.
    if command == "node" and re.search(
        r"\b(?:writefile|appendfile|rename|unlink|rm|mkdir|copyfile)(?:sync)?\s*\(",
        full_text,
    ):
        return "patch"

    # Runtime probes often redirect logs or suppress errors. Once source has
    # been edited, those redirects support validation rather than file editing.
    if command in _CODEX_POST_PATCH_VALIDATION_COMMANDS:
        return "validation" if has_patch else "general"

    try:
        from mapPhase import get_phase
    except ImportError:
        return "general"

    phase = get_phase(
        parsed.get("tool", ""), parsed.get("subcommand", ""),
        parsed.get("command", ""), parsed.get("args", {}),
        prev_phases, parsed.get("flags", {}),
    )
    if phase != "general":
        return phase

    if command in {
        "rg", "grep", "find", "fd", "cat", "ls", "dir", "tree",
        "head", "tail", "type", "get-content", "get-childitem",
        "select-string", "resolve-path",
    }:
        return "localization"
    if command == "git" and re.search(r"\b(status|diff|show|log|grep|ls-files)\b", arg_text.lower()):
        return "localization"
    if command in {"set-content", "add-content", "out-file", "new-item", "remove-item"}:
        return "patch"
    if command in _CODEX_SHELL_EDIT_COMMANDS:
        return "patch"

    validation_markers = (
        "pytest", "unittest", "npm test", "npm run test", "pnpm test",
        "yarn test", "cargo test", "go test", "dotnet test", "mvn test",
        "gradle test", "ruff check", "mypy", "pyright", "eslint",
        "black --check", "isort --check", "node --check", "tsc --noemit",
    )
    if any(marker in full_text for marker in validation_markers):
        return "validation" if has_patch else "localization"
    return "general"


def _codex_call_outcome(call: dict, observation: str,
                        tool: str = "", subcommand: str = "",
                        args: dict | None = None) -> str | None:
    if call.get("is_error") is True:
        return "failure"
    if call.get("is_error") is False:
        return "success"
    if isinstance(call.get("exit_code"), int):
        return "success" if call["exit_code"] == 0 else "failure"

    exit_match = re.search(r"\bExit code:\s*(-?\d+)\b", observation, re.IGNORECASE)
    if exit_match:
        return "success" if int(exit_match.group(1)) == 0 else "failure"
    return check_command_outcome(observation, tool, subcommand, args or {})


def _codex_action_entries(call: dict, cmd_parser, filter_cd: bool) -> list[dict]:
    """Expand one native Codex call into graphable atomic actions."""
    name = str(call.get("name") or "tool")
    lowered = name.lower()
    args = dict(call.get("args") or {})
    raw_input = str(call.get("raw_input") or args.get("_raw") or "")
    observation = str(call.get("observation") or "")
    entries: list[dict] = []

    shell_commands = []
    if lowered in _CODEX_SHELL_TOOLS:
        command_value = args.get("command") or args.get("cmd") or args.get("_raw")
        if isinstance(command_value, list):
            command_value = " ".join(str(part) for part in command_value)
        if command_value:
            shell_commands.append(str(command_value))
    elif lowered == "exec":
        shell_commands.extend(_codex_nested_exec_commands(raw_input))

    for command_text in shell_commands:
        parsed_commands = _codex_parse_shell_commands(command_text, cmd_parser)

        has_cd = False
        if filter_cd and len(parsed_commands) > 1:
            first = parsed_commands[0]
            if str(first.get("command") or "").strip().lower() == "cd":
                has_cd = True
                parsed_commands = parsed_commands[1:]

        for parsed in parsed_commands:
            if _is_shell_noop(parsed):
                continue
            entries.append({
                "parsed": parsed,
                "native_tool": name,
                "observation": observation,
                "has_cd": has_cd,
                "phase": None,
            })

    patch_source = raw_input
    has_nested_patch = lowered == "exec" and "tools.apply_patch" in raw_input
    if lowered in _CODEX_PATCH_TOOLS or has_nested_patch:
        paths = _codex_patch_paths(patch_source) or [""]
        for path in paths:
            patch_args = {"path": path} if path else {"_raw": patch_source}
            entries.append({
                "parsed": {
                    "tool": "str_replace_editor",
                    "subcommand": "str_replace",
                    "flags": {},
                    "args": patch_args,
                    "command": "",
                },
                "native_tool": "apply_patch",
                "observation": observation,
                "has_cd": False,
                "phase": "patch",
                "label": "apply_patch",
            })

    if entries:
        return entries

    entries.append({
        "parsed": {
            "tool": name,
            "subcommand": "",
            "flags": {},
            "args": args,
            "command": "",
        },
        "native_tool": name,
        "observation": observation,
        "has_cd": False,
        "phase": codex_tool_phase(name, args, []),
        "label": "final answer" if lowered == "final_answer" else name,
    })
    return entries


def _build_graph_codex(traj_data: dict, instance_id: str,
                       eval_report_path: str, cmd_parser,
                       filter_cd: bool = True,
                       unique_think: bool = True):
    """Build a Graphectory from a persisted Codex rollout stream."""
    builder = GraphBuilder()
    prev_phases_list: list[str] = []
    prev_thought = ""
    prev_step_first_node: str | None = None

    for step_idx, step in enumerate(iter_codex_steps(traj_data)):
        thought = str(step.get("thought") or "")
        calls = step.get("calls") or []
        thought_len_raw = compute_thought_length_raw(thought)
        thought_len_clean = compute_thought_length_clean(thought)
        action_entries = []
        action_parts = []
        observation_parts = []

        for call in calls:
            name = str(call.get("name") or "tool")
            args = call.get("args") or {}
            raw_input = call.get("raw_input")
            shown_args = raw_input if raw_input else json.dumps(
                args, ensure_ascii=False, indent=2, default=str,
            )
            action_parts.append(f"{name}\n{shown_args}".strip())
            observation = str(call.get("observation") or "")
            if observation:
                observation_parts.append(f"{name}: {observation}")
            for entry in _codex_action_entries(call, cmd_parser, filter_cd):
                entry["call"] = call
                action_entries.append(entry)

        action_str = "\n\n".join(action_parts)
        step_observation = "\n\n".join(observation_parts)
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

            phase = entry.get("phase")
            if phase is None:
                phase = codex_shell_phase(parsed, prev_phases_list)
            elif phase == "general":
                phase = codex_tool_phase(entry["native_tool"], args, prev_phases_list)

            node_label = entry.get("label") or (
                f"{tool}: {subcommand}" if tool and subcommand
                else (tool or command or entry["native_tool"])
            )
            outcome = _codex_call_outcome(
                entry["call"], observation, tool, subcommand,
                args if isinstance(args, dict) else {},
            )
            if outcome and isinstance(args, dict):
                args.setdefault("command_outcome", outcome)

            node_key = builder.add_or_update_node(
                node_label=node_label, args=args, flags=flags, phase=phase,
                step_idx=step_idx, tool=tool, command=command,
                subcommand=subcommand, thought_length=thought_len_raw,
                has_cd=entry["has_cd"],
            )
            node_data = builder.G.nodes[node_key]
            node_data["thought_len_raw"] = thought_len_raw
            node_data["thought_len_clean"] = thought_len_clean
            if outcome:
                node_data["command_outcome"] = outcome
            _accumulate_observation(node_data, observation)
            _accumulate_step_data(
                node_data, step_idx, thought, action_str, step_observation,
            )
            builder.add_execution_edge(
                node_key, step_idx, is_first_in_step=is_first_in_step,
                thought_length_raw=thought_len_raw if is_first_in_step else 0,
                thought_length_clean=thought_len_clean if is_first_in_step else 0,
            )
            if is_first_in_step:
                _mark_thought_continuation(
                    builder.G, prev_step_first_node, node_key,
                    prev_thought, thought,
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
    metadata = traj_data.get("metadata") or {}
    builder.G.graph.update({
        "resolution_status": resolution_status,
        "debug_difficulty": "unknown",
        "trajectory_format": CODEX_TRAJECTORY_FORMAT,
        "instance_name": metadata.get("display_name") or instance_id,
        "session_id": instance_id,
        "model": metadata.get("model") or "",
        "cwd": metadata.get("cwd") or "",
        "thought_source": "visible Codex reasoning summaries and commentary",
    })
    return builder.G


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
    """Fold compatible Kimi Code or Claude Code records into ordered steps."""
    if traj_data.get("trajectory_format") == KIMI_SWE_TOGETHER_FORMAT:
        return _iter_kimi_swe_together_steps(traj_data)

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


def _iter_kimi_swe_together_steps(traj_data: dict) -> list[dict]:
    """Reconstruct Kimi Code steps from the published ShareGPT export.

    The dataset records assistant tool calls inline and places the corresponding
    feedback in the next human turn. It is request-side data, so a session's
    final assistant reply is intentionally unavailable.
    """
    canonical_records = traj_data.get("canonical_records")
    if isinstance(canonical_records, list):
        return _iter_kimi_swe_together_canonical_steps(canonical_records)

    conversations = traj_data.get("conversations")
    if not isinstance(conversations, list):
        return []

    ordered_steps: list[dict] = []
    for index, turn in enumerate(conversations):
        if not isinstance(turn, dict) or turn.get("from") != "gpt":
            continue
        value = str(turn.get("value") or "")
        thought = _KIMI_SWE_TOGETHER_TOOL_CALL_RE.sub("", value).strip()
        next_turn = conversations[index + 1] if index + 1 < len(conversations) else {}
        observation = ""
        if isinstance(next_turn, dict) and next_turn.get("from") == "human":
            observation = str(next_turn.get("value") or "")
        is_error = "<system>ERROR: Tool execution failed.</system>" in observation

        calls = []
        for call_index, match in enumerate(_KIMI_SWE_TOGETHER_TOOL_CALL_RE.finditer(value)):
            args_text = match.group("args").strip()
            try:
                args = json.loads(args_text)
            except json.JSONDecodeError:
                args = {"_raw": args_text}
            if not isinstance(args, dict):
                args = {"_raw": args}
            calls.append({
                "id": f"sharegpt-{index}-{call_index}",
                "name": match.group("name"),
                "args": args,
                "observation": observation,
                "is_error": is_error,
            })

        if calls or thought:
            ordered_steps.append({
                "source_step": index,
                "thought": thought,
                "calls": calls,
            })
    return ordered_steps


def _iter_kimi_swe_together_canonical_steps(records: list[dict]) -> list[dict]:
    """Recover action turns from sequential canonical request snapshots."""
    ordered_steps: list[dict] = []
    seen_steps: set[str] = set()

    for record in sorted(records, key=_kimi_swe_together_call_index):
        messages = record.get("messages")
        if not isinstance(messages, list):
            continue

        assistant_index = next(
            (index for index in range(len(messages) - 1, -1, -1)
             if isinstance(messages[index], dict)
             and messages[index].get("role") == "assistant"),
            None,
        )
        if assistant_index is None:
            continue

        assistant = messages[assistant_index]
        thought = str(
            assistant.get("reasoning_content") or assistant.get("content") or ""
        ).strip()
        raw_calls = assistant.get("tool_calls")
        if not isinstance(raw_calls, list):
            raw_calls = []

        outputs_by_id: dict[str, str] = {}
        for message in messages[assistant_index + 1:]:
            if not isinstance(message, dict) or message.get("role") != "tool":
                continue
            call_id = str(message.get("tool_call_id") or message.get("toolCallId") or "")
            content = _kimi_output_text(message.get("content"))
            if call_id:
                outputs_by_id[call_id] = content

        calls = []
        for call_index, raw_call in enumerate(raw_calls):
            if not isinstance(raw_call, dict):
                continue
            function = raw_call.get("function")
            if not isinstance(function, dict):
                continue
            raw_args = function.get("arguments")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            if not isinstance(args, dict):
                args = {"_raw": args}
            call_id = str(raw_call.get("id") or "")
            observation = outputs_by_id.get(call_id, "")
            calls.append({
                "id": call_id or f"canonical-{_kimi_swe_together_call_index(record)}-{call_index}",
                "name": str(function.get("name") or "tool"),
                "args": args,
                "observation": observation,
                "is_error": "<system>ERROR: Tool execution failed.</system>" in observation,
            })

        signature = json.dumps(
            {
                "thought": thought,
                "calls": [
                    {"name": call["name"], "args": call["args"]}
                    for call in calls
                ],
            },
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
        if signature in seen_steps or (not thought and not calls):
            continue
        seen_steps.add(signature)
        ordered_steps.append({
            "source_step": _kimi_swe_together_call_index(record),
            "thought": thought,
            "calls": calls,
        })

    return ordered_steps


def kimi_tool_phase(tool_name: str, args: dict, prev_phases: list[str]) -> str:
    """Map compatible Kimi Code or Claude Code tools onto the phase taxonomy."""
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
                      unique_think: bool = True,
                      agent_type: str = "kimi"):
    """Build a graph from a compatible Kimi Code or Claude Code wire stream."""
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
    builder.G.graph["trajectory_format"] = (
        CLAUDE_CODE_TRAJECTORY_FORMAT
        if agent_type == "claude" else KIMI_TRAJECTORY_FORMAT
    )
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
    if agent_type == "codex":
        return _build_graph_codex(traj_data, instance_id, eval_report_path,
                                  cmd_parser, filter_cd, unique_think=unique_think)

    if agent_type in {"kimi", "claude", "kimi_swe_together"}:
        return _build_graph_kimi(traj_data, instance_id, eval_report_path,
                                 cmd_parser, filter_cd, unique_think=unique_think,
                                 agent_type=agent_type)

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
