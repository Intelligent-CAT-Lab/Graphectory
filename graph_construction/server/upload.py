"""
server/upload.py

Stage an uploaded trajectory so the normal loading pipeline can read it.

POST /api/analyze receives a trajectory in the request body rather than a path
on disk. Instead of re-implementing format detection for in-memory data, the
body is written into a private temporary directory laid out exactly like the
on-disk datasets, and then handed to the same detect_agent_type() →
scan_trajectories() → load_trajectory() → build_graph() chain that
live_graph_server.py uses for --trajs.

The caller owns the directory's lifetime (use it inside
tempfile.TemporaryDirectory) and must delete it once the graph is built.
"""

import json
import re
from pathlib import Path
from typing import Optional

from server.graph_builder import detect_agent_type, load_trajectory, scan_trajectories

AGENT_TYPES = {"sa", "msa", "oh", "claude", "codex"}

# Used both as a file name and as the graph's instance_id, so it must never be
# able to name another directory.
_INSTANCE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
DEFAULT_INSTANCE_ID = "upload"


class UploadError(ValueError):
    """The upload cannot be turned into a trajectory; message is user-facing."""


def valid_instance_id(value: str) -> bool:
    return bool(_INSTANCE_ID_RE.match(value)) and ".." not in value


def _json_object_agent(obj: dict) -> str:
    """Guess the format of a single-JSON-object trajectory from its keys."""
    if "messages" in obj or str(obj.get("trajectory_format", "")).startswith("mini-swe-agent"):
        return "msa"
    # SWE-agent .traj files carry both "trajectory" and "history"; OpenHands
    # entries only "history" — so test "trajectory" first.
    if "trajectory" in obj:
        return "sa"
    if "history" in obj:
        return "oh"
    raise UploadError(
        "Unrecognised JSON trajectory: expected a SWE-agent .traj ('trajectory'), "
        "mini-swe-agent .traj.json ('messages') or OpenHands entry ('history')."
    )


def stage_upload(body: bytes, workdir: Path, agent_hint: Optional[str] = None,
                 instance_id: str = DEFAULT_INSTANCE_ID) -> tuple[Path, str]:
    """Write *body* under *workdir* and return (graphs_dir, agent_type).

    A body that parses as one JSON object is a SWE-agent, mini-swe-agent or
    OpenHands trajectory; anything else is treated as JSONL (Claude Code,
    Codex or OpenHands output.jsonl). *agent_hint* overrides detection.
    """
    if agent_hint is not None and agent_hint not in AGENT_TYPES:
        raise UploadError(f"Unknown agent '{agent_hint}'; expected one of {sorted(AGENT_TYPES)}.")
    if not valid_instance_id(instance_id):
        raise UploadError("Invalid id: use letters, digits, '.', '_' or '-' (max 128).")

    text = body.decode("utf-8", errors="replace")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        agent = agent_hint or _json_object_agent(parsed)
        if agent == "msa":
            target = workdir / instance_id / f"{instance_id}.traj.json"
        elif agent == "sa":
            target = workdir / instance_id / f"{instance_id}.traj"
        elif agent == "oh":
            # OpenHands reads one entry per line of output.jsonl.
            parsed.setdefault("instance_id", instance_id)
            target = workdir / "output.jsonl"
            text = json.dumps(parsed)
        else:
            raise UploadError(f"A single JSON object cannot be a '{agent}' trajectory; send JSONL.")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        return (target if agent == "oh" else workdir), agent

    if parsed is not None:
        raise UploadError("Body must be a JSON object or JSONL records.")

    # JSONL: Claude Code transcripts and Codex rollouts are recognised by
    # scanning a directory; OpenHands is the remaining JSONL format and is
    # read from the file itself.
    target = workdir / f"{instance_id}.jsonl"
    target.write_text(text, encoding="utf-8")
    agent = agent_hint or detect_agent_type(workdir)
    if agent in {"sa", "msa"}:
        if agent_hint:
            raise UploadError(f"A '{agent}' trajectory is a single JSON object, not JSONL.")
        agent = "oh"
    return (target if agent == "oh" else workdir), agent


def load_staged(graphs_dir: Path, agent_type: str,
                instance_id: Optional[str] = None) -> tuple[dict, str]:
    """Return (traj_data, instance_id) for the staged upload.

    *instance_id* picks one entry when the upload holds several (an OpenHands
    output.jsonl); otherwise the first trajectory found is used.
    """
    found = [entry["instance_id"] for entry in scan_trajectories(graphs_dir, None, agent_type)]
    if not found:
        raise UploadError(f"No {agent_type} trajectory found in the upload.")
    chosen = instance_id if instance_id in found else found[0]
    try:
        return load_trajectory(graphs_dir, chosen, agent_type), chosen
    except FileNotFoundError as exc:
        raise UploadError(str(exc)) from exc
