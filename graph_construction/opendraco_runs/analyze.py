"""
opendraco_runs/analyze.py

``POST /api/analyze?agent=opendraco``: one public-OpenDraco inference event log
in, the run view and every agent's graph out, for one SWE-bench instance
(``id=<instance_id>``, default the first). Stateless: nothing is written.
"""

from pathlib import Path
from typing import Optional

from opendraco2_runs.analyze import OpenDracoError, build_from_run, render
from opendraco_runs.events import EventLogError, looks_like_opendraco, parse_event_log
from opendraco_runs.tools import tool_kind

AGENT_TYPE = "opendraco"

__all__ = ["AGENT_TYPE", "analyze", "looks_like_opendraco"]


def tool_aliases(run) -> dict:
    """{tool name: kind} for every non-shell tool the run called."""
    names = {r.get("call", {}).get("name") for a in run.activations for r in a.records
             if r["type"] == "tool_call" and isinstance(r.get("call"), dict)}
    out = {}
    for name in names:
        if isinstance(name, str) and name != "bash":
            kind = tool_kind(name)
            if kind and kind != "shell":
                out[name] = kind
    return out


def analyze(body: bytes, cmd_parser, assets_dir: Path, *, output: str = "html",
            label: Optional[str] = None, agent_id: Optional[str] = None,
            filter_cd: bool = False, unique_think: bool = True, run_phases: bool = True,
            view: Optional[dict] = None) -> tuple[str, bytes]:
    """Return (content_type, body); *label* (``id=``) picks the instance."""
    try:
        run, instances = parse_event_log(body, label)
    except EventLogError as exc:
        raise OpenDracoError(str(exc)) from exc
    if not run.activations:
        raise OpenDracoError("The event log records no agent activity for "
                             f"{run.meta.get('instance_id') or 'this run'}.")
    graphs, run_graph = build_from_run(run, cmd_parser, agent_id=agent_id, filter_cd=filter_cd,
                                       unique_think=unique_think, run_phases=run_phases,
                                       tool_aliases=tool_aliases(run))
    extra = {
        "instance_id": run.meta.get("instance_id"),
        "instances": run.meta.get("instances"),
        "config_name": run.meta.get("config_name"),
    }
    return render(run, graphs, run_graph, assets_dir, output=output,
                  label=label or run.meta.get("instance_id"), view=view,
                  agent_type=AGENT_TYPE, title="OpenDraco run", extra=extra)
