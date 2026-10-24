"""
opendraco2_runs/analyze.py

``POST /api/analyze?agent=opendraco2``: one OpenDraco2 run log in, the run view
and every agent's graph out. Stateless: nothing is written anywhere.

``build_from_run``, ``to_json`` and ``render`` work on any ``RunLog``, whichever
log it was read from, so another multi-agent format (``opendraco_runs`` for the
public OpenDraco's event log) only has to produce one.
"""

import json
from pathlib import Path
from typing import Optional

import networkx as nx

from opendraco2_runs.agents import build_agent_graph
from opendraco2_runs.log import RunLog, looks_like_opendraco2, parse_run_log
from opendraco2_runs.patterns import PATTERN_CONFIG, describe, detect_patterns
from opendraco2_runs.run_graph import build_run_graph

AGENT_TYPE = "opendraco2"

__all__ = ["AGENT_TYPE", "OpenDracoError", "analyze", "looks_like_opendraco2"]


class OpenDracoError(ValueError):
    """The upload is not a usable run log; message is user-facing (400)."""


def node_link(G) -> dict:
    try:
        return nx.node_link_data(G, edges="edges")
    except TypeError:  # networkx < 3.4 has no ``edges`` keyword
        return nx.node_link_data(G)


def agent_json(G, agent_id: str, agent_type: str = AGENT_TYPE) -> dict:
    """The same shape /api/analyze returns for a single-agent format."""
    return {
        "instance_id": agent_id,
        "agent_type":  agent_type,
        "nodes":       len(G.nodes),
        "edges":       len(G.edges),
        "graph":       node_link(G),
    }


def build_from_run(run: RunLog, cmd_parser, *, agent_id: Optional[str] = None,
                   filter_cd: bool = False, unique_think: bool = True, run_phases: bool = True,
                   tool_aliases: Optional[dict] = None) -> tuple[dict, Optional[nx.MultiDiGraph]]:
    """({agent_id: graph}, run_graph or None when one agent was asked for)."""
    agents = run.agent_ids
    if agent_id is not None:
        if agent_id not in agents:
            raise OpenDracoError(f"Unknown agent_id '{agent_id}'; this run has: {', '.join(agents) or 'none'}.")
        agents = [agent_id]
    graphs = {a: build_agent_graph(run, a, cmd_parser, filter_cd=filter_cd, unique_think=unique_think,
                                   run_phases=run_phases, tool_aliases=tool_aliases)
              for a in agents}
    run_graph = None if agent_id is not None else build_run_graph(run, graphs)
    return graphs, run_graph


def build(body: bytes, cmd_parser, *, agent_id: Optional[str] = None,
          filter_cd: bool = False, unique_think: bool = True,
          run_phases: bool = True) -> tuple[RunLog, dict, Optional[nx.MultiDiGraph]]:
    """Parse an OpenDraco2 *body*; return (run, {agent_id: graph}, run_graph)."""
    run = parse_run_log(body)
    if run.started is None and not run.activations:
        raise OpenDracoError("No OpenDraco2 records found: expected a run.jsonl starting with run_started.")
    graphs, run_graph = build_from_run(run, cmd_parser, agent_id=agent_id, filter_cd=filter_cd,
                                       unique_think=unique_think, run_phases=run_phases)
    return run, graphs, run_graph


def to_json(run: RunLog, graphs: dict, run_graph, *, label: Optional[str] = None,
            agent_type: str = AGENT_TYPE, extra: Optional[dict] = None) -> dict:
    if run_graph is None:  # agent_id=…: just that agent, in the single-agent shape
        (agent_id, G), = graphs.items()
        return agent_json(G, agent_id, agent_type)
    data = {
        "run_id":     run.run_id,
        "id":         label or run.run_id,
        "agent_type": agent_type,
        "status":     run.status,
        "phase_scope": "run" if all(G.graph.get("phase_scope") == "run" for G in graphs.values()) else "agent",
    }
    data.update(extra or {})
    data.update({
        "agents":     {a: agent_json(G, a, agent_type) for a, G in graphs.items()},
        "run_graph":  node_link(run_graph),
        "patterns":   detect_patterns(run),
        "pattern_config": dict(PATTERN_CONFIG),
    })
    return data


def render(run: RunLog, graphs: dict, run_graph, assets_dir: Path, *, output: str,
           label: Optional[str], view: Optional[dict], agent_type: str = AGENT_TYPE,
           title: str = "OpenDraco2 run", extra: Optional[dict] = None) -> tuple[str, bytes]:
    """(content_type, body) for an analysed run, as JSON or the standalone page."""
    if output == "json":
        data = to_json(run, graphs, run_graph, label=label, agent_type=agent_type, extra=extra)
        return "application/json; charset=utf-8", json.dumps(data, default=str).encode()

    from opendraco2_runs.page import render_agent_page, render_run_page

    view = view or {}
    if run_graph is None:
        (_, G), = graphs.items()
        return "text/html; charset=utf-8", render_agent_page(G, assets_dir, view).encode()
    data = to_json(run, graphs, run_graph, label=label, agent_type=agent_type, extra=extra)
    page = render_run_page(data, graphs, assets_dir, view, describe(), title=title)
    return "text/html; charset=utf-8", page.encode()


def analyze(body: bytes, cmd_parser, assets_dir: Path, *, output: str = "html",
            label: Optional[str] = None, agent_id: Optional[str] = None,
            filter_cd: bool = False, unique_think: bool = True, run_phases: bool = True,
            view: Optional[dict] = None) -> tuple[str, bytes]:
    """Return (content_type, body) for the response."""
    run, graphs, run_graph = build(body, cmd_parser, agent_id=agent_id, filter_cd=filter_cd,
                                   unique_think=unique_think, run_phases=run_phases)
    return render(run, graphs, run_graph, assets_dir, output=output, label=label, view=view)
