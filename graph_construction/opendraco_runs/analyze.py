"""
opendraco_runs/analyze.py

``POST /api/analyze?agent=opendraco``: one OpenDraco run log in, the run view and
every agent's graph out. Stateless: nothing is written anywhere.
"""

import json
from pathlib import Path
from typing import Optional

import networkx as nx

from opendraco_runs.agents import build_agent_graph
from opendraco_runs.log import RunLog, looks_like_opendraco, parse_run_log
from opendraco_runs.patterns import PATTERN_CONFIG, describe, detect_patterns
from opendraco_runs.run_graph import build_run_graph

AGENT_TYPE = "opendraco"

__all__ = ["AGENT_TYPE", "OpenDracoError", "analyze", "looks_like_opendraco"]


class OpenDracoError(ValueError):
    """The upload is not a usable OpenDraco run log; message is user-facing (400)."""


def node_link(G) -> dict:
    try:
        return nx.node_link_data(G, edges="edges")
    except TypeError:  # networkx < 3.4 has no ``edges`` keyword
        return nx.node_link_data(G)


def agent_json(G, agent_id: str) -> dict:
    """The same shape /api/analyze returns for a single-agent format."""
    return {
        "instance_id": agent_id,
        "agent_type":  AGENT_TYPE,
        "nodes":       len(G.nodes),
        "edges":       len(G.edges),
        "graph":       node_link(G),
    }


def build(body: bytes, cmd_parser, *, agent_id: Optional[str] = None,
          filter_cd: bool = False, unique_think: bool = True,
          run_phases: bool = True) -> tuple[RunLog, dict, Optional[nx.MultiDiGraph]]:
    """Parse *body*; return (run, {agent_id: graph}, run_graph or None for one agent)."""
    run = parse_run_log(body)
    if run.started is None and not run.activations:
        raise OpenDracoError("No OpenDraco records found: expected a run.jsonl starting with run_started.")
    agents = run.agent_ids
    if agent_id is not None:
        if agent_id not in agents:
            raise OpenDracoError(f"Unknown agent_id '{agent_id}'; this run has: {', '.join(agents) or 'none'}.")
        agents = [agent_id]
    graphs = {a: build_agent_graph(run, a, cmd_parser, filter_cd=filter_cd,
                                   unique_think=unique_think, run_phases=run_phases)
              for a in agents}
    run_graph = None if agent_id is not None else build_run_graph(run, graphs)
    return run, graphs, run_graph


def to_json(run: RunLog, graphs: dict, run_graph, *, label: Optional[str] = None) -> dict:
    if run_graph is None:  # agent_id=…: just that agent, in the single-agent shape
        (agent_id, G), = graphs.items()
        return agent_json(G, agent_id)
    return {
        "run_id":     run.run_id,
        "id":         label or run.run_id,
        "agent_type": AGENT_TYPE,
        "status":     run.status,
        "phase_scope": "run" if all(G.graph.get("phase_scope") == "run" for G in graphs.values()) else "agent",
        "agents":     {a: agent_json(G, a) for a, G in graphs.items()},
        "run_graph":  node_link(run_graph),
        "patterns":   detect_patterns(run),
        "pattern_config": dict(PATTERN_CONFIG),
    }


def analyze(body: bytes, cmd_parser, assets_dir: Path, *, output: str = "html",
            label: Optional[str] = None, agent_id: Optional[str] = None,
            filter_cd: bool = False, unique_think: bool = True, run_phases: bool = True,
            view: Optional[dict] = None) -> tuple[str, bytes]:
    """Return (content_type, body) for the response."""
    run, graphs, run_graph = build(body, cmd_parser, agent_id=agent_id, filter_cd=filter_cd,
                                   unique_think=unique_think, run_phases=run_phases)
    if output == "json":
        data = to_json(run, graphs, run_graph, label=label)
        return "application/json; charset=utf-8", json.dumps(data, default=str).encode()

    from opendraco_runs.page import render_agent_page, render_run_page

    view = view or {}
    if run_graph is None:
        (only_id, G), = graphs.items()
        return "text/html; charset=utf-8", render_agent_page(G, assets_dir, view).encode()
    data = to_json(run, graphs, run_graph, label=label)
    page = render_run_page(data, graphs, assets_dir, view, describe())
    return "text/html; charset=utf-8", page.encode()
