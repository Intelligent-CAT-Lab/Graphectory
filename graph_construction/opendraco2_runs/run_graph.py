"""
opendraco2_runs/run_graph.py

The run-level view: how the agents of one OpenDraco run worked together.

``build_run_graph`` returns a networkx MultiDiGraph whose

* nodes are the topology's nodes (agents, plus START/END when the topology has
  them), each agent carrying its activation count, phase mix (from its own
  Graphectory graph), tokens, duration and whether it ever changed the workspace;
* ``delivery`` edges are the deliveries from → to, one per topology edge id, with
  their count and the supersteps they happened in (edges the run never used keep
  ``count: 0``);
* ``route`` edges are a conditional node's choices (``route_decision``), one per
  (group, choice), with how often it was picked.

``G.graph["timeline"]`` holds the "who handed what to whom, when" data: one block
per activation (agent × superstep) and one arrow per delivery, from the activation
that produced it to every activation it caused.
"""

from collections import Counter, defaultdict
from typing import Optional

import networkx as nx

from opendraco2_runs.log import TOOL_CALL, RunLog

# Longest delivery value kept per timeline arrow; the full text is in the agent tabs.
PREVIEW_CHARS = 280


def _preview(value) -> str:
    text = value if isinstance(value, str) else ""
    return text if len(text) <= PREVIEW_CHARS else text[:PREVIEW_CHARS - 1] + "…"


def _topology(run: RunLog) -> dict:
    topology = (run.started or {}).get("topology")
    return topology if isinstance(topology, dict) else {}


def build_run_graph(run: RunLog, agent_graphs: Optional[dict] = None) -> nx.MultiDiGraph:
    agent_graphs = agent_graphs or {}
    G = nx.MultiDiGraph()
    topology = _topology(run)
    agents_cfg = (run.started or {}).get("agents")
    agents_cfg = agents_cfg if isinstance(agents_cfg, dict) else {}

    for node in topology.get("nodes", []) or []:
        if isinstance(node, dict) and isinstance(node.get("id"), str):
            G.add_node(node["id"], kind=str(node.get("kind") or "agent"),
                       mode=str(node.get("mode") or "unconditional"))

    for agent_id in run.agent_ids:
        acts = run.agent_activations(agent_id)
        abandoned = [a for a in run.agent_activations(agent_id, include_abandoned=True)
                     if a.status == "abandoned"]
        tokens = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}
        duration = 0.0
        for a in acts:
            for key, value in a.usage.items():
                tokens[key] += value
            duration += a.duration_s or 0.0
        phases: Counter = Counter()
        graph = agent_graphs.get(agent_id)
        if graph is not None:
            for _, data in graph.nodes(data=True):
                phases.update(p for p in data.get("phases") or [] if isinstance(p, str))
        changed = sorted({p for a in acts if a.commit
                          for p in a.commit.get("changed_paths") or [] if isinstance(p, str)})
        cfg = agents_cfg.get(agent_id) if isinstance(agents_cfg.get(agent_id), dict) else {}
        model = cfg.get("model")
        if isinstance(model, dict):
            model = model.get("name") or model.get("model")
        if agent_id not in G:
            G.add_node(agent_id, kind="agent", mode="unconditional")
        G.nodes[agent_id].update(
            activations=len(acts),
            abandoned_activations=len(abandoned),
            running_activations=sum(1 for a in acts if a.status == "running"),
            tool_calls=sum(a.count(TOOL_CALL) for a in acts),
            phase_mix=dict(sorted(phases.items())),
            tokens=tokens,
            duration_s=round(duration, 3),
            changed_workspace=bool(changed) or any(a.commit for a in acts),
            commits=sum(1 for a in acts if a.commit),
            changed_paths=changed,
            model=model if isinstance(model, str) else None,
        )

    # Delivery edges: one per topology edge id, aggregated over the run.
    seen: dict[str, dict] = {}
    for edge in topology.get("edges", []) or []:
        if isinstance(edge, dict) and isinstance(edge.get("id"), str):
            seen[edge["id"]] = {"source": str(edge.get("source") or ""), "target": str(edge.get("target") or ""),
                                "slot": str(edge.get("slot") or ""), "count": 0, "supersteps": set(),
                                "from_parent_run": 0}
    for delivery in run.deliveries.values():
        edge_id = str(delivery.get("edge_id") or "")
        entry = seen.setdefault(edge_id, {
            "source": str(delivery.get("from_agent") or ""), "target": str(delivery.get("to_agent") or ""),
            "slot": str(delivery.get("slot") or ""), "count": 0, "supersteps": set(), "from_parent_run": 0,
        })
        entry["count"] += 1
        superstep = run.delivery_superstep(delivery)
        entry["supersteps"].add("start" if superstep is None else superstep)
    for act in run.activations:  # a fork's causes recorded only in the parent run
        if act.missing_causes and run.forked_from:
            for edge_entry in seen.values():
                if edge_entry["target"] == act.agent_id and edge_entry["count"] == 0:
                    edge_entry["from_parent_run"] += len(act.missing_causes)
                    break
    for edge_id, entry in seen.items():
        for end in (entry["source"], entry["target"]):
            if end and end not in G:
                G.add_node(end, kind="external", mode="unconditional")
        steps = sorted(entry["supersteps"], key=lambda s: (s != "start", s if isinstance(s, int) else -1))
        G.add_edge(entry["source"], entry["target"], key=f"delivery:{edge_id}", type="delivery",
                   edge_id=edge_id, slot=entry["slot"], count=entry["count"], supersteps=steps,
                   from_parent_run=entry["from_parent_run"])

    # Route edges: a conditional node's choices.
    routes: Counter = Counter()
    route_meta: dict[tuple, dict] = {}
    for decision in run.route_decisions:
        agent_id = str(decision.get("agent_id") or "")
        group = str(decision.get("group") or "")
        choice = str(decision.get("choice") or "")
        key = (agent_id, group, choice)
        routes[key] += 1
        meta = route_meta.setdefault(key, {"outcomes": [], "supersteps": set()})
        meta["outcomes"] = [str(o) for o in decision.get("outcomes") or []]
        act = run.activation(str(decision.get("parent_uuid") or ""))
        if act is not None:
            meta["supersteps"].add(act.superstep)
    group_totals: Counter = Counter()
    for (agent_id, group, _), count in routes.items():
        group_totals[(agent_id, group)] += count
    for (agent_id, group, choice), count in routes.items():
        target = choice if choice in G else choice or "?"
        if target not in G:
            G.add_node(target, kind="external", mode="unconditional")
        G.add_edge(agent_id, target, key=f"route:{group}:{choice}", type="route", group=group or None,
                   choice=choice, count=count, of=group_totals[(agent_id, group)],
                   outcomes=route_meta[(agent_id, group, choice)]["outcomes"],
                   supersteps=sorted(route_meta[(agent_id, group, choice)]["supersteps"]))

    G.graph.update(
        run_id=run.run_id,
        status=run.status,
        framework_version=(run.started or {}).get("framework_version"),
        forked_from=run.forked_from,
        resumed_at=[r.get("superstep") for r in run.resumes],
        truncated=run.truncated,
        skipped_lines=run.skipped_lines,
        supersteps=max((a.superstep for a in run.activations), default=0),
        timeline=_timeline(run),
        errors=[{"agent_id": e.get("agent_id"), "message": e.get("message")} for e in run.errors],
    )
    G.graph.update(run.meta)
    return G


def _timeline(run: RunLog) -> dict:
    caused: dict[str, list[str]] = defaultdict(list)
    for act in run.activations:
        for uuid in act.new_causes:
            caused[uuid].append(act.uuid)
    blocks = [{
        "uuid": a.uuid,
        "agent_id": a.agent_id,
        "superstep": a.superstep,
        "local_turn": a.local_turn,
        "status": a.status,
        "rerun_of": a.rerun_of,
        "tokens": a.usage["total_tokens"],
        "duration_s": a.duration_s,
        "stop_reason": a.stop_reason,
        "tool_calls": a.count(TOOL_CALL),
        "changed_paths": [p for p in (a.commit or {}).get("changed_paths") or [] if isinstance(p, str)],
        "from_parent_run": len(a.missing_causes) if run.forked_from else 0,
        "started_at": a.timestamp or None,
    } for a in run.activations]
    arrows = [{
        "uuid": d.get("uuid"),
        "from_agent": d.get("from_agent"),
        "to_agent": d.get("to_agent"),
        "edge_id": d.get("edge_id"),
        "from_activation": d.get("parent_uuid") if run.activation(str(d.get("parent_uuid") or "")) else None,
        "to_activations": caused.get(str(d.get("uuid")), []),
        "superstep": run.delivery_superstep(d),
        "preview": _preview(d.get("value")),
    } for d in run.deliveries.values()]
    return {"activations": blocks, "deliveries": arrows}
