"""
opendraco_runs/patterns.py

Multi-agent patterns detected from the log's structure alone (no model involved).

Every threshold lives in ``PATTERN_CONFIG``; ``PATTERN_DESCRIPTIONS`` is the one
sentence the page shows for each pattern kind.
"""

from collections import Counter, defaultdict
from typing import Optional

from opendraco_runs.log import TOOL_CALL, RunLog

PATTERN_CONFIG = {
    # N: deliveries alternating A→B→A… with no workspace commit in between.
    "ping_pong_min_exchanges": 4,
    # M: activations of an agent that never commits, never calls a tool, and only
    # passes its output on.
    "stalled_min_activations": 3,
    # K: times one route_decision group picks the same target.
    "router_churn_min_repeats": 3,
    # X: share of all activations or all tokens taken by one agent (needs at least
    # two active agents).
    "hot_agent_share": 0.7,
}

PATTERN_DESCRIPTIONS = {
    "ping_pong": "Two agents handed work back and forth {ping_pong_min_exchanges} or more times "
                 "in a row without either one changing the workspace.",
    "stalled_handoff": "An agent was activated {stalled_min_activations} or more times but never "
                       "called a tool or changed a file. It only passed its input on.",
    "router_churn": "A conditional node sent work to the same target {router_churn_min_repeats} "
                    "or more times.",
    "hot_agent": "One agent took more than {hot_agent_share_pct}% of all activations or tokens.",
}


def describe(config: Optional[dict] = None) -> dict[str, str]:
    values = dict(PATTERN_CONFIG, **(config or {}))
    values["hot_agent_share_pct"] = round(values["hot_agent_share"] * 100)
    return {kind: text.format(**values) for kind, text in PATTERN_DESCRIPTIONS.items()}


def _seq(record: dict) -> int:
    value = record.get("global_seq")
    return value if isinstance(value, int) else record.get("_line", 0)


def detect_patterns(run: RunLog, config: Optional[dict] = None) -> list[dict]:
    cfg = dict(PATTERN_CONFIG, **(config or {}))
    agents = set(run.agent_ids)
    found: list[dict] = []
    found += _ping_pong(run, agents, cfg["ping_pong_min_exchanges"])
    found += _stalled(run, cfg["stalled_min_activations"])
    found += _router_churn(run, cfg["router_churn_min_repeats"])
    found += _hot_agent(run, cfg["hot_agent_share"])
    return found


def _ping_pong(run: RunLog, agents: set[str], minimum: int) -> list[dict]:
    commit_seqs = sorted(_seq(c) for c in run.commits)

    def commit_between(lo: int, hi: int) -> bool:
        return any(lo < s < hi for s in commit_seqs)

    by_pair: dict[frozenset, list[dict]] = defaultdict(list)
    for delivery in sorted(run.deliveries.values(), key=_seq):
        a, b = delivery.get("from_agent"), delivery.get("to_agent")
        if a in agents and b in agents and a != b:
            by_pair[frozenset((a, b))].append(delivery)

    out = []
    for pair, deliveries in by_pair.items():
        streak: list[dict] = []

        def flush():
            if len(streak) >= minimum:
                first, last = streak[0], streak[-1]
                supersteps = sorted({s for s in (run.delivery_superstep(d) for d in streak) if s is not None})
                names = sorted(pair)
                out.append({
                    "kind": "ping_pong",
                    "agents": names,
                    "supersteps": supersteps,
                    "summary": f"{names[0]} and {names[1]} exchanged {len(streak)} deliveries in a row "
                               f"(supersteps {_span(supersteps)}) with no workspace commit in between.",
                    "evidence": {"deliveries": [d.get("uuid") for d in streak],
                                 "first_from": first.get("from_agent"), "count": len(streak),
                                 "last_seq": _seq(last)},
                })

        for delivery in deliveries:
            if streak and (delivery.get("from_agent") == streak[-1].get("from_agent")
                           or commit_between(_seq(streak[-1]), _seq(delivery))):
                flush()
                streak = []
            streak.append(delivery)
        flush()
    return out


def _stalled(run: RunLog, minimum: int) -> list[dict]:
    produced_by = Counter(str(d.get("parent_uuid")) for d in run.deliveries.values())
    out = []
    for agent_id in run.agent_ids:
        acts = [a for a in run.agent_activations(agent_id) if a.status == "finished"]
        if len(acts) < minimum:
            continue
        if any(a.commit for a in acts) or any(a.count(TOOL_CALL) for a in acts):
            continue
        if not all(produced_by.get(a.uuid) for a in acts):
            continue
        supersteps = sorted({a.superstep for a in acts})
        out.append({
            "kind": "stalled_handoff",
            "agents": [agent_id],
            "supersteps": supersteps,
            "summary": f"{agent_id} ran {len(acts)} times (supersteps {_span(supersteps)}), never "
                       f"called a tool or changed a file, and passed its output on every time.",
            "evidence": {"activations": [a.uuid for a in acts]},
        })
    return out


def _router_churn(run: RunLog, minimum: int) -> list[dict]:
    counts: Counter = Counter()
    supersteps: dict[tuple, set] = defaultdict(set)
    for decision in run.route_decisions:
        key = (str(decision.get("agent_id") or ""), str(decision.get("group") or ""),
               str(decision.get("choice") or ""))
        counts[key] += 1
        act = run.activation(str(decision.get("parent_uuid") or ""))
        if act is not None:
            supersteps[key].add(act.superstep)
    out = []
    for (agent_id, group, choice), count in counts.items():
        if count < minimum:
            continue
        steps = sorted(supersteps[(agent_id, group, choice)])
        label = f"group {group}" if group else agent_id
        out.append({
            "kind": "router_churn",
            "agents": [agent_id],
            "supersteps": steps,
            "summary": f"{agent_id} ({label}) routed to {choice} {count} times "
                       f"(supersteps {_span(steps)}).",
            "evidence": {"group": group or None, "choice": choice, "count": count},
        })
    return out


def _hot_agent(run: RunLog, share: float) -> list[dict]:
    acts = [a for a in run.activations if a.status != "abandoned"]
    by_agent = Counter(a.agent_id for a in acts)
    if len(by_agent) < 2:
        return []
    tokens: Counter = Counter()
    for a in acts:
        tokens[a.agent_id] += a.usage["total_tokens"]
    total_acts, total_tokens = sum(by_agent.values()), sum(tokens.values())
    out = []
    for agent_id in by_agent:
        act_share = by_agent[agent_id] / total_acts
        tok_share = tokens[agent_id] / total_tokens if total_tokens else 0.0
        if act_share > share or tok_share > share:
            out.append({
                "kind": "hot_agent",
                "agents": [agent_id],
                "supersteps": sorted({a.superstep for a in acts if a.agent_id == agent_id}),
                "summary": f"{agent_id} took {act_share:.0%} of activations and {tok_share:.0%} of tokens.",
                "evidence": {"activation_share": round(act_share, 4), "token_share": round(tok_share, 4),
                             "activations": by_agent[agent_id], "tokens": tokens[agent_id]},
            })
    return out


def _span(steps: list[int]) -> str:
    if not steps:
        return "n/a"
    return str(steps[0]) if steps[0] == steps[-1] else f"{steps[0]}–{steps[-1]}"
