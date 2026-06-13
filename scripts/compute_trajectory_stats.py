#!/usr/bin/env python3
"""
Compute descriptive statistics for raw trajectories and Graphectory graphs.

This script measures:
  - Raw trajectory line and character counts
  - Average step counts
  - Graphectory node and edge counts
  - How many action occurrences are deduplicated into shared nodes
  - Revisit/loop counts (occurrences after a node's first visit)
  - Maximum out-degree (graph "visual width")
  - Shortest execution-path length from the first node to the last node

By default, the script reuses the same graph-building pipeline as the live
viewer so the reported numbers match the tool's actual semantics.

Examples
--------
  # SWE-agent directory of .traj files
  python scripts/compute_trajectory_stats.py ^
      --trajs data/samples/SWE-agent/trajectories/anthropic_filemap__deepseek--deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test ^
      --eval-report data/samples/SWE-agent/reports/deepseek-chat.json

  # OpenHands output.jsonl
  python scripts/compute_trajectory_stats.py ^
      --trajs data/samples/OpenHands/trajectories/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/output.jsonl ^
      --eval-report data/samples/OpenHands/trajectories/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/report.json
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import networkx as nx

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = REPO_ROOT / "graph_construction"
if str(GRAPH_ROOT) not in sys.path:
    sys.path.insert(0, str(GRAPH_ROOT))

from commandParser import CommandParser
from server.graph_builder import build_graph, load_trajectory, scan_trajectories


@dataclass(frozen=True)
class DatasetSource:
    """One discoverable trajectory source within a corpus."""

    source_id: str
    trajs_path: Path
    agent_type: str
    eval_report: str | None = None


def detect_agent_type(trajs_path: Path) -> str:
    """Infer the agent family from the supplied trajectory path."""
    if trajs_path.is_file() and trajs_path.suffix == ".jsonl":
        return "oh"
    if trajs_path.is_dir():
        if any(trajs_path.rglob("*.traj.json")):
            return "msa"
        if any(trajs_path.rglob("*.traj")):
            return "sa"
    raise ValueError(
        f"Could not infer agent type from {trajs_path}. "
        "Pass a SWE-agent directory of .traj files, a mini-swe-agent directory "
        "of .traj.json files, or an OpenHands output.jsonl file."
    )


def discover_sources(
    trajs_path: Path,
    agent_hint: str = "auto",
    eval_report: str | None = None,
) -> list[DatasetSource]:
    """Discover one or more dataset sources under *trajs_path*.

    The path may be:
      - a single dataset root (e.g., a SWE-agent model directory)
      - a single OpenHands `output.jsonl`
      - a corpus root containing multiple model folders across agent families

    When an OpenHands source is auto-discovered, a sibling `report.json` is
    attached automatically if present.
    """
    if trajs_path.is_file():
        if trajs_path.suffix != ".jsonl":
            raise ValueError(f"Unsupported file input: {trajs_path}")
        return [
            DatasetSource(
                source_id=trajs_path.parent.name,
                trajs_path=trajs_path,
                agent_type="oh",
                eval_report=eval_report,
            )
        ]

    sources_by_key: dict[tuple[str, str], DatasetSource] = {}

    def add_source(source_root: Path, agent_type: str, report: str | None = None) -> None:
        if agent_hint != "auto" and agent_hint != agent_type:
            return
        label_path = source_root.parent if source_root.is_file() else source_root
        source_id = str(label_path.relative_to(trajs_path)) if label_path != trajs_path else label_path.name
        key = (str(source_root), agent_type)
        final_report = eval_report if eval_report else report
        sources_by_key[key] = DatasetSource(
            source_id=source_id,
            trajs_path=source_root,
            agent_type=agent_type,
            eval_report=final_report,
        )

    # OpenHands model folders: .../<model-run>/output.jsonl
    for jsonl in sorted(trajs_path.rglob("output.jsonl")):
        sibling_report = jsonl.with_name("report.json")
        add_source(jsonl, "oh", str(sibling_report) if sibling_report.exists() else None)

    # mini-swe-agent roots: .../<model>/<instance>/<instance>.traj.json  -> source root is <model>
    for traj_file in sorted(trajs_path.rglob("*.traj.json")):
        if len(traj_file.parents) >= 2:
            add_source(traj_file.parents[1], "msa")

    # SWE-agent roots: .../<model-dir>/*.traj  -> source root is parent directory
    for traj_file in sorted(trajs_path.rglob("*.traj")):
        source_root = traj_file.parent
        # Some corpora store each instance as <model-dir>/<instance>/<instance>.traj.
        # In that case, group all such instances back under the model directory.
        if source_root.name == traj_file.stem and source_root.parent != trajs_path:
            source_root = source_root.parent
        add_source(source_root, "sa")

    if sources_by_key:
        return sorted(
            sources_by_key.values(),
            key=lambda src: (src.agent_type, src.source_id.lower()),
        )

    # Fall back to treating the path as a single dataset root.
    agent_type = detect_agent_type(trajs_path) if agent_hint == "auto" else agent_hint
    return [
        DatasetSource(
            source_id=trajs_path.name,
            trajs_path=trajs_path,
            agent_type=agent_type,
            eval_report=eval_report,
        )
    ]


def setup_cmd_parser(extra_configs: list[str] | None = None) -> CommandParser:
    """Load the same tool configs used by the live viewer."""
    parser = CommandParser()
    tool_configs = [
        REPO_ROOT / "data" / "SWE-agent" / "tools" / "edit_anthropic" / "config.yaml",
        REPO_ROOT / "data" / "SWE-agent" / "tools" / "review_on_submit_m" / "config.yaml",
        REPO_ROOT / "data" / "SWE-agent" / "tools" / "registry" / "config.yaml",
    ]
    existing = [str(path) for path in tool_configs if path.exists()]
    if extra_configs:
        existing.extend(extra_configs)
    if existing:
        parser.load_tool_yaml_files(existing)
    return parser


def should_search_instance_configs(trajs_path: Path) -> bool:
    """Return True if the dataset appears to carry per-instance config YAMLs."""
    if not trajs_path.is_dir():
        return False
    try:
        next(trajs_path.rglob("*.config.yaml"))
        return True
    except StopIteration:
        return False


def find_trajectory_file(trajs_path: Path, instance_id: str, agent_type: str) -> Path | None:
    """Return the backing raw trajectory file when the format is file-backed."""
    if agent_type == "sa":
        for traj_file in trajs_path.rglob(f"{instance_id}.traj"):
            return traj_file
    elif agent_type == "msa":
        canonical = trajs_path / instance_id / f"{instance_id}.traj.json"
        if canonical.exists():
            return canonical
        for traj_file in trajs_path.rglob(f"{instance_id}.traj.json"):
            return traj_file
    return None


def raw_text_metrics(
    trajs_path: Path,
    instance_id: str,
    agent_type: str,
    traj_data: dict[str, Any],
) -> tuple[int, int]:
    """Return (line_count, char_count) for one trajectory.

    For file-backed formats we count the raw file contents directly.
    For OpenHands JSONL entries we pretty-print the single entry before counting
    so each trajectory's size is still meaningful instead of every run
    appearing as a single raw line.
    """
    file_path = find_trajectory_file(trajs_path, instance_id, agent_type)
    if file_path is not None:
        text = file_path.read_text(encoding="utf-8", errors="replace")
        return len(text.splitlines()), len(text)

    pretty = json.dumps(traj_data, indent=2, ensure_ascii=False)
    return len(pretty.splitlines()), len(pretty)


def iter_openhands_entries(jsonl_path: Path) -> dict[str, Any]:
    """Yield parsed OpenHands entries from one output.jsonl file."""
    with jsonl_path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("instance_id"):
                yield entry


def build_exec_graph(G: nx.MultiDiGraph) -> nx.DiGraph:
    """Collapse execution edges into a simple directed graph."""
    exec_graph = nx.DiGraph()
    exec_graph.add_nodes_from(G.nodes)
    for src, dst, data in G.edges(data=True):
        if data.get("type") == "exec":
            exec_graph.add_edge(src, dst)
    return exec_graph


def select_terminal_nodes(G: nx.MultiDiGraph, exec_graph: nx.DiGraph) -> tuple[str | None, str | None]:
    """Pick start/end nodes based on earliest/latest trajectory occurrence.

    We prefer nodes that are execution sources/sinks when there is a tie on the
    first or last step index.
    """
    step_ranges: list[tuple[str, int, int]] = []
    for node, data in G.nodes(data=True):
        step_indices = data.get("step_indices") or []
        if not step_indices:
            continue
        step_ranges.append((node, min(step_indices), max(step_indices)))

    if not step_ranges:
        return None, None

    first_step = min(item[1] for item in step_ranges)
    last_step = max(item[2] for item in step_ranges)

    start_candidates = [node for node, min_step, _ in step_ranges if min_step == first_step]
    end_candidates = [node for node, _, max_step in step_ranges if max_step == last_step]

    start_node = next(
        (node for node in start_candidates if exec_graph.in_degree(node) == 0),
        sorted(start_candidates)[0],
    )
    end_node = next(
        (node for node in end_candidates if exec_graph.out_degree(node) == 0),
        sorted(end_candidates)[-1],
    )
    return start_node, end_node


def graph_metrics(G: nx.MultiDiGraph) -> dict[str, Any]:
    """Compute the graph-level metrics used in the evaluation section."""
    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    total_occurrences = 0
    revisited_node_count = 0

    for _, data in G.nodes(data=True):
        occurrences = len(data.get("step_indices") or [])
        total_occurrences += occurrences
        if occurrences > 1:
            revisited_node_count += 1

    deduplicated_nodes = max(total_occurrences - node_count, 0)
    dedup_shortening_pct = (
        (deduplicated_nodes / total_occurrences) * 100.0 if total_occurrences else 0.0
    )

    full_out_degree_values = [degree for _, degree in G.out_degree()]
    max_out_degree = max(full_out_degree_values, default=0)

    exec_graph = build_exec_graph(G)
    exec_out_degree_values = [degree for _, degree in exec_graph.out_degree()]
    max_exec_out_degree = max(exec_out_degree_values, default=0)

    loop_components = 0
    loop_nodes = 0
    for component in nx.strongly_connected_components(exec_graph):
        component_set = set(component)
        has_self_loop = any(exec_graph.has_edge(node, node) for node in component_set)
        if len(component_set) > 1 or has_self_loop:
            loop_components += 1
            loop_nodes += len(component_set)

    start_node, end_node = select_terminal_nodes(G, exec_graph)
    shortest_exec_path = None
    if start_node is not None and end_node is not None:
        try:
            shortest_exec_path = nx.shortest_path_length(exec_graph, start_node, end_node)
        except nx.NetworkXNoPath:
            shortest_exec_path = None

    return {
        "graph_nodes": node_count,
        "graph_edges": edge_count,
        "total_action_occurrences": total_occurrences,
        "deduplicated_nodes": deduplicated_nodes,
        "revisited_node_count": revisited_node_count,
        "loop_count": loop_components,
        "loop_node_count": loop_nodes,
        "dedup_shortening_pct": dedup_shortening_pct,
        "max_out_degree": max_out_degree,
        "max_exec_out_degree": max_exec_out_degree,
        "shortest_exec_path_length": shortest_exec_path,
        "start_node": start_node,
        "end_node": end_node,
    }


def safe_mean(values: list[float]) -> float | None:
    numeric = [value for value in values if value is not None]
    if not numeric:
        return None
    return statistics.fmean(numeric)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate dataset-level averages from per-instance rows."""
    if not rows:
        return {"trajectory_count": 0}

    summary = {
        "trajectory_count": len(rows),
        "avg_raw_lines": safe_mean([row["raw_lines"] for row in rows]),
        "avg_raw_chars": safe_mean([row["raw_chars"] for row in rows]),
        "avg_step_count": safe_mean([row["step_count"] for row in rows]),
        "avg_graph_nodes": safe_mean([row["graph_nodes"] for row in rows]),
        "avg_graph_edges": safe_mean([row["graph_edges"] for row in rows]),
        "avg_total_action_occurrences": safe_mean([row["total_action_occurrences"] for row in rows]),
        "avg_deduplicated_nodes": safe_mean([row["deduplicated_nodes"] for row in rows]),
        "avg_revisited_node_count": safe_mean([row["revisited_node_count"] for row in rows]),
        "avg_loop_count": safe_mean([row["loop_count"] for row in rows]),
        "avg_loop_node_count": safe_mean([row["loop_node_count"] for row in rows]),
        "avg_dedup_shortening_pct": safe_mean([row["dedup_shortening_pct"] for row in rows]),
        "avg_max_out_degree": safe_mean([row["max_out_degree"] for row in rows]),
        "avg_max_exec_out_degree": safe_mean([row["max_exec_out_degree"] for row in rows]),
        "avg_shortest_exec_path_length": safe_mean(
            [row["shortest_exec_path_length"] for row in rows if row["shortest_exec_path_length"] is not None]
        ),
    }
    return summary


def summarize_by_source(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return one summary per discovered dataset source."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["source_id"], []).append(row)
    return {
        source_id: summarize(source_rows)
        for source_id, source_rows in sorted(grouped.items())
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print a compact, paper-friendly summary block."""
    print("\nDataset summary")
    print("-" * 60)
    for key, value in summary.items():
        label = key.replace("_", " ")
        if isinstance(value, float):
            print(f"{label:32s} {value:.2f}")
        else:
            print(f"{label:32s} {value}")


def print_source_summaries(source_summaries: dict[str, dict[str, Any]]) -> None:
    """Print a short one-line summary per source."""
    if not source_summaries:
        return
    print("\nPer-source summary")
    print("-" * 60)
    for source_id, summary in source_summaries.items():
        print(
            f"{source_id}: "
            f"{summary.get('trajectory_count', 0)} trajectories, "
            f"{(summary.get('avg_graph_nodes') or 0):.2f} avg nodes, "
            f"{(summary.get('avg_dedup_shortening_pct') or 0):.2f}% avg dedup shortening"
        )


def write_csv(rows: list[dict[str, Any]], output_csv: Path) -> None:
    fieldnames = [
        "source_id",
        "source_agent",
        "source_trajs_path",
        "instance_id",
        "status",
        "difficulty",
        "step_count",
        "raw_lines",
        "raw_chars",
        "graph_nodes",
        "graph_edges",
        "total_action_occurrences",
        "deduplicated_nodes",
        "revisited_node_count",
        "loop_count",
        "loop_node_count",
        "dedup_shortening_pct",
        "max_out_degree",
        "max_exec_out_degree",
        "shortest_exec_path_length",
        "start_node",
        "end_node",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure raw trajectory and Graphectory statistics for one dataset or a corpus root.",
    )
    parser.add_argument(
        "--trajs",
        required=True,
        help=(
            "Trajectory dataset path, OpenHands output.jsonl file, or a corpus root "
            "containing multiple trajectory datasets."
        ),
    )
    parser.add_argument(
        "--eval-report",
        default=None,
        help=(
            "Evaluation report JSON used for status labels. This is applied directly "
            "for single-dataset inputs. For discovered OpenHands datasets, sibling "
            "report.json files are also picked up automatically."
        ),
    )
    parser.add_argument("--agent", choices=["auto", "sa", "oh", "msa"], default="auto")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N trajectories.")
    parser.add_argument("--output-csv", default=None, help="Optional CSV path for per-instance metrics.")
    parser.add_argument("--output-json", default=None, help="Optional JSON path for dataset summary + rows.")
    parser.add_argument("--no-filter-cd", action="store_true", help="Keep leading cd commands instead of compressing them.")
    parser.add_argument("--no-unique-think", action="store_true", help="Collapse all think nodes instead of keeping distinct thoughts separate.")
    parser.add_argument(
        "--config-yaml",
        action="append",
        default=[],
        help="Optional extra tool config YAML to load into the parser. Can be passed multiple times.",
    )
    return parser.parse_args()


def build_rows_for_source(
    source: DatasetSource,
    cmd_parser: CommandParser,
    filter_cd: bool,
    unique_think: bool,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Build per-instance metric rows for one discovered source."""
    graphs_dir_for_build = source.trajs_path if should_search_instance_configs(source.trajs_path) else None
    entries = scan_trajectories(source.trajs_path, source.eval_report, agent_type=source.agent_type)
    meta_by_id = {meta["instance_id"]: meta for meta in entries}

    rows: list[dict[str, Any]] = []

    def append_row(instance_id: str, traj_data: dict[str, Any]) -> None:
        meta = meta_by_id.get(instance_id, {})
        raw_lines, raw_chars = raw_text_metrics(source.trajs_path, instance_id, source.agent_type, traj_data)
        G = build_graph(
            traj_data,
            instance_id,
            source.eval_report,
            cmd_parser,
            graphs_dir=graphs_dir_for_build,
            filter_cd=filter_cd,
            agent_type=source.agent_type,
            unique_think=unique_think,
        )
        metrics = graph_metrics(G)
        rows.append(
            {
                "source_id": source.source_id,
                "source_agent": source.agent_type,
                "source_trajs_path": str(source.trajs_path),
                "instance_id": instance_id,
                "status": meta.get("status", "unknown"),
                "difficulty": meta.get("difficulty", "unknown"),
                "step_count": meta.get("step_count", 0),
                "raw_lines": raw_lines,
                "raw_chars": raw_chars,
                **metrics,
            }
        )

    if source.agent_type == "oh":
        seen = 0
        for entry in iter_openhands_entries(source.trajs_path):
            instance_id = entry["instance_id"]
            if instance_id not in meta_by_id:
                continue
            append_row(instance_id, entry)
            seen += 1
            if limit is not None and seen >= limit:
                break
        return rows

    for meta in entries[:limit] if limit is not None else entries:
        instance_id = meta["instance_id"]
        traj_data = load_trajectory(source.trajs_path, instance_id, agent_type=source.agent_type)
        append_row(instance_id, traj_data)

    return rows


def main() -> int:
    args = parse_args()
    trajs_path = Path(args.trajs)
    if not trajs_path.exists():
        raise FileNotFoundError(f"--trajs path does not exist: {trajs_path}")

    eval_report = str(Path(args.eval_report)) if args.eval_report else None
    filter_cd = not args.no_filter_cd
    unique_think = not args.no_unique_think

    extra_configs = [str(Path(path)) for path in args.config_yaml]
    cmd_parser = setup_cmd_parser(extra_configs)
    sources = discover_sources(trajs_path, agent_hint=args.agent, eval_report=eval_report)

    print(f"Discovered {len(sources)} source(s)")
    for source in sources:
        report_label = source.eval_report if source.eval_report else "(none)"
        print(f"  - [{source.agent_type}] {source.source_id} :: {source.trajs_path} :: report={report_label}")

    rows: list[dict[str, Any]] = []
    remaining = args.limit
    for source in sources:
        source_limit = remaining if remaining is not None else None
        source_rows = build_rows_for_source(
            source,
            cmd_parser,
            filter_cd=filter_cd,
            unique_think=unique_think,
            limit=source_limit,
        )
        rows.extend(source_rows)
        if remaining is not None:
            remaining -= len(source_rows)
            if remaining <= 0:
                break

    summary = summarize(rows)
    source_summaries = summarize_by_source(rows)
    print_summary(summary)
    print_source_summaries(source_summaries)

    if args.output_csv:
        output_csv = Path(args.output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        write_csv(rows, output_csv)
        print(f"\nWrote CSV to {output_csv}")

    if args.output_json:
        output_json = Path(args.output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": {
                "trajs": str(trajs_path),
                "eval_report": eval_report,
                "agent_hint": args.agent,
                "filter_cd": filter_cd,
                "unique_think": unique_think,
            },
            "sources": [
                {
                    "source_id": source.source_id,
                    "trajs_path": str(source.trajs_path),
                    "agent_type": source.agent_type,
                    "eval_report": source.eval_report,
                }
                for source in sources
            ],
            "summary": summary,
            "source_summaries": source_summaries,
            "rows": rows,
        }
        output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON to {output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
