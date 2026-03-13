#!/usr/bin/env python3
"""
Graph Generation Script for Agent Trajectories (JSON only)

This script generates trajectory graphs (JSON) from agent execution traces.
Supports SWE-agent and OpenHands trajectories across multiple models.

Usage:
    python generatejson.py --agent sa --model dsk-v3 --trajs path_to_your_trajectory_folder --eval_report path_to_your_report.json --output_dir data/samples
    python generatejson.py --agent oh --model dsk-v3 --trajs path_to_your_output.jsonl --eval_report path_to_your_report.json --output_dir data/samples

Output Structure:
    {output_dir}/SWE-agent/graphs/{model}/{instance_id}/{instance_id}.json
    {output_dir}/OpenHands/graphs/{model}/{instance_id}/{instance_id}.json
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from datasets import load_dataset
from networkx.readwrite import json_graph

from commandParser import CommandParser
# Reuse graph construction logic from buildGraph (returns json_path, html_path)
from buildGraph import build_graph_from_sa_trajectory, build_graph_from_oh_trajectory, build_graph_from_msa_trajectory


# ==================== Configuration ====================
SUPPORTED_AGENTS = {"sa", "oh", "msa"}
SUPPORTED_MODELS = {"dsk-v3", "dsk-r1", "dev", "cld-4", "gpt-5-mini"}

MODEL_NAMES = {
    "dsk-v3": "deepseek-v3",
    "dsk-r1": "deepseek-r1-0528",
    "dev": "devstral-small",
    "cld-4": "claude-sonnet-4",
    "gpt-5-mini": "gpt-5-mini",
}

AGENT_NAMES = {
    "sa": "SWE-agent",
    "oh": "OpenHands",
    "msa": "mini-swe-agent",
}

# -------------------- Data lookups --------------------
swe_bench_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
difficulty_lookup = {row["instance_id"]: row["difficulty"] for row in swe_bench_ds}


# ==================== Data Classes ====================
@dataclass
class ProcessingResult:
    """Result of processing a single trajectory."""
    instance_id: str
    status: str           # "success" or "error"
    json_path: Optional[str] = None
    error: Optional[str] = None


# ==================== Path Management ====================
def get_graph_output_dir(base_output_dir: str, agent: str, model: str) -> Path:
    """Construct the graph output directory path."""
    return Path(base_output_dir) / AGENT_NAMES[agent] / "graphs" / MODEL_NAMES[model]


# ==================== Trajectory Loaders ====================
class TrajectoryLoader:
    """Loaders for SWE-agent and OpenHands trajectories."""

    @staticmethod
    def load_sa_trajectories(trajs_path: Path) -> List[Dict[str, Any]]:
        """Load SWE-agent trajectories from directory structure.

        Directory structure:
            trajs_path/
                ├── instance-1/
                │   ├── instance-1.traj
                │   └── ...
                └── ...
        """
        trajectories = []
        if not trajs_path.is_dir():
            raise ValueError(f"SA trajectories path must be a directory: {trajs_path}")

        for instance_dir in sorted(trajs_path.iterdir()):
            if not instance_dir.is_dir():
                continue
            instance_id = instance_dir.name
            traj_file = instance_dir / f"{instance_id}.traj"

            if not traj_file.exists():
                print(f"[WARN] Missing .traj file for {instance_id}, skipping")
                continue

            try:
                with open(traj_file, "r") as f:
                    traj_data = json.load(f)
                trajectories.append({"instance_id": instance_id, "traj_data": traj_data})
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse {traj_file}: {e}")

        return trajectories

    @staticmethod
    def load_oh_trajectories(trajs_path: Path) -> List[Dict[str, Any]]:
        """Load OpenHands trajectories from output.jsonl file."""
        trajectories = []
        if not trajs_path.is_file():
            raise ValueError(f"OH trajectories path must be a file: {trajs_path}")

        with open(trajs_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    traj_data = json.loads(line)
                    instance_id = traj_data.get("instance_id")
                    if not instance_id:
                        print(f"[WARN] Line {line_num}: Missing instance_id, skipping")
                        continue
                    trajectories.append({"instance_id": instance_id, "traj_data": traj_data})
                except json.JSONDecodeError as e:
                    print(f"[ERROR] Line {line_num}: Failed to parse JSON: {e}")

        return trajectories

    @staticmethod
    def load_msa_trajectories(trajs_path: Path) -> List[Dict[str, Any]]:
        """Load mini-swe-agent trajectories from directory structure.

        Directory structure:
            trajs_path/
                ├── instance-1/
                │   ├── instance-1.traj.json
                │   └── ...
                └── ...
        """
        trajectories = []
        if not trajs_path.is_dir():
            raise ValueError(f"MSA trajectories path must be a directory: {trajs_path}")

        for instance_dir in sorted(trajs_path.iterdir()):
            if not instance_dir.is_dir():
                continue
            instance_id = instance_dir.name
            traj_file = instance_dir / f"{instance_id}.traj.json"

            if not traj_file.exists():
                print(f"[WARN] Missing .traj.json file for {instance_id}, skipping")
                continue

            try:
                with open(traj_file, "r") as f:
                    traj_data = json.load(f)
                trajectories.append({"instance_id": instance_id, "traj_data": traj_data})
            except json.JSONDecodeError as e:
                print(f"[ERROR] Failed to parse {traj_file}: {e}")

        return trajectories


# ==================== Graph Processor ====================
class GraphProcessor:
    """Process trajectories and generate JSON graphs."""

    def __init__(self, agent: str, parser: CommandParser, eval_report_path: str, output_dir: Path):
        self.agent = agent
        self.parser = parser
        self.eval_report_path = eval_report_path
        self.output_dir = output_dir

    def process_trajectory(self, instance_id: str, traj_data: Dict[str, Any]) -> ProcessingResult:
        """Process a single trajectory and generate JSON graph."""
        try:
            if self.agent == "sa":
                json_path, _ = build_graph_from_sa_trajectory(
                    traj_data=traj_data,
                    parser=self.parser,
                    instance_id=instance_id,
                    output_dir=str(self.output_dir),
                    eval_report_path=self.eval_report_path,
                )
            elif self.agent == "oh":
                json_path, _ = build_graph_from_oh_trajectory(
                    traj_data=traj_data,
                    parser=self.parser,
                    instance_id=instance_id,
                    output_dir=str(self.output_dir),
                    eval_report_path=self.eval_report_path,
                )
            elif self.agent == "msa":
                json_path, _ = build_graph_from_msa_trajectory(
                    traj_data=traj_data,
                    parser=self.parser,
                    instance_id=instance_id,
                    output_dir=str(self.output_dir),
                    eval_report_path=self.eval_report_path,
                )
            else:
                raise ValueError(f"Unsupported agent: {self.agent}")

            return ProcessingResult(instance_id=instance_id, status="success", json_path=json_path)

        except Exception as e:
            return ProcessingResult(instance_id=instance_id, status="error", error=str(e))


# ==================== Batch Processing ====================
def setup_parser_for_agent(agent: str) -> CommandParser:
    """Setup CommandParser with appropriate tool configurations."""
    parser = CommandParser()
    tool_configs = []
    if agent == "sa":
        tool_configs = [
            "data/SWE-agent/tools/edit_anthropic/config.yaml",
            "data/SWE-agent/tools/review_on_submit_m/config.yaml",
            "data/SWE-agent/tools/registry/config.yaml",
        ]
    if tool_configs:
        parser.load_tool_yaml_files(tool_configs)
    return parser


def process_batch(
    trajectories: List[Dict[str, Any]],
    processor: GraphProcessor,
    max_workers: int = 8,
) -> Dict[str, List]:
    """Process trajectories in parallel."""
    results: Dict[str, List] = {"success": [], "failed": []}
    total = len(trajectories)

    print(f"\n{'='*70}")
    print(f"Processing {total} trajectories with {max_workers} workers...")
    print(f"{'='*70}\n")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_instance = {
            executor.submit(processor.process_trajectory, traj["instance_id"], traj["traj_data"]): traj["instance_id"]
            for traj in trajectories
        }
        completed = 0
        for future in as_completed(future_to_instance):
            result = future.result()
            completed += 1
            if result.status == "success":
                results["success"].append(result)
                print(f"[{completed}/{total}] ✓ {result.instance_id}")
            else:
                results["failed"].append(result)
                print(f"[{completed}/{total}] ✗ {result.instance_id}: {result.error}")

    return results


def print_summary(results: Dict[str, List], agent: str, model: str, output_dir: Path):
    """Print processing summary."""
    success_count = len(results["success"])
    failed_count = len(results["failed"])
    total = success_count + failed_count

    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY")
    print(f"{'='*70}")
    print(f"Agent:        {AGENT_NAMES[agent]}")
    print(f"Model:        {MODEL_NAMES[model]}")
    print(f"Output:       {output_dir}")
    print(f"Succeeded:    {success_count}/{total}")
    print(f"{'='*70}\n")

    if failed_count > 0:
        print("Failed instances:")
        for result in results["failed"][:10]:
            print(f"  - {result.instance_id}: {result.error}")
        if failed_count > 10:
            print(f"  ... and {failed_count - 10} more")
        print()


# ==================== Entry Point ====================
def main():
    parser = argparse.ArgumentParser(
        description="Generate trajectory graphs (JSON only) for agent executions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # SWE-agent with DeepSeek-V3
  python %(prog)s --agent sa --model dsk-v3 --trajs sa_trajectories --eval_report report.json --output_dir output

  # OpenHands with Claude Sonnet 4
  python %(prog)s --agent oh --model cld-4 --trajs output.jsonl --eval_report report.json --output_dir output

Output Structure:
  {output_dir}/SWE-agent/graphs/deepseek-v3/{instance_id}/{instance_id}.json
  {output_dir}/OpenHands/graphs/claude-sonnet-4/{instance_id}/{instance_id}.json

Supported agents: sa (SWE-agent), oh (OpenHands)
Supported models: dsk-v3 (deepseek-v3), dsk-r1 (deepseek-r1-0528), dev (devstral-small), cld-4 (claude-sonnet-4)
        """,
    )

    parser.add_argument("--agent", type=str, required=True, choices=list(SUPPORTED_AGENTS),
                        help="Agent type: sa (SWE-agent) or oh (OpenHands)")
    parser.add_argument("--model", type=str, required=True, choices=list(SUPPORTED_MODELS),
                        help="Model type: dsk-v3, dsk-r1, dev, or cld-4")
    parser.add_argument("--trajs", type=str, required=True,
                        help="Path to trajectories (directory for SA, output.jsonl for OH)")
    parser.add_argument("--eval_report", type=str, required=True,
                        help="Path to evaluation report JSON file")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Base output directory")
    parser.add_argument("--workers", type=int, default=8,
                        help="Number of parallel workers (default: 8)")

    args = parser.parse_args()

    trajs_path = Path(args.trajs)
    eval_report_path = Path(args.eval_report)

    if not trajs_path.exists():
        print(f"[ERROR] Trajectories path does not exist: {trajs_path}")
        sys.exit(1)
    if not eval_report_path.exists():
        print(f"[ERROR] Evaluation report does not exist: {eval_report_path}")
        sys.exit(1)

    graph_output_dir = get_graph_output_dir(args.output_dir, args.agent, args.model)
    graph_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("CONFIGURATION")
    print(f"{'='*70}")
    print(f"Agent:        {AGENT_NAMES[args.agent]}")
    print(f"Model:        {MODEL_NAMES[args.model]}")
    print(f"Trajectories: {trajs_path}")
    print(f"Eval Report:  {eval_report_path}")
    print(f"Graph Output: {graph_output_dir}")
    print(f"Workers:      {args.workers}")
    print(f"{'='*70}\n")

    print("Loading trajectories...")
    try:
        if args.agent == "sa":
            trajectories = TrajectoryLoader.load_sa_trajectories(trajs_path)
        elif args.agent == "oh":
            trajectories = TrajectoryLoader.load_oh_trajectories(trajs_path)
        else:
            trajectories = TrajectoryLoader.load_msa_trajectories(trajs_path)
    except Exception as e:
        print(f"[ERROR] Failed to load trajectories: {e}")
        sys.exit(1)

    if not trajectories:
        print("[ERROR] No trajectories found")
        sys.exit(1)

    print(f"Loaded {len(trajectories)} trajectories\n")

    cmd_parser = setup_parser_for_agent(args.agent)

    processor = GraphProcessor(
        agent=args.agent,
        parser=cmd_parser,
        eval_report_path=str(eval_report_path),
        output_dir=graph_output_dir,
    )

    results = process_batch(trajectories, processor, max_workers=args.workers)
    print_summary(results, args.agent, args.model, graph_output_dir)

    if results["failed"]:
        sys.exit(1)
    print("✓ All trajectories processed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()