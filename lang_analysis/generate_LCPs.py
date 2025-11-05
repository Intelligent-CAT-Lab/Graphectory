#!/usr/bin/env python3
"""
Mine Longest Common Pattern (LCP) of sequences from trajectory graphs.

- Uses extractSeq to flatten graphs
- Uses buildPhases or buildLang to generate sequences with RLE
- Uses PatternMiner to compute LCPs
- Supports both phase and language role sequences

Three modes:
1. Multi-mode (default): Scans all agents/models, outputs unified matrix
   Output: {output_dir}/LCP/{sequence_type}/all_lcp_matrix.txt

2. Agent-specific mode (--agent): Processes all default models for given agent
   Output: {output_dir}/LCP/{sequence_type}/{agent}_lcp_matrix.txt

3. Single-mode (--agent + --model): Processes specific agent/model
   Output: {output_dir}/LCP/{sequence_type}/{agent}_{model}_lcp_matrix.txt
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Literal
from collections import defaultdict

from lang_construction.extractSeq import extract_node_sequence
from lang_construction.buildPhases import build_phase_sequence_rle
from lang_construction.buildLang import build_lang_sequence_rle
from lang_analysis.computeLCP import PatternMiner


# ----------------------- Configuration -----------------------

SequenceType = Literal["phases", "lang"]

AGENTS = ["SWE-agent", "OpenHands"]
DISPLAY_MODELS = [
    "deepseek-v3",
    "deepseek-r1-0528",
    "devstral-small",
    "claude-sonnet-4",
]

AGENT_ABBR = {"SWE-agent": "SA", "OpenHands": "OH"}
MODEL_ABBR = {
    "deepseek-v3": "DSK-V3",
    "deepseek-r1-0528": "DSK-R1",
    "devstral-small": "Dev",
    "claude-sonnet-4": "CLD-4",
}

# Difficulty mapping
DIFF_KEYS = ["under15min", "under1h", "under4h", "over4h"]
DIFF_LABELS_LOWER = ["easy", "medium", "hard", "very hard"]

DIFFICULTY_RENAME = {
    "<15 min fix": "under15min",
    "15 min - 1 hour": "under1h",
    "1-4 hours": "under4h",
    ">4 hours": "over4h",
}

# PatternMiner settings
MIN_SUPPORT = 0.30
MAX_PERIOD_LEN = 12


# ----------------------- Metadata Extraction -----------------------

def get_metadata(graph_json: dict, key: str, default=None):
    """Extract metadata from graph JSON (supports both flat and nested structures)."""
    if not isinstance(graph_json, dict):
        return default

    if key in graph_json:
        return graph_json.get(key, default)

    graph_obj = graph_json.get("graph", {})
    if isinstance(graph_obj, dict):
        return graph_obj.get(key, default)

    return default


def normalize_metadata(graph_json: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract and normalize resolution_status and debug_difficulty.

    Returns:
        (status, difficulty) where both are normalized or None
    """
    status = get_metadata(graph_json, "resolution_status", "unknown")
    status = str(status).strip().lower()
    if status not in {"resolved", "unresolved"}:
        status = None

    raw_diff = get_metadata(graph_json, "debug_difficulty", "unknown")
    diff_str = str(raw_diff).strip()
    diff_norm = DIFFICULTY_RENAME.get(diff_str, diff_str).lower()

    if diff_norm not in DIFF_KEYS:
        diff_norm = None

    return status, diff_norm


# ----------------------- Sequence Collection -----------------------

def collect_sequences(
    graph_dir: Path,
    sequence_type: SequenceType = "phases"
) -> Dict[Tuple[str, str], List[dict]]:
    """
    Collect RLE sequences from all graphs in a directory, grouped by (status, difficulty).

    Args:
        graph_dir: Directory containing graph JSON files
        sequence_type: Type of sequence to extract ('phases' or 'lang')

    Returns:
        Dictionary mapping (status, difficulty) -> list of RLE sequences
        Each RLE sequence: {'seq': [...], 'lens': [...]}
    """
    grouped: Dict[Tuple[str, str], List[dict]] = defaultdict(list)

    build_fn = build_phase_sequence_rle if sequence_type == "phases" else build_lang_sequence_rle

    for json_file in graph_dir.rglob("*.json"):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                graph_data = json.load(f)
        except (json.JSONDecodeError, IOError):
            continue

        status, difficulty = normalize_metadata(graph_data)
        if not status or not difficulty:
            continue

        step_nodes = extract_node_sequence(graph_data)
        if not step_nodes:
            continue

        seq, lens = build_fn(step_nodes)
        if seq:
            grouped[(status, difficulty)].append({"seq": seq, "lens": lens})

    return grouped


# ----------------------- Matrix Rendering -----------------------

def center_text(text: str, width: int) -> str:
    """Center text within a given width."""
    if len(text) >= width:
        return text[:width]
    pad = width - len(text)
    left = pad // 2
    right = pad - left
    return " " * left + text + " " * right


class MultiMatrixRenderer:
    """Renders multi-agent/model LCP matrix with dynamic layout."""

    COL_STATUS = 10
    COL_DIFF = 12
    COL_CELL = 24
    SEP_BETWEEN_GROUPS = "  "

    def __init__(self, miner: PatternMiner, agents: List[str], models: List[str]):
        self.miner = miner
        self.agents = agents
        self.models = models

    def format_cell(self, sequences_rle: List[dict]) -> str:
        """Format a single cell with the top-1 longest pattern."""
        result = self.miner.longest_ranked_top1(sequences_rle)
        if not result:
            return "—"
        pattern, percentage, lower_bounds = result
        formatted = self.miner.format_pattern_with_lbs(pattern, lower_bounds)
        return f"{formatted} ({percentage}%)"

    def get_cell_for_group(
        self,
        grouped: Dict[Tuple[str, str], List[dict]],
        status: str,
        difficulty: str
    ) -> str:
        """Get formatted cell for a given status and difficulty."""
        sequences = grouped.get((status, difficulty), [])
        return self.format_cell(sequences)

    def render_matrix(
        self,
        per_unit_data: Dict[Tuple[str, str], Dict[Tuple[str, str], List[dict]]]
    ) -> str:
        """
        Render the complete multi-agent/model matrix.

        Args:
            per_unit_data: Maps (agent, model) -> grouped sequences dict

        Returns:
            Formatted matrix string
        """
        lines = []
        left_pad = " " * (self.COL_STATUS + 2 + self.COL_DIFF)

        # Single agent mode: simpler header
        if len(self.agents) == 1:
            agent = self.agents[0]
            agent_abbr = AGENT_ABBR.get(agent, agent)

            # Header with agent name
            lines.append(f"{left_pad}Agent: {agent_abbr}")

            # Model headers
            model_headers = [f"{MODEL_ABBR.get(m, m):<{self.COL_CELL}}" for m in self.models]
            header_line = left_pad + "  ".join(model_headers)
            lines.append(header_line)
            lines.append("-" * len(header_line))

        else:
            # Multi-agent mode: agent groups with model columns
            group_width = len(self.models) * self.COL_CELL + (len(self.models) - 1) * 2

            # Header row 1: Agent abbreviations
            agent_headers = []
            for agent in self.agents:
                agent_abbr = AGENT_ABBR.get(agent, agent)
                agent_headers.append(center_text(agent_abbr, group_width))
            header1 = left_pad + self.SEP_BETWEEN_GROUPS.join(agent_headers)
            lines.append(header1)

            # Header row 2: Model abbreviations
            model_headers = []
            for _ in self.agents:
                for model in self.models:
                    model_headers.append(f"{MODEL_ABBR.get(model, model):<{self.COL_CELL}}")
            header2 = left_pad + "  ".join(model_headers)
            lines.append(header2)
            lines.append("-" * len(header2))

        # Body: Iterate through statuses and difficulties
        for status in ["resolved", "unresolved"]:
            lines.append(f"{status.capitalize():<{self.COL_STATUS}}")

            for diff_key, diff_label in zip(DIFF_KEYS, DIFF_LABELS_LOWER):
                left = f"{'':>{self.COL_STATUS}}  {diff_label:<{self.COL_DIFF}}"

                # Collect cells for all agent/model combinations
                cells = []
                for agent in self.agents:
                    for model in self.models:
                        grouped = per_unit_data.get((agent, model), {})
                        cell = self.get_cell_for_group(grouped, status, diff_key)
                        cells.append(f"{cell:<{self.COL_CELL}}")

                lines.append(left + "  " + "  ".join(cells))

            lines.append("")

        return "\n".join(lines)


class SingleMatrixRenderer:
    """Renders single agent/model LCP matrix."""

    COL_DIFF = 15
    COL_CELL = 50

    def __init__(self, miner: PatternMiner, agent: str, model: str):
        self.miner = miner
        self.agent = agent
        self.model = model

    def format_cell(self, sequences_rle: List[dict]) -> str:
        """Format a single cell with the top-1 longest pattern."""
        result = self.miner.longest_ranked_top1(sequences_rle)
        if not result:
            return "—"
        pattern, percentage, lower_bounds = result
        formatted = self.miner.format_pattern_with_lbs(pattern, lower_bounds)
        return f"{formatted} ({percentage}%)"

    def render_matrix(
        self,
        grouped: Dict[Tuple[str, str], List[dict]]
    ) -> str:
        """
        Render single agent/model matrix.

        Args:
            grouped: Dictionary mapping (status, difficulty) -> sequences

        Returns:
            Formatted matrix string
        """
        lines = []

        # Header
        agent_abbr = AGENT_ABBR.get(self.agent, self.agent)
        model_abbr = MODEL_ABBR.get(self.model, self.model)
        lines.append(f"Agent: {agent_abbr} | Model: {model_abbr}")
        lines.append("=" * (self.COL_DIFF + 2 + self.COL_CELL))

        # Body
        for status in ["resolved", "unresolved"]:
            lines.append(f"\n{status.capitalize()}")

            for diff_key, diff_label in zip(DIFF_KEYS, DIFF_LABELS_LOWER):
                sequences = grouped.get((status, diff_key), [])
                cell = self.format_cell(sequences)
                lines.append(f"  {diff_label:<{self.COL_DIFF}}: {cell}")

        return "\n".join(lines)


# ----------------------- Main Processing -----------------------

def generate_lcp_multi(
    base_data_dir: Path,
    output_dir: Path,
    sequence_type: SequenceType,
    min_support: float,
    max_period_len: int,
    target_agent: Optional[str] = None
) -> None:
    """
    Generate unified LCP matrix for agents and models.

    Args:
        base_data_dir: Base data directory
        output_dir: Output directory
        sequence_type: Type of sequence ('phases' or 'lang')
        min_support: Minimum support threshold
        max_period_len: Maximum pattern length
        target_agent: If specified, only process this agent

    Output:
        - All agents: {output_dir}/LCP/{sequence_type}/all_lcp_matrix.txt
        - Single agent: {output_dir}/LCP/{sequence_type}/{agent}_lcp_matrix.txt
    """
    # Determine which agents to process
    agents_to_process = [target_agent] if target_agent else AGENTS

    per_unit_data: Dict[Tuple[str, str], Dict[Tuple[str, str], List[dict]]] = {}

    for agent in agents_to_process:
        for model in DISPLAY_MODELS:
            graph_dir = base_data_dir / agent / "graphs" / model

            if not graph_dir.exists():
                per_unit_data[(agent, model)] = defaultdict(list)
                print(f"[WARN] Graph directory not found: {graph_dir}")
                continue

            print(f"[INFO] Processing {agent}/{model}...")
            grouped = collect_sequences(graph_dir, sequence_type)
            per_unit_data[(agent, model)] = grouped

            total_seqs = sum(len(v) for v in grouped.values())
            print(f"       Collected {total_seqs} sequences from {len(grouped)} groups")

    # Render and write
    miner = PatternMiner(min_support=min_support, max_period_len=max_period_len)
    renderer = MultiMatrixRenderer(miner, agents=agents_to_process, models=DISPLAY_MODELS)

    print("\n[INFO] Rendering matrix...")
    matrix_text = renderer.render_matrix(per_unit_data)

    # Determine output filename
    if target_agent:
        out_filename = f"{target_agent}_lcp_matrix.txt"
    else:
        out_filename = "all_lcp_matrix.txt"

    output_path = output_dir / "LCP" / sequence_type / out_filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(matrix_text + "\n")

    print(f"[OK] Wrote {output_path}")


def generate_lcp_single(
    base_data_dir: Path,
    output_dir: Path,
    agent: str,
    model: str,
    sequence_type: SequenceType,
    min_support: float,
    max_period_len: int
) -> None:
    """
    Generate LCP matrix for a single agent/model.

    Output: {output_dir}/LCP/{sequence_type}/{agent}_{model}_lcp_matrix.txt
    """
    graph_dir = base_data_dir / agent / "graphs" / model

    if not graph_dir.exists():
        print(f"[ERROR] Graph directory not found: {graph_dir}")
        return

    print(f"[INFO] Processing {agent}/{model}...")
    grouped = collect_sequences(graph_dir, sequence_type)

    total_seqs = sum(len(v) for v in grouped.values())
    print(f"       Collected {total_seqs} sequences from {len(grouped)} groups")

    # Render and write
    miner = PatternMiner(min_support=min_support, max_period_len=max_period_len)
    renderer = SingleMatrixRenderer(miner, agent, model)

    print("\n[INFO] Rendering matrix...")
    matrix_text = renderer.render_matrix(grouped)

    output_path = output_dir / "LCP" / sequence_type / f"{agent}_{model}_lcp_matrix.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(matrix_text + "\n")

    print(f"[OK] Wrote {output_path}")


def main(
    agent: Optional[str] = None,
    model: Optional[str] = None,
    output_dir: Optional[str] = None,
    sequence_type: SequenceType = "phases"
) -> None:
    """
    Main entry point.

    Args:
        agent: Agent name (if model given: single mode; else: agent-specific mode)
        model: Model name (used with agent for single mode)
        output_dir: Output directory (defaults to stats/)
        sequence_type: Type of sequence to use ('phases' or 'lang')
    """
    # Resolve paths
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    base_data_dir = project_root / "data"

    output_path = Path(output_dir) if output_dir else project_root / "stats"

    if not base_data_dir.exists():
        print(f"[ERROR] Data directory not found: {base_data_dir}")
        return

    # Execute based on mode
    print(f"[INFO] Data directory: {base_data_dir}")
    print(f"[INFO] Output directory: {output_path}")
    print(f"[INFO] Sequence type: {sequence_type}")

    if agent and model:
        # Single agent/model mode
        print(f"[INFO] Mode: Single agent/model ({agent}/{model})")
        print()
        generate_lcp_single(
            base_data_dir=base_data_dir,
            output_dir=output_path,
            agent=agent,
            model=model,
            sequence_type=sequence_type,
            min_support=MIN_SUPPORT,
            max_period_len=MAX_PERIOD_LEN
        )
    elif agent:
        # Agent-specific multi-model mode
        if agent not in AGENTS:
            print(f"[WARN] Agent '{agent}' not in default AGENTS list")

        print(f"[INFO] Mode: Agent-specific multi-model ({agent})")
        print()
        generate_lcp_multi(
            base_data_dir=base_data_dir,
            output_dir=output_path,
            sequence_type=sequence_type,
            min_support=MIN_SUPPORT,
            max_period_len=MAX_PERIOD_LEN,
            target_agent=agent
        )
    else:
        # Full multi-agent/model mode
        print(f"[INFO] Mode: Full multi-agent/model")
        print()
        generate_lcp_multi(
            base_data_dir=base_data_dir,
            output_dir=output_path,
            sequence_type=sequence_type,
            min_support=MIN_SUPPORT,
            max_period_len=MAX_PERIOD_LEN
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate LCP matrices from trajectory graphs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 1. Full multi-mode: All agents/models (default)
  #    Output: stats/LCP/phases/all_lcp_matrix.txt
  python generate_LCPs.py

  # 2. Agent-specific mode: All models for one agent
  #    Output: stats/LCP/phases/OpenHands_lcp_matrix.txt
  python generate_LCPs.py --agent OpenHands

  # 3. Single agent/model mode
  #    Output: stats/LCP/phases/OpenHands_deepseek-v3_lcp_matrix.txt
  python generate_LCPs.py --agent OpenHands --model deepseek-v3

  # 4. Generate language sequences with custom output directory
  #    Output: results/LCP/lang/all_lcp_matrix.txt
  python generate_LCPs.py --sequence-type lang --output-dir ./results
        """
    )
    parser.add_argument(
        "--agent",
        type=str,
        help="Agent name (alone: all models for agent; with --model: single mode)"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model name (requires --agent for single mode)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (default: stats/)"
    )
    parser.add_argument(
        "--sequence-type",
        type=str,
        choices=["phases", "lang"],
        default="phases",
        help="Type of sequence to use (default: phases)"
    )

    args = parser.parse_args()

    main(
        agent=args.agent,
        model=args.model,
        output_dir=args.output_dir,
        sequence_type=args.sequence_type
    )
