#!/usr/bin/env python3
"""
Languatory/Phase Extraction Script for Graphectories

Extracts languatory (run-length encoded role sequences) or phases from graphectory JSON files.
Supports both SWE-agent and OpenHands graphectories across multiple models.

Usage:
    # Extract languatory (default)
    python lang_construction/get_lang.py

    # Extract phases
    python lang_construction/get_lang.py --mode phase

    # Process specific agent/model
    python lang_construction/get_lang.py --agent oh --model cld-4

    # Process from custom path
    python lang_construction/get_lang.py --graphs_path data/OpenHands/graphs/claude-sonnet-4

    # Specific instance only
    python lang_construction/get_lang.py --instance_id django__django-10914

Output Structure:
    Languatory mode: {output_dir}/{agent}/langs/{model}/languatory.json
    Phase mode: {output_dir}/{agent}/langs/{model}/phases.json

    Format:
    [
        {
            "instance_id": "django__django-10914",
            "resolution_status": "resolved",
            "debug_difficulty": "<15 min fix",
            "languatory": ["L_navigate_5", "L_reproduce_3", "P_2", "V_regression_test_4"]
            // OR "phases": ["L_5", "L_3", "P_2", "V_4"]
        },
        ...
    ]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Literal
from dataclasses import dataclass, asdict

from lang_construction.extractSeq import extract_node_sequence
from lang_construction.buildPhases import build_phase_sequence_rle


# ==================== Configuration ====================
SUPPORTED_AGENTS = {"sa", "oh"}
SUPPORTED_MODELS = {"dsk-v3", "dsk-r1", "dev", "cld-4"}

MODEL_NAMES = {
    "dsk-v3": "deepseek-v3",
    "dsk-r1": "deepseek-r1-0528",
    "dev": "devstral-small",
    "cld-4": "claude-sonnet-4"
}

AGENT_NAMES = {
    "sa": "SWE-agent",
    "oh": "OpenHands"
}

# Phase abbreviations for phase mode
PHASE_ABBR = {
    'localization': 'L',
    'patch': 'P',
    'validation': 'V',
}

# Reverse mappings for auto-detection
MODEL_NAMES_REV = {v: k for k, v in MODEL_NAMES.items()}
AGENT_NAMES_REV = {v: k for k, v in AGENT_NAMES.items()}


# ==================== Data Classes ====================
@dataclass
class Languatory:
    """Languatory data for a single instance."""
    instance_id: str
    resolution_status: str
    debug_difficulty: str
    languatory: List[str]  # Each element is "Role_runlength", e.g., "P_2", "L_navigate_3"


@dataclass
class Phases:
    """Phase sequence data for a single instance."""
    instance_id: str
    resolution_status: str
    debug_difficulty: str
    phases: List[str]  # Each element is "PhaseAbbr_runlength", e.g., "P_2", "L_3", "V_4"


# ==================== Path Management ====================
def get_lang_output_path(base_output_dir: str, agent: str, model: str, mode: Literal["lang", "phase"] = "lang") -> Path:
    """Construct the output file path based on mode."""
    agent_name = AGENT_NAMES[agent]
    model_name = MODEL_NAMES[model]
    filename = "languatory.json" if mode == "lang" else "phases.json"
    return Path(base_output_dir) / agent_name / "langs" / model_name / filename


def discover_agent_model_paths(data_dir: Path) -> List[Tuple[str, str, Path]]:
    """Discover all agent/model combinations in data directory.

    Returns:
        List of (agent_abbr, model_abbr, graphs_path) tuples
    """
    paths = []

    for agent_abbr, agent_name in AGENT_NAMES.items():
        agent_path = data_dir / agent_name / "graphs"
        if not agent_path.exists():
            continue

        for model_abbr, model_name in MODEL_NAMES.items():
            model_path = agent_path / model_name
            if model_path.exists() and model_path.is_dir():
                paths.append((agent_abbr, model_abbr, model_path))

    return paths


# ==================== Graphectory Loading ====================
def load_graphectory(json_path: Path) -> Optional[Dict[str, Any]]:
    """Load a single graphectory JSON file."""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error loading {json_path.name}: {e}", file=sys.stderr)
        return None


def find_graphectories(path: Path, instance_id: Optional[str] = None) -> List[Path]:
    """Find graphectory JSON files in a path.

    Expected structure: {model_path}/{instance_id}/{instance_id}.json
    """
    if not path.exists():
        return []

    if path.is_file():
        return [path] if path.suffix == '.json' else []

    # Find all JSON files in subdirectories
    json_files = []
    for json_path in path.glob("*/*.json"):
        # Validate structure: parent dir name should match filename stem
        if json_path.parent.name == json_path.stem:
            if instance_id is None or json_path.stem == instance_id:
                json_files.append(json_path)

    return sorted(json_files)


# ==================== Languatory Extraction ====================
def extract_languatory(graph_json: Dict[str, Any]) -> Optional[Languatory]:
    """Extract languatory from a graphectory JSON."""
    try:
        # Extract from graph metadata
        instance_id = graph_json.get("graph", {}).get("instance_name")
        if not instance_id:
            return None
        
        resolution_status = graph_json.get("graph", {}).get("resolution_status")
        if not resolution_status:
            return None

        debug_difficulty = graph_json.get("graph", {}).get("debug_difficulty")
        if not debug_difficulty:
            return None

        # Extract node sequence
        step_nodes = extract_node_sequence(graph_json)
        if not step_nodes:
            return None

        # Build RLE languatory
        roles, run_lengths = build_lang_sequence_rle(step_nodes)
        if not roles:
            return None

        # Format as "Role_runlength" strings
        languatory = [f"{role}_{length}" for role, length in zip(roles, run_lengths)]

        return Languatory(
            instance_id=instance_id,
            resolution_status=resolution_status,
            debug_difficulty=debug_difficulty,
            languatory=languatory
        )

    except Exception as e:
        print(f"  Extraction error: {e}", file=sys.stderr)
        return None


def extract_phases(graph_json: Dict[str, Any]) -> Optional[Phases]:
    """Extract phase sequence from a graphectory JSON."""
    try:
        # Extract from graph metadata
        instance_id = graph_json.get("graph", {}).get("instance_name")
        if not instance_id:
            return None

        resolution_status = graph_json.get("graph", {}).get("resolution_status")
        if not resolution_status:
            return None

        debug_difficulty = graph_json.get("graph", {}).get("debug_difficulty")
        if not debug_difficulty:
            return None

        # Extract node sequence
        step_nodes = extract_node_sequence(graph_json)
        if not step_nodes:
            return None

        # Build RLE phase sequence
        phases_full, run_lengths = build_phase_sequence_rle(step_nodes)
        if not phases_full:
            return None

        # Convert to abbreviations
        phase_abbrs = [PHASE_ABBR.get(p.lower(), p) for p in phases_full]

        # Format as "PhaseAbbr_runlength" strings
        phases = [f"{phase}_{length}" for phase, length in zip(phase_abbrs, run_lengths)]

        return Phases(
            instance_id=instance_id,
            resolution_status=resolution_status,
            debug_difficulty=debug_difficulty,
            phases=phases
        )

    except Exception as e:
        print(f"  Extraction error: {e}", file=sys.stderr)
        return None


# ==================== Output Management ====================
def load_existing_data(output_path: Path, mode: Literal["lang", "phase"]) -> Dict[str, Any]:
    """Load existing data from output file."""
    if not output_path.exists():
        return {}

    try:
        with open(output_path, 'r') as f:
            data = json.load(f)

        result = {}
        for item in data:
            instance_id = item["instance_id"]

            if mode == "lang":
                # Handle both old format (roles + run_lengths) and new format (languatory)
                if "languatory" in item:
                    result[instance_id] = Languatory(**item)
                elif "roles" in item and "run_lengths" in item:
                    # Convert old format to new format
                    languatory = [f"{role}_{length}" for role, length in zip(item["roles"], item["run_lengths"])]
                    result[instance_id] = Languatory(
                        instance_id=instance_id,
                        resolution_status=item.get("resolution_status", ""),
                        debug_difficulty=item.get("debug_difficulty", ""),
                        languatory=languatory
                    )
                else:
                    print(f"  Warning: Skipping malformed entry for {instance_id}", file=sys.stderr)
            else:  # phase mode
                if "phases" in item:
                    result[instance_id] = Phases(**item)
                elif "roles" in item and "run_lengths" in item:
                    # Convert old format
                    phases = [f"{role}_{length}" for role, length in zip(item["roles"], item["run_lengths"])]
                    result[instance_id] = Phases(
                        instance_id=instance_id,
                        resolution_status=item.get("resolution_status", ""),
                        debug_difficulty=item.get("debug_difficulty", ""),
                        phases=phases
                    )
                else:
                    print(f"  Warning: Skipping malformed entry for {instance_id}", file=sys.stderr)

        return result
    except Exception as e:
        print(f"  Warning: Could not load {output_path}: {e}", file=sys.stderr)
        return {}


def save_data(data_list: List[Any], output_path: Path, merge: bool = True, mode: Literal["lang", "phase"] = "lang"):
    """Save data to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Merge with existing data
    existing = load_existing_data(output_path, mode) if merge else {}
    for item in data_list:
        existing[item.instance_id] = item

    # Convert to sorted list
    output_data = [
        asdict(item) for item in
        sorted(existing.values(), key=lambda x: x.instance_id)
    ]

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    return len(output_data)


# ==================== Main Processing ====================
def process_single_config(
    agent: str,
    model: str,
    graphs_path: Path,
    instance_id: Optional[str],
    output_dir: str,
    mode: Literal["lang", "phase"] = "lang"
) -> Tuple[int, int]:
    """Process graphectories for a single agent/model configuration.

    Returns:
        (num_processed, num_total) tuple
    """
    # Find graphectory files
    json_paths = find_graphectories(graphs_path, instance_id)
    if not json_paths:
        return 0, 0

    # Process each graphectory
    results = []
    for json_path in json_paths:
        graph_json = load_graphectory(json_path)
        if graph_json is None:
            continue

        if mode == "lang":
            result = extract_languatory(graph_json)
        else:  # phase
            result = extract_phases(graph_json)

        if result is not None:
            results.append(result)

    # Save results
    if results:
        output_path = get_lang_output_path(output_dir, agent, model, mode)
        total_saved = save_data(results, output_path, merge=True, mode=mode)
        return len(results), len(json_paths)

    return 0, len(json_paths)


def process_all(
    data_dir: str = "data",
    agent: Optional[str] = None,
    model: Optional[str] = None,
    graphs_path: Optional[Path] = None,
    instance_id: Optional[str] = None,
    output_dir: Optional[str] = None,
    mode: Literal["lang", "phase"] = "lang"
) -> int:
    """Process graphectories based on provided arguments.

    Returns:
        Total number of successfully processed instances
    """
    output_dir = output_dir or data_dir
    total_processed = 0

    # Case 1: Specific graphs_path provided
    if graphs_path is not None:
        # Auto-detect agent/model from path structure
        detected_agent, detected_model = None, None
        parts = graphs_path.parts

        for i, part in enumerate(parts):
            if part in AGENT_NAMES_REV:
                detected_agent = AGENT_NAMES_REV[part]
                if i + 2 < len(parts) and parts[i + 1] == "graphs":
                    potential_model = parts[i + 2]
                    if potential_model in MODEL_NAMES_REV:
                        detected_model = MODEL_NAMES_REV[potential_model]
                break

        agent = agent or detected_agent
        model = model or detected_model

        if agent is None or model is None:
            print(f"Error: Cannot auto-detect agent/model from {graphs_path}", file=sys.stderr)
            print(f"  Please specify --agent and --model explicitly", file=sys.stderr)
            return 0

        print(f"\n{'='*60}")
        print(f"Processing: {AGENT_NAMES[agent]} / {MODEL_NAMES[model]}")
        print(f"{'='*60}")

        processed, total = process_single_config(agent, model, graphs_path, instance_id, output_dir, mode)
        print(f"  Processed: {processed}/{total} instances")
        total_processed += processed

    # Case 2: Specific agent/model or process all
    else:
        data_path = Path(data_dir)

        # Determine which configurations to process
        if agent is not None and model is not None:
            # Single specific configuration
            configs = [(agent, model, data_path / AGENT_NAMES[agent] / "graphs" / MODEL_NAMES[model])]
        elif agent is not None:
            # All models for specific agent
            configs = [
                (agent, m, data_path / AGENT_NAMES[agent] / "graphs" / MODEL_NAMES[m])
                for m in SUPPORTED_MODELS
            ]
        elif model is not None:
            # All agents for specific model
            configs = [
                (a, model, data_path / AGENT_NAMES[a] / "graphs" / MODEL_NAMES[model])
                for a in SUPPORTED_AGENTS
            ]
        else:
            # Discover all available configurations
            configs = discover_agent_model_paths(data_path)

        # Process each configuration
        for agent, model, graphs_path in configs:
            if not graphs_path.exists():
                continue

            print(f"\n{'='*60}")
            print(f"Processing: {AGENT_NAMES[agent]} / {MODEL_NAMES[model]}")
            print(f"{'='*60}")

            processed, total = process_single_config(agent, model, graphs_path, instance_id, output_dir, mode)
            if total > 0:
                print(f"  Processed: {processed}/{total} instances")
                total_processed += processed

    return total_processed


# ==================== CLI ====================
def main():
    parser = argparse.ArgumentParser(
        description="Extract languatories or phases from graphectory JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract languatory (default)
  python lang_construction/get_lang.py

  # Extract phases
  python lang_construction/get_lang.py --mode phase

  # Process specific agent
  python lang_construction/get_lang.py --agent oh --mode phase

  # Process specific model across all agents
  python lang_construction/get_lang.py --model cld-4

  # Process specific agent and model
  python lang_construction/get_lang.py --agent sa --model dsk-v3 --mode lang

  # Process from custom path
  python lang_construction/get_lang.py --graphs_path data/samples/OpenHands/graphs/deepseek-v3

  # Process specific instance only
  python lang_construction/get_lang.py --instance_id django__django-10914 --mode phase
        """
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["lang", "phase"],
        default="lang",
        help="Extraction mode: 'lang' for languatory (detailed roles), 'phase' for phases (abbreviated). Default: lang"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Base data directory (default: data/)"
    )

    parser.add_argument(
        "--graphs_path",
        type=str,
        help="Path to specific graphectory file or directory (overrides data_dir)"
    )

    parser.add_argument(
        "--agent",
        type=str,
        choices=SUPPORTED_AGENTS,
        help="Agent type (sa=SWE-agent, oh=OpenHands). Process all if not specified."
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=SUPPORTED_MODELS,
        help="Model type (dsk-v3, dsk-r1, dev, cld-4). Process all if not specified."
    )

    parser.add_argument(
        "--instance_id",
        type=str,
        help="Specific instance to process (optional)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        help="Base output directory (default: same as data_dir)"
    )

    args = parser.parse_args()

    # Convert graphs_path to Path if provided
    graphs_path = Path(args.graphs_path) if args.graphs_path else None

    # Process graphectories
    mode_name = "Languatory" if args.mode == "lang" else "Phase Sequence"
    print(f"{mode_name} Extraction")
    print("=" * 60)

    total = process_all(
        data_dir=args.data_dir,
        agent=args.agent,
        model=args.model,
        graphs_path=graphs_path,
        instance_id=args.instance_id,
        output_dir=args.output_dir,
        mode=args.mode
    )

    print(f"\n{'='*60}")
    print(f"Total: {total} instances processed successfully")
    print(f"{'='*60}")

    sys.exit(0 if total > 0 else 1)


if __name__ == "__main__":
    main()