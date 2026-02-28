#!/usr/bin/env python3
"""
live_graph_server.py

Single entry point.  Run:

    python live_graph_server.py --trajs <dir> --eval_report <file>

Then open http://localhost:8000 in your browser.

All graph data is rendered on the fly; no HTML files are pre-generated.
Use the toggle in the sidebar to switch between cd-filtered (▲ hat) and
cd-as-separate-node mode in real time.
"""

import argparse
import sys
from http.server import HTTPServer
from pathlib import Path

# Allow sibling imports (buildGraph, mapPhase, commandParser…)
sys.path.insert(0, str(Path(__file__).parent))

from server.handler import GraphHandler


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live trajectory graph browser (on-demand rendering)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # SWE-agent: pass a directory containing .traj files
  python live_graph_server.py \\
      --trajs output/SWE-agent/graphs/deepseek-v3 \\
      --eval_report report.json

  # OpenHands: pass the output.jsonl file directly
  python live_graph_server.py \\
      --trajs trajectories/OpenHands/output.jsonl \\
      --eval_report report.json

  python live_graph_server.py \\
      --trajs trajectories \\
      --eval_report report.json \\
      --assets_dir custom_templates \\
      --port 8080
        """,
    )
    p.add_argument("--trajs",    required=True,
                   help="Directory that contains .traj files (SWE-agent), "
                        "or path to an output.jsonl file (OpenHands)")
    p.add_argument("--eval_report",   required=True,
                   help="Evaluation report JSON used for resolution status")
    p.add_argument("--assets_dir",    default=None,
                   help="Directory with graph_template.html / styles.css / "
                        "graph_renderer.js  (defaults to same dir as this script)")
    p.add_argument("--port",          type=int, default=8000)
    return p.parse_args()


def setup_cmd_parser():
    """Return a CommandParser loaded with all available SWE-agent tool configs.

    Each config file defines a distinct set of tools (editor, reviewer, registry),
    so all present configs are loaded — not just the first one found.
    Returns None only if commandParser cannot be imported.
    """
    try:
        from commandParser import CommandParser
        parser = CommandParser()

        tool_configs = [
            "data/SWE-agent/tools/edit_anthropic/config.yaml",
            "data/SWE-agent/tools/review_on_submit_m/config.yaml",
            "data/SWE-agent/tools/registry/config.yaml",
        ]
        loaded = []
        for cfg in tool_configs:
            cfg_path = Path(cfg)
            if cfg_path.exists():
                parser.load_tool_yaml_files([str(cfg_path)])
                loaded.append(cfg_path.name)

        if loaded:
            print(f"  [parser] Loaded tool configs: {', '.join(loaded)}")

        return parser

    except ImportError:
        print("[WARN] commandParser not found – install it or add it to the Python path")
        return None


def main() -> int:
    args = parse_args()

    trajs = Path(args.trajs)
    if not trajs.exists():
        print(f"[ERROR] trajs does not exist: {trajs}")
        return 1

    eval_report = Path(args.eval_report)
    if not eval_report.exists():
        print(f"[ERROR] eval_report does not exist: {eval_report}")
        return 1

    assets_dir = Path(args.assets_dir) if args.assets_dir else Path(__file__).parent
    if not assets_dir.exists():
        print(f"[ERROR] assets_dir does not exist: {assets_dir}")
        return 1

    # Detect agent type: a .jsonl file → OpenHands; a directory → SWE-agent
    if trajs.is_file() and trajs.suffix == ".jsonl":
        agent_type = "oh"
    elif trajs.is_dir():
        agent_type = "sa"
    else:
        print(f"[ERROR] --trajs must be a directory (SWE-agent) or a .jsonl file (OpenHands): {trajs}")
        return 1

    # Inject configuration into the handler class
    GraphHandler.graphs_dir       = trajs        # may be a file (OH) or dir (SA)
    GraphHandler.agent_type       = agent_type
    GraphHandler.eval_report_path = str(eval_report)
    GraphHandler.cmd_parser       = setup_cmd_parser()
    GraphHandler.assets_dir       = assets_dir

    httpd = HTTPServer(("", args.port), GraphHandler)

    print(f"\n{'─'*60}")
    print(f"  Trajectory Graph Server")
    print(f"{'─'*60}")
    print(f"  Agent type   : {agent_type.upper()} ({'OpenHands (.jsonl)' if agent_type == 'oh' else 'SWE-agent (directory)'})")
    print(f"  Trajs path   : {trajs.absolute()}")
    print(f"  Eval report  : {eval_report.absolute()}")
    print(f"  Assets dir   : {assets_dir.absolute()}")
    print(f"  URL          : http://localhost:{args.port}")
    print(f"{'─'*60}\n")
    print("  Press Ctrl+C to stop.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")

    return 0


if __name__ == "__main__":
    sys.exit(main())