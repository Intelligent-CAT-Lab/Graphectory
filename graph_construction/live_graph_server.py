#!/usr/bin/env python3
"""
live_graph_server.py

Entry point for the trajectory graph browser.  Run:

    python live_graph_server.py --trajs <dir-or-jsonl> --eval_report <file>

Then open http://localhost:8000 in your browser.

--trajs and --eval_report can also be omitted; the data source can be set or
changed at any time from the browser UI without restarting the server.

Graphs are rendered on demand — no HTML files are pre-generated.  Each HTTP
request is handled in its own thread, so navigating quickly between instances
or changing toggle settings never blocks the UI.  The agent type (SWE-agent or
OpenHands) is inferred automatically from the path passed to --trajs.
"""

import argparse
import logging
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

# Allow sibling imports (buildGraph, mapPhase, commandParser, …).
sys.path.insert(0, str(Path(__file__).parent))

from server.handler import GraphHandler

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Live trajectory graph browser (on-demand rendering)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples
--------
  # SWE-agent — pass a directory of .traj files:
  python live_graph_server.py \\
      --trajs path/to/trajectories \\
      --eval_report report.json

  # OpenHands — pass the output.jsonl file directly:
  python live_graph_server.py \\
      --trajs path/to/output.jsonl \\
      --eval_report report.json

  # Start without paths and configure via the browser UI:
  python live_graph_server.py --port 8080

  # Custom assets directory:
  python live_graph_server.py \\
      --trajs trajectories \\
      --eval_report report.json \\
      --assets_dir custom_templates
        """,
    )
    p.add_argument(
        "--trajs", default=None,
        help="Directory of .traj files (SWE-agent) or path to output.jsonl (OpenHands). "
             "Can be omitted and set later from the browser UI.",
    )
    p.add_argument(
        "--eval_report", default=None,
        help="Evaluation report JSON with 'resolved_ids' and 'unresolved_ids' arrays. "
             "Can be omitted and set later from the browser UI.",
    )
    p.add_argument(
        "--assets_dir", default=None,
        help="Directory containing graph_template.html, styles.css, and graph_renderer.js. "
             "Defaults to the directory containing this script.",
    )
    p.add_argument("--port", type=int, default=8000)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Command parser setup
# ---------------------------------------------------------------------------

def _setup_cmd_parser():
    """Load a CommandParser with all available SWE-agent tool configs.

    All present config files are loaded (each defines a distinct tool set),
    so tool parsing covers editor, reviewer, and registry variants in one pass.
    Returns ``None`` if commandParser is not importable.
    """
    try:
        from commandParser import CommandParser
    except ImportError:
        logger.warning("commandParser not found — tool actions will use fallback parsing.")
        return None

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
        logger.info("Loaded tool configs: %s", ", ".join(loaded))
    else:
        logger.debug("No tool config files found; CommandParser using defaults.")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        datefmt="%H:%M:%S",
    )

    assets_dir = Path(args.assets_dir) if args.assets_dir else Path(__file__).parent
    if not assets_dir.exists():
        logger.error("--assets_dir does not exist: %s", assets_dir)
        return 1

    # Validate CLI-supplied paths when provided.
    trajs: Path | None       = None
    eval_report: Path | None = None
    agent_type: str          = "sa"

    if args.trajs:
        trajs = Path(args.trajs)
        if not trajs.exists():
            logger.error("--trajs path does not exist: %s", trajs)
            return 1

        if trajs.is_file() and trajs.suffix == ".jsonl":
            agent_type = "oh"
        elif trajs.is_dir():
            # Peek inside: .traj.json files indicate mini-swe-agent; otherwise SWE-agent
            if any(trajs.rglob("*.traj.json")):
                agent_type = "msa"
            else:
                agent_type = "sa"
        else:
            logger.error(
                "--trajs must be a directory (SWE-agent or mini-swe-agent) "
                "or a .jsonl file (OpenHands): %s", trajs,
            )
            return 1

        if args.eval_report:
            eval_report = Path(args.eval_report)
            if not eval_report.exists():
                logger.error("--eval_report path does not exist: %s", eval_report)
                return 1

    # Inject configuration into the handler class before the server starts.
    GraphHandler.graphs_dir       = trajs
    GraphHandler.agent_type       = agent_type
    GraphHandler.eval_report_path = str(eval_report) if eval_report else None
    GraphHandler.cmd_parser       = _setup_cmd_parser()
    GraphHandler.assets_dir       = assets_dir

    # ThreadingHTTPServer handles each request in its own thread, so slow
    # graph builds never block the browser UI or subsequent requests.
    httpd = ThreadingHTTPServer(("", args.port), GraphHandler)

    agent_label = (
        "OpenHands (.jsonl)"      if agent_type == "oh"  else
        "mini-swe-agent (directory)" if agent_type == "msa" else
        "SWE-agent (directory)"
    )
    print(f"\n{'─' * 60}")
    print( "  Trajectory Graph Server")
    print(f"{'─' * 60}")
    if trajs:
        print(f"  Agent      : {agent_type.upper()}  ({agent_label})")
        print(f"  Trajs      : {trajs.absolute()}")
        if eval_report:
            print(f"  Report     : {eval_report.absolute()}")
        else:
            print( "  Report     : (none — status badges will not be shown)")
    else:
        print( "  Data source: not set — configure via the browser UI")
    print(f"  Assets     : {assets_dir.absolute()}")
    print(f"  URL        : http://localhost:{args.port}")
    print(f"{'─' * 60}\n")
    print("  Press Ctrl+C to stop.\n")

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")

    return 0


if __name__ == "__main__":
    sys.exit(main())