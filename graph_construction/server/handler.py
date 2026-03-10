"""
server/handler.py

HTTP request handler.  Routing only — all business logic lives in
graph_builder.py and graph_renderer.py.

Routes
------
GET /                           → browser UI  (index.html)
GET /static/<file>              → static assets  (browser.css, browser.js, dagre.min.js, …)
GET /api/graphs                 → JSON list of available trajectories
GET /api/graph?id=X[&…]         → on-demand graph HTML for instance X
"""

import json
import logging
import traceback
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from server.graph_builder  import scan_trajectories, load_trajectory, build_graph
from server.graph_renderer import render_graph_html

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"

_MIME: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
}


class GraphHandler(BaseHTTPRequestHandler):
    """Thin HTTP handler — delegates all logic to the server modules."""

    # Injected by live_graph_server.py before the server starts.
    graphs_dir:       Path = None   # directory (SWE-agent) or .jsonl file (OpenHands)
    agent_type:       str  = "sa"   # "sa" | "oh"
    eval_report_path: str  = None
    cmd_parser             = None
    assets_dir:       Path = None   # directory containing graph_template.html etc.

    # ── Logging ──────────────────────────────────────────────────────────────

    def log_message(self, fmt, *args):
        status = args[1] if len(args) > 1 else "?"
        logger.info("%s %s  →  %s", self.command, self.path, status)

    # ── Dispatch ─────────────────────────────────────────────────────────────

    def do_GET(self):
        parsed = urlparse(self.path)
        path   = parsed.path
        params = parse_qs(parsed.query)

        try:
            if path in ("/", "/index.html"):
                self._send_file(STATIC_DIR / "index.html")

            elif path.startswith("/static/"):
                self._send_file(STATIC_DIR / path[len("/static/"):])

            elif path == "/api/graphs":
                self._api_graphs()

            elif path == "/api/graph":
                instance_id = params.get("id", [""])[0]
                if not instance_id:
                    self._error(400, "Missing required query parameter: id")
                    return
                self._api_graph(
                    instance_id      = instance_id,
                    filter_cd        = _bool_param(params, "filter_cd",        default=False),
                    thought_quotes   = _bool_param(params, "thought_quotes",   default=True),
                    node_verbosity   = _bool_param(params, "node_verbosity",   default=True),
                    show_observation = _bool_param(params, "show_observation", default=False),
                    unique_think     = _bool_param(params, "unique_think",     default=True),
                )

            else:
                self._error(404, "Not found")

        except Exception as exc:
            logger.exception("[handler] Unhandled error for %s %s", self.command, self.path)
            self._error(500, str(exc))

    # ── Route handlers ────────────────────────────────────────────────────────

    def _api_graphs(self):
        graphs = scan_trajectories(
            self.graphs_dir, self.eval_report_path, agent_type=self.agent_type,
        )
        self._respond_json(graphs)

    def _api_graph(
        self,
        instance_id:      str,
        filter_cd:        bool,
        thought_quotes:   bool,
        node_verbosity:   bool,
        show_observation: bool,
        unique_think:     bool,
    ):
        logger.info("[handler] Building graph for '%s'", instance_id)

        traj_data = load_trajectory(
            self.graphs_dir, instance_id, agent_type=self.agent_type,
        )
        G = build_graph(
            traj_data        = traj_data,
            instance_id      = instance_id,
            eval_report_path = self.eval_report_path,
            cmd_parser       = self.cmd_parser,
            filter_cd        = filter_cd,
            agent_type       = self.agent_type,
            unique_think     = unique_think,
        )
        html = render_graph_html(
            G, filter_cd, thought_quotes, node_verbosity, show_observation, self.assets_dir,
        )
        self._respond(200, "text/html; charset=utf-8", html.encode())

    # ── Low-level helpers ─────────────────────────────────────────────────────

    def _send_file(self, path: Path):
        if not path.exists():
            self._error(404, f"File not found: {path.name}")
            return
        content_type = _MIME.get(path.suffix.lower(), "application/octet-stream")
        self._respond(200, content_type, path.read_bytes())

    def _respond_json(self, data):
        self._respond(200, "application/json; charset=utf-8", json.dumps(data).encode())

    def _respond(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type",   content_type)
        self.send_header("Content-Length", len(body))
        self.send_header("Cache-Control",  "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str):
        self._respond(status, "application/json; charset=utf-8",
                      json.dumps({"error": message}).encode())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bool_param(params: dict, key: str, *, default: bool) -> bool:
    """Parse a boolean query-string parameter, returning *default* if absent."""
    raw = params.get(key, [None])[0]
    if raw is None:
        return default
    return raw.lower() == "true"