"""
server/handler.py

HTTP request handler.  Routing only — all business logic lives in
graph_builder.py and graph_renderer.py.

The server runs in threaded mode (ThreadingHTTPServer), so every public method
on this class must be thread-safe.  Two in-memory caches are maintained as
class-level attributes protected by a single RLock:

  _graphs_cache   — the /api/graphs JSON list, rebuilt whenever the data source changes.
  _render_cache   — rendered HTML keyed by (instance_id, settings…), flushed on reconfigure.

Routes
------
GET  /                          → browser UI  (index.html)
GET  /static/<file>             → static assets (browser.css, browser.js, …)
GET  /api/graphs                → JSON list of available trajectories
GET  /api/graph?id=X[&…]        → on-demand graph HTML for instance X
GET  /api/config                → currently active trajs path and eval_report path
POST /api/config                → swap trajs/eval_report live; validates paths and overlap
"""

import json
import logging
import threading
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
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

# Minimum fraction of trajectory instance IDs that must appear in the report
# (or vice-versa) for the pairing to be considered valid.
_OVERLAP_THRESHOLD = 0.05


class GraphHandler(BaseHTTPRequestHandler):
    """Thin, thread-safe HTTP handler — delegates all logic to the server modules."""

    # ── Injected by live_graph_server.py before the server starts ────────────
    graphs_dir:       Path = None   # directory (SWE-agent) or .jsonl file (OpenHands)
    agent_type:       str  = "sa"   # "sa" | "oh"
    eval_report_path: str  = None
    cmd_parser             = None
    assets_dir:       Path = None   # directory containing graph_template.html etc.

    # ── In-memory caches (class-level, shared across all handler instances) ──
    _cache_lock:   threading.RLock  = threading.RLock()
    _graphs_cache: Optional[list]   = None
    _render_cache: dict[tuple, str] = {}

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

            elif path == "/api/config":
                self._api_get_config()

            else:
                self._error(404, "Not found")

        except Exception as exc:
            logger.exception("[handler] Unhandled error for GET %s", self.path)
            self._error(500, str(exc))

    def do_POST(self):
        parsed = urlparse(self.path)
        path   = parsed.path

        try:
            if path == "/api/config":
                length = int(self.headers.get("Content-Length", 0))
                body   = self.rfile.read(length)
                data   = json.loads(body)
                self._api_post_config(data)
            else:
                self._error(404, "Not found")

        except Exception as exc:
            logger.exception("[handler] Unhandled error for POST %s", self.path)
            self._error(500, str(exc))

    # ── Route handlers ────────────────────────────────────────────────────────

    def _api_get_config(self):
        """Return the currently active data-source paths."""
        self._respond_json({
            "trajs":       str(self.graphs_dir)       if self.graphs_dir       else "",
            "eval_report": str(self.eval_report_path) if self.eval_report_path else "",
            "agent_type":  self.agent_type,
        })

    def _api_post_config(self, data: dict):
        """Validate and apply a new trajs/eval_report pair.

        Checks performed:
          1. Both paths exist on disk.
          2. trajs is a directory (.traj files) or a .jsonl file.
          3. eval_report is valid JSON containing recognised ID arrays.
          4. At least _OVERLAP_THRESHOLD of trajectory IDs appear in the report
             (or the report is non-empty and contains at least one matching ID),
             so that an accidental mismatch between datasets is caught early.

        On success the class-level state is updated and both caches are flushed
        so the next /api/graphs and /api/graph requests use the new data source.
        On failure a 400 response is returned with a human-readable error message;
        the existing configuration is left unchanged.
        """
        raw_trajs  = (data.get("trajs")       or "").strip()
        raw_report = (data.get("eval_report") or "").strip()

        if not raw_trajs or not raw_report:
            self._error(400, "Both 'trajs' and 'eval_report' are required.")
            return

        trajs  = Path(raw_trajs)
        report = Path(raw_report)

        # ── Path existence ────────────────────────────────────────────────────
        if not trajs.exists():
            self._error(400, f"Trajectories path not found: {trajs}")
            return
        if not report.exists():
            self._error(400, f"Eval report not found: {report}")
            return

        # ── Agent type inference ──────────────────────────────────────────────
        if trajs.is_file() and trajs.suffix == ".jsonl":
            agent_type = "oh"
        elif trajs.is_dir():
            agent_type = "sa"
        else:
            self._error(
                400,
                f"Trajectories path must be a directory (SWE-agent) "
                f"or an output.jsonl file (OpenHands): {trajs}",
            )
            return

        # ── Report validity ───────────────────────────────────────────────────
        try:
            with open(report) as fh:
                report_data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            self._error(400, f"Could not read eval report as JSON: {exc}")
            return

        report_ids: set[str] = set(
            report_data.get("resolved_ids", []) + report_data.get("unresolved_ids", [])
        )

        # ── Overlap check ─────────────────────────────────────────────────────
        error = _check_overlap(trajs, agent_type, report_ids)
        if error:
            self._error(400, error)
            return

        # ── Apply ─────────────────────────────────────────────────────────────
        logger.info(
            "[handler] Reconfiguring data source: trajs=%s  report=%s", trajs, report,
        )
        with self._cache_lock:
            GraphHandler.graphs_dir       = trajs
            GraphHandler.agent_type       = agent_type
            GraphHandler.eval_report_path = str(report)
            GraphHandler._graphs_cache    = None
            GraphHandler._render_cache    = {}

        self._respond_json({
            "ok":          True,
            "trajs":       str(trajs),
            "eval_report": str(report),
            "agent_type":  agent_type,
        })

    def _api_graphs(self):
        """Return the trajectory list, using the cached copy when available."""
        with self._cache_lock:
            if self._graphs_cache is None:
                logger.info("[handler] Building trajectory list (cache miss).")
                GraphHandler._graphs_cache = scan_trajectories(
                    self.graphs_dir, self.eval_report_path, agent_type=self.agent_type,
                )
            graphs = self._graphs_cache

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
        """Build (or retrieve from cache) and serve the graph HTML for *instance_id*."""
        cache_key = (instance_id, filter_cd, thought_quotes,
                     node_verbosity, show_observation, unique_think)

        with self._cache_lock:
            cached = self._render_cache.get(cache_key)
        if cached is not None:
            logger.info("[handler] Cache hit for '%s'.", instance_id)
            self._respond(200, "text/html; charset=utf-8", cached.encode())
            return

        logger.info("[handler] Building graph for '%s'.", instance_id)
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

        with self._cache_lock:
            self._render_cache[cache_key] = html

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
# Validation helpers
# ---------------------------------------------------------------------------

def _traj_instance_ids(trajs: Path, agent_type: str) -> set[str]:
    """Return the set of instance IDs found in *trajs* without building graphs."""
    ids: set[str] = set()
    if agent_type == "oh":
        try:
            with open(trajs) as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        iid   = entry.get("instance_id")
                        if iid:
                            ids.add(iid)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass
    else:
        for traj_file in trajs.rglob("*.traj"):
            ids.add(traj_file.stem)
    return ids


def _check_overlap(trajs: Path, agent_type: str, report_ids: set[str]) -> Optional[str]:
    """Return an error string if the trajs/report pair looks mismatched, else None.

    The check is intentionally lenient: it only fires when *both* sides are
    non-empty and the overlap is below _OVERLAP_THRESHOLD.  An empty report
    (no resolved/unresolved IDs) is allowed through — it just means everything
    will be marked 'unsubmitted', which is valid during development.
    """
    if not report_ids:
        # Empty report — nothing to compare against; pass through.
        return None

    traj_ids = _traj_instance_ids(trajs, agent_type)
    if not traj_ids:
        return "No trajectory instances found at the given path."

    overlap = traj_ids & report_ids
    if not overlap:
        sample_traj   = sorted(traj_ids)[:3]
        sample_report = sorted(report_ids)[:3]
        return (
            f"The trajectories and eval report appear to be mismatched — "
            f"no instance IDs overlap.\n"
            f"  Trajectory IDs (sample): {sample_traj}\n"
            f"  Report IDs (sample):     {sample_report}"
        )

    ratio = len(overlap) / max(len(traj_ids), len(report_ids))
    if ratio < _OVERLAP_THRESHOLD:
        return (
            f"Very few instance IDs overlap between the trajectories and eval report "
            f"({len(overlap)} of {len(traj_ids)} trajectories matched). "
            f"Check that both paths refer to the same evaluation run."
        )

    return None


# ---------------------------------------------------------------------------
# Query-string helpers
# ---------------------------------------------------------------------------

def _bool_param(params: dict, key: str, *, default: bool) -> bool:
    """Parse a boolean query-string parameter, returning *default* if absent."""
    raw = params.get(key, [None])[0]
    if raw is None:
        return default
    return raw.lower() == "true"