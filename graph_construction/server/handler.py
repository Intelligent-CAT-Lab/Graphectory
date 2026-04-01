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
GET  /api/sankey                → aggregated phase-per-step data for Sankey diagram
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
    _sankey_cache: Optional[dict]   = None   # keyed on graphs_dir + eval_report_path

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

            elif path == "/api/sankey":
                self._api_sankey()

            elif path == "/sankey":
                # Serve the Sankey page from static dir
                self._send_file(STATIC_DIR / "sankey.html")

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

        if not raw_trajs:
            self._error(400, "'trajs' is required.")
            return

        trajs  = Path(raw_trajs)
        report = Path(raw_report) if raw_report else None

        # ── Path existence ────────────────────────────────────────────────────
        if not trajs.exists():
            self._error(400, f"Trajectories path not found: {trajs}")
            return
        if report is not None and not report.exists():
            self._error(400, f"Eval report not found: {report}")
            return

        # ── Agent type inference ──────────────────────────────────────────────
        if trajs.is_file() and trajs.suffix == ".jsonl":
            agent_type = "oh"
        elif trajs.is_dir():
            # Peek inside: .traj.json files indicate mini-swe-agent; otherwise SWE-agent
            if any(trajs.rglob("*.traj.json")):
                agent_type = "msa"
            else:
                agent_type = "sa"
        else:
            self._error(
                400,
                f"Trajectories path must be a directory (SWE-agent or mini-swe-agent) "
                f"or an output.jsonl file (OpenHands): {trajs}",
            )
            return

        # ── Report validity ───────────────────────────────────────────────────
        report_ids: set[str] = set()
        if report is not None:
            try:
                with open(report) as fh:
                    report_data = json.load(fh)
            except (json.JSONDecodeError, OSError) as exc:
                self._error(400, f"Could not read eval report as JSON: {exc}")
                return

            report_ids = set(
                report_data.get("resolved_ids", []) + report_data.get("unresolved_ids", [])
            )

            # ── Overlap check ─────────────────────────────────────────────────
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
            GraphHandler.eval_report_path = str(report) if report else None
            GraphHandler._graphs_cache    = None
            GraphHandler._render_cache    = {}
            GraphHandler._sankey_cache    = None

        self._respond_json({
            "ok":          True,
            "trajs":       str(trajs),
            "eval_report": str(report) if report else "",
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

    def _api_sankey(self):
        """Return aggregated phase-per-step data for the Sankey diagram.

        Response shape:
        {
          "trajectories": [
            { "instance_id": "...", "status": "resolved", "phases": ["general","localization",...] },
            ...
          ]
        }

        Each entry's ``phases`` list is indexed by step (step 0, 1, 2, …).  The
        phase is the dominant phase of the first parsed command at that step.

        We build this by loading every trajectory lightly — we only need the
        per-step phase sequence, not the full node-link graph.  Results are
        cached after the first call and invalidated when the data source changes.
        """
        with self._cache_lock:
            cached = self._sankey_cache
        if cached is not None:
            logger.info("[handler] Sankey cache hit.")
            self._respond_json(cached)
            return

        logger.info("[handler] Building Sankey data (cache miss).")

        # Ensure graph list is built first (cheap; uses its own cache)
        with self._cache_lock:
            if self._graphs_cache is None:
                GraphHandler._graphs_cache = scan_trajectories(
                    self.graphs_dir, self.eval_report_path, agent_type=self.agent_type,
                )
            graphs = self._graphs_cache

        trajectories = []
        for meta in graphs:
            instance_id = meta["instance_id"]
            try:
                traj_data = load_trajectory(
                    self.graphs_dir, instance_id, agent_type=self.agent_type,
                )
                phases = _extract_phase_sequence(
                    traj_data, self.agent_type, self.cmd_parser,
                )
            except Exception as exc:
                logger.warning("[sankey] Skipping %s: %s", instance_id, exc)
                phases = []

            trajectories.append({
                "instance_id": instance_id,
                "status":      meta.get("status", "none"),
                "phases":      phases,
            })

        result = {"trajectories": trajectories}

        with self._cache_lock:
            GraphHandler._sankey_cache = result

        self._respond_json(result)

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
# Sankey phase-extraction helper
# ---------------------------------------------------------------------------

def _extract_phase_sequence(traj_data: dict, agent_type: str, cmd_parser) -> list[str]:
    """Return a list of phase strings, one per trajectory step.

    This mirrors the logic in build_graph / _build_graph_oh but is intentionally
    lightweight: it only needs the dominant phase of each step, not the full
    graph structure.  Unrecognised or empty steps are represented as "general".
    """
    try:
        from mapPhase import get_phase
    except ImportError:
        def get_phase(*_args, **_kwargs):
            return "general"

    phases: list[str] = []

    if agent_type == "oh":
        # ── OpenHands ────────────────────────────────────────────────────────
        prev_phases_list: list[str] = []
        for step in traj_data.get("history", []):
            obs_type = step.get("observation")
            if obs_type in ("system", "message") or obs_type is None:
                continue

            tool_call_meta = step.get("tool_call_metadata", {})
            model_response = tool_call_meta.get("model_response", {})
            choices        = model_response.get("choices", [])

            step_phase = "general"
            for choice in choices:
                msg = choice.get("message", {})
                for tc in (msg.get("tool_calls") or []):
                    fn = tc.get("function", {})
                    tool_name = fn.get("name", "")
                    args_raw  = fn.get("arguments", "{}")
                    try:
                        args = json.loads(args_raw) if isinstance(args_raw, str) else args_raw
                    except Exception:
                        args = {}
                    subcommand = args.pop("command", None)
                    step_phase = get_phase(
                        tool_name, subcommand, "", args, prev_phases_list, {},
                    )
                    break  # use first tool call's phase
                if step_phase != "general":
                    break

            phases.append(step_phase)
            prev_phases_list.append(step_phase)
        return phases

    if agent_type == "msa":
        # ── mini-swe-agent ────────────────────────────────────────────────────
        prev_phases_list: list[str] = []
        messages = traj_data.get("messages", [])

        # v1.0 text format
        if traj_data.get("trajectory_format") == "mini-swe-agent-1":
            import re
            i = 2
            while i < len(messages):
                msg = messages[i]
                if msg.get("role") != "assistant":
                    i += 1
                    continue
                content = msg.get("content", "")
                if not isinstance(content, str) or not content.strip():
                    i += 2
                    continue
                import re as _re
                bash_match = _re.search(r'```bash\s*(.*?)```', content, _re.DOTALL)
                action_str = bash_match.group(1).strip() if bash_match else ""
                step_phase = "general"
                if action_str and cmd_parser:
                    cmds = cmd_parser.parse(action_str)
                    if cmds:
                        p = cmds[0]
                        step_phase = get_phase(
                            p.get("tool",""), p.get("subcommand",""),
                            p.get("command",""), p.get("args",{}),
                            prev_phases_list, p.get("flags",{}),
                        )
                phases.append(step_phase)
                prev_phases_list.append(step_phase)
                i += 2
            return phases

        # Default MSA structured format
        i = 2
        while i < len(messages):
            msg = messages[i]
            if not isinstance(msg.get("output"), list):
                i += 1
                continue
            step_phase = "general"
            for block in msg["output"]:
                if isinstance(block, dict) and block.get("type") == "function_call":
                    try:
                        args_json = json.loads(block.get("arguments", "{}"))
                    except Exception:
                        args_json = {}
                    cmd_str = args_json.get("command", "")
                    if cmd_str and cmd_parser:
                        cmds = cmd_parser.parse(cmd_str)
                        if cmds:
                            p = cmds[0]
                            step_phase = get_phase(
                                p.get("tool",""), p.get("subcommand",""),
                                p.get("command",""), p.get("args",{}),
                                prev_phases_list, p.get("flags",{}),
                            )
                    break
            phases.append(step_phase)
            prev_phases_list.append(step_phase)
            i += 2
        return phases

    # ── SWE-agent ─────────────────────────────────────────────────────────────
    prev_phases_list: list[str] = []
    for step in traj_data.get("trajectory", []):
        action_str = step.get("action", "")
        step_phase = "general"
        if action_str.strip() and cmd_parser:
            cmds = cmd_parser.parse(action_str)
            if cmds:
                p = cmds[0]
                step_phase = get_phase(
                    p.get("tool",""), p.get("subcommand",""),
                    p.get("command",""), p.get("args",{}),
                    prev_phases_list, p.get("flags",{}),
                )
        phases.append(step_phase)
        prev_phases_list.append(step_phase)
    return phases


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
    elif agent_type == "msa":
        for traj_file in trajs.rglob("*.traj.json"):
            ids.add(traj_file.name[: -len(".traj.json")])
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