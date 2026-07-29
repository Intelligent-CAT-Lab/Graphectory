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
GET  /api/graphs                → incremental trajectory metadata and progress
GET  /api/graph?id=X[&…]        → on-demand graph HTML for instance X
GET  /api/sankey                → aggregated phase-per-step data for Sankey diagram
GET  /api/config                → currently active trajs path and eval_report path
GET  /api/select-path           → local native file or folder picker
POST /api/config                → swap trajs/eval_report live; validates paths and overlap
"""

import json
import logging
import os
import socket
import threading
import time
from collections import OrderedDict
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse

from server.graph_builder import (
    _codex_action_entries,
    build_graph,
    codex_shell_phase,
    codex_tool_phase,
    detect_agent_type,
    iter_codex_steps,
    claude_tool_phase,
    iter_claude_steps,
    load_trajectory,
    _claude_code_session_files,
    clear_trajectory_caches,
    scan_trajectories,
)
from server.graph_renderer import render_graph_html

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).parent / "static"
BROWSER_LOGO_PATH = STATIC_DIR / "browser_logo.png"

_MIME: dict[str, str] = {
    ".html": "text/html; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".js":   "application/javascript; charset=utf-8",
    ".png":  "image/png",
}

# Minimum fraction of trajectory instance IDs that must appear in the report
# (or vice-versa) for the pairing to be considered valid.
_OVERLAP_THRESHOLD = 0.05

_CLIENT_DISCONNECT_ERRORS = (
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionResetError,
)

_PATH_PICKER_LOCK = threading.Lock()
_LOCALHOST_ADDRESSES = {"127.0.0.1", "::1"}


class GraphHandler(BaseHTTPRequestHandler):
    """Thin, thread-safe HTTP handler — delegates all logic to the server modules."""

    # ── Injected by live_graph_server.py before the server starts ────────────
    graphs_dir:       Path = None
    agent_type:       str  = "sa"   # "sa" | "oh" | "msa" | "claude" | "codex"
    eval_report_path: str  = None
    cmd_parser             = None
    assets_dir:       Path = None   # directory containing graph_template.html etc.

    # ── In-memory caches (class-level, shared across all handler instances) ──
    _cache_lock:   threading.RLock  = threading.RLock()
    _sankey_build_lock: threading.Lock = threading.Lock()
    _graphs_cache: Optional[list]   = None
    _graphs_loading: bool = False
    _graphs_progress: dict = {
        "loaded": 0, "total": 0, "graphs": [], "error": None,
    }
    _graphs_generation: int = 0
    # Rendered graph pages can be large because they include the raw step
    # tuples needed by the detail sidebar. Keep this cache bounded so browsing
    # a large corpus cannot gradually exhaust the server process's memory.
    _render_cache: OrderedDict[tuple, str] = OrderedDict()
    _render_cache_chars: int = 0
    _RENDER_CACHE_MAX_ENTRIES = 8
    _RENDER_CACHE_MAX_CHARS = 64 * 1024 * 1024
    # Graph construction is independent of thought-length display,
    # observations, and repeated-edge filtering. Retain a few recent graphs
    # so changing a view option only rerenders HTML instead of reparsing data.
    _graph_cache: OrderedDict[tuple, object] = OrderedDict()
    _GRAPH_CACHE_MAX_ENTRIES = 4
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

            elif path in {"/browser_logo.png", "/favicon.png"}:
                self._send_file(BROWSER_LOGO_PATH)

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
                    filter_repeated  = _bool_param(params, "filter_repeated",  default=False),
                    thought_quotes   = _bool_param(params, "thought_quotes",   default=True),
                    show_observation = _bool_param(params, "show_observation", default=False),
                    unique_think     = _bool_param(params, "unique_think",     default=True),
                    prefetch         = _bool_param(params, "prefetch",         default=False),
                )

            elif path == "/api/sankey":
                self._api_sankey()

            elif path == "/api/config":
                self._api_get_config()

            elif path == "/api/select-path":
                self._api_select_path(params)

            else:
                self._error(404, "Not found")

        except _CLIENT_DISCONNECT_ERRORS as exc:
            logger.info("[handler] Client disconnected during GET %s: %s", self.path, exc)
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

        except _CLIENT_DISCONNECT_ERRORS as exc:
            logger.info("[handler] Client disconnected during POST %s: %s", self.path, exc)
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

        trajs = Path(os.path.expandvars(raw_trajs)).expanduser()
        report = Path(os.path.expandvars(raw_report)).expanduser() if raw_report else None

        # ── Path existence ────────────────────────────────────────────────────
        if not trajs.exists():
            self._error(400, f"Trajectories path not found: {trajs}")
            return
        if report is not None and not report.exists():
            self._error(400, f"Eval report not found: {report}")
            return
        if report is not None and report.is_dir():
            resolved_report = _report_from_directory(report)
            if resolved_report is None:
                self._error(
                    400,
                    "The selected eval-report folder must contain exactly one JSON report "
                    "or a file named report.json.",
                )
                return
            report = resolved_report

        # ── Agent type inference ──────────────────────────────────────────────
        if trajs.is_file() and trajs.suffix == ".jsonl":
            agent_type = detect_agent_type(trajs)
        elif trajs.is_dir():
            agent_type = detect_agent_type(trajs)
        else:
            self._error(
                400,
                f"Trajectories path must be a supported trajectory directory "
                f"or JSONL file: {trajs}",
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
        clear_trajectory_caches()
        with self._cache_lock:
            GraphHandler.graphs_dir       = trajs
            GraphHandler.agent_type       = agent_type
            GraphHandler.eval_report_path = str(report) if report else None
            GraphHandler._graphs_cache    = None
            GraphHandler._graphs_generation += 1
            GraphHandler._graphs_loading  = False
            GraphHandler._graphs_progress  = {
                "loaded": 0, "total": 0, "graphs": [], "error": None,
            }
            GraphHandler._render_cache    = OrderedDict()
            GraphHandler._render_cache_chars = 0
            GraphHandler._graph_cache      = OrderedDict()
            GraphHandler._sankey_cache    = None

        self._respond_json({
            "ok":          True,
            "trajs":       str(trajs),
            "eval_report": str(report) if report else "",
            "agent_type":  agent_type,
        })

    def _api_select_path(self, params: dict[str, list[str]]):
        """Open a native picker on the machine running the local server."""
        purpose = params.get("purpose", [""])[0]
        kind = params.get("kind", [""])[0]
        if purpose not in {"trajs", "report"} or kind not in {"file", "folder"}:
            self._error(400, "Picker requires purpose=trajs|report and kind=file|folder.")
            return
        if self.client_address[0] not in _LOCALHOST_ADDRESSES:
            self._error(403, "Path selection is available only from the local machine.")
            return

        try:
            selected = _select_local_path(purpose, kind)
        except RuntimeError as exc:
            self._error(503, str(exc))
            return

        self._respond_json({"path": selected or ""})

    @staticmethod
    def _estimate_graph_total(trajs: Optional[Path], agent_type: str) -> int:
        """Estimate source records without parsing every trajectory."""
        if trajs is None:
            return 0
        if trajs.is_file():
            if agent_type == "oh":
                try:
                    with open(trajs, encoding="utf-8", errors="replace") as stream:
                        return sum(1 for line in stream if line.strip())
                except OSError:
                    return 0
            return 1
        if agent_type == "claude":
            # A Claude Code dataset may contain auxiliary JSONL files beside
            # each transcript. Count only the files the scanner will expose.
            return len(_claude_code_session_files(trajs))
        patterns = {
            "codex": "*.jsonl",
            "claude": "*.jsonl",
            "msa": "*.traj.json",
            "sa": "*.traj",
        }
        return sum(1 for _ in trajs.rglob(patterns.get(agent_type, "*.traj")))

    def _start_graph_loading(self):
        """Start a background trajectory scan if one is not already running."""
        with self._cache_lock:
            if self._graphs_cache is not None or self._graphs_loading:
                return
            if self.graphs_dir is None:
                GraphHandler._graphs_cache = []
                return

            GraphHandler._graphs_loading = True
            generation = self._graphs_generation
            trajs = self.graphs_dir
            report = self.eval_report_path
            agent_type = self.agent_type
            GraphHandler._graphs_progress = {
                "loaded": 0,
                "total": self._estimate_graph_total(trajs, agent_type),
                "graphs": [],
                "error": None,
            }

        def on_progress(loaded: int, total: int, partial: list[dict]):
            with self._cache_lock:
                if generation != self._graphs_generation:
                    return
                GraphHandler._graphs_progress = {
                    "loaded": loaded,
                    "total": total,
                    "graphs": partial,
                    "error": None,
                }

        def worker():
            try:
                graphs = scan_trajectories(
                    trajs,
                    report,
                    agent_type=agent_type,
                    progress_callback=on_progress,
                    progress_batch_size=10,
                )
                with self._cache_lock:
                    if generation != self._graphs_generation:
                        return
                    GraphHandler._graphs_cache = graphs
                    GraphHandler._graphs_loading = False
                    GraphHandler._graphs_progress = {
                        "loaded": len(graphs),
                        "total": max(
                            self._graphs_progress.get("total", 0), len(graphs),
                        ),
                        "graphs": graphs,
                        "error": None,
                    }
            except Exception as exc:
                logger.exception("[handler] Failed to build trajectory list")
                with self._cache_lock:
                    if generation != self._graphs_generation:
                        return
                    GraphHandler._graphs_loading = False
                    GraphHandler._graphs_progress = {
                        "loaded": 0,
                        "total": self._graphs_progress.get("total", 0),
                        "graphs": [],
                        "error": str(exc),
                    }

        threading.Thread(
            target=worker,
            name="graphectory-trajectory-loader",
            daemon=True,
        ).start()

    def _api_graphs(self):
        """Return cached or incremental trajectory metadata."""
        self._start_graph_loading()
        with self._cache_lock:
            progress = dict(self._graphs_progress)
            loading = self._graphs_loading
            error = progress.get("error")

        self._respond_json({
            "status": "error" if error else ("loading" if loading else "complete"),
            "loaded": progress.get("loaded", 0),
            "total": progress.get("total", 0),
            "graphs": progress.get("graphs", []),
            "error": error,
        })

    def _api_graph(
        self,
        instance_id:      str,
        filter_cd:        bool,
        filter_repeated:  bool,
        thought_quotes:   bool,
        show_observation: bool,
        unique_think:     bool,
        prefetch:         bool = False,
    ):
        """Build, cache, and serve graph HTML for *instance_id*.

        Prefetch requests populate the same render cache as normal requests but
        return an empty response because the browser only needs the cache warm.
        """
        cache_key = (instance_id, filter_cd, filter_repeated,
                     thought_quotes, show_observation, unique_think)

        with self._cache_lock:
            cached = self._render_cache.pop(cache_key, None)
            if cached is not None:
                # Treat recently viewed pages as the most useful cache entries.
                self._render_cache[cache_key] = cached
        if cached is not None:
            logger.info("[handler] Cache hit for '%s'.", instance_id)
            if prefetch:
                self._respond(204, "text/plain; charset=utf-8", b"")
                return
            self._respond(200, "text/html; charset=utf-8", cached.encode())
            return

        graph_key = (instance_id, filter_cd, unique_think)
        with self._cache_lock:
            G = self._graph_cache.pop(graph_key, None)
            if G is not None:
                self._graph_cache[graph_key] = G

        if G is None:
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
            with self._cache_lock:
                self._graph_cache[graph_key] = G
                while len(self._graph_cache) > self._GRAPH_CACHE_MAX_ENTRIES:
                    self._graph_cache.popitem(last=False)
        else:
            logger.info("[handler] Graph cache hit for '%s'.", instance_id)

        html = render_graph_html(
            G, filter_cd, thought_quotes, show_observation,
            self.assets_dir, filter_repeated=filter_repeated,
        )

        with self._cache_lock:
            html_chars = len(html)
            if html_chars <= self._RENDER_CACHE_MAX_CHARS:
                old = self._render_cache.pop(cache_key, None)
                if old is not None:
                    self._render_cache_chars -= len(old)
                self._render_cache[cache_key] = html
                self._render_cache_chars += html_chars

                while (
                    len(self._render_cache) > self._RENDER_CACHE_MAX_ENTRIES
                    or self._render_cache_chars > self._RENDER_CACHE_MAX_CHARS
                ):
                    _, evicted = self._render_cache.popitem(last=False)
                    self._render_cache_chars -= len(evicted)

        if prefetch:
            self._respond(204, "text/plain; charset=utf-8", b"")
            return
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
        self._start_graph_loading()
        while True:
            with self._cache_lock:
                loading = self._graphs_loading
            if not loading:
                break
            time.sleep(0.05)

        with self._cache_lock:
            cached = self._sankey_cache
        if cached is not None:
            logger.info("[handler] Sankey cache hit.")
            self._respond_json(cached)
            return

        with self._sankey_build_lock:
            # Another request may have filled the cache while we were waiting.
            with self._cache_lock:
                cached = self._sankey_cache
                if cached is not None:
                    logger.info("[handler] Sankey cache hit after wait.")
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

            # OpenHands is stored in a single large JSONL file. Loading each
            # instance individually would rescan the file once per trajectory,
            # which makes Sankey generation quadratic in dataset size.
            # Instead, stream the file exactly once and then emit results in the
            # same order as the cached metadata list.
            if self.agent_type == "oh" and self.graphs_dir and self.graphs_dir.is_file():
                status_by_id = {
                    meta["instance_id"]: meta.get("status", "none")
                    for meta in graphs
                }
                phases_by_id: dict[str, list[str]] = {}

                with open(self.graphs_dir, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        instance_id = entry.get("instance_id")
                        if not instance_id or instance_id not in status_by_id:
                            continue
                        try:
                            phases_by_id[instance_id] = _extract_phase_sequence(
                                entry, self.agent_type, self.cmd_parser,
                            )
                        except Exception as exc:
                            logger.warning("[sankey] Skipping %s: %s", instance_id, exc)
                            phases_by_id[instance_id] = []

                for meta in graphs:
                    instance_id = meta["instance_id"]
                    trajectories.append({
                        "instance_id": instance_id,
                        "status":      meta.get("status", "none"),
                        "phases":      phases_by_id.get(instance_id, []),
                    })
            else:
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
        try:
            self.send_response(status)
            self.send_header("Content-Type",   content_type)
            self.send_header("Content-Length", len(body))
            self.send_header("Cache-Control",  "no-cache")
            self.end_headers()
            self.wfile.write(body)
        except _CLIENT_DISCONNECT_ERRORS:
            raise
        except OSError as exc:
            # Windows may surface mid-response client drops as a generic OSError.
            if isinstance(exc, socket.timeout):
                raise
            if getattr(exc, "winerror", None) in {10053, 10054}:
                logger.info("[handler] Client disconnected while sending %s", self.path)
                return
            raise

    def _error(self, status: int, message: str):
        try:
            self._respond(status, "application/json; charset=utf-8",
                          json.dumps({"error": message}).encode())
        except _CLIENT_DISCONNECT_ERRORS:
            logger.info("[handler] Client disconnected before error response for %s", self.path)


# ---------------------------------------------------------------------------
# Local path-picker helpers
# ---------------------------------------------------------------------------

def _select_local_path(purpose: str, kind: str) -> str:
    """Return a native picker selection, or an empty string when cancelled.

    The picker deliberately lives in the local Python process rather than the
    browser: web pages are not permitted to reveal absolute filesystem paths.
    This also keeps the manual text fields as a reliable fallback for headless
    or containerised server deployments.
    """
    if not _PATH_PICKER_LOCK.acquire(blocking=False):
        raise RuntimeError("A file picker is already open.")

    root = None
    try:
        try:
            import tkinter as tk
            from tkinter import filedialog
        except ImportError as exc:
            raise RuntimeError("Native file selection is unavailable; enter the path manually.") from exc

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        if kind == "folder":
            selected = filedialog.askdirectory(title="Select trajectory or report folder", parent=root)
        else:
            filetypes = (
                [("Trajectory files", "*.jsonl *.json"), ("All files", "*.*")]
                if purpose == "trajs"
                else [("JSON reports", "*.json"), ("All files", "*.*")]
            )
            selected = filedialog.askopenfilename(
                title="Select trajectory file" if purpose == "trajs" else "Select evaluation report",
                filetypes=filetypes,
                parent=root,
            )
        return str(Path(selected)) if selected else ""
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "Native file selection is unavailable in this server environment; enter the path manually."
        ) from exc
    finally:
        if root is not None:
            root.destroy()
        _PATH_PICKER_LOCK.release()


def _report_from_directory(directory: Path) -> Optional[Path]:
    """Resolve a report folder only when it has an unambiguous JSON report."""
    named_report = directory / "report.json"
    if named_report.is_file():
        return named_report

    candidates = sorted(path for path in directory.glob("*.json") if path.is_file())
    return candidates[0] if len(candidates) == 1 else None


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

    if agent_type == "codex":
        prev_phases_list: list[str] = []
        for step in iter_codex_steps(traj_data):
            step_phase = "general"
            for call in step.get("calls", []):
                entries = _codex_action_entries(call, cmd_parser, filter_cd=False)
                for entry in entries:
                    parsed = entry["parsed"]
                    candidate = entry.get("phase")
                    if candidate is None:
                        candidate = codex_shell_phase(parsed, prev_phases_list)
                    elif candidate == "general":
                        candidate = codex_tool_phase(
                            entry["native_tool"], parsed.get("args", {}),
                            prev_phases_list,
                        )
                    if candidate != "general":
                        step_phase = candidate
                        break
                if step_phase != "general":
                    break
            phases.append(step_phase)
            prev_phases_list.append(step_phase)
        return phases

    if agent_type == "claude":
        prev_phases_list: list[str] = []
        for step in iter_claude_steps(traj_data):
            step_phase = "general"
            calls = step.get("calls", [])
            for call in calls:
                tool_name = str(call.get("name") or "")
                args = dict(call.get("args") or {})
                candidate = "general"
                if tool_name.lower() in {"bash", "shell", "execute_bash"}:
                    command_text = str(args.get("command") or "").strip()
                    commands = cmd_parser.parse(command_text) if command_text and cmd_parser else []
                    for parsed in commands:
                        command_phase = get_phase(
                            parsed.get("tool", ""), parsed.get("subcommand", ""),
                            parsed.get("command", ""), parsed.get("args", {}),
                            prev_phases_list, parsed.get("flags", {}),
                            framework="claude",
                        )
                        if command_phase != "general":
                            candidate = command_phase
                            break
                else:
                    candidate = claude_tool_phase(tool_name, args, prev_phases_list)
                if candidate != "general":
                    step_phase = candidate
                    break
            phases.append(step_phase)
            prev_phases_list.append(step_phase)
        return phases

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
    elif agent_type == "claude":
        ids.update(
            item["instance_id"]
            for item in scan_trajectories(trajs, agent_type="claude")
        )
    elif agent_type == "codex":
        ids.update(
            item["instance_id"]
            for item in scan_trajectories(trajs, agent_type="codex")
        )
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
