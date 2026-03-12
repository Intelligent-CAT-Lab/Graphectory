"""
server/graph_renderer.py

Converts a NetworkX MultiDiGraph into a self-contained HTML page by filling
in graph_template.html with inlined CSS, JS, and serialised graph data.

Public surface:

    html: str = render_graph_html(G, filter_cd, thought_quotes,
                                   node_verbosity, show_observation, assets_dir)
"""

import json
import logging
import os
from pathlib import Path
from typing import Any

import networkx as nx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FONT_FAMILY: str = os.environ.get("GRAPH_FONT", "DejaVu Sans, Arial, sans-serif")

PHASE_COLORS: dict[str, str] = {
    "localization": "#C5B3F0",
    "patch":        "#FCC9B0",
    "validation":   "#A8E6F0",
    "general":      "#CFE0F6",
}

_DAGRE_VERSION    = "0.8.5"
_DAGRE_CDN_URL    = f"https://unpkg.com/dagre@{_DAGRE_VERSION}/dist/dagre.min.js"
_DAGRE_LOCAL_NAME = "dagre.min.js"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def render_graph_html(
    G: nx.MultiDiGraph,
    filter_cd: bool,
    thought_quotes: bool,
    node_verbosity: bool,
    show_observation: bool,
    assets_dir: Path,
) -> str:
    """Return a complete, self-contained HTML page for *G*.

    All CSS, JS, and graph data are inlined so the page has no runtime
    dependencies beyond the dagre script tag (local copy preferred over CDN).
    """
    instance_name = G.graph.get("instance_name", "Unknown")
    logger.info(
        "[renderer] Rendering '%s'  nodes=%d  edges=%d",
        instance_name, G.number_of_nodes(), G.number_of_edges(),
    )

    nodes_data = _prepare_nodes(G)
    edges_data = _prepare_edges(G)

    resolution_status = G.graph.get("resolution_status", "none") or "none"
    meta = {
        "instance_name":     instance_name,
        "resolution_status": resolution_status,
        "difficulty":        str(G.graph.get("debug_difficulty", "unknown")),
        "node_count":        str(len(nodes_data)),
        "edge_count":        str(len(edges_data)),
    }

    settings = {
        "thoughtQuotes":   thought_quotes,
        "nodeVerbosity":   node_verbosity,
        "showObservation": show_observation,
    }

    template = _load(assets_dir / "graph_template.html")
    css      = _load(assets_dir / "styles.css").replace("{{FONT_FAMILY}}", FONT_FAMILY)
    js       = _load(assets_dir / "graph_renderer.js")

    html = template

    # Dagre — serve from local static/ when available, otherwise fall back to CDN.
    html = html.replace("{{DAGRE_SCRIPT_TAG}}", _dagre_script_tag(assets_dir))

    # Status item — omit entirely when no report was provided
    if resolution_status in ("none", "unknown", ""):
        status_item_html = ""
    else:
        status_item_html = (
            '<div class="metadata-item">'
            "<strong>Status:</strong>"
            f' <span class="status-badge status-{_esc(resolution_status)}">'
            f"{_esc(resolution_status)}</span>"
            "</div>"
        )

    # Metadata
    html = html.replace("{{STATUS_ITEM}}",       status_item_html)
    html = html.replace("{{INSTANCE_NAME}}",     _esc(meta["instance_name"]))
    html = html.replace("{{DIFFICULTY}}",        _esc(meta["difficulty"]))
    html = html.replace("{{NODE_COUNT}}",        meta["node_count"])
    html = html.replace("{{EDGE_COUNT}}",        meta["edge_count"])

    # Graph data
    html = html.replace("{{NODES_DATA}}",   _safe_json(nodes_data))
    html = html.replace("{{EDGES_DATA}}",   _safe_json(edges_data))
    html = html.replace("{{PHASE_COLORS}}", _safe_json(PHASE_COLORS))
    html = html.replace("{{SETTINGS}}",     _safe_json(settings))

    # Inline assets so the rendered page is fully self-contained.
    html = html.replace(
        '<link rel="stylesheet" href="styles.css">',
        f"<style>{css}</style>",
    )
    html = html.replace(
        '<script src="graph_renderer.js"></script>',
        f"<script>{js}</script>",
    )

    logger.info("[renderer] Done — %d bytes", len(html))
    return html


# ---------------------------------------------------------------------------
# Dagre script tag
# ---------------------------------------------------------------------------

def _dagre_script_tag(assets_dir: Path) -> str:
    """Return a ``<script>`` tag for dagre, preferring a local copy over the CDN.

    Searches for ``dagre.min.js`` in *assets_dir* and in its ``static/``
    sub-directory (the path already served by the HTTP handler).  If found,
    the file is referenced via ``/static/dagre.min.js``; otherwise the CDN
    URL is used with an ``onerror`` console warning.
    """
    for search_path in (assets_dir, assets_dir / "static"):
        if (search_path / _DAGRE_LOCAL_NAME).exists():
            logger.debug("[renderer] Using local dagre at '%s'", search_path)
            return f'<script src="/static/{_DAGRE_LOCAL_NAME}"></script>'

    logger.warning(
        "[renderer] dagre.min.js not found locally in '%s' — falling back to CDN.",
        assets_dir,
    )
    return (
        f'<script src="{_DAGRE_CDN_URL}" '
        f"onerror=\"console.error('[graph] Failed to load dagre from CDN.')\"></script>"
    )



# ---------------------------------------------------------------------------
# Node preparation
# ---------------------------------------------------------------------------

def _sanitize_text(s: str) -> str:
    """Remove or replace control characters that break JSON/HTML rendering.

    Keeps common whitespace (newline, tab, carriage return) and strips all
    other C0/C1 control characters (U+0000–U+001F excluding \\t \\n \\r,
    and U+007F–U+009F).  Surrogate code points that can't be JSON-encoded
    are also replaced with the Unicode replacement character (U+FFFD).
    """
    if not s:
        return s
    # Replace surrogates
    try:
        s = s.encode("utf-16", "surrogatepass").decode("utf-16")
    except (UnicodeDecodeError, UnicodeEncodeError):
        s = s.encode("utf-8", "replace").decode("utf-8")
    # Strip non-printable control chars except \\t \\n \\r
    return "".join(
        ch if (ch in ("\t", "\n", "\r") or (ord(ch) >= 0x20 and ord(ch) != 0x7F and ord(ch) < 0x9F)
               or ord(ch) >= 0xA0)
        else "\ufffd"
        for ch in s
    )


def _sanitize_step_data(step_data: list) -> list:
    """Return a copy of step_data with all text fields sanitized."""
    cleaned = []
    for entry in step_data:
        cleaned.append({
            "step_idx":    entry.get("step_idx", 0),
            "thought":     _sanitize_text(entry.get("thought",     "") or ""),
            "action":      _sanitize_text(entry.get("action",      "") or ""),
            "observation": _sanitize_text(entry.get("observation", "") or ""),
        })
    return cleaned


def _prepare_nodes(G: nx.MultiDiGraph) -> list[dict[str, Any]]:
    nodes = []
    for node_id, data in G.nodes(data=True):
        colors = _node_colors(data)
        nodes.append({
            "id":                 node_id,
            "label":              _make_label(data),
            "label_minimal":      _make_label_minimal(data),
            "tooltip":            _make_tooltip(data),
            "color":              colors[0] if colors else PHASE_COLORS["general"],
            "colors":             colors,
            "has_cd":             bool(data.get("has_cd", False)),
            "observation_length": int(data.get("observation_length", 0)),
            "tool":               data.get("tool", ""),
            "subcommand":         data.get("subcommand", ""),
            "step_data":          _sanitize_step_data(data.get("step_data", [])),
        })
    return nodes


def _node_colors(data: dict) -> list[str]:
    """Return an ordered list of phase colour hex strings for a node.

    Colours are emitted in canonical phase order so that gradient rendering
    is consistent across nodes that share the same set of phases.
    """
    phases = data.get("phases") or ["general"]
    order  = ["localization", "patch", "validation", "general"]
    seen:   set[str]  = set()
    result: list[str] = []

    for ph in order:
        if ph in phases and ph not in seen:
            seen.add(ph)
            result.append(ph)
    for ph in phases:
        if ph not in seen:
            seen.add(ph)
            result.append(ph)

    return [PHASE_COLORS.get(ph, PHASE_COLORS["general"]) for ph in result]


def _make_label_minimal(data: dict) -> str:
    """Single-token label used when verbose node labels are disabled.

    Examples::

        "str_replace_editor: view"  →  "view"
        "grep -r foo src/"          →  "grep"
        "python"                    →  "python"
    """
    raw = (data.get("label") or "").strip()
    if not raw:
        return "action"
    if ": " in raw:
        return raw.split(": ", 1)[1].split()[0][:20]
    tokens = raw.split()
    return tokens[0][:20] if tokens else raw[:20]


def _make_label(data: dict) -> str:
    """Multi-line verbose label: action title, step index, file path, view range.

    Line 1 — action title (subcommand for tool nodes; command verb otherwise).
    Line 2 — step index(es).
    Line 3 — shortened file path or first positional argument.
    Line 4 — view range when present.
    """
    lines: list[str] = []
    args       = data.get("args", {}) or {}
    tool       = (data.get("tool")       or "").strip()
    subcommand = (data.get("subcommand") or "").strip()
    command    = (data.get("command")    or "").strip()
    raw_label  = (data.get("label")      or "").strip()

    # Line 1 — clean action title
    if tool and subcommand:
        title = subcommand
    elif tool:
        title = tool
    elif command:
        title = command.split()[0][:20]
    else:
        tokens = raw_label.split()
        title  = tokens[0][:20] if tokens else "action"

    if isinstance(args, dict):
        status = args.get("edit_status", "") or args.get("command_outcome", "")
        if status == "success":
            title += " ✓"
        elif status and str(status).startswith("failure"):
            title += " ✗"

    lines.append(title)

    # Line 2 — step index(es)
    step_indices = data.get("step_indices", [])
    if step_indices:
        if len(step_indices) == 1:
            lines.append(f"step {step_indices[0]}")
        elif len(step_indices) <= 3:
            lines.append(f"steps {','.join(map(str, step_indices))}")
        else:
            lines.append(f"steps {step_indices[0]}–{step_indices[-1]} (×{len(step_indices)})")

    # Line 3 — key path argument
    path = args.get("path") if isinstance(args, dict) else None

    if path is None:
        _PY_CMDS = {"python", "python3", "python2", "pytest"}
        verb     = command.split()[0].lower() if command.split() else ""
        if verb in _PY_CMDS:
            candidates = args if isinstance(args, list) else []
            for tok in candidates:
                tok = str(tok)
                if not tok.startswith("-") and (tok.endswith(".py") or "/" in tok or "\\" in tok):
                    path = tok
                    break

    if path:
        p     = str(path).replace("\\", "/")
        parts = [pt for pt in p.split("/") if pt]
        short = ("…/" + "/".join(parts[-2:])) if len(parts) > 2 else p
        lines.append(short[:35] + ("…" if len(short) > 35 else ""))
    elif isinstance(args, dict) and args.get("_raw"):
        raw = str(args["_raw"])
        lines.append(raw[:28] + ("…" if len(raw) > 28 else ""))

    # Line 4 — view range
    if isinstance(args, dict):
        vr = args.get("view_range")
        if isinstance(vr, (list, tuple)) and len(vr) == 2:
            lines.append(f"L{vr[0]}–{vr[1]}")

    return "\\n".join(lines)


def _make_tooltip(data: dict) -> str:
    """Rich HTML tooltip shown on node hover."""
    parts: list[str] = []

    def _row(label: str, value: str) -> str:
        return (
            '<div style="display:flex;gap:8px;margin:2px 0;">'
            f'<span style="color:#a0c4ff;min-width:110px;flex-shrink:0;">{_esc(label)}</span>'
            f'<span style="word-break:break-all;">{_esc(value)}</span>'
            "</div>"
        )

    def _section(title: str) -> str:
        return (
            '<div style="margin-top:10px;margin-bottom:3px;font-weight:700;'
            f'color:#f0c27f;border-bottom:1px solid #555;padding-bottom:2px;">{_esc(title)}</div>'
        )

    # Header
    raw_label = (data.get("label") or "").strip()
    header    = raw_label[:80] + ("…" if len(raw_label) > 80 else "")
    parts.append(
        '<div style="font-weight:700;font-size:14px;margin-bottom:8px;'
        f'color:#fff;border-bottom:1px solid #666;padding-bottom:5px;">{_esc(header)}</div>'
    )

    tool       = (data.get("tool")       or "").strip()
    subcommand = (data.get("subcommand") or "").strip()
    command    = (data.get("command")    or "").strip()

    if tool:
        parts.append(_row("Tool", tool))
    if subcommand:
        parts.append(_row("Subcommand", subcommand))
    if command and not tool:
        cmd_display = command if len(command) <= 150 else command[:150] + "…"
        parts.append(_row("Command", cmd_display))

    phases = data.get("phases") or ["general"]
    parts.append(_row("Phase(s)", ", ".join(sorted(set(phases)))))

    step_indices    = data.get("step_indices", [])
    thought_lengths = data.get("thought_lengths", [])

    if len(step_indices) == 1:
        parts.append(_row("Step", str(step_indices[0])))
    elif step_indices:
        parts.append(_row("Steps", ", ".join(map(str, step_indices))))

    if len(thought_lengths) == 1:
        parts.append(_row("Thought len", str(thought_lengths[0])))
    elif thought_lengths:
        parts.append(_row("Thought lens", ", ".join(map(str, thought_lengths))))

    obs_lengths = data.get("observation_lengths", [])
    obs_length  = data.get("observation_length",  0)
    if len(obs_lengths) > 1:
        parts.append(_row("Observation lens", ", ".join(map(str, obs_lengths))))
    elif obs_length:
        parts.append(_row("Observation len", str(obs_length)))

    # Outcome badge
    args = data.get("args", {}) or {}
    if isinstance(args, dict):
        outcome = args.get("edit_status", "") or args.get("command_outcome", "")
        if outcome:
            color = "#7defa7" if outcome == "success" else "#ff8080"
            parts.append(
                '<div style="display:flex;gap:8px;margin:4px 0;">'
                '<span style="color:#a0c4ff;min-width:110px;flex-shrink:0;">Outcome</span>'
                f'<span style="color:{color};font-weight:600;">{_esc(outcome)}</span>'
                "</div>"
            )

    # Arguments
    _SKIP_KEYS = {"edit_status", "command_outcome"}
    if isinstance(args, dict) and args:
        visible = {k: v for k, v in args.items() if k not in _SKIP_KEYS and v is not None}
        if visible:
            if list(visible.keys()) == ["_raw"]:
                raw_val = str(visible["_raw"])
                if raw_val:
                    parts.append(_row("Args", raw_val[:200] + ("…" if len(raw_val) > 200 else "")))
            else:
                parts.append(_section("Arguments"))
                for k, v in visible.items():
                    if k == "_raw":
                        continue
                    v_str   = str(v)
                    display = (v_str[:300].replace("\n", "↵") + "…") if len(v_str) > 300 \
                              else v_str.replace("\n", "↵")
                    parts.append(_row(k, display))
                if "_raw" in visible and visible["_raw"]:
                    parts.append(_row("raw args", str(visible["_raw"])[:200]))

    return "".join(parts)


# ---------------------------------------------------------------------------
# Edge preparation
# ---------------------------------------------------------------------------

def _prepare_edges(G: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """Serialise edges for the JS renderer, including both thought-length variants."""
    edges = []

    for u, v, _k, d in G.edges(keys=True, data=True):
        etype            = d.get("type", "exec")
        edge_label       = str(d.get("label", ""))
        is_first_in_step = bool(d.get("is_first_in_step", False))

        if etype == "exec":
            thought_len_raw         = int(d.get("thought_length_raw",   0))
            thought_len_clean       = int(d.get("thought_length_clean", 0))
            u_steps                 = set(G.nodes[u].get("step_indices", []))
            v_steps                 = set(G.nodes[v].get("step_indices", []))
            is_multi_node           = bool(u_steps & v_steps) and not is_first_in_step
            is_thought_continuation = bool(d.get("is_thought_continuation", False))
            # Carry the source node's observation length on first-in-step edges
            # so the JS renderer can encode it as a visual indicator.
            obs_length = int(G.nodes[u].get("observation_length", 0)) if is_first_in_step else 0
        else:
            thought_len_raw         = 0
            thought_len_clean       = 0
            is_multi_node           = False
            is_thought_continuation = False
            obs_length              = 0

        edges.append({
            "from":                    u,
            "to":                      v,
            "type":                    etype,
            "label":                   edge_label if etype == "exec" else "",
            "thought_length_raw":      thought_len_raw,
            "thought_length_clean":    thought_len_clean,
            "is_multi_node_step":      is_multi_node,
            "is_first_in_step":        is_first_in_step,
            "is_thought_continuation": is_thought_continuation,
            "obs_length":              obs_length,
        })

    return edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_json(obj: Any) -> str:
    """Serialise *obj* to JSON, escaping ``</`` so embedded text cannot
    inadvertently close a surrounding ``<script>`` tag.
    """
    return json.dumps(obj).replace("</", r"<\/")


def _load(path: Path) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _esc(s: Any) -> str:
    """Minimal HTML escaping for attribute values and text content."""
    return (
        str(s) if s else ""
    ).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")