"""
server/graph_renderer.py

Converts a NetworkX MultiDiGraph into a self-contained HTML string
by filling in the graph_template.html with inlined CSS/JS and graph data.

The public surface is a single function:

    html = render_graph_html(G, filter_cd, assets_dir)
"""

import json
import os
from pathlib import Path


def _safe_json(obj) -> str:
    """Serialize obj to JSON, escaping '</' so observation text containing
    '</script>' cannot terminate the surrounding <script> tag in the HTML.
    This is the standard technique for embedding JSON in HTML script blocks.
    """
    return json.dumps(obj).replace('</', r'<\/')
from typing import Any

import networkx as nx


FONT_FAMILY = os.environ.get("GRAPH_FONT", "DejaVu Sans, Arial, sans-serif")

PHASE_COLORS = {
    "localization": "#C5B3F0",
    "patch":        "#FCC9B0",
    "validation":   "#A8E6F0",
    "general":      "#CFE0F6",
}


# ── Public API ──────────────────────────────────────────────────────────────

def render_graph_html(G: nx.MultiDiGraph, filter_cd: bool,
                      thought_quotes: bool, node_verbosity: bool,
                      show_observation: bool, assets_dir: Path) -> str:
    """Return a complete, self-contained HTML string for the graph."""
    nodes_data = _prepare_nodes(G)
    edges_data = _prepare_edges(G)

    meta = {
        "instance_name":     G.graph.get("instance_name", "Unknown"),
        "resolution_status": G.graph.get("resolution_status", "unknown"),
        "difficulty":        str(G.graph.get("debug_difficulty", "unknown")),
        "node_count":        str(len(nodes_data)),
        "edge_count":        str(len(edges_data)),
        "metadata_comment":  f"Mode: {'cd filtered (▲ hat)' if filter_cd else 'cd as node'}",
    }

    # Render settings for JS
    settings = {
        "thoughtQuotes":   thought_quotes,
        "nodeVerbosity":   node_verbosity,
        "showObservation": show_observation,
    }

    template = _load(assets_dir / "graph_template.html")
    css      = _load(assets_dir / "styles.css").replace("{{FONT_FAMILY}}", FONT_FAMILY)
    js       = _load(assets_dir / "graph_renderer.js")

    html = template

    # Metadata substitutions
    html = html.replace("{{INSTANCE_NAME}}",     _esc(meta["instance_name"]))
    html = html.replace("{{RESOLUTION_STATUS}}", meta["resolution_status"])
    html = html.replace("{{DIFFICULTY}}",        _esc(meta["difficulty"]))
    html = html.replace("{{NODE_COUNT}}",        meta["node_count"])
    html = html.replace("{{EDGE_COUNT}}",        meta["edge_count"])
    html = html.replace("{{METADATA_COMMENT}}",  _esc(meta["metadata_comment"]))

    # Data substitutions
    html = html.replace("{{NODES_DATA}}",   _safe_json(nodes_data))
    html = html.replace("{{EDGES_DATA}}",   _safe_json(edges_data))
    html = html.replace("{{PHASE_COLORS}}", _safe_json(PHASE_COLORS))
    html = html.replace("{{SETTINGS}}",     _safe_json(settings))

    # Inline CSS and JS so the response is fully self-contained
    html = html.replace(
        '<link rel="stylesheet" href="styles.css">',
        f'<style>{css}</style>',
    )
    html = html.replace(
        '<script src="graph_renderer.js"></script>',
        f'<script>{js}</script>',
    )

    return html


# ── Node preparation ────────────────────────────────────────────────────────

def _prepare_nodes(G: nx.MultiDiGraph) -> list[dict[str, Any]]:
    nodes = []
    for node_id, data in G.nodes(data=True):
        colors        = _node_colors(data)
        primary_color = colors[0] if colors else PHASE_COLORS["general"]

        has_cd = bool(data.get("has_cd", False))

        # Observation data (for last node of each step)
        obs_length = data.get("observation_length", 0)

        nodes.append({
            "id":                  node_id,
            "label":               _make_label(data),
            "label_minimal":       _make_label_minimal(data),
            "tooltip":             _make_tooltip(data),
            "color":               primary_color,
            "colors":              colors,
            "has_cd":              has_cd,
            "observation_length":  obs_length,
            "tool":                data.get("tool", ""),
            "subcommand":          data.get("subcommand", ""),
            "step_data":           data.get("step_data", []),
        })
    return nodes


def _node_colors(data: dict) -> list[str]:
    """Return ordered list of phase colour hexes for a node."""
    phases = data.get("phases") or ["general"]
    order  = ["localization", "patch", "validation", "general"]
    seen, result = set(), []
    for ph in order:
        if ph in phases and ph not in seen:
            seen.add(ph); result.append(ph)
    for ph in phases:
        if ph not in seen:
            seen.add(ph); result.append(ph)
    return [PHASE_COLORS.get(ph, PHASE_COLORS["general"]) for ph in result]


def _make_label_minimal(data: dict) -> str:
    """Minimal label: the action verb extracted from the stored label.

    The stored ``label`` is the canonical title set during node creation
    (e.g. ``"str_replace_editor: view"``, ``"grep -r foo src/"``).
    For compact display we extract just the leading verb/subcommand:
      - ``str_replace_editor: view``  → ``view``
      - ``grep -r foo src/``          → ``grep``
      - ``python``                    → ``python``
    """
    raw_label = (data.get("label") or "").strip()
    if not raw_label:
        return "action"

    # For "tool: subcommand" format, the subcommand is the meaningful part
    if ": " in raw_label:
        return raw_label.split(": ", 1)[1].split()[0][:20]

    # Otherwise take the first token (the command verb)
    return raw_label.split()[0][:20] if raw_label.split() else raw_label[:20]


def _make_label(data: dict) -> str:
    """Verbose node label: title + subtitle lines for step/path/range.

    Line 1 (title): for tool nodes, just the subcommand (e.g. ``view``);
                    for bare commands, the verb (e.g. ``python``).
    Line 2: step index(es).
    Line 3: shortened file path (args["path"] for tool nodes;
            first positional arg for python/pytest nodes).
    Line 4: view range (when present in args).
    """
    lines = []
    args = data.get("args", {}) or {}

    tool       = (data.get("tool")       or "").strip()
    subcommand = (data.get("subcommand") or "").strip()
    command    = (data.get("command")    or "").strip()
    raw_label  = (data.get("label")      or "").strip()

    # ── Line 1: clean action title ────────────────────────────────────────────
    # For tool nodes (str_replace_editor, etc.) show only the subcommand.
    # For bare shell commands show just the verb.
    if tool and subcommand:
        title = subcommand                          # e.g. "view", "str_replace"
    elif tool:
        title = tool
    elif command:
        title = command.split()[0][:20]             # verb only, e.g. "python"
    else:
        title = raw_label.split()[0][:20] if raw_label.split() else "action"

    # Outcome badge
    if isinstance(args, dict):
        status = args.get("edit_status", "") or args.get("command_outcome", "")
        if status == "success":
            title += " ✓"
        elif status and str(status).startswith("failure"):
            title += " ✗"

    lines.append(title)

    # ── Line 2: step index(es) ───────────────────────────────────────────────
    step_indices = data.get("step_indices", [])
    if step_indices:
        if len(step_indices) == 1:
            lines.append(f"step {step_indices[0]}")
        elif len(step_indices) <= 3:
            lines.append(f"steps {','.join(map(str, step_indices))}")
        else:
            lines.append(f"steps {step_indices[0]}–{step_indices[-1]} (×{len(step_indices)})")

    # ── Line 3: key argument info ────────────────────────────────────────────
    path = None

    if isinstance(args, dict):
        # Tool nodes (str_replace_editor, etc.) store path as a dict key
        path = args.get("path")

    if path is None:
        # Python/pytest nodes: commandParser puts the script as the first
        # positional arg in a list, e.g. args = ["tests/test_foo.py", "-v"]
        _PY_CMDS = {"python", "python3", "python2", "pytest"}
        verb = command.split()[0].lower() if command.split() else ""
        if verb in _PY_CMDS:
            # args may be a list of positional arguments
            candidates = args if isinstance(args, list) else []
            for tok in candidates:
                tok = str(tok)
                if not tok.startswith("-") and (
                    tok.endswith(".py") or "/" in tok or "\\" in tok
                ):
                    path = tok
                    break

    if path:
        p = str(path).replace("\\", "/")
        parts = [pt for pt in p.split("/") if pt]
        short = ("…/" + "/".join(parts[-2:])) if len(parts) > 2 else p
        lines.append(short[:35] + ("…" if len(short) > 35 else ""))
    elif isinstance(args, dict) and args.get("_raw"):
        raw = str(args["_raw"])
        lines.append(raw[:28] + ("…" if len(raw) > 28 else ""))

    # ── Line 4: view range ───────────────────────────────────────────────────
    if isinstance(args, dict):
        vr = args.get("view_range")
        if isinstance(vr, (list, tuple)) and len(vr) == 2:
            lines.append(f"L{vr[0]}–{vr[1]}")

    return "\\n".join(lines)


def _make_tooltip(data: dict) -> str:
    """Rich HTML tooltip for a node.

    Matches the reference format:
        <Title line>
        Tool: …        Subcommand: …
        Phase(s): …
        Step: …        Thought len: …
        --- Arguments ---
        path: …
        view_range: …
        …
    """
    parts: list[str] = []

    # ── helpers ──────────────────────────────────────────────────────────────
    def _row(label: str, value: str) -> str:
        return (
            f'<div style="display:flex;gap:8px;margin:2px 0;">'
            f'<span style="color:#a0c4ff;min-width:110px;flex-shrink:0;">{_esc(label)}</span>'
            f'<span style="word-break:break-all;">{_esc(value)}</span>'
            f'</div>'
        )

    def _section(title: str) -> str:
        return (
            f'<div style="margin-top:10px;margin-bottom:3px;'
            f'font-weight:700;color:#f0c27f;'
            f'border-bottom:1px solid #555;padding-bottom:2px;">'
            f'{_esc(title)}</div>'
        )

    # ── Header: use the stored label as the canonical title ──────────────────
    raw_label = (data.get("label") or "").strip()
    header_title = raw_label[:80] + ("…" if len(raw_label) > 80 else "")

    # Re-read tool/subcommand/command for the identity rows below
    tool       = (data.get("tool")       or "").strip()
    subcommand = (data.get("subcommand") or "").strip()
    command    = (data.get("command")    or "").strip()

    parts.append(
        f'<div style="font-weight:700;font-size:14px;margin-bottom:8px;'
        f'color:#fff;border-bottom:1px solid #666;padding-bottom:5px;">'
        f'{_esc(header_title)}</div>'
    )

    # ── Identity section ─────────────────────────────────────────────────────
    if tool:
        parts.append(_row("Tool", tool))
    if subcommand:
        parts.append(_row("Subcommand", subcommand))
    if command and not tool:
        # For shell commands show the full command (truncated) when no tool
        cmd_display = command if len(command) <= 150 else command[:150] + "…"
        parts.append(_row("Command", cmd_display))

    phases = data.get("phases") or ["general"]
    parts.append(_row("Phase(s)", ", ".join(sorted(set(phases)))))

    # ── Execution section ────────────────────────────────────────────────────
    step_indices    = data.get("step_indices", [])
    thought_lengths = data.get("thought_lengths", [])

    if step_indices:
        if len(step_indices) == 1:
            parts.append(_row("Step", str(step_indices[0])))
        else:
            parts.append(_row("Steps", ", ".join(map(str, step_indices))))

    if thought_lengths:
        if len(thought_lengths) == 1:
            parts.append(_row("Thought len", str(thought_lengths[0])))
        else:
            # Show as compact comma-separated list
            parts.append(_row("Thought lens", ", ".join(map(str, thought_lengths))))

    # ── Observation section ──────────────────────────────────────────────────
    obs_lengths = data.get("observation_lengths", [])
    obs_length  = data.get("observation_length",  0)

    if obs_lengths and len(obs_lengths) > 1:
        parts.append(_row("Observation lens", ", ".join(map(str, obs_lengths))))
    elif obs_length:
        parts.append(_row("Observation len", str(obs_length)))

    # ── Outcome (shown before arguments for visibility) ──────────────────────
    args = data.get("args", {}) or {}
    if isinstance(args, dict):
        edit_status     = args.get("edit_status", "")
        command_outcome = args.get("command_outcome", "")
        outcome_display = edit_status or command_outcome
        if outcome_display:
            color = "#7defa7" if outcome_display == "success" else "#ff8080"
            parts.append(
                f'<div style="display:flex;gap:8px;margin:4px 0;">'
                f'<span style="color:#a0c4ff;min-width:110px;flex-shrink:0;">Outcome</span>'
                f'<span style="color:{color};font-weight:600;">{_esc(outcome_display)}</span>'
                f'</div>'
            )

    # ── Arguments section ────────────────────────────────────────────────────
    # Internal bookkeeping keys that shouldn't be shown to the user
    _SKIP_KEYS = {"edit_status", "command_outcome"}

    if isinstance(args, dict) and args:
        visible = {k: v for k, v in args.items()
                   if k not in _SKIP_KEYS and v is not None}

        if visible:
            # If the only arg is _raw (fallback-parsed shell command remainder),
            # surface it as "Args" directly without a section header
            if list(visible.keys()) == ["_raw"]:
                raw_val = str(visible["_raw"])
                if raw_val:
                    parts.append(_row("Args", raw_val[:200] + ("…" if len(raw_val) > 200 else "")))
            else:
                parts.append(_section("Arguments"))
                for k, v in visible.items():
                    if k == "_raw":
                        continue
                    v_str = str(v)
                    display = (v_str[:300].replace("\n", "↵") + "…") if len(v_str) > 300 \
                              else v_str.replace("\n", "↵")
                    parts.append(_row(k, display))
                # Show _raw at the bottom if it exists alongside structured args
                if "_raw" in visible and visible["_raw"]:
                    parts.append(_row("raw args", str(visible["_raw"])[:200]))

    return "".join(parts)


# ── Edge preparation ────────────────────────────────────────────────────────

def _prepare_edges(G: nx.MultiDiGraph) -> list[dict[str, Any]]:
    """Serialise edges for the JS renderer with both thought length variants."""
    edges = []

    for u, v, _k, d in G.edges(keys=True, data=True):
        etype            = d.get("type", "exec")
        edge_label       = str(d.get("label", ""))
        is_first_in_step = bool(d.get("is_first_in_step", False))

        if etype == "exec":
            thought_len_raw   = int(d.get("thought_length_raw", 0))
            thought_len_clean = int(d.get("thought_length_clean", 0))

            u_steps = set(G.nodes[u].get("step_indices", []))
            v_steps = set(G.nodes[v].get("step_indices", []))
            is_multi_node = bool(u_steps & v_steps) and not is_first_in_step
            is_thought_continuation = bool(d.get("is_thought_continuation", False))

            # For first-in-step edges, carry the source node's observation
            # length so the edge body thickness can encode it.
            if is_first_in_step:
                src_data   = G.nodes[u]
                obs_length = int(src_data.get("observation_length", 0))
            else:
                obs_length = 0
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


# ── Helpers ─────────────────────────────────────────────────────────────────

def _load(path: Path) -> str:
    with open(path, encoding="utf-8") as f:
        return f.read()


def _esc(s: str) -> str:
    return (str(s) if s else "").replace("&", "&amp;").replace("<", "&lt;") \
                                 .replace(">", "&gt;").replace('"', "&quot;")