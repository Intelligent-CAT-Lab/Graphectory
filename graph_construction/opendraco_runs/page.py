"""
opendraco_runs/page.py

The standalone HTML page for one OpenDraco run: the run view on top (agent graph,
superstep timeline, detected patterns), and one tab per agent holding that agent's
ordinary Graphectory graph page.

Standalone means it works saved to disk or served by another app inside a
sandboxed iframe: dagre is inlined (once) and nothing is fetched.

All run text (prompts, thoughts, messages, tool output) is model output and is
treated as hostile:

* the run data is embedded as JSON inside ``<script type="application/json">`` with
  ``<``, ``>`` and ``&`` escaped, so no string can close the element;
* the page's script only ever puts strings into the DOM through ``textContent``
  (and SVG ``<title>`` text), never ``innerHTML``;
* each agent page (rendered by the upstream renderer, which escapes its own data)
  is embedded base64-encoded and only decoded into a ``sandbox="allow-scripts"``
  iframe's ``srcdoc`` when its tab is opened, so it runs in an opaque origin with
  no access to this page.
"""

import base64
import json
import re
from html import escape
from pathlib import Path

from server.graph_renderer import render_graph_html

_DAGRE_PATH = Path(__file__).resolve().parent.parent / "server" / "static" / "dagre.min.js"
# The renderer emits either the local /static/ tag or the CDN tag; both are replaced.
_DAGRE_TAG_RE = re.compile(r'<script[^>]*\bsrc="[^"]*dagre[^"]*"[^>]*>\s*</script>', re.IGNORECASE)
_DAGRE_MARKER = "<!--opendraco:dagre-->"


def _dagre_source() -> str:
    source = _DAGRE_PATH.read_text(encoding="utf-8")
    if "</script" in source.lower():  # would end the inline element early
        raise RuntimeError("dagre.min.js cannot be inlined: it contains '</script'")
    return source


def _agent_html(G, assets_dir: Path, view: dict) -> str:
    return render_graph_html(
        G,
        bool(view.get("filter_cd", False)),
        bool(view.get("thought_quotes", True)),
        bool(view.get("show_observation", False)),
        assets_dir,
        filter_repeated=bool(view.get("filter_repeated", False)),
    )


def render_agent_page(G, assets_dir: Path, view: dict) -> str:
    """One agent's ordinary graph page, made standalone (dagre inlined)."""
    html = _agent_html(G, assets_dir, view)
    inline = f"<script>{_dagre_source()}</script>"
    return _DAGRE_TAG_RE.sub(lambda _m: inline, html, count=1)


def _json_for_script(value) -> str:
    """JSON that is inert inside a <script type="application/json"> element."""
    return (json.dumps(value, default=str, ensure_ascii=False)
            .replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
            .replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))


def render_run_page(data: dict, graphs: dict, assets_dir: Path, view: dict,
                    descriptions: dict) -> str:
    agent_pages = {}
    for agent_id, G in graphs.items():
        html = _DAGRE_TAG_RE.sub(lambda _m: _DAGRE_MARKER, _agent_html(G, assets_dir, view), count=1)
        agent_pages[agent_id] = base64.b64encode(html.encode("utf-8")).decode("ascii")

    payload = {
        "run": {k: v for k, v in data.items() if k != "agents"},
        "agents": {a: {"nodes": g["nodes"], "edges": g["edges"]} for a, g in data["agents"].items()},
        "descriptions": descriptions,
    }
    title = f"OpenDraco run {data.get('id') or data.get('run_id') or ''}".strip()
    return (_PAGE
            .replace("{{TITLE}}", escape(title))
            .replace("{{DAGRE}}", _dagre_source())
            .replace("{{RUN_DATA}}", _json_for_script(payload))
            .replace("{{AGENT_PAGES}}", _json_for_script(agent_pages))
            .replace("{{DAGRE_MARKER}}", _json_for_script(_DAGRE_MARKER)))


_PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}}</title>
<style>
:root {
  --bg: #f6f7f9; --panel: #ffffff; --ink: #1f2933; --muted: #616e7c; --line: #d9dee4;
  --accent: #3e63dd; --warn: #b54708; --bad: #c0392b;
  --localization: #b9a2ed; --patch: #f8ac84; --validation: #70d1df; --plan: #f4d06f; --general: #c4cfda;
}
* { box-sizing: border-box; }
body { margin: 0; font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
header { padding: 14px 20px; background: var(--panel); border-bottom: 1px solid var(--line); }
header h1 { margin: 0 0 4px; font-size: 18px; word-break: break-all; }
.meta { color: var(--muted); display: flex; flex-wrap: wrap; gap: 4px 16px; }
.badge { display: inline-block; padding: 0 8px; border-radius: 10px; font-size: 12px; background: #e8ecf1; }
.badge.completed { background: #dff3e4; } .badge.failed { background: #fbe0dd; }
.badge.in_progress, .badge.paused { background: #fdf0d5; }
.warn { color: var(--warn); }
main { padding: 16px 20px; display: grid; gap: 16px; }
section { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 12px 14px; min-width: 0; }
section h2 { margin: 0 0 8px; font-size: 15px; }
.row { display: grid; grid-template-columns: minmax(0, 3fr) minmax(0, 2fr); gap: 16px; }
@media (max-width: 900px) { .row { grid-template-columns: 1fr; } }
.scroll { overflow: auto; max-height: 520px; }
svg text { font: 12px system-ui, sans-serif; fill: var(--ink); }
.node rect { fill: #fff; stroke: #8796a8; stroke-width: 1.2; rx: 6; }
.node.agent rect { stroke: var(--accent); }
.node.external rect, .node.start rect, .node.end rect { fill: #eef1f5; stroke-dasharray: 3 3; }
.node.agent { cursor: pointer; }
.edge path { fill: none; stroke: #7b8794; stroke-width: 1.4; }
.edge.route path { stroke: var(--warn); stroke-dasharray: 5 4; }
.edge.unused path { stroke: #c9d1da; stroke-dasharray: 2 4; }
.edge text { fill: var(--muted); font-size: 11px; }
.block rect { stroke: #52606d; stroke-width: 1; rx: 4; }
.block.abandoned rect { fill: url(#hatch) !important; stroke-dasharray: 4 3; }
.block.running rect { stroke-dasharray: 2 2; }
.arrow { fill: none; stroke: var(--accent); stroke-width: 1.2; opacity: .75; }
.arrow.parent { stroke: var(--warn); stroke-dasharray: 3 3; }
.grid line { stroke: #edf0f3; }
.patterns { list-style: none; margin: 0; padding: 0; display: grid; gap: 8px; }
.patterns li { border-left: 3px solid var(--warn); padding: 4px 10px; background: #fffaf2; }
.patterns .kind { font-weight: 600; }
.patterns .why { color: var(--muted); font-size: 12px; }
.legend { display: flex; flex-wrap: wrap; gap: 4px 12px; color: var(--muted); font-size: 12px; margin-top: 6px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 4px; vertical-align: -1px; }
table.stats { border-collapse: collapse; width: 100%; font-size: 13px; }
table.stats th, table.stats td { text-align: left; padding: 4px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
table.stats td.num { text-align: right; font-variant-numeric: tabular-nums; }
.tabs { display: flex; flex-wrap: wrap; gap: 4px; border-bottom: 1px solid var(--line); margin: -4px 0 0; }
.tabs button { border: 1px solid var(--line); border-bottom: none; background: #eef1f5; padding: 6px 12px; border-radius: 6px 6px 0 0; cursor: pointer; font: inherit; }
.tabs button[aria-selected="true"] { background: var(--panel); font-weight: 600; }
.frame-wrap { height: 78vh; min-height: 480px; }
.frame-wrap iframe { width: 100%; height: 100%; border: 0; }
.empty { color: var(--muted); font-style: italic; }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <div class="meta" id="meta"></div>
</header>
<main>
  <div class="row">
    <section><h2>Agents working together</h2><div class="scroll" id="run-graph"></div>
      <div class="legend"><span>— delivery (count × edge)</span><span class="warn">- - route choice (picked / decisions)</span><span>· · edge never used</span></div>
    </section>
    <section><h2>Agents</h2><div class="scroll"><table class="stats" id="agent-table"></table></div></section>
  </div>
  <section><h2>Timeline: who handed what to whom, when</h2><div class="scroll" id="timeline"></div>
    <div class="legend" id="timeline-legend"></div>
  </section>
  <section><h2>Patterns</h2><ul class="patterns" id="patterns"></ul><div class="legend" id="pattern-config"></div></section>
  <section><h2>Per-agent graphs</h2><div class="tabs" id="tabs" role="tablist"></div><div class="frame-wrap" id="frame"></div></section>
</main>
<script id="dagre-lib">{{DAGRE}}</script>
<script type="application/json" id="run-data">{{RUN_DATA}}</script>
<script type="application/json" id="agent-pages">{{AGENT_PAGES}}</script>
<script type="application/json" id="dagre-marker">{{DAGRE_MARKER}}</script>
<script>
(function () {
  "use strict";
  const SVG = "http://www.w3.org/2000/svg";
  const PHASES = ["localization", "patch", "validation", "plan", "general"];
  const PHASE_COLOR = {localization: "#b9a2ed", patch: "#f8ac84", validation: "#70d1df", plan: "#f4d06f", general: "#c4cfda"};
  const data = JSON.parse(document.getElementById("run-data").textContent);
  const pages = JSON.parse(document.getElementById("agent-pages").textContent);
  const marker = JSON.parse(document.getElementById("dagre-marker").textContent);
  const run = data.run, rg = run.run_graph, info = rg.graph;
  const nodes = rg.nodes, edges = rg.edges || rg.links || [];
  const agentIds = Object.keys(data.agents);

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function svg(tag, attrs, text) {
    const e = document.createElementNS(SVG, tag);
    for (const k in (attrs || {})) e.setAttribute(k, attrs[k]);
    if (text !== undefined && text !== null) e.textContent = String(text);
    return e;
  }
  function tip(parent, text) { parent.appendChild(svg("title", {}, text)); }
  function fmtTokens(n) { return n >= 10000 ? (n / 1000).toFixed(1) + "k" : String(n); }
  function fmtSec(s) { return s == null ? "–" : (s >= 60 ? (s / 60).toFixed(1) + " min" : s.toFixed(1) + " s"); }

  // ── Header ────────────────────────────────────────────────────────────
  document.getElementById("title").textContent = "OpenDraco run " + (run.id || run.run_id || "");
  const meta = document.getElementById("meta");
  meta.appendChild(el("span", {"class": "badge " + run.status}, run.status));
  if (info.framework_version) meta.appendChild(el("span", {}, "OpenDraco " + info.framework_version));
  meta.appendChild(el("span", {}, agentIds.length + " agents · " + info.timeline.activations.length +
                      " activations · " + info.timeline.deliveries.length + " deliveries · " +
                      (info.supersteps || 0) + " supersteps"));
  if (info.forked_from) meta.appendChild(el("span", {"class": "warn"}, "forked from run " + info.forked_from.run_id +
                      " at superstep " + info.forked_from.superstep));
  if ((info.resumed_at || []).length) meta.appendChild(el("span", {}, "resumed at superstep " + info.resumed_at.join(", ")));
  if (info.truncated) meta.appendChild(el("span", {"class": "warn"}, "last line half-written (run still writing, or cut off)"));
  if (info.skipped_lines) meta.appendChild(el("span", {"class": "warn"}, info.skipped_lines + " unreadable line(s) skipped"));
  (info.errors || []).forEach(function (e) {
    meta.appendChild(el("span", {"class": "warn"}, "error" + (e.agent_id ? " in " + e.agent_id : "") + ": " + (e.message || "")));
  });

  // ── Run graph (dagre layout) ──────────────────────────────────────────
  (function drawRunGraph() {
    const g = new dagre.graphlib.Graph({multigraph: true});
    g.setGraph({rankdir: "LR", nodesep: 40, ranksep: 90, marginx: 16, marginy: 16});
    g.setDefaultEdgeLabel(function () { return {}; });
    nodes.forEach(function (n) {
      const agent = n.kind === "agent";
      g.setNode(n.id, {width: agent ? 190 : 80, height: agent ? 74 : 32, data: n});
    });
    edges.forEach(function (e, i) {
      let label;
      if (e.type === "route") label = "→ " + e.choice + " " + e.count + "/" + e.of;
      else label = e.count + "× " + e.edge_id + (e.from_parent_run ? " (+parent run)" : "");
      g.setEdge(e.source, e.target, {label: label, width: Math.min(160, label.length * 6.5), height: 14, data: e}, "e" + i);
    });
    dagre.layout(g);
    const gw = g.graph().width || 200, gh = g.graph().height || 100;
    const root = svg("svg", {width: "100%", viewBox: "0 0 " + gw + " " + gh, role: "img", "aria-label": "Run graph",
                              style: "max-width:" + gw + "px;height:auto;display:block"});
    const defs = svg("defs");
    ["arrow", "arrow-route"].forEach(function (id) {
      const m = svg("marker", {id: id, viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "7", markerHeight: "7", orient: "auto-start-reverse"});
      m.appendChild(svg("path", {d: "M0,0 L10,5 L0,10 z", fill: id === "arrow" ? "#7b8794" : "#b54708"}));
      defs.appendChild(m);
    });
    root.appendChild(defs);
    g.edges().forEach(function (key) {
      const e = g.edge(key), d = e.data;
      const cls = "edge " + d.type + (d.type === "delivery" && !d.count && !d.from_parent_run ? " unused" : "");
      const grp = svg("g", {"class": cls});
      const pts = e.points.map(function (p) { return p.x + "," + p.y; });
      grp.appendChild(svg("path", {d: "M" + pts.join(" L"), "marker-end": "url(#" + (d.type === "route" ? "arrow-route" : "arrow") + ")"}));
      if (e.x !== undefined) grp.appendChild(svg("text", {x: e.x, y: e.y + 4, "text-anchor": "middle"}, e.label));
      tip(grp, d.type === "route"
        ? (d.source + " routed to " + d.choice + " " + d.count + " of " + d.of + " decisions" +
           (d.group ? " (group " + d.group + ")" : "") + "; outcomes: " + (d.outcomes || []).join(", ") +
           "; supersteps " + (d.supersteps || []).join(", "))
        : (d.edge_id + ": " + d.source + " → " + d.target + " (slot " + d.slot + "), " + d.count +
           " deliveries; supersteps " + (d.supersteps || []).join(", ") +
           (d.from_parent_run ? "; " + d.from_parent_run + " from the parent run" : "")));
      root.appendChild(grp);
    });
    g.nodes().forEach(function (id) {
      const n = g.node(id), d = n.data, x = n.x - n.width / 2, y = n.y - n.height / 2;
      const grp = svg("g", {"class": "node " + d.kind, transform: "translate(" + x + "," + y + ")"});
      grp.appendChild(svg("rect", {width: n.width, height: n.height}));
      if (d.kind !== "agent") {
        grp.appendChild(svg("text", {x: n.width / 2, y: 20, "text-anchor": "middle"}, id));
      } else {
        grp.appendChild(svg("text", {x: 10, y: 18, "font-weight": "600"}, id + (d.changed_workspace ? "  ✎" : "")));
        grp.appendChild(svg("text", {x: 10, y: 35}, d.activations + " act · " + fmtTokens(d.tokens.total_tokens) +
                                                    " tok · " + fmtSec(d.duration_s)));
        // Phase mix bar
        const total = PHASES.reduce(function (s, p) { return s + (d.phase_mix[p] || 0); }, 0);
        let off = 10;
        PHASES.forEach(function (p) {
          const v = d.phase_mix[p] || 0;
          if (!v || !total) return;
          const w = (n.width - 20) * v / total;
          const r = svg("rect", {x: off, y: 46, width: w, height: 12, style: "fill:" + PHASE_COLOR[p] + ";stroke:none"});
          tip(r, p + ": " + v);
          grp.appendChild(r);
          off += w;
        });
        if (!total) grp.appendChild(svg("text", {x: 10, y: 57, "class": "empty"}, "no actions"));
        tip(grp, id + ": " + d.activations + " activations" + (d.abandoned_activations ? " (+" + d.abandoned_activations + " abandoned)" : "") +
          ", " + d.tool_calls + " tool calls, " + d.tokens.total_tokens + " tokens, " + fmtSec(d.duration_s) +
          (d.changed_workspace ? ", changed: " + d.changed_paths.join(", ") : ", never changed the workspace") +
          (d.model ? ", model " + d.model : ""));
        grp.addEventListener("click", function () { selectTab(id); });
      }
      root.appendChild(grp);
    });
    document.getElementById("run-graph").appendChild(root);
  })();

  // ── Agent table ───────────────────────────────────────────────────────
  (function drawTable() {
    const t = document.getElementById("agent-table");
    const head = el("tr");
    ["agent", "activations", "tool calls", "tokens", "duration", "workspace", "phase mix"].forEach(function (h) { head.appendChild(el("th", {}, h)); });
    t.appendChild(head);
    agentIds.forEach(function (id) {
      const d = nodes.find(function (n) { return n.id === id; }) || {};
      const tr = el("tr");
      tr.appendChild(el("td", {}, id));
      tr.appendChild(el("td", {"class": "num"}, (d.activations || 0) + (d.abandoned_activations ? " (+" + d.abandoned_activations + " abandoned)" : "")));
      tr.appendChild(el("td", {"class": "num"}, d.tool_calls || 0));
      tr.appendChild(el("td", {"class": "num"}, (d.tokens || {}).total_tokens || 0));
      tr.appendChild(el("td", {"class": "num"}, fmtSec(d.duration_s)));
      tr.appendChild(el("td", {}, d.changed_workspace ? (d.changed_paths || []).join(", ") || "changed" : "–"));
      tr.appendChild(el("td", {}, PHASES.filter(function (p) { return (d.phase_mix || {})[p]; })
        .map(function (p) { return p + " " + d.phase_mix[p]; }).join(", ") || "–"));
      t.appendChild(tr);
    });
  })();

  // ── Timeline: supersteps × agents ─────────────────────────────────────
  (function drawTimeline() {
    const tl = info.timeline, rows = agentIds.slice();
    const steps = tl.activations.map(function (a) { return a.superstep; });
    const minStep = steps.length ? Math.min.apply(null, steps) : 0;
    const maxStep = Math.max(minStep, info.supersteps || 0);
    const colW = 110, rowH = 46, left = 110, top = 28, bw = 84, bh = 26;
    const span = maxStep - minStep + 1;
    const startX = left - 40, endX = left + span * colW + 10;
    const width = endX + 50, height = top + rows.length * rowH + 20;
    const root = svg("svg", {width: width, height: height, viewBox: "0 0 " + width + " " + height, role: "img", "aria-label": "Timeline"});
    const defs = svg("defs");
    const hatch = svg("pattern", {id: "hatch", width: "6", height: "6", patternUnits: "userSpaceOnUse", patternTransform: "rotate(45)"});
    hatch.appendChild(svg("rect", {width: "6", height: "6", fill: "#f1f3f6"}));
    hatch.appendChild(svg("line", {x1: "0", y1: "0", x2: "0", y2: "6", stroke: "#b8c2cc", "stroke-width": "3"}));
    defs.appendChild(hatch);
    const m = svg("marker", {id: "tarrow", viewBox: "0 0 10 10", refX: "9", refY: "5", markerWidth: "6", markerHeight: "6", orient: "auto"});
    m.appendChild(svg("path", {d: "M0,0 L10,5 L0,10 z", fill: "#3e63dd"}));
    defs.appendChild(m);
    root.appendChild(defs);
    const grid = svg("g", {"class": "grid"});
    for (let s = minStep; s <= maxStep; s++) {
      const x = left + (s - minStep) * colW;
      grid.appendChild(svg("line", {x1: x, y1: top - 6, x2: x, y2: height - 10}));
      root.appendChild(svg("text", {x: x + colW / 2, y: 16, "text-anchor": "middle", fill: "#616e7c"}, "superstep " + s));
    }
    root.appendChild(grid);
    root.appendChild(svg("text", {x: startX, y: 16, "text-anchor": "middle", fill: "#616e7c"}, "START"));
    root.appendChild(svg("text", {x: endX + 20, y: 16, "text-anchor": "middle", fill: "#616e7c"}, "END"));
    rows.forEach(function (id, i) {
      root.appendChild(svg("text", {x: 8, y: top + i * rowH + bh / 2 + 4, "font-weight": "600"}, id));
    });
    // Place blocks; an abandoned attempt and its rerun share a cell, side by side.
    const pos = {}, cellCount = {};
    tl.activations.forEach(function (a) {
      const r = rows.indexOf(a.agent_id);
      if (r < 0) return;
      const key = r + ":" + a.superstep;
      const k = cellCount[key] = (cellCount[key] || 0) + 1;
      const w = bw / Math.min(k, 2);
      const x = left + (a.superstep - minStep) * colW + (colW - bw) / 2 + (k - 1) * (bw / 2);
      pos[a.uuid] = {x: x, y: top + r * rowH, w: k > 1 ? bw / 2 : bw};
      if (k > 1) {  // shrink the earlier block of this cell
        tl.activations.forEach(function (b) {
          if (b !== a && pos[b.uuid] && rows.indexOf(b.agent_id) === r && b.superstep === a.superstep) pos[b.uuid].w = bw / 2;
        });
      }
    });
    const arrows = svg("g");
    function centre(p) { return {x: p.x + p.w / 2, y: p.y + bh / 2}; }
    tl.deliveries.forEach(function (d) {
      const from = d.from_activation && pos[d.from_activation];
      const targets = (d.to_activations || []).map(function (u) { return pos[u]; }).filter(Boolean);
      function path(a, b, cls) {
        const mx = (a.x + b.x) / 2;
        const p = svg("path", {"class": "arrow" + (cls ? " " + cls : ""), d: "M" + a.x + "," + a.y + " C" + mx + "," + a.y + " " + mx + "," + b.y + " " + b.x + "," + b.y, "marker-end": "url(#tarrow)"});
        tip(p, d.from_agent + " → " + d.to_agent + " via " + d.edge_id + (d.superstep != null ? " (superstep " + d.superstep + ")" : "") + ": " + (d.preview || ""));
        arrows.appendChild(p);
      }
      if (!targets.length) {
        if (from && d.to_agent === "end") { const c = centre(from); path({x: from.x + from.w, y: c.y}, {x: endX, y: c.y}); }
        return;
      }
      targets.forEach(function (t) {
        const c = centre(t), to = {x: t.x, y: c.y};
        if (from) { const f = centre(from); path({x: from.x + from.w, y: f.y}, to); }
        else path({x: startX + 18, y: c.y}, to);
      });
    });
    tl.activations.forEach(function (a) {
      if (a.from_parent_run && pos[a.uuid]) {
        const p = pos[a.uuid], c = centre(p);
        const s = svg("path", {"class": "arrow parent", d: "M" + (startX + 18) + "," + c.y + " L" + p.x + "," + c.y, "marker-end": "url(#tarrow)"});
        tip(s, a.from_parent_run + " delivery(ies) from the parent run (" + (info.forked_from || {}).run_id + ")");
        arrows.appendChild(s);
      }
    });
    root.appendChild(arrows);
    tl.activations.forEach(function (a) {
      const p = pos[a.uuid];
      if (!p) return;
      const grp = svg("g", {"class": "block " + a.status, transform: "translate(" + p.x + "," + p.y + ")"});
      const fill = a.changed_paths.length ? "#f8ac84" : (a.tool_calls ? "#b9a2ed" : "#c4cfda");
      grp.appendChild(svg("rect", {width: p.w, height: bh, fill: fill}));
      grp.appendChild(svg("text", {x: 5, y: 17}, (p.w < bw ? "" : "t" + (a.local_turn == null ? "?" : a.local_turn) + " ") + fmtTokens(a.tokens)));
      tip(grp, a.agent_id + " turn " + a.local_turn + ", superstep " + a.superstep + " — " + a.status +
        (a.rerun_of ? " (rerun of an abandoned attempt)" : "") + "; " + a.tokens + " tokens, " + fmtSec(a.duration_s) +
        ", " + a.tool_calls + " tool calls" + (a.changed_paths.length ? "; changed " + a.changed_paths.join(", ") : "") +
        (a.stop_reason ? "; stop: " + a.stop_reason : ""));
      grp.style.cursor = "pointer";
      grp.addEventListener("click", function () { selectTab(a.agent_id); });
      root.appendChild(grp);
    });
    document.getElementById("timeline").appendChild(root);
    const legend = document.getElementById("timeline-legend");
    [["#f8ac84", "changed the workspace"], ["#b9a2ed", "called tools"], ["#c4cfda", "text only"]].forEach(function (l) {
      const s = el("span"); const i = el("i"); i.style.background = l[0]; s.appendChild(i); s.appendChild(document.createTextNode(l[1])); legend.appendChild(s);
    });
    legend.appendChild(el("span", {}, "hatched = abandoned by a pause (its rerun sits beside it) · dashed = still running · arrows = deliveries"));
  })();

  // ── Patterns ──────────────────────────────────────────────────────────
  (function drawPatterns() {
    const list = document.getElementById("patterns");
    const found = run.patterns || [];
    if (!found.length) list.appendChild(el("li", {"class": "empty"}, "No multi-agent patterns detected."));
    found.forEach(function (p) {
      const li = el("li");
      li.appendChild(el("div", {"class": "kind"}, p.kind.replace(/_/g, " ")));
      li.appendChild(el("div", {}, p.summary));
      li.appendChild(el("div", {"class": "why"}, data.descriptions[p.kind] || ""));
      list.appendChild(li);
    });
    const cfg = document.getElementById("pattern-config");
    const seen = new Set(found.map(function (p) { return p.kind; }));
    const rest = Object.keys(data.descriptions).filter(function (k) { return !seen.has(k); });
    if (rest.length) cfg.appendChild(el("span", {}, "Also checked, not found:"));
    rest.forEach(function (k) { cfg.appendChild(el("span", {}, k.replace(/_/g, " ") + ": " + data.descriptions[k])); });
  })();

  // ── Agent tabs ────────────────────────────────────────────────────────
  const tabs = document.getElementById("tabs"), frame = document.getElementById("frame");
  const dagreSource = document.getElementById("dagre-lib").textContent;
  function decode(b64) {
    const bin = atob(b64), bytes = new Uint8Array(bin.length);
    for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
    return new TextDecoder("utf-8").decode(bytes);
  }
  function selectTab(id) {
    if (!(id in pages)) return;
    Array.prototype.forEach.call(tabs.children, function (b) { b.setAttribute("aria-selected", String(b.dataset.agent === id)); });
    frame.textContent = "";
    const iframe = el("iframe", {sandbox: "allow-scripts", title: "Graph of agent " + id});
    iframe.srcdoc = decode(pages[id]).split(marker).join("<script>" + dagreSource + "<\/script>");
    frame.appendChild(iframe);
  }
  agentIds.forEach(function (id) {
    const b = el("button", {role: "tab", "aria-selected": "false"}, id + " (" + data.agents[id].nodes + " nodes)");
    b.dataset.agent = id;
    b.addEventListener("click", function () { selectTab(id); });
    tabs.appendChild(b);
  });
  if (agentIds.length) selectTab(agentIds[0]); else frame.appendChild(el("p", {"class": "empty"}, "This run has no agents."));
})();
</script>
</body>
</html>
"""
