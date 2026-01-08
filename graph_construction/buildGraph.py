import json
import os
import sys
import hashlib
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from pathlib import Path
from networkx.readwrite import json_graph
from graph_construction.commandParser import CommandParser
from datasets import load_dataset
from collections import defaultdict
import getpass
import tempfile
import multiprocessing
from graph_construction.mapPhase import get_phase
import pygraphviz as pgv
_HAS_PYGRAPHVIZ = True

FONT_FAMILY = os.environ.get("GRAPH_FONT", "DejaVu Sans")

# -------------------- Data lookups --------------------
swe_bench_ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
difficulty_lookup = {row["instance_id"]: row["difficulty"] for row in swe_bench_ds}

# -------------------- Helpers --------------------
def hash_node_signature(label, args, flags):
    normalized = json.dumps({"label": label, "args": args, "flags": flags}, sort_keys=True)
    return hashlib.md5(normalized.encode("utf-8")).hexdigest()

def check_edit_status(tool, subcommand, args, observation):
    def check_str_edit_status(obs):
        if not obs:
            return None
        if "has been edited." in obs:
            return "success"
        if "did not appear verbatim" in obs:
            return "failure: not found"
        if "Multiple occurrences of old_str" in obs:
            return "failure: multiple occurrences"
        if "old_str" in obs and "is the same as new_str" in obs:
            return "failure: no change"
        return "failure: unknown"

    if tool == "str_replace_editor" and subcommand in {"str_replace"}:
        return check_str_edit_status(observation)
    return None

def determine_resolution_status(instance_id: str, eval_report_path: str) -> str:
    """Determine resolution status from eval report given an instance ID."""
    if not os.path.isfile(eval_report_path):
        return "N/A"
    with open(eval_report_path, 'r') as f:
        report = json.load(f)
    if instance_id in report.get("resolved_ids", []):
        return "resolved"
    elif instance_id in report.get("unresolved_ids", []):
        return "unresolved"
    return "unsubmitted"

# -------------------- Graph Builder Class --------------------
class GraphBuilder:
    """Utility class for managing graph construction operations.

    This class encapsulates all shared graph construction logic for building
    trajectory graphs from agent execution traces.
    """

    def __init__(self):
        self.G = nx.MultiDiGraph()
        self.node_signature_to_key = {}
        self.localization_nodes = []
        self.prev_phases = set()
        self.previous_node = None

    def add_or_update_node(self, node_label, args, flags, phase, step_idx,
                          tool=None, command=None, subcommand=None):
        """Add a new node or update existing node with a new occurrence.

        Args:
            node_label: Display label for the node
            args: Command arguments dictionary
            flags: Command flags dictionary
            phase: Phase classification (localization/patch/validation/general)
            step_idx: Step index in trajectory
            tool: Tool name (if applicable)
            command: Command name (if applicable)
            subcommand: Subcommand name (if applicable)

        Returns:
            node_key: The key of the added or updated node
        """
        node_signature = hash_node_signature(node_label, args, flags)

        if node_signature in self.node_signature_to_key:
            # Update existing node
            node_key = self.node_signature_to_key[node_signature]
            self.G.nodes[node_key]["step_indices"].append(step_idx)
            if "phases" not in self.G.nodes[node_key]:
                self.G.nodes[node_key]["phases"] = []
            self.G.nodes[node_key]["phases"].append(phase)
        else:
            # Add new node
            node_key = f"{len(self.G.nodes)}:{node_label}"
            self.G.add_node(
                node_key,
                label=node_label,
                args=args,
                flags=flags,
                phases=[phase],
                step_indices=[step_idx],
                tool=tool,
                command=command,
                subcommand=subcommand
            )
            self.node_signature_to_key[node_signature] = node_key

            # Track localization nodes
            if tool == "str_replace_editor" and subcommand == "view":
                self.localization_nodes.append(node_key)

        return node_key

    def add_execution_edge(self, node_key, step_idx):
        """Add execution edge from previous node to current node.

        Args:
            node_key: Target node key
            step_idx: Step index for edge label
        """
        if self.previous_node:
            self.G.add_edge(self.previous_node, node_key, label=str(step_idx), type="exec")

    def update_previous_node(self, node_key):
        """Update the previous node pointer.

        Args:
            node_key: Node to set as previous
        """
        self.previous_node = node_key

    def add_phase(self, phase):
        """Add phase to the set of previous phases.

        Args:
            phase: Phase to add
        """
        self.prev_phases.add(phase)

    def finalize_and_save(self, output_dir, instance_id, eval_report_path):
        """Build hierarchical edges, add metadata, and save graph.

        Args:
            output_dir: Base output directory
            instance_id: Instance identifier
            eval_report_path: Path to evaluation report

        Returns:
            tuple: (json_path, pdf_path) paths to saved files
        """
        build_hierarchical_edges(self.G, self.localization_nodes)

        resolution_status = determine_resolution_status(instance_id, eval_report_path)
        self.G.graph["resolution_status"] = resolution_status
        self.G.graph["instance_name"] = instance_id
        self.G.graph["debug_difficulty"] = difficulty_lookup.get(instance_id, "unknown")

        # Construct output paths: output_dir/{instance_id}/{instance_id}.{json,pdf}
        instance_dir = os.path.join(output_dir, instance_id)
        os.makedirs(instance_dir, exist_ok=True)

        json_path = os.path.join(instance_dir, f"{instance_id}.json")
        pdf_path = os.path.join(instance_dir, f"{instance_id}.pdf")

        with open(json_path, "w") as f:
            json.dump(json_graph.node_link_data(self.G, edges="edges"), f, indent=2)

        GraphVisualizer.draw_with_timeout(self.G, pdf_path, timeout_sec=60)

        return json_path, pdf_path

# -------------------- Build graph --------------------
def build_graph_from_sa_trajectory(traj_data, parser: CommandParser, instance_id, output_dir, eval_report_path):
    """Build graph from SWE-agent trajectory data.

    Args:
        traj_data: SWE-agent trajectory dictionary containing 'trajectory' key
        parser: CommandParser instance for parsing action strings
        instance_id: Instance identifier (e.g., 'django__django-12345')
        output_dir: Base output directory for saving graphs
        eval_report_path: Path to evaluation report JSON file

    Returns:
        tuple: (json_path, pdf_path) paths to the saved graph files

    Output Structure:
        {output_dir}/{instance_id}/{instance_id}.json
        {output_dir}/{instance_id}/{instance_id}.pdf
    """
    builder = GraphBuilder()
    trajectory = traj_data.get("trajectory", [])

    for step_idx, step in enumerate(trajectory):
        action_str = step.get("action", "")

        # Handle explicit "think" steps (blank action)
        if action_str.strip() == "":
            node_key = builder.add_or_update_node(
                node_label="think",
                args={},
                flags={},
                phase="general",
                step_idx=step_idx,
                tool=None,
                command=None,
                subcommand=None
            )
            builder.add_execution_edge(node_key, step_idx)
            builder.update_previous_node(node_key)
            builder.add_phase("general")
            continue

        # Parse actionable commands
        parsed_commands = parser.parse(action_str)
        if not parsed_commands:
            continue

        for parsed in parsed_commands:
            tool = parsed.get("tool", "").strip() if parsed.get("tool") else ""
            subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
            command = parsed.get("command", "").strip() if parsed.get("command") else ""
            args = parsed.get("args", {})
            flags = parsed.get("flags", {})

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, builder.prev_phases)

            edit_status = check_edit_status(tool, subcommand, args, step.get("observation", ""))
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_key = builder.add_or_update_node(
                node_label=node_label,
                args=args,
                flags=flags,
                phase=phase,
                step_idx=step_idx,
                tool=tool,
                command=command,
                subcommand=subcommand
            )
            builder.add_execution_edge(node_key, step_idx)
            builder.update_previous_node(node_key)
            builder.add_phase(phase)

    return builder.finalize_and_save(output_dir, instance_id, eval_report_path)

def build_graph_from_oh_trajectory(traj_data, parser: CommandParser, instance_id, output_dir, eval_report_path):
    """Build graph from OpenHands trajectory data.

    Args:
        traj_data: OpenHands trajectory dictionary containing 'history' key
        parser: CommandParser instance for parsing action strings
        instance_id: Instance identifier (e.g., 'django__django-12345')
        output_dir: Base output directory for saving graphs
        eval_report_path: Path to evaluation report JSON file

    Returns:
        tuple: (json_path, pdf_path) paths to the saved graph files

    Output Structure:
        {output_dir}/{instance_id}/{instance_id}.json
        {output_dir}/{instance_id}/{instance_id}.pdf
    """
    builder = GraphBuilder()
    step_idx = 0

    for step in traj_data.get("history", []):
        action = step.get("observation") if step.get("observation") else None
        if action in ("system", "message") or action is None:
            continue

        # Use action text only as a fallback when command string is empty
        action_str = action or ""

        tool_calls = step.get("tool_call_metadata", {}).get("model_response", {}).get("choices", [])
        if not tool_calls and "tool_call_metadata" in step:
            tool_calls = [step["tool_call_metadata"]]

        parsed_commands = []
        for call in tool_calls:
            function_call = None
            if isinstance(call, dict):
                if "function" in call:
                    function_call = call["function"]
                elif "message" in call and "tool_calls" in call["message"]:
                    for tc in call["message"]["tool_calls"]:
                        if "function" in tc:
                            function_call = tc["function"]

            if not function_call:
                continue

            tool_name = function_call.get("name")
            args_raw = function_call.get("arguments", "{}")

            try:
                args_loaded = json.loads(args_raw)
            except json.JSONDecodeError:
                args_loaded = {}

            if tool_name == "execute_bash":
                cmd = args_loaded.get("command", "").strip()
                parsed_commands = parser.parse(cmd)
                if not parsed_commands:
                    continue
            else:
                subcommand = args_loaded.pop("command", None)  # remove 'command' key from args
                parsed_commands = [{
                    "tool": tool_name,
                    "subcommand": subcommand,
                    "args": args_loaded,
                }]

        if not parsed_commands:
            continue

        for parsed in parsed_commands:
            tool = parsed.get("tool", "").strip()
            # ---- THINK NODES ----
            if tool == "think":
                node_key = builder.add_or_update_node(
                    node_label="think",
                    args={},
                    flags={},
                    phase="general",
                    step_idx=step_idx,
                    tool=None,
                    command=None,
                    subcommand=None
                )
                builder.add_execution_edge(node_key, step_idx)
                builder.update_previous_node(node_key)
                builder.add_phase("general")
                continue

            subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
            command = parsed.get("command", "").strip() if parsed.get("command") else ""
            args = parsed.get("args", {})
            flags = parsed.get("flags", {})

            if tool:
                node_label = f"{tool}: {subcommand}" if subcommand else tool
            else:
                node_label = command.strip() or action_str.strip()

            phase = get_phase(tool, subcommand, command, args, builder.prev_phases)

            edit_status = check_edit_status(tool, subcommand, args, step.get("content", ""))
            if edit_status and isinstance(args, dict):
                args["edit_status"] = edit_status

            node_key = builder.add_or_update_node(
                node_label=node_label,
                args=args,
                flags=flags,
                phase=phase,
                step_idx=step_idx,
                tool=tool,
                command=command,
                subcommand=subcommand
            )
            builder.add_execution_edge(node_key, step_idx)
            builder.update_previous_node(node_key)
            builder.add_phase(phase)

        step_idx += 1

    return builder.finalize_and_save(output_dir, instance_id, eval_report_path)

def build_hierarchical_edges(G: nx.MultiDiGraph, localization_nodes):
    path_nodes = []  # [(node_id, Path)]
    range_nodes_by_path = defaultdict(list)  # path_str -> [(node_id, [start, end])]

    for node in localization_nodes:
        data = G.nodes[node]
        path = data.get("args", {}).get("path")
        view_range = data.get("args", {}).get("view_range")

        if path:
            path_obj = Path(path)
            if view_range is None:
                path_nodes.append((node, path_obj))
            elif (
                isinstance(view_range, (list, tuple)) and
                len(view_range) == 2 and
                all(isinstance(x, int) for x in view_range)
            ):
                range_nodes_by_path[str(path_obj)].append((node, view_range))
            else:
                print(f"[WARN] Skipping invalid view_range for node {node}: {view_range}")

    # --- 1) Path hierarchy by folder containment ---
    for child_node, child_path in path_nodes:
        best_parent_node = None
        best_parent_path = None
        for parent_node, parent_path in path_nodes:
            if parent_node == child_node:
                continue
            if (len(parent_path.parts) < len(child_path.parts) and
                child_path.parts[:len(parent_path.parts)] == parent_path.parts):
                if best_parent_path is None or len(parent_path.parts) > len(best_parent_path.parts):
                    best_parent_node = parent_node
                    best_parent_path = parent_path
        if best_parent_node:
            G.add_edge(best_parent_node, child_node, type="hier")

    # --- 2) Range nodes: handle nesting + link outermost ---
    path_to_node = {str(p): n for n, p in path_nodes}

    for path_str, range_nodes in range_nodes_by_path.items():
        is_nested = {n: False for n, _ in range_nodes}

        # detect nesting: mark inner ranges
        for i, (node_i, r_i) in enumerate(range_nodes):
            for j, (node_j, r_j) in enumerate(range_nodes):
                if i == j:
                    continue
                try:
                    a1, a2 = r_i
                    b1, b2 = r_j
                    if b1 >= a1 and b2 <= a2:
                        G.add_edge(node_i, node_j, type="hier")
                        is_nested[node_j] = True
                except Exception as e:
                    print(f"[WARN] Failed to unpack ranges for nesting check: {r_i}, {r_j} ({e})")

        # link outermost ranges to:
        #   - exact path node if exists
        #   - else closest parent path node whose path contains this path
        path_node = path_to_node.get(path_str)
        if path_node:
            for node, _ in range_nodes:
                if not is_nested[node]:
                    G.add_edge(path_node, node, type="hier")
        else:
            # No exact path node → find nearest ancestor
            path_parts = Path(path_str).parts
            best_ancestor_node = None
            best_ancestor_depth = -1
            for pn, pp in path_nodes:
                if len(pp.parts) < len(path_parts) and path_parts[:len(pp.parts)] == pp.parts:
                    if len(pp.parts) > best_ancestor_depth:
                        best_ancestor_node = pn
                        best_ancestor_depth = len(pp.parts)
            for node, _ in range_nodes:
                if not is_nested[node] and best_ancestor_node:
                    G.add_edge(best_ancestor_node, node, type="hier")

# ==================== Visualization Class ====================
class GraphVisualizer:
    """Encapsulates all plot-related helpers and renderers."""

    phase_colors = {
        "localization": "#C5B3F0",  # light purple
        "patch":        "#FCC9B0",  # light coral
        "validation": "#A8E6F0",  # light cyan
        "general":      "#CFE0F6",  # light sky
    }

    def __init__(self):
        # Built at draw time: maps each unique string to a stable ID "str_#"
        self._str_id_map = {}
    
    def _node_phase_colors(self, node_data):
        """Return an ordered list of color hexes for this node based on its phases list."""
        phases = node_data.get("phases") or ["general"]
        uniq = []
        seen = set()
        # stable, human-friendly ordering for stripes
        order = ["localization", "patch", "validation", "general"]
        for ph in order:
            if ph in phases and ph not in seen:
                seen.add(ph); uniq.append(ph)
        # append any remaining unknowns in their first-seen order
        for ph in phases:
            if ph not in seen:
                seen.add(ph); uniq.append(ph)
        return [self.phase_colors.get(ph, self.phase_colors["general"]) for ph in uniq]

    def _draw_node_with_stripes(self, ax, x, y, label, colors, font_size=25):
        """
        Matplotlib: draw a rounded box at (x,y) with vertical color stripes behind text.
        Keep existing styling (rounded, black border). 'colors' is a list of hexes.
        """
        # Measure text size by creating a temporary, invisible text object
        t = ax.text(x, y, label, fontsize=font_size, fontweight='bold',
                    ha="center", va="center", alpha=0.0)
        fig = ax.figure
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        bbox = t.get_window_extent(renderer=renderer).transformed(ax.transData.inverted())
        t.remove()

        pad_x, pad_y = 0.35, 0.28  # similar to previous bbox padding
        width = bbox.width * 1.0 + pad_x
        height = bbox.height * 1.0 + pad_y
        left = x - width / 2.0
        bottom = y - height / 2.0

        # Stripes (equal widths)
        n = max(1, len(colors))
        for i, c in enumerate(colors):
            w_i = width / n
            ax.add_patch(
                FancyBboxPatch(
                    (left + i * w_i, bottom),
                    w_i, height,
                    boxstyle="round,pad=0.0,rounding_size=0.2",
                    linewidth=0.0,  # no inner borders between stripes
                    facecolor=c,
                    edgecolor="none",
                    zorder=0.5,
                )
            )

        # Border on top
        ax.add_patch(
            FancyBboxPatch(
                (left, bottom),
                width, height,
                boxstyle="round,pad=0.0,rounding_size=0.2",
                linewidth=1.2,
                facecolor="none",
                edgecolor="black",
                zorder=0.8,
            )
        )

        # Foreground text
        ax.text(x, y, label, fontsize=font_size, fontweight='bold',
                ha="center", va="center", color="black", zorder=1.0)

    def draw_graph_pdf(self, G: nx.MultiDiGraph, pdf_path: str):
        # Build the mapping once per graph (JSON graph remains unchanged)
        self._str_id_map = self._build_str_id_map(G)

        if _HAS_PYGRAPHVIZ:
            try:
                self._draw_graph_graphviz_with_compact_legend(G, pdf_path)
                return
            except OSError as e:
                print("[WARN] Graphviz failed, falling back to Matplotlib:", e)
        self._draw_graph_matplotlib_with_compact_legend(G, pdf_path)
    
    # ----- TIMEOUT WRAPPER -----
    @staticmethod
    def _pdf_worker(G: nx.MultiDiGraph, pdf_path: str):
        gv = GraphVisualizer()
        gv.draw_graph_pdf(G, pdf_path)

    @classmethod
    def draw_with_timeout(cls, G: nx.MultiDiGraph, pdf_path: str, timeout_sec: int = 100) -> bool:
        """
        Try to render PDF via GraphVisualizer; if it takes longer than timeout_sec
        (default 5 min) or fails, terminate and fall back to the simple graph drawer.
        Returns True if PDF succeeded; False if fell back to simple graph.
        """
        p = multiprocessing.Process(target=cls._pdf_worker, args=(G, pdf_path))
        p.start()
        p.join(timeout_sec)

        if p.exitcode is None:
            # Timed out: terminate and fall back.
            try:
                p.terminate()
                p.join(5)
                if p.is_alive():
                    try:
                        p.kill()
                    except Exception:
                        pass
            finally:
                pass
            print(f"[WARN] GraphVisualizer exceeded {timeout_sec}s. Too large to display.")
            return False

        if p.exitcode != 0:
            # Crashed: fall back.
            print(f"[WARN] GraphVisualizer failed (exit {p.exitcode}). Too large to display.")
            return False

        return True

    # ---- Mapping helpers for str_replace display ----
    def _build_str_id_map(self, G: nx.MultiDiGraph) -> dict:
        """
        Deduplicate all strings seen in str_replace actions (both old_str and new_str)
        and assign stable IDs: str_1, str_2, ...
        """
        mapping = {}
        next_id = 1
        for _, d in G.nodes(data=True):
            if d.get("subcommand") == "str_replace" and isinstance(d.get("args"), dict):
                for key in ("old_str", "new_str"):
                    s = d["args"].get(key)
                    if isinstance(s, str) and s not in mapping:
                        mapping[s] = f"str_{next_id}"
                        next_id += 1
        return mapping

    def _str_ids_for_node(self, node_data):
        """Return 'str_i, str_j' for str_replace nodes, else ''."""
        if node_data.get("subcommand") != "str_replace":
            return ""
        args = node_data.get("args", {})
        if not isinstance(args, dict):
            return ""
        old_s = args.get("old_str")
        new_s = args.get("new_str")
        if not isinstance(old_s, str) or not isinstance(new_s, str):
            return ""
        old_id = self._str_id_map.get(old_s)
        new_id = self._str_id_map.get(new_s)
        if not old_id or not new_id:
            return ""
        return f"{old_id}, {new_id}"

    # ---- Label helpers ----
    @staticmethod
    def _shorten_path(p: str, maxlen: int = 18) -> str:
        p = (p or "").replace("\\", "/")
        if len(p) <= maxlen:
            return p
        parts = [x for x in p.split("/") if x]
        base = parts[-1] if parts else p
        return f".../{base}"

    @staticmethod
    def _first_script_arg(args_list):
        for tok in args_list:
            if not isinstance(tok, str):
                continue
            if tok.startswith("-"):
                continue
            if "/" in tok or tok.endswith(".py"):
                return tok
        for tok in args_list:
            if isinstance(tok, str) and not tok.startswith("-"):
                return tok
        return None

    @staticmethod
    def _escape_html(s: str) -> str:
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    @staticmethod
    def _format_view_range(args) -> str:
        """Return a pretty 'Lstart–end' if args has a valid view_range."""
        if isinstance(args, dict) and isinstance(args.get("view_range"), (list, tuple)) and len(args["view_range"]) == 2:
            a, b = args["view_range"]
            if isinstance(a, int) and isinstance(b, int):
                return f"L{a}–{b}"
        return ""

    def _make_display_label_plain(self, node_data):
        """Text label for Matplotlib fallback (includes view_range and str_#,# for str_replace)."""
        base = (node_data.get("command") or node_data.get("subcommand") or node_data.get("label") or "").strip()
        tool = (node_data.get("tool") or "").strip()
        if base == tool:
            base = ""
        args = node_data.get("args", {})
        cmd = (node_data.get("command") or "").lower()
        path_lc = ""
        if isinstance(args, dict):
            p = args.get("path")
            path_lc = self._shorten_path(str(p).lower()) if p else ""
        elif isinstance(args, (list, tuple)) and cmd in {"python", "python3"}:
            cand = self._first_script_arg(args)
            path_lc = self._shorten_path(cand.lower()) if cand else ""

        vr = self._format_view_range(args)

        # Status badge
        badge = ""
        if isinstance(args, dict):
            status = args.get("edit_status")
            if status == "success":
                badge = " ✓"
            elif status and str(status).startswith("failure"):
                badge = " ✗"

        # 'str_i, str_j' for str_replace nodes
        str_pair = self._str_ids_for_node(node_data)

        lines = []
        if base or badge:
            lines.append((base or node_data.get("label", "")).strip() + (badge or ""))
        # --- CHANGED ORDER: path first, then str_pair ---
        if path_lc:
            lines.append(path_lc)
        if str_pair:
            lines.append(str_pair)
        if vr:
            lines.append(vr)

        text = "\n".join([l for l in lines if l]).strip()
        return text if text else (node_data.get("label", "") or "")

    def _make_display_label_html(self, node_data):
        """
        HTML-like label for Graphviz: first line command (+badge),
        then path, then 'str_i, str_j' for str_replace, then view_range.
        """
        base = (node_data.get("command") or node_data.get("subcommand") or node_data.get("label") or "").strip()
        tool = (node_data.get("tool") or "").strip()
        if base == tool:
            base = ""
        args = node_data.get("args", {})
        cmd = (node_data.get("command") or "").lower()
        path_lc = ""
        if isinstance(args, dict):
            p = args.get("path")
            path_lc = self._shorten_path(str(p).lower()) if p else ""
        elif isinstance(args, (list, tuple)) and cmd in {"python", "python3"}:
            cand = self._first_script_arg(args)
            path_lc = self._shorten_path(cand.lower()) if cand else ""

        vr = self._format_view_range(args)

        # Badge
        badge = ""
        if isinstance(args, dict):
            status = args.get("edit_status")
            if status == "success":
                badge = " ✓"
            elif status and str(status).startswith("failure"):
                badge = " ✗"

        # 'str_i, str_j' for str_replace nodes
        str_pair = self._str_ids_for_node(node_data)

        lines = []
        if base or badge:
            lines.append(f"<B>{self._escape_html((base or node_data.get('label','')) + (f' {badge}' if badge else ''))}</B>")
        # --- CHANGED ORDER: path first, then str_pair ---
        if path_lc:
            lines.append(self._escape_html(path_lc))
        if str_pair:
            lines.append(self._escape_html(str_pair))
        if vr:
            lines.append(self._escape_html(vr))
        if not lines:
            lines.append(self._escape_html(node_data.get("label", "")))

        inner = "<BR/>".join(lines)
        html = f'<FONT FACE="{FONT_FAMILY}" POINT-SIZE="20">{inner}</FONT>'
        return f"<{html}>"


    # ---- Graphviz path (main graph) + COMPACT LEGEND placed INSIDE near center ----
    def _draw_graph_graphviz_with_compact_legend(self, G: nx.MultiDiGraph, pdf_path: str):
        A = pgv.AGraph(directed=True, strict=False)
        A.graph_attr.update(
            rankdir="LR",
            overlap="false",
            splines="true",
            nodesep="0.9",
            ranksep="1.15",
            margin="0.15",
            ratio="compress",
            newrank="true",
            fontname=FONT_FAMILY
        )
        A.node_attr.update(
            shape="box",
            style="rounded,filled",
            fontsize="25",
            color="black",
            penwidth="1.0",
            fontname=FONT_FAMILY
        )
        A.edge_attr.update(
            fontsize="20",
            arrowsize="1.3",
            arrowhead="normal",
            color="#808080",
            fontname=FONT_FAMILY
        )

        # Nodes
        for n, d in G.nodes(data=True):
            label = self._make_display_label_html(d)
            colors = self._node_phase_colors(d)  # list of hex colors
            if len(colors) <= 1:
                fill = colors[0] if colors else self.phase_colors["general"]
                A.add_node(n, label=label, fillcolor=fill, style="rounded,filled")
            else:
                # striped fill with equal slices
                fill = ":".join(colors)
                A.add_node(n, label=label, fillcolor=fill, style="rounded,striped")


        # Edges with staggered labels
        grouped = defaultdict(list)
        for u, v, k, d in G.edges(keys=True, data=True):
            grouped[(u, v)].append((k, d))

        for (u, v), lst in grouped.items():
            for idx, (k, d) in enumerate(lst):
                etype = d.get("type", "exec")
                atr = {"style": "solid", "color": "#808080", "minlen": "1"}
                if etype == "hier":
                    atr["style"] = "dashed"
                    atr["color"] = "#2E8B57"
                if etype == "exec" and "label" in d:
                    atr["label"] = str(d["label"])
                    atr["labelfontsize"] = "20"
                    atr["labeldistance"] = str(1.0 + 0.4 * idx)
                    sign = 1 if (str(u) < str(v)) else -1
                    atr["labelangle"] = str(sign * (20 + 12 * (idx % 3)))
                A.add_edge(u, v, **atr)

        # Compact legend
        # row1, row2 = phases[:3], phases[3:]
        phases = ["localization", "patch", "validation", "general"]

        def _legend_row(items):
            cells = []
            for ph in items:
                color = self.phase_colors[ph]
                swatch = (
                    "<TABLE BORDER='0' CELLBORDER='1' COLOR='#C8C8C8' CELLPADDING='0' CELLSPACING='0'>"
                    f"<TR><TD BGCOLOR='{color}' WIDTH='24' HEIGHT='12' FIXEDSIZE='TRUE'></TD></TR>"
                    "</TABLE>"
                )
                cells.append(f"<TD>{swatch}</TD>")
                cells.append(f"<TD ALIGN='LEFT'><FONT FACE='{FONT_FAMILY}' POINT-SIZE='18' COLOR='#333333'>{ph}</FONT></TD>")
                cells.append("<TD WIDTH='10'></TD>")
            return "<TR>" + "".join(cells) + "</TR>"

        legend_label = (
            "<"
            "<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='6'>"
            f"{_legend_row(phases)}"
            # f"{_legend_row(row1)}"
            # f"{_legend_row(row2)}"
            "</TABLE>"
            ">"
        )
        A.graph_attr.update(labelloc="b", labeljust="l", label=legend_label)
        A.draw(pdf_path, prog="dot")

    # ---- Matplotlib fallback ----
    def _draw_graph_matplotlib_with_compact_legend(self, G: nx.MultiDiGraph, pdf_path: str):
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['font.family'] = FONT_FAMILY

        fig_w = max(16, min(36, 1.0 + 0.8 * G.number_of_nodes()))
        fig_h = max(11, 13)
        fig = plt.figure(figsize=(fig_w, fig_h))

        ax = fig.add_axes([0.05, 0.16, 0.90, 0.78])

        try:
            from networkx.drawing.nx_agraph import graphviz_layout
            pos = graphviz_layout(
                G, prog="dot",
                args="-Grankdir=LR -Goverlap=false -Gsplines=true -Granksep=1.3 -Gnodesep=0.9 -Gmargin=0.2"
            )
        except Exception:
            pos = nx.spring_layout(G, seed=42, k=2.5 / max(1, G.number_of_nodes()), iterations=300)

        labels = {n: self._make_display_label_plain(d) for n, d in G.nodes(data=True)}

        # Nodes
        for n, (x, y) in pos.items():
            label = labels[n]
            colors = self._node_phase_colors(G.nodes[n])
            self._draw_node_with_stripes(ax, x, y, label, colors, font_size=25)

        # Edges
        exec_edges, hier_edges = [], []
        for u, v, k, d in G.edges(keys=True, data=True):
            (hier_edges if d.get("type") == "hier" else exec_edges).append((u, v, k, d))

        def draw_edges_group(edges, solid=True, color="gray", base=0.22, step=0.15):
            group = defaultdict(list)
            for u, v, k, d in edges:
                group[(u, v)].append((k, d))
            for (u, v), lst in group.items():
                for i, (k, d) in enumerate(lst):
                    sign = 1 if str(u) < str(v) else -1
                    rad = (base + step * i) * sign
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=[(u, v, k)],
                        connectionstyle=f"arc3,rad={rad}",
                        style="solid" if solid else "dashed",
                        edge_color=color,
                        width=1.8,
                        arrows=True,
                        arrowstyle="-|>",
                        arrowsize=22,
                        min_source_margin=18,
                        min_target_margin=18,
                        alpha=1.0,
                        ax=ax
                    )
                    if d.get("type") == "exec" and "label" in d:
                        x1, y1 = pos[u]; x2, y2 = pos[v]
                        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        cx = mx - rad * (y2 - y1)
                        cy = my + rad * (x2 - x1)
                        t = 0.5
                        bx = (1 - t) ** 2 * x1 + 2 * (1 - t) * t * cx + t ** 2 * x2
                        by = (1 - t) ** 2 * y1 + 2 * (1 - t) * t * cy + t ** 2 * y2
                        ax.text(bx, by, str(d["label"]), fontsize=25, color="black",
                                ha="center", va="center", zorder=5)

        draw_edges_group(exec_edges, solid=True, color="gray")
        draw_edges_group(hier_edges, solid=False, color="#2E8B57")

        # Legend
        legend_ax = fig.add_axes([0.05, 0.05, 0.90, 0.07])
        legend_ax.axis("off")
        phases_order = ["localization", "patch", "validation", "general"]
        x = 0.01
        y = 0.55
        rect_w = 0.016
        rect_h = 0.20
        gap_after_rect = 0.045
        font_size = 25
        fig.canvas.draw()
        for ph in phases_order:
            color = self.phase_colors[ph]
            t = legend_ax.text(x, y, f"{ph}:", ha="left", va="center",
                               fontsize=font_size, fontweight="bold",
                               transform=legend_ax.transAxes)
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            bbox_disp = t.get_window_extent(renderer=renderer)
            bbox_axes = bbox_disp.transformed(legend_ax.transAxes.inverted())
            text_w = bbox_axes.width
            rx = x + text_w + 0.01
            ry = y - rect_h/2
            legend_ax.add_patch(Rectangle((rx, ry), rect_w, rect_h,
                                          transform=legend_ax.transAxes,
                                          facecolor=color, edgecolor="black", linewidth=1.0))
            x = rx + rect_w + gap_after_rect

        ax.axis("off")
        plt.savefig(pdf_path, dpi=300, bbox_inches="tight")
        plt.close(fig)