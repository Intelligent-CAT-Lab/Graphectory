import json
import os
import sys
import re
import hashlib
import networkx as nx
from pathlib import Path
from networkx.readwrite import json_graph
from collections import defaultdict
import tempfile
import multiprocessing
from plan_monitor.commandParser import CommandParser
from plan_monitor.mapLang import get_action_role
import pygraphviz as pgv

FONT_FAMILY = os.environ.get("GRAPH_FONT", "DejaVu Sans")

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
    
TEST_COMMANDS = {"python", "python2", "python3", "pytest", "unittest", "nosetests", "tox"}

# More precise signals first (pytest structured summaries)
RE_PYTEST_FAIL = re.compile(r"\b(\d+)\s+failed\b", re.IGNORECASE)
RE_PYTEST_ERROR = re.compile(r"\b(\d+)\s+errors?\b", re.IGNORECASE)
RE_PYTEST_PASS = re.compile(r"\b(\d+)\s+passed\b", re.IGNORECASE)

# Exception patterns (precise, avoid generic words like "error")
EXCEPTION_SIGNS = [
    "Traceback (most recent call last):",
    # "AssertionError",
    # "SyntaxError",
    # "TypeError",
    # "ValueError",
    # "NameError",
    # "RuntimeError",
    # "ImportError",
    # "ModuleNotFoundError",
    # "MemoryError",
    # "Segmentation fault",
]


def check_command_outcome(command: str, observation: str, tool: str = None, subcommand: str = None, args: dict = None):
    """Check if a command outcome is successful or failure.

    Args:
        command: The command string
        observation: The observation/output from the command
        tool: Tool name (optional, for edit status checking)
        subcommand: Subcommand name (optional, for edit status checking)
        args: Command arguments (optional, for edit status checking)

    Returns:
        "success" or "failure" or None (if cannot determine)
    """
    obs = observation or ""

    # Check edit status if tool/subcommand provided
    if tool and subcommand:
        edit_status = check_edit_status(tool, subcommand, args, observation)
        if edit_status and isinstance(edit_status, str) and edit_status.startswith("failure"):
            return "failure"

    # Check for EXCEPTION_SIGNS
    for sig in EXCEPTION_SIGNS:
        if sig in obs:
            return "failure"

    # Check for structured test output patterns
    if RE_PYTEST_FAIL.search(obs) or RE_PYTEST_ERROR.search(obs):
        return "failure"
    if RE_PYTEST_PASS.search(obs):
        return "success"

    # Check for pytest failure blocks
    if "FAILURES" in obs or "ERRORS" in obs or "INTERNALERROR" in obs:
        return "failure"

    # If no failure markers found, assume success
    return "success"


# Keep check_test_status for backward compatibility
def check_test_status(command: str, observation: str):
    def is_test_command(cmd: str):
        head = cmd.strip().split()[0]
        return head in TEST_COMMANDS

    if not is_test_command(command):
        return None

    obs = observation or ""

    # --- 1) Structured Pytest test-summary (most reliable) ---
    if RE_PYTEST_FAIL.search(obs) or RE_PYTEST_ERROR.search(obs):
        return "failure"
    if RE_PYTEST_PASS.search(obs):
        # Passed does not guarantee no failures elsewhere, but if summary says passed, trust it
        return "success"

    # --- 2) General robust exception indicators ---
    for sig in EXCEPTION_SIGNS:
        if sig in obs:
            return "failure"

    # --- 3) Fallback: Detect pytest failure blocks (e.g., "=== FAILURES ===") ---
    if "FAILURES" in obs or "ERRORS" in obs or "INTERNALERROR" in obs:
        return "failure"

    # --- 4) Default assumption: no failure markers found ---
    return "success"

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
        self.created_tests = set()
        self.previous_node = None

    def add_or_update_node(self, node_label, args, flags, phase, step_idx,
                          tool=None, command=None, subcommand=None, thoughts="", observations="", outcome=None):
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
            thoughts: Thought/reasoning for this occurrence
            observations: Observation/output for this occurrence
            outcome: Outcome of the command ("success", "failure", or None)

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

            # Append thoughts and observations for this occurrence
            if "thoughts" not in self.G.nodes[node_key]:
                self.G.nodes[node_key]["thoughts"] = []
            self.G.nodes[node_key]["thoughts"].append(thoughts)

            if "observations" not in self.G.nodes[node_key]:
                self.G.nodes[node_key]["observations"] = []
            self.G.nodes[node_key]["observations"].append(observations)

            # Append outcome for this occurrence
            if "outcome" not in self.G.nodes[node_key]:
                self.G.nodes[node_key]["outcome"] = []
            self.G.nodes[node_key]["outcome"].append(outcome)
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
                thoughts=[thoughts],
                observations=[observations],
                outcome=[outcome],
                tool=tool,
                command=command,
                subcommand=subcommand
            )
            self.node_signature_to_key[node_signature] = node_key

            # Track localization nodes and update hierarchical edges incrementally
            if tool == "str_replace_editor" and subcommand == "view":
                self.localization_nodes.append(node_key)
                self._update_hierarchical_edges_for_new_node(node_key)

        return node_key

    def _update_hierarchical_edges_for_new_node(self, new_node_key):
        """Update hierarchical edges when a new localization node is added.

        This implements the same logic as build_hierarchical_edges but incrementally:
        - For path nodes: find closest parent path among existing nodes
        - For range nodes: detect nesting with existing ranges, link to path node if exists

        Args:
            new_node_key: The newly added localization node key
        """
        new_data = self.G.nodes[new_node_key]
        new_path = new_data.get("args", {}).get("path")
        new_view_range = new_data.get("args", {}).get("view_range")

        if not new_path:
            return

        new_path_obj = Path(new_path)
        new_path_str = str(new_path_obj)

        # Validate view_range if present
        if new_view_range is not None:
            if not (isinstance(new_view_range, (list, tuple)) and
                    len(new_view_range) == 2 and
                    all(isinstance(x, int) for x in new_view_range)):
                print(f"[WARN] Skipping invalid view_range for node {new_node_key}: {new_view_range}")
                return

        # Check relationship with each existing localization node
        for existing_node in self.localization_nodes[:-1]:  # Exclude new node itself
            existing_data = self.G.nodes[existing_node]
            existing_path = existing_data.get("args", {}).get("path")
            existing_view_range = existing_data.get("args", {}).get("view_range")

            if not existing_path:
                continue

            existing_path_obj = Path(existing_path)
            existing_path_str = str(existing_path_obj)

            # --- Case 1: Both are path nodes (no view_range) ---
            if new_view_range is None and existing_view_range is None:
                # Check if existing is parent of new
                if (len(existing_path_obj.parts) < len(new_path_obj.parts) and
                    new_path_obj.parts[:len(existing_path_obj.parts)] == existing_path_obj.parts):
                    self.G.add_edge(existing_node, new_node_key, type="hier")
                # Check if new is parent of existing
                elif (len(new_path_obj.parts) < len(existing_path_obj.parts) and
                      existing_path_obj.parts[:len(new_path_obj.parts)] == new_path_obj.parts):
                    self.G.add_edge(new_node_key, existing_node, type="hier")

            # --- Case 2: New is range node, existing is path node ---
            elif new_view_range is not None and existing_view_range is None:
                # If existing path matches new path, existing is parent of new range
                if existing_path_str == new_path_str:
                    self.G.add_edge(existing_node, new_node_key, type="hier")
                # If existing is ancestor directory of new
                elif (len(existing_path_obj.parts) < len(new_path_obj.parts) and
                      new_path_obj.parts[:len(existing_path_obj.parts)] == existing_path_obj.parts):
                    self.G.add_edge(existing_node, new_node_key, type="hier")

            # --- Case 3: New is path node, existing is range node ---
            elif new_view_range is None and existing_view_range is not None:
                # If new path matches existing path, new is parent of existing range
                if new_path_str == existing_path_str:
                    self.G.add_edge(new_node_key, existing_node, type="hier")
                # If new is ancestor directory of existing
                elif (len(new_path_obj.parts) < len(existing_path_obj.parts) and
                      existing_path_obj.parts[:len(new_path_obj.parts)] == new_path_obj.parts):
                    self.G.add_edge(new_node_key, existing_node, type="hier")

            # --- Case 4: Both are range nodes ---
            elif new_view_range is not None and existing_view_range is not None:
                # Only check nesting if same path
                if new_path_str == existing_path_str:
                    try:
                        n_start, n_end = new_view_range
                        e_start, e_end = existing_view_range
                        # Check if new contains existing
                        if n_start <= e_start and e_end <= n_end and (n_start < e_start or e_end < n_end):
                            self.G.add_edge(new_node_key, existing_node, type="hier")
                        # Check if existing contains new
                        elif e_start <= n_start and n_end <= e_end and (e_start < n_start or n_end < e_end):
                            self.G.add_edge(existing_node, new_node_key, type="hier")
                    except Exception as e:
                        print(f"[WARN] Failed to check range nesting: {e}")

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

    def finalize_and_save(self, output_dir, instance_id):
        """Add metadata and save graph.

        Args:
            output_dir: Base output directory (default: graphs)
            instance_id: Instance identifier

        Returns:
            str: Path to saved JSON file

        Note:
            Hierarchical edges are built incrementally in _update_hierarchical_edges_for_new_node,
            so they don't need to be built here.
        """
        self.G.graph["instance_name"] = instance_id

        # Construct output paths: output_dir/{instance_id}.json
        os.makedirs(output_dir, exist_ok=True)

        json_path = os.path.join(output_dir, f"{instance_id}.json")
        pdf_path = os.path.join(output_dir, f"{instance_id}.pdf")

        with open(json_path, "w") as f:
            json.dump(json_graph.node_link_data(self.G, link="edges"), f, indent=2)
        
        GraphVisualizer.draw_with_timeout(self.G, pdf_path, timeout_sec=30)

        return json_path

# -------------------- Build graph --------------------
def build_online_graph_from_trajectory(
    builder: GraphBuilder,
    step_idx: int,
    thought: str,
    action: str,
    observation: str,
    parser: CommandParser
):
    """Build graph online from a single trajectory step.

    This function processes one trajectory step at a time and updates the graph builder.
    Unlike the batch functions (build_graph_from_sa_trajectory, build_graph_from_oh_trajectory),
    this builds the graph incrementally as steps arrive.

    Args:
        builder: GraphBuilder instance maintaining the graph state
        step_idx: Current step index in trajectory
        thought: Thought/reasoning text from this step
        action: Action/command string from this step
        observation: Observation/output from this step
        parser: CommandParser instance for parsing action strings

    Returns:
        None (modifies builder in-place)

    Note:
        - Compound commands (connected by && || ;) are split into separate nodes
        - Thought and observation are stored as node properties
        - Each node tracks which steps it appeared in via step_indices
        - Hierarchical edges are NOT built here; call builder.finalize_and_save() when done
    """
    # Handle empty action (explicit "think" step)
    if not action or action.strip() == "":
        node_key = builder.add_or_update_node(
            node_label="think",
            args={},
            flags={},
            phase="general",
            step_idx=step_idx,
            tool=None,
            command=None,
            subcommand=None,
            thoughts=thought,
            observations=observation,
            outcome=None
        )
        builder.add_execution_edge(node_key, step_idx)
        builder.update_previous_node(node_key)
        builder.add_phase("general")
        return

    # Parse actionable commands
    parsed_commands = parser.parse(action)
    if not parsed_commands:
        return

    # Process each parsed command
    for cmd_idx, parsed in enumerate(parsed_commands):
        tool = parsed.get("tool", "").strip() if parsed.get("tool") else ""
        subcommand = parsed.get("subcommand", "").strip() if parsed.get("subcommand") else ""
        command = parsed.get("command", "").strip() if parsed.get("command") else ""
        args = parsed.get("args", {})
        flags = parsed.get("flags", {})

        # Build node label
        if tool:
            node_label = f"{tool}: {subcommand}" if subcommand else tool
        else:
            node_label = command.strip() or action.strip()

        # Get phase classification
        phase = get_action_role(
            tool, subcommand, command, args, flags,
            prev_roles=builder.prev_phases,
            created_tests=builder.created_tests
        )

        # Check edit status (keep for backward compatibility)
        edit_status = check_edit_status(tool, subcommand, args, observation)
        if edit_status and isinstance(args, dict):
            args["edit_status"] = edit_status

        # Determine thoughts and observations for this command
        # For compound commands: thoughts applies to all, observations to last
        cmd_thoughts = thought
        cmd_observations = observation if cmd_idx == len(parsed_commands) - 1 else ""

        # Check outcome using the new check_command_outcome function
        outcome = check_command_outcome(command, cmd_observations, tool, subcommand, args)

        # Add or update node
        node_key = builder.add_or_update_node(
            node_label=node_label,
            args=args,
            flags=flags,
            phase=phase,
            step_idx=step_idx,
            tool=tool,
            command=command,
            subcommand=subcommand,
            thoughts=cmd_thoughts,
            observations=cmd_observations,
            outcome=outcome
        )

        builder.add_execution_edge(node_key, step_idx)
        builder.update_previous_node(node_key)
        builder.add_phase(phase)

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

        # Map phase prefixes to full names: L_* -> localization, P -> patch, V_* -> validation
        def normalize_phase(phase):
            if phase.startswith("L_"):
                return "localization"
            elif phase.startswith("V_"):
                return "validation"
            elif phase == "P" or phase.startswith("P_"):
                return "patch"
            else:
                return "general"

        normalized = [normalize_phase(p) for p in phases]

        uniq = []
        seen = set()
        # Stable ordering for stripes
        order = ["localization", "patch", "validation", "general"]
        for ph in order:
            if ph in normalized and ph not in seen:
                seen.add(ph)
                uniq.append(ph)

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
        try:
            self._draw_graph_graphviz_with_compact_legend(G, pdf_path)
            return
        except OSError as e:
            print("[WARN] Graphviz failed, install first:", e)
    
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