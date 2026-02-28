"""
Graph Visualization Module

Generates HTML visualizations of trajectory graphs using external templates.
"""

import json
import os
import multiprocessing
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any


FONT_FAMILY = os.environ.get("GRAPH_FONT", "DejaVu Sans, Arial, sans-serif")


class GraphVisualizer:
    """Encapsulates all HTML graph rendering with horizontal flow and edge visualization."""

    phase_colors = {
        "localization": "#C5B3F0",  # light purple
        "patch":        "#FCC9B0",  # light coral
        "validation":   "#A8E6F0",  # light cyan
        "general":      "#CFE0F6",  # light sky
    }

    def __init__(self, template_dir: Path = None):
        """Initialize visualizer with template directory.
        
        Args:
            template_dir: Path to directory containing HTML/CSS/JS templates.
                         Defaults to same directory as this file.
        """
        if template_dir is None:
            template_dir = Path(__file__).parent
        
        self.template_dir = Path(template_dir)
        self._str_id_map = {}
        
        # Load templates
        self.html_template = self._load_template("graph_template.html")
        self.css_template = self._load_template("styles.css")
        self.js_template = self._load_template("graph_renderer.js")
    
    def _load_template(self, filename: str) -> str:
        """Load template file from template directory."""
        template_path = self.template_dir / filename
        if not template_path.exists():
            raise FileNotFoundError(f"Template not found: {template_path}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _node_phase_colors(self, node_data: Dict[str, Any]) -> List[str]:
        """Return an ordered list of color hexes for this node based on its phases list."""
        phases = node_data.get("phases") or ["general"]
        uniq = []
        seen = set()
        order = ["localization", "patch", "validation", "general"]
        for ph in order:
            if ph in phases and ph not in seen:
                seen.add(ph)
                uniq.append(ph)
        for ph in phases:
            if ph not in seen:
                seen.add(ph)
                uniq.append(ph)
        return [self.phase_colors.get(ph, self.phase_colors["general"]) for ph in uniq]

    @staticmethod
    def _html_worker(G: nx.MultiDiGraph, html_path: str, template_dir: Path, metadata_comment: str):
        """Worker process for HTML generation."""
        gv = GraphVisualizer(template_dir=template_dir)
        gv.draw_graph_html(G, html_path, metadata_comment)

    @classmethod
    def draw_with_timeout(cls, G: nx.MultiDiGraph, html_path: str, 
                         timeout_sec: int = 100, template_dir: Path = None,
                         metadata_comment: str = "") -> bool:
        """Try to render HTML; return False if timeout/failure."""
        if template_dir is None:
            template_dir = Path(__file__).parent
        
        p = multiprocessing.Process(
            target=cls._html_worker, 
            args=(G, html_path, template_dir, metadata_comment)
        )
        p.start()
        p.join(timeout_sec)

        if p.exitcode is None:
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
            print(f"[WARN] GraphVisualizer failed (exit {p.exitcode}). Too large to display.")
            return False

        return True

    def _build_str_id_map(self, G: nx.MultiDiGraph) -> Dict[str, str]:
        """Deduplicate strings for str_replace nodes."""
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

    @staticmethod
    def _escape_html(s: str) -> str:
        """Escape HTML special characters."""
        return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")

    def _make_simple_label(self, node_data: Dict[str, Any]) -> str:
        """Make a multi-line label with key details for node display."""
        lines = []
        
        # Main command/label
        base = (node_data.get("command") or node_data.get("subcommand") or 
                node_data.get("label") or "").strip()
        
        # Add status badge
        args = node_data.get("args", {})
        if isinstance(args, dict):
            status = args.get("edit_status")
            if status == "success":
                base += " ✓"
            elif status and str(status).startswith("failure"):
                base += " ✗"
        
        if base:
            lines.append(base)
        else:
            lines.append(node_data.get("label", ""))
        
        # Add step indices (compact)
        step_indices = node_data.get("step_indices", [])
        if step_indices:
            if len(step_indices) <= 3:
                lines.append(f"steps: {','.join(map(str, step_indices))}")
            else:
                lines.append(f"steps: {step_indices[0]}..{step_indices[-1]} ({len(step_indices)})")
        
        # Add key argument info
        if isinstance(args, dict):
            # Path (shortened)
            path = args.get("path")
            if path:
                path_str = str(path).replace("\\", "/")
                if "/" in path_str:
                    path_parts = path_str.split("/")
                    if len(path_parts) > 2:
                        lines.append(f".../{'/'.join(path_parts[-2:])}")
                    else:
                        lines.append(path_str[-30:] if len(path_str) > 30 else path_str)
                else:
                    lines.append(path_str[-30:] if len(path_str) > 30 else path_str)
            
            # View range
            view_range = args.get("view_range")
            if isinstance(view_range, (list, tuple)) and len(view_range) == 2:
                lines.append(f"L{view_range[0]}-{view_range[1]}")
        
        return "\\n".join(lines)

    def _build_tooltip(self, node_data: Dict[str, Any]) -> str:
        """Build detailed tooltip content for a node."""
        tooltip_parts = []
        
        # Node label/command
        label = node_data.get("label", "")
        tooltip_parts.append(f"<strong>{self._escape_html(label)}</strong>")
        
        # Tool and subcommand
        tool = node_data.get("tool")
        subcommand = node_data.get("subcommand")
        if tool:
            tooltip_parts.append(f"Tool: {self._escape_html(tool)}")
        if subcommand:
            tooltip_parts.append(f"Subcommand: {self._escape_html(subcommand)}")
        
        # Phases
        phases = node_data.get("phases", ["general"])
        tooltip_parts.append(f"Phases: {', '.join(set(phases))}")
        
        # Step indices and thought lengths
        step_indices = node_data.get("step_indices", [])
        thought_lengths = node_data.get("thought_lengths", [])

        if step_indices:
            tooltip_parts.append(f"Step: {', '.join(map(str, step_indices))}")

        if thought_lengths:
            if len(thought_lengths) == 1:
                tooltip_parts.append(f"Thought len: {thought_lengths[0]}")
            else:
                tooltip_parts.append(f"Thought lengths: {', '.join(map(str, thought_lengths))}")
        
        # Arguments (formatted nicely)
        args = node_data.get("args", {})
        if isinstance(args, dict) and args:
            tooltip_parts.append("<br><strong>Arguments:</strong>")
            for k, v in args.items():
                if k == "edit_status":
                    tooltip_parts.append(f"  • {k}: <strong>{v}</strong>")
                elif isinstance(v, str) and len(v) > 100:
                    tooltip_parts.append(f"  • {k}: {self._escape_html(v[:100])}...")
                else:
                    tooltip_parts.append(f"  • {k}: {self._escape_html(str(v))}")
        
        return "<br>".join(tooltip_parts)

    def _prepare_nodes_data(self, G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        """Prepare node data for JSON serialization."""
        nodes_data = []
        for node_id, node_data in G.nodes(data=True):
            colors = self._node_phase_colors(node_data)
            primary_color = colors[0] if colors else self.phase_colors["general"]
            
            display_label = self._make_simple_label(node_data)
            tooltip = self._build_tooltip(node_data)
            
            # Check if this action failed
            args = node_data.get("args", {})
            has_failure = False
            if isinstance(args, dict):
                edit_status = args.get("edit_status", "")
                if edit_status and str(edit_status).startswith("failure"):
                    has_failure = True
            
            # Check if this node had cd stripped
            has_cd = node_data.get("has_cd", False)
            
            nodes_data.append({
                "id": node_id,
                "label": display_label,
                "tooltip": tooltip,
                "color": primary_color,
                "colors": colors,
                "has_failure": has_failure,
                "has_cd": has_cd
            })
        
        return nodes_data

    def _prepare_edges_data(self, G: nx.MultiDiGraph) -> List[Dict[str, Any]]:
        """Prepare edge data for JSON serialization with thought length information.
        
        Key Logic:
        - Each edge connects two nodes
        - Only the FIRST edge in a trajectory step gets the thought_length
        - Subsequent edges in same step get thought_length=0
        - Multi-node steps are detected when multiple nodes share the same trajectory step_idx
        - Edges between nodes in the same step get special blue dotted styling
        """
        edges_data = []
        
        # Build map: step_idx -> list of node_ids in that step
        step_to_nodes = {}
        for node_id, node_data in G.nodes(data=True):
            step_indices = node_data.get("step_indices", [])
            for step_idx in step_indices:
                if step_idx not in step_to_nodes:
                    step_to_nodes[step_idx] = []
                step_to_nodes[step_idx].append(node_id)
        
        # Process each edge
        for u, v, k, d in G.edges(keys=True, data=True):
            etype = d.get("type", "exec")
            edge_label = str(d.get("label", ""))
            is_first_in_step = d.get("is_first_in_step", False)
            
            thought_length = 0
            is_multi_node_step = False
            
            if etype == "exec":
                # Get the source node's data
                u_data = G.nodes[u]
                u_step_indices = u_data.get("step_indices", [])
                u_thought_lengths = u_data.get("thought_lengths", [])
                
                # Get the target node's data
                v_data = G.nodes[v]
                v_step_indices = v_data.get("step_indices", [])
                
                # Find common steps between source and target
                common_steps = set(u_step_indices) & set(v_step_indices)
                
                if common_steps:
                    # Nodes share the same trajectory step - this is a multi-node step
                    is_multi_node_step = True
                    # Intra-step edges always have thought_length = 0
                    thought_length = 0
                else:
                    # Normal inter-step edge
                    # Only the FIRST edge in a trajectory step gets the thought length
                    if is_first_in_step:
                        # Use the source node's thought length
                        if u_step_indices and u_thought_lengths:
                            # Use the most recent (last) thought length from source node
                            thought_length = u_thought_lengths[-1] if u_thought_lengths else 0
                    else:
                        # Not the first edge in step - no thought visualization
                        thought_length = 0
            
            edges_data.append({
                "from": u,
                "to": v,
                "type": etype,
                "label": edge_label if etype == "exec" else "",
                "thought_length": thought_length,
                "is_multi_node_step": is_multi_node_step,
                "is_first_in_step": is_first_in_step
            })
        
        return edges_data

    def draw_graph_html(self, G: nx.MultiDiGraph, html_path: str, metadata_comment: str = ""):
        """Generate complete HTML visualization with external templates.
        
        Args:
            G: NetworkX MultiDiGraph to visualize
            html_path: Output path for HTML file
            metadata_comment: Optional comment about model/plan (e.g., "Model: GPT-4, Plan: ReAct")
        """
        # Build the mapping once per graph
        self._str_id_map = self._build_str_id_map(G)
        
        # Prepare data
        nodes_data = self._prepare_nodes_data(G)
        edges_data = self._prepare_edges_data(G)
        
        # Get graph metadata
        instance_name = G.graph.get("instance_name", "Unknown")
        resolution_status = G.graph.get("resolution_status", "unknown")
        difficulty = G.graph.get("debug_difficulty", "unknown")
        
        # Generate inline CSS and JS with templates
        css_content = self.css_template.replace("{{FONT_FAMILY}}", FONT_FAMILY)
        js_content = self.js_template
        
        # Build complete HTML with inline styles and scripts
        html_content = self.html_template
        
        # Replace metadata placeholders
        html_content = html_content.replace("{{INSTANCE_NAME}}", self._escape_html(instance_name))
        html_content = html_content.replace("{{RESOLUTION_STATUS}}", resolution_status)
        html_content = html_content.replace("{{DIFFICULTY}}", self._escape_html(str(difficulty)))
        html_content = html_content.replace("{{NODE_COUNT}}", str(len(nodes_data)))
        html_content = html_content.replace("{{EDGE_COUNT}}", str(len(edges_data)))
        html_content = html_content.replace("{{METADATA_COMMENT}}", self._escape_html(metadata_comment))
        
        # Replace data placeholders
        html_content = html_content.replace("{{NODES_DATA}}", json.dumps(nodes_data))
        html_content = html_content.replace("{{EDGES_DATA}}", json.dumps(edges_data))
        html_content = html_content.replace("{{PHASE_COLORS}}", json.dumps(self.phase_colors))
        
        # Inline CSS and JS
        html_content = html_content.replace(
            '<link rel="stylesheet" href="styles.css">',
            f'<style>{css_content}</style>'
        )
        html_content = html_content.replace(
            '<script src="graph_renderer.js"></script>',
            f'<script>{js_content}</script>'
        )
        
        # Write HTML file
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
