#!/usr/bin/env python3
"""
Inefficiency multi-set 'venn'-style plots

Inputs:
    - trajectory_metrics.csv from each unit (SWE-agent, OpenHands × various models)

Outputs (../figures/):
  - phase_transition_overview.png
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, List, Tuple
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe

# pip install venn
try:
    from venn import venn as venn_draw
except Exception as e:
    raise ImportError("Please install the 'venn' package: pip install venn") from e

# ----------------------- Config -----------------------

AGENTS = ["SWE-agent", "OpenHands"]
MODELS = [
    "deepseek-chat",
    "deepseek-r1-0528",
    "devstral-small",
    "claude-sonnet-4",
]

AGENT_ABBR = {"SWE-agent": "SA", "OpenHands": "OH"}
MODEL_ABBR = {
    "deepseek-chat": "DSK-V3",
    "deepseek-r1-0528": "DSK-R1",
    "devstral-small": "Dev",
    "claude-sonnet-4": "CLD-4",
}

UNITS: List[Tuple[str, str]] = [
    ("SWE-agent", "deepseek-chat"),               # SA+DSK-V3
    ("OpenHands", "deepseek-chat"),               # OH+DSK-V3
    ("SWE-agent", "deepseek-r1-0528"), # SA+DSK-R1
    ("OpenHands", "deepseek-r1-0528"), # OH+DSK-R1
    ("SWE-agent", "devstral-small"),  # SA+Dev
    ("OpenHands", "devstral-small"),  # OH+Dev
    ("SWE-agent", "claude-sonnet-4"), # SA+CLD-4
    ("OpenHands", "claude-sonnet-4"), # OH+CLD-4
]

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"

# Fixed label orders so legend ↔ fills are consistent
LOC_LABEL_ORDER = ["RepeatedView", "ZoomOut", "Scroll", "OverlyDeepZoom"]
TRANS_LABEL_ORDER = ["LV", "PL", "VL", "VP"]

# Pastel colors mapped to labels above
LOC_COLORS = ["#A8D5BA", "#1982c4", "#F7C29B", "#DAB6FC"]                 # green, blue, peach, lilac
PAT_COLORS = ["#E6A8C2", "#A8E6CF", "#FFD3B6", "#ef7c8e", "#bc5090", "#a0c4ff"]
TRANS_COLORS = ["#A8D5BA", "#1982c4", "#F7C29B", "#DAB6FC"]              # green, blue, peach, lilac

ALPHA = 0.60
RESOLUTION_STYLES = {
    "resolved": {"label": "Resolved", "color": "#2A9D8F"},
    "unresolved": {"label": "Unresolved", "color": "#E76F51"},
}

# ----------------------- Paths -----------------------

MODEL_DIR_NAMES = {
    "deepseek-chat": "deepseek-v3",
}

def csv_path(agent: str, model: str) -> Path:
    model_dir = MODEL_DIR_NAMES.get(model, model)
    return DATA_DIR / agent / "analysis" / model_dir / "trajectory_metrics.csv"

def graph_dir(agent: str, model: str) -> Path:
    model_dir = MODEL_DIR_NAMES.get(model, model)
    return DATA_DIR / agent / "graphs" / model_dir

# ----------------------- Flags -----------------------

def _to_bool(series: pd.Series, numeric_positive: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(False, index=pd.RangeIndex(0))
    if numeric_positive:
        return pd.to_numeric(series, errors="coerce").fillna(0) > 0
    return series.map(lambda x: False if pd.isna(x) else (str(x).strip().lower() in {"1","true","t","yes","y"}))

def build_phase_transition_flags(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Build flags for phase transitions: LV, PL, VL, VP"""
    cols = df.columns
    return {
        "LV": _to_bool(df["LV"], numeric_positive=True) if "LV" in cols else pd.Series(False, index=df.index),
        "PL": _to_bool(df["PL"], numeric_positive=True) if "PL" in cols else pd.Series(False, index=df.index),
        "VL": _to_bool(df["VL"], numeric_positive=True) if "VL" in cols else pd.Series(False, index=df.index),
        "VP": _to_bool(df["VP"], numeric_positive=True) if "VP" in cols else pd.Series(False, index=df.index),
    }

def build_localization_flags(df: pd.DataFrame) -> Dict[str, pd.Series]:
    cols = df.columns
    deep_col = "deep_zooms_without_edit" if "deep_zooms_without_edit" in cols else \
               "num_deep_zooms_without_edit" if "num_deep_zooms_without_edit" in cols else None
    return {
        "RepeatedView":    _to_bool(df["repeated_view"], numeric_positive=True) if "repeated_view" in cols else pd.Series(False, index=df.index),
        "ZoomOut":         _to_bool(df["zoom_out"]) if "zoom_out" in cols else pd.Series(False, index=df.index),
        "Scroll":          _to_bool(df["scroll_behavior"]) if "scroll_behavior" in cols else pd.Series(False, index=df.index),
        "OverlyDeepZoom":  _to_bool(df[deep_col], numeric_positive=True) if deep_col is not None else pd.Series(False, index=df.index),
    }

def build_patching_flags(df: pd.DataFrame) -> Dict[str, pd.Series]:
    cols = df.columns
    def fnum(name):
        return _to_bool(df[name], numeric_positive=True) if name in cols else pd.Series(False, index=df.index)
    return {
        "UnresolvedRetry":  _to_bool(df["abandonment"]) if "abandonment" in cols else pd.Series(False, index=df.index),
        "EditReversion":         _to_bool(df["flip_flop"]) if "flip_flop" in cols else pd.Series(False, index=df.index),
        "StrNotFound":         fnum("fail_type_not found"),
        "NoEffectEdit":         fnum("fail_type_no change"),
        "AmbiguousTarget":       fnum("fail_type_multiple occurrences"),
    }

def flags_to_sets(flags: Dict[str, pd.Series]) -> Dict[str, set]:
    out: Dict[str, set] = {}
    for label, ser in flags.items():
        idxs = set(ser[ser.fillna(False)].index.tolist())
        if idxs:
            out[label] = idxs
    return out

def _iter_nodes(graph_json: dict):
    if isinstance(graph_json, dict):
        if "nodes" in graph_json and isinstance(graph_json["nodes"], list):
            yield from graph_json["nodes"]
        elif "graph" in graph_json and isinstance(graph_json["graph"], dict) and "nodes" in graph_json["graph"]:
            yield from graph_json["graph"]["nodes"]


def _extract_phase_sequence(graph_json: dict) -> List[str]:
    phase_abbr = {"localization": "L", "patch": "P", "validation": "V"}
    step_phase: List[Tuple[int, str]] = []

    for node in _iter_nodes(graph_json):
        step_indices = node.get("step_indices") or []
        phases = node.get("phases") or node.get("phase")

        if isinstance(phases, list):
            if len(phases) == len(step_indices):
                for idx, phase in zip(step_indices, phases):
                    step_phase.append((idx, str(phase).lower()))
        else:
            for idx in step_indices:
                step_phase.append((idx, str(phases).lower()))

    if not step_phase:
        return []

    step_phase.sort(key=lambda x: x[0])
    seq: List[str] = []
    prev = None
    for _, phase in step_phase:
        abbr = phase_abbr.get(phase)
        if not abbr:
            continue
        if abbr != prev:
            seq.append(abbr)
            prev = abbr
    return seq


def _build_phase_transition_frame(agent: str, model: str) -> pd.DataFrame:
    rows = []
    root = graph_dir(agent, model)
    if not root.exists():
        return pd.DataFrame(columns=["instance", "LV", "PL", "VL", "VP"])

    for graph_path in sorted(root.rglob("*.json")):
        instance = graph_path.stem
        try:
            with graph_path.open("r", encoding="utf-8") as f:
                graph_json = json.load(f)
            seq = _extract_phase_sequence(graph_json)
        except Exception:
            continue

        transitions = {a + b for a, b in zip(seq, seq[1:])}
        rows.append(
            {
                "instance": instance,
                "LV": int("LV" in transitions),
                "PL": int("PL" in transitions),
                "VL": int("VL" in transitions),
                "VP": int("VP" in transitions),
            }
        )

    return pd.DataFrame(rows)


def load_phase_transition_df(agent: str, model: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path(agent, model))
    if {"LV", "PL", "VL", "VP"}.issubset(df.columns):
        return df

    if "instance" not in df.columns:
        return df

    phase_df = _build_phase_transition_frame(agent, model)
    if phase_df.empty:
        return df

    merged = df.merge(phase_df, on="instance", how="left")
    for col in ("LV", "PL", "VL", "VP"):
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0).astype(int)
    return merged

# ------------ Title helper (subscript model) ------------
def format_title(agent_name: str, model_abbr: str) -> str:
    # e.g. "SWE-agent$_{\mathrm{DSK\text{-}V3}}$"
    safe_model = model_abbr.replace("-", r"\text{-}")
    return rf"{agent_name}$_{{\mathrm{{{safe_model}}}}}$"

# ----------------------- Label utilities -----------------------
def _hide_zero_labels(ax: plt.Axes) -> None:
    for t in ax.texts:
        if t.get_text() in ("0", "0.0", ""):
            t.set_visible(False)

def _repel_labels(ax: plt.Axes, max_iter: int = 280, step: float = 1.0) -> None:
    """Overlap avoidance: nudge labels apart in display coords, keep inside axes."""
    fig = ax.figure
    fig.canvas.draw()
    texts = [t for t in ax.texts if t.get_visible()]
    if len(texts) < 2:
        return

    for _ in range(max_iter):
        moved = False
        bboxes = [t.get_window_extent(fig.canvas.get_renderer()).expanded(1.05, 1.2) for t in texts]
        for i in range(len(texts)):
            for j in range(i+1, len(texts)):
                if not bboxes[i].overlaps(bboxes[j]):
                    continue
                moved = True
                dx = step if (i % 2 == 0) else -step
                dy = (step * 0.35) if (j % 2 == 0) else -(step * 0.35)
                for t, ddx, ddy in ((texts[i], dx, dy), (texts[j], -dx, -dy)):
                    x, y = t.get_position()
                    inv = ax.transData.inverted()
                    new_disp = ax.transData.transform((x, y)) + np.array([ddx, ddy])
                    new_xy = inv.transform(new_disp)
                    t.set_position(new_xy)
        if not moved:
            break

    # clamp within axes area
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim()
    for t in texts:
        x, y = t.get_position()
        x = min(max(x, x0 + 0.01*(x1-x0)), x1 - 0.01*(x1-x0))
        y = min(max(y, y0 + 0.01*(y1-y0)), y1 - 0.01*(y1-y0))
        t.set_position((x, y))

def _recolor_patches_to_labels(ax: plt.Axes, labels_in_order: List[str], color_map: Dict[str, str]) -> None:
    """Match patch colors to our label→color mapping; remove edges; round joins."""
    patches = list(ax.patches)
    n = min(len(patches), len(labels_in_order))
    for i in range(n):
        p = patches[i]
        c = color_map[labels_in_order[i]]
        try:
            p.set_facecolor(c)
            p.set_alpha(ALPHA)
            p.set_edgecolor("none")
            if hasattr(p, "set_joinstyle"):
                p.set_joinstyle("round")
            if hasattr(p, "set_capstyle"):
                p.set_capstyle("round")
        except Exception:
            pass

# ----------------------- Drawing -----------------------
def _add_totals_strip(ax: plt.Axes,
                      labels_in_use: List[str],
                      sets_dict: Dict[str, set],
                      color_map: Dict[str, str]) -> None:
    """
    Add a compact, per-subplot totals legend (counts per type), with abbreviations.
    """
    from matplotlib.patches import Patch
    counts = [len(sets_dict.get(lab, set())) for lab in labels_in_use]
    if sum(counts) == 0:
        return

    # abbreviation mapping
    ABBR = {
        "RepeatedView": "RV",
        "ZoomOut": "ZO",
        "Scroll": "S",
        "OverlyDeepZoom": "DZ",
        "UnresolvedRetry": "UR",
        "EditReversion": "ER",
        "StrNotFound": "NF",
        "NoEffectEdit": "NE",
        "AmbiguousTarget": "AT",
        "LV": "LV",
        "PL": "PL",
        "VL": "VL",
        "VP": "VP",
    }

    handles = [Patch(facecolor=color_map[lab], edgecolor="none", alpha=ALPHA)
               for lab in labels_in_use]
    labels  = [f"{ABBR.get(lab, lab[0])}:{cnt}" for lab, cnt in zip(labels_in_use, counts)]

    ncol = 3 #if len(labels_in_use) > 4 else 1
    leg = ax.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=False,
        fontsize=9.5,
        labelspacing=0.2,
        handlelength=0.9,
        borderaxespad=0.2,
        ncol=ncol,
        columnspacing=0.8,
    )
    for txt in leg.get_texts():
        txt.set_path_effects([pe.withStroke(linewidth=2, foreground="white")])

def _draw_single_venn(ax: plt.Axes,
                      sets_dict: Dict[str, set],
                      label_order: List[str],
                      color_map: Dict[str, str],
                      title: str,
                      total_count: int = None) -> None:
    """
    Draw one subplot with consistent coloring, compact labels,
    title below the axes. If total_count is provided, show percentages.
    """
    if len(sets_dict) < 2:
        ax.text(0.5, 0.5, "Insufficient active types", ha="center", va="center", fontsize=10.5)
        ax.set_title("")  # we'll use xlabel for titles
        ax.set_xlabel(title, fontsize=11.5, fontweight="bold", labelpad=2)
        ax.set_xticks([]); ax.set_yticks([]);
        for sp in ax.spines.values(): sp.set_visible(False)
        return

    # Canonical label order for deterministic coloring
    labels_in_use = [lab for lab in label_order if lab in sets_dict]
    extras = [lab for lab in sets_dict.keys() if lab not in labels_in_use]
    labels_in_use += sorted(extras)

    ordered = OrderedDict((lab, sets_dict[lab]) for lab in labels_in_use)

    venn_draw(ordered, ax=ax)
    if (leg := ax.get_legend()) is not None:
        try: leg.remove()
        except Exception: pass

    _recolor_patches_to_labels(ax, labels_in_use, color_map)
    _hide_zero_labels(ax)

    # Convert counts to percentages if total_count provided
    if total_count and total_count > 0:
        for t in ax.texts:
            try:
                count = int(float(t.get_text()))
                percentage = round((count / total_count) * 100)
                if percentage != 0:
                    t.set_text(f"{percentage}")
            except (ValueError, AttributeError):
                pass

    # Emphasize region counts/percentages
    for t in ax.texts:
        t.set_fontsize(12)
        # t.set_fontweight("bold")
        t.set_color("#1d232a")
        t.set_path_effects([pe.withStroke(linewidth=1.0, foreground="white")])

    # Slightly faster/lighter repel (enough to avoid collisions)
    _repel_labels(ax, max_iter=160, step=0.7)

    # Clean frame
    for sp in ax.spines.values(): sp.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # Put the "title" under the subplot to save vertical space
    ax.set_title("")
    # ax.set_xlabel(title, fontsize=11.5, fontweight="bold", labelpad=2)
    from matplotlib.offsetbox import AnchoredOffsetbox, TextArea, HPacker

    counts = [len(sets_dict.get(lab, set())) for lab in labels_in_use]
    ABBR = {
        "RepeatedView": "RV",
        "ZoomOut": "ZO",
        "Scroll": "S",
        "OverlyDeepZoom": "DZ",
        "UnresolvedRetry": "UR",
        "EditReversion": "ER",
        "StrNotFound": "NF",
        "NoEffectEdit": "NE",
        "AmbiguousTarget": "AT",
        "LV": "LV",
        "PL": "PL",
        "VL": "VL",
        "VP": "VP",
    }

    # clear xlabel (we'll build our own colored strip below the axis)
    ax.set_xlabel(title, fontsize=11.5, fontweight="bold", labelpad=2)

    # build colored parts
    parts = []
    for lab, cnt in zip(labels_in_use, counts):
        abbr = ABBR.get(lab, lab[0])
        if total_count and total_count > 0:
            pct = round((cnt / total_count) * 100)
            txt = f"{abbr}:{pct}%"
        else:
            txt = f"{abbr}:{cnt}"
        part = TextArea(
            txt,
            textprops=dict(color=color_map[lab],
                        fontsize=10,
                        fontweight="bold")
        )
        parts.append(part)

    # pack horizontally with a little spacing
    # totals_box = HPacker(children=parts, align="center", pad=2, sep=3)
    from matplotlib.offsetbox import HPacker, VPacker

    # split into two roughly equal halves
    if len(parts) >= 6:
        mid = len(parts) // 2
        row1 = HPacker(children=parts[:mid], align="center", pad=0.5, sep=1)
        row2 = HPacker(children=parts[mid:], align="center", pad=0.5, sep=1)

        totals_box = VPacker(children=[row1, row2], align="center", pad=0.5, sep=1)
    else:
        totals_box = HPacker(children=parts, align="center", pad=0.5, sep=1)

    # anchor below the xlabel
    anchored = AnchoredOffsetbox(
        loc="lower center",
        child=totals_box,
        pad=0.0,
        frameon=False,
        bbox_to_anchor=(0.5, 0.0),  # shift downward
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    ax.add_artist(anchored)

    # Compact totals strip, only if any active
    # _add_totals_strip(ax, labels_in_use, sets_dict, color_map)


def _filter_by_resolution(df: pd.DataFrame, res_filter: str) -> pd.DataFrame:
    if "resolution" not in df.columns:
        return df.iloc[0:0].copy()
    df_res = df.copy()
    df_res["resolution"] = df_res["resolution"].astype(str).str.strip().str.lower()
    return df_res[df_res["resolution"] == res_filter]


def _add_resolution_badge(ax: plt.Axes, resolution: str) -> None:
    style = RESOLUTION_STYLES[resolution]
    ax.text(
        0.03,
        0.97,
        style["label"],
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color=style["color"],
        bbox=dict(
            boxstyle="round,pad=0.25,rounding_size=0.7",
            facecolor="white",
            edgecolor=style["color"],
            linewidth=1.2,
        ),
    )


def _draw_phase_grid(fig,
                     gs,
                     units: List[Tuple[str, str]],
                     resolution: str,
                     builder,
                     label_order: List[str],
                     colors: List[str]) -> List[str]:
    color_map = {lab: colors[i % len(colors)] for i, lab in enumerate(label_order)}
    active_labels_overall: List[str] = []

    for i, (agent, model) in enumerate(units):
        r, c = divmod(i, 2)
        ax = fig.add_subplot(gs[r, c])

        path = csv_path(agent, model)
        if not path.exists():
            ax.axis("off")
            ax.set_xlabel(f"{AGENT_ABBR[agent]}+{MODEL_ABBR[model]}  (n=0)",
                          fontsize=10, fontweight="bold", labelpad=2)
            continue

        df = load_phase_transition_df(agent, model)
        df_res = _filter_by_resolution(df, resolution)
        total_cases = len(df_res)

        flags = builder(df_res)
        sets_dict = flags_to_sets(flags)
        if sets_dict:
            union = set.union(*sets_dict.values())
            if len(union) == 0:
                sets_dict = {}

        active_labels_overall.extend(list(sets_dict.keys()))

        title = format_title(agent, MODEL_ABBR[model])
        n_used = len(set.union(*sets_dict.values())) if sets_dict else 0
        pct_with_ineff = round((n_used / total_cases) * 100) if total_cases > 0 else 0
        title_with_pct = f"{title}  ({pct_with_ineff}%)"

        _draw_single_venn(ax, sets_dict, label_order, color_map, title_with_pct, total_count=total_cases)

    return [lab for lab in label_order if lab in set(active_labels_overall)]


def plot_phase_transition_overview(units: List[Tuple[str, str]],
                                   builder,
                                   label_order: List[str],
                                   colors: List[str],
                                   out_path: Path) -> None:
    """Render resolved and unresolved grids side by side under one shared legend."""
    fig = plt.figure(figsize=(10.2, 9.6))
    outer = fig.add_gridspec(
        3, 2,
        height_ratios=[0.12, 1.0, 1.0],
        width_ratios=[1.0, 1.0],
        wspace=0.04,
        hspace=0.08,
    )

    left_gs = outer[1:, 0].subgridspec(4, 2, wspace=0.04, hspace=0.08)
    right_gs = outer[1:, 1].subgridspec(4, 2, wspace=0.04, hspace=0.08)

    labels_resolved = _draw_phase_grid(
        fig, left_gs, units, "resolved", builder, label_order, colors
    )
    labels_unresolved = _draw_phase_grid(
        fig, right_gs, units, "unresolved", builder, label_order, colors
    )

    color_map = {lab: colors[i % len(colors)] for i, lab in enumerate(label_order)}
    labels_in_use = [lab for lab in label_order if lab in set(labels_resolved + labels_unresolved)]

    legend_ax = fig.add_subplot(outer[0, :])
    legend_ax.axis("off")
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=color_map[lab], edgecolor="none", alpha=ALPHA, label=lab)
               for lab in labels_in_use]
    if handles:
        legend_ax.legend(
            handles=handles,
            loc="center",
            ncol=len(handles),
            frameon=False,
            fontsize=11,
            handlelength=1.1,
            columnspacing=1.0,
        )

    fig.text(0.25, 0.872, "Resolved", ha="center", va="bottom",
             fontsize=14, fontweight="bold", color=RESOLUTION_STYLES["resolved"]["color"])
    fig.text(0.75, 0.872, "Unresolved", ha="center", va="bottom",
             fontsize=14, fontweight="bold", color=RESOLUTION_STYLES["unresolved"]["color"])

    fig.subplots_adjust(left=0.035, right=0.985, top=0.955, bottom=0.04)
    fig.savefig(out_path, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    out_path = FIG_DIR / "phase_transition_overview.png"
    plot_phase_transition_overview(
        UNITS,
        builder=build_phase_transition_flags,
        label_order=TRANS_LABEL_ORDER,
        colors=TRANS_COLORS,
        out_path=out_path,
    )

if __name__ == "__main__":              
    main()
