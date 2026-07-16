#!/bin/bash
set -e

# Figure Generation: Create publication plots from analysis results
# Usage: bash scripts/reproduce.sh [--output-dir DIR]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

show_help() {
    cat << 'EOF'
Usage: bash scripts/reproduce.sh [OPTIONS]

Generate all publication figures (RQ1-RQ3) from precomputed analysis results.

Requires: Precomputed analysis data in data/{agent}/analysis/{model}/

OPTIONS:
  --output-dir DIR, -o DIR   Output directory for figures (default: figures/)
  --help, -h                 Show this help

GENERATED FIGURES:
  figures/median_iqr_trajectory_heatmap.png     (RQ1: Process metrics)
  figures/sankey_grid.png                       (RQ2: Phase transitions)
  figures/end_phase_donuts.png                  (RQ2: End phase distribution)
  figures/phase_transition_overview.png         (RQ2: Phase flow)
  figures/inefficiency_venn/*.pdf               (RQ3: Inefficiency patterns)

EXAMPLES:
  bash scripts/reproduce.sh
  bash scripts/reproduce.sh --output-dir /tmp/figs
  bash scripts/reproduce.sh -o ./output/figures

EOF
}

OUTPUT_DIR="figures"

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir|-o) OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h) show_help; exit 0 ;;
        *) error "Unknown option: $1" ;;
    esac
done

mkdir -p "$OUTPUT_DIR/inefficiency_venn"

# Verify plot scripts exist
declare -a SCRIPTS=(
    "plot/trajectory_heatmap_plot.py"
    "plot/sankey_phase_transition_plot.py"
    "plot/end_phase_plot.py"
    "plot/phase_transition_plot.py"
    "plot/inefficiency_plot.py"
)

for SCRIPT in "${SCRIPTS[@]}"; do
    [[ -f "$SCRIPT" ]] || error "Plot script not found: $SCRIPT"
done

info "Generating publication figures..."
echo

for SCRIPT in "${SCRIPTS[@]}"; do
    NAME=$(basename "$SCRIPT" .py)
    info "Running: $NAME"
    python "$SCRIPT" || error "Plot generation failed: $NAME"
done

echo
info "✓ All figures generated"
info "Output: $OUTPUT_DIR/"
