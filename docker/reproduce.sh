#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

[[ $# -eq 0 ]] && {
    cat << 'EOF'
Usage: reproduce.sh [options]

Generate all publication figures from precomputed graphs and analysis results.

This script requires precomputed analysis data in data/*/analysis/.
To generate graphs and analysis first, use:
  - docker/construct.sh (trajectories → graphs)
  - docker/analyze.sh    (graphs → analysis metrics)

OPTIONS:
  -o, --output-dir DIR    Save figures to DIR (default: figures/)
  -h, --help              Show this help

FIGURES GENERATED (RQ1-RQ3):
  figures/median_iqr_trajectory_heatmap.png     (RQ1: Metrics heatmap)
  figures/sankey_grid.png                       (RQ2: Phase transitions)
  figures/end_phase_donuts.png                  (RQ2: End phase distribution)
  figures/phase_transition_overview.png         (RQ2: Phase flow overview)
  figures/inefficiency_venn/*.pdf               (RQ3: Inefficiency patterns)

EXAMPLE:
  reproduce.sh
  reproduce.sh -o /output

EOF
    exit 1
}

OUTPUT_DIR="figures"

while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) $0; exit 0 ;;
        *) error "Unknown option: $1" ;;
    esac
done

mkdir -p "$OUTPUT_DIR/inefficiency_venn"

# Verify plot scripts exist
declare -a PLOTS=(
    "plot/trajectory_heatmap_plot.py"
    "plot/sankey_phase_transition_plot.py"
    "plot/end_phase_plot.py"
    "plot/phase_transition_plot.py"
    "plot/inefficiency_plot.py"
)

MISSING=0
for PLOT in "${PLOTS[@]}"; do
    [[ -f "$PLOT" ]] || { warn "Not found: $PLOT"; ((MISSING++)); }
done

[[ $MISSING -eq 0 ]] || error "$MISSING plot scripts missing"

echo
info "====== Generating Publication Figures ======"
echo

for PLOT in "${PLOTS[@]}"; do
    NAME=$(basename "$PLOT" .py)
    info "Running: $NAME"
    python "$PLOT" || error "Plot generation failed: $NAME"
done

echo
info "====== Complete! ======"
info "Figures saved to: $OUTPUT_DIR/"
echo
