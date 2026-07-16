#!/bin/bash
set -e

# Graph Analysis: Compute metrics from graphs
# Usage: bash scripts/analyze.sh <data_dir> [output_dir] [--agent AGENT] [--model MODEL]

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }

show_help() {
    cat << 'EOF'
Usage: bash scripts/analyze.sh <data_dir> [output_dir] [OPTIONS]

Analyze precomputed graphs and compute trajectory metrics.

ARGS:
  data_dir            Base directory containing agent/graphs/ subdirectories
  output_dir          Output directory for analysis results (default: .)

OPTIONS:
  --agent AGENT       Filter by agent: SWE-agent or OpenHands (optional)
  --model MODEL       Filter by model substring (optional)

EXPECTED STRUCTURE:
  data_dir/
    ├── SWE-agent/graphs/model1/instance/*.json
    └── OpenHands/graphs/model2/instance/*.json

OUTPUT:
  data_dir/{agent}/analysis/{model}/trajectory_metrics.csv

EXAMPLES:
  bash scripts/analyze.sh data/
  bash scripts/analyze.sh data/ ./analysis_output
  bash scripts/analyze.sh data/ . --agent SWE-agent --model deepseek-r1

EOF
}

[[ $# -lt 1 ]] && { show_help; exit 1; }

DATA_DIR="$1"
OUTPUT_DIR="${2:-.}"

[[ -d "$DATA_DIR" ]] || error "Data directory not found: $DATA_DIR"

python -c "import graph_analysis" 2>/dev/null || error "graph_analysis module not installed"

# Find graphs directory
GRAPHS_PATH=$(find "$DATA_DIR" -type d -name "graphs" -print -quit 2>/dev/null)
if [[ -z "$GRAPHS_PATH" ]]; then
    error "No precomputed graphs found in $DATA_DIR (expected: */graphs/)"
fi

DATA_BASE="$(dirname "$GRAPHS_PATH")"

info "Starting graph analysis..."
info "Data: $DATA_BASE"
info "Output: $OUTPUT_DIR"
echo

# Build command
CMD="python -m graph_analysis.batch_runner --data-dir $DATA_BASE"
[[ "$OUTPUT_DIR" != "." ]] && CMD="$CMD --output-dir $OUTPUT_DIR"

# Pass remaining arguments (--agent, --model)
shift 2 || shift 1
while [[ $# -gt 0 ]]; do
    CMD="$CMD $1"
    shift
done

eval "$CMD" || error "Graph analysis failed"

info "✓ Analysis complete. Results in $OUTPUT_DIR/"
