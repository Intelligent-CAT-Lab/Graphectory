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

[[ $# -ge 1 ]] || {
    cat << 'EOF'
Usage: analyze.sh <data_dir> [output_dir] [--agent AGENT] [--model MODEL]

Analyze precomputed graphs and compute metrics.

ARGS:
  data_dir            Directory containing agent/graphs/ subdirectories
  output_dir          Output directory for analysis results (default: .)

OPTIONS:
  --agent AGENT       Filter: SWE-agent or OpenHands (optional)
  --model MODEL       Filter by model substring (optional)

STRUCTURE:
  data_dir/
    ├── SWE-agent/graphs/model1/instance/*.json
    └── OpenHands/graphs/model2/instance/*.json

EXAMPLE:
  analyze.sh data/
  analyze.sh data/ ./output --agent SWE-agent
  analyze.sh data/ ./output --model deepseek-r1

EOF
    exit 1
}

DATA_DIR="$1"
OUTPUT_DIR="${2:-.}"
shift 2 || shift 1

[[ -d "$DATA_DIR" ]] || error "Not found: $DATA_DIR"

python -c "import graph_analysis" 2>/dev/null || error "graph_analysis module not installed"

# Check if graphs exist
GRAPHS_PATH=$(find "$DATA_DIR" -type d -name "graphs" -print -quit 2>/dev/null)
if [[ -z "$GRAPHS_PATH" ]]; then
    error "No precomputed graphs found in $DATA_DIR"
fi

DATA_BASE="$(dirname "$GRAPHS_PATH")"

info "Data directory: $DATA_BASE"
info "Output directory: $OUTPUT_DIR"
echo

CMD="python -m graph_analysis.batch_runner --data-dir $DATA_BASE"
[[ "$OUTPUT_DIR" != "." ]] && CMD="$CMD --output-dir $OUTPUT_DIR"

# Append remaining arguments (--agent, --model)
while [[ $# -gt 0 ]]; do
    CMD="$CMD $1"
    shift
done

eval "$CMD" || error "Analysis failed"

info "✓ Analysis complete"
