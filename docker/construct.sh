#!/bin/bash
set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }

[[ $# -ge 2 ]] || {
    cat << 'EOF'
Usage: construct.sh <trajectories_path> <eval_report.json> [output_dir] [model]

Convert raw trajectories to graph JSON files.

ARGS:
  trajectories_path   Directory with .traj files (SWE-agent) or output.jsonl (OpenHands)
  eval_report.json    SWE-bench evaluation report
  output_dir          Output directory (default: .)
  model               Model code (auto-detected or: dsk-v3, dsk-r1, dev, cld-4, gpt-5-mini)

EXAMPLE:
  construct.sh data/samples/SWE-agent/trajectories data/SWE-agent/reports/deepseek-chat.json

EOF
    exit 1
}

TRAJECTORIES="$1"
EVAL_REPORT="$2"
OUTPUT_DIR="${3:-.}"
MODEL="${4:-}"

[[ -e "$TRAJECTORIES" ]] || error "Not found: $TRAJECTORIES"
[[ -f "$EVAL_REPORT" ]] || error "Not found: $EVAL_REPORT"

python -c "import graph_construction" 2>/dev/null || error "graph_construction module not installed"

# Auto-detect agent and set paths
if [[ -f "$TRAJECTORIES/output.jsonl" ]]; then
    AGENT="oh"
    TRAJS_PATH="$TRAJECTORIES/output.jsonl"
    info "Detected: OpenHands"
elif ls "$TRAJECTORIES"/*.traj >/dev/null 2>&1; then
    AGENT="sa"
    TRAJS_PATH="$TRAJECTORIES"
    info "Detected: SWE-agent"
else
    error "Expected .traj files (SWE-agent) or output.jsonl (OpenHands)"
fi

# Auto-infer model if not provided
if [[ -z "$MODEL" ]]; then
    if [[ $TRAJS_PATH =~ deepseek-chat ]]; then MODEL="dsk-v3"
    elif [[ $TRAJS_PATH =~ deepseek-r1 ]]; then MODEL="dsk-r1"
    elif [[ $TRAJS_PATH =~ devstral ]]; then MODEL="dev"
    elif [[ $TRAJS_PATH =~ claude|sonnet ]]; then MODEL="cld-4"
    elif [[ $TRAJS_PATH =~ gpt-5 ]]; then MODEL="gpt-5-mini"
    else
        error "Cannot infer model. Provide as 4th argument"
    fi
fi

info "Agent: $AGENT"
info "Model: $MODEL"
info "Trajectories: $TRAJECTORIES"
info "Report: $EVAL_REPORT"
info "Output: $OUTPUT_DIR"
echo

python -m graph_construction.generatejson \
    --agent "$AGENT" \
    --model "$MODEL" \
    --trajs "$TRAJS_PATH" \
    --eval_report "$EVAL_REPORT" \
    --output_dir "$OUTPUT_DIR" || error "Graph construction failed"

info "✓ Graphs generated successfully"
