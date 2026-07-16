#!/bin/bash
set -e

# Graph Construction: Convert trajectories to JSON graphs
# Usage: bash scripts/construct.sh <trajectories_path> <eval_report.json> [output_dir] [model]

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

error() { echo -e "${RED}[ERROR]${NC} $*" >&2; exit 1; }
info()  { echo -e "${GREEN}[INFO]${NC} $*"; }

show_help() {
    cat << 'EOF'
Usage: bash scripts/construct.sh <trajectories_path> <eval_report.json> [output_dir] [model]

Convert raw agent trajectories to JSON graph files.

ARGS:
  trajectories_path   Directory with .traj files (SWE-agent) or path to output.jsonl (OpenHands)
  eval_report.json    SWE-bench evaluation report JSON file
  output_dir          Output directory for graphs (default: .)
  model               Model identifier (auto-detected or explicit: dsk-v3, dsk-r1, dev, cld-4, gpt-5-mini)

AGENT AUTO-DETECTION:
  - SWE-agent:  Directory containing .traj files
  - OpenHands:  Directory containing or path to output.jsonl

MODEL AUTO-INFERENCE:
  Inferred from directory name: deepseek-chat, deepseek-r1, devstral, claude, gpt-5

EXAMPLES:
  bash scripts/construct.sh data/samples/SWE-agent/trajectories/anthropic_filemap__deepseek--deepseek-chat__t-0.00__p-1.00__c-2.00___swe_bench_verified_test data/samples/SWE-agent/reports/deepseek-chat.json
  bash scripts/construct.sh data/samples/OpenHands/trajectories/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/sample_output.jsonl data/samples/OpenHands/trajectories/deepseek-chat_maxiter_100_N_v0.40.0-no-hint-run_1/report.json ./graphs
  bash scripts/construct.sh data/trajs report.json . dsk-r1

EOF
}

[[ $# -lt 2 ]] && { show_help; exit 1; }

TRAJECTORIES="$1"
EVAL_REPORT="$2"
OUTPUT_DIR="${3:-.}"
MODEL="${4:-}"

# Validation
[[ -e "$TRAJECTORIES" ]] || error "Trajectories not found: $TRAJECTORIES"
[[ -f "$EVAL_REPORT" ]] || error "Report not found: $EVAL_REPORT"

python -c "import graph_construction" 2>/dev/null || error "graph_construction module not installed"

info "Starting graph construction..."
info "Trajectories: $TRAJECTORIES"
info "Report: $EVAL_REPORT"
info "Output: $OUTPUT_DIR"

# Detect agent type
OH_JSONL=$(find "$TRAJECTORIES" -type f -name "output.jsonl" -print -quit)
SA_TRAJ=$(find "$TRAJECTORIES" -type f -name "*.traj" -print -quit)

if [[ -n "$OH_JSONL" ]]; then
    AGENT="oh"
    TRAJS_PATH="$OH_JSONL"
    info "Agent: OpenHands"
elif [[ -n "$SA_TRAJ" ]]; then
    AGENT="sa"
    TRAJS_PATH="$TRAJECTORIES"
    info "Agent: SWE-agent"
else
    error "Cannot detect agent type. Expected output.jsonl or .traj files under $TRAJECTORIES"
fi


# Infer model if not provided
if [[ -z "$MODEL" ]]; then
    MODEL_PATH="${TRAJS_PATH}:${TRAJECTORIES}"
    if [[ $MODEL_PATH =~ deepseek-chat ]]; then MODEL="dsk-v3"
    elif [[ $MODEL_PATH =~ deepseek-r1 ]]; then MODEL="dsk-r1"
    elif [[ $MODEL_PATH =~ devstral ]]; then MODEL="dev"
    elif [[ $MODEL_PATH =~ claude|sonnet ]]; then MODEL="cld-4"
    elif [[ $MODEL_PATH =~ gpt-5 ]]; then MODEL="gpt-5-mini"
    else
        error "Cannot infer model from path. Provide as 4th argument: dsk-v3, dsk-r1, dev, cld-4, gpt-5-mini"
    fi
fi

info "Model: $MODEL"
echo

python graph_construction/generatejson.py \
    --agent "$AGENT" \
    --model "$MODEL" \
    --trajs "$TRAJS_PATH" \
    --eval_report "$EVAL_REPORT" \
    --output_dir "$OUTPUT_DIR" \
    || error "Graph construction failed"

info "✓ Graphs generated in $OUTPUT_DIR/{agent_name}/graphs/{model}/"
