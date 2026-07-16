#!/usr/bin/env bash
# run_graph_analysis.sh — Analyze trajectory graphs (SWE-agent or OpenHands)
# Usage:
#   bash run_graph_analysis.sh [agent] [model] [config]
# Behavior:
#   - If agent is omitted, run the specified model on ALL valid agents.
#   - If model is omitted, run the specified agent on ALL valid models.
#   - If both omitted, run ALL agent×model pairs.
# Notes:
#   - Agents: SWE-agent | OpenHands
#   - Models:
#       - deepseek/deepseek-chat
#       - openrouter/mistralai/devstral-small
#       - openrouter/deepseek/deepseek-r1-0528
#   - SWE-agent: [config] optional, defaults to anthropic_filemap
#   - OpenHands: [config] ignored

set -euo pipefail

VALID_AGENTS=("SWE-agent" "OpenHands")
VALID_MODELS=("deepseek/deepseek-chat" "openrouter/mistralai/devstral-small" "openrouter/deepseek/deepseek-r1-0528" "openrouter/anthropic/claude-sonnet-4")

usage() {
  cat >&2 <<'EOF'
Usage:
  bash run_graph_analysis.sh [agent] [model] [config]

Examples:
  bash run_graph_analysis.sh                         # all pairs
  bash run_graph_analysis.sh SWE-agent               # SWE-agent × all models
  bash run_graph_analysis.sh openrouter/deepseek/deepseek-r1-0528   # model × both agents
  bash run_graph_analysis.sh SWE-agent deepseek/deepseek-chat       # specific pair
  bash run_graph_analysis.sh OpenHands openrouter/mistralai/devstral-small
  bash run_graph_analysis.sh SWE-agent deepseek/deepseek-chat my_cfg # SWE-agent with config

Allowed agents: ${VALID_AGENTS[*]}
Allowed models: ${VALID_MODELS[*]}
EOF
  exit 1
}

# Trim CRLF and outer whitespace
_trim() {
  # prints trimmed version of $1
  # shellcheck disable=SC2001
  echo -n "$1" | sed -e 's/\r$//' -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//'
}

is_in_list() {
  local needle="$1"; shift
  local x
  for x in "$@"; do
    [[ "$x" == "$needle" ]] && return 0
  done
  return 1
}

# --- Parse args (agent/model optional; config only for SWE-agent) ---
AGENT=""
MODEL=""
CONFIG=""

case $# in
  0)
    # all pairs
    ;;
  1)
    A="$(_trim "$1")"
    if is_in_list "$A" "${VALID_AGENTS[@]}"; then
      AGENT="$A"
    elif is_in_list "$A" "${VALID_MODELS[@]}"; then
      MODEL="$A"
    else
      echo "Error: '$A' is neither a valid agent nor a valid model." >&2
      usage
    fi
    ;;
  2|3)
    A1="$(_trim "$1")"
    A2="$(_trim "$2")"
    # Accept either order: [agent model] OR [model agent]
    if is_in_list "$A1" "${VALID_AGENTS[@]}" && is_in_list "$A2" "${VALID_MODELS[@]}"; then
      AGENT="$A1"; MODEL="$A2"
    elif is_in_list "$A1" "${VALID_MODELS[@]}" && is_in_list "$A2" "${VALID_AGENTS[@]}"; then
      MODEL="$A1"; AGENT="$A2"
    else
      # Give targeted diagnostics
      err1=""; err2=""
      is_in_list "$A1" "${VALID_AGENTS[@]}" || is_in_list "$A1" "${VALID_MODELS[@]}" || err1=" '$A1'"
      is_in_list "$A2" "${VALID_AGENTS[@]}" || is_in_list "$A2" "${VALID_MODELS[@]}" || err2=" '$A2'"
      echo "Error: could not interpret arguments.${err1}${err2}" >&2
      echo "  Agents: ${VALID_AGENTS[*]}" >&2
      echo "  Models: ${VALID_MODELS[*]}" >&2
      usage
    fi
    if [[ $# -eq 3 ]]; then
      CONFIG="$(_trim "$3")"
    fi
    ;;
  *)
    usage
    ;;
esac

# Default config for SWE-agent if needed
: "${CONFIG:=anthropic_filemap}"

# Resolve script dir; run from there so relative paths in Python match.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Allow overriding Python via $PYTHON; fallback to python3 then python.
PY="${PYTHON:-python3}"
command -v "$PY" >/dev/null 2>&1 || PY=python

# Build execution matrix
AGENTS_TO_RUN=()
MODELS_TO_RUN=()

if [[ -z "$AGENT" ]]; then
  AGENTS_TO_RUN=("${VALID_AGENTS[@]}")
else
  AGENTS_TO_RUN=("$AGENT")
fi

if [[ -z "$MODEL" ]]; then
  MODELS_TO_RUN=("${VALID_MODELS[@]}")
else
  MODELS_TO_RUN=("$MODEL")
fi

echo "==> Plan:"
echo "    Agents: ${AGENTS_TO_RUN[*]}"
echo "    Models: ${MODELS_TO_RUN[*]}"
[[ " ${AGENTS_TO_RUN[*]} " =~ " SWE-agent " ]] && echo "    SWE-agent config: '${CONFIG}'"

TOTAL=$(( ${#AGENTS_TO_RUN[@]} * ${#MODELS_TO_RUN[@]} ))
i=0
for a in "${AGENTS_TO_RUN[@]}"; do
  for m in "${MODELS_TO_RUN[@]}"; do
    i=$((i+1))
    if [[ "$a" == "SWE-agent" ]]; then
      echo ""
      echo "[$i/$TOTAL] ==> Analyzing (agent='${a}', model='${m}', config='${CONFIG}')"
      "$PY" analyze_graph.py "$a" "$m" "$CONFIG"
    else
      echo ""
      echo "[$i/$TOTAL] ==> Analyzing (agent='${a}', model='${m}')"
      "$PY" analyze_graph.py "$a" "$m"
    fi
    echo "[✓] Done: (agent='${a}', model='${m}')"
  done
done

echo ""
echo "✅ All analyses complete."
