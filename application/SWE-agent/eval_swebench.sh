RUN_ID="${1:-1}"
CONFIG="oscillation"
MODEL_NAME=openrouter/mistralai/devstral-small
MODEL="${MODEL_NAME##*/}"
RESULTS_DIR="trajectories/$CONFIG/exp-${RUN_ID}/$MODEL"
PREDICTION_PATH="$RESULTS_DIR/preds.json"
python -m swebench.harness.run_evaluation \
    --dataset_name SWE-bench/SWE-bench_Verified \
    --predictions_path "$PREDICTION_PATH" \
    --run_id "tmp" \
    --report_dir "reports"
