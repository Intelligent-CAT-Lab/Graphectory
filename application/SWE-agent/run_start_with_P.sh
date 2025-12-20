#!/bin/bash

# Configuration
RUN_ID="${1:-1}"
CONFIG="start_with_P"

CSV_FILE="/home/shuyang/Graphectory/stats/flawed_trajs/starts_with_P.csv"

# Extract SWE-agent instances and group by model
declare -A MODEL_INSTANCES

while IFS=, read -r agent model resolution_status debug_difficulty instance_id phases link; do
    # Skip header line
    if [[ "$agent" == "agent" ]]; then
        continue
    fi

    # Only process SWE-agent entries
    if [[ "$agent" == "SWE-agent" ]]; then
        # Map model names to OpenRouter format
        case "$model" in
            "deepseek-r1-0528")
                openrouter_model="openrouter/deepseek/deepseek-r1-0528"
                ;;
            "deepseek-v3")
                openrouter_model="openrouter/deepseek/deepseek-chat-v3-0324"
                ;;
            "devstral-small")
                openrouter_model="openrouter/mistralai/devstral-small"
                ;;
            *)
                echo "Unknown model: $model"
                continue
                ;;
        esac

        # Append instance to the model's list (joined by |)
        if [[ -z "${MODEL_INSTANCES[$openrouter_model]}" ]]; then
            MODEL_INSTANCES[$openrouter_model]="$instance_id"
        else
            MODEL_INSTANCES[$openrouter_model]="${MODEL_INSTANCES[$openrouter_model]}|$instance_id"
        fi
    fi
done < "$CSV_FILE"

# Run sweagent for each model with its instances
for model in "${!MODEL_INSTANCES[@]}"; do
    instances="${MODEL_INSTANCES[$model]}"

    # Extract model name from OpenRouter format: openrouter/provider/model-name -> model-name
    MODEL="${model##*/}"
    OUTPUT_DIR="trajectories/$CONFIG/exp-${RUN_ID}/$MODEL"

    echo "Running SWE-agent with model: $model"
    echo "Output directory: $OUTPUT_DIR"
    echo "Instances: $instances"

    sweagent run-batch \
        --config config/$CONFIG.yaml \
        --agent.model.api_base https://openrouter.ai/api/v1 \
        --agent.model.name "$model" \
        --agent.model.api_key $OPENROUTER_API_KEY \
        --num_workers 5 \
        --agent.model.per_instance_cost_limit 2.0 \
        --instances.deployment.docker_args=--memory=10g \
        --agent.model.max_output_tokens 64000 \
        --agent.model.litellm_model_registry litellm_model_registry.json \
        --instances.type swe_bench \
        --instances.subset verified \
        --instances.split test \
        --instances.filter "$instances" \
        --instances.shuffle=False \
        --output_dir "$OUTPUT_DIR"

    echo "Completed batch for model: $model"

    # Evaluate the results
    echo "Starting evaluation for model: $model"
    PREDICTION_PATH="$OUTPUT_DIR/preds.json"

    if [[ -f "$PREDICTION_PATH" ]]; then
        python -m swebench.harness.run_evaluation \
            --dataset_name SWE-bench/SWE-bench_Verified \
            --predictions_path "$PREDICTION_PATH" \
            --run_id "$RUN_ID" \
            --report_dir "reports/$CONFIG"

        if [[ $? -eq 0 ]]; then
            echo "✓ Evaluation completed for model: $model"
        else
            echo "✗ Evaluation failed for model: $model"
        fi
    else
        echo "✗ Predictions file not found: $PREDICTION_PATH"
    fi

    echo "----------------------------------------"
done

echo "All SWE-agent runs completed!"