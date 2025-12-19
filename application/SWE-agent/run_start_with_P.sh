RUN_ID="${1:-1}"
CONFIG="default"
MODEL_NAME=openrouter/mistralai/devstral-small
MODEL="${MODEL_NAME##*/}"
OUTPUT_DIR="trajectories/$CONFIG/exp-${RUN_ID}/$MODEL"

sweagent run-batch \
    --config config/$CONFIG.yaml \
    --agent.model.api_base https://openrouter.ai/api/v1 \
    --agent.model.name $MODEL_NAME\
    --agent.model.api_key $OPENROUTER_API_KEY \
    --num_workers 5 \
    --output_dir $OUTPUT_DIR \
    --agent.model.per_instance_cost_limit 2.0 \
    --instances.deployment.docker_args=--memory=10g \
    --agent.model.max_output_tokens 64000 \
    --agent.model.litellm_model_registry litellm_model_registry.json \
    --instances.type swe_bench \
    --instances.subset verified \
    --instances.split test  \
    --instances.filter django__django-12858 \
    --instances.shuffle=False \