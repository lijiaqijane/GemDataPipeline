#!/bin/bash
# Batch generation script with vLLM
# Uses Docker and SandboxFusion by default for secure execution

cd /home/victor/project/general_agent_bundle
source /home/victor/anaconda3/bin/activate tb

# Common env for local vLLM
export LLM_PROVIDER=vllm
export VLLM_MODEL=qwen3-4b
export VLLM_BASE_URL=http://localhost:8000/v1
export VLLM_API_KEY=local

# Docker configuration
export DOCKER_IMAGE=python:3.11-slim
export DOCKER_TIMEOUT=30

# SandboxFusion configuration (optional)
export SANDBOX_FUSION_URL=http://localhost:8080

# Check Docker
if ! docker ps > /dev/null 2>&1; then
    echo "⚠️  Warning: Docker is not running. Code execution may fail."
    echo "   Start Docker with: sudo systemctl start docker"
fi

# Batch categories with moderate rounds to improve success rate
for cat in "travel itinerary planning" "personal finance planning" "coding interview prep" "meal planning" "event scheduling"; do
  echo "Processing category: $cat"
  python -m general_agent \
    --category "$cat" \
    --sandbox "./sandbox/$(echo "$cat" | tr ' ' '-' )" \
    --rounds 1
done
