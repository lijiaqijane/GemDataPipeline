#!/bin/bash
# Test script with Deepseek API
# By default, uses Docker and SandboxFusion for secure execution

cd /home/victor/project/general_agent_bundle
source /home/victor/anaconda3/bin/activate tb

# Deepseek v3.2 configuration
export LLM_PROVIDER=deepseek
export VOLCANO_MODEL=deepseek-v3-2-251201
export VOLCANO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
# Set your actual API key before running
export VOLCANO_API_KEY=${VOLCANO_API_KEY:-"replace-with-your-api-key"}

# Docker configuration (optional)
export DOCKER_IMAGE=python:3.11-slim
export DOCKER_TIMEOUT=30

# SandboxFusion configuration (optional, if service is running)
export SANDBOX_FUSION_URL=http://localhost:8080
export SANDBOX_FUSION_TIMEOUT=30
export SANDBOX_FUSION_LANGUAGE=python

# Check Docker
if ! docker ps > /dev/null 2>&1; then
    echo "⚠️  Warning: Docker is not running. Code execution may fail."
    echo "   Start Docker with: sudo systemctl start docker"
fi

# Test with default secure execution (Docker + SandboxFusion enabled by default)
python -m general_agent \
  --category "travel itinerary planning" \
  --sandbox "./sandbox/test-deepseek" \
  --rounds 1
