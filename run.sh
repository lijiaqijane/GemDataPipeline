#!/usr/bin/env bash
# General run script: quickly switch between Deepseek or other vLLM-compatible backends via environment variables

set -euo pipefail

# ---------- Environment variable configuration (can be overridden via environment variables) ----------
# LLM related (default: Deepseek/Volcano)
export LLM_PROVIDER="${LLM_PROVIDER:-deepseek}"
export VOLCANO_BASE_URL="${VOLCANO_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
export VOLCANO_MODEL="${VOLCANO_MODEL:-deepseek-v3-2-251201}"
export VOLCANO_API_KEY="${VOLCANO_API_KEY:-47041ffc-3c83-49ee-9d79-4f70592850d2}"

# OpenAI/vLLM (set if using)
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_MODEL="${VLLM_MODEL:-local-model}"
export VLLM_API_KEY="${VLLM_API_KEY:-}"

# SandboxFusion
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-http://localhost:8080}"
export SANDBOX_FUSION_TIMEOUT="${SANDBOX_FUSION_TIMEOUT:-30}"
export SANDBOX_FUSION_PORT="${SANDBOX_FUSION_PORT:-8080}"

# Runtime parameters
CATEGORY="${CATEGORY:-Paris Travel Planning}"
SANDBOX="${SANDBOX:-./sandbox/run}"
ROUNDS="${ROUNDS:-2}"
VALIDATE="${VALIDATE:-1}"
USE_SANDBOX_FUSION="${USE_SANDBOX_FUSION:-1}"

# ---------- Assemble command arguments ----------
args=(
  --category "$CATEGORY"
  --sandbox "$SANDBOX"
  --rounds "$ROUNDS"
)

if [[ "$VALIDATE" == "0" ]]; then
  args+=(--no-validate)
fi

if [[ "$USE_SANDBOX_FUSION" == "1" ]]; then
  args+=(--use-sandbox-fusion)
else
  args+=(--no-sandbox-fusion)
fi

# ---------- Execute main program ----------
# Environment checking and logging handling have been moved to Python code
PYTHONUNBUFFERED=1 PYTHONPATH="$(pwd)/general_agent_bundle:${PYTHONPATH:-}" python -u -m general_agent "${args[@]}" "$@"

