#!/usr/bin/env bash
# General run script: quickly switch between Deepseek or other vLLM-compatible backends via environment variables

set -euo pipefail

# ---------- Environment variable configuration (can be overridden via environment variables) ----------
# LLM related (default: Deepseek/Volcano). Supply real keys via env or edit here.
export LLM_PROVIDER="${LLM_PROVIDER:-deepseek}"
export VOLCANO_BASE_URL="${VOLCANO_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
export VOLCANO_MODEL="${VOLCANO_MODEL:-deepseek-v3-2-251201}"
export VOLCANO_API_KEY="${VOLCANO_API_KEY:-47041ffc-3c83-49ee-9d79-4f70592850d2}"

# OpenAI/vLLM (set if using; leave empty to avoid accidental key leaks)
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_MODEL="${VLLM_MODEL:-local-model}"
export VLLM_API_KEY="${VLLM_API_KEY:-}"

# Web search (Serper) for sandbox data retrieval
export SERPER_API_KEY="${SERPER_API_KEY:-359135a8666e6c3934dc758cd2c48fdb21621fc9}"
# Jina API key for embeddings/rerank cleanup
export JINA_API_KEY="${JINA_API_KEY:-jina_b7ac238911474f91a7c06eddede292d7qFimJoGGPvlsGNfxxUU8duLXqjRi}"
export JINA_TIMEOUT="${JINA_TIMEOUT:-15}"

# Maximum bytes to download per dataset file (20M)
export DATASET_MAX_BYTES="${DATASET_MAX_BYTES:-20000000}"
# Data file output limits (0 means no limit)
export MAX_DATA_FILES="${MAX_DATA_FILES:-6}"
export MAX_SAMPLE_ROWS="${MAX_SAMPLE_ROWS:-1000}"
# Raw page capture toggle (1 = save raw HTML + web_pages.db, 0 = skip raw HTML)
export USE_RAW_PAGES="${USE_RAW_PAGES:-0}"

# SandboxFusion
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-http://localhost:8080}"
export SANDBOX_FUSION_TIMEOUT="${SANDBOX_FUSION_TIMEOUT:-30}"
export SANDBOX_FUSION_PORT="${SANDBOX_FUSION_PORT:-8080}"

# Runtime parameters
CATEGORY="${CATEGORY:-Paris Travel Planning}"
SANDBOX="${SANDBOX:-./sandbox/run}"
ROUNDS="${ROUNDS:-5}"
VALIDATE="${VALIDATE:-1}"
MAX_VALIDATION_ROUNDS="${MAX_VALIDATION_ROUNDS:-5}"
USE_SANDBOX_FUSION="${USE_SANDBOX_FUSION:-1}"
MERGE="${MERGE:-0}"  # Default: overwrite (0), set to 1 to merge
MAX_TOKENS="${MAX_TOKENS:-10000}"  # Maximum tokens for LLM generation
# LLM I/O logging (set LLM_LOG_IO=1 to enable, optional LLM_LOG_IO_FILE for path)
export LLM_LOG_IO="${LLM_LOG_IO:-1}"
export LLM_LOG_IO_FILE="${LLM_LOG_IO_FILE:-sandbox/run/llm_io.log}"

# ---------- Assemble command arguments ----------
args=(
  --category "$CATEGORY"
  --sandbox "$SANDBOX"
  --rounds "$ROUNDS"
  --max-tokens "$MAX_TOKENS"
  --max-validation-rounds "$MAX_VALIDATION_ROUNDS"
)

if [[ "$VALIDATE" == "0" ]]; then
  args+=(--no-validate)
fi

if [[ "$USE_SANDBOX_FUSION" != "1" ]]; then
  echo "ERROR: SandboxFusion is required. Set USE_SANDBOX_FUSION=1." >&2
  exit 1
fi
args+=(--use-sandbox-fusion)

if [[ "$MERGE" == "1" ]]; then
  args+=(--merge)
else
  args+=(--no-merge)
fi

# ---------- Execute main program ----------
# Environment checking and logging handling have been moved to Python code
PYTHONUNBUFFERED=1 PYTHONPATH="$(pwd)/general_agent_bundle:${PYTHONPATH:-}" python -u -m agent_gem synthesize "${args[@]}" "$@"
