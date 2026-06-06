#!/usr/bin/env bash
# General run script: quickly switch between Deepseek or other vLLM-compatible backends via environment variables

set -euo pipefail

# ---------- Environment variable configuration (can be overridden via environment variables) ----------
# LLM related (default: Deepseek/Volcano). Supply real keys via env or edit here.
export LLM_PROVIDER="${LLM_PROVIDER:-deepseek}"
export VOLCANO_BASE_URL="${VOLCANO_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
export VOLCANO_MODEL="${VOLCANO_MODEL:-deepseek-v4-flash-260425}"
# export VOLCANO_API_KEY="${VOLCANO_API_KEY:-47041ffc-3c83-49ee-9d79-4f70592850d2}"
# export VOLCANO_API_KEY="${VOLCANO_API_KEY:-f5166447-c00e-484f-94f9-cee55ef9139e}"
# export VOLCANO_API_KEY="${VOLCANO_API_KEY:-76f961c5-df43-48e5-88c4-aa50d88bb792}"
export VOLCANO_API_KEY="${VOLCANO_API_KEY:-f5166447-c00e-484f-94f9-cee55ef9139e}"


# OpenAI/vLLM (set if using; leave empty to avoid accidental key leaks)
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-}"
export OPENAI_MODEL="${OPENAI_MODEL:-}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_MODEL="${VLLM_MODEL:-local-model}"
export VLLM_API_KEY="${VLLM_API_KEY:-}"

# Web search (Serper) for sandbox data retrieval
export SERPER_API_KEY="${SERPER_API_KEY:-197d0e578b52e0177e271974bb004329571c9a05}"

# Firecrawl
# fc-64f3247dc29b4ccfba207223fd3a3633
# export FIRECRAWL_API_KEY="${FIRECRAWL_API_KEY:-fc-2993ffd188e94e0e97fb97a82aa21488}"
# export FIRECRAWL_API_URL="https://api.firecrawl.dev/v2"
# export FIRECRAWL_API_KEY="fc-64f3247dc29b4ccfba207223fd3a3633"
export FIRECRAWL_API_URL="http://localhost:3102"
# if [[ -z "${FIRECRAWL_API_URL:-}" ]]; then
#   if [[ -n "${FIRECRAWL_API_KEY:-}" ]]; then
#     export FIRECRAWL_API_URL="https://api.firecrawl.dev/v2"
#     export FIRECRAWL_API_KEY="${FIRECRAWL_API_KEY:-fc-2993ffd188e94e0e97fb97a82aa21488}"
#   else
#     # export FIRECRAWL_API_URL="http://localhost:3000"
#     # export FIRECRAWL_API_URL="http://localhost:3001"
#     # export FIRECRAWL_API_URL="http://localhost:3002"
#     export FIRECRAWL_API_URL="http://localhost:3002"
#   fi
# fi

# Maximum bytes to download per dataset file (20M)
export DATASET_MAX_BYTES="${DATASET_MAX_BYTES:-20000000}"
# Data file output limits (0 means no limit)
export MAX_DATA_FILES="${MAX_DATA_FILES:-10}"
export MAX_SAMPLE_ROWS="${MAX_SAMPLE_ROWS:-1000}"

# SandboxFusion
# docker run -d --privileged -p 8080:8080 volcengine/sandbox-fusion:server-20250609 make run-online
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-http://localhost:8080}"
export SANDBOX_FUSION_TIMEOUT="${SANDBOX_FUSION_TIMEOUT:-30}"
export SANDBOX_FUSION_PORT="${SANDBOX_FUSION_PORT:-8080}"

# Runtime parameters
NUM_CATEGORIES="${NUM_CATEGORIES:-300}"
NUM_TASKS="${NUM_TASKS:-2}"
SANDBOX="${SANDBOX:-./sandbox/run3}"
ROUNDS="${ROUNDS:-3}"
VALIDATE="${VALIDATE:-1}"
MAX_VALIDATION_ROUNDS="${MAX_VALIDATION_ROUNDS:-5}"
USE_SANDBOX_FUSION="${USE_SANDBOX_FUSION:-1}"
MAX_TOKENS="${MAX_TOKENS:-10000}"  # Maximum tokens for LLM generation
CATEGORY="${CATEGORY:-}"  # Optional: specify category manually (if NUM_CATEGORIES=1)


export SCENARIO_INDEX="${SCENARIO_INDEX:-300}"

# LLM I/O logging (set LLM_LOG_IO=1 to enable, optional LLM_LOG_IO_FILE for path)
export LLM_LOG_IO="${LLM_LOG_IO:-1}"
export LLM_LOG_IO_FILE="${LLM_LOG_IO_FILE:-${SANDBOX}/llm_io.log}"

# LLM retry optimization (enhanced retry logic with exponential backoff)
export LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-6}"  # Maximum retry attempts (increased from default 3)
export LLM_RETRY_BASE_DELAY="${LLM_RETRY_BASE_DELAY:-2.0}"  # Base delay in seconds
export LLM_RETRY_MAX_DELAY="${LLM_RETRY_MAX_DELAY:-60.0}"  # Maximum delay cap in seconds
export LLM_RETRY_BACKOFF_FACTOR="${LLM_RETRY_BACKOFF_FACTOR:-3.0}"  # Exponential backoff multiplier
export LLM_RETRY_JITTER="${LLM_RETRY_JITTER:-1}"  # Enable jitter to avoid thundering herd (1=yes, 0=no)
export LLM_TIMEOUT="${LLM_TIMEOUT:-600}"  # Timeout in seconds

# ---------- Assemble command arguments ----------
args=(
  --sandbox "$SANDBOX"
  --rounds "$ROUNDS"
  --max-tokens "$MAX_TOKENS"
  --max-validation-rounds "$MAX_VALIDATION_ROUNDS"
  --num "$NUM_TASKS"
  --num-categories "$NUM_CATEGORIES"
)

# Only add --category if specified and NUM_CATEGORIES is 1
if [[ "$NUM_CATEGORIES" == "1" && -n "${CATEGORY:-}" ]]; then
  args+=(--category "$CATEGORY")
fi

if [[ "$VALIDATE" == "0" ]]; then
  args+=(--no-validate)
fi

if [[ "$USE_SANDBOX_FUSION" != "1" ]]; then
  echo "ERROR: SandboxFusion is required. Set USE_SANDBOX_FUSION=1." >&2
  exit 1
fi
args+=(--use-sandbox-fusion)

# ---------- Execute main program ----------
# Prefer the conda environment managed by the user
CONDA_ENV_NAME="${CONDA_ENV_NAME:-tau2}"
if [[ -n "${CONDA_PREFIX:-}" && "$(basename "$CONDA_PREFIX")" == "$CONDA_ENV_NAME" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x "$HOME/anaconda3/envs/$CONDA_ENV_NAME/bin/python" ]]; then
  PYTHON_BIN="$HOME/anaconda3/envs/$CONDA_ENV_NAME/bin/python"
else
  echo "ERROR: Conda environment '$CONDA_ENV_NAME' not found. Activate it first or set CONDA_ENV_NAME." >&2
  exit 1
fi

PYTHONUNBUFFERED=1 PYTHONPATH="$(pwd):$(pwd)/general_agent_bundle:${PYTHONPATH:-}" "$PYTHON_BIN" -u -m agent_gem synthesize "${args[@]}" "$@"
