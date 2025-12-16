# AgentGEM: Agent Generative Environment Maker

High-performance generator for RL-ready agentic tasks. It ships four pipelines (Search, Code, Code Interpreter, General) and packages each task into isolated sandboxes for large-scale training.

## Install

### With uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync --group dev
uv run pre-commit install
```

### With pip

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Copy your environment template to `.env` and fill in provider keys:

<<<<<<< HEAD
```bash
cp .env.example .env
# then edit .env with your API keys/models
```

## Model Configuration (DeepSeek-first)

- Default provider: `deepseek`. Set `DEEPSEEK_API_KEY` (or `DEEPSEEK_API`) and optionally `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`.
- Other providers: `LLM_PROVIDER=openai` with `OPENAI_API_KEY`/`OPENAI_MODEL`, or `LLM_PROVIDER=vllm` with `VLLM_BASE_URL`/`VLLM_MODEL`.
- Tunables: `LLM_TIMEOUT` (seconds), `LLM_MAX_RETRIES` (default 3).

## Quickstart

Generate two general tasks about retrieval:

```bash
agent_gem --agent-type general_agent --topic "retrieval-augmented QA" --count 2 --sandbox-root sandbox/raq
```
=======
| Variable | Description | Default |
| --- | --- | --- |
| `LLM_PROVIDER` | `vllm`, `openai`, or `deepseek` | `vllm` |
| `VLLM_BASE_URL` | vLLM OpenAI-compatible base URL | `http://localhost:8000/v1` |
| `VLLM_MODEL` | Local model name | `local-model` |
| `VLLM_API_KEY` | Optional if auth enabled | empty |
| `VOLCANO_BASE_URL` | Deepseek API base URL | `https://ark.cn-beijing.volces.com/api/v3` |
| `VOLCANO_MODEL` | Deepseek model ID | `deepseek-v3-2-251201` |
| `VOLCANO_API_KEY` | Deepseek API key | empty |
| `OPENAI_BASE_URL` | OpenAI-compatible base URL | `https://api.openai.com/v1` |
| `OPENAI_MODEL` | Remote model name | `gpt-4o-mini` |
| `OPENAI_API_KEY` | API key for OpenAI/compatible service | empty |
| `LLM_TIMEOUT` | Request timeout seconds | `60` |
| `SANDBOX_FUSION_URL` | SandboxFusion service URL | `http://localhost:8080` |
| `SANDBOX_FUSION_TIMEOUT` | SandboxFusion request timeout | `30` |
| `SANDBOX_FUSION_LANGUAGE` | Default programming language | `python` |
| `DOCKER_IMAGE` | Docker image for code execution | `python:3.11-slim` |
| `DOCKER_TIMEOUT` | Docker execution timeout | `30` |

## Quickstart

### Prerequisites

1. **Docker** (required for secure code execution):
```bash
# Check if Docker is running
docker ps

# If not running, start Docker service
sudo systemctl start docker  # Linux
# Or start Docker Desktop
```

2. **SandboxFusion** (optional, but recommended):
```bash
# Deploy SandboxFusion service
docker run -it -p 8080:8080 volcengine/sandbox-fusion:server-20250609
```

### Basic Usage

**By default, Docker and SandboxFusion are enabled for secure execution:**

```bash
# Configure LLM (Deepseek example)
export LLM_PROVIDER=deepseek
export VOLCANO_MODEL=deepseek-v3-2-251201
export VOLCANO_BASE_URL=https://ark.cn-beijing.volces.com/api/v3
export VOLCANO_API_KEY=your-api-key

# Optional: Configure SandboxFusion (if service is running)
export SANDBOX_FUSION_URL=http://localhost:8080

# Run with default secure execution (Docker + SandboxFusion)
python -m general_agent \
  --category "travel itinerary planning" \
  --sandbox ./sandbox/travel \
  --rounds 1
```

**Or use the provided test script:**
```bash
bash test_deepseek.sh
```

### Disable Security Features (Not Recommended)

If you need to disable Docker or SandboxFusion:

```bash
# Disable Docker
python -m general_agent --category "..." --sandbox ./sandbox/travel --no-docker

# Disable SandboxFusion
python -m general_agent --category "..." --sandbox ./sandbox/travel --no-sandbox-fusion

# Disable both (use local execution - less secure)
python -m general_agent --category "..." --sandbox ./sandbox/travel --no-docker --no-sandbox-fusion
```

Generated DB and tasks are saved under `sandbox/travel`.
>>>>>>> main

Run a code-focused batch:

```bash
agent_gem --agent-type code_agent --topic "python data pipelines" --count 1 --difficulty Hard
```

## Project Layout

- `agent_gem/config.py`, `agent_gem/llm.py`: DeepSeek-first client with retries.
- `agent_gem/core/`: task schema, validation, scoring, and helpers.
- `agent_gem/agents/`: agent implementations (search, code, code interpreter, general).
- `agent_gem/env_generator/`: orchestrator for routing requests and prioritizing tasks.
- `agent_gem/sandbox/`: isolation and persistence of generated tasks.

## Output

Tasks are saved as `sandbox/<agent>/<slug>/task.json` with task schema, reference solution, verification snippet, and metadata. Use these sandboxes directly for RL rollouts or dataset curation.
