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

```bash
cp .env.example .env
# then edit .env with your API keys/models
```

## Model Configuration (DeepSeek-first)

- Default provider: `deepseek`. Set `DEEPSEEK_API_KEY` (or `DEEPSEEK_API`) and optionally `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`.
- Other providers: `LLM_PROVIDER=openai` with `OPENAI_API_KEY`/`OPENAI_MODEL`, or `LLM_PROVIDER=vllm` with `VLLM_BASE_URL`/`VLLM_MODEL`.
- Tunables: `LLM_TIMEOUT` (seconds), `LLM_MAX_RETRIES` (default 3).

## Sandbox Configuration

Build the image locally:

```bash
# change the base image in Dockerfile.server
docker build -f ./sandbox_fusion/scripts/Dockerfile.server -t code_sandbox:server .
docker run -d --rm --privileged --it \
  -v "$PWD/sandbox_fusion":/root/sandbox \
  -p 8080:8080 code_sandbox:server
```

## Quickstart

Generate two general tasks about retrieval:

```bash
agent_gem --agent-type general_agent --topic "retrieval-augmented QA" --count 2 --sandbox-root sandbox/raq
```

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

Tasks are saved as `sandbox/<agent>/task-<task-id>/task.json` with task schema, reference solution, verification snippet, and metadata. Use these sandboxes directly for RL rollouts or dataset curation.