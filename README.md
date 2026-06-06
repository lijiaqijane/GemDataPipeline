# AgentGEM: Agent Generative Environment Maker

High-performance generator for RL-ready agentic tasks. It ships four pipelines (Search, Code, Code Interpreter, General) and packages each task into isolated sandboxes for large-scale training.

## Install

## Model Configuration (DeepSeek-first)

- Default provider: `deepseek`. Set `DEEPSEEK_API_KEY` (or `DEEPSEEK_API`) and optionally `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL`.
- Other providers: `LLM_PROVIDER=openai` with `OPENAI_API_KEY`/`OPENAI_MODEL`, or `LLM_PROVIDER=vllm` with `VLLM_BASE_URL`/`VLLM_MODEL`.
- Tunables: `LLM_TIMEOUT` (seconds), `LLM_MAX_RETRIES` (default 3).

## Sandbox Configuration

Build the image locally:

```bash
docker build -f ./sandbox_fusion/scripts/Dockerfile.server -t code_sandbox:server .
docker run -d --rm --privileged --it \
  -v "$PWD/sandbox_fusion":/root/sandbox \
  -p 8080:8080 code_sandbox:server
```

## FireCrawl Configuration
```bash

```

## Quickstart

Generate general tasks:

```bash
bash run.sh
```

## Output

Tasks are saved as `sandbox/<agent>/task-<task-id>/` with task schema, reference solution, verification snippet, and metadata. Use these sandboxes directly for RL rollouts or dataset curation.
