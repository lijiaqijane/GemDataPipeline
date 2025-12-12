# General Agent Synthesis

Lightweight automatic environment and task synthesis project with:
- Local vLLM or any OpenAI-compatible API
- Retrieval + sandbox tools to seed the database automatically
- Verifiable tasks with auto-repair when solutions fail validation
- Both CLI and Python API entrypoints

## Install
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Model Configuration
Environment variables to switch between local vLLM and OpenAI-compatible endpoints:

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

### Python API
```python
from pathlib import Path
from general_agent import LLMClient, EnvironmentSynthesizer

llm = LLMClient.from_env()
synth = EnvironmentSynthesizer(llm)
bundles = synth.synthesize(category="travel itinerary planning", sandbox=Path("sandbox/travel"), rounds=3)
for b in bundles:
    print(b.name, b.difficulty)
```

## Project Layout
- `general_agent/llm.py`: unified client for vLLM & OpenAI-compatible APIs
- `general_agent/tools.py`: sandbox tools (restricted bash, DuckDuckGo search) and registry
- `general_agent/database.py`: local JSON storage
- `general_agent/synthesis.py`: main pipeline for environment/tool/task synthesis and validation
- `general_agent/cli.py`: CLI entrypoint

## Workflow
1. **Build context**: create sandbox dir, load DB, inject default tools.
2. **Seed database**: fetch seed data with search tool; LLM structures and stores it.
3. **Synthesize tools**: LLM proposes task-specific tools and registers them.
4. **Generate tasks**: produce solution and verifier; auto-run and repair on failure.
5. **Refine**: iterate difficulty over multiple rounds.
6. **Persist**: save tools, DB, and tasks into `tasks.json`.

## Notes
- DuckDuckGo API is used by default; network access is required.
- Solutions/verifiers run in a constrained environment; still prefer running inside a controlled sandbox directory.

