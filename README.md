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

## Code Agent

The Code Agent generates function-implementation tasks by mining GitHub repositories. It extracts real Python functions, creates implementation challenges, and generates test suites with validation.

### Workflow

1. **Triple Generation**: Mines GitHub repos → filters quality files → extracts suitable functions
2. **Task Creation**: Removes function body → generates task description → creates test suite
3. **Validation**: Verifies tests pass with original code → validates in clean environment
4. **Output**: Saves task + tests + solution to `{taskdb_root}/task_{id}/`

### Usage

See [examples/run_code_agent.sh](examples/run_code_agent.sh) for a complete example:

```bash
# Batch mode: generate from pre-computed triples
python -m agent_gem code_synthesize --config config/code_agent.yaml

# Single mode: target specific function
python -m agent_gem code_synthesize \
  --config config/code_agent.yaml \
  --repo numpy/numpy \
  --file numpy/matlib.py \
  --function identity
```

Configuration (scoring weights, filters, test counts) is in [config/code_agent.yaml](config/code_agent.yaml).

## Search Agent

The search agent extracts real information from web, sample entity and generate questions for search agent training.

### Pipeline

The **Search Agent** pipeline is as follows:

1. **Domain & Entity Sampling**: You can either specify domains directly or allow the LLM to generate domains. For each domain, the agent samples relevant entities, extracting each entity's name, description, and domain.

2. **Context Extraction & Question Generation**: For every sampled entity, the agent collects relevant multi-hop context (information requiring multiple reasoning steps) using searching and browsing tools. Then, the LLM generates question-answer pairs based on the gathered context, ensuring the questions require reasoning over several pieces of information.

3. **Multi-Agent Answer Generation**: Multiple answer candidates are generated for each question using different agents (with varied system prompts and temperatures) to simulate diverse answering strategies.

4. **Filtering by Verification**: Only samples where the ground-truth answer is verified correct and *all* answer candidates are verified incorrect are retained. This helps curate harder, high-quality datasets for robust evaluation.

### How to run

Optional hyperparameters for running the Search Agent:

- `--domain`: Comma-separated list of target domains (e.g. `medicine,sports,science`). If not provided, the agent will use the LLM to generate domains.
- `--num_domains`: Number of domains to generate (if `--domain` not set). Default: 3.
- `--num_entities_each_domain`: Number of entities to sample per domain. Default: 10.
- `--num_tasks_each_entity`: Number of question-answer pairs to generate per entity. Default: 3.
- `--search_depth`: How many pages deep to search (higher gets more obscure data). Default: 2.
- `--search_breadth`: Number of results to retrieve per page. Default: 5.

Example usage:

```bash
python -m agent_gem search_synthesize \
  --num_domains 10 \
  --num_entities_each_domain 8 \
  --num_tasks_each_entity 2 \
  --search_depth 3 \
  --search_breadth 8 \
  --output ./results.json \
  --require_all_incorrect
```

Minimum requirements:
- Set `SERPER_API_KEY` (for search) and `JINA_API_KEY` (for content extraction) in your environment.
- Set `SEARCH_CACHE_PATH` for persistent caching of search queries.

## Project Layout

- `agent_gem/config.py`, `agent_gem/llm.py`: DeepSeek-first client with retries.
- `agent_gem/core/`: task schema, validation, scoring, and helpers.
- `agent_gem/agents/`: agent implementations (search, code, code interpreter, general).
- `agent_gem/env_generator/`: orchestrator for routing requests and prioritizing tasks.
- `agent_gem/sandbox/`: isolation and persistence of generated tasks.

## Output

Tasks are saved as `sandbox/<agent>/task-<task-id>/task.json` with task schema, reference solution, verification snippet, and metadata. Use these sandboxes directly for RL rollouts or dataset curation.