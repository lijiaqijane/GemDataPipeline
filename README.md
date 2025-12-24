# Agent Task Pipeline

Automated training data generation pipeline that creates high-quality programming tasks from GitHub repositories.

## Overview

Agent Task Pipeline is an intelligent task generation system that automatically generates programming tasks and test data from real GitHub codebases. The system supports multiple generation modes, including function implementation tasks, bug fix tasks, and more.

## Core Features

- 🤖 **Intelligent Task Generation**: Automatically generate high-quality programming tasks using LLMs
- 🔍 **GitHub Integration**: Extract code snippets from real open-source projects
- 🧪 **Complete Testing**: Auto-generate verification tests and evaluation scripts
- 🐳 **Sandbox Execution**: Safely execute and verify code using SandboxFusion
- 📦 **Batch Processing**: Support for large-scale batch task generation
- ⚙️ **Flexible Configuration**: Control the entire generation pipeline via YAML configuration

## Code Agent Workflow

Code Agent is the core task generator that automatically creates function implementation tasks from GitHub repositories.

### Complete Pipeline

```
1. Triple Generation (Optional)
   ├─ Search GitHub for qualifying repositories
   ├─ Analyze repository structure and filter files
   ├─ Extract high-quality functions
   └─ Generate (repo, file, function) triples

2. Task Generation
   ├─ Clone repository to sandbox
   ├─ Analyze dependencies and environment
   ├─ Delete target function
   ├─ Generate function implementation task description
   ├─ Create test cases
   └─ Verify test executability

3. Task Saving
   ├─ Save task definition (task.json)
   ├─ Save solution (solution.txt)
   ├─ Save test information (test_info.txt)
   ├─ Save generation logs (logs/)
   └─ Clone repository locally (repo/)
```

### Detailed Steps

#### Stage 0: Preparation
- Start SandboxFusion service (if auto-start is configured)
- Initialize task directory and logging system

#### Stage 1: Triple Generation (Batch Mode)
When no specific `(repo, file, function)` is specified, the system automatically generates triples:

1. **Repository Filtering**
   - Filter by star count, size, update time
   - Exclude heavy ML/DL projects
   - Filter by topic tags (library, tool, cli, etc.)

2. **File Scoring**
   - Focus on main package directories
   - Exclude tests, docs, example code
   - Score by line count, function count, documentation completeness

3. **Function Extraction**
   - Skip private functions (starting with `_`)
   - Score by complexity, documentation, type hints
   - Select high-quality functions (quality score > 0.3)

#### Stage 2: Task Generation

1. **Repository Setup**
   ```bash
   # Clone repository in sandbox
   git clone --depth 1 <repo_url> /workspace/repo
   
   # Analyze language and dependencies
   - Detect project language (Python/JS/Java, etc.)
   - Extract dependency list (requirements.txt, package.json, etc.)
   - Identify test framework (pytest, unittest, jest, etc.)
   ```

2. **Function Deletion**
   ```python
   # Use AST to locate target function
   # Use git diff to generate deletion patch
   deletion_patch = git_diff(function_location)
   ```

3. **Issue Generation**
   - LLM generates task description from function signature and docstring
   - Includes implementation hints and edge case explanations
   - Creates GitHub Issue-style task description

4. **Test Generation**
   ```python
   # LLM generates test cases based on original function
   # Covers normal cases, edge cases, error handling
   test_code = generate_test(function_spec, original_code)
   
   # Verify tests pass on clean code
   apply_test_patch()
   run_tests() # Should PASS
   ```

5. **Task Packaging**
   - Save all generated files
   - Clone complete repository locally
   - Record metadata and generation logs

## Usage Guide

### Prerequisites

### Prerequisites

```bash
# 1. Install dependencies
pip install -e .

# 2. Start SandboxFusion (if not using auto_start)
# Refer to SandboxFusion documentation

# 3. Configure environment variables
export GITHUB_TOKEN=your_github_token          # GitHub API access
export VOLCANO_API_KEY=your_volcano_api_key    # LLM API
export SANDBOX_IMAGE=your_image
export SANDBOX_CMD=your_cmd
export SANDBOX_FUSION_URL=you_url
```

### Configuration File

The configuration file `config/code_agent.yaml` controls the entire generation pipeline:

### Usage Examples

#### 1. Batch Generation (Auto Triple)

Automatically discover and generate tasks from GitHub:

```bash
# Generate 10 tasks (using default configuration)
python -m agent_gem batch

# Use custom configuration
python -m agent_gem batch -c config/code_agent.yaml

# Enable verbose logging
python -m agent_gem batch -c config/code_agent.yaml -v
```

The system will:
1. Search GitHub for qualifying repositories
2. Extract high-quality functions to generate triples
3. Generate complete tasks for each triple
4. Save to `taskdb/code_agent/task_<uuid>/`

#### 2. Targeted Generation

Generate tasks for specific repository, file, and function:

```bash
# Generate task for numpy.matlib.identity
python -m agent_gem batch \
  --repo numpy/numpy \
  --file numpy/matlib.py \
  --function identity

# Use full URL
python -m agent_gem batch \
  --repo https://github.com/requests/requests \
  --file requests/api.py \
  --function get
```

#### 3. Using Shell Script (Recommended)

Use `run_code_agent.sh` for better experience:

```bash
# Batch generation
./run_code_agent.sh

# Targeted generation
./run_code_agent.sh numpy/numpy numpy/matlib.py identity

# Script automatically manages sandbox lifecycle
```

### Generated Output

Each task is saved in a separate directory:

```
taskdb/code_agent/task_<uuid>/
├── logs/                          # Generation process logs
│   ├── 00_repo_directory_listing.txt
│   ├── 01_repo_metadata.json
│   ├── 02_extracted_source_files.json
│   ├── llm_response_01.txt
│   ├── llm_response_02.txt
│   └── ...
├── task.json                      # Task information
└── repo/                          # Cloned repository (for training environment)
```

## Configuration Reference

### Triple Generation Configuration

| Config Item | Description | Default |
|-------------|-------------|---------|
| `repo_filter.stars.min/max` | Repository star count range | 100-10000 |
| `repo_filter.size.max` | Repository size limit (KB) | 10240 |
| `repo_filter.topics` | Filter by topic tags | [library, tool, cli] |
| `file_filter.line_count` | File line count range | 50-2000 |
| `function_filter.min_quality_score` | Function quality threshold | 0.3 |
| `output.target_triples` | Target number of triples | 100 |

### Task Generation Configuration

| Config Item | Description | Default |
|-------------|-------------|---------|
| `taskdb_root` | Task save root directory | taskdb/code_agent |
| `difficulty` | Task difficulty (1-3) | 1 |
| `max_tokens` | LLM max tokens | 8192 |
| `temperature` | LLM sampling temperature | 0.7 |
| `save_logs` | Save generation logs | true |

### Batch Processing Configuration

| Config Item | Description | Default |
|-------------|-------------|---------|
| `num_tasks` | Number of batch tasks | 10 |
| `skip_errors` | Continue on error | false |
| `max_retries` | Maximum retry attempts | 5 |


## Project Structure

```
Agent_Task_Pipeline/
├── agent_gem/              # Core code
│   ├── agents/            # Agent implementations
│   │   ├── code_agent.py          # Code Agent
│   │   ├── triple_generator.py    # Triple generator
│   │   └── feature_requester_generator.py  # Task generator
│   ├── config.py          # Configuration management
│   ├── cli.py             # Command-line interface
│   └── sandbox/           # Sandbox integration
├── config/                # Configuration files
│   └── code_agent.yaml   # Code Agent configuration
├── taskdb/                # Generated task database
├── triple_cache/          # Triple cache
└── run_code_agent.sh      # Launch script
```

## License

[Add license information]

## Contributing

Issues and Pull Requests are welcome!

## Contact

[Add contact information]
