# Triple Generation Configuration Guide

This guide explains how to use the YAML configuration system for the Triple Generator and Code Agent.

## Configuration Files

1. **triple_generation.yaml** - Controls triple (repo, file, function) generation from GitHub
2. **code_agent.yaml** - Controls task generation parameters for CodeAgent

## Quick Start

### Using with CLI

```bash
# Generate triples and tasks in one command with configs
python -m agent_gem batch \
    --triple-config config/triple_generation.yaml \
    --agent-config config/code_agent.yaml \
    --generate-triples

# Use existing triples with custom agent config
python -m agent_gem batch \
    --triples triple_cache/function_triples.json \
    --agent-config config/code_agent.yaml

# Override config values with command line
python -m agent_gem batch \
    --agent-config config/code_agent.yaml \
    --num-tasks 20 \
    --difficulty 3
```

### Using in Python

```python
from agent_gem.config import TripleGenerationConfig, CodeAgentConfig
from agent_gem.agents.triple_generator import TripleGenerator
from agent_gem.agents.code_agent import CodeAgent

# Load configurations
triple_config = TripleGenerationConfig.from_file("config/triple_generation.yaml")
agent_config = CodeAgentConfig.from_file("config/code_agent.yaml")

# Use with TripleGenerator
generator = TripleGenerator(
    github_token="your_token",
    config_path="config/triple_generation.yaml"
)
triples = generator.generate_triples()

# Use with CodeAgent
agent = CodeAgent.from_env(
    taskdb_root=agent_config.taskdb_root,
    max_tokens=agent_config.max_tokens,
    temperature=agent_config.temperature
)
```

## Configuration Files

### 1. triple_generation.yaml

Controls which repositories are selected from GitHub and how functions are extracted.

### 1. Repository Filter (`repo_filter`)

Controls which repositories are selected from GitHub:

```yaml
repo_filter:
  stars:
    min: 100      # Minimum star count
    max: 10000    # Maximum star count
  
  size:
    max: 51200    # Maximum size in KB (50MB)
  
  updated_within_days: 730  # Recently updated (within 2 years)
  
  topics:         # GitHub topics to search for
    - library
    - tool
    - cli
  
  exclude_keywords:  # Keywords that mark repos as "heavy"
    - tensorflow
    - pytorch
    - gpu
  
  exclude_topics:    # Topics to exclude
    - machine-learning
    - deep-learning
  
  max_repos: 50    # Maximum repos to analyze
```

### 2. File Filter (`file_filter`)

Controls which files are selected from each repository:

```yaml
file_filter:
  # Focus only on main package (e.g., numpy/ in numpy repo)
  focus_main_package: true
  
  # Directories to skip
  exclude_directories:
    - tests
    - docs
    - examples
  
  # Files to skip
  exclude_files:
    - __init__.py
    - setup.py
    - config.py
  
  # Line count requirements
  line_count:
    min: 50          # Minimum lines
    max: 2000        # Maximum lines
    optimal_min: 100 # Optimal range start
    optimal_max: 500 # Optimal range end
  
  min_functions: 2   # Minimum number of functions
  
  # Scoring weights
  scoring:
    line_count_optimal: 20    # Score for optimal size
    function_count: 10        # Score per function
    function_count_max: 50    # Max score from functions
    docstring: 5              # Score per docstring
    not_test: 15              # Bonus for non-test files
    utility_penalty: -10      # Penalty for utility files
  
  utility_keywords:  # Keywords that identify utility files
    - util
    - helper
  
  max_files_per_repo: 10  # Maximum files to extract per repo
```

### 3. Function Filter (`function_filter`)

Controls which functions are extracted from files:

```yaml
function_filter:
  skip_private: true  # Skip functions starting with _
  
  # Line count requirements
  line_count:
    min: 5           # Too small
    max: 200         # Too large
    optimal_min: 10  # Best range start
    optimal_max: 100 # Best range end
  
  max_arguments: 5   # Maximum function parameters
  
  min_quality_score: 0.3  # Minimum quality (0-1)
  
  # Scoring weights (should sum to ~1.0)
  scoring:
    line_count_optimal: 0.3   # Good size
    has_docstring: 0.2        # Documented
    has_type_hints: 0.1       # Type annotated
    has_control_flow: 0.2     # Has if/loop/try
    has_return: 0.1           # Returns value
    arg_count_ok: 0.1         # Not too many args
```

### 4. Output Configuration (`output`)

Controls overall generation parameters:

```yaml
output:
  target_triples: 100       # Number of triples to generate
  quality_threshold: 0.5    # Minimum quality (0-1)
  cache_dir: triple_cache   # Where to cache data
  refresh_repos: false      # Re-fetch repos from GitHub
```

## Example Use Cases

### 1. Find Small, Well-Documented Libraries

```yaml
repo_filter:
  stars: {min: 500, max: 5000}
  size: {max: 20480}  # 20MB
  topics: [library, utility]

file_filter:
  line_count: {min: 100, max: 500}
  scoring:
    docstring: 10  # Increase docstring weight

function_filter:
  min_quality_score: 0.6  # Higher quality bar
  scoring:
    has_docstring: 0.3  # Prioritize documentation
```

### 2. Focus on CLI Tools

```yaml
repo_filter:
  topics: [cli, command-line, terminal]
  exclude_keywords: []  # Remove ML exclusions

file_filter:
  focus_main_package: true
  exclude_directories: [tests, docs, examples, benchmarks]

function_filter:
  line_count: {optimal_min: 20, optimal_max: 150}
```

### 3. Search Entire Repos (Not Just Main Package)

```yaml
file_filter:
  focus_main_package: false  # Look at all Python files
  max_files_per_repo: 20     # Increase file limit
```

### 4. Get More Triples Quickly

```yaml
repo_filter:
  max_repos: 100  # Analyze more repos

file_filter:
  min_functions: 1  # Lower bar
  max_files_per_repo: 15

function_filter:
  min_quality_score: 0.2  # Lower quality threshold

output:
  target_triples: 500
  quality_threshold: 0.3
```

## 2. code_agent.yaml

Controls task generation parameters for CodeAgent.

#### Key Sections

**task_generation** - Task creation parameters
```yaml
task_generation:
  difficulty: 2           # 1-5, controls task complexity
  max_tokens: 8000        # Max tokens for generation
  temperature: 0.7        # Creativity (0.0-1.0)
  max_retries: 3          # Retry failed generations
```

**batch_processing** - Batch workflow parameters
```yaml
batch_processing:
  num_tasks: 10           # Tasks per batch
  taskdb_root: taskdb     # Output directory
  skip_errors: true       # Continue on errors
```

**sandbox** - Execution environment
```yaml
sandbox:
  timeout: 120            # Seconds per test
  memory_limit: 512       # MB
```

## Parameter Precedence

When running with both config file and command-line arguments:

**CLI args > Config file > Defaults**

Example:
```bash
# Config has num_tasks: 10
# Command line specifies --num-tasks 20
# Result: Uses 20 (CLI takes precedence)
python -m agent_gem batch --agent-config config/code_agent.yaml --num-tasks 20
```

## Command Line Usage

## Command Line Usage

### Using batch command

```bash
# Generate triples and create tasks with both configs
python -m agent_gem batch \
    --generate-triples \
    --triple-config config/triple_generation.yaml \
    --agent-config config/code_agent.yaml \
    --num-tasks 10 \
    -v

# Use existing triples, custom agent config
python -m agent_gem batch \
    --triples triple_cache/function_triples.json \
    --agent-config config/code_agent.yaml

# Override config values on command line
python -m agent_gem batch \
    --agent-config config/code_agent.yaml \
    --num-tasks 20 \
    --difficulty 4 \
    --temperature 0.8

# Use shell script for triple generation
./run_generate_triples.sh
```

### Direct Python usage

```python
from agent_gem.config import TripleGenerationConfig, CodeAgentConfig
from agent_gem.agents.triple_generator import TripleGenerator
from agent_gem.agents.code_agent import CodeAgent

# Load configs
triple_config = TripleGenerationConfig.from_file("config/triple_generation.yaml")
agent_config = CodeAgentConfig.from_file("config/code_agent.yaml")

# Generate triples
generator = TripleGenerator(
    github_token="your_token",
    config_path="config/triple_generation.yaml"
)
triples = generator.generate_triples()

# Create tasks
agent = CodeAgent.from_env(
    taskdb_root=agent_config.taskdb_root,
    max_tokens=agent_config.max_tokens,
    temperature=agent_config.temperature
)

for triple in triples[:agent_config.num_tasks]:
    task = agent.generate_task(triple)
```

## Tips

### Triple Generation
1. **Start Conservative**: Begin with restrictive filters, then relax them if needed
2. **Monitor Quality**: Check the quality_score distribution in your results
3. **Use Cache**: Set `refresh_repos: false` to speed up repeated runs
4. **Iterate**: Adjust scoring weights based on your results
5. **Balance**: Higher thresholds = fewer but better triples

### Task Generation
1. **Match Difficulty**: Align difficulty with your use case (1=simple, 5=complex)
2. **Token Budget**: Increase max_tokens for complex tasks with detailed explanations
3. **Temperature**: Lower (0.3-0.5) for consistency, higher (0.7-0.9) for creativity
4. **Batch Size**: Start with small batches (5-10) to test parameters

## Configuration Validation

The system validates your configuration on load:
- Checks for required sections
- Validates numeric ranges (min < max)
- Ensures thresholds are in valid range (0-1)

If validation fails, you'll see a clear error message.

## Advanced: Multiple Configurations

Create different config files for different purposes:

```
config/
├── triple_generation.yaml       # Default/example
├── high_quality.yaml           # Strict filters
├── cli_tools.yaml              # CLI-specific
└── web_frameworks.yaml         # Web-specific
```

Then use them as needed:

```python
# Generate different types of triples
high_quality = TripleGenerator(config_path="config/high_quality.yaml")
cli_triples = TripleGenerator(config_path="config/cli_tools.yaml")
```
