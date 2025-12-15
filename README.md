# General Agent Synthesis

Lightweight automatic environment and task synthesis agent. It generates specialized tools, tasks, and verification logic for a given topic, verifying them in a secure execution environment.

## Features

- **Automatic Tool Synthesis**: Generates specialized Python tools based on topic data.
- **Task Generation**: Creates challenging tasks with corresponding solutions and verifiers.
- **Secure Execution**: Supports **SandboxFusion** (remote execution) and **Docker** for running untrusted code.
- **Iterative Refinement**: Automatically increases task difficulty and repairs failing tasks.

## Quick Start

### Run Demo (No setup required)

To see the agent workflow in action without configuring external services (LLM/Docker/Sandbox), run the self-contained demo:

```bash
python run_demo.py
```

### Run with Real Services

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: requests is the main dependency)*

2. **Configure Environment**
   Set environment variables for your LLM provider (vLLM, OpenAI, or Deepseek/Volcano) and Sandbox.

   ```bash
   # Example for Deepseek/Volcano
   export LLM_PROVIDER=deepseek
   export VOLCANO_API_KEY="your-api-key"
   
   # Example for SandboxFusion (if running locally)
   export SANDBOX_FUSION_URL="http://localhost:8080"
   ```

3. **Run Synthesis**
   ```bash
   # Run for a specific category
   python -m general_agent --category "travel planning" --sandbox ./sandbox/travel
   
   # Or use the helper script
   ./run.sh
   ```

## Project Structure

- `general_agent/`: Main package source
  - `synthesis.py`: Core logic for environment and task synthesis.
  - `executor.py`: SandboxFusion execution environment client.
  - `tools.py`: Tool definitions (Bash, Search, etc.).
  - `llm.py`: Unified LLM client.
- `run_demo.py`: Mocked demonstration script.
- `examples/`: Usage examples.
- `tests/`: Integration tests.

## License

MIT
