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

## CodeAgent 任务构建流程

### 概述

CodeAgent 用于生成代码仓库 Bug Fix 任务。它会自动克隆目标仓库，分析代码，生成合理的安全漏洞或逻辑错误，并配套生成 Issue、Pull Request 描述和测试用例。所有操作在 SandboxFusion 隔离环境中执行，确保安全性。

### 10 步工作流程

1. **启动 SandboxFusion 容器**
   - 使用自定义 Docker 命令启动代码沙盒
   - 自动检测端口并更新连接 URL
   - 生命周期管理（启动/停止）

2. **克隆代码仓库**
   - 在沙盒中 git clone 目标仓库
   - 支持公开和私有仓库
   - 记录仓库元数据（语言、结构等）

3. **分析代码仓库**
   - 扫描 Python/JavaScript/其他语言源文件
   - 提取函数、类、模块定义
   - 生成代码结构摘要

4. **抽取目标源文件**
   - 选择适合注入 Bug 的文件
   - 优先选择核心功能模块
   - 提取函数签名和代码片段

5. **LLM 生成 Bug**
   - 使用 DeepSeek v3 生成合理的 Bug
   - 支持类型：security（安全）、logic（逻辑）、performance（性能）
   - 可配置 `max_tokens` 和 `temperature` 参数
   - 输出：Bug 描述、受影响文件、修复提示

6. **LLM 生成 Issue/PR**
   - 自动生成 GitHub Issue 格式的问题报告
   - 包含：问题描述、重现步骤、期望行为
   - 生成对应的 Pull Request 描述
   - 包含：修复摘要、安全影响、测试说明

7. **LLM 生成测试用例**
   - 生成至少 2 个测试用例
   - 测试特性：在 Buggy 代码上失败，在 Fixed 代码上通过
   - 支持多种测试框架（pytest、jest 等）
   - 包含完整的测试代码和断言说明

8. **测试验证**（可选）
   - 在沙盒中验证测试的有效性
   - 失败时使用降级策略（继续使用所有测试）
   - 记录验证结果

9. **生成环境设置**
   - 提取项目依赖（requirements.txt、package.json）
   - 生成安装命令
   - 配置测试运行环境

10. **创建并持久化任务包**
    - 生成最终 TaskPackage（符合 Pydantic schema）
    - 包含：task.json、solution.txt、context.json
    - 保存到 `{repo_name}_taskdb/code_agent/task-{uuid}/`
    - 支持 JSON 和 JSONL 格式

### 核心特性

#### 可配置的 LLM 参数
```bash
# 增加生成长度（默认 2000）
MAX_TOKENS=4000 bash ./run_code_agent.sh

# 调整采样温度（默认 0.7）
TEMPERATURE=0.5 bash ./run_code_agent.sh

# 两者结合
MAX_TOKENS=3000 TEMPERATURE=0.8 bash ./run_code_agent.sh
```

#### 完整的中间过程日志
每次任务生成会创建 7 个中间 JSON 文件：
- `01_repo_metadata.json` - 仓库元数据
- `02_extracted_source_files.json` - 抽取的源文件列表
- `03_generated_bug.json` - 生成的 Bug 定义
- `04_issue_pr.json` - Issue 和 PR 内容
- `05_test_cases.json` - 完整测试代码
- `06_test_validation_results.json` - 测试验证结果
- `07_final_task_package.json` - 最终任务包

以及 4 个原始 LLM 响应日志：
- `llm_response_01.txt` - Bug 生成响应
- `llm_response_02.txt` - Issue/PR 生成响应
- `llm_response_03.txt` - 补充响应
- `llm_response_04.txt` - 测试用例生成响应

#### 健壮的错误处理
- JSON 解析失败时使用降级策略（extract_json_from_response）
- 处理 Markdown 代码块包裹的 JSON
- LLM 生成失败时使用默认值
- 测试验证失败时继续流程

#### 自定义 Docker 支持
```bash
# 使用自定义 Docker 启动命令
SANDBOX_CMD="docker run -d --rm --privileged -p 8080:8080 code_sandbox:server make run-online"
```

### 使用示例

#### 为 pandas 仓库生成任务
```bash
bash ./run_code_agent.sh
```

#### 批量生成多个任务
```bash
NUM_TASKS=10 bash ./run_code_agent.sh
```

#### 为其他仓库生成
```bash
REPO_URL=https://github.com/numpy/numpy bash ./run_code_agent.sh
```

#### 使用 Python 脚本直接调用
```bash
python run_pandas_generation.py \
  --max-tokens 3000 \
  --temperature 0.6 \
  --repo-url https://github.com/pandas-dev/pandas \
  --difficulty 2
```

### 输出结构

任务保存在 `{repo_name}_taskdb/code_agent/task-{uuid}/`：

```
task-a0bde1df-00b9-437a-932e-ab5bf3537fa1/
├── {uuid}.json              # 任务元数据
├── {uuid}.jsonl             # JSONL 格式任务数据
└── training_data.json       # 完整训练数据（含 Bug、修复、测试）
```

中间输出保存在 `output/{uuid}/`：

```
output/a0bde1df-00b9-437a-932e-ab5bf3537fa1/
├── 01_repo_metadata.json
├── 02_extracted_source_files.json
├── 03_generated_bug.json
├── 04_issue_pr.json
├── 05_test_cases.json
├── 06_test_validation_results.json
├── 07_final_task_package.json
├── llm_response_01.txt
├── llm_response_02.txt
├── llm_response_03.txt
└── llm_response_04.txt
```

### 技术架构

- **代码分析**: `code_repo_analyzer.py` - 仓库结构分析和元数据提取
- **Bug 生成**: `code_bug_generator.py` - LLM 驱动的 Bug/Issue/PR 生成
- **测试生成**: `code_test_validator.py` - 测试用例生成和验证
- **沙盒集成**: `sandbox_integration.py` - SandboxFusion 生命周期管理
- **主控制器**: `code_agent.py` - 10 步工作流编排

### 性能指标

- **每个任务生成时间**: 约 3 分钟
- **每个任务数据量**: 约 128 KB（包含中间输出和最终任务）
- **吞吐量**: 约 20 任务/小时
- **批量生成 100 个任务**: 约 5 小时

## Output

Tasks are saved as `sandbox/<agent>/task-<task-id>/task.json` with task schema, reference solution, verification snippet, and metadata. Use these sandboxes directly for RL rollouts or dataset curation.