#!/usr/bin/env bash
# CodeAgent pandas 数据生成脚本：为 pandas 仓库生成训练任务
# 环境变量已通过外部提前设置，脚本负责启动 CodeAgent 和沙盒的自动化

set -euo pipefail

# ---------- 沙盒配置 ----------
# Docker 镜像（与 .env 或环境变量一致）
export SANDBOX_IMAGE="${SANDBOX_IMAGE:-code_sandbox:server}"

# Docker 启动命令（沙盒会自动启动和关闭）
export SANDBOX_CMD="${SANDBOX_CMD:-docker run -d --rm --privileged -p 8080:8080 code_sandbox:server make run-online}"

# 沙盒服务 URL
export SANDBOX_URL="${SANDBOX_URL:-http://localhost:8080}"

# ---------- CodeAgent 生成参数 ----------
# 生成的任务保存目录
TASKDB_ROOT="${TASKDB_ROOT:-pandas_taskdb}"

# 仓库信息
REPO_URL="${REPO_URL:-https://github.com/pandas-dev/pandas}"
DIFFICULTY="${DIFFICULTY:-2}"
AGENT_TYPE="${AGENT_TYPE:-code_agent}"

# 生成任务数量
NUM_TASKS="${NUM_TASKS:-1}"

# 日志级别
LOG_LEVEL="${LOG_LEVEL:-INFO}"

# LLM 生成参数
MAX_TOKENS="${MAX_TOKENS:-4096}"  # 最大生成长度
TEMPERATURE="${TEMPERATURE:-0.7}"  # 采样温度 (0.0-1.0)

# ---------- 执行主程序 ----------
# 运行 CodeAgent 为 pandas 生成训练任务
PYTHONUNBUFFERED=1 python -u run_pandas_generation.py \
  --taskdb-root "$TASKDB_ROOT" \
  --repo-url "$REPO_URL" \
  --difficulty "$DIFFICULTY" \
  --agent-type "$AGENT_TYPE" \
  --num-tasks "$NUM_TASKS" \
  --log-level "$LOG_LEVEL" \
  --max-tokens "$MAX_TOKENS" \
  --temperature "$TEMPERATURE" \
  "$@"

