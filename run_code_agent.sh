#!/usr/bin/env bash
# CodeAgent 任务生成脚本：使用新的 batch 命令生成训练任务
# 支持两种模式：
#   1. 批量生成：从 triples 自动生成多个任务
#   2. 单个生成：为指定的 (repo, file, function) 生成任务
# 自动管理沙盒服务的启动和停止

set -euo pipefail

# ---------- 沙盒配置 ----------
# Docker 镜像名称
export SANDBOX_IMAGE="${SANDBOX_IMAGE:-code_sandbox:server}"

# Docker 启动命令
export SANDBOX_CMD="${SANDBOX_CMD:-docker run -d --rm --privileged -p 8080:8080 ${SANDBOX_IMAGE} make run-online}"

# 沙盒服务 URL
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-http://localhost:8080}"

# ---------- 环境变量配置 ----------
# 确保必要的环境变量已设置
if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "⚠️  警告: GITHUB_TOKEN 未设置，GitHub API 限流将很低"
  echo "   建议设置: export GITHUB_TOKEN=your_github_token"
fi

# LLM 超时和重试设置
export LLM_TIMEOUT="${LLM_TIMEOUT:-300}"
export LLM_MAX_RETRIES="${LLM_MAX_RETRIES:-5}"

# ---------- 脚本参数配置 ----------
# 配置文件路径（所有参数都在配置文件中管理）
CONFIG_FILE="${CONFIG_FILE:-config/code_agent.yaml}"

# 可选：指定特定的 (repo, file, function) 三元组
# 如果不指定，将使用批量模式从 triples 生成任务
REPO_NAME="${REPO_NAME:-numpy/numpy}"           # 例如: numpy/numpy
FILE_PATH="${FILE_PATH:-numpy/matlib.py}"           # 例如: numpy/core/numeric.py
FUNCTION_NAME="${FUNCTION_NAME:-identity}"   # 例如: array_equal

# 是否启用详细日志
VERBOSE="${VERBOSE:-false}"

# ---------- 执行主程序 ----------
echo ""
echo "=========================================="
echo "  CodeAgent Batch Task Generation"
echo "=========================================="
echo ""

# 构建参数列表
ARGS=(
  --config "$CONFIG_FILE"
)

# 检查是否指定了特定的三元组
if [ -n "$REPO_NAME" ]; then
  if [ -z "$FILE_PATH" ] || [ -z "$FUNCTION_NAME" ]; then
    echo "❌ 错误: 当指定 REPO_NAME 时，必须同时指定 FILE_PATH 和 FUNCTION_NAME"
    echo ""
    echo "示例："
    echo "  export REPO_NAME='numpy/numpy'"
    echo "  export FILE_PATH='numpy/core/numeric.py'"
    echo "  export FUNCTION_NAME='array_equal'"
    echo "  ./run_code_agent.sh"
    exit 1
  fi
  
  echo "🎯 模式: 单个任务生成"
  echo "   Repository: $REPO_NAME"
  echo "   File: $FILE_PATH"
  echo "   Function: $FUNCTION_NAME"
  echo ""
  
  ARGS+=(
    --repo "$REPO_NAME"
    --file "$FILE_PATH"
    --function "$FUNCTION_NAME"
  )
else
  echo "🚀 模式: 批量任务生成（从 triples）"
  echo "   配置文件: $CONFIG_FILE"
  echo ""
fi

# 添加 verbose 标志
if [ "$VERBOSE" = "true" ]; then
  ARGS+=(-v)
  echo "📝 详细日志: 已启用"
  echo ""
fi

echo "正在执行: python -m agent_gem batch ${ARGS[@]}"
echo ""

# 运行命令
PYTHONUNBUFFERED=1 python -u -m agent_gem batch "${ARGS[@]}" "$@"

