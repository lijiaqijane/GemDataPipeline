#!/usr/bin/env bash
# 通用运行脚本：通过环境变量快速切换 Deepseek 或其它兼容 vLLM 的后端

set -euo pipefail

# ---------- 可改参数（导出/修改这些变量即可） ----------
# LLM 相关（默认用 Deepseek/Volcano，已内置 Key）
export LLM_PROVIDER="${LLM_PROVIDER:-deepseek}"                        # 可选：vllm | openai | volcano | deepseek
export VOLCANO_BASE_URL="${VOLCANO_BASE_URL:-https://ark.cn-beijing.volces.com/api/v3}"
export VOLCANO_MODEL="${VOLCANO_MODEL:-deepseek-v3-2-251201}"
export VOLCANO_API_KEY="${VOLCANO_API_KEY:-47041ffc-3c83-49ee-9d79-4f70592850d2}"  # 示例 Key，建议替换

# OpenAI/vLLM 示例（如使用则自行设置）
export OPENAI_BASE_URL="${OPENAI_BASE_URL:-https://api.openai.com/v1}"
export OPENAI_MODEL="${OPENAI_MODEL:-gpt-4o-mini}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-}"
export VLLM_BASE_URL="${VLLM_BASE_URL:-http://localhost:8000/v1}"
export VLLM_MODEL="${VLLM_MODEL:-local-model}"
export VLLM_API_KEY="${VLLM_API_KEY:-}"

# SandboxFusion
export SANDBOX_FUSION_URL="${SANDBOX_FUSION_URL:-http://localhost:8080}"
export SANDBOX_FUSION_TIMEOUT="${SANDBOX_FUSION_TIMEOUT:-30}"
export SANDBOX_FUSION_LANGUAGE="${SANDBOX_FUSION_LANGUAGE:-python}"

# 运行参数
CATEGORY="${CATEGORY:-Paris Travel Planning}"
SANDBOX="${SANDBOX:-./sandbox/run}"
ROUNDS="${ROUNDS:-5}"
VALIDATE="${VALIDATE:-1}"                     # 1=执行校验，0=跳过
USE_SANDBOX_FUSION="${USE_SANDBOX_FUSION:-1}" # 默认启用 SandboxFusion
SANDBOX_FUSION_PORT="${SANDBOX_FUSION_PORT:-8080}"

# ---------- 健康检查函数 ----------
check_sandbox_fusion() {
  local url="${SANDBOX_FUSION_URL}"
  local timeout=3
  local max_attempts=3
  local attempt=1
  
  # 尝试连接 SandboxFusion 服务
  while [[ $attempt -le $max_attempts ]]; do
    if command -v curl &> /dev/null; then
      if curl -s --max-time "$timeout" -X POST "${url}/run_code" \
         -H "Content-Type: application/json" \
         -d '{"code":"print(1)","language":"python"}' &> /dev/null 2>&1; then
        return 0
      fi
    elif command -v wget &> /dev/null; then
      if wget -q --timeout="$timeout" --tries=1 -O /dev/null \
         --post-data='{"code":"print(1)","language":"python"}' \
         --header="Content-Type: application/json" \
         "${url}/run_code" &> /dev/null 2>&1; then
        return 0
      fi
    else
      # 使用 Python 检查连接和 API
      if python3 -c "
import sys
import socket
import json
import urllib.request
import urllib.error
try:
    # 先检查端口连接
    socket.setdefaulttimeout($timeout)
    host = '${url#http://}'
    host = host.split('/')[0]
    if ':' in host:
        host, port = host.split(':')
        port = int(port)
    else:
        port = ${SANDBOX_FUSION_PORT}
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    result = sock.connect_ex((host, port))
    sock.close()
    if result != 0:
        sys.exit(1)
    
    # 尝试调用 API
    req = urllib.request.Request('${url}/run_code',
        data=json.dumps({'code': 'print(1)', 'language': 'python'}).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    urllib.request.urlopen(req, timeout=$timeout)
    sys.exit(0)
except Exception:
    sys.exit(1)
" 2>/dev/null; then
        return 0
      fi
    fi
    
    if [[ $attempt -lt $max_attempts ]]; then
      sleep 1
    fi
    attempt=$((attempt + 1))
  done
  return 1
}

# ---------- 确保 SandboxFusion 服务可用 ----------
echo "[INFO] 检查 SandboxFusion 服务..."
if ! check_sandbox_fusion; then
  echo "[ERROR] SandboxFusion 服务不可用 (${SANDBOX_FUSION_URL})"
  echo "[ERROR] 请先启动 SandboxFusion 服务，然后重试。"
  exit 1
else
  echo "[INFO] SandboxFusion 服务可用"
fi

# ---------- 调试信息 ----------
if [[ "${DEBUG:-0}" == "1" ]]; then
  echo "[DEBUG] === 环境变量检查 ==="
  echo "[DEBUG] LLM_PROVIDER: ${LLM_PROVIDER:-未设置}"
  echo "[DEBUG] VOLCANO_API_KEY: ${VOLCANO_API_KEY:+已设置} ${VOLCANO_API_KEY:+(${#VOLCANO_API_KEY}字符)}"
  echo "[DEBUG] SANDBOX_FUSION_URL: ${SANDBOX_FUSION_URL:-未设置}"
  echo "[DEBUG] CATEGORY: ${CATEGORY:-未设置}"
  echo "[DEBUG] SANDBOX: ${SANDBOX:-未设置}"
  echo "[DEBUG] ROUNDS: ${ROUNDS:-未设置}"
  echo "[DEBUG] USE_SANDBOX_FUSION: ${USE_SANDBOX_FUSION:-未设置}"
fi

# ---------- 组装命令 ----------
args=(
  --category "$CATEGORY"
  --sandbox "$SANDBOX"
  --rounds "$ROUNDS"
)

if [[ "$VALIDATE" == "0" ]]; then
  args+=(--no-validate)
fi

# 执行模式
if [[ "$USE_SANDBOX_FUSION" == "1" ]]; then
  args+=(--use-sandbox-fusion)
else
  args+=(--no-sandbox-fusion)
fi

# ---------- 执行主程序 ----------
echo "[INFO] 启动主程序（实时输出）..."
if [[ "${DEBUG:-0}" == "1" ]]; then
  echo "[DEBUG] 执行命令: PYTHONUNBUFFERED=1 PYTHONPATH=$(pwd)/general_agent_bundle:${PYTHONPATH:-} python -u -m general_agent ${args[*]} $@"
fi

# 实时输出 + 保留日志以便后续检测验证失败
tmp_log=$(mktemp)
set +e
PYTHONUNBUFFERED=1 PYTHONPATH="$(pwd)/general_agent_bundle:${PYTHONPATH:-}" python -u -m general_agent "${args[@]}" "$@" 2>&1 | tee "$tmp_log"
exit_code=${PIPESTATUS[0]}
set -e

# 检查是否有验证失败的警告
if grep -q "Task failed validation" "$tmp_log"; then
  echo ""
  echo "[WARNING] 检测到任务验证失败"
  echo "[INFO] 这通常是由于以下原因："
  echo "  1. 工具匹配逻辑过于严格，数据库记录中缺少相关关键词"
  echo "  2. LLM生成的验证代码过于严格"
  echo "  3. 任务复杂度与可用数据不匹配"
  echo ""
  echo "[INFO] 建议解决方案："
  echo "  - 增加更多相关的数据库记录"
  echo "  - 调整任务难度或重新生成"
  echo "  - 检查数据库内容是否包含任务需要的关键词"
fi

rm -f "$tmp_log"
exit $exit_code

