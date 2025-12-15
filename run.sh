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
ROUNDS="${ROUNDS:-2}"
VALIDATE="${VALIDATE:-1}"                     # 1=执行校验，0=跳过
USE_SANDBOX_FUSION="${USE_SANDBOX_FUSION:-1}" # 默认启用 SandboxFusion
USE_DOCKER="${USE_DOCKER:-0}"                 # 默认关闭 Docker（SandboxFusion 优先）
# 注意：USE_SANDBOX_FUSION 和 USE_DOCKER 是互斥的执行模式
# 如果两者都启用，SandboxFusion 优先

# SandboxFusion Docker 配置
SANDBOX_FUSION_IMAGE="${SANDBOX_FUSION_IMAGE:-volcengine/sandbox-fusion:server-20250609}"
SANDBOX_FUSION_CONTAINER_NAME="${SANDBOX_FUSION_CONTAINER_NAME:-sandbox-fusion-server}"
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

# ---------- 检查 Docker 容器是否运行 ----------
check_docker_container() {
  if ! command -v docker &> /dev/null; then
    return 1
  fi
  
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^${SANDBOX_FUSION_CONTAINER_NAME}$"; then
    return 0
  fi
  return 1
}

# ---------- 检查 Docker 镜像是否存在 ----------
check_docker_image() {
  if ! command -v docker &> /dev/null; then
    return 1
  fi
  
  if docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null | grep -q "^${SANDBOX_FUSION_IMAGE}$"; then
    return 0
  fi
  return 1
}

# ---------- 拉取 Docker 镜像 ----------
pull_docker_image() {
  echo "[INFO] Docker 镜像不存在，正在拉取: ${SANDBOX_FUSION_IMAGE}"
  echo "[INFO] 这可能需要几分钟时间，请耐心等待..."
  
  if ! docker pull "${SANDBOX_FUSION_IMAGE}"; then
    echo "[ERROR] 无法拉取 Docker 镜像: ${SANDBOX_FUSION_IMAGE}"
    echo "[ERROR] 请检查网络连接或手动拉取："
    echo "       docker pull ${SANDBOX_FUSION_IMAGE}"
    exit 1
  fi
  
  echo "[INFO] Docker 镜像拉取完成"
}

# ---------- 启动 SandboxFusion Docker 容器 ----------
start_sandbox_fusion() {
  echo "[INFO] 正在启动 SandboxFusion 服务..."
  
  if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker 未安装或不在 PATH 中，无法启动 SandboxFusion"
    echo "[ERROR] 请安装 Docker 或手动启动 SandboxFusion 服务"
    exit 1
  fi
  
  # 检查 Docker 服务是否运行
  if ! docker info &> /dev/null; then
    echo "[ERROR] Docker 服务未运行，请先启动 Docker 服务"
    echo "[ERROR] 通常可以通过以下命令启动："
    echo "       sudo systemctl start docker"
    echo "       或"
    echo "       sudo service docker start"
    exit 1
  fi
  
  # 检查并拉取镜像
  if ! check_docker_image; then
    pull_docker_image
  else
    echo "[INFO] Docker 镜像已存在: ${SANDBOX_FUSION_IMAGE}"
  fi
  
  # 检查端口是否被占用
  local host="${SANDBOX_FUSION_URL#http://}"
  host="${host%%/*}"
  local port="${SANDBOX_FUSION_PORT}"
  if [[ "$host" == *:* ]]; then
    port="${host##*:}"
  fi
  
  # 检查是否有容器在运行但端口不对
  if check_docker_container; then
    echo "[INFO] 发现已存在的容器 ${SANDBOX_FUSION_CONTAINER_NAME}，尝试重启..."
    if docker restart "${SANDBOX_FUSION_CONTAINER_NAME}" 2>&1; then
      sleep 2
      if check_sandbox_fusion; then
        echo "[INFO] SandboxFusion 服务已就绪"
        return 0
      fi
    fi
  fi
  
  # 停止并删除旧容器（如果存在）
  if docker ps -a --format '{{.Names}}' 2>/dev/null | grep -q "^${SANDBOX_FUSION_CONTAINER_NAME}$"; then
    echo "[INFO] 清理旧容器..."
    docker stop "${SANDBOX_FUSION_CONTAINER_NAME}" 2>&1 || true
    docker rm "${SANDBOX_FUSION_CONTAINER_NAME}" 2>&1 || true
  fi
  
  # 启动新容器
  echo "[INFO] 启动 Docker 容器: ${SANDBOX_FUSION_IMAGE}"
  echo "[INFO] 端口映射: ${port}:8080"
  if ! docker run -d \
    --name "${SANDBOX_FUSION_CONTAINER_NAME}" \
    -p "${port}:8080" \
    "${SANDBOX_FUSION_IMAGE}" 2>&1; then
    echo "[ERROR] 无法启动 SandboxFusion Docker 容器"
    echo "[ERROR] 请检查 Docker 是否正常运行，或手动启动："
    echo "       docker run -d --name ${SANDBOX_FUSION_CONTAINER_NAME} -p ${port}:8080 ${SANDBOX_FUSION_IMAGE}"
    exit 1
  fi
  
  echo "[INFO] Docker 容器已启动，等待服务就绪..."
  
  # 等待服务启动
  local wait_time=0
  local max_wait=60
  while [[ $wait_time -lt $max_wait ]]; do
    if check_sandbox_fusion; then
      echo "[INFO] ✅ SandboxFusion 服务已就绪 (${SANDBOX_FUSION_URL})"
      return 0
    fi
    sleep 2
    wait_time=$((wait_time + 2))
    if [[ $((wait_time % 10)) -eq 0 ]]; then
      echo "[INFO] 等待服务启动中... (${wait_time}/${max_wait}秒)"
      # 显示容器状态
      if docker ps --format '{{.Names}}\t{{.Status}}' 2>/dev/null | grep -q "^${SANDBOX_FUSION_CONTAINER_NAME}"; then
        echo "[INFO] 容器运行状态: $(docker ps --format '{{.Status}}' --filter "name=${SANDBOX_FUSION_CONTAINER_NAME}" 2>/dev/null || echo '未知')"
      fi
    fi
  done
  
  echo "[ERROR] SandboxFusion 服务启动超时 (${max_wait}秒)"
  echo "[ERROR] 容器状态："
  docker ps -a --filter "name=${SANDBOX_FUSION_CONTAINER_NAME}" 2>&1 || true
  echo "[ERROR] 容器日志（最后 30 行）："
  docker logs --tail 30 "${SANDBOX_FUSION_CONTAINER_NAME}" 2>&1 || true
  exit 1
}

# ---------- 确保 SandboxFusion 服务可用 ----------
echo "[INFO] 检查 SandboxFusion 服务..."
if ! check_sandbox_fusion; then
  echo "[WARNING] SandboxFusion 服务不可用 (${SANDBOX_FUSION_URL})"
  start_sandbox_fusion
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
  echo "[DEBUG] USE_DOCKER: ${USE_DOCKER:-未设置}"
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

# 处理执行模式（SandboxFusion 和 Docker 互斥）
if [[ "$USE_SANDBOX_FUSION" == "1" ]]; then
  args+=(--use-sandbox-fusion)
  args+=(--no-docker)  # SandboxFusion 启用时禁用 Docker
else
  args+=(--no-sandbox-fusion)
  if [[ "$USE_DOCKER" == "1" ]]; then
    args+=(--use-docker)
  else
    args+=(--no-docker)
  fi
fi

# ---------- 执行主程序 ----------
echo "[INFO] 启动主程序（实时输出）..."
if [[ "${DEBUG:-0}" == "1" ]]; then
  echo "[DEBUG] 执行命令: PYTHONUNBUFFERED=1 python -u -m general_agent ${args[*]} $@"
fi

# 实时输出 + 保留日志以便后续检测验证失败
tmp_log=$(mktemp)
set +e
PYTHONUNBUFFERED=1 python -u -m general_agent "${args[@]}" "$@" 2>&1 | tee "$tmp_log"
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

