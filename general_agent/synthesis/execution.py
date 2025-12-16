"""SandboxFusion execution with real tool implementations."""

import json
import logging
import os
import base64
import subprocess
from typing import Any, Dict, List

from ..executor import SandboxFusionExecutor

logger = logging.getLogger(__name__)


def smart_db_query(records: List[Dict[str, Any]], tool_key: str, query: str) -> List[Dict[str, Any]]:
    """智能数据库查询函数（从EnvironmentSynthesizer.smart_db_query复制逻辑）"""
    if not query or not isinstance(query, str):
        return records
    query_lower = query.lower().strip()
    
    # 精确匹配
    for record in records:
        title = str(record.get("title", "")).lower()
        summary = str(record.get("summary", "")).lower()
        if query_lower in title or query_lower in summary:
            return [record]
    
    # 关键词匹配
    query_words = query_lower.split()
    scored_records = []
    for record in records:
        title = str(record.get("title", "")).lower()
        summary = str(record.get("summary", "")).lower()
        score = 0
        for word in query_words:
            if word in title:
                score += 2
            if word in summary:
                score += 1
        if score > 0:
            scored_records.append((score, title, record))
    
    scored_records.sort(reverse=True)
    result = [record for _, _, record in scored_records[:5]]
    return result if result else records[:5]


def create_sandbox_wrapper_code(
    code: str,
    tools: Dict[str, Any],
    func_name: str,
    db_records: List[Dict[str, Any]],
    *args: Any
) -> str:
    """创建SandboxFusion包装代码，注入工具实现和数据库记录。"""
    # 序列化数据库记录 - 使用base64编码避免JSON字符串中的引号问题
    db_records_json = json.dumps(db_records, ensure_ascii=False)
    db_records_b64 = base64.b64encode(db_records_json.encode('utf-8')).decode('utf-8')
    
    # 序列化工具名称列表 - 使用base64编码避免JSON字符串中的引号问题
    tool_names = list(tools.keys())
    tool_names_json = json.dumps(tool_names, ensure_ascii=False)
    tool_names_b64 = base64.b64encode(tool_names_json.encode('utf-8')).decode('utf-8')
    
    # 使用base64编码代码，避免转义问题
    code_b64 = base64.b64encode(code.encode('utf-8')).decode('utf-8')
    answer_json = json.dumps(args[0], default=str, ensure_ascii=False) if args else 'null'
    answer_b64 = base64.b64encode(answer_json.encode('utf-8')).decode('utf-8')
    
    # 创建包装代码用于SandboxFusion执行，注入真正的工具实现
    wrapper_code = f"""
import json
import sys
import os
import base64
import subprocess

# 注入数据库记录（使用base64解码避免JSON字符串转义问题）
_db_records_json = base64.b64decode('{db_records_b64}').decode('utf-8')
db_records = json.loads(_db_records_json)

# 智能数据库查询函数
def smart_db_query(records, tool_key, query):
    if not query or not isinstance(query, str):
        return records
    query_lower = query.lower().strip()
    
    # 精确匹配
    for record in records:
        title = str(record.get("title", "")).lower()
        summary = str(record.get("summary", "")).lower()
        if query_lower in title or query_lower in summary:
            return [record]
    
    # 关键词匹配
    query_words = query_lower.split()
    scored_records = []
    for record in records:
        title = str(record.get("title", "")).lower()
        summary = str(record.get("summary", "")).lower()
        score = 0
        for word in query_words:
            if word in title:
                score += 2
            if word in summary:
                score += 1
        if score > 0:
            scored_records.append((score, title, record))
    
    scored_records.sort(reverse=True)
    result = [record for _, _, record in scored_records[:5]]
    return result if result else records[:5]

# 数据库访问器（供验证代码使用）
class DatabaseAccessor:
    def __init__(self, db_records):
        self._records = db_records
    
    def get(self, key):
        # 返回所有记录或按key过滤
        if key:
            return [r for r in self._records if key in str(r.get("title", "")).lower() or key in str(r.get("summary", "")).lower()]
        return self._records
    
    def query(self, field, value):
        return [r for r in self._records if r.get(field) == value]

# 真正的工具实现
class RealTools:
    def __init__(self, tool_names, db_records):
        self._names = tool_names
        self._db_records = db_records
        self._calls = []
        # 为验证代码提供数据库访问（验证代码可以访问数据库）
        self._names.append("database")
    
    def __getitem__(self, key):
        # 验证代码可以访问database工具
        if key == "database":
            return DatabaseAccessor(self._db_records)
        if key not in self._names:
            raise KeyError(f"Tool '{{key}}' not found. Available: {{self._names}}")
        return self._create_tool_handler(key)
    
    def __getattr__(self, key):
        # 验证代码可以访问database工具
        if key == "database":
            return DatabaseAccessor(self._db_records)
        if key not in self._names:
            raise AttributeError(f"Tool '{{key}}' not found. Available: {{self._names}}")
        return self._create_tool_handler(key)
    
    def _create_tool_handler(self, tool_name):
        def handler(*args, **kwargs):
            self._calls.append({{"tool": tool_name, "args": list(args) if args else [], "kwargs": kwargs}})
            
            # bash工具：在SandboxFusion中执行bash命令
            if tool_name == "bash":
                command = args[0] if args else (kwargs.get("command", ""))
                try:
                    result = subprocess.run(
                        command,
                        shell=True,
                        capture_output=True,
                        text=True,
                        timeout=10
                    )
                    return {{
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }}
                except Exception as e:
                    return {{"returncode": -1, "stdout": "", "stderr": str(e)}}
            
            # search工具：从数据库记录中搜索（模拟网络搜索）
            elif tool_name == "search":
                query = args[0] if args else kwargs.get("query", "")
                max_results = kwargs.get("max_results", 5) if "max_results" in kwargs else (args[1] if len(args) > 1 else 5)
                # 使用智能查询从数据库记录中查找
                results = smart_db_query(self._db_records, tool_name, query)
                # 转换为search工具期望的格式
                search_results = []
                for record in results[:max_results]:
                    search_results.append({{
                        "title": record.get("title", ""),
                        "url": f"https://example.com/{{record.get('title', '').replace(' ', '-').lower()}}"
                    }})
                return search_results
            
            # 其他生成的工具：使用智能数据库查询
            else:
                candidate = None
                if args:
                    candidate = args[0]
                if "query" in kwargs:
                    candidate = kwargs["query"]
                if candidate is None and kwargs:
                    candidate = " ".join(f"{{k}}:{{v}}" for k, v in kwargs.items())
                if isinstance(candidate, dict):
                    candidate = json.dumps(candidate, ensure_ascii=False)
                if candidate is None:
                    return self._db_records[:5]
                elif not isinstance(candidate, str):
                    candidate = str(candidate)
                
                # 使用智能查询
                result = smart_db_query(self._db_records, tool_name, candidate)
                
                # 根据工具类型返回不同格式
                if any(word in tool_name.lower() for word in ["matcher", "finder"]):
                    return [r.get("title", "") for r in result[:5]]
                elif any(word in tool_name.lower() for word in ["recommendation", "advisor"]):
                    return [r.get("summary", r.get("title", "")) for r in result[:3]]
                elif any(word in tool_name.lower() for word in ["checker", "analyzer"]):
                    return {{"present": [r.get("title", "") for r in result[:3]], "missing": []}}
                else:
                    return result[:5]
        
        return handler
    
    def get_calls(self):
        return self._calls

# 创建工具实例（使用base64解码避免JSON字符串转义问题）
_tool_names_json = base64.b64decode('{tool_names_b64}').decode('utf-8')
tool_names = json.loads(_tool_names_json)
tools = RealTools(tool_names, db_records)

# 解码并执行用户代码
code_bytes = base64.b64decode('{code_b64}')
exec(code_bytes)

if '{func_name}' == 'solve':
    result = solve(tools)
    # 输出结果和工具调用记录
    output = {{
        "result": result,
        "tool_calls": tools.get_calls()
    }}
    print(json.dumps(output, default=str))
elif '{func_name}' == 'verify':
    # 使用base64解码避免JSON字符串转义问题
    _answer_json = base64.b64decode('{answer_b64}').decode('utf-8')
    try:
        answer = json.loads(_answer_json)
    except:
        answer = _answer_json
    result = verify(tools, answer)
    output = {{
        "result": result,
        "tool_calls": tools.get_calls()
    }}
    print(json.dumps(output, default=str))
"""
    return wrapper_code


def execute_in_sandbox_fusion(
    code: str,
    tools: Dict[str, Any],
    func_name: str,
    db_records: List[Dict[str, Any]],
    *args: Any
) -> Any:
    """Execute code in SandboxFusion service with real tool implementations.
    
    Args:
        code: Code to execute
        tools: Dictionary of available tools
        func_name: Function name to call ('solve' or 'verify')
        db_records: Database records to inject
        *args: Additional arguments (e.g., answer for verify)
    
    Returns:
        Execution result
    """
    wrapper_code = create_sandbox_wrapper_code(code, tools, func_name, db_records, *args)
    
    executor = SandboxFusionExecutor(
        base_url=os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080"),
        timeout=int(os.getenv("SANDBOX_FUSION_TIMEOUT", "30")),
    )
    
    # 重试机制：最多重试3次
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # 记录代码长度用于调试
            if attempt == 0:
                logger.debug(f"Executing code in SandboxFusion: func_name={func_name}, code_length={len(code)}, wrapper_length={len(wrapper_code)}")
            
            result = executor(wrapper_code, language="python")
            
            # 从raw响应中提取实际的stdout和stderr（SandboxFusion的响应格式是嵌套的）
            raw_response = result.get('raw', {})
            run_result = raw_response.get('run_result', {}) if raw_response else {}
            
            # 优先使用run_result中的值，如果没有则使用result中的值
            output = run_result.get('stdout', result.get('stdout', '')).strip()
            stderr_output = run_result.get('stderr', result.get('stderr', '')).strip()
            return_code = run_result.get('return_code', result.get('return_code', 0))
            status = raw_response.get('status', result.get('status', 'unknown'))
            
            # 记录详细的执行结果用于调试
            if attempt == 0:  # 只在第一次尝试时记录详细信息
                logger.debug(f"SandboxFusion execution result: status={status}, return_code={return_code}, stdout_length={len(output)}, stderr_length={len(stderr_output)}")
                if raw_response:
                    logger.debug(f"Raw response keys: {list(raw_response.keys())}")
            
            # 检查执行状态
            if return_code != 0 or status in ("error", "Failed"):
                error_msg = stderr_output or "Unknown error"
                stdout_preview = output[:500]
                # 检查 raw 响应中是否有更多错误信息
                if raw_response and "message" in raw_response:
                    error_msg = f"{error_msg} | Message: {raw_response.get('message', '')}"
                if attempt < max_retries - 1:
                    logger.warning(f"SandboxFusion execution failed (attempt {attempt + 1}/{max_retries}): {error_msg[:200]}")
                    if stdout_preview:
                        logger.debug(f"SandboxFusion stdout preview: {stdout_preview}")
                    continue
                raise RuntimeError(f"SandboxFusion execution failed after {max_retries} attempts: {error_msg}. stdout: {stdout_preview}")
            
            # 如果输出为空，检查是否有其他信息
            if not output:
                # 检查 raw 响应中的其他可能位置
                if raw_response:
                    if "output" in raw_response:
                        output = str(raw_response.get("output", "")).strip()
                    elif "result" in raw_response:
                        output = str(raw_response.get("result", "")).strip()
                
                if not output:
                    if stderr_output:
                        logger.warning(f"SandboxFusion stderr (empty stdout): {stderr_output[:500]}")
                    # 记录完整的 raw 响应用于调试
                    if raw_response and attempt == 0:
                        logger.debug(f"Full raw response structure: {list(raw_response.keys())}")
                    if attempt < max_retries - 1:
                        logger.warning(f"SandboxFusion returned empty output (attempt {attempt + 1}/{max_retries})")
                        continue
                    raise RuntimeError(f"SandboxFusion returned empty output after retries. stderr: {stderr_output[:500]}")
            
            # 尝试提取JSON
            # 首先尝试直接解析整个输出
            try:
                parsed = json.loads(output)
                # 如果成功，检查是否是期望的格式
                if isinstance(parsed, dict) and "result" in parsed:
                    tool_calls = parsed.get("tool_calls", [])
                    if tool_calls:
                        logger.debug(f"Tool calls made in SandboxFusion: {tool_calls}")
                    return parsed.get("result")
                elif isinstance(parsed, dict) and "tool_calls" in parsed:
                    tool_calls = parsed.get("tool_calls", [])
                    if tool_calls:
                        logger.debug(f"Tool calls made in SandboxFusion: {tool_calls}")
                    return parsed.get("result")
                else:
                    return parsed
            except json.JSONDecodeError:
                pass  # 继续尝试提取JSON片段
            
            # 尝试从输出中提取JSON对象
            start = output.find('{')
            end = output.rfind('}') + 1
            if start >= 0 and end > start:
                output_json = output[start:end]
            elif output.startswith('['):
                end = output.rfind(']') + 1
                if end > 0:
                    output_json = output[:end]
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"Invalid JSON array in output (attempt {attempt + 1}/{max_retries}): {output[:200]}")
                        continue
                    raise RuntimeError(f"Invalid JSON array in output: {output[:200]}")
            else:
                # 尝试查找任何JSON结构
                import re
                json_match = re.search(r'\{.*\}|\[.*\]', output, re.DOTALL)
                if json_match:
                    output_json = json_match.group(0)
                else:
                    if attempt < max_retries - 1:
                        logger.warning(f"No valid JSON found in output (attempt {attempt + 1}/{max_retries}): {output[:500]}")
                        continue
                    raise RuntimeError(f"No valid JSON found in output: {output[:500]}")
            
            parsed = json.loads(output_json)
            
            # 如果输出包含工具调用记录，记录日志
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls = parsed.get("tool_calls", [])
                if tool_calls:
                    logger.debug(f"Tool calls made in SandboxFusion: {tool_calls}")
                # 返回实际结果
                return parsed.get("result")
            
            return parsed
            
        except json.JSONDecodeError as e:
            last_error = f"Failed to parse SandboxFusion output as JSON: {output[:200]}. Error: {e}"
            if attempt < max_retries - 1:
                logger.warning(f"JSON decode error (attempt {attempt + 1}/{max_retries}): {last_error}")
                continue
            raise RuntimeError(last_error)
        except Exception as e:
            last_error = str(e)
            if attempt < max_retries - 1:
                logger.warning(f"Execution error (attempt {attempt + 1}/{max_retries}): {last_error}")
                continue
            raise
    
    # 如果所有重试都失败
    raise RuntimeError(f"SandboxFusion execution failed after {max_retries} attempts. Last error: {last_error}")

