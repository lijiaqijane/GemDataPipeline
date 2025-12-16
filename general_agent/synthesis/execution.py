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
    """Smart database query function (logic copied from EnvironmentSynthesizer.smart_db_query)"""
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
    """Create SandboxFusion wrapper code, inject tool implementations and database records."""
    # Serialize database records - use base64 encoding to avoid quote issues in JSON strings
    db_records_json = json.dumps(db_records, ensure_ascii=False)
    db_records_b64 = base64.b64encode(db_records_json.encode('utf-8')).decode('utf-8')
    
    # Serialize tool name list - use base64 encoding to avoid quote issues in JSON strings
    tool_names = list(tools.keys())
    tool_names_json = json.dumps(tool_names, ensure_ascii=False)
    tool_names_b64 = base64.b64encode(tool_names_json.encode('utf-8')).decode('utf-8')
    
    # Use base64 encoding for code to avoid escape issues
    code_b64 = base64.b64encode(code.encode('utf-8')).decode('utf-8')
    answer_json = json.dumps(args[0], default=str, ensure_ascii=False) if args else 'null'
    answer_b64 = base64.b64encode(answer_json.encode('utf-8')).decode('utf-8')
    
    # Create wrapper code for SandboxFusion execution, inject real tool implementations
    wrapper_code = f"""
import json
import sys
import os
import base64
import subprocess

# Inject database records (use base64 decoding to avoid JSON string escape issues)
_db_records_json = base64.b64decode('{db_records_b64}').decode('utf-8')
db_records = json.loads(_db_records_json)

# Smart database query function
def smart_db_query(records, tool_key, query):
    if not query or not isinstance(query, str):
        return records
    query_lower = query.lower().strip()
    
    # Exact match
    for record in records:
        title = str(record.get("title", "")).lower()
        summary = str(record.get("summary", "")).lower()
        if query_lower in title or query_lower in summary:
            return [record]
    
    # Keyword matching
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

# Database accessor (for verification code use)
class DatabaseAccessor:
    def __init__(self, db_records):
        self._records = db_records
    
    def get(self, key):
        # Return all records or filter by key
        if key:
            return [r for r in self._records if key in str(r.get("title", "")).lower() or key in str(r.get("summary", "")).lower()]
        return self._records
    
    def query(self, field, value):
        return [r for r in self._records if r.get(field) == value]

# Real tool implementations
class RealTools:
    def __init__(self, tool_names, db_records):
        self._names = tool_names
        self._db_records = db_records
        self._calls = []
        # Provide database access for verification code (verification code can access database)
        self._names.append("database")
    
    def __getitem__(self, key):
        # Verification code can access database tool
        if key == "database":
            return DatabaseAccessor(self._db_records)
        if key not in self._names:
            raise KeyError(f"Tool '{{key}}' not found. Available: {{self._names}}")
        return self._create_tool_handler(key)
    
    def __getattr__(self, key):
        # Verification code can access database tool
        if key == "database":
            return DatabaseAccessor(self._db_records)
        if key not in self._names:
            raise AttributeError(f"Tool '{{key}}' not found. Available: {{self._names}}")
        return self._create_tool_handler(key)
    
    def _create_tool_handler(self, tool_name):
        def handler(*args, **kwargs):
            self._calls.append({{"tool": tool_name, "args": list(args) if args else [], "kwargs": kwargs}})
            
            # bash tool: Execute bash commands in SandboxFusion
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
            
            # search tool: Search from database records (simulate web search)
            elif tool_name == "search":
                query = args[0] if args else kwargs.get("query", "")
                max_results = kwargs.get("max_results", 5) if "max_results" in kwargs else (args[1] if len(args) > 1 else 5)
                # Use smart query to find from database records
                results = smart_db_query(self._db_records, tool_name, query)
                # Convert to format expected by search tool
                search_results = []
                for record in results[:max_results]:
                    search_results.append({{
                        "title": record.get("title", ""),
                        "url": f"https://example.com/{{record.get('title', '').replace(' ', '-').lower()}}"
                    }})
                return search_results
            
            # Other generated tools: Use smart database query
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
                
                # Use smart query
                result = smart_db_query(self._db_records, tool_name, candidate)
                
                # Return different formats based on tool type
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

# Create tool instance (use base64 decoding to avoid JSON string escape issues)
_tool_names_json = base64.b64decode('{tool_names_b64}').decode('utf-8')
tool_names = json.loads(_tool_names_json)
tools = RealTools(tool_names, db_records)

# Decode and execute user code
code_bytes = base64.b64decode('{code_b64}')
exec(code_bytes)

if '{func_name}' == 'solve':
    result = solve(tools)
    # Output result and tool call records
    output = {{
        "result": result,
        "tool_calls": tools.get_calls()
    }}
    print(json.dumps(output, default=str))
elif '{func_name}' == 'verify':
    # Use base64 decoding to avoid JSON string escape issues
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
    
    # Retry mechanism: retry at most 3 times
    max_retries = 3
    last_error = None
    
    for attempt in range(max_retries):
        try:
            # Record code length for debugging
            if attempt == 0:
                logger.debug(f"Executing code in SandboxFusion: func_name={func_name}, code_length={len(code)}, wrapper_length={len(wrapper_code)}")
            
            result = executor(wrapper_code, language="python")
            
            # Extract actual stdout and stderr from raw response (SandboxFusion response format is nested)
            raw_response = result.get('raw', {})
            run_result = raw_response.get('run_result', {}) if raw_response else {}
            
            # Prefer values from run_result, if not available use values from result
            output = run_result.get('stdout', result.get('stdout', '')).strip()
            stderr_output = run_result.get('stderr', result.get('stderr', '')).strip()
            return_code = run_result.get('return_code', result.get('return_code', 0))
            status = raw_response.get('status', result.get('status', 'unknown'))
            
            # Record detailed execution results for debugging
            if attempt == 0:  # Only record detailed info on first attempt
                logger.debug(f"SandboxFusion execution result: status={status}, return_code={return_code}, stdout_length={len(output)}, stderr_length={len(stderr_output)}")
                if raw_response:
                    logger.debug(f"Raw response keys: {list(raw_response.keys())}")
            
            # Check execution status
            if return_code != 0 or status in ("error", "Failed"):
                error_msg = stderr_output or "Unknown error"
                stdout_preview = output[:500]
                # Check if there's more error info in raw response
                if raw_response and "message" in raw_response:
                    error_msg = f"{error_msg} | Message: {raw_response.get('message', '')}"
                if attempt < max_retries - 1:
                    logger.warning(f"SandboxFusion execution failed (attempt {attempt + 1}/{max_retries}): {error_msg[:200]}")
                    if stdout_preview:
                        logger.debug(f"SandboxFusion stdout preview: {stdout_preview}")
                    continue
                raise RuntimeError(f"SandboxFusion execution failed after {max_retries} attempts: {error_msg}. stdout: {stdout_preview}")
            
            # If output is empty, check for other information
            if not output:
                # Check other possible locations in raw response
                if raw_response:
                    if "output" in raw_response:
                        output = str(raw_response.get("output", "")).strip()
                    elif "result" in raw_response:
                        output = str(raw_response.get("result", "")).strip()
                
                if not output:
                    if stderr_output:
                        logger.warning(f"SandboxFusion stderr (empty stdout): {stderr_output[:500]}")
                    # Record full raw response for debugging
                    if raw_response and attempt == 0:
                        logger.debug(f"Full raw response structure: {list(raw_response.keys())}")
                    if attempt < max_retries - 1:
                        logger.warning(f"SandboxFusion returned empty output (attempt {attempt + 1}/{max_retries})")
                        continue
                    raise RuntimeError(f"SandboxFusion returned empty output after retries. stderr: {stderr_output[:500]}")
            
            # Try to extract JSON
            # First try to directly parse entire output
            try:
                parsed = json.loads(output)
                # If successful, check if it's the expected format
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
                pass  # Continue trying to extract JSON fragment
            
            # Try to extract JSON object from output
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
                # Try to find any JSON structure
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
            
            # If output contains tool call records, log them
            if isinstance(parsed, dict) and "tool_calls" in parsed:
                tool_calls = parsed.get("tool_calls", [])
                if tool_calls:
                    logger.debug(f"Tool calls made in SandboxFusion: {tool_calls}")
                # Return actual result
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
    
    # If all retries failed
    raise RuntimeError(f"SandboxFusion execution failed after {max_retries} attempts. Last error: {last_error}")

