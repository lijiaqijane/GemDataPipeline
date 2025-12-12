from __future__ import annotations

import base64
import json
import os
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

from agent_gem.tools import DockerTool


class ToolSpec(BaseModel):
    tool_name: str = Field(..., min_length=1)
    tool_description: str = Field(..., min_length=1)
    tool_functionality: str = Field(..., min_length=1)


class EvaluationCriteria(BaseModel):
    correctness: float = Field(0.6, ge=0.0, le=1.0)
    diversity: float = Field(0.6, ge=0.0, le=1.0)
    complexity: float = Field(0.6, ge=0.0, le=1.0)
    solution_verifiability: float = Field(0.6, ge=0.0, le=1.0)

    @field_validator("*", mode="before")
    @classmethod
    def default_to_float(cls, value: float) -> float:
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return value


class TaskDefinition(BaseModel):
    task_title: str = Field(..., min_length=4)
    task_content: str = Field(..., min_length=10)
    submit_result_format: dict = Field(default_factory=dict)
    tool_set: List[ToolSpec] = Field(default_factory=list)
    evaluation_criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)
    difficulty_level: int = Field(default=1)

    def summary(self) -> str:
        return f"{self.task_title} [{self.difficulty_level}]"


class TaskPackage(BaseModel):
    task: TaskDefinition
    solution: str = Field(..., min_length=3)
    verification: str = Field(..., min_length=3)
    agent_type: str = Field(..., min_length=3)
    metadata: Dict[str, str] = Field(default_factory=dict)
    sandbox_path: Optional[str] = None
    use_docker: bool = False

    def as_payload(self) -> Dict[str, object]:
        return {
            "agent_type": self.agent_type,
            "task": self.task.model_dump(),
            "solution": self.solution,
            "verification": self.verification,
            "metadata": self.metadata,
            "sandbox_path": self.sandbox_path,
        }

    def run_solution(self, tools: Dict[str, Any]) -> Any:
        if self.use_docker:
            return self._run_in_docker(self.solution, tools, "solve")

        env = self._build_exec_env(tools)
        exec(self.solution, env, env)
        if "solve" not in env:
            raise RuntimeError("solution_code must define solve(tools)")
        return env["solve"](tools)

    def verify(self, tools: Dict[str, Any], answer: Any) -> bool:
        if self.use_docker:
            result = self._run_in_docker(self.verification, tools, "verify", answer)
            return bool(result)

        env = self._build_exec_env(tools)
        exec(self.verification, env, env)
        if "verify" not in env:
            raise RuntimeError("verification_code must define verify(tools, answer)")
        return bool(env["verify"](tools, answer))

    def _run_in_docker(self, code: str, tools: Dict[str, Any], func_name: str, *args: Any) -> Any:
        """Execute code in Docker container.

        Note: Tools cannot be directly passed to Docker, so we create a simplified
        tool interface that uses subprocess to call back to the host.
        """

        # Serialize tools metadata (names and descriptions only)
        tools_meta = {name: {"name": name} for name in tools.keys()}
        tools_meta_json = json.dumps(tools_meta)

        # Create a wrapper that provides a mock tools interface
        # In a real implementation, tools would communicate via a shared mechanism
        # For now, we'll create a simplified version that works with the code structure
        wrapper_code = f"""
import json
import sys

# Simplified tools interface - tools are accessed but execution happens locally
class ToolProxy:
    def __init__(self, tool_names):
        self._names = tool_names
    
    def __getitem__(self, key):
        return self
    
    def __getattr__(self, key):
        return self
    
    def __call__(self, *args, **kwargs):
        # Return mock data for tool calls
        return {{"result": "tool_executed", "args": str(args), "kwargs": str(kwargs)}}

tools_meta = json.loads('{tools_meta_json}')
tools = ToolProxy(tools_meta.keys())

{code}

if '{func_name}' == 'solve':
    result = solve(tools)
    print(json.dumps(result, default=str))
elif '{func_name}' == 'verify':
    answer_str = sys.argv[1] if len(sys.argv) > 1 else 'null'
    try:
        answer = json.loads(answer_str)
    except:
        answer = answer_str
    result = verify(tools, answer)
    print(json.dumps(result, default=str))
"""

        docker_tool = DockerTool(
            image=os.getenv("DOCKER_IMAGE", "python:3.11-slim"),
            timeout=int(os.getenv("DOCKER_TIMEOUT", "30")),
        )

        # For verify, pass answer as JSON string argument
        if func_name == "verify" and args:
            answer_json = json.dumps(args[0], default=str)
            # Escape for shell
            answer_json_escaped = answer_json.replace("'", "'\"'\"'")
            wrapper_code = wrapper_code.replace("sys.argv[1]", f"'{answer_json_escaped}'")

        result = docker_tool(code=wrapper_code, language="python")

        if result["returncode"] != 0:
            raise RuntimeError(f"Docker execution failed: {result['stderr']}")

        try:
            output = result["stdout"].strip()
            # Remove any non-JSON prefix (like print statements)
            if output:
                # Try to find JSON in output
                start = output.find("{")
                end = output.rfind("}") + 1
                if start >= 0 and end > start:
                    output = output[start:end]
                elif output.startswith("["):
                    end = output.rfind("]") + 1
                    if end > 0:
                        output = output[:end]
                return json.loads(output)
            return None
        except json.JSONDecodeError as e:
            # Fallback: try to evaluate as Python literal
            try:
                return eval(output)
            except:
                raise RuntimeError(f"Failed to parse Docker output: {output[:200]}. Error: {e}")

    @staticmethod
    def _build_exec_env(tools: Dict[str, Any]) -> Dict[str, Any]:
        safe_builtins = {
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "any": any,
            "all": all,
            "sorted": sorted,
            "enumerate": enumerate,
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "isinstance": isinstance,
        }
        return {"__builtins__": safe_builtins, "tools": tools}
