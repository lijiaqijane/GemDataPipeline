from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.handler(*args, **kwargs)


@dataclass
class DockerTool:
    """Execute code and commands in Docker container for secure isolation."""

    image: str = "python:3.11-slim"
    timeout: int = 30
    workdir: Path | None = None
    memory_limit: str = "512m"
    cpu_limit: str = "1.0"

    def __call__(self, code: str | None = None, command: str | None = None, language: str = "python") -> Dict[str, Any]:
        """Execute code or command in Docker container.
        
        Args:
            code: Python code to execute (for language='python')
            command: Shell command to execute (for language='bash')
            language: 'python' or 'bash'
            
        Returns:
            Dict with execution results
        """
        if code is None and command is None:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "Either 'code' or 'command' must be provided",
            }
        
        # Prepare Docker command
        if language == "python" and code:
            # Execute Python code
            docker_cmd = [
                "docker", "run",
                "--rm",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                "--network=none",  # Disable network for security
                "--read-only",  # Read-only root filesystem
                "--tmpfs=/tmp:rw,noexec,nosuid,size=100m",  # Temporary writable space
            ]
            
            if self.workdir:
                docker_cmd.extend(["-v", f"{self.workdir}:/workspace:ro"])
                docker_cmd.extend(["-w", "/workspace"])
            
            docker_cmd.extend([
                self.image,
                "python", "-c", code
            ])
        elif language == "bash" and command:
            # Execute bash command
            docker_cmd = [
                "docker", "run",
                "--rm",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                "--network=none",
                "--read-only",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=100m",
            ]
            
            if self.workdir:
                docker_cmd.extend(["-v", f"{self.workdir}:/workspace:ro"])
                docker_cmd.extend(["-w", "/workspace"])
            
            docker_cmd.extend([
                self.image,
                "bash", "-c", command
            ])
        else:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Invalid combination: language={language}, code={code is not None}, command={command is not None}",
            }
        
        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Execution timeout after {self.timeout} seconds",
            }
        except Exception as e:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": f"Docker execution failed: {str(e)}",
            }


@dataclass
class BashTool:
    """Restricted bash tool with configurable working directory.
    
    Note: For better security, consider using DockerTool instead.
    """

    workdir: Path
    timeout: int = 20
    use_docker: bool = False
    docker_image: str = "python:3.11-slim"

    def __call__(self, command: str) -> Dict[str, Any]:
        if self.use_docker:
            docker_tool = DockerTool(
                image=self.docker_image,
                timeout=self.timeout,
                workdir=self.workdir,
            )
            return docker_tool(command=command, language="bash")
        
        # Fallback to local execution
        proc = subprocess.run(
            command,
            shell=True,
            cwd=self.workdir,
            capture_output=True,
            text=True,
            timeout=self.timeout,
        )
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }


@dataclass
class SearchTool:
    """Simple DuckDuckGo search wrapper for sandbox lookups."""

    def __call__(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        topics = data.get("RelatedTopics", [])[:max_results]
        results: List[Dict[str, str]] = []
        for item in topics:
            if "Text" in item and "FirstURL" in item:
                results.append({"title": item.get("Text", ""), "url": item.get("FirstURL", "")})
        if not results and data.get("Heading"):
            results.append({"title": data["Heading"], "url": url})
        return results

@dataclass
class SerperSearchTool:
    """Serper API search wrapper for high-quality web search results."""

    api_key: str
    base_url: str = "https://google.serper.dev"

    def __call__(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "search",
    ) -> Dict[str, Any]:
        """
        Search using Serper API.
        
        Args:
            query: Search query string
            max_results: Maximum number of results to return
            search_type: Type of search - "search" (default) or "images" or "videos"
        
        Returns:
            Dictionary containing organic results, answerBox, knowledgeGraph, etc.
        """
        url = f"{self.base_url}/{search_type}"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = json.dumps({
            "q": query,
            "num": max_results,
        })
        
        response = requests.request("POST", url, headers=headers, data=payload)
        return json.loads(response.text)

    def get_organic_results(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Get only organic search results."""
        data = self(query, max_results=max_results)
        return data.get("organic", [])[:max_results]

    def get_answer_box(self, query: str) -> Dict[str, Any]:
        """Get answer box if available."""
        data = self(query, max_results=1)
        return data.get("answerBox", {})

    def get_knowledge_graph(self, query: str) -> Dict[str, Any]:
        """Get knowledge graph if available."""
        data = self(query, max_results=1)
        return data.get("knowledgeGraph", {})


@dataclass
class SandboxFusionTool:
    """Secure code execution tool using SandboxFusion service.
    
    SandboxFusion is a secure code sandbox that supports 23+ programming languages.
    It provides safe execution environment for LLM-generated code.
    """

    base_url: str = "http://localhost:8080"
    timeout: int = 30
    default_language: str = "python"

    def __call__(self, code: str, language: str | None = None) -> Dict[str, Any]:
        """Execute code in SandboxFusion sandbox.
        
        Args:
            code: Code to execute
            language: Programming language (default: python)
            
        Returns:
            Dict with execution results including:
            - status: Execution status
            - stdout: Standard output
            - stderr: Standard error
            - execution_time: Execution time in seconds
            - return_code: Return code (if applicable)
        """
        url = f"{self.base_url.rstrip('/')}/run_code"
        payload = {
            "code": code,
            "language": language or self.default_language,
        }
        
        try:
            resp = requests.post(url, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
            
            # Normalize response format
            return {
                "status": result.get("status", "unknown"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time", 0),
                "return_code": result.get("return_code", 0),
                "raw": result,  # Keep raw response for debugging
            }
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "stdout": "",
                "stderr": f"SandboxFusion request failed: {str(e)}",
                "execution_time": 0,
                "return_code": -1,
                "raw": {},
            }


@dataclass
class ToolRegistry:
    """Registry that manages tools exposed to synthesis and verification."""

    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, name: str, description: str, func: Callable[..., Any]) -> None:
        self.tools[name] = Tool(name=name, description=description, handler=func)

    def ensure_defaults(self, bash: BashTool, search: SearchTool, sandbox_fusion: SandboxFusionTool | None = None) -> None:
        if "bash" not in self.tools:
            self.register("bash", "Execute bash commands inside the sandbox", bash)
        if "search" not in self.tools:
            self.register("search", "Search the web via DuckDuckGo", search)
        if sandbox_fusion is not None and "sandbox_fusion" not in self.tools:
            self.register("sandbox_fusion", "Execute code securely in SandboxFusion sandbox (supports 23+ languages)", sandbox_fusion)

    def as_callable_dict(self) -> Dict[str, Callable[..., Any]]:
        return {name: tool.handler for name, tool in self.tools.items()}

    def describe(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self.tools.values()]

