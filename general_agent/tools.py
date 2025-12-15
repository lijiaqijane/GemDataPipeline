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
class ToolRegistry:
    """Registry that manages tools exposed to synthesis and verification."""

    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, name: str, description: str, func: Callable[..., Any]) -> None:
        self.tools[name] = Tool(name=name, description=description, handler=func)

    def ensure_defaults(self, bash: BashTool, search: SearchTool) -> None:
        """Register default tools. Note: SandboxFusion is an execution environment, not a tool."""
        if "bash" not in self.tools:
            self.register("bash", "Execute bash commands inside the sandbox", bash)
        if "search" not in self.tools:
            self.register("search", "Search the web via DuckDuckGo", search)

    def as_callable_dict(self) -> Dict[str, Callable[..., Any]]:
        return {name: tool.handler for name, tool in self.tools.items()}

    def describe(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self.tools.values()]

