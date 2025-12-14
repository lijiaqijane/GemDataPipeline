from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agent_gem.tools.base import BaseTool, ToolExecutionError


class DockerTool(BaseTool):
    """Execute code and commands in a Docker container for isolation."""

    def __init__(
        self,
        *,
        image: str = "python:3.11-slim",
        timeout_s: int = 30,
        workdir: Path | None = None,
        memory_limit: str = "512m",
        cpu_limit: str = "1.0",
        name: str = "docker",
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.image = image
        self.timeout_s = timeout_s
        self.workdir = workdir
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit

    def execute(
        self,
        code: str | None = None,
        command: str | None = None,
        language: str = "python",
    ) -> dict[str, Any]:
        if code is None and command is None:
            raise ToolExecutionError(
                "invalid_input",
                {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "Either 'code' or 'command' must be provided",
                },
            )

        if language == "python" and code:
            docker_cmd: list[str] = [
                "docker",
                "run",
                "--rm",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                "--network=none",
                "--read-only",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=100m",
            ]
            if self.workdir:
                docker_cmd.extend(["-v", f"{self.workdir}:/workspace:ro", "-w", "/workspace"])
            docker_cmd.extend([self.image, "python", "-c", code])
        elif language == "bash" and command:
            docker_cmd = [
                "docker",
                "run",
                "--rm",
                f"--memory={self.memory_limit}",
                f"--cpus={self.cpu_limit}",
                "--network=none",
                "--read-only",
                "--tmpfs=/tmp:rw,noexec,nosuid,size=100m",
            ]
            if self.workdir:
                docker_cmd.extend(["-v", f"{self.workdir}:/workspace:ro", "-w", "/workspace"])
            docker_cmd.extend([self.image, "bash", "-c", command])
        else:
            raise ToolExecutionError(
                "invalid_input",
                {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": (
                        "Invalid combination: "
                        f"language={language}, code={code is not None}, command={command is not None}"
                    ),
                },
            )

        try:
            proc = subprocess.run(
                docker_cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
            )
            return {
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
            }
        except subprocess.TimeoutExpired as exc:
            raise ToolExecutionError(
                "timeout",
                {
                    "returncode": -1,
                    "stdout": exc.stdout or "",
                    "stderr": (exc.stderr or "") + f"\nTimeout after {self.timeout_s}s",
                },
            )
        except Exception as exc:
            raise ToolExecutionError(
                "docker_failed",
                {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": f"Docker execution failed: {exc}",
                },
            )
