from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from agent_gem.tools.base import BaseTool, ToolExecutionError
from agent_gem.tools.docker import DockerTool


class BashTool(BaseTool):
    """Execute shell commands with cwd pinned to the sandbox directory."""

    _BANNED_TOKENS = {
        "rm -rf",
        "shutdown",
        "reboot",
        ":(){:|:&};:",
        "mkfs",
        "dd if=",
        "sudo ",
    }

    def __init__(
        self,
        *,
        workdir: Path,
        timeout_s: int = 20,
        use_docker: bool = False,
        docker_image: str = "python:3.11-slim",
        name: str = "bash",
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.workdir = workdir
        self.timeout_s = timeout_s
        self.use_docker = use_docker
        self.docker_image = docker_image

    def execute(self, command: str, timeout_s: int | None = None) -> dict[str, Any]:
        lower = (command or "").lower()
        if any(token in lower for token in self._BANNED_TOKENS):
            raise ToolExecutionError(
                "blocked_command",
                {
                    "returncode": -1,
                    "stdout": "",
                    "stderr": "Command blocked by policy",
                },
            )

        effective_timeout = timeout_s or self.timeout_s

        if self.use_docker:
            docker_tool = DockerTool(
                image=self.docker_image,
                timeout_s=effective_timeout,
                workdir=self.workdir,
                name="docker",
            )
            return docker_tool.execute(command=command, language="bash")

        try:
            proc = subprocess.run(
                command,
                shell=True,
                cwd=self.workdir,
                capture_output=True,
                text=True,
                timeout=effective_timeout,
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
                    "stderr": (exc.stderr or "") + f"\nTimeout after {effective_timeout}s",
                },
            )
