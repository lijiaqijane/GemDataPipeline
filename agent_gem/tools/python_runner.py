from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from agent_gem.tools.base import BaseTool, ToolExecutionError


class PythonRunnerTool(BaseTool):
    """Execute Python code via `python -c` inside the sandbox directory."""

    def __init__(
        self,
        *,
        workdir: Path,
        timeout_s: int = 20,
        name: str = "python_runner",
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.workdir = workdir
        self.timeout_s = timeout_s

    def execute(self, code: str, timeout_s: int | None = None) -> dict[str, Any]:
        effective_timeout = timeout_s or self.timeout_s
        try:
            proc = subprocess.run(
                [sys.executable, "-c", code],
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
