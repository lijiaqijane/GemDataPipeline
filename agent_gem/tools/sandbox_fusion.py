from __future__ import annotations

from typing import Any

import requests

from agent_gem.tools.base import BaseTool, ToolExecutionError


class SandboxFusionTool(BaseTool):
    """Execute code in a SandboxFusion service instance."""

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:8080",
        timeout_s: int = 30,
        default_language: str = "python",
        name: str = "sandbox_fusion",
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.base_url = base_url
        self.timeout_s = timeout_s
        self.default_language = default_language

    def execute(self, code: str, language: str | None = None) -> dict[str, Any]:
        url = f"{self.base_url.rstrip('/')}/run_code"
        payload = {
            "code": code,
            "language": language or self.default_language,
        }

        try:
            resp = requests.post(url, json=payload, timeout=self.timeout_s)
            resp.raise_for_status()
            result = resp.json()
            return {
                "status": result.get("status", "unknown"),
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "execution_time": result.get("execution_time", 0),
                "return_code": result.get("return_code", 0),
                "raw": result,
            }
        except requests.exceptions.RequestException as exc:
            raise ToolExecutionError(
                "sandbox_fusion_failed",
                {
                    "status": "error",
                    "stdout": "",
                    "stderr": f"SandboxFusion request failed: {exc}",
                    "execution_time": 0,
                    "return_code": -1,
                    "raw": {},
                },
            )
