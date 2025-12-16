from __future__ import annotations

import requests
from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class SandboxFusionExecutor:
    """Secure code execution environment using SandboxFusion service.
    
    This is an execution environment, not a tool. It executes the entire
    solution/verification logic in an isolated container.
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
