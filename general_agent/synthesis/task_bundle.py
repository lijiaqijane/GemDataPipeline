"""Task bundle for task definition, solution, and verification."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List

from .execution import execute_in_sandbox_fusion


@dataclass
class TaskBundle:
    """Bundle of task definition, solution code, and verification code."""

    name: str
    description: str
    difficulty: int
    solution_code: str
    verification_code: str
    use_sandbox_fusion: bool = True  # Default to using SandboxFusion, ensure all code executes in sandbox

    def run_solution(self, tools: Dict[str, Any], db_records: List[Dict[str, Any]] | None = None) -> Any:
        """Execute solution code in SandboxFusion.
        
        Args:
            tools: Dictionary of available tools
            db_records: Database records to inject into SandboxFusion (optional)
        """
        if not self.use_sandbox_fusion:
            raise RuntimeError("SandboxFusion is required for code execution. Set use_sandbox_fusion=True.")
        
        code = self._normalize_code(self.solution_code)
        return execute_in_sandbox_fusion(code, tools, "solve", db_records or [])

    def verify(self, tools: Dict[str, Any], answer: Any, db_records: List[Dict[str, Any]] | None = None) -> bool:
        """Execute verification code in SandboxFusion.
        
        Args:
            tools: Dictionary of available tools
            answer: The answer from run_solution to verify
            db_records: Database records to inject into SandboxFusion (optional)
        """
        if not self.use_sandbox_fusion:
            raise RuntimeError("SandboxFusion is required for code execution. Set use_sandbox_fusion=True.")
        
        code = self._normalize_code(self.verification_code)
        result = execute_in_sandbox_fusion(code, tools, "verify", db_records or [], answer)
        return bool(result)

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize common lowercase literals to valid Python to avoid trivial runtime errors."""
        if not code:
            return code
        # Replace standalone true/false/null (case-insensitive) with Python literals
        code = re.sub(r"\btrue\b", "True", code, flags=re.IGNORECASE)
        code = re.sub(r"\bfalse\b", "False", code, flags=re.IGNORECASE)
        code = re.sub(r"\bnull\b", "None", code, flags=re.IGNORECASE)
        return code

