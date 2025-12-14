from __future__ import annotations

from typing import TYPE_CHECKING, List

from agent_gem.core.task_schema import ToolSpec

from .base import BaseAgent

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.env.generator import GenerationRequest


class CodeAgent(BaseAgent):
    agent_type = "code_agent"
    description = "Synthesizes code issue + patch validation tasks"

    def _build_prompt(self, request: GenerationRequest) -> str:
        return (
            "You are the Code Agent creating repository repair tasks. "
            f"Generate exactly 1 task for topic '{request.topic or 'a codebase you choose'}' that mimics issue/PR workflows. "
            "Each task JSON must include: task_title, task_content describing the bug and expectations, "
            "submit_result_format (e.g., 'patch diff' or 'pytest output'), tool_set, evaluation_criteria, "
            "difficulty_level, solution (reference patch or algorithm sketch), verification (python function verify(tools, answer)). "
            "Verification should check that tests would pass or invariants hold, not just type checks. Return only JSON."
        )

    def _default_tools(self) -> List[ToolSpec]:
        def git(command: str) -> str:
            """Inspect repository state and apply patches."""
            raise RuntimeError("tool spec only")

        def tests(target: str = "all") -> dict[str, object]:
            """Run the project's test suite."""
            raise RuntimeError("tool spec only")

        def analyzer(file: str) -> list[str]:
            """Static analysis for code quality."""
            raise RuntimeError("tool spec only")

        return [
            ToolSpec.from_function(git, name="git"),
            ToolSpec.from_function(tests, name="tests"),
            ToolSpec.from_function(analyzer, name="analyzer"),
        ]
