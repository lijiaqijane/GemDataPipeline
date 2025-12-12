from __future__ import annotations

from typing import List

from agent_gem.core.task_schema import ToolSpec

from .base import AgentRequest, BaseAgent


class CodeAgent(BaseAgent):
    agent_type = "code_agent"
    description = "Synthesizes code issue + patch validation tasks"

    def _build_prompt(self, request: AgentRequest) -> str:
        return (
            "You are the Code Agent creating repository repair tasks. "
            f"Generate exactly 1 task for topic '{request.topic or 'a codebase you choose'}' that mimics issue/PR workflows. "
            "Each task JSON must include: task_title, task_content describing the bug and expectations, "
            "submit_result_format (e.g., 'patch diff' or 'pytest output'), tool_set, evaluation_criteria, "
            "difficulty_level, solution (reference patch or algorithm sketch), verification (python function verify(tools, answer)). "
            "Verification should check that tests would pass or invariants hold, not just type checks. Return only JSON."
        )

    def _default_tools(self) -> List[ToolSpec]:
        return [
            ToolSpec(
                tool_name="git",
                tool_description="Inspect repository state and apply patches.",
                tool_functionality="git(command: str) -> str",
            ),
            ToolSpec(
                tool_name="tests",
                tool_description="Run the project's test suite.",
                tool_functionality="tests(target: str = 'all') -> {stdout, stderr, returncode}",
            ),
            ToolSpec(
                tool_name="analyzer",
                tool_description="Static analysis for code quality.",
                tool_functionality="analyzer(file: str) -> list[str]",
            ),
        ]
