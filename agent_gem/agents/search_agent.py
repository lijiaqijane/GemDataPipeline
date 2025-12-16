from __future__ import annotations

from typing import TYPE_CHECKING, List

from agent_gem.core.task_schema import ToolSpec

from .base import BaseAgent

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest


class SearchAgent(BaseAgent):
    agent_type = "search_agent"
    description = "Generates long-tail QA and retrieval-heavy tasks"

    def _build_prompt(self, request: GenerationRequest) -> str:
        return (
            "You are the Search Agent crafting retrieval-heavy QA tasks. "
            f"Generate exactly 1 task targeting long-tail entities within '{request.topic or 'a topic you choose'}'. "
            "Each task must follow JSON fields: task_title, task_content, submit_result_format, "
            "tool_set (list of {tool_name, tool_description, tool_functionality}), "
            "evaluation_criteria {correctness, diversity, complexity, solution_verifiability}, difficulty_level, "
            "solution, verification. Solution should include an expected answer string and brief rationale. "
            "Verification must be Python code with verify(tools, answer) to compare against the expected answer "
            "case-insensitively. Prefer difficult reasoning and multi-hop retrieval. Return only JSON."
        )

    def _default_tools(self) -> List[ToolSpec]:
        def search(query: str, k: int = 5) -> list[str]:
            """Retrieve passages from mirrored web corpora."""
            raise RuntimeError("tool spec only")

        def summarize(texts: list[str]) -> str:
            """Summarize retrieved passages."""
            raise RuntimeError("tool spec only")

        return [
            ToolSpec.from_function(search, name="search"),
            ToolSpec.from_function(summarize, name="summarize"),
        ]
