from __future__ import annotations

from typing import TYPE_CHECKING, List

from agent_gem.core.task_schema import ToolSpec

from .base import BaseAgent

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.env.generator import GenerationRequest


class CodeInterpreterAgent(BaseAgent):
    agent_type = "code_interpreter_agent"
    description = "Generates notebook-style reasoning and execution tasks"

    def _build_prompt(self, request: GenerationRequest) -> str:
        return (
            "You are the Code Interpreter Agent designing math/data-science tasks requiring execution. "
            f"Create exactly 1 task for '{request.topic or 'a domain you choose'}' that requires code to solve. "
            "Return JSON with task_title, task_content (include dataset snippet or function signature), "
            "submit_result_format (e.g., 'notebook cells', 'json metrics'), tool_set, evaluation_criteria, difficulty_level, "
            "solution (reference python code), verification (verify(tools, answer) validating numerical closeness or dataframe shape). "
            "Encourage multiple reasoning steps and validation of outputs. Return only JSON."
        )

    def _default_tools(self) -> List[ToolSpec]:
        def python_runner(code: str) -> dict[str, object]:
            """Execute Python code with scientific stack."""
            raise RuntimeError("tool spec only")

        def dataset_loader(name: str) -> object:
            """Load provided in-sandbox datasets."""
            raise RuntimeError("tool spec only")

        return [
            ToolSpec.from_function(python_runner, name="python_runner"),
            ToolSpec.from_function(dataset_loader, name="dataset_loader"),
        ]
