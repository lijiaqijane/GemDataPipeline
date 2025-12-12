from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from agent_gem.core.task_schema import EvaluationCriteria, TaskDefinition, TaskPackage, ToolSpec
from agent_gem.core.validation import validate_task_package
from agent_gem.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class AgentRequest:
    topic: Optional[str] = None
    difficulty: str = "Medium"
    seed_tools: Optional[List[ToolSpec]] = None
    submit_result_format: str = "json"
    hints: Dict[str, str] = field(default_factory=dict)


@dataclass
class TaskState:
    """Tracks the current state during task synthesis"""

    current_tools: List[ToolSpec]
    database_content: Dict[str, Any]
    current_difficulty: str
    task_history: List[Dict[str, Any]] = field(default_factory=list)
    tool_implementations: Dict[str, str] = field(default_factory=dict)


class BaseAgent:
    agent_type: str = "base"
    description: str = "Base agent"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.task_state = TaskState(
            current_tools=self._default_tools(),
            database_content={},
            current_difficulty="easy",
        )

    def generate(self, request: AgentRequest) -> List[TaskPackage]:
        logger.info(
            "[agent:%s] Generating 1 task (topic=%s, difficulty=%s)",
            self.agent_type,
            request.topic or "auto-generate",
            request.difficulty,
        )
        prompt = self._build_prompt(request)
        logger.debug("[agent:%s] Prompt preview: %s", self.agent_type, _preview_text(prompt))
        raw = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.55,
            max_tokens=1800,
        )
        logger.debug("[agent:%s] Raw completion: %s", self.agent_type, _preview_text(raw))
        packages = self._parse_response(raw, request)
        if not packages:
            logger.warning("[agent:%s] No tasks parsed; using fallback task.", self.agent_type)
            packages = [self._fallback_package(request)]
        validated = []
        for idx, pkg in enumerate(packages, start=1):
            try:
                validated.append(validate_task_package(pkg))
                logger.info(
                    "[agent:%s] Accepted task %d: %s [%s]",
                    self.agent_type,
                    idx,
                    pkg.task.task_title,
                    pkg.task.difficulty_level,
                )
                logger.debug(
                    "[agent:%s] Solution preview: %s",
                    self.agent_type,
                    _preview_text(pkg.solution),
                )
                logger.debug(
                    "[agent:%s] Verification preview: %s",
                    self.agent_type,
                    _preview_text(pkg.verification),
                )
            except Exception as exc:
                logger.warning(
                    "[agent:%s] Dropping task %d due to validation error: %s",
                    self.agent_type,
                    idx,
                    exc,
                )
                continue
        return validated

    def _build_prompt(self, request: AgentRequest) -> str:
        topic_hint = (
            f"Topic: {request.topic}." if request.topic else "Topic: choose a high-value category yourself."
        )
        return (
            f"You are the {self.agent_type} for RL environment generation. "
            "Produce exactly 1 task as a JSON object (or a single-element array) matching the schema: "
            "{task_title, task_content, submit_result_format, tool_set:[{tool_name, tool_description, tool_functionality}], "
            "evaluation_criteria:{correctness, diversity, complexity, solution_verifiability}, difficulty_level, "
            "solution, verification}. "
            f"{topic_hint} Difficulty: {request.difficulty}. "
            "Return only JSON."
        )

    def _parse_response(self, raw: str, request: AgentRequest) -> List[TaskPackage]:
        data = self._extract_json(raw)
        if not data:
            return []
        if isinstance(data, dict):
            data = [data]
        packages: List[TaskPackage] = []
        for item in data:
            pkg = self._build_package(item, request)
            if pkg:
                packages.append(pkg)
        return packages

    def _build_package(self, item: Dict[str, object], request: AgentRequest) -> Optional[TaskPackage]:
        try:
            tools = item.get("tool_set") or request.seed_tools or self._default_tools()
            tool_specs = [ToolSpec(**tool) if not isinstance(tool, ToolSpec) else tool for tool in tools]  # type: ignore[arg-type]
            evaluation = item.get("evaluation_criteria", {})  # type: ignore[assignment]
            criteria = (
                EvaluationCriteria(**evaluation)
                if not isinstance(evaluation, EvaluationCriteria)
                else evaluation
            )
            fallback_topic = request.topic or "a relevant domain you select"
            task = TaskDefinition(
                task_title=item.get("task_title", f"{self.agent_type.title()} Task"),  # type: ignore[arg-type]
                task_content=item.get("task_content", f"Create a task about {fallback_topic}."),  # type: ignore[arg-type]
                submit_result_format=item.get("submit_result_format", request.submit_result_format),  # type: ignore[arg-type]
                tool_set=tool_specs,
                evaluation_criteria=criteria,
                difficulty_level=item.get("difficulty_level", request.difficulty),  # type: ignore[arg-type]
            )
            solution = item.get("solution") or self._default_solution(task.task_title)
            verification = item.get("verification") or self._default_verification()
            metadata = {"prompt_tokens": str(len(raw) if isinstance(raw := json.dumps(item), str) else 0)}
            return TaskPackage(
                task=task,
                solution=str(solution),
                verification=str(verification),
                agent_type=self.agent_type,
                metadata=metadata,
            )
        except Exception:
            return None

    def _default_tools(self) -> List[ToolSpec]:
        return [
            ToolSpec(
                tool_name="search",
                tool_description="Search over curated sources or web mirrors.",
                tool_functionality="search(query: str) -> list[str]",
            ),
            ToolSpec(
                tool_name="bash",
                tool_description="Execute shell commands in an isolated sandbox.",
                tool_functionality="bash(command: str) -> {stdout, stderr, returncode}",
            ),
        ]

    def _default_solution(self, title: str) -> str:
        return (
            "def solve(tools):\n"
            f"    # Placeholder solution for: {title}\n"
            "    return {'status': 'pending'}\n"
        )

    def _default_verification(self) -> str:
        return "def verify(tools, answer):\n" "    return isinstance(answer, dict) and 'status' in answer\n"

    def _extract_json(self, raw: str) -> object:
        text = raw.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        fence = text.split("```")
        for block in fence:
            block = block.strip()
            if not block:
                continue
            try:
                return json.loads(block)
            except json.JSONDecodeError:
                continue
        if "[" in text and "]" in text:
            start = text.find("[")
            end = text.rfind("]")
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        if "{" in text and "}" in text:
            start = text.find("{")
            end = text.rfind("}")
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None

    def _fallback_package(self, request: AgentRequest) -> TaskPackage:
        tools = request.seed_tools or self._default_tools()
        fallback_topic = request.topic or "a relevant domain you select"
        task = TaskDefinition(
            task_title=f"{self.agent_type.title()} Fallback Task",
            task_content=f"Synthesize a verifiable scenario about {fallback_topic} with {self.agent_type}.",
            submit_result_format=request.submit_result_format,
            tool_set=tools,
            difficulty_level=request.difficulty,
        )
        return TaskPackage(
            task=task,
            solution=self._default_solution(task.task_title),
            verification=self._default_verification(),
            agent_type=self.agent_type,
        )


def _preview_text(text: str, limit: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}... (truncated)"
