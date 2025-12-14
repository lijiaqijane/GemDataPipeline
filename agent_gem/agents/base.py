from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent_gem.core.task_schema import EvaluationCriteria, TaskDefinition, TaskPackage, ToolSpec
from agent_gem.core.validation import validate_task_package
from agent_gem.database import LocalDatabase
from agent_gem.llm import LLMClient
from agent_gem.sandbox import SandboxExecutor

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest


logger = logging.getLogger(__name__)


@dataclass
class TaskStep:
    parentUuid: uuid.UUID | None
    sessionId: uuid.UUID | None
    message: dict[str, Any]
    requestId: str
    uuid: uuid.UUID
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "parentUuid": str(self.parentUuid) if self.parentUuid else None,
            "sessionId": str(self.sessionId) if self.sessionId else None,
            "message": self.message,
            "requestId": self.requestId,
            "uuid": str(self.uuid),
            "timestamp": self.timestamp,
        }


@dataclass
class TaskState:
    """Tracks the current state during task synthesis"""

    current_difficulty: str = "easy"
    steps: List[TaskStep] = field(default_factory=list)
    session_id: uuid.UUID = field(default_factory=uuid.uuid4)
    task_id: uuid.UUID = field(default_factory=lambda _: f"req_{uuid.uuid4().hex}")

    def add_step(
        self,
        message: dict[str, Any],
        *,
        task_id: str | None = None,
        parent_id: uuid.UUID | None = None,
    ) -> TaskStep:
        step = TaskStep(
            parentUuid=(parent_id or self.steps[-1].uuid if len(self.steps) > 0 else None),
            sessionId=self.session_id,
            message=message,
            task_id=task_id or self.task_id,
            uuid=uuid.uuid4(),
        )
        self.steps.append(step)
        self.parent_uuid = step.uuid
        return step


class BaseAgent:
    agent_type: str = "base"
    description: str = "Base agent"

    def __init__(self, llm: LLMClient, taskdb_root: str = "taskdb") -> None:
        self.llm = llm
        self.sandbox: Optional[SandboxExecutor] = None
        self.taskdb_root = taskdb_root
        self.taskdb = LocalDatabase(Path(taskdb_root))

        self.db: Optional[LocalDatabase] = None
        self._trivial_solution_patterns = [
            r"return\s+list\(tools\.keys\(\)\)",
            r"return\s+tools\.keys\(\)",
            r"return\s+\[.*tools",
            r"return\s+tools",
        ]
        self._trivial_verifier_patterns = [
            r"return\s+isinstance\(answer,\s*list\)",
            r"return\s+True",
        ]

    def register_tools(self) -> None:
        """Hook for subclasses to register additional tools on the sandbox."""

    def generate(self, request: GenerationRequest) -> Optional[TaskPackage]:
        self.reset()
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
        package = self._parse_response(raw, request)
        if not package:
            logger.warning("[agent:%s] No tasks parsed; using fallback task.", self.agent_type)
            return None
        if not request.validate:
            return package
        validated = None
        try:
            validated = validate_task_package(package)
            logger.info(
                "[agent:%s] Accepted task: %s [%d]",
                self.agent_type,
                package.task.task_title,
                package.task.difficulty_level,
            )
            logger.debug(
                "[agent:%s] Solution preview: %s",
                self.agent_type,
                _preview_text(package.solution),
            )
            logger.debug(
                "[agent:%s] Verification preview: %s",
                self.agent_type,
                _preview_text(package.verification),
            )
        except Exception as exc:
            logger.warning(
                "[agent:%s] Dropping task due to validation error: %s",
                self.agent_type,
                exc,
            )
        return validated

    def _build_prompt(self, request: GenerationRequest) -> str:
        topic_hint = (
            f"Topic: {request.topic}." if request.topic else "Topic: choose a high-value category yourself."
        )
        return (
            f"You are the {self.agent_type} for RL environment generation. "
            "Produce exactly 1 task as a JSON object (or a single-element array) with: "
            "{task_title, task_content, submit_result_format, difficulty_level, evaluation_criteria, solution, verification}. "
            f"{topic_hint} Difficulty_level: {request.difficulty}. "
            "Return only JSON."
        )

    def _parse_response(self, raw: str, request: GenerationRequest) -> Optional[TaskPackage]:
        data = self._extract_json(raw)
        if not data:
            return None
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        if not isinstance(data, dict):
            return None
        return self._build_package(data, request)

    def _build_package(self, item: Dict[str, object], request: GenerationRequest) -> Optional[TaskPackage]:
        try:
            tool_specs: List[ToolSpec] = []
            if request.seed_tools:
                tool_specs = [
                    spec if isinstance(spec, ToolSpec) else ToolSpec(**spec) for spec in request.seed_tools
                ]
            else:
                tool_specs = self._default_tools()

            evaluation = item.get("evaluation_criteria", {})  # type: ignore[assignment]
            criteria = (
                EvaluationCriteria(**evaluation)
                if not isinstance(evaluation, EvaluationCriteria)
                else evaluation
            )
            fallback_topic = request.topic or "a relevant domain you select"
            submit_result_format = item.get("submit_result_format", {})
            if isinstance(submit_result_format, str):
                submit_result_format = {"type": submit_result_format}
            task = TaskDefinition(
                task_title=str(item.get("task_title") or f"{self.agent_type.title()} Task"),
                task_content=str(item.get("task_content") or f"Create a task about {fallback_topic}."),
                submit_result_format=submit_result_format,  # type: ignore[arg-type]
                tool_set=tool_specs,
                evaluation_criteria=criteria,
                difficulty_level=int(item.get("difficulty_level") or request.difficulty),
            )
            solution = item.get("solution") or self._default_solution(task.task_title)
            verification = item.get("verification") or self._default_verification()
            metadata = {
                "source": "llm",
                "item_chars": str(len(json.dumps(item, ensure_ascii=False))),
            }
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
        def search(query: str, k: int = 5) -> list[str]:
            """Search over curated sources or web mirrors."""
            raise RuntimeError("tool spec only")

        def bash(command: str) -> dict[str, Any]:
            """Execute shell commands in an isolated sandbox."""
            raise RuntimeError("tool spec only")

        return [
            ToolSpec.from_function(search, name="search"),
            ToolSpec.from_function(bash, name="bash"),
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


def _preview_text(text: str, limit: int = 240) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return f"{cleaned[:limit]}... (truncated)"
