from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

from agent_gem.agents import (
    AgentRequest,
    BaseAgent,
    CodeAgent,
    CodeInterpreterAgent,
    GeneralAgent,
    SearchAgent,
)
from agent_gem.core import TaskPackage, score_task, validate_task_package
from agent_gem.sandbox import SandboxManager

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    agent_type: str
    topic: Optional[str] = None
    count: int = 1
    difficulty: str = "Medium"
    sandbox_root: Path = Path("sandbox")
    validate: bool = True
    submit_result_format: str = "json"


class EnvironmentGenerator:
    """Central orchestrator that routes generation to the right agent and persists sandboxes."""

    def __init__(self, llm, sandbox_root: Path | str = "sandbox") -> None:
        self.llm = llm
        self.sandbox = SandboxManager(Path(sandbox_root))
        self.agent_factories: Dict[str, Type[BaseAgent]] = {
            "search_agent": SearchAgent,
            "code_agent": CodeAgent,
            "code_interpreter_agent": CodeInterpreterAgent,
            "general_agent": GeneralAgent,
        }

    def generate(self, request: GenerationRequest) -> List[TaskPackage]:
        logger.info(
            "Starting generation: agent=%s topic=%s count=%d difficulty=%s sandbox_root=%s",
            request.agent_type,
            request.topic or "auto-generate",
            request.count,
            request.difficulty,
            request.sandbox_root,
        )
        agent = self._resolve_agent(request.agent_type)
        packages: List[TaskPackage] = []
        for idx in range(request.count):
            logger.debug("Invoking agent iteration %d/%d", idx + 1, request.count)
            agent_request = AgentRequest(
                topic=request.topic,
                difficulty=request.difficulty,
                submit_result_format=request.submit_result_format,
            )
            generated = agent.generate(agent_request)
            if generated:
                packages.append(generated[0])
            else:
                logger.warning("Agent returned no packages on iteration %d", idx + 1)
        if request.validate:
            logger.info("Validating %d package(s)", len(packages))
            packages = [validate_task_package(pkg) for pkg in packages]

        packages = self.sandbox.persist(packages)
        logger.info(
            "Persisted %d package(s) to sandbox root '%s'",
            len(packages),
            self.sandbox.root,
        )
        return self._prioritize(packages)

    def _prioritize(self, packages: List[TaskPackage]) -> List[TaskPackage]:
        scored = [(score_task(pkg.task).composite, pkg) for pkg in packages]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        for rank, (composite, pkg) in enumerate(scored, start=1):
            logger.info("Rank #%d: %s (score=%.3f)", rank, pkg.task.summary(), composite)
        return [pkg for composite, pkg in scored]

    def _resolve_agent(self, agent_type: str) -> BaseAgent:
        key = agent_type.lower()
        if key not in self.agent_factories:
            raise ValueError(f"Unsupported agent type: {agent_type}")
        logger.debug("Resolved agent handler for type=%s", key)
        return self.agent_factories[key](self.llm)
