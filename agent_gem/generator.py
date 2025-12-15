from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Type

from agent_gem.agents import BaseAgent, CodeAgent, CodeInterpreterAgent, GeneralAgent, SearchAgent
from agent_gem.core import TaskPackage, ToolSpec, score_task, validate_task_package
from agent_gem.database import LocalDatabase

logger = logging.getLogger(__name__)


@dataclass
class GenerationRequest:
    agent_type: str
    topic: Optional[str] = None
    num: int = 1
    difficulty: int = 1
    validate: bool = True
    use_sandbox_fusion: bool = False
    use_docker: bool = False
    max_refine_rounds: int = 1
    max_validation_rounds: int = 1
    persist_result: bool = True
    fail_soft: bool = False
    seed_tools: Optional[List[ToolSpec]] = None


class EnvironmentGenerator:
    """Central orchestrator that routes generation to the right agent and persists sandboxes."""

    def __init__(self, llm, taskdb: Path | str = "taskdb") -> None:
        self.llm = llm
        self.localdb = LocalDatabase(root=Path(taskdb))
        self.agent_factories: Dict[str, Type[BaseAgent]] = {
            "search_agent": SearchAgent,
            "code_agent": CodeAgent,
            "code_interpreter_agent": CodeInterpreterAgent,
            "general_agent": GeneralAgent,
        }

    def generate(self, request: GenerationRequest) -> List[TaskPackage]:
        logger.info(
            "Starting generation: agent=%s topic=%s count=%d difficulty=%s taskdb_root=%s",
            request.agent_type,
            request.topic or "auto-generate",
            request.num,
            request.difficulty,
            self.localdb.root,
        )
        agent = self._resolve_agent(request.agent_type)
        packages: List[TaskPackage] = []
        for idx in range(request.num):
            logger.debug("Invoking agent iteration %d/%d", idx + 1, request.num)
            generated = agent.generate(request)
            if generated:
                if request.validate:
                    logger.info("Validating package %d", idx + 1)
                    generated = validate_task_package(generated)
                packages.append(generated)

            else:
                logger.warning("Agent returned no packages on iteration %d", idx + 1)

        persisted = self.localdb.persist(packages) if request.persist_result else packages
        return self._prioritize(persisted)

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
