from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List

from agent_gem.core.task_schema import TaskPackage
from agent_gem.core.utils import dump_json, slugify

logger = logging.getLogger(__name__)


class SandboxManager:
    """Creates isolated sandboxes for generated tasks."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def persist(self, packages: Iterable[TaskPackage]) -> List[TaskPackage]:
        updated: List[TaskPackage] = []
        for package in packages:
            slug = slugify(package.task.task_title)
            task_dir = self.root / package.agent_type / slug
            task_dir.mkdir(parents=True, exist_ok=True)
            payload = package.as_payload() | {"slug": slug}
            dump_json(task_dir / "task.json", payload)
            logger.info(
                "Sandbox persisted: agent=%s title='%s' -> %s",
                package.agent_type,
                package.task.task_title,
                task_dir,
            )
            updated.append(package.copy(update={"task_path": str(task_dir)}))
        return updated
