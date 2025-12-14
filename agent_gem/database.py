from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent_gem.core.task_schema import TaskPackage
from agent_gem.core.utils import dump_json, slugify

logger = logging.getLogger(__name__)


@dataclass
class LocalDatabase:
    """Lightweight JSON storage for scraped or generated data."""

    root: Path
    records: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.path = self.root / "db.json"
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text())
                self.records = data.get("records", []) if isinstance(data, dict) else []
            except Exception:
                self.records = []

    def persist(self, packages: Iterable[TaskPackage]) -> List[TaskPackage]:
        updated: List[TaskPackage] = []
        for package in packages:
            task_dir = self.root / package.agent_type / f"task-{package.task.task_id}"
            task_dir.mkdir(parents=True, exist_ok=True)
            payload = package.as_payload()
            dump_json(task_dir / "task.json", payload)
            logger.info(
                "LocalDB task persisted: agent=%s title='%s' -> %s",
                package.agent_type,
                package.task.task_title,
                task_dir,
            )
            updated.append(package.copy(update={"task_path": str(task_dir)}))
        return updated

    def task_dir(self, task_id: str, agent_type: str) -> Path:
        return self.root / agent_type / f"task-{task_id}"

    def record_steps(
        self,
        task_id: str,
        agent_type: str,
        steps: List[Dict[str, Any]],
        extra: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Persist a structured step trace for a task under its sandbox directory."""
        task_dir = self.task_dir(task_id, agent_type)
        task_dir.mkdir(parents=True, exist_ok=True)
        payload: Dict[str, Any] = {
            "task_id": task_id,
            "agent_type": agent_type,
            "steps": steps,
        }
        if extra:
            payload.update(extra)
        path = task_dir / f"{task_id}.jsonl"
        dump_json(path, payload)
        return path
