from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from urllib.parse import urlparse
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent_gem.core.task_schema import TaskPackage, TaskStep
from agent_gem.core.utils import dump_json, slugify

logger = logging.getLogger(__name__)


@dataclass
class TaskWriter:
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
    
    def merge_records(self, new_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new records with existing records, deduplicating by title and URL."""
        def _canon_url(value: str) -> str:
            if not value:
                return ""
            try:
                parsed = urlparse(value)
            except Exception:
                return value.strip()
            scheme = (parsed.scheme or "https").lower()
            netloc = parsed.netloc.lower()
            path = parsed.path or ""
            if path != "/" and path.endswith("/"):
                path = path[:-1]
            return f"{scheme}://{netloc}{path}"

        existing_titles = {
            r.get("title", "").lower().strip() for r in self.records if r.get("title")
        }
        existing_urls = {
            _canon_url(r.get("url", ""))
            for r in self.records
            if r.get("url")
        }
        merged = list(self.records)
        for record in new_records:
            title = record.get("title", "").lower().strip()
            url = _canon_url(record.get("url", ""))
            if url and url in existing_urls:
                continue
            if title and title in existing_titles:
                continue
            merged.append(record)
            if title:
                existing_titles.add(title)
            if url:
                existing_urls.add(url)
        return merged

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
        return Path(self.root, agent_type, f"task-{task_id}")

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
        }
        if extra:
            payload.update(extra)
        path = Path(task_dir, f"{task_id}.json")
        dump_json(path, payload)
        with open(Path(task_dir, f"{task_id}.jsonl"), "w") as f:
            for step in steps:
                f.write(
                    json.dumps(
                        step.to_payload() if isinstance(step, TaskStep) else step
                    )
                    + "\n"
                )
        return path