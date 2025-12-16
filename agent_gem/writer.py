from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
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

    def persist_quadruple_format(
        self,
        category: str,
        records: List[Dict[str, Any]],
        packages: Iterable[TaskPackage],
        output_path: Optional[Path] = None,
    ) -> Path:
        """Persist tasks in quadruple format compatible with general_agent output.

        Format: <environment, tools, task, verifier>
        Output structure matches general_agent/synthesis/_persist format.

        Args:
            category: Task category name
            records: Database records for the environment
            packages: Task packages to persist
            output_path: Optional path to write tasks.json (default: root/tasks.json)

        Returns:
            Path to the written tasks.json file
        """
        packages_list = list(packages)
        if not packages_list:
            logger.warning("No packages to persist in quadruple format")
            return output_path or (self.root / "tasks.json")

        # Collect all tools from all packages
        all_tools = []
        tool_names_seen = set()
        for package in packages_list:
            for tool_spec in package.task.tool_set:
                if tool_spec.name not in tool_names_seen:
                    all_tools.append({
                        "name": tool_spec.name,
                        "description": tool_spec.description,
                    })
                    tool_names_seen.add(tool_spec.name)

        # Build tasks with task and verifier structure
        tasks_with_verifiers = []
        for package in packages_list:
            task_entry = {
                # Task part: task definition
                "task": {
                    "name": package.task.task_title,
                    "description": package.task.task_content,
                    "difficulty": package.task.difficulty_level,
                    "solution_code": package.solution or "",
                },
                # Verifier part: verifier definition
                "verifier": {
                    "verification_code": package.verification or "",
                },
                # Retain complete information (backward compatible)
                "name": package.task.task_title,
                "description": package.task.task_content,
                "difficulty": package.task.difficulty_level,
                "solution_code": package.solution or "",
                "verification_code": package.verification or "",
            }
            tasks_with_verifiers.append(task_entry)

        # Build quadruple format payload
        payload = {
            # Standard quadruple format
            "environment": {
                "category": category,
                "records": records,
                "record_count": len(records),
            },
            "tools": all_tools,
            "tasks": tasks_with_verifiers,
            # Compatible fields (backward compatible)
            "category": category,
            "tooling": all_tools,  # Same as tools
            "records": records,  # Same as environment.records
            # Metadata
            "metadata": {
                "version": "1.0",
                "format": "quadruple",  # <environment, tools, task, verifier>
                "task_count": len(packages_list),
                "tool_count": len(all_tools),
                "generation_timestamp": datetime.now().isoformat(),
            },
        }

        target = output_path or (self.root / "tasks.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        dump_json(target, payload)
        logger.info("Quadruple format tasks saved to %s", target)
        return target
