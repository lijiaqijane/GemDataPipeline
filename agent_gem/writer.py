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
    
    def merge_records(self, new_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge new records with existing records, deduplicating by title."""
        existing_titles = {r.get("title", "").lower().strip() for r in self.records if r.get("title")}
        merged = list(self.records)
        for record in new_records:
            title = record.get("title", "").lower().strip()
            if title and title not in existing_titles:
                merged.append(record)
                existing_titles.add(title)
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

    def persist_quadruple_format(
        self,
        category: str,
        records: List[Dict[str, Any]],
        packages: Iterable[TaskPackage],
        output_path: Optional[Path] = None,
        merge: bool = True,
    ) -> Path:
        """Persist tasks in quadruple format compatible with general_agent output.

        Format: <environment, tools, task, verifier>
        Output structure matches general_agent/synthesis/_persist format.

        Extended fields:
        - setup_bundle: setup_env.py code + stdout/stderr/returncode
        - tools_interface: synthesized tools.py code (preview)
        - reward_function: verification code (full text)
        - reference_solution: solution code (full text)
        - state_hash: pre/post hashes captured during validation
        
        Args:
            category: Task category name
            records: Records for this category
            packages: Task packages to persist
            output_path: Optional custom output path (default: root/tasks.json)
            merge: If True, merge with existing tasks.json; if False, overwrite
        """
        packages_list = list(packages)
        if not packages_list:
            logger.warning("No packages to persist in quadruple format")
            return output_path or (self.root / "tasks.json")

        target = output_path or (self.root / "tasks.json")
        
        # Load existing tasks if merging
        existing_tasks = []
        existing_tools = []
        existing_task_ids = set()
        if merge and target.exists():
            try:
                existing_data = json.loads(target.read_text())
                existing_tasks = existing_data.get("tasks", [])
                existing_tools = existing_data.get("tools", [])
                existing_task_ids = {t.get("task", {}).get("id") or t.get("name", "") for t in existing_tasks}
            except Exception as e:
                logger.warning(f"Failed to load existing tasks.json for merging: {e}")

        first_meta = packages_list[0].metadata if packages_list else {}
        setup_bundle = None
        if first_meta:
            setup_bundle = {
                "script": first_meta.get("setup_code", ""),
                "stdout": first_meta.get("setup_stdout_preview", ""),
                "stderr": first_meta.get("setup_stderr_preview", ""),
                "returncode": first_meta.get("setup_returncode", ""),
            }
        tools_interface = first_meta.get("tools_code", "")

        # Collect all tools from all packages and existing tasks
        all_tools = list(existing_tools)  # Start with existing tools
        tool_names_seen = {t.get("name", "") for t in existing_tools}
        for package in packages_list:
            for tool_spec in package.task.tool_set:
                if tool_spec.name not in tool_names_seen:
                    all_tools.append(
                        {
                            "name": tool_spec.name,
                            "description": tool_spec.description,
                        }
                    )
                    tool_names_seen.add(tool_spec.name)

        # Build tasks with task and verifier structure
        tasks_with_verifiers = list(existing_tasks)  # Start with existing tasks
        for package in packages_list:
            # Skip if task already exists (by task_id)
            if package.task.task_id in existing_task_ids:
                logger.debug(f"Skipping duplicate task: {package.task.task_id}")
                continue
                
            task_entry = {
                # Task part: task definition
                "task": {
                    "id": package.task.task_id,
                    "name": package.task.task_title,
                    "description": package.task.task_content,
                    "difficulty": package.task.difficulty_level,
                    "solution_code": package.solution or "",
                    "reference_solution": package.solution or "",
                    "state_hash": {
                        "pre": package.metadata.get("pre_state_hash", ""),
                        "post": package.metadata.get("post_state_hash", ""),
                    },
                },
                # Verifier part: verifier definition
                "verifier": {
                    "verification_code": package.verification or "",
                    "reward_function": package.verification or "",
                },
                # Retain complete information (backward compatible)
                "name": package.task.task_title,
                "description": package.task.task_content,
                "difficulty": package.task.difficulty_level,
                "solution_code": package.solution or "",
                "reference_solution": package.solution or "",
                "verification_code": package.verification or "",
                "metadata": package.metadata,
            }
            tasks_with_verifiers.append(task_entry)

        # Merge records if merging (use existing records from db.json if available)
        final_records = records
        if merge and self.path.exists():
            try:
                db_data = json.loads(self.path.read_text())
                existing_db_records = db_data.get("records", [])
                if existing_db_records:
                    # Merge records, preferring existing ones (they may have more data)
                    record_titles = {r.get("title", "").lower() for r in existing_db_records}
                    for r in records:
                        if r.get("title", "").lower() not in record_titles:
                            existing_db_records.append(r)
                    final_records = existing_db_records
            except Exception:
                pass
        
        # Build quadruple format payload
        payload = {
            # Standard quadruple format
            "environment": {
                "category": category,
                "records": final_records,
                "record_count": len(final_records),
                "setup_bundle": setup_bundle or {},
            },
            "tools": all_tools,
            "tools_interface": tools_interface,
            "tasks": tasks_with_verifiers,
            # Compatible fields (backward compatible)
            "category": category,
            "tooling": all_tools,  # Same as tools
            "records": final_records,  # Same as environment.records
            # Metadata
            "metadata": {
                "version": "1.1",
                "format": "quadruple",  # <environment, tools, task, verifier>
                "task_count": len(tasks_with_verifiers),
                "tool_count": len(all_tools),
                "generation_timestamp": datetime.now().isoformat(),
                "merged": merge and target.exists(),
            },
        }

        target.parent.mkdir(parents=True, exist_ok=True)
        dump_json(target, payload)
        logger.info(
            "Quadruple format tasks saved to %s (tasks: %d, tools: %d, records: %d, merged: %s)",
            target,
            len(tasks_with_verifiers),
            len(all_tools),
            len(final_records),
            merge and target.exists(),
        )
        return target
