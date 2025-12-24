from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from agent_gem.core.task_schema import TaskPackage
from agent_gem.core.utils import dump_json
from agent_gem.writer import TaskWriter

logger = logging.getLogger(__name__)


def persist_quadruple_format(
    writer: TaskWriter,
    *,
    category: str,
    records: List[Dict[str, Any]],
    packages: Iterable[TaskPackage],
    output_path: Optional[Path] = None,
    merge: bool = True,
) -> Path:
    """Persist tasks in quadruple format compatible with general_agent output."""
    packages_list = list(packages)
    if not packages_list:
        logger.warning("No packages to persist in quadruple format")
        return output_path or (writer.root / "tasks.json")

    target = output_path or (writer.root / "tasks.json")

    # Load existing tasks if merging
    existing_tasks = []
    existing_tools = []
    existing_task_keys = set()
    if merge and target.exists():
        try:
            existing_data = json.loads(target.read_text())
            existing_tasks = existing_data.get("tasks", [])
            existing_tools = existing_data.get("tools", [])
            for entry in existing_tasks:
                task = entry.get("task", {}) if isinstance(entry, dict) else {}
                task_id = task.get("id") or entry.get("name", "")
                difficulty = task.get("difficulty") or entry.get("difficulty")
                existing_task_keys.add((task_id, difficulty))
        except Exception as exc:
            logger.warning("Failed to load existing tasks.json for merging: %s", exc)

    first_meta = packages_list[0].metadata if packages_list else {}
    tools_interface = first_meta.get("tools_code", "")
    if packages_list:
        candidate = packages_list[0]
        tools_path = writer.task_dir(candidate.task.task_id, candidate.agent_type) / "_sandbox" / "tools.py"
        if tools_path.exists():
            tools_interface = tools_path.read_text(encoding="utf-8")

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
        key = (package.task.task_id, package.task.difficulty_level)
        if key in existing_task_keys:
            logger.debug(
                "Skipping duplicate task: %s (difficulty %s)",
                package.task.task_id,
                package.task.difficulty_level,
            )
            continue
        if not isinstance(package.verification, str) or "def verify" not in package.verification:
            logger.warning("Skipping task with invalid verification code: %s", package.task.task_id)
            continue
        meta = package.metadata or {}
        if any(meta.get(key) for key in ("validation_error", "verification_error", "repair_failed")):
            logger.warning("Skipping task with validation errors: %s", package.task.task_id)
            continue

        task_entry = {
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
            "verifier": {
                "verification_code": package.verification or "",
                "reward_function": package.verification or "",
            },
            "metadata": package.metadata,
        }
        tasks_with_verifiers.append(task_entry)
        existing_task_keys.add(key)

    # Merge records if merging (use existing records from db.json if available)
    final_records = records
    if not final_records and writer.path.exists():
        try:
            db_data = json.loads(writer.path.read_text())
            final_records = db_data.get("records", [])
        except Exception:
            final_records = records
    if merge and writer.path.exists():
        try:
            db_data = json.loads(writer.path.read_text())
            existing_db_records = db_data.get("records", [])
            if existing_db_records:
                record_titles = {r.get("title", "").lower() for r in existing_db_records}
                for record in records:
                    if record.get("title", "").lower() not in record_titles:
                        existing_db_records.append(record)
                final_records = existing_db_records
        except Exception:
            pass

    payload = {
        "environment": {
            "category": category,
            "records": final_records,
            "record_count": len(final_records),
        },
        "tools": all_tools,
        "tools_interface": tools_interface,
        "tasks": tasks_with_verifiers,
        "metadata": {
            "version": "1.1",
            "format": "quadruple",
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
