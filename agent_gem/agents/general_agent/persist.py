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
) -> Path:
    """Persist tasks in quadruple format compatible with general_agent output.
    
    Each task gets its own tasks.json file in its task directory.
    """
    packages_list = list(packages)
    if not packages_list:
        logger.warning("No packages to persist in quadruple format")
        # If no packages, use first package's task directory if output_path not provided
        if output_path:
            return output_path
        # Fallback to writer.root if we can't determine task directory
        return writer.root / "tasks.json"

    # Use the first package to determine task directory
    first_package = packages_list[0]
    task_dir = writer.task_dir(first_package.task.task_id, first_package.agent_type)
    target = output_path or (task_dir / "tasks.json")

    # Get tools_interface from the task's sandbox
    first_meta = first_package.metadata if packages_list else {}
    tools_interface = first_meta.get("tools_code", "")
    if packages_list:
        tools_path = task_dir / "_sandbox" / "tools.py"
        if tools_path.exists():
            tools_interface = tools_path.read_text(encoding="utf-8")

    # Collect all tools from packages (no merging with existing)
    all_tools = []
    tool_names_seen = set()
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

    # Build tasks with task and verifier structure (no merging)
    tasks_with_verifiers = []
    for package in packages_list:
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
                "state_hash": {
                    "pre": package.metadata.get("pre_state_hash", ""),
                    "post": package.metadata.get("post_state_hash", ""),
                },
            },
            "verifier": {
                "verification_code": package.verification or "",
            },
            "metadata": package.metadata,
        }
        tasks_with_verifiers.append(task_entry)

    payload = {
        "environment": {
            "category": category,
            "records": records,
            "record_count": len(records),
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
        },
    }

    target.parent.mkdir(parents=True, exist_ok=True)
    dump_json(target, payload)
    logger.info(
        "Quadruple format tasks saved to %s (tasks: %d, tools: %d, records: %d)",
        target,
        len(tasks_with_verifiers),
        len(all_tools),
        len(records),
    )
    return target
