from __future__ import annotations

from typing import Iterable, Set

from .task_schema import TaskPackage


def validate_task_package(package: TaskPackage) -> TaskPackage:
    """Lightweight validation guardrails for generated tasks."""
    task = package.task
    if not task.tool_set:
        raise ValueError(f"Task '{task.task_title}' must declare at least one tool.")

    if len(task.task_content.split()) < 8:
        raise ValueError(f"Task '{task.task_title}' lacks sufficient detail.")

    if not _looks_runnable(package.solution):
        raise ValueError(f"Solution for '{task.task_title}' is not runnable.")

    if "verify" not in package.verification:
        raise ValueError(f"Verification for '{task.task_title}' must define a check.")

    return package


def _looks_runnable(code: str) -> bool:
    banned: Set[str] = {"rm -rf", "shutdown", ":(){:|:&};:"}
    lowered = code.lower()
    return not any(token in lowered for token in banned)
