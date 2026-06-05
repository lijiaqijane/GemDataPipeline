"""
Task adapter for code_agent module.

This module provides Task interface adaptation, allowing code_agent to work
with tasks from app module or provide its own implementation.
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

try:
    from app.task import Task as AppTask, SweTask as AppSweTask
    from docker import DockerClient
    APP_TASK_AVAILABLE = True
except ImportError:
    APP_TASK_AVAILABLE = False
    AppTask = ABC
    AppSweTask = None
    DockerClient = Any


class Task(ABC):
    """Abstract base class for tasks."""
    
    @property
    @abstractmethod
    def project_path(self) -> str:
        """Get the project path."""
        raise NotImplementedError("abstract method")

    @abstractmethod
    def get_issue_statement(self) -> str:
        """Get the issue statement."""
        raise NotImplementedError("abstract method")

    @abstractmethod
    def setup_project(self) -> None:
        """Set up the project before starting to resolve the task."""
        raise NotImplementedError("abstract method")

    @abstractmethod
    def reset_project(self) -> None:
        """Reset project to initial state."""
        raise NotImplementedError("abstract method")


@dataclass(kw_only=True)
class SweTask(Task):
    """SWE-bench task implementation."""
    
    task_id: str
    problem_statement: str
    repo_path: str
    repo_cache_path: str
    commit: str
    repo_name: str
    patch: str
    test_patch: str
    language: str
    version: str | None = None
    client: DockerClient
    task_info: dict
    
    @property
    def project_path(self) -> str:
        return self.repo_path
    
    @project_path.setter
    def project_path(self, value: str) -> None:
        self.repo_path = value

    def get_issue_statement(self) -> str:
        return self.problem_statement
    
    def setup_project(self) -> None:
        """Set up the project by checking out the correct commit."""
        # Import here to avoid circular dependency
        from .utils_adapter import cd, repo_reset_and_clean_checkout, repo_commit_current_changes
        
        with cd(self.project_path):
            repo_reset_and_clean_checkout(self.commit)
        
        # Commit the current changes so that resetting later does not erase them
        with cd(self.project_path):
            repo_commit_current_changes()

    def reset_project(self) -> None:
        """Reset project to initial commit state."""
        from .utils_adapter import cd, repo_reset_and_clean_checkout
        
        with cd(self.repo_path):
            repo_reset_and_clean_checkout(self.commit)

    def remove_project(self) -> None:
        """Remove the entire project repository."""
        if os.path.exists(self.repo_path):
            shutil.rmtree(self.repo_path)


class TaskAdapter:
    """Adapter for converting between app.task and code_agent.task."""
    
    @staticmethod
    def from_app_task(app_task: AppTask) -> Task:
        """Convert an app.task.Task to code_agent Task."""
        if not APP_TASK_AVAILABLE:
            raise ValueError("app.task module is not available")
        
        if isinstance(app_task, AppSweTask):
            return SweTask(
                task_id=app_task.task_id,
                problem_statement=app_task.problem_statement,
                repo_path=app_task.repo_path,
                repo_cache_path=app_task.repo_cache_path,
                commit=app_task.commit,
                repo_name=app_task.repo_name,
                patch=app_task.patch,
                test_patch=app_task.test_patch,
                language=app_task.language,
                version=getattr(app_task, 'version', None),
                client=app_task.client,
                task_info=app_task.task_info,
            )
        else:
            # For other task types, create a wrapper
            # This is a simplified implementation
            raise NotImplementedError(f"Task type {type(app_task)} not supported")
    
    @staticmethod
    def to_app_task(task: Task) -> AppTask:
        """Convert a code_agent Task to app.task.Task."""
        if not APP_TASK_AVAILABLE:
            raise ValueError("app.task module is not available")
        
        if isinstance(task, SweTask):
            return AppSweTask(
                task_id=task.task_id,
                problem_statement=task.problem_statement,
                repo_path=task.repo_path,
                repo_cache_path=task.repo_cache_path,
                commit=task.commit,
                repo_name=task.repo_name,
                patch=task.patch,
                test_patch=task.test_patch,
                language=task.language,
                version=task.version if task.version else 'unknown',
                client=task.client,
                task_info=task.task_info,
            )
        else:
            raise NotImplementedError(f"Task type {type(task)} not supported")
