"""
Raw task classes for code_agent module.

This module provides RawSweTask and related classes without dependencies on app module.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from os.path import join as pjoin
from typing import Any, Optional

try:
    from docker import DockerClient
except ImportError:
    DockerClient = Any


class RawTask(ABC):
    """Abstract base class for raw tasks."""
    
    @property
    @abstractmethod
    def task_id(self) -> str:
        """Get the task ID."""
        raise NotImplementedError("abstract base class")

    @abstractmethod
    def to_task(self):
        """Convert to a Task object."""
        raise NotImplementedError("abstract base class")

    @abstractmethod
    def dump_meta_data(self, output_dir: str) -> None:
        """Dump metadata to output directory."""
        raise NotImplementedError("abstract base class")


class RawSweTask(RawTask):
    """
    Encapsulate everything required to run one SWE-bench task.
    
    This class is independent of the app module and can be used in code_agent
    without requiring app.raw_tasks to be importable.
    """

    def __init__(
        self, 
        task_id: str, 
        setup_info: dict, 
        task_info: dict, 
        client: Optional[DockerClient] = None
    ):
        """
        Initialize RawSweTask.
        
        Args:
            task_id: Task identifier
            setup_info: Setup information dict with keys like:
                - 'repo_path': Path to task repository
                - 'repo_cache_path': Path to repository cache
                - 'env_name': Environment name (optional)
                - 'pre_install': Pre-install commands (optional)
                - 'install': Install command (optional)
                - 'test_cmd': Test command (optional)
            task_info: Task information dict with keys like:
                - 'base_commit': Base commit hash
                - 'hints_text': Hints text (optional)
                - 'created_at': Creation timestamp (optional)
                - 'test_patch': Test patch
                - 'repo': Repository name
                - 'problem_statement': Problem statement
                - 'version': Version
                - 'instance_id': Instance ID
                - 'FAIL_TO_PASS': Failing tests (optional)
                - 'PASS_TO_PASS': Passing tests (optional)
                - 'environment_setup_commit': Environment setup commit (optional)
                - 'patch': Developer patch
            client: Optional Docker client
        """
        self._task_id = task_id
        self.setup_info = setup_info
        self.task_info = task_info
        self.client = client

    @property
    def task_id(self) -> str:
        """Get the task ID."""
        return self._task_id

    def to_task(self):
        """
        Convert to a SweTask object from code_agent.dependencies.
        
        Returns:
            SweTask instance
        """
        from .environment_setup_utils.task_adapter import SweTask
        
        task_id = self.task_id
        setup_info = self.setup_info
        task_info = self.task_info
        language = task_info.get('language', 'None')
        client = self.client
        
        return SweTask(
            task_id=task_id,
            problem_statement=task_info["problem_statement"],
            repo_path=setup_info["repo_path"],
            repo_cache_path=setup_info["repo_cache_path"],
            commit=task_info["base_commit"],
            repo_name=task_info["repo"],
            patch=task_info["patch"],
            test_patch=task_info["test_patch"],
            language=language,
            version=task_info.get('version'),
            client=client,
            task_info=task_info
        )

    def dump_meta_data(self, output_dir: str) -> None:
        """
        Dump task metadata to output directory.
        
        Args:
            output_dir: Output directory path
        """
        meta = {
            "task_id": self.task_id,
            "setup_info": self.setup_info,
            "task_info": self.task_info,
        }
        with open(pjoin(output_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=4)
        with open(pjoin(output_dir, "problem_statement.txt"), "w") as f:
            f.write(self.task_info["problem_statement"])
        with open(pjoin(output_dir, "developer_patch.diff"), "w") as f:
            f.write(self.task_info["patch"])
