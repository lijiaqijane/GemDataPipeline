from __future__ import annotations

import contextlib
import logging
from pathlib import Path
from typing import Generator, Iterable, List, Optional

from agent_gem.core.task_schema import TaskPackage
from agent_gem.core.utils import dump_json, slugify
from agent_gem.sandbox.executor import DockerAPIRunner, SandboxFusionExecutor

logger = logging.getLogger(__name__)


class SandboxManager:
    """
    Unified sandbox manager for both task persistence and SandboxFusion container lifecycle.
    
    Features:
    1. Task persistence: Save generated tasks to filesystem
    2. Container management: Start/stop SandboxFusion containers
    """

    def __init__(
        self, 
        root: Optional[Path] = None,
        *,
        sandbox_url: Optional[str] = None,
        use_docker_runner: bool = True,
        silent: bool = False,
        use_china_mirror: bool = True
    ) -> None:
        # Task persistence
        self.root = root or Path("taskdb")
        self.root.mkdir(parents=True, exist_ok=True)
        
        # SandboxFusion container management
        self.sandbox_url = sandbox_url or "http://localhost:8080"
        self.use_docker_runner = use_docker_runner
        self.silent = silent
        self.use_china_mirror = use_china_mirror
        self._runner: Optional[DockerAPIRunner] = None

    def persist(self, packages: Iterable[TaskPackage]) -> List[TaskPackage]:
        """Persist generated task packages to filesystem."""
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

    @contextlib.contextmanager
    def managed_executor(self, *, timeout_s: int = 20) -> Generator[SandboxFusionExecutor, None, None]:
        """
        Context manager that yields a running SandboxFusionExecutor.
        
        Usage:
            with sandbox_manager.managed_executor() as executor:
                executor.run_code("print('hello')", language="python")
        """
        executor: Optional[SandboxFusionExecutor] = None
        try:
            base_url = self.sandbox_url
            if self.use_docker_runner:
                self._runner = DockerAPIRunner(
                    use_china_mirror=self.use_china_mirror,
                    silent=self.silent,
                )
                if not self._runner.start():
                    raise RuntimeError("Failed to start SandboxFusion container")
                self._runner.wait_ready()
                base_url = f"http://localhost:{self._runner.port}"

            executor = SandboxFusionExecutor(self.root, base_url=base_url, timeout_s=timeout_s)
            yield executor
        finally:
            if self._runner is not None:
                try:
                    self._runner.stop()
                except Exception:
                    logger.exception("Failed to stop SandboxFusion container cleanly")
                self._runner = None


__all__ = ["SandboxManager"]
