"""Code agent package."""

# Keep imports lazy and tolerant of missing optional dependencies so that
# lightweight utilities (e.g., repo_crawler) can be imported without
# requiring the full code_agent stack to be available.

try:  # pragma: no cover - optional imports
    from .agent import CodeAgent
except Exception:  # noqa: BLE001
    CodeAgent = None  # type: ignore[assignment]

try:  # pragma: no cover - optional imports
    from .environment_setup_agent import EnvironmentSetupAgent
except Exception:  # noqa: BLE001
    EnvironmentSetupAgent = None  # type: ignore[assignment]

try:  # pragma: no cover - optional imports
    from .task_executor import TaskExecutor
except Exception:  # noqa: BLE001
    TaskExecutor = None  # type: ignore[assignment]

try:  # pragma: no cover - optional imports
    from .raw_tasks import RawSweTask, RawTask
except Exception:  # noqa: BLE001
    RawSweTask = None  # type: ignore[assignment]
    RawTask = None  # type: ignore[assignment]

__all__ = [
    "CodeAgent",
    "EnvironmentSetupAgent",
    "TaskExecutor",
    "RawSweTask",
    "RawTask",
]