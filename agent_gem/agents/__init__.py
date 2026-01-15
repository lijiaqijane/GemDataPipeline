from .base import BaseAgent

# The other agent types depend on optional or heavy submodules.
# Import them lazily and tolerate import errors so that lightweight
# utilities (e.g., repo_crawler) can be used without pulling in
# the entire code_agent stack.
try:  # pragma: no cover - optional imports
    from .code_agent import CodeAgent
except Exception:  # noqa: BLE001
    CodeAgent = None  # type: ignore[assignment]

try:  # pragma: no cover - optional imports
    from .code_interpreter_agent import CodeInterpreterAgent
except Exception:  # noqa: BLE001
    CodeInterpreterAgent = None  # type: ignore[assignment]

try:  # pragma: no cover - optional imports
    from .general_agent import GeneralAgent
except Exception:  # noqa: BLE001
    GeneralAgent = None  # type: ignore[assignment]

# from .search_agent import SearchAgent

__all__ = [
    "BaseAgent",
    "CodeAgent",
    "CodeInterpreterAgent",
    "GeneralAgent",
    "SearchAgent",
]
