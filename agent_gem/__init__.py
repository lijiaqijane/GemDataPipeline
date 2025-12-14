from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "BaseAgent",
    "CodeAgent",
    "CodeInterpreterAgent",
    "GeneralAgent",
    "SearchAgent",
    "LLMConfig",
    "EnvironmentGenerator",
    "GenerationRequest",
    "LLMClient",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "BaseAgent": ("agent_gem.agents", "BaseAgent"),
    "CodeAgent": ("agent_gem.agents", "CodeAgent"),
    "CodeInterpreterAgent": ("agent_gem.agents", "CodeInterpreterAgent"),
    "GeneralAgent": ("agent_gem.agents", "GeneralAgent"),
    "SearchAgent": ("agent_gem.agents", "SearchAgent"),
    "LLMConfig": ("agent_gem.config", "LLMConfig"),
    "EnvironmentGenerator": ("agent_gem.env", "EnvironmentGenerator"),
    "GenerationRequest": ("agent_gem.env", "GenerationRequest"),
    "LLMClient": ("agent_gem.llm", "LLMClient"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value
