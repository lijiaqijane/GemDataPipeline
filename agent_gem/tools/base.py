from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from agent_gem.core.task_schema import ToolSpec


class ToolExecutionError(RuntimeError):
    """Raised by a tool to return a structured error result."""

    def __init__(self, kind: str, tool_output: Any, message: str | None = None) -> None:
        super().__init__(message or kind)
        self.kind = kind
        self.tool_output = tool_output


class BaseTool(ABC):
    """Base class for all executable tools registered in a sandbox."""

    name: str
    description: str

    def __init__(self, *, name: str, description: str | None = None) -> None:
        self.name = name
        self.description = (
            description.strip()
            if isinstance(description, str) and description.strip()
            else (self.__class__.__doc__ or "").strip()
        )

    @abstractmethod
    def execute(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.execute(*args, **kwargs)

    def to_tool_spec(self) -> ToolSpec:
        """Convert this tool into a schema-level ToolSpec for prompting/packaging."""
        return ToolSpec.from_function(
            fn=self.execute,
            name=self.name,
            description=self.description or self.name,
        )


class CallableTool(BaseTool):
    """Wrap an arbitrary callable as a sandbox tool."""

    def __init__(
        self,
        *,
        name: str,
        handler: Callable[..., Any],
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description or (handler.__doc__ or ""))
        self._handler = handler

    def execute(self, *args: Any, **kwargs: Any) -> Any:
        return self._handler(*args, **kwargs)

    def to_tool_spec(self) -> ToolSpec:
        return ToolSpec.from_function(
            fn=self._handler,
            name=self.name,
            description=self.description or self.name,
        )
