"""
general_agent

Lightweight automatic environment and task synthesis agent that supports local vLLM or OpenAI-compatible endpoints.
"""

from .config import LLMConfig
from .llm import LLMClient
from .database import LocalDatabase
from .tools import BashTool, SearchTool, ToolRegistry
from .synthesis import (
    EnvironmentSynthesizer,
    SynthesisContext,
    TaskBundle,
)

__all__ = [
    "LLMConfig",
    "LLMClient",
    "LocalDatabase",
    "BashTool",
    "SearchTool",
    "ToolRegistry",
    "EnvironmentSynthesizer",
    "SynthesisContext",
    "TaskBundle",
]

