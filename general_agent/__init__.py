"""
general_agent

Lightweight automatic environment and task synthesis agent that supports local vLLM or OpenAI-compatible endpoints.
"""

from .config import LLMConfig
from .database import LocalDatabase
from .executor import SandboxFusionExecutor
from .llm import LLMClient
from .synthesis import EnvironmentSynthesizer, SynthesisContext, TaskBundle
from .tools import BashTool, SearchTool, ToolRegistry
from .executor import SandboxFusionExecutor
from .constraints import (
    CodeValidator,
    SolutionContext,
    ToolContext,
    VerificationContext,
)
from .synthesis import (
    EnvironmentSynthesizer,
    SynthesisContext,
    TaskBundle,
    EnvironmentAgent,
    ToolAgent,
    TaskAgent,
    ValidationAgent,
)

__all__ = [
    "LLMConfig",
    "LLMClient",
    "LocalDatabase",
    "BashTool",
    "SearchTool",
    "ToolRegistry",
    "SandboxFusionExecutor",
    "CodeValidator",
    "SolutionContext",
    "ToolContext",
    "VerificationContext",
    "EnvironmentSynthesizer",
    "SynthesisContext",
    "TaskBundle",
]
