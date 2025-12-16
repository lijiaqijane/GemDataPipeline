"""Synthesis context definitions."""

from dataclasses import dataclass
from pathlib import Path

from ..database import LocalDatabase
from ..tools import ToolRegistry
from ..llm import LLMClient


@dataclass
class SynthesisContext:
    """Context for environment and task synthesis."""
    
    category: str
    sandbox: Path
    db: LocalDatabase
    registry: ToolRegistry
    llm: LLMClient

