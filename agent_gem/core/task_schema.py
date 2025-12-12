from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ToolSpec(BaseModel):
    tool_name: str = Field(..., min_length=1)
    tool_description: str = Field(..., min_length=1)
    tool_functionality: str = Field(..., min_length=1)


class EvaluationCriteria(BaseModel):
    correctness: float = Field(0.6, ge=0.0, le=1.0)
    diversity: float = Field(0.6, ge=0.0, le=1.0)
    complexity: float = Field(0.6, ge=0.0, le=1.0)
    solution_verifiability: float = Field(0.6, ge=0.0, le=1.0)

    @field_validator("*", mode="before")
    @classmethod
    def default_to_float(cls, value: float) -> float:
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return value


class TaskDefinition(BaseModel):
    task_title: str = Field(..., min_length=4)
    task_content: str = Field(..., min_length=10)
    submit_result_format: str = Field("json", min_length=3)
    tool_set: List[ToolSpec] = Field(default_factory=list)
    evaluation_criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)
    difficulty_level: str = Field("Medium", min_length=4)

    def summary(self) -> str:
        return f"{self.task_title} [{self.difficulty_level}]"


class TaskPackage(BaseModel):
    task: TaskDefinition
    solution: str = Field(..., min_length=3)
    verification: str = Field(..., min_length=3)
    agent_type: str = Field(..., min_length=3)
    metadata: Dict[str, str] = Field(default_factory=dict)
    sandbox_path: Optional[str] = None

    def as_payload(self) -> Dict[str, object]:
        return {
            "agent_type": self.agent_type,
            "task": self.task.model_dump(),
            "solution": self.solution,
            "verification": self.verification,
            "metadata": self.metadata,
            "sandbox_path": self.sandbox_path,
        }
