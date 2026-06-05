from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from .task_schema import TaskDefinition


@dataclass
class TaskScore:
    composite: float
    detail: Dict[str, float]


def score_task(task: TaskDefinition) -> TaskScore:
    """Compute a lightweight score for prioritization across generated tasks."""
    weights = {
        "correctness": 0.4,
        "complexity": 0.25,
        "diversity": 0.2,
        "solution_verifiability": 0.15,
    }
    crit = task.evaluation_criteria
    detail = {
        "correctness": crit.correctness,
        "complexity": crit.complexity,
        "diversity": crit.diversity,
        "solution_verifiability": crit.solution_verifiability,
    }
    composite = sum(detail[key] * weight for key, weight in weights.items())
    return TaskScore(composite=composite, detail=detail)
