from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from agent_gem.sandbox import SandboxExecutor, SandboxFusionExecutor


def _load_submitted_result(sandbox_dir: Path) -> Any | None:
    submitted = sandbox_dir / "submitted_result.json"
    if not submitted.exists():
        return None
    try:
        return json.loads(submitted.read_text(encoding="utf-8"))
    except Exception:
        return None


def _should_use_submitted_result(answer: Any) -> bool:
    if answer is None:
        return True
    if isinstance(answer, str):
        lowered = answer.lower()
        return any(
            token in lowered
            for token in (
                "submitted_result.json",
                "result submitted",
                "result saved",
                "submitted successfully",
                "saved to submitted_result",
            )
        )
    return False


def _unwrap_submitted_result(answer: Any) -> Any:
    if not isinstance(answer, dict):
        return answer
    status = answer.get("status")
    message = answer.get("message")
    if not (isinstance(status, str) or isinstance(message, str)):
        return answer
    if "submitted_data" in answer:
        return answer.get("submitted_data")
    if "data" in answer:
        return answer.get("data")
    return answer


def _persist_verified_result(sandbox: SandboxExecutor, run_group: str) -> None:
    source = sandbox.sandbox_dir / "submitted_result.json"
    if not source.exists():
        return
    group_dir = sandbox._run_group_dir(run_group)
    try:
        group_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, group_dir / "submitted_result.json")
    except Exception:
        pass


class GeneralAgentSandboxExecutor(SandboxExecutor):
    def _process_solution_answer(self, answer: Any) -> Any:
        if _should_use_submitted_result(answer):
            submitted_payload = _load_submitted_result(self.sandbox_dir)
            if submitted_payload is not None:
                answer = submitted_payload
        return _unwrap_submitted_result(answer)

    def persist_verified_result(self, run_group: str) -> None:
        _persist_verified_result(self, run_group)


class GeneralAgentSandboxFusionExecutor(SandboxFusionExecutor):
    def _extra_runner_helpers(self) -> str:
        return """
def _load_submitted_result():
    submitted_path = Path("submitted_result.json")
    if not submitted_path.exists():
        return None
    try:
        return json.loads(submitted_path.read_text(encoding="utf-8"))
    except Exception:
        return None

def _should_use_submitted_result(value):
    if value is None:
        return True
    if isinstance(value, str):
        lowered = value.lower()
        return any(
            token in lowered
            for token in (
                "submitted_result.json",
                "result submitted",
                "result saved",
                "submitted successfully",
                "saved to submitted_result",
            )
        )
    return False

def _unwrap_answer(value):
    if not isinstance(value, dict):
        return value
    status = value.get("status")
    message = value.get("message")
    if not (isinstance(status, str) or isinstance(message, str)):
        return value
    if "submitted_data" in value:
        return value.get("submitted_data")
    if "data" in value:
        return value.get("data")
    return value
"""

    def _answer_postprocess_snippet(self) -> str:
        return """
if _should_use_submitted_result(answer):
    submitted_payload = _load_submitted_result()
    if submitted_payload is not None:
        answer = submitted_payload
answer = _unwrap_answer(answer)
"""

    def persist_verified_result(self, run_group: str) -> None:
        _persist_verified_result(self, run_group)
