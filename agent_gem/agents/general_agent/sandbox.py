from __future__ import annotations

import json
import logging
import shutil
import textwrap
import time
import traceback
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from agent_gem.core.task_schema import TaskPackage
from agent_gem.core.utils import dump_json
from agent_gem.sandbox import SandboxExecutor, SandboxFusionExecutor
from agent_gem.sandbox.executor import TaskRunRecord

logger = logging.getLogger(__name__)


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
    if isinstance(answer, dict):
        # Check if this is a submit_result tool response
        status = answer.get("status")
        message = answer.get("message", "").lower()
        file_path = answer.get("file_path", "").lower()
        return (
            status == "success" and
            any(token in message for token in ("submitted", "saved")) and
            "submitted_result.json" in file_path
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

    def run_task(
        self,
        package: TaskPackage,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        run_group: str = "default",
    ) -> TaskRunRecord:
        run_id = uuid.uuid4().hex
        started = time.time()
        tool_proxy = self.as_tools(extra=tools, cache_calls=True)
        group_dir = self._run_group_dir(run_group)
        self._record_run_code(run_id, package, group_dir=group_dir)

        raw_answer: Any = None
        processed_answer: Any = None
        verified: Optional[bool] = None
        verification_score: Optional[float] = None
        verification_details: Any = None
        verification_message: Optional[str] = None
        error: Optional[str] = None
        verification_error: Optional[str] = None
        try:
            raw_answer = package.run_solution(tool_proxy)
            processed_answer = self._process_solution_answer(raw_answer)
            verified, verification_score, verification_details, verification_message = package.verify_with_meta(
                tool_proxy, processed_answer
            )
        except Exception:
            error = traceback.format_exc()
        else:
            if verified is False:
                verification_error = verification_message or "verification returned False"
            elif verified is None:
                verification_error = verification_message or "verification returned no boolean result"

        ended = time.time()
        record = TaskRunRecord(
            run_id=run_id,
            task_id=package.task.task_id,
            task_title=package.task.task_title,
            run_group=run_group,
            started_at=started,
            ended_at=ended,
            duration_s=ended - started,
            answer=self._to_jsonable(raw_answer),
            verified=verified,
            verification_score=verification_score,
            verification_details=self._to_jsonable(verification_details),
            error=error,
            verification_error=verification_error,
        )
        group_dir.mkdir(parents=True, exist_ok=True)
        dump_json(group_dir / f"{record.run_id}.json", record.model_dump())
        logger.debug("Saved task run record: %s.json in %s", record.run_id, group_dir)
        return record

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

    def _build_runner_code(self, package: TaskPackage) -> str:
        solution_code = package.solution or ""
        verification_code = package.verification or ""
        required_tools = [spec.name for spec in package.task.tool_set]

        extra_helpers = textwrap.dedent(self._extra_runner_helpers() or "").strip("\n")
        if extra_helpers:
            extra_helpers = extra_helpers + "\n"

        postprocess = textwrap.dedent(self._answer_postprocess_snippet() or "").strip("\n")
        if postprocess:
            postprocess = textwrap.indent(postprocess, "    ") + "\n"

        runner = textwrap.dedent(
            f'''
import json
import importlib.util
import traceback
from pathlib import Path
class ToolProxy(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        def _missing(*args, **kwargs):
            return {{"error": f"Tool not available: {{name}}", "args": args, "kwargs": kwargs}}
        return _missing

tools = {{}}
try:
    spec = importlib.util.spec_from_file_location("generated_tools", "tools.py")
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for key in dir(module):
            if key.startswith("_"):
                continue
            value = getattr(module, key)
            if callable(value):
                tools[key] = value
except Exception:
    pass

{extra_helpers}# User requirement: all required tools MUST be implemented by the agent in tools.py.
missing = [name for name in {json.dumps(required_tools)} if name not in tools or not callable(tools.get(name))]
if missing:
    print(json.dumps({{"error": f"missing_required_tools: {{missing}}"}}, ensure_ascii=False))
    raise SystemExit(0)

tool_cache = {{}}
def _cache_key(args, kwargs):
    payload = {{"args": args, "kwargs": kwargs}}
    try:
        return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        return repr(payload)

def _wrap_tool(name, fn):
    def _call(*args, **kwargs):
        cache_key = name + ":" + _cache_key(args, kwargs)
        if cache_key in tool_cache:
            return tool_cache[cache_key]
        result = fn(*args, **kwargs)
        tool_cache[cache_key] = result
        return result
    return _call

for name in list(tools.keys()):
    if callable(tools[name]):
        tools[name] = _wrap_tool(name, tools[name])

tool_proxy = ToolProxy(**tools)

def _emit(payload):
    print(json.dumps(payload, ensure_ascii=False))

try:
    solution_src = {json.dumps(solution_code)}
    verification_src = {json.dumps(verification_code)}
    def _coerce_score(value):
        try:
            return float(value)
        except Exception:
            return None

    def _normalize_verification_output(output):
        verified = None
        score = None
        details = None
        message = None
        if isinstance(output, dict):
            for key in ("passed", "success", "ok", "result"):
                if key in output:
                    verified = bool(output.get(key))
                    break
            score = _coerce_score(output.get("score"))
            details = output.get("details") or output
            message = output.get("message") or output.get("error")
        elif isinstance(output, (list, tuple)) and output:
            if isinstance(output[0], bool):
                verified = output[0]
            if len(output) > 1:
                score = _coerce_score(output[1])
                if score is None and isinstance(output[1], str):
                    message = output[1]
            if len(output) > 2:
                details = output[2]
            if len(output) > 3 and message is None and isinstance(output[3], str):
                message = output[3]
        elif isinstance(output, bool):
            verified = output
        else:
            details = output
        return verified, score, details, message

    exec(solution_src, globals())
    exec(verification_src, globals())
    answer = solve(tool_proxy)
{postprocess}    raw_verified = verify(tool_proxy, answer)
    verified, score, details, message = _normalize_verification_output(raw_verified)
    if verified is False and message is None:
        message = "verification returned False"
    if verified is None and message is None:
        message = f"verification returned unsupported type: {{type(raw_verified).__name__}}"
    _emit({{"answer": answer, "verified": verified, "verification_score": score, "verification_details": details, "verification_message": message}})
except Exception:
    _emit({{"error": traceback.format_exc()}})
'''
        ).strip()
        return runner

    def run_task(
        self,
        package: TaskPackage,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
        run_group: str = "default",
    ) -> TaskRunRecord:
        run_id = uuid.uuid4().hex
        started = time.time()
        group_dir = self._run_group_dir(run_group)
        self._record_run_code(run_id, package, group_dir=group_dir)
        runner = self._build_runner_code(package)
        exec_result = self.execute_python(runner, timeout_s=self.timeout_s)
        answer: Any = None
        verified: Optional[bool] = None
        verification_score: Optional[float] = None
        verification_details: Any = None
        verification_message: Optional[str] = None
        error: Optional[str] = None
        verification_error: Optional[str] = None
        try:
            stdout = (exec_result.get("stdout") or "").strip()
            stderr = (exec_result.get("stderr") or "").strip()
            lines = stdout.splitlines() if stdout else []
            payload = json.loads(lines[-1]) if lines else {}
            if "error" in payload:
                error = payload.get("error")
            else:
                answer = payload.get("answer")
                verified = payload.get("verified")
                verification_score = payload.get("verification_score")
                verification_details = payload.get("verification_details")
                verification_message = payload.get("verification_message")
            if not lines and (stderr or exec_result.get("return_code", 0) not in (0, None)):
                error = stderr or "sandbox_no_output"
        except Exception:
            error = "failed_to_parse_sandbox_output"

        ended = time.time()
        if verified is False:
            verification_error = verification_message or "verification returned False"
        elif verified is None:
            verification_error = verification_message or "verification returned no boolean result"

        record = TaskRunRecord(
            run_id=run_id,
            task_id=package.task.task_id,
            task_title=package.task.task_title,
            run_group=run_group,
            started_at=started,
            ended_at=ended,
            duration_s=ended - started,
            answer=self._to_jsonable(answer),
            verified=verified,
            verification_score=verification_score,
            verification_details=self._to_jsonable(verification_details),
            error=error,
            verification_error=verification_error,
        )
        group_dir.mkdir(parents=True, exist_ok=True)
        dump_json(group_dir / f"{record.run_id}.json", record.model_dump())
        logger.debug("Saved task run record: %s.json in %s", record.run_id, group_dir)
        return record

    def persist_verified_result(self, run_group: str) -> None:
        _persist_verified_result(self, run_group)
