from __future__ import annotations

import asyncio
import base64
import hashlib
import inspect
import json
import logging
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from io import BytesIO
import os
from pathlib import Path
import requests
import shutil
import tarfile
import textwrap
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field
from sandbox_fusion import set_endpoint, run_code, RunCodeRequest
import socket

import docker
from docker.errors import DockerException
from agent_gem.core.task_schema import TaskPackage
from agent_gem.core.utils import dump_json, slugify
from agent_gem.tools import (
    BaseTool,
    BashTool,
    CallableTool,
    PythonRunnerTool,
    SearchTool,
    ToolExecutionError,
)


_SANDBOX_FUSION_IMAGES = {
    "global": "volcengine/sandbox-fusion:server-20250609",
    "china": "vemlp-cn-beijing.cr.volces.com/preset-images/code-sandbox:server-20250609",
}

logger = logging.getLogger(__name__)


class DockerAPIRunner:
    def __init__(self, use_china_mirror: bool = True, silent: bool = False) -> None:
        self.image = (
            _SANDBOX_FUSION_IMAGES["china"]
            if use_china_mirror
            else _SANDBOX_FUSION_IMAGES["global"]
        )
        self.container = None
        self.silent = silent
        self.client = docker.from_env()
        self.port = self._find_free_port()

    @staticmethod
    def _find_free_port() -> int:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("", 0))
            sock.listen(1)
            return int(sock.getsockname()[1])

    def start(self) -> bool:
        try:
            if not self.silent:
                logger.info("Pulling image: %s", self.image)
            self.client.images.pull(self.image)
            self.container = self.client.containers.run(
                self.image,
                ports={"8080/tcp": self.port},
                detach=True,
                remove=True,
            )
            if not self.silent:
                logger.info(
                    "SandboxFusion container started: %s", self.container.short_id
                )
            return True
        except DockerException as exc:
            if not self.silent:
                logger.error("Error starting SandboxFusion container: %s", exc)
            return False

    def stop(self) -> bool:
        if not self.container:
            return False
        try:
            self.container.stop()
            if not self.silent:
                logger.info("SandboxFusion container stopped")
            return True
        except DockerException as exc:
            if not self.silent:
                logger.error("Error stopping SandboxFusion container: %s", exc)
            return False

    def wait_ready(self, max_wait_time: int = 60, check_interval: float = 1.0) -> None:
        if not self.container:
            raise RuntimeError("Container not started")
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            self.container.reload()
            if self.container.status == "running":
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                        sock.settimeout(2)
                        if sock.connect_ex(("localhost", self.port)) == 0:
                            return
                except OSError:
                    pass
            elif self.container.status in {"exited", "dead"}:
                logs = self.container.logs().decode("utf-8")
                raise RuntimeError(
                    f"Container failed to start ({self.container.status}): {logs[:500]}"
                )
            time.sleep(check_interval)
        logs = self.container.logs().decode("utf-8")
        raise RuntimeError(
            f"Container not ready after {max_wait_time}s: status={self.container.status}; logs={logs[:500]}"
        )


class ToolCallRecord(BaseModel):
    """Structured record of a single tool invocation."""

    call_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    tool: str
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    tool_output: Any = None
    error: Optional[str] = None
    started_at: float
    ended_at: float
    duration_s: float


class TaskRunRecord(BaseModel):
    """Structured record of running a TaskPackage (solve + verify)."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    task_id: Optional[str] = None
    task_title: Optional[str] = None
    started_at: float
    ended_at: float
    duration_s: float
    answer: Any = None
    verified: Optional[bool] = None
    verification_score: Optional[float] = None
    verification_details: Any = None
    error: Optional[str] = None
    verification_error: Optional[str] = None


class ToolProxy(dict):
    """Tool mapping that supports both dict and attribute access."""

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        return self.__missing__(name)

    def __missing__(self, key: str) -> Any:
        def _missing_tool(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return {
                "error": f"Tool not available: {key}",
                "args": args,
                "kwargs": kwargs,
            }

        return _missing_tool


class SandboxExecutor:
    """Local sandbox executor that runs tools and stores results under a directory.

    This is intentionally simple: commands execute locally with `cwd` pinned to
    `sandbox_dir`, and each tool call is appended to `logs/tool_calls.jsonl`.
    """

    def __init__(self, sandbox_dir: Path | str, *, timeout_s: int = 20) -> None:
        self.sandbox_dir = Path(sandbox_dir)
        self.sandbox_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s

        self.logs_dir = self.sandbox_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir = self.sandbox_dir / "runs"
        self.runs_dir.mkdir(parents=True, exist_ok=True)

        self.tool_calls_path = self.logs_dir / "tool_calls.jsonl"
        self.search_cache_path = self.sandbox_dir / "search_cache.json"
        self._tools: Dict[str, BaseTool] = {}
        self._tool_calls_lock = threading.Lock()
        self._on_tool_call: Callable[[ToolCallRecord], None] | None = None

    @classmethod
    def for_package(cls, package: TaskPackage, root: Path | str) -> "SandboxExecutor":
        root = Path(root)
        return cls(
            root / package.agent_type / f"task-{package.task.task_id}" / "_sandbox"
        )

    def as_tools(self, extra: Optional[Dict[str, Any]] = None) -> ToolProxy:
        if extra:
            for name, value in extra.items():
                self.register_tool(name, value)

        tools: Dict[str, Callable[..., Any]] = {}
        for name in sorted(self._tools):
            tools[name] = self._make_tool_callable(name)
        return ToolProxy(**tools)

    def describe_tools(self) -> List[Dict[str, str]]:
        """Return a lightweight description list suitable for prompting."""
        described: List[Dict[str, str]] = []
        for name in sorted(self._tools):
            described.append(
                {"name": name, "description": self._tools[name].description}
            )
        return described

    def tool_names(self) -> set[str]:
        return set(self._tools.keys())

    def tool_specs(self) -> List[dict[str, Any]]:
        """Return ToolSpec payloads (fn excluded) for all registered tools."""
        specs: List[dict[str, Any]] = []
        for name in sorted(self._tools):
            try:
                specs.append(self._tools[name].to_tool_spec().model_dump())
            except Exception:
                specs.append(
                    {
                        "name": name,
                        "description": self._tools[name].description,
                        "parameters": {},
                        "meta": {"error": "failed_to_build_tool_spec"},
                    }
                )
        return specs

    def execute_bash(
        self, command: str, timeout_s: Optional[int] = None
    ) -> Dict[str, Any]:
        result = self.execute("bash", command, timeout_s=timeout_s)
        return result  # type: ignore[return-value]

    def execute_python(
        self, code: str, timeout_s: Optional[int] = None
    ) -> Dict[str, Any]:
        result = self.execute("python_runner", code, timeout_s=timeout_s)
        return result  # type: ignore[return-value]

    def execute_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        result = self.execute("search", query, max_results=max_results)
        return result  # type: ignore[return-value]

    def register_tool(
        self,
        name_or_tool: str | BaseTool,
        handler: Callable[..., Any] | BaseTool | None = None,
        *,
        description: str = "",
    ) -> None:
        if isinstance(name_or_tool, BaseTool):
            tool = name_or_tool
            if not tool.name:
                raise ValueError("Tool name must be a non-empty string")
            if description:
                tool.description = description
            self._tools[tool.name] = tool
            return

        name = name_or_tool
        if not name or not isinstance(name, str):
            raise ValueError("Tool name must be a non-empty string")
        if handler is None:
            raise ValueError("handler must be provided when registering by name")

        if isinstance(handler, BaseTool):
            tool = handler
        else:
            tool = CallableTool(name=name, handler=handler, description=description)
        if description:
            tool.description = description
        self._tools[name] = tool

    def set_tool_call_callback(
        self, callback: Callable[[ToolCallRecord], None] | None
    ) -> None:
        self._on_tool_call = callback

    def execute(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """Execute a registered tool by name.

        Supports tool-call style payloads where a single dict positional argument
        is treated as keyword arguments.
        """
        started = time.time()
        call_id = uuid.uuid4().hex

        normalized_args = args
        normalized_kwargs = kwargs
        if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
            normalized_args = ()
            normalized_kwargs = dict(args[0])

        tool = self._tools.get(tool_name)
        if tool is None:
            ended = time.time()
            record = ToolCallRecord(
                call_id=call_id,
                tool=tool_name,
                tool_input={
                    "args": self._to_jsonable(list(normalized_args)),
                    "kwargs": self._to_jsonable(normalized_kwargs),
                },
                tool_output={"error": f"Tool not available: {tool_name}"},
                error="missing_tool",
                started_at=started,
                ended_at=ended,
                duration_s=ended - started,
            )
            self._append_tool_call(record)
            return record.tool_output

        error: Optional[str] = None
        output: Any
        try:
            output = tool.execute(*normalized_args, **normalized_kwargs)
        except ToolExecutionError as exc:
            output = exc.tool_output
            error = exc.kind
        except Exception:
            output = {"error": "exception", "traceback": traceback.format_exc()}
            error = "exception"

        ended = time.time()
        record = ToolCallRecord(
            call_id=call_id,
            tool=tool_name,
            tool_input={
                "args": self._to_jsonable(list(normalized_args)),
                "kwargs": self._to_jsonable(normalized_kwargs),
            },
            tool_output=self._to_jsonable(output),
            error=error,
            started_at=started,
            ended_at=ended,
            duration_s=ended - started,
        )
        self._append_tool_call(record)
        return output

    async def aexecute(self, tool_name: str, *args: Any, **kwargs: Any) -> Any:
        """Async version of `execute`."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return self.execute(tool_name, *args, **kwargs)

        if inspect.iscoroutinefunction(tool.execute):
            started = time.time()
            call_id = uuid.uuid4().hex

            normalized_args = args
            normalized_kwargs = kwargs
            if len(args) == 1 and isinstance(args[0], dict) and not kwargs:
                normalized_args = ()
                normalized_kwargs = dict(args[0])

            error: Optional[str] = None
            output: Any
            try:
                output = await tool.execute(*normalized_args, **normalized_kwargs)  # type: ignore[misc]
            except ToolExecutionError as exc:
                output = exc.tool_output
                error = exc.kind
            except Exception:
                output = {"error": "exception", "traceback": traceback.format_exc()}
                error = "exception"

            ended = time.time()
            record = ToolCallRecord(
                call_id=call_id,
                tool=tool_name,
                tool_input={
                    "args": self._to_jsonable(list(normalized_args)),
                    "kwargs": self._to_jsonable(normalized_kwargs),
                },
                tool_output=self._to_jsonable(output),
                error=error,
                started_at=started,
                ended_at=ended,
                duration_s=ended - started,
            )
            self._append_tool_call(record)
            return output

        return await asyncio.to_thread(self.execute, tool_name, *args, **kwargs)

    def _make_tool_callable(self, name: str) -> Callable[..., Any]:
        def _call(*args: Any, **kwargs: Any) -> Any:
            return self.execute(name, *args, **kwargs)

        return _call

    def snapshot_fs(self, exclude_prefixes: Optional[List[str]] = None) -> Dict[str, str]:
        """Return a deterministic file snapshot {relative_path: sha256} for the sandbox.

        Directories are skipped; logs/runs are excluded by default to reduce noise.
        """
        exclude_prefixes = exclude_prefixes or ["logs/", "runs/"]
        snapshot: Dict[str, str] = {}

        for path in self.sandbox_dir.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(self.sandbox_dir).as_posix()
            if any(rel.startswith(prefix) for prefix in exclude_prefixes):
                continue
            try:
                data = path.read_bytes()
                digest = hashlib.sha256(data).hexdigest()
                snapshot[rel] = digest
            except OSError:
                continue
        return snapshot

    def run_task(
        self,
        package: TaskPackage,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> TaskRunRecord:
        started = time.time()
        tool_proxy = self.as_tools(extra=tools)

        answer: Any = None
        verified: Optional[bool] = None
        verification_score: Optional[float] = None
        verification_details: Any = None
        verification_message: Optional[str] = None
        error: Optional[str] = None
        verification_error: Optional[str] = None
        try:
            answer = package.run_solution(tool_proxy)
            verified, verification_score, verification_details, verification_message = package.verify_with_meta(
                tool_proxy, answer
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
            task_id=package.task.task_id,  # Use task.task_id instead of package.id for consistency
            task_title=package.task.task_title,
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
        # Ensure runs_dir exists before saving
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        dump_json(self.runs_dir / f"{record.run_id}.json", record.model_dump())
        logger.debug(f"Saved task run record: {record.run_id}.json in {self.runs_dir}")
        return record

    def _append_tool_call(self, record: ToolCallRecord) -> None:
        line = json.dumps(record.model_dump(), ensure_ascii=False, default=str)
        self.tool_calls_path.parent.mkdir(parents=True, exist_ok=True)
        with self._tool_calls_lock:
            with self.tool_calls_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        if self._on_tool_call is not None:
            try:
                self._on_tool_call(record)
            except Exception:
                logger.debug("on_tool_call callback failed", exc_info=True)

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        try:
            return json.loads(json.dumps(value, ensure_ascii=False, default=str))
        except Exception:
            return str(value)


class SandboxFusionExecutor(SandboxExecutor):
    """Secure code execution environment using SandboxFusion service.

    This is an execution environment, not a tool. It executes the entire
    solution/verification logic in an isolated container.
    """

    def __init__(
        self,
        sandbox_dir: Path | str,
        base_url: str = "http://localhost:8080",
        *,
        timeout_s: int = 20,
        default_language: str = "python",
    ) -> None:
        super().__init__(sandbox_dir=sandbox_dir, timeout_s=timeout_s)
        self.sandbox_base_url = base_url
        set_endpoint(self.sandbox_base_url)
        self.default_language = default_language
        self._archive_input_name = "_input.tar.gz"
        self._archive_output_name = "_output.tar.gz"
        # Register remote-backed bash/python tools
        self.register_tool("bash", self.execute_bash, description="Remote bash via SandboxFusion")
        self.register_tool("python_runner", self.execute_python, description="Remote python via SandboxFusion")

    def _build_input_archive(self) -> str:
        buffer = BytesIO()
        with tarfile.open(mode="w:gz", fileobj=buffer) as tar:
            for path in self.sandbox_dir.rglob("*"):
                if path.is_dir():
                    continue
                rel = path.relative_to(self.sandbox_dir).as_posix()
                if rel.startswith("logs/") or rel.startswith("runs/"):
                    continue
                tar.add(path, arcname=rel)
        return base64.b64encode(buffer.getvalue()).decode("ascii")

    def _apply_output_archive(self, encoded: str) -> None:
        if not encoded:
            return
        data = base64.b64decode(encoded)
        buffer = BytesIO(data)
        tmp_dir = self.sandbox_dir / "__remote_sync__"
        if tmp_dir.exists():
            shutil.rmtree(tmp_dir)
        tmp_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(mode="r:gz", fileobj=buffer) as tar:
            tar.extractall(path=tmp_dir)

        # Clear existing workspace except logs/runs
        for path in self.sandbox_dir.iterdir():
            if path.name in {"logs", "runs", "__remote_sync__"}:
                continue
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()

        # Move synced files back
        for path in tmp_dir.rglob("*"):
            rel = path.relative_to(tmp_dir)
            target = self.sandbox_dir / rel
            if path.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
        shutil.rmtree(tmp_dir, ignore_errors=True)

    def _run_remote(
        self,
        code: str,
        *,
        language: str,
        timeout_s: Optional[int] = None,
    ) -> Dict[str, Any]:
        archive_b64 = self._build_input_archive()
        files = {self._archive_input_name: archive_b64}
        request = RunCodeRequest(
            code=code,
            language=language,
            files=files,
            fetch_files=[self._archive_output_name],
            run_timeout=float(timeout_s or self.timeout_s),
        )
        try:
            response = run_code(
                request,
                client_timeout=timeout_s or self.timeout_s,
            )
            if hasattr(response, "model_dump"):
                result = response.model_dump()
            elif hasattr(response, "dict"):
                result = response.dict()
            else:
                result = response  # type: ignore[assignment]
        except requests.exceptions.RequestException as e:
            return {
                "status": "error",
                "stdout": "",
                "stderr": f"SandboxFusion request failed: {str(e)}",
                "execution_time": 0,
                "return_code": -1,
                "raw": {},
            }

        files_out = result.get("files", {}) or {}
        output_b64 = files_out.get(self._archive_output_name, "")
        self._apply_output_archive(output_b64)

        run_result = result.get("run_result") or {}
        return {
            "status": result.get("status", "unknown"),
            "stdout": run_result.get("stdout", ""),
            "stderr": run_result.get("stderr", ""),
            "execution_time": run_result.get("execution_time", 0),
            "return_code": run_result.get("return_code", 0),
            "returncode": run_result.get("return_code", 0),
            "raw": result,
        }

    def execute_bash(
        self, command: str, timeout_s: Optional[int] = None
    ) -> Dict[str, Any]:
        wrapped = "\n".join(
            [
                "set -e",
                f"if [ -f {self._archive_input_name} ]; then tar -xzf {self._archive_input_name}; fi",
                command,
                f"tar -czf {self._archive_output_name} --warning=no-file-changed --warning=no-file-removed --ignore-failed-read --exclude={self._archive_output_name} --exclude={self._archive_input_name} .",
            ]
        )
        result = self._run_remote(wrapped, language="bash", timeout_s=timeout_s)
        return result  # type: ignore[return-value]

    def execute_python(
        self, code: str, timeout_s: Optional[int] = None
    ) -> Dict[str, Any]:
        wrapped = "\n".join(
            [
                "set -e",
                f"if [ -f {self._archive_input_name} ]; then tar -xzf {self._archive_input_name}; fi",
                "python - <<'PY'",
                code,
                "PY",
                f"tar -czf {self._archive_output_name} --warning=no-file-changed --warning=no-file-removed --ignore-failed-read --exclude={self._archive_output_name} --exclude={self._archive_input_name} .",
            ]
        )
        result = self._run_remote(wrapped, language="bash", timeout_s=timeout_s)
        return result  # type: ignore[return-value]

    def run_task(
        self,
        package: TaskPackage,
        *,
        tools: Optional[Dict[str, Callable[..., Any]]] = None,
    ) -> TaskRunRecord:
        started = time.time()
        # Run solve/verify remotely using tools.py and local data files.
        solution_code = package.solution or ""
        verification_code = package.verification or ""
        runner = textwrap.dedent(
            f'''
import json
import importlib.util
import traceback

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

tool_proxy = ToolProxy(**tools)

def _emit(payload):
    print(json.dumps(payload, ensure_ascii=False))

try:
    {solution_code}
    {verification_code}
    answer = solve(tool_proxy)
    verified = verify(tool_proxy, answer)
    _emit({{"answer": answer, "verified": verified}})
except Exception:
    _emit({{"error": traceback.format_exc()}})
'''
        ).strip()
        exec_result = self.execute_python(runner, timeout_s=self.timeout_s)
        answer: Any = None
        verified: Optional[bool] = None
        verification_score: Optional[float] = None
        verification_details: Any = None
        verification_message: Optional[str] = None
        error: Optional[str] = None
        verification_error: Optional[str] = None
        try:
            lines = (exec_result.get("stdout") or "").strip().splitlines()
            payload = json.loads(lines[-1]) if lines else {}
            if "error" in payload:
                error = payload.get("error")
            else:
                answer = payload.get("answer")
                verified = payload.get("verified")
        except Exception:
            error = "failed_to_parse_sandbox_output"

        ended = time.time()
        record = TaskRunRecord(
            task_id=package.task.task_id,
            task_title=package.task.task_title,
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
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        dump_json(self.runs_dir / f"{record.run_id}.json", record.model_dump())
        logger.debug(f"Saved task run record: {record.run_id}.json in {self.runs_dir}")
        return record

    def execute_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        result = self.execute("search", query, max_results=max_results)
        return result  # type: ignore[return-value]
