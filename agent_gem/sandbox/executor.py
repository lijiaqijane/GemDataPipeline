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
from io import BytesIO
from pathlib import Path
import requests
import shutil
import tarfile
from typing import Any, Callable, Dict, List, Optional, Tuple

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
                # privileged=True,
                # command="make run-online"

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
    run_group: str = "default"
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

    def as_tools(
        self,
        extra: Optional[Dict[str, Any]] = None,
        *,
        cache_calls: bool = False,
    ) -> ToolProxy:
        if extra:
            for name, value in extra.items():
                self.register_tool(name, value)

        tools: Dict[str, Callable[..., Any]] = {}
        cache: Dict[str, Any] = {}

        def _cache_key(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
            payload = {"args": args, "kwargs": kwargs}
            try:
                return json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
            except Exception:
                return repr(payload)

        for name in sorted(self._tools):
            base = self._make_tool_callable(name)
            if cache_calls:
                def _call(*args: Any, _base=base, _name=name, **kwargs: Any) -> Any:
                    cache_key = f"{_name}:{_cache_key(args, kwargs)}"
                    if cache_key in cache:
                        return cache[cache_key]
                    result = _base(*args, **kwargs)
                    cache[cache_key] = result
                    return result

                tools[name] = _call
            else:
                tools[name] = base
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

    @staticmethod
    def _run_group_dir_from(base: Path, run_group: str) -> Path:
        text = (run_group or "").strip()
        if not text:
            return base / "default"
        parts = []
        for raw in text.split("/"):
            raw = raw.strip()
            if not raw or raw in {".", ".."}:
                continue
            parts.append(slugify(raw))
        if not parts:
            parts = ["default"]
        return base.joinpath(*parts)

    def _run_group_dir(self, run_group: str) -> Path:
        return self._run_group_dir_from(self.runs_dir, run_group)

    def _record_run_code(self, run_id: str, package: TaskPackage, *, group_dir: Path) -> None:
        """Persist solution/verification code per run for debugging."""
        try:
            solutions_dir = group_dir / "solutions"
            verifications_dir = group_dir / "verifications"
            solutions_dir.mkdir(parents=True, exist_ok=True)
            verifications_dir.mkdir(parents=True, exist_ok=True)
            (solutions_dir / f"{run_id}.py").write_text(
                package.solution or "", encoding="utf-8"
            )
            (verifications_dir / f"{run_id}.py").write_text(
                package.verification or "", encoding="utf-8"
            )
        except Exception:
            logger.debug("Failed to record run code", exc_info=True)

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

    def annotate_run_record(
        self, *, run_group: str, run_id: str, updates: Dict[str, Any]
    ) -> None:
        if not updates:
            return
        path = self._run_group_dir(run_group) / f"{run_id}.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                return
            data.update(updates)
            dump_json(path, data)
        except Exception:
            pass

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

        # Consolidate temp scripts from SandboxFusion runs
        tmp_scripts_dir = self.logs_dir / "tmp_scripts"
        tmp_scripts_dir.mkdir(parents=True, exist_ok=True)
        for path in self.sandbox_dir.iterdir():
            if path.is_file() and path.name.startswith("tmp") and path.suffix == ".sh":
                shutil.move(str(path), str(tmp_scripts_dir / path.name))

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

    def execute_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        result = self.execute("search", query, max_results=max_results)
        return result  # type: ignore[return-value]


class CodeExecutor:
    """Utility wrapper around SandboxFusionExecutor for repo operations."""

    def __init__(self, executor: SandboxFusionExecutor) -> None:
        self.executor = executor

    # ---------- low-level runners ----------
    def run_command(self, command: str, *, timeout_s: int = 30) -> Dict[str, Any]:
        return self.executor._run_remote(command, language="bash", timeout_s=timeout_s)

    def run_code(self, code: str, *, language: str = "python", timeout_s: int = 30) -> Dict[str, Any]:
        return self.executor._run_remote(code, language=language, timeout_s=timeout_s)

    # ---------- helpers ----------
    @staticmethod
    def _extract_stdout(result: Dict[str, Any]) -> str:
        raw = result.get("raw", {}) or {}
        if isinstance(raw, dict):
            for key in ("run_result", "result", "output"):
                inner = raw.get(key)
                if isinstance(inner, dict) and inner.get("stdout"):
                    return str(inner.get("stdout", ""))
            if raw.get("stdout"):
                return str(raw.get("stdout", ""))
        return str(result.get("stdout", ""))

    @staticmethod
    def _is_success(result: Dict[str, Any]) -> bool:
        return result.get("status", "") == "Success"

    # ---------- repo operations ----------

    def detect_language(self, repo_path: str = "/workspace/repo") -> str:
        code = f"""
import os
from pathlib import Path

repo = Path("{repo_path}")
ext_map = {{
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}}

counts = {{}}
for ext, lang in ext_map.items():
    count = len(list(repo.rglob(f"*{{ext}}")))
    if count:
        counts[lang] = count

print(max(counts, key=counts.get) if counts else "unknown")
"""
        result = self.run_code(code, language="python")
        output = self._extract_stdout(result).strip()
        return output.split("\n")[-1] if output else "unknown"

    def extract_dependencies(self, repo_path: str = "/workspace/repo", language: str = "python") -> List[str]:
        if language == "python":
            code = f"""
import json
import re
from pathlib import Path

repo = Path("{repo_path}")
deps = []

# 1. Check requirements.txt
req_file = repo / "requirements.txt"
if req_file.exists():
    deps.extend([
        line.strip()
        for line in req_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ])

# 2. Check setup.py
setup_py = repo / "setup.py"
if setup_py.exists():
    content = setup_py.read_text()
    match = re.search(r"install_requires\\s*=\\s*\\[(.*?)\\]", content, re.DOTALL)
    if match:
        deps_str = match.group(1)
        pairs = re.findall(r'"([^"]+)"|\\\'([^\\\']+)\\\'', deps_str)
        deps.extend([p[0] or p[1] for p in pairs])

# 3. Check pyproject.toml
pyproject = repo / "pyproject.toml"
if pyproject.exists():
    content = pyproject.read_text()
    match = re.search(r"dependencies\\s*=\\s*\\[(.*?)\\]", content, re.DOTALL)
    if match:
        deps_str = match.group(1)
        pairs = re.findall(r'"([^"]+)"|\\\'([^\\\']+)\\\'', deps_str)
        deps.extend([p[0] or p[1] for p in pairs])

# 4. Check environment.yml (conda)
env_yml_files = [
    repo / "environment.yml",
    repo / "environment.yaml",
    repo / "conda.yml",
    repo / "conda.yaml"
]

for env_file in env_yml_files:
    if env_file.exists():
        try:
            import yaml
            content = yaml.safe_load(env_file.read_text())
            
            # Extract conda dependencies
            if 'dependencies' in content:
                for dep in content['dependencies']:
                    if isinstance(dep, str):
                        # Simple package name or package=version
                        # Extract just the package name before = or version specifier
                        pkg_name = re.split(r'[=<>!]', dep)[0].strip()
                        if pkg_name and not pkg_name.startswith('-'):
                            deps.append(pkg_name)
                    elif isinstance(dep, dict) and 'pip' in dep:
                        # Pip dependencies nested in conda environment
                        deps.extend(dep['pip'])
        except ImportError:
            # yaml not available, try simple parsing
            try:
                content = env_file.read_text()
                # Find dependencies section
                in_deps = False
                in_pip = False
                for line in content.splitlines():
                    line = line.strip()
                    if line.startswith('dependencies:'):
                        in_deps = True
                        continue
                    if in_deps:
                        if line.startswith('-'):
                            if 'pip:' in line:
                                in_pip = True
                                continue
                            elif in_pip:
                                # pip dependency
                                pkg = line.lstrip('- ').strip()
                                deps.append(pkg)
                            else:
                                # conda dependency
                                pkg = line.lstrip('- ').strip()
                                pkg_name = re.split(r'[=<>!]', pkg)[0].strip()
                                if pkg_name:
                                    deps.append(pkg_name)
                        elif not line.startswith('#') and line and not line.endswith(':'):
                            in_deps = False
            except Exception:
                pass
        break  # Only process the first found environment file

print(json.dumps(sorted(set(deps))))
"""
            result = self.run_code(code, language="python")
            output = self._extract_stdout(result).strip()
            
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return []

        if language in {"javascript", "typescript"}:
            code = f"""
import json
from pathlib import Path

repo = Path("{repo_path}")
pkg_json = repo / "package.json"
deps = []

if pkg_json.exists():
    try:
        data = json.loads(pkg_json.read_text())
        deps.extend(list(data.get("dependencies", {{}}).keys()))
        deps.extend(list(data.get("devDependencies", {{}}).keys()))
    except Exception:
        pass

print(json.dumps(sorted(set(deps))))
"""
            result = self.run_code(code, language="python")
            output = self._extract_stdout(result).strip()
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                return []

        return []

    def list_files(self, directory: str, extensions: Optional[tuple] = None) -> List[str]:
        """
        List files in the core source directory of a repository.
        
        Args:
            directory: Repository directory path
            extensions: Optional tuple of file extensions to filter (e.g., (".py", ".js"))
                       If None, uses default code extensions
                       
        Returns:
            List of file paths relative to the repository root
        """
        # Convert extensions tuple to JSON-compatible list
        if extensions is None:
            extensions_json = '[".py", ".js", ".ts", ".java", ".go", ".rs", ".cpp", ".c", ".h", ".hpp"]'
        else:
            extensions_json = json.dumps(list(extensions))
        
        code = f"""
import json
import subprocess
from pathlib import Path

base = Path("{directory}")

# Find the core source directory
core_dir = base

# Strategy 1: Try to get real repo name from git remote
try:
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=str(base),
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        remote_url = result.stdout.strip()
        # Extract repo name from URL (e.g., "https://github.com/numpy/numpy.git" -> "numpy")
        repo_name = remote_url.rstrip("/").rstrip(".git").split("/")[-1]
    else:
        repo_name = None
except Exception:
    repo_name = None

# Strategy 2: Find directories with __init__.py at top level (likely package directories)
if not repo_name or not (base / repo_name).is_dir():
    top_level_packages = [
        d for d in base.iterdir()
        if d.is_dir() and (d / "__init__.py").exists() and not d.name.startswith(".")
    ]
    if top_level_packages:
        # Use the first package directory found
        repo_name = top_level_packages[0].name

# Strategy 3: Try common source directory names
if repo_name and (base / repo_name).is_dir():
    candidate = base / repo_name
    if (candidate / "__init__.py").exists():
        core_dir = candidate
elif top_level_packages:
    core_dir = top_level_packages[0]
else:
    # Fallback: check common source directories
    for common_name in ["src", "lib"]:
        candidate = base / common_name
        if candidate.is_dir():
            # Check if it contains Python packages
            if (candidate / "__init__.py").exists() or any(candidate.glob("**/__init__.py")):
                core_dir = candidate
                break

# List files, filtering by specified extensions
code_extensions = set({extensions_json})
files = []
for p in core_dir.rglob("*"):
    if p.is_file() and p.suffix in code_extensions:
        try:
            files.append(str(p.relative_to(base)))
        except ValueError:
            files.append(str(p))
        if len(files) >= 50:
            break

print(json.dumps(files))
"""
        result = self.run_code(code, language="python")
        output = self._extract_stdout(result).strip()
        
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return []

    def read_file(self, file_path: str) -> Optional[str]:
        code = f"""
from pathlib import Path
try:
    print(Path("{file_path}").read_text())
except Exception as exc:
    print(f"ERROR: {{exc}}")
"""
        result = self.run_code(code, language="python")
        output = self._extract_stdout(result)
        if output.startswith("ERROR:"):
            return None
        return output

    def write_file(self, file_path: str, content: str) -> bool:
        code = f"""
from pathlib import Path
path = Path("{file_path}")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text('''{content}''')
print("OK")
"""
        result = self.run_code(code, language="python")
        output = self._extract_stdout(result)
        return "OK" in output

    def run_tests_in_repo(self, test_command: str = "pytest -v", repo_path: str = "/workspace/repo") -> Dict[str, Any]:
        cmd = f"cd {repo_path} && {test_command}"
        return self.run_command(cmd)

    # ---------- git helpers ----------
    def git_command(self, args: List[str], cwd: str = "/workspace/repo") -> Dict[str, Any]:
        cmd = f"cd {cwd} && git " + " ".join(args)
        return self.run_command(cmd)

    def apply_code_to_file(self, file_path: str, code: str, repo_path: str = "/workspace/repo") -> bool:
        full_path = f"{repo_path}/{file_path}"
        cmd = f"cat > {full_path} << 'EOF'\n{code}\nEOF"
        result = self.run_command(cmd)
        return self._is_success(result)

    def apply_patch(self, patch_content: str, repo_path: str = "/workspace/repo") -> bool:
        tmp_name = f"/workspace/tmp/patch_{uuid.uuid4().hex[:8]}.patch"
        write_cmd = f"mkdir -p /workspace/tmp && cat > {tmp_name} << 'EOF'\n{patch_content}\nEOF"
        apply_cmd = f"cd {repo_path} && patch -p1 --verbose < {tmp_name}"
        result_write = self.run_command(write_cmd)
        if not self._is_success(result_write):
            return False
        result_apply = self.run_command(apply_cmd)
        return self._is_success(result_apply)
    
    def apply_patch_with_result(self, patch_content: str, repo_path: str = "/workspace/repo", fuzz: int = 3) -> Tuple[bool, str]:
        """
        Apply patch and return success status with error message.
        
        Args:
            patch_content: Patch content in unified diff format
            repo_path: Path to repository
            fuzz: Fuzz factor for patch application
            
        Returns:
            Tuple of (success, error_message)
        """
        tmp_name = f"/workspace/tmp/patch_{uuid.uuid4().hex[:8]}.patch"
        write_cmd = f"mkdir -p /workspace/tmp && cat > {tmp_name} << 'EOF'\n{patch_content}\nEOF"
        apply_cmd = f"cd {repo_path} && patch -p1 --fuzz={fuzz} --verbose < {tmp_name}"
        result_write = self.run_command(write_cmd)
        if not self._is_success(result_write):
            return False, f"Failed to write patch file: {self._extract_stdout(result_write)}"
        result_apply = self.run_command(apply_cmd)
        ok = self._is_success(result_apply)
        err = self._extract_stdout(result_apply) if not ok else ""
        return ok, err

    def validate_patch(self, patch_content: str, repo_path: str = "/workspace/repo", fuzz: int = 3) -> Tuple[bool, str]:
        tmp_name = f"/workspace/tmp/validate_{uuid.uuid4().hex[:8]}.patch"
        write_cmd = f"mkdir -p /workspace/tmp && cat > {tmp_name} << 'EOF'\n{patch_content}\nEOF"
        validate_cmd = f"cd {repo_path} && patch -p1 --dry-run --fuzz={fuzz} --verbose < {tmp_name}"
        result_write = self.run_command(write_cmd)
        if not self._is_success(result_write):
            return False, self._extract_stdout(result_write)
        result_validate = self.run_command(validate_cmd)
        ok = self._is_success(result_validate)
        err = self._extract_stdout(result_validate) if not ok else ""
        return ok, err

    def create_git_branch(self, branch_name: str, repo_path: str = "/workspace/repo") -> bool:
        result = self.git_command(["checkout", "-b", branch_name], cwd=repo_path)
        return self._is_success(result)

    def checkout_git_branch(self, branch_name: str, repo_path: str = "/workspace/repo") -> bool:
        result = self.git_command(["checkout", branch_name], cwd=repo_path)
        return self._is_success(result)

    def git_commit(self, message: str, file_paths: Optional[List[str]] = None, repo_path: str = "/workspace/repo") -> bool:
        self.git_command(["config", "user.email", "codeagent@example.com"], cwd=repo_path)
        self.git_command(["config", "user.name", "CodeAgent"], cwd=repo_path)
        if file_paths:
            for path in file_paths:
                self.git_command(["add", path], cwd=repo_path)
        else:
            self.git_command(["add", "-A"], cwd=repo_path)
        result = self.git_command(["commit", "-m", message], cwd=repo_path)
        return self._is_success(result)

    def get_git_diff(self, branch1: str, branch2: str, repo_path: str = "/workspace/repo") -> str:
        result = self.git_command(["diff", branch1, branch2], cwd=repo_path)
        return self._extract_stdout(result)
