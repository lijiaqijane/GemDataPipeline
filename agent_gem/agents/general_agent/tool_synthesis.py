from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any, TYPE_CHECKING

from agent_gem.core.task_schema import ToolSpec
from agent_gem.sandbox import SandboxExecutor
from agent_gem.tools import CallableTool

from ..base import BaseAgent, TaskContext

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest  # noqa: F401

logger = logging.getLogger(__name__)


class ToolSynthesisMixin:
    """Tool generation, compilation, and registration helpers."""
    _SUBMIT_RESULT_TOOL = "submit_result"

    @staticmethod
    def _tool_code_uses_mcp_tool(code: str) -> bool:
        """Detect whether code uses @mcp.tool decorators (which require a runtime `mcp` binding)."""
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                # @mcp.tool
                if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
                    if dec.value.id == "mcp" and dec.attr == "tool":
                        return True
                # @mcp.tool(...)
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and isinstance(dec.func.value, ast.Name):
                    if dec.func.value.id == "mcp" and dec.func.attr == "tool":
                        return True
        return False

    @staticmethod
    def _tool_code_has_mcp_binding(code: str) -> bool:
        """Detect an import or assignment that binds the name `mcp` in module scope."""
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.asname == "mcp" or alias.name == "mcp":
                        return True
            if isinstance(node, ast.ImportFrom) and node.module == "mcp":
                # This binds specific names, not `mcp` itself, but allow it if user still uses @mcp.tool
                # (In practice, LLM should prefer `import mcp` for @mcp.tool.)
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "mcp":
                        return True
            if isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name) and node.target.id == "mcp":
                    return True
            if isinstance(node, ast.Try):
                # allow patterns like:
                # try: import mcp
                # except: mcp = ...
                for inner in node.body:
                    if isinstance(inner, ast.Import):
                        for alias in inner.names:
                            if alias.asname == "mcp" or alias.name == "mcp":
                                return True
                for inner in node.handlers:
                    for hstmt in inner.body:
                        if isinstance(hstmt, ast.Assign):
                            for target in hstmt.targets:
                                if isinstance(target, ast.Name) and target.id == "mcp":
                                    return True
        return False

    @staticmethod
    def _tool_code_reads_data(code: str) -> bool:
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                return True
            if isinstance(func, ast.Attribute):
                attr = func.attr
                if attr in {"read_text", "read_bytes", "read_csv", "read_json", "connect"}:
                    return True
                if isinstance(func.value, ast.Name):
                    base = func.value.id
                    if base == "csv" and attr == "DictReader":
                        return True
                    if base == "json" and attr == "load":
                        return True
                    if base == "sqlite3" and attr == "connect":
                        return True
                    if base in {"pd", "pandas"} and attr in {"read_csv", "read_json"}:
                        return True
        return False

    @staticmethod
    def _tool_code_embeds_dataset(code: str) -> bool:
        """Detect large embedded list-of-dict literals (likely sample data leakage)."""
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                value = node.value
            else:
                continue
            if isinstance(value, (ast.List, ast.Tuple)):
                dict_items = [elt for elt in value.elts if isinstance(elt, ast.Dict)]
                if len(dict_items) >= 3:
                    for dct in dict_items:
                        key_count = sum(1 for k in dct.keys if k is not None)
                        if key_count >= 2:
                            return True
        return False

    @classmethod
    def _validate_tool_code(cls, code: str) -> tuple[bool, list[str]]:
        """Return (ok, reasons). ok means code reads from local data sources."""
        reasons: list[str] = []
        uses_mcp = cls._tool_code_uses_mcp_tool(code)
        has_mcp = cls._tool_code_has_mcp_binding(code)
        if uses_mcp and not has_mcp:
            reasons.append("missing_mcp_import_or_binding")
            return False, reasons
        reads_data = cls._tool_code_reads_data(code)
        embeds_data = cls._tool_code_embeds_dataset(code)
        if not reads_data:
            reasons.append("no_data_io_detected")
            if embeds_data:
                reasons.append("embedded_dataset_literal")
            return False, reasons
        if embeds_data:
            reasons.append("embedded_dataset_literal")
        return True, reasons

    @staticmethod
    def _sanitize_llm_code(raw: str) -> str:
        """Best-effort cleanup for LLM outputs that omit code fences."""
        text = (raw or "").strip()
        if not text:
            return ""
        # If code blocks exist, join them.
        blocks = BaseAgent._extract_code_blocks(text)
        if blocks:
            return "\n\n".join(BaseAgent._strip_code_fences(block) for block in blocks).strip()
        # Otherwise, strip leading prose until a likely code line.
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("import ", "from ", "@", "def ", "class ")):
                start = i
                break
        cleaned = "\n".join(lines[start:]).strip()
        return cleaned.replace("```", "").strip()

    @staticmethod
    def _submit_result_handler(result: Any) -> Any:
        return result

    def _submit_result_spec(self) -> ToolSpec:
        def submit_result(result: Any) -> Any:
            """Submit the final answer payload from solve()."""
            return result

        return ToolSpec.from_function(
            submit_result,
            name=self._SUBMIT_RESULT_TOOL,
            description="Submit the final answer payload.",
            meta={"system": True},
        )

    def _ensure_submit_result_tool(self, tool_specs: list[ToolSpec]) -> list[ToolSpec]:
        if any(spec.name == self._SUBMIT_RESULT_TOOL for spec in tool_specs):
            return tool_specs
        return tool_specs + [self._submit_result_spec()]

    def _synthesize_task_tools(
        self,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        data_profile: dict[str, Any],
    ) -> tuple[list[ToolSpec], str]:
        """Generate task-specific tools using detected data sources."""
        data_profile_json = json.dumps(self._compact_data_profile(data_profile), ensure_ascii=False)
        if len(data_profile_json) > 12000:
            data_profile_json = data_profile_json[:12000] + "..."
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        last_raw = ""
        specs: list[ToolSpec] = []
        tools_code = ""
        submit_persists = False
        last_code_reasons: list[str] = []
        for attempt in range(1, 4):
            prompt = (
                "You are designing a deterministic toolset for a sandboxed RL agent.\n"
                "Goal: generate 3-5 @mcp.tool functions tailored to the AVAILABLE local data sources.\n"
                "Hard rules:\n"
                "- IMPORTANT: decorators execute at import-time. Since you MUST use @mcp.tool, you MUST also bind `mcp`.\n"
                "  Add `import mcp` near the top (preferred). If unavailable, you may create a tiny fallback shim that defines\n"
                "  `mcp.tool` as an identity decorator. Do NOT implement tool logic in the framework; all tool bodies must be yours.\n"
                "- NO network, NO external APIs. Only read the listed local files/SQLite.\n"
                "- Deterministic: set random.seed(0) if randomness is used.\n"
                "- Tools must accept (query: str, max_results: int = 5) and return list[dict].\n"
                "- Implement the body to actually parse the data sources; do not stub/raise.\n"
                "- Read from the local files/SQLite shown in the data profile; do NOT hardcode sample records.\n"
                "- You MUST define a tool named submit_result with signature submit_result(result) that returns `result`.\n"
                "  submit_result MUST be implemented (no pass/raise) and MUST persist the payload to submitted_result.json\n"
                "  in the current working directory for inspection.\n"
                "- If fields are insufficient, add additional tools or extend returned dicts to cover likely needs.\n"
                "- Prefer descriptive snake_case names; avoid 'bash'/'search'/'python_runner'.\n"
                "Data sources (JSON, sampled schemas):\n"
                f"{data_profile_json}\n"
                f"Topic: {topic}\n"
                f"Curated records sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
                "Output ONLY one Python code block defining the tools (including imports).\n"
                "Do NOT include prose. Ensure each function has the @mcp.tool decorator."
            )
            if attempt > 1:
                prompt += "\nPrevious attempt returned no valid @mcp.tool code. Try again with a valid code block."
                logger.warning("Tool synthesis retrying (attempt %s/3).", attempt)
            if last_code_reasons:
                prompt += (
                    "\nPrevious attempt issues: "
                    + ", ".join(last_code_reasons)
                    + ". Ensure tools READ local files (csv/json/sqlite) and avoid inline sample datasets."
                )
            raw = self.llm.simple_complete(prompt, temperature=0.6, max_tokens=max_tokens)
            last_raw = raw
            ctx.add_step({"type": "tool_synthesis", "content": raw, "attempt": attempt})
            specs = self._extract_mcp_tools(raw)
            tools_code = self._sanitize_llm_code(raw)
            # Enforce: submit_result must be present and implemented by the agent.
            has_submit = any(spec.name == self._SUBMIT_RESULT_TOOL for spec in specs)
            submit_stubbed = False
            try:
                if tools_code and "def submit_result" in tools_code:
                    tree = ast.parse(tools_code)
                    for node in tree.body:
                        if isinstance(node, ast.FunctionDef) and node.name == self._SUBMIT_RESULT_TOOL:
                            body = node.body
                            if (not body) or all(isinstance(stmt, ast.Pass) for stmt in body):
                                submit_stubbed = True
                            if len(body) == 1 and isinstance(body[0], ast.Raise):
                                submit_stubbed = True
                            for stmt in ast.walk(node):
                                if isinstance(stmt, ast.Raise):
                                    submit_stubbed = True
                                    break
            except Exception:
                # If parsing fails, treat as invalid.
                submit_stubbed = True

            submit_persists = bool(tools_code and "submitted_result.json" in tools_code)
            code_ok, code_reasons = (False, ["no_tools_code"])
            if tools_code:
                code_ok, code_reasons = self._validate_tool_code(tools_code)

            if specs and has_submit and not submit_stubbed and submit_persists and code_ok:
                if code_reasons:
                    ctx.add_step(
                        {
                            "type": "tool_synthesis_warning",
                            "tool_code_reasons": code_reasons,
                        }
                    )
                break
            if specs and (not has_submit or submit_stubbed or not submit_persists or not code_ok):
                last_code_reasons = code_reasons or []
                ctx.add_step(
                    {
                        "type": "tool_synthesis_missing_submit_result",
                        "has_submit_result": bool(has_submit),
                        "submit_stubbed": bool(submit_stubbed),
                        "submit_persists": bool(submit_persists),
                        "tool_code_valid": bool(code_ok),
                        "tool_code_reasons": code_reasons,
                    }
                )

        seen: set[str] = set()
        filtered: list[ToolSpec] = []
        for spec in specs:
            if spec.name in {"bash", "search", "python_runner"}:
                continue
            if spec.name in seen:
                continue
            seen.add(spec.name)
            filtered.append(
                spec.copy(
                    update={
                        "description": spec.description
                        or f"Query curated records about {topic}",
                        "meta": (spec.meta or {}) | {"topic": topic},
                    }
                )
            )

        if not filtered:
            ctx.add_step(
                {
                    "type": "tool_synthesis_failed",
                    "error": "no_tools_parsed",
                    "raw_preview": (last_raw or "")[:400],
                }
            )
            raise RuntimeError("Tool synthesis failed: no tools parsed from LLM output.")
        if not submit_persists:
            raise RuntimeError(
                "Tool synthesis failed: submit_result must persist to submitted_result.json."
            )
        if not tools_code or not self._validate_tool_code(tools_code)[0]:
            raise RuntimeError("Tool synthesis failed: tool code not data-driven.")
        # Keep submit_result in the tool_set; it must be implemented in tools.py by the agent.
        specs = self._ensure_submit_result_tool(filtered)
        ctx.add_step(
            {
                "type": "tool_synthesis",
                "tool_count": len(specs),
                "tools": [spec.model_dump() for spec in specs],
            }
        )
        # Write tools.py with actual implementations for standalone execution
        if tools_code:
            tools_path = sandbox.sandbox_dir / "tools.py"
            implemented_code = self._generate_tool_implementations(
                tools_code=tools_code,
                sandbox_dir=sandbox.sandbox_dir,
                tool_specs=specs,
                topic=topic,
                data_profile=data_profile,
            )
            tools_path.write_text(implemented_code, encoding="utf-8")
            compile_result = sandbox.execute_bash("python -m py_compile tools.py")
            ctx.add_step(
                {
                    "type": "tool_compile",
                    "returncode": compile_result.get("returncode"),
                    "stdout": compile_result.get("stdout", "")[:500],
                    "stderr": compile_result.get("stderr", "")[:500],
                }
            )
            # Smoke test: ensure tools.py is importable (py_compile does NOT execute decorators).
            try:
                import_result = sandbox.execute_bash('python -c "import tools; print(\'TOOLS_IMPORT_OK\')"')
                ctx.add_step(
                    {
                        "type": "tool_import_smoketest",
                        "returncode": import_result.get("returncode"),
                        "stdout": import_result.get("stdout", "")[:500],
                        "stderr": import_result.get("stderr", "")[:500],
                    }
                )
            except Exception as exc:
                ctx.add_step(
                    {
                        "type": "tool_import_smoketest_failed",
                        "error": str(exc)[:500],
                    }
                )
                raise RuntimeError(
                    "Tool synthesis failed: tools.py is not importable. "
                    "If you used @mcp.tool, ensure `import mcp` (or a small shim defining mcp.tool) exists."
            )
        return specs, tools_code

    def _compact_data_profile(self, data_profile: dict[str, Any]) -> dict[str, Any]:
        """Trim large samples to keep prompts within limits."""
        compact: dict[str, Any] = {}
        for key, items in data_profile.items():
            if not isinstance(items, list):
                compact[key] = items
                continue
            trimmed = []
            for item in items[:5]:
                if not isinstance(item, dict):
                    trimmed.append(item)
                    continue
                small = dict(item)
                if "sample_rows" in small and isinstance(small["sample_rows"], list):
                    small["sample_rows"] = small["sample_rows"][:2]
                if "sample_items" in small and isinstance(small["sample_items"], list):
                    small["sample_items"] = small["sample_items"][:2]
                if "tables" in small and isinstance(small["tables"], list):
                    tables = []
                    for t in small["tables"][:4]:
                        if isinstance(t, dict):
                            t = dict(t)
                            if "rows" in t and isinstance(t["rows"], list):
                                t["rows"] = t["rows"][:2]
                            tables.append(t)
                        else:
                            tables.append(t)
                    small["tables"] = tables
                trimmed.append(small)
            compact[key] = trimmed
        return compact

    def _generate_tool_implementations(
        self,
        *,
        tools_code: str,
        sandbox_dir: Path,
        tool_specs: list[ToolSpec],
        topic: str,
        data_profile: dict[str, Any],
    ) -> str:
        """Return agent-authored tools code with minimal formatting cleanup.

        User requirement: do NOT inject/override any tool implementations here.
        We only validate and fail-fast so the agent can regenerate.
        """
        code = (tools_code or "").strip()
        if not code:
            raise RuntimeError("tools.py generation failed: empty tools code")

        # Validate: tool code must be data-driven (read local files) and avoid embedding datasets.
        code_ok, reasons = self._validate_tool_code(code)
        if not code_ok:
            raise RuntimeError(f"tools.py invalid (not data-driven): {reasons}")

        # Validate: submit_result must be implemented by the agent and persist the payload.
        if "def submit_result" not in code:
            raise RuntimeError("tools.py missing required submit_result implementation (agent-authored).")
        if "submitted_result.json" not in code:
            raise RuntimeError("tools.py submit_result must persist to submitted_result.json (agent-authored).")

        # Validate: required tool functions should exist in the module source.
        try:
            tree = ast.parse(code)
            defined = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
        except Exception as exc:
            raise RuntimeError(f"tools.py parse failed: {exc}")
        required = {spec.name for spec in tool_specs}
        missing = sorted(name for name in required if name not in defined)
        if missing:
            raise RuntimeError(f"tools.py missing required tool defs: {missing}")

        return code + ("\n" if not code.endswith("\n") else "")

    def _self_test_tools(
        self,
        tool_specs: list[ToolSpec],
        sandbox: SandboxExecutor,
        topic: str,
        ctx: TaskContext,
    ) -> dict[str, Any]:
        """Invoke each tool with a few deterministic queries to capture schema samples."""
        profile: dict[str, Any] = {}
        queries = [topic, "sample", "test"]
        for spec in tool_specs:
            if spec.name == self._SUBMIT_RESULT_TOOL:
                continue
            samples = []
            keys: set[str] = set()
            for q in queries:
                try:
                    output = sandbox.execute(spec.name, q, max_results=3)
                except Exception as exc:
                    output = {"error": str(exc)}
                samples.append(output)
                if isinstance(output, list):
                    for item in output:
                        if isinstance(item, dict):
                            keys.update(item.keys())
            profile[spec.name] = {
                "queries": queries,
                "samples": samples[:2],
                "fields": sorted(keys),
            }
        ctx.add_step(
            {
                "type": "tool_self_test",
                "content": json.dumps(profile, ensure_ascii=False)[:2000],
            }
        )
        return profile

    @staticmethod
    def _is_non_empty_sample(sample: Any) -> bool:
        if sample is None:
            return False
        if isinstance(sample, dict):
            return len(sample) > 0
        if isinstance(sample, list):
            return len(sample) > 0
        if isinstance(sample, str):
            return bool(sample.strip())
        return True

    @staticmethod
    def _expected_fields_from_format(submit_result_format: Any) -> set[str]:
        fmt = submit_result_format
        if isinstance(fmt, str):
            try:
                fmt = json.loads(fmt)
            except Exception:
                return set()
        fields: set[str] = set()
        if isinstance(fmt, dict):
            props = fmt.get("properties") if isinstance(fmt.get("properties"), dict) else None
            if props:
                fields.update([k for k in props.keys() if isinstance(k, str)])
            required = fmt.get("required")
            if isinstance(required, list):
                fields.update([k for k in required if isinstance(k, str)])
        if isinstance(fmt, list) and fmt and isinstance(fmt[0], dict):
            fields.update([k for k in fmt[0].keys() if isinstance(k, str)])
        return fields

    def _needs_tool_regeneration(
        self,
        tool_selftest: dict[str, Any],
        required_fields: set[str] | None = None,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not tool_selftest:
            return True, ["selftest_missing"]
        union_fields = set()
        for name, info in tool_selftest.items():
            if name == self._SUBMIT_RESULT_TOOL:
                continue
            if not isinstance(info, dict):
                reasons.append(f"{name}:invalid_profile")
                continue
            fields = set(info.get("fields") or [])
            union_fields |= {f for f in fields if isinstance(f, str)}
            samples = info.get("samples") or []
            has_data = any(self._is_non_empty_sample(s) for s in samples)
            if not fields or not has_data:
                reasons.append(f"{name}:empty_output")
        if required_fields and required_fields and not (required_fields & union_fields):
            reasons.append("missing_required_fields")
        return bool(reasons), reasons

    def _regenerate_tools_with_selftest(
        self,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        data_profile: dict[str, Any],
        tool_selftest: dict[str, Any],
        existing_specs: list[ToolSpec],
        previous_code: str = "",
        required_fields: set[str] | None = None,
    ) -> tuple[list[ToolSpec], str, dict[str, Any]]:
        prompt = (
            "Self-tests show missing fields or empty outputs. Regenerate an executable toolset.\n"
            "Rules:\n"
            "- Use ONLY local CSV/JSON/SQLite shown in the data profile; no network.\n"
            "- Implement full logic (reading/parsing/filtering); do NOT emit stubs, pass, or RuntimeError placeholders.\n"
            "- Keep existing tool names when possible; you may add up to 2 new tools to expose missing fields.\n"
            "- Each tool signature: (query: str, max_results: int = 5) -> list[dict].\n"
            "- Deterministic: set random.seed(0) if randomness appears.\n"
            f"Topic: {topic}\n"
            f"Data profile: {json.dumps(data_profile, ensure_ascii=False)[:1200]}\n"
            f"Self-test report: {json.dumps(tool_selftest, ensure_ascii=False)[:1200]}\n"
            f"Existing tools: {json.dumps([s.model_dump() for s in existing_specs], ensure_ascii=False)[:1200]}\n"
            f"Required fields to expose (if any): {sorted(required_fields) if required_fields else []}\n"
            f"Prior tool code (truncated): {previous_code[:800]}\n"
            "Return exactly ONE Python code block with @mcp.tool definitions (imports included)."
        )
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        raw = self.llm.simple_complete(prompt, temperature=0.55, max_tokens=max_tokens)
        ctx.add_step({"type": "tool_regeneration", "content": raw})

        specs = self._extract_mcp_tools(raw)
        new_code = self._sanitize_llm_code(raw)
        if not specs or not new_code:
            return existing_specs, previous_code, tool_selftest
        code_ok, code_reasons = self._validate_tool_code(new_code)
        if not code_ok:
            ctx.add_step(
                {
                    "type": "tool_regeneration_invalid_tool_code",
                    "tool_code_reasons": code_reasons,
                }
            )
            return existing_specs, previous_code, tool_selftest
        if "submitted_result.json" not in new_code:
            ctx.add_step(
                {
                    "type": "tool_regeneration_missing_submit_result",
                    "submit_persists": False,
                }
            )
            return existing_specs, previous_code, tool_selftest
        specs = self._ensure_submit_result_tool(specs)

        tools_path = sandbox.sandbox_dir / "tools.py"
        implemented_code = self._generate_tool_implementations(
            tools_code=new_code,
            sandbox_dir=sandbox.sandbox_dir,
            tool_specs=specs,
            topic=topic,
            data_profile=data_profile,
        )
        tools_path.write_text(implemented_code, encoding="utf-8")
        sandbox.execute_bash("python -m py_compile tools.py")
        # Ensure import-time decorators don't fail.
        sandbox.execute_bash('python -c "import tools; print(\'TOOLS_IMPORT_OK\')"')
        self._register_task_tools(specs, sandbox, ctx, tools_code=new_code)
        updated_selftest = self._self_test_tools(specs, sandbox, topic, ctx)
        return specs, new_code, updated_selftest

    def _register_task_tools(
        self,
        tool_specs: list[ToolSpec],
        sandbox: SandboxExecutor,
        ctx: TaskContext,
        *,
        tools_code: str | None = None,
    ) -> None:
        tool_specs = self._ensure_submit_result_tool(list(tool_specs))
        records_path = sandbox.sandbox_dir / self._RECORDS_FILENAME
        tools_path = sandbox.sandbox_dir / "tools.py"
        module = None
        if tools_path.exists():
            try:
                # Ensure a usable @mcp.tool decorator exists at import-time.
                # This does NOT provide any tool implementations; it only prevents import failures
                # when a real `mcp` package is present but lacks `tool`.
                import sys
                import types
                try:
                    import mcp  # type: ignore
                except Exception:
                    mcp = types.ModuleType("mcp")
                if not hasattr(mcp, "tool"):
                    def _tool(func=None, **kwargs):
                        if func is None:
                            def wrapper(f):
                                return f
                            return wrapper
                        return func
                    mcp.tool = _tool  # type: ignore[attr-defined]
                sys.modules["mcp"] = mcp

                spec_module = importlib.util.spec_from_file_location("generated_tools", tools_path)
                if spec_module and spec_module.loader:
                    module = importlib.util.module_from_spec(spec_module)
                    spec_module.loader.exec_module(module)
            except Exception:
                logger.debug("Failed to import generated tools.py", exc_info=True)

        for spec in tool_specs:
            registered = False
            if module and hasattr(module, spec.name):
                handler = getattr(module, spec.name)
                if callable(handler):
                    sandbox.register_tool(
                        CallableTool(
                            name=spec.name,
                            description=spec.description,
                            handler=handler,
                        )
                    )
                    registered = True
            if not registered:
                # User requirement: ALL tool implementations must be agent-authored.
                # Therefore, missing tools are a hard error (no fallback).
                raise RuntimeError(
                    f"Missing agent-authored tool implementation in tools.py: {spec.name}"
                    )

        ctx.add_step(
            {
                "type": "tool_registration",
                "registered_tools": [spec.name for spec in tool_specs],
                "tools_code_preview": (tools_code or "")[:300],
            }
        )
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)

    def _augment_toolset(
        self,
        topic: str,
        records: list[dict[str, Any]],
        existing_specs: list[ToolSpec],
        data_profile: dict[str, Any] | None,
        ctx: TaskContext,
        sandbox: SandboxExecutor,
    ) -> tuple[list[ToolSpec], str, bool]:
        """Generate incremental tools to augment the current toolset."""
        existing_names = {spec.name for spec in existing_specs}
        data_profile_json = json.dumps(data_profile or {}, ensure_ascii=False)
        prompt = (
            "You are augmenting an existing toolset to make a task solvable and verifiable.\n"
            "Return Python code defining 1-2 NEW tools decorated with @mcp.tool(...), avoiding duplicates.\n"
            "Constraints:\n"
            "- Tool names must be unique, snake_case, and NOT 'bash'/'search'/'python_runner'.\n"
            "- Each tool MUST accept (query: str, max_results: int = 5) and return list[dict].\n"
            "- Implement real logic that reads ONLY local CSV/JSON/SQLite or the provided records; no stubs, no pass/raise placeholders.\n"
            "- Deterministic: set random.seed(0) if any randomness is used.\n"
            "Only output a single Python code block.\n"
            f"Topic: {topic}\n"
            f"Existing tools: {sorted(existing_names)}\n"
            f"Records (JSON sample): {json.dumps(records[:5], ensure_ascii=False)}\n"
            f"Data profile (JSON): {data_profile_json[:800]}\n"
        )
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        raw = self.llm.simple_complete(prompt, temperature=0.55, max_tokens=max_tokens)
        ctx.add_step({"type": "tool_augmentation", "content": raw})

        specs = self._extract_mcp_tools(raw)
        tools_code = self._sanitize_llm_code(raw)

        new_specs: list[ToolSpec] = []
        for spec in specs:
            if spec.name in {"bash", "search", "python_runner"}:
                continue
            if spec.name in existing_names:
                continue
            existing_names.add(spec.name)
            new_specs.append(
                spec.copy(
                    update={
                        "description": spec.description
                        or f"Augmented tool for {topic}",
                        "meta": (spec.meta or {}) | {"topic": topic, "augmented": True},
                    }
                )
            )

        if not new_specs:
            return existing_specs, tools_code, False
        if tools_code:
            code_ok, code_reasons = self._validate_tool_code(tools_code)
            if not code_ok:
                ctx.add_step(
                    {
                        "type": "tool_augmentation_invalid_tool_code",
                        "tool_code_reasons": code_reasons,
                    }
                )
            return existing_specs, tools_code, False

        if tools_code:
            tools_path = sandbox.sandbox_dir / "tools.py"
            implemented_code = self._generate_tool_implementations(
                tools_code=tools_code,
                sandbox_dir=sandbox.sandbox_dir,
                tool_specs=new_specs,
                topic=topic,
                data_profile=data_profile or {},
            )
            mode = "a" if tools_path.exists() else "w"
            with tools_path.open(mode, encoding="utf-8") as handle:
                if tools_path.exists():
                    handle.write("\n\n")
                handle.write(implemented_code)
            sandbox.execute_bash("python -m py_compile tools.py")
            sandbox.execute_bash('python -c "import tools; print(\'TOOLS_IMPORT_OK\')"')

        self._register_task_tools(new_specs, sandbox, ctx, tools_code=tools_code)
        combined_specs = self._ensure_submit_result_tool(existing_specs + new_specs)
        return combined_specs, tools_code, True

    @staticmethod
    def _extract_mcp_tools(raw: str) -> list[ToolSpec]:
        """
        Extract MCP tool decorators and generate ToolSpec objects from the provided Python code string.

        Args:
        - raw: The Python code string containing function definitions with @mcp.tool decorators.

        Returns:
        - A list of ToolSpec objects.
        """
        text = (raw or "").strip()
        if not text:
            return []

        # Extract code blocks wrapped in markdown
        code_blocks = BaseAgent._extract_code_blocks(text)
        mcp_tools: list[ToolSpec] = []
        if code_blocks:
            for code in code_blocks:
                mcp_tools.extend(ToolSynthesisMixin._extract_mcp_tools_from_python(code))
        return mcp_tools
        # Fallback: attempt to parse raw output as code.
        cleaned = ToolSynthesisMixin._sanitize_llm_code(text)
        if cleaned:
            return ToolSynthesisMixin._extract_mcp_tools_from_python(cleaned)
        return []

    @staticmethod
    def _extract_mcp_tools_from_python(raw: str) -> list[ToolSpec]:
        text = (raw or "").strip()
        if not text:
            return []
        normalized = re.sub(r"@mcp\.tool\s*(?=\n)", "@mcp.tool()", text)
        mcp_tools: list[ToolSpec] = []
        try:
            module = ast.parse(normalized)
            lines = normalized.splitlines()
            for node in module.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not any(
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "tool"
                    for dec in node.decorator_list
                ):
                    continue
                if node.lineno is None or node.end_lineno is None:
                    continue
                decorator_lines = [
                    dec.lineno for dec in node.decorator_list if getattr(dec, "lineno", None)
                ]
                start_line = min(decorator_lines) if decorator_lines else node.lineno
                function_string = "\n".join(lines[start_line - 1 : node.end_lineno]).strip()
                try:
                    tool = ToolSpec.from_function_string(function_string)
                    mcp_tools.append(tool)
                except ValueError as e:
                    logger.error(
                        f"Error creating ToolSpec for function {node.name}: {e}"
                    )
            if mcp_tools:
                return mcp_tools
        except Exception:
            logger.debug("AST tool extraction failed; falling back to regex.", exc_info=True)

        # Regex fallback for malformed code blocks.
        mcp_pattern = r"(@mcp\.tool(?:\((.*?)\))?\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*\s*\([^\)]*\))\s*(->\s*[^:]+)?\s*:([\s\S]+?))(?=\n\s*@mcp|\Z)"
        matches = re.findall(mcp_pattern, normalized, re.DOTALL)
        for match in matches:
            function_signature = match[2]
            function_string = match[0]
            try:
                tool = ToolSpec.from_function_string(function_string)
                mcp_tools.append(tool)
            except ValueError as e:
                logger.error(
                    f"Error creating ToolSpec for function {function_signature}: {e}"
                )
        return mcp_tools
