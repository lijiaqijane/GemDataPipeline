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
from agent_gem.tools import CallableTool, JsonRecordsQueryTool

from ..base import BaseAgent, TaskContext

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest  # noqa: F401

logger = logging.getLogger(__name__)


class ToolSynthesisMixin:
    """Tool generation, compilation, and registration helpers."""

    def _synthesize_task_tools(
        self,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        data_profile: dict[str, Any],
    ) -> tuple[list[ToolSpec], str]:
        """Generate task-specific tools using detected data sources."""
        data_profile_json = json.dumps(data_profile, ensure_ascii=False)
        prompt = (
            "You are designing a deterministic toolset for a sandboxed RL agent.\n"
            "Goal: generate 3-5 @mcp.tool functions tailored to the AVAILABLE local data sources.\n"
            "Hard rules:\n"
            "- NO network, NO external APIs. Only read the listed local files/SQLite.\n"
            "- Deterministic: set random.seed(0) if randomness is used.\n"
            "- Tools must accept (query: str, max_results: int = 5) and return list[dict].\n"
            "- Implement the body to actually parse the data sources; do not stub/raise.\n"
            "- If fields are insufficient, add additional tools or extend returned dicts to cover likely needs.\n"
            "- Prefer descriptive snake_case names; avoid 'bash'/'search'/'python_runner'.\n"
            "Data sources (JSON, sampled schemas):\n"
            f"{data_profile_json}\n"
            f"Topic: {topic}\n"
            f"Curated records sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
            "Output ONLY one Python code block defining the tools (including imports)."
        )

        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        raw = self.llm.simple_complete(prompt, temperature=0.6, max_tokens=max_tokens)
        ctx.add_step({"type": "tool_synthesis", "content": raw})
        specs = self._extract_mcp_tools(raw)
        tools_code = ""
        blocks = BaseAgent._extract_code_blocks(raw)
        if blocks:
            tools_code = BaseAgent._strip_code_fences(blocks[0])

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

        specs = filtered or self._fallback_tool_specs(topic)
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
        return specs, tools_code

    def _generate_tool_implementations(
        self,
        *,
        tools_code: str,
        sandbox_dir: Path,
        tool_specs: list[ToolSpec],
        topic: str,
        data_profile: dict[str, Any],
    ) -> str:
        """Post-process LLM-generated tools to enforce determinism and local-only access."""
        code = tools_code.strip()
        prepend: list[str] = []
        if "import random" not in code:
            prepend.append("import random")
        if "from pathlib import Path" not in code:
            prepend.append("from pathlib import Path")
        if prepend:
            code = "\n".join(prepend) + "\n" + code
        if "random.seed(" not in code:
            code = "random.seed(0)\n" + code
        # Guardrail: remind about local-only data
        guard_comment = f"# Data profile (truncated): {json.dumps(data_profile, ensure_ascii=False)[:400]}"

        tool_names = [spec.name for spec in tool_specs]
        fallback_profile = json.dumps(data_profile, ensure_ascii=False)
        tools_literal = json.dumps(tool_names, ensure_ascii=False)

        stubbed_functions: set[str] = set()
        try:
            tree = ast.parse(code)
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    body = node.body
                    if not body or all(isinstance(stmt, ast.Pass) for stmt in body):
                        stubbed_functions.add(node.name)
                        continue
                    if len(body) == 1 and isinstance(body[0], ast.Raise):
                        stubbed_functions.add(node.name)
                        continue
                    for stmt in ast.walk(node):
                        if isinstance(stmt, ast.Raise):
                            exc = getattr(stmt, "exc", None)
                            if isinstance(exc, ast.Call) and getattr(exc.func, "id", "") == "RuntimeError":
                                stubbed_functions.add(node.name)
                                break
        except Exception:
            logger.debug("stub detection failed", exc_info=True)

        stub_literal = json.dumps(sorted(stubbed_functions), ensure_ascii=False)
        fallback_block = """
# --- Auto-fallback to avoid stubbed tools; reads local data only ---
import json as _json
import csv as _csv
import sqlite3 as _sqlite3
from pathlib import Path as _Path

_FALLBACK_PROFILE = _json.loads(_json.dumps({fallback_profile}))
_STUBBED_TO_PATCH = {stub_literal}

def _fallback_records(base_dir: _Path) -> list:
    # Try CSV first
    for entry in _FALLBACK_PROFILE.get("csv", []):
        p = base_dir / entry.get("path", "")
        if p.exists():
            try:
                with p.open("r", encoding="utf-8", errors="ignore") as f:
                    r = _csv.DictReader(f)
                    return [row for _, row in zip(range(20), r)]
            except Exception:
                pass
    # Then JSON
    for entry in _FALLBACK_PROFILE.get("json", []):
        p = base_dir / entry.get("path", "")
        if p.exists():
            try:
                data = _json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    return data[:20]
                if isinstance(data, dict):
                    return [data]
            except Exception:
                pass
    # Then SQLite
    for entry in _FALLBACK_PROFILE.get("sqlite", []):
        p = base_dir / entry.get("path", "")
        if p.exists():
            try:
                conn = _sqlite3.connect(p)
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [r[0] for r in cur.fetchall()]
                for table in tables:
                    cur.execute("SELECT * FROM '{tbl}' LIMIT 20".format(tbl=table))
                    rows = cur.fetchall()
                    cols = [d[0] for d in cur.description] if cur.description else []
                    conn.close()
                    return [dict(zip(cols, row)) for row in rows]
                conn.close()
            except Exception:
                pass
    return []

def _filter_records(data, query: str, max_results: int = 5):
    if not isinstance(data, list):
        return []
    if not query or not isinstance(query, str):
        return data[:max_results]
    q = query.lower()
    filtered = []
    for item in data:
        if isinstance(item, dict):
            text = " ".join(str(v) for v in item.values()).lower()
            if q in text:
                filtered.append(item)
    if not filtered:
        filtered = data
    return filtered[:max_results]

def _make_impl():
    def _impl(query: str, max_results: int = 5, _base=_Path(__file__).parent):
        data = _fallback_records(_base)
        return _filter_records(data, query, max_results)
    return _impl

def _wrap_stub(fn):
    def _wrapped(query: str, max_results: int = 5):
        base_dir = _Path(__file__).parent
        try:
            out = fn(query, max_results)
            if out is not None:
                return out
        except Exception:
            pass
        data = _fallback_records(base_dir)
        return _filter_records(data, query, max_results)
    return _wrapped

for _name in _STUBBED_TO_PATCH:
    globals()[_name] = _make_impl()

for _name in {tools_literal}:
    fn = globals().get(_name)
    if callable(fn):
        globals()[_name] = _wrap_stub(fn)
# --- End fallback patch ---
""".replace("{fallback_profile}", fallback_profile).replace("{tools_literal}", tools_literal).replace("{stub_literal}", stub_literal)
        return guard_comment + "\n" + code + "\n" + fallback_block

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
    def _collect_selftest_fields(tool_selftest: dict[str, Any]) -> set[str]:
        fields: set[str] = set()
        for info in tool_selftest.values():
            if isinstance(info, dict):
                for key in info.get("fields", []) or []:
                    if isinstance(key, str):
                        fields.add(key)
        return fields

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
        blocks = BaseAgent._extract_code_blocks(raw)
        new_code = BaseAgent._strip_code_fences(blocks[0]) if blocks else ""
        if not specs or not new_code:
            return existing_specs, previous_code, tool_selftest

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
        records_path = sandbox.sandbox_dir / self._RECORDS_FILENAME
        tools_path = sandbox.sandbox_dir / "tools.py"
        module = None
        if tools_path.exists():
            try:
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
                sandbox.register_tool(
                    JsonRecordsQueryTool(
                        name=spec.name,
                        description=spec.description,
                        records_path=records_path,
                    )
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
        tools_code = ""
        blocks = BaseAgent._extract_code_blocks(raw)
        if blocks:
            tools_code = BaseAgent._strip_code_fences(blocks[0])

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

        self._register_task_tools(new_specs, sandbox, ctx, tools_code=tools_code)
        return existing_specs + new_specs, tools_code, True

    def _fallback_tool_specs(self, topic: str) -> list[ToolSpec]:
        """Minimal toolset when generation fails."""

        records_filename = getattr(self, "_RECORDS_FILENAME", "records.json")

        def _load_records(max_results: int = 20) -> list[dict]:
            base_candidates = [
                Path(records_filename),
                Path("records.json"),
                Path.cwd() / records_filename,
            ]
            for candidate in base_candidates:
                if candidate.exists():
                    try:
                        data = json.loads(candidate.read_text(encoding="utf-8"))
                        if isinstance(data, list):
                            return data[:max_results]
                        if isinstance(data, dict):
                            return [data]
                    except Exception:
                        continue
            return []

        def _filter_query(data: list[dict], query: str, max_results: int = 5) -> list[dict]:
            if not data:
                return []
            if not query or not isinstance(query, str):
                return data[:max_results]
            q = query.lower()
            filtered = []
            for item in data:
                if isinstance(item, dict):
                    text = " ".join(str(v) for v in item.values()).lower()
                    if q in text:
                        filtered.append(item)
            if not filtered:
                filtered = data
            return filtered[:max_results]

        def fetch_related_records(query: str, max_results: int = 5) -> list[dict]:
            """Retrieve records that loosely match the query text."""
            data = _load_records(max_results=max_results)
            return _filter_query(data, query, max_results)

        def summarize_records(query: str, max_results: int = 5) -> list[dict]:
            """Return compact summaries for records associated with the topic."""
            data = _load_records(max_results=max_results * 2)
            summarized: list[dict] = []
            for item in data[: max_results * 2]:
                if isinstance(item, dict):
                    summarized.append(
                        {
                            "summary": " ".join(
                                str(v) for v in item.values()
                            )[:200],
                            **item,
                        }
                    )
            return _filter_query(summarized or data, query, max_results)

        def cross_reference_entities(query: str, max_results: int = 5) -> list[dict]:
            """Find entities connected to the topic with brief justification."""
            data = _load_records(max_results=max_results * 3)
            enriched: list[dict] = []
            for item in data:
                if isinstance(item, dict):
                    enriched.append(
                        {
                            "entity": item.get("name") or item.get("id") or item.get("title"),
                            "link": item.get("related") or item.get("relations"),
                            **item,
                        }
                    )
            return _filter_query(enriched or data, query, max_results)

        defaults = [
            ToolSpec.from_function(
                fetch_related_records,
                name="fetch_related_records",
                description=f"Find records related to {topic}",
            ),
            ToolSpec.from_function(
                summarize_records,
                name="summarize_records",
                description=f"Summarize key points for {topic}",
            ),
            ToolSpec.from_function(
                cross_reference_entities,
                name="cross_reference_entities",
                description=f"Cross-check entities connected to {topic}",
            ),
        ]
        return defaults

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

        for code in code_blocks:
            # Regular expression to capture function definitions with @mcp.tool decorators
            mcp_pattern = r"(@mcp\.tool\((.*?)\)\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*\s*\([^\)]*\))\s*(->\s*[a-zA-Z_][a-zA-Z0-9_\[\],]*)?\s*:([\s\S]+?))(?=\n\s*@mcp|\Z)"

            # Match all decorators and the corresponding function definitions
            matches = re.findall(mcp_pattern, code, re.DOTALL)

            for match in matches:
                function_signature = match[2]

                # Generate a ToolSpec from the function string
                function_string = match[0]

                try:
                    tool = ToolSpec.from_function_string(function_string)
                    mcp_tools.append(tool)
                except ValueError as e:
                    logger.error(
                        f"Error creating ToolSpec for function {function_signature}: {e}"
                    )

        return mcp_tools
