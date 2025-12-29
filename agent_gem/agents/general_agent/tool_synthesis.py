from __future__ import annotations

import ast
import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import Any

from agent_gem.core.task_schema import ToolSpec
from agent_gem.sandbox import SandboxExecutor
from agent_gem.tools import CallableTool

from ..base import BaseAgent, TaskContext

logger = logging.getLogger(__name__)


class ToolSynthesisMixin:
    """Tool generation, compilation, and registration helpers."""
    _SUBMIT_RESULT_TOOL = "submit_result"
    _SUBMITTED_RESULT_FILE = "submitted_result.json"
    _MAX_REGEN_ATTEMPTS = 3
    _MAX_DATA_PROFILE_SIZE = 12000
    _FILTERED_TOOL_NAMES = {"bash", "search", "python_runner"}

    @staticmethod
    def _tool_code_uses_mcp_tool(code: str) -> bool:
        """Detect whether code uses @mcp.tool decorators (which require a FastMCP instance)."""
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
        """Detect a module-scope binding for a FastMCP instance named `mcp`."""
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        def _assigns_mcp(nodes: list[ast.stmt]) -> bool:
            for stmt in nodes:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "mcp":
                            return True
                if isinstance(stmt, ast.AnnAssign):
                    if isinstance(stmt.target, ast.Name) and stmt.target.id == "mcp":
                        return True
            return False

        if _assigns_mcp(tree.body):
            return True

        for node in tree.body:
            if isinstance(node, ast.Try):
                # allow patterns like:
                # try: mcp = FastMCP(...)
                # except: mcp = ...
                if _assigns_mcp(node.body):
                    return True
                for handler in node.handlers:
                    if _assigns_mcp(handler.body):
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
            reasons.append("missing_mcp_instance")
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

    def _validate_submit_result_tool(
        self, tools_code: str, specs: list[ToolSpec]
    ) -> tuple[bool, bool, bool]:
        """Validate submit_result tool.
        
        Returns:
            (has_submit, not_stubbed, persists) tuple
        """
        has_submit = any(spec.name == self._SUBMIT_RESULT_TOOL for spec in specs)
        submit_stubbed = False
        submit_persists = bool(tools_code and self._SUBMITTED_RESULT_FILE in tools_code)
        
        if tools_code and "def submit_result" in tools_code:
            try:
                tree = ast.parse(tools_code)
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == self._SUBMIT_RESULT_TOOL:
                        body = node.body
                        if (not body) or all(isinstance(stmt, ast.Pass) for stmt in body):
                            submit_stubbed = True
                        elif len(body) == 1 and isinstance(body[0], ast.Raise):
                            submit_stubbed = True
                        else:
                            # Check for raise statements in function body
                            for stmt in ast.walk(node):
                                if isinstance(stmt, ast.Raise):
                                    submit_stubbed = True
                                    break
                        break
            except Exception:
                # If parsing fails, treat as invalid
                submit_stubbed = True
        
        return has_submit, not submit_stubbed, submit_persists

    def _ensure_mcp_installed(self, sandbox: SandboxExecutor) -> tuple[bool, str]:
        """Ensure mcp package (FastMCP) is available in the sandbox."""
        # Check if FastMCP is importable
        import_check = sandbox.execute_bash(
            'python -c "from mcp.server.fastmcp import FastMCP; print(\'MCP_IMPORT_OK\')" 2>&1'
        )
        import_output = (import_check.get("stdout", "") or "") + (import_check.get("stderr", "") or "")
        import_success = import_check.get("returncode") == 0 or "MCP_IMPORT_OK" in import_output
        
        if not import_success:
            # mcp is not available, install it via pip
            install_result = sandbox.execute_bash("pip install mcp 2>&1")
            # Verify installation
            verify_import = sandbox.execute_bash(
                'python -c "from mcp.server.fastmcp import FastMCP; print(\'MCP_IMPORT_OK\')" 2>&1'
            )
            verify_output = (verify_import.get("stdout", "") or "") + (verify_import.get("stderr", "") or "")
            verify_success = verify_import.get("returncode") == 0 or "MCP_IMPORT_OK" in verify_output
            
            if not verify_success:
                error_msg = verify_import.get("stderr", "") or verify_import.get("stdout", "")
                if not error_msg:
                    error_msg = install_result.get("stderr", "") or install_result.get("stdout", "") or "Unknown installation error"
                return False, f"Failed to install or import mcp: {error_msg[:500]}"

        return True, ""


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

    def _validate_and_register_tools(
        self,
        raw: str,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        data_profile: dict[str, Any],
        attempt: int,
    ) -> tuple[bool, list[ToolSpec], str, dict[str, Any], list[str]]:
        """Validate generated tools and perform all checks (code, import, registration, selftest).
        
        Returns:
            (success, specs, tools_code, tool_selftest, errors) tuple
            If success is False, errors contains all validation errors
        """
        errors: list[str] = []
        specs = self._extract_mcp_tools(raw)
        tools_code = self._sanitize_llm_code(raw)
        
        # Check if we successfully parsed any tools
        if not specs:
            errors.append("no_tools_parsed_from_llm_output")
            ctx.add_step(
                {
                    "type": "tool_synthesis_no_tools_parsed",
                    "attempt": attempt,
                    "raw_preview": raw[:500],
                }
            )
            return False, [], "", {}, errors
        
        # Validate code
        code_ok, code_reasons = self._validate_tool_code(tools_code) if tools_code else (False, ["no_tools_code"])
        if not code_ok:
            errors.extend(code_reasons)
        
        # Validate submit_result
        has_submit, not_stubbed, submit_persists = self._validate_submit_result_tool(tools_code, specs)
        if not has_submit:
            errors.append("missing_submit_result")
        if not not_stubbed:
            errors.append("submit_result_stubbed")
        if not submit_persists:
            errors.append("submit_result_not_persisting")
        
        # If basic validation failed, return early
        if errors:
            ctx.add_step(
                {
                    "type": "tool_synthesis_validation_failed",
                    "attempt": attempt,
                    "errors": errors,
                    "has_submit_result": bool(has_submit),
                    "submit_stubbed": not not_stubbed,
                    "submit_persists": bool(submit_persists),
                    "tool_code_valid": bool(code_ok),
                }
            )
            return False, specs, tools_code, {}, errors
        
        # Ensure mcp is installed
        mcp_installed, mcp_error = self._ensure_mcp_installed(sandbox)
        if not mcp_installed:
            error_msg = f"mcp_installation_failed: {mcp_error}" if mcp_error else "mcp_installation_failed"
            errors.append(error_msg)
            ctx.add_step(
                {
                    "type": "mcp_installation_failed",
                    "attempt": attempt,
                    "error": mcp_error,
                }
            )
            return False, specs, tools_code, {}, errors
        
        # Filter and prepare specs
        seen: set[str] = set()
        filtered: list[ToolSpec] = []
        for spec in specs:
            if spec.name in self._FILTERED_TOOL_NAMES:
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
        filtered = self._ensure_submit_result_tool(filtered)
        
        # Write tools.py and test import
        tools_path = sandbox.sandbox_dir / "tools.py"
        try:
            implemented_code = self._generate_tool_implementations(
                tools_code=tools_code,
                sandbox_dir=sandbox.sandbox_dir,
                tool_specs=filtered,
                topic=topic,
                data_profile=data_profile,
                sandbox=sandbox,
            )
            tools_path.write_text(implemented_code, encoding="utf-8")
            
            # Test import
            import_result = sandbox.execute_bash('python -c "import tools; print(\'TOOLS_IMPORT_OK\')"')
            stderr = (import_result.get("stderr") or "").strip()
            stdout = (import_result.get("stdout") or "").strip()
            import_ok = import_result.get("returncode") == 0 or "TOOLS_IMPORT_OK" in stdout
            if not import_ok:
                detail = (
                    f"stdout={stdout[:1000]!r} stderr={stderr[:1000]!r}"
                    if (stdout or stderr)
                    else "stdout/stderr empty"
                )
                errors.append(f"import_failed: {detail}")
                ctx.add_step(
                    {
                        "type": "tool_import_test_failed",
                        "returncode": import_result.get("returncode"),
                        "stderr": stderr[:2000],
                        "stdout": stdout[:2000],
                        "attempt": attempt,
                    }
                )
                return False, filtered, tools_code, {}, errors
        except Exception as exc:
            errors.append(f"import_test_exception: {str(exc)[:200]}")
            ctx.add_step(
                {
                    "type": "tool_import_test_exception",
                    "error": str(exc)[:500],
                    "attempt": attempt,
                }
            )
            return False, filtered, tools_code, {}, errors
        
        # Register tools
        registration_ok, registration_errors = self._register_task_tools(filtered, sandbox, ctx, tools_code=tools_code)
        if not registration_ok:
            errors.extend(registration_errors)
            ctx.add_step(
                {
                    "type": "tool_registration_failed",
                    "errors": registration_errors,
                    "attempt": attempt,
                }
            )
            return False, filtered, tools_code, {}, errors
        
        # Self-test
        tool_selftest = self._self_test_tools(filtered, sandbox, topic, ctx, data_profile)
        
        # Check if selftest indicates regeneration is needed
        regen_needed, regen_reasons = self._needs_tool_regeneration(tool_selftest)
        if regen_needed:
            errors.extend([f"Self-test issue: {reason}" for reason in regen_reasons])
            ctx.add_step(
                {
                    "type": "tool_selftest_failed",
                    "reasons": regen_reasons,
                    "attempt": attempt,
                }
            )
            return False, filtered, tools_code, tool_selftest, errors
        
        # All validations passed
        return True, filtered, tools_code, tool_selftest, []

    def _build_base_tool_rules(self, include_examples: bool = True) -> str:
        """Build the base rules for tool generation that should be included in all prompts.
        
        Args:
            include_examples: If True, include example code snippets for file paths.
        """
        rules = (
            "- IMPORTANT: decorators execute at import-time. Use @mcp.tool() on a FastMCP instance.\n"
            "  Add `from mcp.server.fastmcp import FastMCP` and `mcp = FastMCP(\"Tools\")` near the top.\n"
            "  Do NOT call mcp.run() or start any server.\n"
            "  Do NOT implement tool logic in the framework; all tool bodies must be yours.\n"
        )
        
        if include_examples:
            rules += (
                "- CRITICAL: File paths must be absolute. Use `Path(__file__).parent` to get the directory containing this tools.py file.\n"
                "  Example: `from pathlib import Path; BASE_DIR = Path(__file__).parent; csv_path = BASE_DIR / \"data\" / \"file.csv\"`\n"
                "  NEVER use relative paths like `\"data/file.csv\"` - they will fail if the working directory changes.\n"
            )
        else:
            rules += (
                "- CRITICAL: File paths must be absolute. Use `Path(__file__).parent` to get the directory containing this tools.py file.\n"
            )
        
        rules += (
            "- NO network, NO external APIs. Only read the listed local files (CSV/JSON/TXT/SQLite).\n"
            "- Deterministic: set random.seed(0) if randomness is used.\n"
            "- RECOMMENDED: Tools should accept (query: str, max_results: int = 5) and return list[dict] for consistency.\n"
            "  However, you may use different signatures if the tool's purpose requires it (e.g., submit_result).\n"
            "- Implement the body to actually parse the data sources; do not stub/raise.\n"
            "- Read from the local files (CSV/JSON/TXT/SQLite) shown in the data profile; do NOT hardcode sample records.\n"
            f"- You MUST define a tool named submit_result with signature submit_result(result) that returns `result`.\n"
            f"  submit_result MUST be implemented (no pass/raise) and MUST persist the payload to {self._SUBMITTED_RESULT_FILE}\n"
            f"  in the directory containing tools.py (use `Path(__file__).parent / \"{self._SUBMITTED_RESULT_FILE}\"`).\n"
            "- If fields are insufficient, add additional tools or extend returned dicts to cover likely needs.\n"
            f"- Prefer descriptive snake_case names; avoid {', '.join(repr(name) for name in self._FILTERED_TOOL_NAMES)}.\n"
        )
        return rules

    def _generate_tool_code(
        self,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        data_profile: dict[str, Any],
        data_profile_json: str,
        all_errors: list[str] | None = None,
    ) -> str:
        """Generate tool code using LLM. Returns raw LLM output."""
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        
        if all_errors:
            # Regeneration prompt with error context - use full base rules
            error_context = (
                "Previous attempts encountered the following issues:\n"
                + "\n".join(f"- {error}" for error in all_errors)
                + "\n\n"
            )
            base_rules = self._build_base_tool_rules(include_examples=False)
            prompt = (
                f"{error_context}"
                "Regenerate an executable toolset to fix the issues above.\n"
                "Hard rules:\n"
                f"{base_rules}"
                f"Topic: {topic}\n"
                f"Data profile: {data_profile_json}\n"
                "Output ONLY one Python code block defining the tools (including imports).\n"
                "Do NOT include prose. Ensure each function has the @mcp.tool() decorator."
            )
            raw = self.llm.simple_complete(prompt, temperature=0.55, max_tokens=max_tokens)
            ctx.add_step({"type": "tool_regeneration", "content": raw})
        else:
            # Initial generation prompt
            base_rules = self._build_base_tool_rules(include_examples=True)
            prompt = (
                "You are designing a deterministic toolset for a sandboxed RL agent.\n"
                "Goal: generate 3-5 @mcp.tool() functions tailored to the AVAILABLE local data sources.\n"
                "Hard rules:\n"
                f"{base_rules}"
                "Data sources (JSON, sampled schemas):\n"
                f"{data_profile_json}\n"
                f"Topic: {topic}\n"
                f"Curated records sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
                "Output ONLY one Python code block defining the tools (including imports).\n"
                "Do NOT include prose. Ensure each function has the @mcp.tool() decorator."
            )
            raw = self.llm.simple_complete(prompt, temperature=0.6, max_tokens=max_tokens)
            ctx.add_step({"type": "tool_synthesis", "content": raw, "attempt": 1})
        
        return raw

    def _synthesize_task_tools(
        self,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        data_profile: dict[str, Any],
    ) -> tuple[list[ToolSpec], str, dict[str, Any]]:
        """Generate task-specific tools using detected data sources.
        
        Returns:
            (tool_specs, tools_code, tool_selftest) tuple
        """
        data_profile_json = json.dumps(self._compact_data_profile(data_profile), ensure_ascii=False)
        if len(data_profile_json) > self._MAX_DATA_PROFILE_SIZE:
            data_profile_json = data_profile_json[:self._MAX_DATA_PROFILE_SIZE] + "..."
        
        # Main generation and validation loop
        all_errors: list[str] | None = None
        for attempt in range(1, self._MAX_REGEN_ATTEMPTS + 1):
            if attempt > 1:
                logger.warning("Tool synthesis retrying (attempt %s/%s).", attempt, self._MAX_REGEN_ATTEMPTS)
            
            # Generate code
            raw = self._generate_tool_code(topic, records, ctx, data_profile, data_profile_json, all_errors)
            
            # Validate and register (does all validation steps)
            success, specs, tools_code, tool_selftest, errors = self._validate_and_register_tools(
                raw, topic, records, ctx, sandbox, data_profile, attempt
            )
            
            if success:
                # All validations passed
                ctx.add_step(
                    {
                        "type": "tool_synthesis",
                        "tool_count": len(specs),
                        "tools": [spec.model_dump() for spec in specs],
                    }
                )
                return specs, tools_code, tool_selftest
            
            # Validation failed, collect errors for next attempt
            all_errors = errors
        
        # All attempts failed
        raise RuntimeError(
            f"Tool synthesis failed after {self._MAX_REGEN_ATTEMPTS} attempts. "
            f"Last errors: {', '.join(all_errors) if all_errors else 'unknown'}"
        )

    def _ensure_tools_meet_format_requirements(
        self,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        data_profile: dict[str, Any],
        tool_specs: list[ToolSpec],
        tools_code: str,
        tool_selftest: dict[str, Any],
        expected_fields: set[str],
    ) -> tuple[list[ToolSpec], str, dict[str, Any]]:
        """Ensure tools meet format requirements, regenerating if needed.
        
        Returns:
            (tool_specs, tools_code, tool_selftest) tuple, potentially regenerated
        """
        for regen_attempt in range(self._MAX_REGEN_ATTEMPTS):
            regen_needed, regen_reasons = self._needs_tool_regeneration(
                tool_selftest, required_fields=expected_fields
            )
            
            if not regen_needed:
                break
            
            logger.info(
                "Tool regeneration for format requirements (attempt %s/%s): %s, fields: %s",
                regen_attempt + 1,
                self._MAX_REGEN_ATTEMPTS,
                regen_reasons,
                sorted(expected_fields),
            )
            # Convert regen_reasons to error format
            all_errors = [f"Format requirement issue: {reason}" for reason in regen_reasons] if regen_reasons else None
            if expected_fields:
                all_errors = (all_errors or []) + [f"Missing required fields: {sorted(expected_fields)}"]
            tool_specs, tools_code, tool_selftest = self._regenerate_tools_with_selftest(
                topic,
                records,
                ctx,
                sandbox,
                data_profile,
                tool_selftest,
                tool_specs,
                tools_code,
                required_fields=expected_fields,
                all_errors=all_errors,
            )
            ctx.add_step(
                {
                    "type": "tool_regeneration_for_format",
                    "reasons": regen_reasons,
                    "attempt": regen_attempt + 1,
                    "expected_fields": sorted(expected_fields),
                }
            )
        
        return tool_specs, tools_code, tool_selftest

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
        sandbox: SandboxExecutor,
    ) -> str:
        """Return agent-authored tools code with minimal formatting cleanup.

        User requirement: do NOT inject/override any tool implementations here.
        We only validate and fail-fast so the agent can regenerate.
        """
        code = (tools_code or "").strip()
        if not code:
            raise RuntimeError("tools.py generation failed: empty tools code")

        # Fix file paths: convert relative paths to absolute paths based on __file__
        code = self._fix_file_paths(code)
        
        # Validate: tool code must be data-driven (read local files) and avoid embedding datasets.
        code_ok, reasons = self._validate_tool_code(code)
        if not code_ok:
            raise RuntimeError(f"tools.py invalid (not data-driven): {reasons}")

        # Validate: submit_result must be implemented by the agent and persist the payload.
        if "def submit_result" not in code:
            raise RuntimeError("tools.py missing required submit_result implementation (agent-authored).")
        if self._SUBMITTED_RESULT_FILE not in code:
            raise RuntimeError(f"tools.py submit_result must persist to {self._SUBMITTED_RESULT_FILE} (agent-authored).")

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
    
    @staticmethod
    def _fix_file_paths(code: str) -> str:
        """Convert relative file paths to absolute paths based on __file__.
        
        This ensures tools can find data files regardless of the current working directory.
        Uses regex-based replacement for reliability.
        """
        import re
        
        lines = code.splitlines()
        
        # Step 1: Add Path import if missing
        has_pathlib = any(
            "from pathlib import Path" in line or 
            ("import pathlib" in line and "Path" in line)
            for line in lines
        )
        
        if not has_pathlib:
            # Find insertion point (after other imports)
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    insert_idx = i + 1
                elif stripped and not stripped.startswith("#") and insert_idx > 0:
                    break
            lines.insert(insert_idx, "from pathlib import Path")
        
        # Step 2: Add BASE_DIR definition if missing
        has_base_dir = any("BASE_DIR" in line and "=" in line for line in lines)
        
        if not has_base_dir:
            # Find insertion point (after imports, before first function/class)
            insert_idx = 0
            for i, line in enumerate(lines):
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    insert_idx = i + 1
                elif stripped and not stripped.startswith("#") and insert_idx > 0:
                    # Check if this is a function/class definition
                    if not (stripped.startswith(("def ", "class ", "@"))):
                        insert_idx = i
                    break
            lines.insert(insert_idx, "BASE_DIR = Path(__file__).parent")
        
        code = "\n".join(lines)
        
        # Step 3: Replace relative file paths with BASE_DIR / path
        # Match patterns like: "data/file.csv", 'data/file.csv', "records.json"
        # But avoid replacing if already using BASE_DIR or absolute paths
        file_ext_pattern = r'\.(csv|json|sqlite|db|txt|parquet)'
        
        def replace_file_path(match):
            full_match = match.group(0)
            quote_char = match.group(1)
            path = match.group(2)
            
            # Skip if already using BASE_DIR
            if "BASE_DIR" in code[max(0, match.start()-50):match.end()+50]:
                return full_match
            
            # Skip absolute paths
            if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
                return full_match
            
            # Skip if it's part of a larger expression that already uses BASE_DIR
            context_start = max(0, match.start() - 100)
            context = code[context_start:match.end()+50]
            if "BASE_DIR" in context:
                return full_match
            
            # Replace the path
            if "/" in path or "\\" in path:
                # Multi-part path: BASE_DIR / "part1" / "part2" / "file.ext"
                parts = [p for p in path.replace("\\", "/").split("/") if p]
                path_expr = " / ".join([f'{quote_char}{part}{quote_char}' for part in parts])
                return f'BASE_DIR / {path_expr}'
            else:
                # Single filename
                return f'BASE_DIR / {quote_char}{path}{quote_char}'
        
        # Pattern: matches quoted strings that look like file paths
        # Matches: "data/file.csv", 'data/file.csv', "records.json"
        # Excludes: already absolute paths, paths in BASE_DIR expressions
        pattern = rf'(["\'])((?:data/|\./)?[^"\']+{file_ext_pattern})(["\'])'
        
        # Replace in code
        code = re.sub(pattern, replace_file_path, code)
        
        # Also handle os.path.exists("path") patterns
        def replace_os_path(match):
            prefix = match.group(1)  # "os.path.exists(" or similar
            quote = match.group(2)
            path = match.group(3)
            suffix = match.group(4)  # closing quote and paren
            
            # Skip if already absolute or using BASE_DIR
            if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
                return match.group(0)
            
            # Replace
            if "/" in path or "\\" in path:
                parts = [p for p in path.replace("\\", "/").split("/") if p]
                path_expr = " / ".join([f'{quote}{p}{quote}' for p in parts])
                return f'{prefix}BASE_DIR / {path_expr}{suffix}'
            else:
                return f'{prefix}BASE_DIR / {quote}{path}{quote}{suffix}'
        
        os_path_pattern = rf'(os\.path\.(?:exists|join|isfile|isdir)\()(["\'])((?:data/|\./)?[^"\']+{file_ext_pattern})(["\']\))'
        code = re.sub(os_path_pattern, replace_os_path, code)
        
        return code

    @staticmethod
    def _convert_paths_to_strings(obj: Any) -> Any:
        """Recursively convert PosixPath objects to strings for JSON serialization."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: ToolSynthesisMixin._convert_paths_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ToolSynthesisMixin._convert_paths_to_strings(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(ToolSynthesisMixin._convert_paths_to_strings(item) for item in obj)
        else:
            return obj

    def _self_test_tools(
        self,
        tool_specs: list[ToolSpec],
        sandbox: SandboxExecutor,
        topic: str,
        ctx: TaskContext,
        data_profile: dict[str, Any],
    ) -> dict[str, Any]:
        """Lightweight tool smoke-test: ensure tool functions exist and are non-empty."""
        tools_path = sandbox.sandbox_dir / "tools.py"
        profile: dict[str, Any] = {}
        if not tools_path.exists():
            ctx.add_step({"type": "tool_self_test", "content": "tools.py missing"})
            return profile
        try:
            source = tools_path.read_text(encoding="utf-8")
        except Exception as exc:
            ctx.add_step({"type": "tool_self_test", "content": f"read_failed: {exc}"})
            return profile
        if not source.strip():
            ctx.add_step({"type": "tool_self_test", "content": "tools.py empty"})
            return profile
        try:
            tree = ast.parse(source)
        except Exception as exc:
            ctx.add_step({"type": "tool_self_test", "content": f"parse_failed: {exc}"})
            return profile

        func_nodes = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def _is_docstring(stmt: ast.stmt) -> bool:
            return (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )

        def _is_empty_stmt(stmt: ast.stmt) -> bool:
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Raise):
                return True
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
                return True
            return False

        for spec in tool_specs:
            if spec.name == self._SUBMIT_RESULT_TOOL:
                continue
            node = func_nodes.get(spec.name)
            if node is None:
                profile[spec.name] = {"ok": False, "error": "missing_function"}
                continue
            body = list(node.body)
            if body and _is_docstring(body[0]):
                body = body[1:]
            meaningful = any(not _is_empty_stmt(stmt) for stmt in body)
            if not body or not meaningful:
                profile[spec.name] = {"ok": False, "error": "empty_body"}
            else:
                profile[spec.name] = {"ok": True}

        profile_serializable = self._convert_paths_to_strings(profile)
        ctx.add_step(
            {
                "type": "tool_self_test",
                "content": json.dumps(profile_serializable, ensure_ascii=False)[:2000],
            }
        )
        return profile

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
                continue
            if info.get("ok") is False or info.get("error"):
                reasons.append(f"{name}:error")
                continue
            fields = set(info.get("fields") or [])
            union_fields |= {f for f in fields if isinstance(f, str)}
        if required_fields and union_fields and not (required_fields & union_fields):
            reasons.append("missing_required_fields")
        return bool(reasons), reasons

    def _generate_regeneration_code(
        self,
        topic: str,
        ctx: TaskContext,
        data_profile: dict[str, Any],
        tool_selftest: dict[str, Any],
        existing_specs: list[ToolSpec],
        previous_code: str = "",
        required_fields: set[str] | None = None,
        all_errors: list[str] | None = None,
    ) -> str:
        """Generate tool code for regeneration with additional context (selftest, existing tools, etc.).
        
        Returns:
            Raw LLM output containing tool code
        """
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        
        # Build error context if errors are provided
        error_context = ""
        if all_errors:
            error_context = (
                "Previous attempts encountered the following issues:\n"
                + "\n".join(f"- {error}" for error in all_errors)
                + "\n\n"
            )
        
        data_profile_json = json.dumps(self._compact_data_profile(data_profile), ensure_ascii=False)
        if len(data_profile_json) > self._MAX_DATA_PROFILE_SIZE:
            data_profile_json = data_profile_json[:self._MAX_DATA_PROFILE_SIZE] + "..."
        
        # Use the same base rules as initial generation, ensuring all requirements are included
        base_rules = self._build_base_tool_rules(include_examples=False)
        
        prompt = (
            f"{error_context}"
            "Regenerate an executable toolset to fix the issues above.\n"
            "Hard rules:\n"
            f"{base_rules}"
            "- Keep existing tool names when possible; you may add up to 2 new tools to expose missing fields.\n"
            f"Topic: {topic}\n"
            f"Data profile: {data_profile_json}\n"
            f"Self-test report: {json.dumps(self._convert_paths_to_strings(tool_selftest), ensure_ascii=False)[:1200]}\n"
            f"Existing tools: {json.dumps([s.model_dump() for s in existing_specs], ensure_ascii=False)[:1200]}\n"
            f"Required fields to expose (if any): {sorted(required_fields) if required_fields else []}\n"
            f"Prior tool code (truncated): {previous_code[:800]}\n"
            "Output ONLY one Python code block defining the tools (including imports).\n"
            "Do NOT include prose. Ensure each function has the @mcp.tool() decorator."
        )
        raw = self.llm.simple_complete(prompt, temperature=0.55, max_tokens=max_tokens)
        ctx.add_step({"type": "tool_regeneration", "content": raw})
        return raw

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
        all_errors: list[str] | None = None,
    ) -> tuple[list[ToolSpec], str, dict[str, Any]]:
        """Regenerate tools with additional context and validate them.
        
        This method generates new tool code based on selftest results, existing tools, and errors,
        then validates and registers the tools using the unified validation flow.
        
        Returns:
            (specs, tools_code, tool_selftest) tuple. If regeneration fails, returns existing values.
        """
        # Generate code with regeneration-specific context
        raw = self._generate_regeneration_code(
            topic, ctx, data_profile, tool_selftest, existing_specs, previous_code, required_fields, all_errors
        )
        
        # Use unified validation flow
        success, specs, tools_code, updated_selftest, errors = self._validate_and_register_tools(
            raw, topic, records, ctx, sandbox, data_profile, attempt=1
        )
        
        if success:
            return specs, tools_code, updated_selftest
        else:
            # Regeneration failed, return existing values
            ctx.add_step(
                {
                    "type": "tool_regeneration_failed",
                    "errors": errors,
                }
            )
            return existing_specs, previous_code, tool_selftest

    def _register_task_tools(
        self,
        tool_specs: list[ToolSpec],
        sandbox: SandboxExecutor,
        ctx: TaskContext,
        *,
        tools_code: str | None = None,
    ) -> tuple[bool, list[str]]:
        """Register tools from tools.py.
        
        Returns:
            (success, error_reasons) tuple. If success is False, error_reasons contains error messages.
        """
        tool_specs = self._ensure_submit_result_tool(list(tool_specs))
        tools_path = sandbox.sandbox_dir / "tools.py"
        module = None
        import_error = None
        if tools_path.exists():
            try:
                # Tools are expected to import FastMCP directly; rely on the real mcp package.
                spec_module = importlib.util.spec_from_file_location("generated_tools", tools_path)
                if spec_module and spec_module.loader:
                    module = importlib.util.module_from_spec(spec_module)
                    spec_module.loader.exec_module(module)
            except Exception as exc:
                import_error = str(exc)
                logger.debug("Failed to import generated tools.py", exc_info=True)

        registration_errors: list[str] = []
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
                error_msg = f"Missing agent-authored tool implementation in tools.py: {spec.name}"
                if import_error:
                    error_msg += f"\nImport error: {import_error}"
                elif not module:
                    error_msg += "\nFailed to import tools.py module (check logs for details)"
                elif not hasattr(module, spec.name):
                    available_names = [name for name in dir(module) if not name.startswith("_") and callable(getattr(module, name, None))]
                    error_msg += f"\nAvailable functions in tools.py: {available_names}"
                registration_errors.append(error_msg)
        
        if registration_errors:
            return False, registration_errors

        ctx.add_step(
            {
                "type": "tool_registration",
                "registered_tools": [spec.name for spec in tool_specs],
                "tools_code_preview": (tools_code or "")[:300],
            }
        )
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)
        return True, []

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
            "Return Python code defining 1-2 NEW tools decorated with @mcp.tool(), avoiding duplicates.\n"
            "Include `from mcp.server.fastmcp import FastMCP` and `mcp = FastMCP(\"Tools\")` in the output.\n"
            "Do NOT call mcp.run() or start any server.\n"
            "Constraints:\n"
            f"- Tool names must be unique, snake_case, and NOT {', '.join(repr(name) for name in self._FILTERED_TOOL_NAMES)}.\n"
            "- RECOMMENDED: Each tool should accept (query: str, max_results: int = 5) and return list[dict] for consistency.\n"
            "  However, you may use different signatures if the tool's purpose requires it.\n"
            "- Implement real logic that reads ONLY local files (CSV/JSON/TXT/SQLite) or the provided records; no stubs, no pass/raise placeholders.\n"
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
            if spec.name in self._FILTERED_TOOL_NAMES:
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
                sandbox=sandbox,
            )
            mode = "a" if tools_path.exists() else "w"
            with tools_path.open(mode, encoding="utf-8") as handle:
                if tools_path.exists():
                    handle.write("\n\n")
                handle.write(implemented_code)
            # Ensure mcp is installed before import test
            mcp_installed, _ = self._ensure_mcp_installed(sandbox)
            if not mcp_installed:
                logger.warning("mcp installation failed during tool augmentation, continuing anyway")
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

    @staticmethod
    def _sanitize_tool_block(function_string: str) -> str:
        lines = function_string.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return function_string
        def_idx = None
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                def_idx = idx
                break
        if def_idx is None:
            return function_string
        decorators = [line for line in lines[:def_idx] if line.lstrip().startswith("@")]
        trimmed = decorators + lines[def_idx:]
        def_line = trimmed[len(decorators)]
        def_indent = len(def_line) - len(def_line.lstrip(" "))
        if def_indent <= 0:
            return "\n".join(trimmed)
        fixed: list[str] = []
        for idx, line in enumerate(trimmed):
            if idx < len(decorators):
                fixed.append(line.lstrip())
                continue
            if not line.strip():
                fixed.append("")
                continue
            fixed.append(line[def_indent:] if len(line) >= def_indent else line.lstrip())
        return "\n".join(fixed)

    @staticmethod
    def _build_tool_spec(function_string: str, name_hint: str) -> ToolSpec | None:
        try:
            return ToolSpec.from_function_string(function_string)
        except (ValueError, SyntaxError, IndentationError) as exc:
            repaired = ToolSynthesisMixin._sanitize_tool_block(function_string)
            if repaired != function_string:
                try:
                    return ToolSpec.from_function_string(repaired)
                except (ValueError, SyntaxError, IndentationError) as repaired_exc:
                    logger.error(
                        "Error creating ToolSpec for function %s after repair: %s",
                        name_hint,
                        repaired_exc,
                    )
                    return None
            logger.error("Error creating ToolSpec for function %s: %s", name_hint, exc)
            return None

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
                tool = ToolSynthesisMixin._build_tool_spec(function_string, node.name)
                if tool is not None:
                    mcp_tools.append(tool)
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
            tool = ToolSynthesisMixin._build_tool_spec(function_string, function_signature)
            if tool is not None:
                mcp_tools.append(tool)
        return mcp_tools
