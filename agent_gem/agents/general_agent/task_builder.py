from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, Optional, TYPE_CHECKING

from agent_gem.core.task_schema import (
    EvaluationCriteria,
    TaskDefinition,
    TaskPackage,
    ToolSpec,
)
from agent_gem.core.validation import CodeValidator

from ..base import BaseAgent, TaskContext

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest  # noqa: F401


class TaskBuilderMixin:
    """Task proposal, refinement, and solution/verification assembly."""
    logger = logging.getLogger(__name__)
    _SUBMIT_RESULT_TOOL = "submit_result"

    def _sanitize_python_code(self, raw: Any) -> str:
        """Extract a single python code string from a raw LLM field (string/markdown)."""
        if raw is None:
            return ""
        text = raw if isinstance(raw, str) else str(raw)
        text = text.strip()
        if not text:
            return ""
        blocks = BaseAgent._extract_code_blocks(text)
        if blocks:
            return "\n\n".join(BaseAgent._strip_code_fences(b) for b in blocks).strip() + "\n"
        # If no fences, drop leading prose until a likely code line.
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("def ", "import ", "from ", "@")):
                start = i
                break
        cleaned = "\n".join(lines[start:]).replace("```", "").strip()
        return cleaned + ("\n" if cleaned and not cleaned.endswith("\n") else "")

    def _generate_agent_solution_and_verification(
        self,
        *,
        request: "GenerationRequest",
        ctx: TaskContext,
        task_content: str,
        submit_result_format: dict[str, Any],
        tool_specs: list[ToolSpec],
        records: list[dict[str, Any]],
        tool_selftest: dict[str, Any] | None = None,
        previous_solution: str | None = None,
        previous_verification: str | None = None,
        repair_error: str | None = None,
        repair_target: str | None = None,
    ) -> tuple[str, str]:
        """Ask the agent to author solve()/verify() (no framework synthesis), then validate."""
        allowed_tool_names = [s.name for s in tool_specs]
        data_tool_names = [n for n in allowed_tool_names if n != self._SUBMIT_RESULT_TOOL]
        tool_list = [{"name": s.name, "description": s.description} for s in tool_specs]
        max_tokens = getattr(ctx.request, "max_tokens", 10000)

        base_prompt = (
            "You are writing agent-authored Python code for a sandboxed environment.\n"
            "Return ONLY JSON with keys: solution, verification.\n\n"
            "Rules for `solution`:\n"
            "- Must define exactly: def solve(tools):\n"
            "- No imports, no helper functions, no classes.\n"
            "- Must call at least TWO different data tools from the allowed tool names (exclude submit_result).\n"
            "- FINAL LINE MUST BE: return tools['submit_result'](answer)\n"
            "  (This ensures solve() returns the answer payload; do NOT omit `return`.)\n"
            "- Must call submit_result exactly once.\n"
            "- Must construct `answer` to match submit_result_format.\n\n"
            "Rules for `verification`:\n"
            "- Must define: def verify(tools, answer):\n"
            "- Deterministic; may import json.\n"
            "- Must NOT be trivial (no unconditional pass/return True).\n"
            "- Must be exception-safe: DO NOT raise. If any unexpected condition or exception occurs, return False.\n"
            "- Must handle answer is None / missing keys / None values without throwing (avoid NoneType is not iterable).\n"
            "- Must ALWAYS return a boolean OR a dict with keys like {passed: bool, message: str, details: ...}.\n"
            "- If returning a dict, include a short failure message to help repairs.\n"
            "- If answer is wrapped by submit_result (keys like status/message/submitted_data), verify the submitted_data payload.\n"
            "- Must call at least ONE data tool (exclude submit_result) to cross-check outputs.\n"
            "- If tools return matching records, verification must fail when the answer is empty/missing those records.\n"
            "- Should verify `answer` matches submit_result_format and is consistent with tool outputs.\n\n"
            f"Topic: {request.topic}\n"
            f"Task content: {task_content}\n"
            f"Allowed tool names (EXACT): {json.dumps(allowed_tool_names, ensure_ascii=False)}\n"
            f"Tools (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"submit_result_format (JSON): {json.dumps(submit_result_format, ensure_ascii=False)}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
            f"Tool self-tests (schemas): {json.dumps(tool_selftest or {}, ensure_ascii=False)[:1200]}\n"
        )
        if repair_error:
            base_prompt += (
                "\nYou are REPAIRING a previously generated package.\n"
                f"Repair target: {repair_target or 'solution'}\n"
                f"Observed error: {repair_error[:1200]}\n"
                "Fix the issue and return updated code.\n"
                "- If repairing verification, keep the solution logic unchanged unless strictly necessary.\n"
                "- If repairing solution, ensure answer matches submit_result_format and still call submit_result exactly once.\n"
                "- If repairing both, rewrite both solve() and verify(); do not reuse the previous logic.\n"
            )

        last_err = ""
        for attempt in range(1, 4):
            prompt = base_prompt
            if previous_solution and previous_verification:
                prompt += (
                    "\nPrevious code (may be improved but keep constraints):\n"
                    f"SOLUTION:\n{previous_solution[:1200]}\n"
                    f"VERIFICATION:\n{previous_verification[:1200]}\n"
                )
            if last_err:
                prompt += f"\nPrevious attempt errors: {last_err}\nFix them and try again.\n"

            raw = self.llm.simple_complete(prompt, temperature=0.6, max_tokens=max_tokens)
            ctx.add_step({"type": "agent_code_raw", "attempt": attempt, "content": raw})
            extracted = self._extract_json(raw) or {}
            sol_code = self._sanitize_python_code(extracted.get("solution"))
            ver_code = self._sanitize_python_code(extracted.get("verification"))

            sol_ok, sol_err = CodeValidator.validate_solution_code(sol_code)
            ver_ok, ver_err = CodeValidator.validate_verification_code(ver_code)
            if not sol_ok or not ver_ok:
                last_err = f"solution_ok={sol_ok} solution_err={sol_err}; verification_ok={ver_ok} verification_err={ver_err}"
                ctx.add_step({"type": "agent_code_validation_failed", "attempt": attempt, "error": last_err})
                continue

            verify_called = CodeValidator.extract_tool_calls(ver_code)
            verify_used_data = sorted(set(verify_called) - {self._SUBMIT_RESULT_TOOL})
            if not verify_used_data:
                last_err = "verification must call at least one data tool"
                ctx.add_step({"type": "agent_code_validation_failed", "attempt": attempt, "error": last_err})
                continue
            invented_ver = sorted(set(verify_called) - set(allowed_tool_names))
            if invented_ver:
                last_err = f"verification calls tools not in allowed list: {invented_ver}"
                ctx.add_step({"type": "agent_code_validation_failed", "attempt": attempt, "error": last_err})
                continue

            called = CodeValidator.extract_tool_calls(sol_code)
            if self._SUBMIT_RESULT_TOOL not in called:
                last_err = "solution must call submit_result"
                ctx.add_step({"type": "agent_code_validation_failed", "attempt": attempt, "error": last_err})
                continue
            used_data = sorted(set(called) - {self._SUBMIT_RESULT_TOOL})
            if len(used_data) < 2:
                last_err = f"solution must call >=2 different data tools; used={used_data}"
                ctx.add_step({"type": "agent_code_validation_failed", "attempt": attempt, "error": last_err})
                continue
            # Ensure no invented tools.
            invented = sorted(set(called) - set(allowed_tool_names))
            if invented:
                last_err = f"solution calls tools not in allowed list: {invented}"
                ctx.add_step({"type": "agent_code_validation_failed", "attempt": attempt, "error": last_err})
                continue

            # Ensure solve() returns the submitted payload (avoid answer=None).
            if "return tools['submit_result']" not in sol_code and 'return tools["submit_result"]' not in sol_code:
                last_err = "solution must end with: return tools['submit_result'](answer)"
                ctx.add_step({"type": "agent_code_validation_failed", "attempt": attempt, "error": last_err})
                continue
            try:
                tree = ast.parse(sol_code)
                solve_fn = None
                for node in tree.body:
                    if isinstance(node, ast.FunctionDef) and node.name == "solve":
                        solve_fn = node
                    break
                if solve_fn is None or not solve_fn.body:
                    raise ValueError("missing solve() body")
                last_stmt = solve_fn.body[-1]
                if not isinstance(last_stmt, ast.Return):
                    raise ValueError("last statement in solve() must be a return")
            except Exception as exc:
                last_err = f"solution must end with a return of submit_result call (parse_check_failed: {exc})"
                ctx.add_step({"type": "agent_code_validation_failed", "attempt": attempt, "error": last_err})
                continue

            ctx.add_step(
                {
                    "type": "agent_code_validated",
                    "attempt": attempt,
                    "tools_used": used_data,
                }
            )
            return sol_code, ver_code

        raise RuntimeError(f"Agent failed to generate valid solution/verification after retries: {last_err}")

    def _ensure_task_content(
        self,
        task_content: str,
        submit_result_format: Any,
        topic: str | None,
        tool_names: list[str],
    ) -> str:
        content = (task_content or "").strip()
        sentences = [s for s in re.split(r"[.!?]", content) if s.strip()]
        if content and len(content) >= 220 and len(sentences) >= 3:
            return content

        fmt = submit_result_format
        if isinstance(fmt, str):
            try:
                fmt = json.loads(fmt)
            except Exception:
                fmt = {}
        fields: list[str] = []
        if isinstance(fmt, dict):
            if isinstance(fmt.get("properties"), dict):
                fields = [k for k in fmt["properties"].keys() if isinstance(k, str)]
            else:
                fields = [k for k in fmt.keys() if isinstance(k, str) and k not in {"type", "required"}]
        elif isinstance(fmt, list) and fmt and isinstance(fmt[0], dict):
            fields = [k for k in fmt[0].keys() if isinstance(k, str)]

        topic_label = topic or "the topic"
        tool_hint = ", ".join(tool_names[:3]) if tool_names else "the available tools"
        field_hint = ", ".join(fields[:6]) if fields else "the required output fields"
        requirements = (
            f"Use at least two different tools ({tool_hint}) and ensure the output strictly follows "
            f"submit_result_format with fields like {field_hint}. Focus on entries explicitly tied to "
            f"{topic_label}, and exclude generic or unrelated records. If results are sparse, explain "
            "how you selected fallback entries from the available tools."
        )
        if content:
            if content[-1] not in ".!?":
                content += "."
            return f"{content} {requirements}"
        return f"Use the available tools to solve a {topic_label} data task. {requirements}"

    def _propose_task(
        self,
        task_id: str,
        request: "GenerationRequest",
        records: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        ctx: TaskContext,
        retry: int = 0,
        tools_code: str = "",
        data_profile: dict[str, Any] | None = None,
        tool_selftest: dict[str, Any] | None = None,
    ) -> TaskPackage:
        tool_list = [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in tool_specs
            if spec.name != self._SUBMIT_RESULT_TOOL
        ]
        tool_names = [spec.name for spec in tool_specs if spec.name != self._SUBMIT_RESULT_TOOL]

        prompt = (
            "You are a task generator.\n"
            "Create ONE verifiable task that will be progressively strengthened in later refinement rounds. Start with a"
            " reasonable, solvable baseline and align with the available tools/data.\n"
            "Return ONLY JSON with keys: task_title, task_content, submit_result_format, difficulty_level.\n"
            f"Topic: {request.topic}\n"
            f"Tool list (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"Allowed tool names (you MUST call from these): {json.dumps(tool_names, ensure_ascii=False)}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
            f"Local data sources (detected): {json.dumps(data_profile or {}, ensure_ascii=False)[:1200]}\n"
            f"Tool self-tests (schemas): {json.dumps(tool_selftest or {}, ensure_ascii=False)[:1200]}\n"
            "CRITICAL: design submit_result_format and task_content ONLY using fields actually available from the tools/data above.\n"
        )

        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        raw = self.llm.simple_complete(prompt, temperature=0.35, max_tokens=max_tokens)
        ctx.add_step({"type": "task_proposed", "content": raw})
        self.writer.record_steps(task_id, self.agent_type, ctx.history)

        extracted = self._extract_json(raw)
        if isinstance(extracted, list) and extracted and isinstance(extracted[0], dict):
            extracted = extracted[0]

        if extracted is None and retry < 3:
            self.logger.warning(
                "Retrying task proposal (attempt %s/3). Previous error: invalid JSON.",
                retry + 2,
            )
            return self._propose_task(
                task_id,
                request,
                records,
                tool_specs,
                ctx,
                retry=retry + 1,
                tools_code=tools_code,
                data_profile=data_profile,
                tool_selftest=tool_selftest,
            )

        if not isinstance(extracted, dict):
            raise RuntimeError("Task proposal failed: invalid JSON response.")

        task_content = (extracted.get("task_content") or "").strip()
        submit_result_format = extracted.get("submit_result_format") or {
            "type": "object",
            "properties": {"result": {"type": "array", "items": {"type": "object"}}},
            "required": ["result"],
        }
        task_content = self._ensure_task_content(
            task_content,
            submit_result_format,
            request.topic,
            tool_names,
        )

        difficulty_raw = extracted.get("difficulty_level")
        if isinstance(difficulty_raw, str):
            mapped = {
                "beginner": 1,
                "easy": 1,
                "medium": 2,
                "hard": 3,
                "expert": 4,
            }.get(difficulty_raw.strip().lower())
            difficulty_raw = mapped if mapped is not None else None
        ctx.current_difficulty = int(difficulty_raw or ctx.current_difficulty or 1)
        ctx.add_step(
            {
                "type": "parse_task_info",
                "content": {
                    "submit_result_format": submit_result_format,
                    "difficulty_level": ctx.current_difficulty,
                },
            }
        )
        self.writer.record_steps(task_id, self.agent_type, ctx.history)

        # Agent-authored code (no framework synthesis).
        solution_code, verification_code = self._generate_agent_solution_and_verification(
            request=request,
            ctx=ctx,
            task_content=task_content,
            submit_result_format=submit_result_format,
            tool_specs=tool_specs,
            records=records,
            tool_selftest=tool_selftest,
        )

        ctx.add_step(
            {"type": "solution_and_verification_code", "content": {"solution_code": solution_code, "verification_code": verification_code}}
        )
        pkg = TaskPackage(
            task=TaskDefinition(
                task_id=task_id,
                task_title=request.topic,
                task_content=(task_content if len(task_content) >= 10 else f"Solve a task about {request.topic}."),
                submit_result_format=submit_result_format,
                tool_set=tool_specs,
                evaluation_criteria=EvaluationCriteria(),
                difficulty_level=ctx.current_difficulty,
            ),
            agent_type=self.agent_type,
            solution=solution_code,
            verification=verification_code,
            metadata={"topic": request.topic, "tools_code": tools_code[:5000] if tools_code else ""},
        )

        return pkg

    def _build_permissive_verifier(self, submit_result_format: dict[str, Any]) -> str:
        fmt = repr(submit_result_format)
        return (
            "def verify(tools, answer):\n"
            "    details = []\n"
            f"    fmt = {fmt}\n"
            "    def _is_schema(obj):\n"
            "        return isinstance(obj, dict) and 'type' in obj and ('properties' in obj or 'items' in obj or 'enum' in obj)\n"
            "    def _check_schema(schema, value, path):\n"
            "        if not isinstance(schema, dict):\n"
            "            return True\n"
            "        if 'enum' in schema and isinstance(schema['enum'], list):\n"
            "            return value in schema['enum']\n"
            "        stype = schema.get('type') or 'object'\n"
            "        if stype == 'object':\n"
            "            if not isinstance(value, dict):\n"
            "                details.append({'name': path, 'passed': False, 'msg': 'expected object'})\n"
            "                return False\n"
            "            props = schema.get('properties') or {}\n"
            "            ok = True\n"
            "            for k, v in props.items():\n"
            "                ok = _check_schema(v, value.get(k), path + '.' + str(k)) and ok\n"
            "            return ok\n"
            "        if stype == 'array':\n"
            "            if not isinstance(value, list):\n"
            "                details.append({'name': path, 'passed': False, 'msg': 'expected array'})\n"
            "                return False\n"
            "            item_schema = schema.get('items') or {}\n"
            "            ok = True\n"
            "            for idx, item in enumerate(value[:3]):\n"
            "                ok = _check_schema(item_schema, item, path + '[' + str(idx) + ']') and ok\n"
            "            return ok\n"
            "        if stype == 'string':\n"
            "            return isinstance(value, str)\n"
            "        if stype == 'integer':\n"
            "            return isinstance(value, int)\n"
            "        if stype == 'number':\n"
            "            return isinstance(value, (int, float))\n"
            "        if stype == 'boolean':\n"
            "            return isinstance(value, bool)\n"
            "        return True\n"
            "    def _check_example(example, value, path):\n"
            "        if isinstance(example, dict):\n"
            "            if not isinstance(value, dict):\n"
            "                details.append({'name': path, 'passed': False, 'msg': 'expected object'})\n"
            "                return False\n"
            "            ok = True\n"
            "            for k, v in example.items():\n"
            "                ok = _check_example(v, value.get(k), path + '.' + str(k)) and ok\n"
            "            return ok\n"
            "        if isinstance(example, list):\n"
            "            return isinstance(value, list)\n"
            "        if isinstance(example, str):\n"
            "            return isinstance(value, str)\n"
            "        if isinstance(example, bool):\n"
            "            return isinstance(value, bool)\n"
            "        if isinstance(example, int):\n"
            "            return isinstance(value, int)\n"
            "        if isinstance(example, float):\n"
            "            return isinstance(value, (int, float))\n"
            "        return True\n"
            "    if _is_schema(fmt):\n"
            "        passed = _check_schema(fmt, answer, 'answer')\n"
            "    else:\n"
            "        passed = _check_example(fmt, answer, 'answer')\n"
            "    score = 1.0 if passed else 0.0\n"
            "    return {'passed': passed, 'score': score, 'details': details}\n"
        )

    def _refine_task(
        self,
        previous: TaskPackage,
        records: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        ctx: TaskContext,
        target_difficulty: int,
    ) -> TaskPackage:
        tool_list = [{"name": spec.name, "description": spec.description} for spec in tool_specs]
        prompt = (
            "Increase the task difficulty while keeping it verifiable.\n"
            "Return ONLY JSON with keys: task_content, submit_result_format, difficulty_level, solution, verification.\n"
            "Keep solve(tools) / verify(tools, answer) signatures unchanged.\n"
            "Do not introduce new tools; only use tools from the provided list.\n"
            f"Target difficulty_level: {target_difficulty}\n"
            f"Tools (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
            f"Previous task (JSON): {json.dumps(previous.as_payload(), ensure_ascii=False)}\n"
        )
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        raw = self.llm.simple_complete(prompt, temperature=0.7, max_tokens=max_tokens)
        extracted = self._extract_json(raw) or {}

        task_content = str(extracted.get("task_content") or previous.task.task_content).strip()
        submit_result_format = extracted.get("submit_result_format", previous.task.submit_result_format)
        if isinstance(submit_result_format, str):
            submit_result_format = {"type": submit_result_format}
        tool_names = [spec.name for spec in tool_specs if spec.name != self._SUBMIT_RESULT_TOOL]
        task_content = self._ensure_task_content(
            task_content,
            submit_result_format,
            ctx.request.topic if ctx and ctx.request else previous.task.task_title,
            tool_names,
        )

        # Prefer agent-provided solution/verification from the refinement JSON; validate strictly.
        candidate_solution = self._sanitize_python_code(extracted.get("solution"))
        candidate_verification = self._sanitize_python_code(extracted.get("verification"))
        sol_ok, sol_err = CodeValidator.validate_solution_code(candidate_solution)
        ver_ok, ver_err = CodeValidator.validate_verification_code(candidate_verification)
        if sol_ok and ver_ok:
            solution = candidate_solution
            verification = candidate_verification
        else:
            ctx.add_step(
                {
                    "type": "refine_code_invalid",
                    "solution_ok": sol_ok,
                    "solution_err": sol_err,
                    "verification_ok": ver_ok,
                    "verification_err": ver_err,
                }
            )
            solution, verification = self._generate_agent_solution_and_verification(
                request=ctx.request,
                ctx=ctx,
                task_content=task_content,
                submit_result_format=submit_result_format,
                tool_specs=tool_specs,
                records=records,
                tool_selftest=None,
                previous_solution=previous.solution or "",
                previous_verification=previous.verification or "",
            )

        pkg = TaskPackage(
            task=TaskDefinition(
                task_id=previous.task.task_id,
                task_title=previous.task.task_title,
                task_content=(
                    task_content if len(task_content) >= 10 else previous.task.task_content
                ),
                submit_result_format=submit_result_format,
                tool_set=tool_specs,
                evaluation_criteria=previous.task.evaluation_criteria,
                difficulty_level=int(extracted.get("difficulty_level") or ctx.current_difficulty),
            ),
            solution=solution,
            verification=verification,
            agent_type=self.agent_type,
            metadata={**previous.metadata, "topic": previous.task.task_title, "refined": "true"},
        )
        ctx.add_step({"type": "task_refined", "task_title": pkg.task.task_title, "difficulty_level": pkg.task.difficulty_level})
        return pkg
