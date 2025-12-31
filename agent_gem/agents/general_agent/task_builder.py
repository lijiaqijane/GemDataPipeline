from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any, TYPE_CHECKING

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

    def _coerce_text(self, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, list):
            return "; ".join(str(item) for item in value if item is not None).strip()
        if isinstance(value, dict):
            try:
                return json.dumps(value, ensure_ascii=True)
            except Exception:
                return str(value).strip()
        return str(value).strip()

    def _generate_repair_guidance(
        self,
        *,
        request: "GenerationRequest",
        task_content: str,
        submit_result_format: dict[str, Any],
        tool_specs: list[ToolSpec],
        previous_solution: str | None,
        previous_verification: str | None,
        repair_error: str,
        repair_target: str | None,
        records: list[dict[str, Any]],
        tool_selftest: dict[str, Any] | None,
        max_tokens: int,
    ) -> dict[str, str]:
        tool_list = [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
                "output_schema": s.output_schema,
                "output_keys": (s.meta or {}).get("output_keys"),
            }
            for s in tool_specs
        ]
        prompt = (
            "You are a debugging assistant. Analyze the failure and propose a prompt addition that forces a different approach.\n"
            "Return ONLY JSON with keys: diagnosis, required_changes, prompt_additions.\n"
            "- diagnosis: concise and specific root cause guess.\n"
            "- required_changes: 2-4 concrete changes to apply (list or string).\n"
            "- prompt_additions: 3-6 sentences to append to the generation prompt.\n"
            "  It must include at least one constraint that changes data selection strategy\n"
            "  (e.g., adjust filters, combine tools, change query terms) and one constraint\n"
            "  about output structure. Avoid repeating the original prompt. English only.\n\n"
            f"Repair target: {repair_target or 'solution'}\n"
            f"Failure: {repair_error[:1200]}\n"
            f"Task content: {task_content[:1200]}\n"
            f"submit_result_format (JSON): {json.dumps(submit_result_format, ensure_ascii=True)}\n"
            f"Tools (JSON): {json.dumps(tool_list, ensure_ascii=True)}\n"
            f"Database sample (JSON): {json.dumps(records[:3], ensure_ascii=True)}\n"
            f"Tool self-tests (schemas): {json.dumps(tool_selftest or {}, ensure_ascii=True)[:1200]}\n"
            f"Previous solution snippet:\n{(previous_solution or '')[:800]}\n"
            f"Previous verification snippet:\n{(previous_verification or '')[:800]}\n"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.3, max_tokens=max_tokens)
        parsed = self._extract_json(raw) or {}
        return {
            "diagnosis": self._coerce_text(parsed.get("diagnosis")),
            "required_changes": self._coerce_text(parsed.get("required_changes")),
            "prompt_additions": self._coerce_text(parsed.get("prompt_additions")),
        }

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
        """Ask the agent to author solve()/verify() in separate calls, then validate."""
        allowed_tool_names = [s.name for s in tool_specs]
        tool_list = [
            {
                "name": s.name,
                "description": s.description,
                "parameters": s.parameters,
                "output_schema": s.output_schema,
                "output_keys": (s.meta or {}).get("output_keys"),
            }
            for s in tool_specs
        ]
        max_tokens = getattr(ctx.request, "max_tokens", 10000)

        common_context = (
            f"Topic: {request.topic}\n"
            f"Task content: {task_content}\n"
            f"Allowed tool names (EXACT): {json.dumps(allowed_tool_names, ensure_ascii=False)}\n"
            f"Tools (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"submit_result_format (JSON): {json.dumps(submit_result_format, ensure_ascii=False)}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
            f"Tool self-tests (schemas): {json.dumps(tool_selftest or {}, ensure_ascii=False)[:1200]}\n"
        )

        solution_base = (
            "You are writing agent-authored Python code for a sandboxed environment.\n"
            "Return ONLY JSON with key: solution.\n\n"
            "Rules for `solution`:\n"
            "- Must define exactly: def solve(tools):\n"
            "- No imports, no helper functions, no classes.\n"
            "- You do NOT have direct database access; rely only on tool outputs.\n"
            "- Must call at least TWO different data tools from the allowed tool names (exclude submit_result).\n"
            "- Tools are plain Python callables; do NOT pass a single dict positional argument.\n"
            "  Use keyword arguments: tools['get_vulnerabilities'](project_name=..., security_risk=...).\n"
            "- Data tools (everything except submit_result) return a dict wrapper with key 'result'.\n"
            "  Unwrap with out['result'] or out.get('result', out).\n"
            "- Respect tool output_schema exactly. If a tool returns list[str], treat items as strings (no .get()).\n"
            "- When mapping tool outputs, use the exact keys returned by the tools; do NOT assume snake_case or raw headers.\n"
            "  Do NOT invent Title Case keys from raw datasets unless the tool output uses them.\n"
            "- If the tool list provides output_keys, use those exact keys.\n"
            "- If tool outputs include nested structures (e.g., matching_excerpts/context), extract usable text fields.\n"
            "- If a required output list is empty, try alternative queries or another available tool to populate it.\n"
            "- FINAL LINE MUST BE: return tools['submit_result'](answer)\n"
            "  (This ensures solve() returns the answer payload; do NOT omit `return`.)\n"
            "- Must call submit_result exactly once.\n"
            "- Must construct `answer` to match submit_result_format.\n"
            "- Do NOT convert `answer` to a string (no str/repr/json.dumps/f-strings). Pass the dict directly.\n"
            "- Use tool parameters exactly as defined in the tool schemas; do NOT invent parameter names.\n"
            "- If a parameter has a default, you may omit it; otherwise pass it explicitly.\n"
            "  NOTE: submit_result_format is a JSON Schema that defines the structure of `answer`.\n"
            "  IMPORTANT: The submit_result tool contains NO validation - it's a simple pass-through wrapper.\n"
            "  It just returns whatever you pass: def submit_result(result: Any) -> Any: return result\n"
            "  Workflow: (1) Query data using data tools, (2) Transform results into answer dict matching submit_result_format, "
            "(3) Call tools['submit_result'](answer) to submit (no validation happens here), (4) Return the result.\n\n"
            f"{common_context}"
        )

        verification_base = (
            "You are writing agent-authored Python code for a sandboxed environment.\n"
            "Return ONLY JSON with key: verification.\n\n"
            "Rules for `verification`:\n"
            "- Must define: def verify(tools, answer):\n"
            "- Deterministic; may import json, re.\n"
            "- Must NOT be trivial (no unconditional pass/return True).\n"
            "- Must be exception-safe: DO NOT raise. If any unexpected condition or exception occurs, return False.\n"
            "- Must handle answer is None / missing keys / None values without throwing (avoid NoneType is not iterable).\n"
            "- Must ALWAYS return a boolean OR a dict with keys like {passed: bool, message: str, details: ...}.\n"
            "- If returning a dict, include a short failure message to help repairs.\n"
            "- If answer is wrapped by submit_result (keys like status/message/submitted_data or status/message/data), verify the wrapped payload.\n"
            "- May call data tools (exclude submit_result) to cross-check outputs, but it is not required.\n"
            "- Do not assume anything about the solution logic; validate using schema and tool outputs only.\n"
            "- Tools are plain Python callables; do NOT pass a single dict positional argument.\n"
            "  Use keyword arguments: tools['get_vulnerabilities'](project_name=..., security_risk=...).\n"
            "- Data tools (everything except submit_result) return a dict wrapper with key 'result'.\n"
            "  Unwrap with out['result'] or out.get('result', out).\n"
            "- Respect tool output_schema exactly. If a tool returns list[str], treat items as strings (no .get()).\n"
            "- When checking fields derived from tool outputs, use the keys actually returned by the tools; do NOT assume snake_case or raw headers.\n"
            "- If the tool list provides output_keys, use those exact keys.\n"
            "- Use tool parameters exactly as defined in the tool schemas; do NOT invent parameter names.\n"
            "- Should verify `answer` matches submit_result_format STRUCTURE and is consistent with tool outputs.\n"
            "  NOTE: submit_result_format is a JSON Schema. Verify that answer has all required fields, correct types, "
            "and that the data values are consistent with what the data tools returned.\n"
            "  IMPORTANT: The submit_result tool does NOT validate format - it contains NO validation logic at all.\n"
            "  All format validation is YOUR responsibility in verify(). The submit_result tool is just a pass-through wrapper.\n\n"
            f"{common_context}"
        )
        repair_guidance: dict[str, str] | None = None
        if repair_error:
            try:
                repair_guidance = self._generate_repair_guidance(
                    request=request,
                    task_content=task_content,
                    submit_result_format=submit_result_format,
                    tool_specs=tool_specs,
                    previous_solution=previous_solution,
                    previous_verification=previous_verification,
                    repair_error=repair_error,
                    repair_target=repair_target,
                    records=records,
                    tool_selftest=tool_selftest if isinstance(tool_selftest, dict) else {},
                    max_tokens=min(800, max_tokens),
                )
                ctx.add_step({"type": "repair_guidance", "content": repair_guidance})
            except Exception:
                repair_guidance = None

        target = (repair_target or "").strip().lower()
        regen_solution = target in {"", "solution", "both"}
        regen_verification = target in {"", "verification", "both"}

        solution_code = (previous_solution or "").strip()
        verification_code = (previous_verification or "").strip()

        def _repair_context(label: str) -> str:
            if not repair_error:
                return ""
            if target and label not in {target, "both"}:
                return ""
            return (
                "\nYou are REPAIRING a previously generated package.\n"
                f"Repair target: {repair_target or label}\n"
                f"Observed error: {repair_error[:1200]}\n"
                "Fix the issue and return updated code.\n"
                "- If repairing verification, keep the solution logic unchanged unless strictly necessary.\n"
                "- If repairing solution, ensure answer matches submit_result_format and still call submit_result exactly once.\n"
                "- If repairing both, rewrite both solve() and verify(); do not reuse the previous logic.\n"
            )

        def _repair_guidance_block(label: str) -> str:
            if not repair_guidance:
                return ""
            if target and label not in {target, "both"}:
                return ""
            return (
                "\nRepair guidance (must follow):\n"
                f"Diagnosis: {repair_guidance.get('diagnosis', '')}\n"
                f"Required changes: {repair_guidance.get('required_changes', '')}\n"
                f"Prompt additions: {repair_guidance.get('prompt_additions', '')}\n"
                "Apply the required changes and avoid repeating the previous approach.\n"
            )

        if regen_solution or not solution_code:
            last_err = ""
            for attempt in range(1, 4):
                prompt = solution_base
                prompt += _repair_context("solution")
                if previous_solution:
                    prompt += f"\nPrevious solution snippet:\n{previous_solution[:1200]}\n"
                prompt += _repair_guidance_block("solution")
                if last_err:
                    prompt += f"\nPrevious attempt errors: {last_err}\nFix them and try again.\n"

                raw = self.llm.simple_complete(prompt, temperature=0.6, max_tokens=max_tokens)
                ctx.add_step({"type": "agent_solution_raw", "attempt": attempt, "content": raw})
                extracted = self._extract_json(raw) or {}
                sol_code = self._sanitize_python_code(extracted.get("solution"))

                sol_ok, sol_err = CodeValidator.validate_solution_code(sol_code)
                if not sol_ok:
                    last_err = f"solution_ok={sol_ok} solution_err={sol_err}"
                    ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
                    continue

                called = CodeValidator.extract_tool_calls(sol_code)
                if self._SUBMIT_RESULT_TOOL not in called:
                    last_err = "solution must call submit_result"
                    ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
                    continue
                used_data = sorted(set(called) - {self._SUBMIT_RESULT_TOOL})
                if len(used_data) < 2:
                    last_err = f"solution must call >=2 different data tools; used={used_data}"
                    ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
                    continue
                invented = sorted(set(called) - set(allowed_tool_names))
                if invented:
                    last_err = f"solution calls tools not in allowed list: {invented}"
                    ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
                    continue

                if "return tools['submit_result']" not in sol_code and 'return tools["submit_result"]' not in sol_code:
                    last_err = "solution must end with: return tools['submit_result'](answer)"
                    ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
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
                    ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
                    continue

                ctx.add_step(
                    {
                        "type": "agent_solution_validated",
                        "attempt": attempt,
                        "tools_used": used_data,
                    }
                )
                solution_code = sol_code
                break
            if not solution_code:
                raise RuntimeError(f"Agent failed to generate valid solution after retries: {last_err}")

        if regen_verification or not verification_code:
            last_err = ""
            for attempt in range(1, 4):
                prompt = verification_base
                prompt += _repair_context("verification")
                if previous_verification:
                    prompt += f"\nPrevious verification snippet:\n{previous_verification[:1200]}\n"
                prompt += _repair_guidance_block("verification")
                if last_err:
                    prompt += f"\nPrevious attempt errors: {last_err}\nFix them and try again.\n"

                raw = self.llm.simple_complete(prompt, temperature=0.6, max_tokens=max_tokens)
                ctx.add_step({"type": "agent_verification_raw", "attempt": attempt, "content": raw})
                extracted = self._extract_json(raw) or {}
                ver_code = self._sanitize_python_code(extracted.get("verification"))

                ver_ok, ver_err = CodeValidator.validate_verification_code(ver_code)
                if not ver_ok:
                    last_err = f"verification_ok={ver_ok} verification_err={ver_err}"
                    ctx.add_step({"type": "agent_verification_validation_failed", "attempt": attempt, "error": last_err})
                    continue

                ctx.add_step({"type": "agent_verification_validated", "attempt": attempt})
                verification_code = ver_code
                break
            if not verification_code:
                raise RuntimeError(f"Agent failed to generate valid verification after retries: {last_err}")

        return solution_code, verification_code

    def _ensure_task_content(
        self,
        task_content: str,
        submit_result_format: Any,
        topic: str | None,
        tool_names: list[str],
    ) -> str:
        content = (task_content or "").strip()
        sentences = [s for s in re.split(r"[.!?]", content) if s.strip()]
        # Increased minimum requirements: 400 chars and 5 sentences for more detailed task descriptions
        if content and len(content) >= 400 and len(sentences) >= 5:
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
                "output_schema": spec.output_schema,
                "output_keys": (spec.meta or {}).get("output_keys"),
            }
            for spec in tool_specs
            if spec.name != self._SUBMIT_RESULT_TOOL
        ]
        tool_names = [spec.name for spec in tool_specs if spec.name != self._SUBMIT_RESULT_TOOL]

        data_profile_payload: Any = data_profile or {}
        tool_selftest_payload: Any = tool_selftest or {}
        try:
            data_profile_payload = self._convert_paths_to_strings(data_profile_payload)
            tool_selftest_payload = self._convert_paths_to_strings(tool_selftest_payload)
        except Exception:
            pass

        prompt = (
            "You are a task generator.\n"
            "Create ONE verifiable task that will be progressively strengthened in later refinement rounds. Start with a"
            " reasonable, solvable baseline and align with the available tools/data.\n"
            "Return ONLY JSON with keys: task_title, task_content, submit_result_format, difficulty_level.\n\n"
            "REQUIREMENTS for task_content (MUST be detailed and comprehensive):\n"
            "- Must be at least 400-500 characters long with 5-8 complete sentences.\n"
            "- Must clearly describe: (1) What data to query/retrieve, (2) What operations to perform, "
            "(3) What specific criteria or filters to apply, (4) What the expected output should contain, "
            "(5) Any edge cases or special considerations.\n"
            "- Must reference specific tool names and explain how they should be used together.\n"
            "- Must include concrete examples of what fields/values to look for in the data.\n"
            "- Must specify any data quality requirements (e.g., exclude nulls, filter by date ranges).\n"
            "- If you suggest specific search queries, they MUST appear in the provided data samples or tool outputs.\n"
            "- Write in clear, actionable language that leaves no ambiguity about the task requirements.\n\n"
            f"Topic: {request.topic}\n"
            f"Tool list (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"Allowed tool names (you MUST call from these): {json.dumps(tool_names, ensure_ascii=False)}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
            f"Local data sources (detected): {json.dumps(data_profile_payload, ensure_ascii=False)[:1200]}\n"
            f"Tool self-tests (schemas): {json.dumps(tool_selftest_payload, ensure_ascii=False)[:1200]}\n"
            + "CRITICAL: design submit_result_format and task_content ONLY using fields actually available from the tools/data above.\n"
            + "CRITICAL: difficulty_level must be the integer 1.\n"
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
        # submit_result_format: JSON Schema defining the structure of the answer
        # This format is used to:
        # 1. Guide solve() to construct the answer dict with correct structure
        # 2. Guide verify() to validate that answer matches the expected schema
        # 
        # IMPORTANT: The submit_result tool itself contains NO validation or schema checking.
        # It is a simple pass-through wrapper: def submit_result(result: Any) -> Any: return result
        # All validation is done by verify() function, not by submit_result tool.
        # 
        # Relationship:
        # - solve() constructs answer matching submit_result_format -> calls submit_result(answer) -> returns result
        # - verify() receives answer and validates it matches submit_result_format structure
        # - submit_result tool does NOT check format, it just returns what it receives
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

        # Force difficulty_level to 1 for initial task proposal.
        ctx.current_difficulty = 1
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
            "Return ONLY JSON with keys: task_content, submit_result_format, difficulty_level.\n"
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
