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

    def _sanitize_task_content(self, content: str, tool_names: list[str]) -> str:
        if not content:
            return content
        cleaned = content
        for name in sorted(set(tool_names), key=len, reverse=True):
            if not name:
                continue
            pattern = re.compile(rf"(?i)\b{re.escape(name)}\b")
            cleaned = pattern.sub("a data tool", cleaned)
        cleaned = re.sub(r"(?i)\btools\.py\b", "a tool module", cleaned)
        return cleaned

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
        max_tokens: int,
    ) -> str:
        """Generate repair guidance based on the failure.
        
        Returns:
            str: Prompt additions that include diagnosis, required changes, and guidance.
                 This text will be appended to the generation prompt to guide the repair.
        """
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
        
        # Determine which code to include based on repair_target
        target = (repair_target or "").strip().lower()
        include_solution = target in {"", "solution", "both"}
        include_verification = target in {"", "verification", "both"}
        
        # Include full code (no truncation) for repair guidance generation
        code_snippet_parts = []
        if include_solution and previous_solution:
            code_snippet_parts.append(f"Previous solution:\n{previous_solution}\n")
        if include_verification and previous_verification:
            code_snippet_parts.append(f"Previous verification:\n{previous_verification}\n")
        
        code_snippet = "\n".join(code_snippet_parts)
        
        prompt = (
            "You are a debugging assistant. Analyze the failure and propose prompt additions that guide a different approach.\n"
            "Return ONLY JSON with key: prompt_additions.\n"
            "- prompt_additions: A comprehensive text (5-10 sentences) that includes:\n"
            "  (1) A concise diagnosis of the root cause,\n"
            "  (2) 2-4 concrete required changes to fix the issue,\n"
            "  (3) Specific guidance on how to modify the approach.\n"
            "  It must include at least one constraint that changes data selection strategy\n"
            "  (e.g., adjust filters, combine tools, change query terms) and one constraint\n"
            "  about output structure. Avoid repeating the original prompt. \n\n"
            f"Repair target: {repair_target or 'solution'}\n"
            f"Failure: {repair_error[:1200]}\n"
            f"Task content: {task_content[:1200]}\n"
            f"The task content above includes the submit_result_format description. "
            f"Refer to the task content for the required output structure.\n"
            f"Tools (JSON): {json.dumps(tool_list, ensure_ascii=True)}\n"
            f"{code_snippet}"
        )
        self.logger.info(f"LLM call: Generating repair guidance for {repair_target or 'solution/verification'}")
        raw = self.llm.simple_complete(prompt, temperature=0.3, max_tokens=max_tokens)
        parsed = self._extract_json(raw) or {}
        return self._coerce_text(parsed.get("prompt_additions", ""))

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
        submit_result_function_name: str | None = None,
    ) -> tuple[str, str]:
        """Ask the agent to author solve()/verify() in separate calls, then validate."""
        # Determine submit_result function name
        if submit_result_function_name is None:
            # Try to find it from tool_specs, matching current difficulty
            difficulty = getattr(ctx, "current_difficulty", 1)
            expected_name = f"submit_result_difficulty_{difficulty}"
            
            # First, try to find exact match for current difficulty
            submit_result_specs = [s for s in tool_specs if s.name == expected_name]
            if submit_result_specs:
                submit_result_function_name = submit_result_specs[0].name
            else:
                # Fallback: find any submit_result function (but prefer current difficulty)
                all_submit_specs = [s for s in tool_specs if s.name.startswith("submit_result_difficulty_")]
                if all_submit_specs:
                    # Sort by difficulty number, prefer current difficulty
                    def get_difficulty_num(name):
                        try:
                            return int(name.split("_")[-1])
                        except:
                            return 0
                    all_submit_specs.sort(key=lambda s: (abs(get_difficulty_num(s.name) - difficulty), get_difficulty_num(s.name)))
                    submit_result_function_name = all_submit_specs[0].name
                else:
                    # Final fallback to default
                    submit_result_function_name = expected_name
        
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
            f"Submit result format (JSON Schema): {json.dumps(submit_result_format, ensure_ascii=False, indent=2)}\n"
            f"The submit_result_format above specifies the EXACT structure and keys required for the answer output.\n"
            f"Allowed tool names (EXACT): {json.dumps(allowed_tool_names, ensure_ascii=False)}\n"
            f"Tools (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"You MUST call tools['{submit_result_function_name}'](<your_answer>) to submit the final answer.\n"
            f"DO NOT use any other submit_result function name. The function name '{submit_result_function_name}' is REQUIRED and matches the current difficulty level.\n"
            f"IMPORTANT: Use positional argument (not keyword argument). The function signature is: {submit_result_function_name}(result: ...)\n"
            f"Example: tools['{submit_result_function_name}'](your_data_structure)\n"
            f"Your code MUST end with: return tools['{submit_result_function_name}'](<your_answer>)\n"
        )

        # Extract common tool rules shared by solution and verification
        common_tool_rules = self._build_common_tool_rules()

        solution_base = (
            "You are writing agent-authored Python code for a sandboxed environment.\n"
            "Return ONLY JSON with key: solution.\n\n"
            "Rules for `solution`:\n"
            "- Must define exactly: def solve(tools):\n"
            "- You CANNOT define any other functions, classes, or imports.\n"
            "  All logic must be written directly inside the solve function body.\n"
            "- You do NOT have direct data access; rely only on tool outputs.\n"
            f"- Must call at least TWO different data tools from the allowed tool names (exclude {submit_result_function_name}).\n"
            f"{common_tool_rules}"
            "- Use exact keys from tool outputs (check output_keys if provided). Do not invent keys.\n"
            "- Handle nested structures and empty results by trying alternative queries or other tools.\n"
            "- Most tool calls must include meaningful parameters (aim for 70%+).\n"
            "  For get_* tools: always pass at least one filter (heading_filter, keyword, search_term, etc.).\n"
            "  For list_* tools: call first to see allowed values, then use concrete values.\n"
            f"- FINAL LINE MUST BE: return tools['{submit_result_function_name}'](answer)\n"
            f"- Use EXACTLY '{submit_result_function_name}'.\n"
            "- The answer MUST exactly match the submit_result_format schema provided above.\n"
            "  - If submit_result_format is an object, answer must be a dict with EXACTLY the keys specified in 'properties'.\n"
            "  - If submit_result_format is an array, answer must be a list of dicts, each dict matching the 'items' schema.\n"
            "  - Use the EXACT key names from the schema. Do not change or abbreviate key names.\n"
            "  - Fill in the values based on the task content and tool outputs.\n"
            "- Do NOT convert answer to string. Pass dict/list directly. answer MUST NEVER be None.\n"
            "- submit_result is a pass-through wrapper (no validation). Build answer matching the submit_result_format schema exactly.\n\n"
            f"{common_context}"
        )

        verification_base = (
            "You are writing agent-authored Python code for a sandboxed environment.\n"
            "Return ONLY JSON with key: verification.\n\n"
            "Rules for `verification`:\n"
            "- Must define: def verify(tools, answer):\n"
            "- Deterministic; may import json, re.\n"
            "- Must NOT be trivial (no unconditional pass/return True).\n"
            "- Must be exception-safe: DO NOT raise. Return False on any unexpected condition or exception.\n"
            "- Must handle answer is None / missing keys / None values without throwing.\n"
            "- If you define helpers like safe_get, accept default= as a keyword to avoid key/arg confusion.\n"
            "- Must ALWAYS return a boolean OR a dict with keys like {passed: bool, message: str, details: ...}.\n"
            "- If returning a dict, include a short failure message to help repairs.\n"
            "- If answer is wrapped by submit_result (keys like status/message/submitted_data), extract and verify the wrapped payload.\n"
            "- Do NOT check data types, key names, or structure format.\n"
            "  The submit_result_format schema is already enforced - you should assume the answer has the correct structure.\n"
            "  Focus ONLY on verifying the VALUES (content correctness), not the format.\n"
            "- You MUST call data tools to cross-check and verify the content correctness of the answer VALUES.\n"
            "  - Extract the actual values from the answer (assuming correct structure).\n"
            "  - Call appropriate data tools to retrieve the actual data from the source.\n"
            "  - Compare the answer values with the actual data to verify correctness.\n"
            "  - For example, if the answer claims a specific value exists, call tools to verify it actually exists in the data.\n"
            "  - Verify that extracted values match what's in the data sources.\n"
            "  - Use at least one tool call to verify content correctness.\n"
            "- Do not assume anything about the solution logic; validate using tool outputs only.\n"
            f"{common_tool_rules}"
            "- Use exact keys from tool outputs (check output_keys if provided). Do not invent keys.\n"
            "- The submit_result_format schema is provided above for reference, but you should NOT validate structure/format.\n"
            "  Only verify that the VALUES in the answer are correct according to the actual data.\n\n"
            f"{common_context}"
        )
        repair_guidance: str | None = None
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
                f"- If repairing solution, ensure answer matches the submit_result_format described in task content and still call {submit_result_function_name} exactly once.\n"
                "- If repairing both, rewrite both solve() and verify(); do not reuse the previous logic.\n"
                "- Remember: submit_result_format is part of the task content above.\n"
            )

        def _repair_guidance_block(label: str) -> str:
            if not repair_guidance:
                return ""
            if target and label not in {target, "both"}:
                return ""
            return (
                "\nRepair guidance (must follow):\n"
                f"{repair_guidance}\n"
                "Apply the required changes and avoid repeating the previous approach.\n"
            )

        if regen_solution or not solution_code:
            last_err = ""
            for attempt in range(1, 4):
                # Put error message at the beginning if retrying
                error_prefix = ""
                if last_err:
                    error_prefix = f"PREVIOUS ATTEMPT FAILED - Fix this error:\n{last_err}\n\n"
                
                prompt = error_prefix + solution_base
                prompt += _repair_context("solution")
                if previous_solution:
                    prompt += f"\nPrevious solution:\n{previous_solution}\n"
                prompt += _repair_guidance_block("solution")

                self.logger.info(f"LLM call: Generating solution code (attempt {attempt}/3)")
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
                if submit_result_function_name not in called:
                    last_err = f"solution must call {submit_result_function_name}"
                    ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
                    continue
                used_data = sorted(set(called) - {submit_result_function_name})
                if len(used_data) < 2:
                    last_err = f"solution must call >=2 different data tools; used={used_data}"
                    ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
                    continue
                invented = sorted(set(called) - set(allowed_tool_names))
                if invented:
                    last_err = f"solution calls tools not in allowed list: {invented}"
                    ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
                    continue
                # Check that most tool calls have parameters (exclude submit_result)
                try:
                    tree = ast.parse(sol_code)
                    tool_calls_with_params = 0
                    tool_calls_without_params = 0
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Call):
                            func = node.func
                            tool_name = None
                            # Extract tool name
                            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "tools":
                                tool_name = func.attr
                            elif isinstance(func, ast.Subscript) and isinstance(func.value, ast.Name) and func.value.id == "tools":
                                if isinstance(func.slice, ast.Constant) and isinstance(func.slice.value, str):
                                    tool_name = func.slice.value
                                elif isinstance(func.slice, ast.Str):
                                    tool_name = func.slice.s
                            # Skip submit_result from the check
                            if tool_name and tool_name != submit_result_function_name:
                                has_args = len(node.args) > 0
                                has_kwargs = len(node.keywords) > 0
                                # Count calls with at least one keyword argument as having params
                                # For positional args, check if they're not just empty dict/list
                                has_meaningful_params = has_kwargs
                                if has_args and not has_kwargs:
                                    # Check if args contain non-empty structures
                                    for arg in node.args:
                                        if isinstance(arg, ast.Dict) and len(arg.keys) == 0:
                                            continue  # Empty dict, skip
                                        elif isinstance(arg, ast.List) and len(arg.elts) == 0:
                                            continue  # Empty list, skip
                                        else:
                                            has_meaningful_params = True
                                            break
                                if has_meaningful_params:
                                    tool_calls_with_params += 1
                                else:
                                    tool_calls_without_params += 1
                    total_data_tool_calls = tool_calls_with_params + tool_calls_without_params
                    if total_data_tool_calls > 2:  # Only check if there are enough calls to matter
                        params_ratio = tool_calls_with_params / total_data_tool_calls if total_data_tool_calls > 0 else 0
                        # Require at least 50% of tool calls to have parameters
                        if params_ratio < 0.5:
                            last_err = f"Too many tool calls without parameters: {tool_calls_without_params}/{total_data_tool_calls} calls have no args/kwargs (need >=50% with params, got {params_ratio:.1%})"
                            ctx.add_step({"type": "agent_solution_validation_failed", "attempt": attempt, "error": last_err})
                            continue
                except Exception as exc:
                    # If AST parsing fails, skip this check but log
                    self.logger.debug(f"Failed to check tool call parameters: {exc}")

                if f"return tools['{submit_result_function_name}']" not in sol_code and f'return tools["{submit_result_function_name}"]' not in sol_code:
                    last_err = f"solution must end with: return tools['{submit_result_function_name}'](answer)"
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
                    last_err = f"solution must end with a return of {submit_result_function_name} call (parse_check_failed: {exc})"
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
                # Put error message at the beginning if retrying
                error_prefix = ""
                if last_err:
                    error_prefix = f"PREVIOUS ATTEMPT FAILED - Fix this error:\n{last_err}\n\n"
                
                prompt = error_prefix + verification_base
                prompt += _repair_context("verification")
                if previous_verification:
                    prompt += f"\nPrevious verification:\n{previous_verification}\n"
                prompt += _repair_guidance_block("verification")

                self.logger.info(f"LLM call: Generating verification code (attempt {attempt}/3)")
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
        # Format submit_result_format for display in task_content
        format_description = ""
        if isinstance(fmt, dict):
            if isinstance(fmt.get("properties"), dict):
                props = fmt["properties"]
                format_description = f"The output must be a JSON object with the following structure: "
                prop_descriptions = []
                for key, prop_def in props.items():
                    prop_type = prop_def.get("type", "unknown")
                    prop_desc = prop_def.get("description", "")
                    if prop_desc:
                        prop_descriptions.append(f"'{key}' ({prop_type}): {prop_desc}")
                    else:
                        prop_descriptions.append(f"'{key}' ({prop_type})")
                format_description += ", ".join(prop_descriptions) + ". "
                if fmt.get("required"):
                    format_description += f"Required fields: {', '.join(fmt['required'])}. "
        requirements = (
            f"Use at least two different tools from the available toolset (do not name them) and ensure "
            f"the output strictly follows the submit_result_format. {format_description}"
            f"Focus on entries explicitly tied to {topic_label}, and exclude generic or unrelated records. If "
            "results are sparse, explain how you selected fallback entries from the available information sources."
        )
        if content:
            if content[-1] not in ".!?":
                content += "."
            return f"{content} {requirements}"
        return f"Use the available tools and information to solve a {topic_label} task. {requirements}"

    def _extract_file_info_from_profile(
        self, data_profile: dict[str, Any] | None, sandbox_dir: Any | None = None
    ) -> list[dict[str, Any]]:
        """Extract file information (path, title, summary) from data_profile.
        
        Args:
            data_profile: Data profile containing file entries
            sandbox_dir: Optional sandbox directory to read files from
            
        Returns:
            List of dicts with keys: path, title (optional), summary (optional)
            
        Raises:
            RuntimeError: If no data files are found (file_info_list is empty)
        """
        file_info_list: list[dict[str, Any]] = []
        if not data_profile:
            raise RuntimeError("No data_profile provided. Cannot extract file information.")
        
        json_entries = data_profile.get("json", [])
        if not isinstance(json_entries, list):
            raise RuntimeError("data_profile.json is not a list. Cannot extract file information.")
        
        for entry in json_entries:
            if not isinstance(entry, dict):
                continue
            path = entry.get("path")
            if not isinstance(path, str) or not path.strip():
                continue
            
            file_info: dict[str, Any] = {"path": path}
            has_title = False
            has_summary = False
            
            # Try to read title and summary from the file if sandbox_dir is provided
            if sandbox_dir:
                try:
                    from pathlib import Path
                    file_path = Path(sandbox_dir) / path
                    if file_path.exists():
                        with open(file_path, 'r', encoding='utf-8') as f:
                            file_data = json.load(f)
                        if isinstance(file_data, dict):
                            if "title" in file_data:
                                file_info["title"] = file_data["title"]
                                has_title = True
                            if "summary" in file_data:
                                file_info["summary"] = file_data["summary"]
                                has_summary = True
                except Exception as e:
                    # If reading fails, log warning
                    self.logger.warning(f"Failed to read file {path}: {e}")
            
            # Warn if title or summary is missing
            if not has_title:
                self.logger.warning(f"File {path} is missing 'title' field")
            if not has_summary:
                self.logger.warning(f"File {path} is missing 'summary' field")
            
            file_info_list.append(file_info)
        
        # Raise error if no files found
        if not file_info_list:
            raise RuntimeError("No valid data files found in data_profile. Cannot generate task.")
        
        return file_info_list

    def _build_common_tool_rules(self) -> str:
        """Build common tool usage rules shared by solution and verification prompts."""
        return (
            "- Tools are callables: use keyword args, e.g., tools['get_x'](param=value).\n"
            "- Data tools return {'result': ...}; unwrap with out['result'] or out.get('result', out).\n"
            "- Respect tool output_schema exactly. Use tool parameters as defined in schemas.\n"
        )

    def _generate_submit_result_function(
        self,
        submit_result_format: dict[str, Any],
        difficulty: int,
        ctx: TaskContext,
        max_tokens: int = 2000,
        previous_errors: list[str] | None = None,
    ) -> str:
        """Generate submit_result function based on submit_result_format and difficulty level.
        
        Returns:
            Python code string for the submit_result function with difficulty-specific name.
        """
        function_name = f"submit_result_difficulty_{difficulty}"
        format_json = json.dumps(submit_result_format, ensure_ascii=False, indent=2)
        
        # Extract schema information for validation
        schema_type = submit_result_format.get("type", "object")
        properties = submit_result_format.get("properties", {})
        required_fields = submit_result_format.get("required", [])
        
        # Build description from schema
        description_parts = []
        if schema_type == "object" and properties:
            description_parts.append("Submit the final answer matching the required schema.")
            if required_fields:
                description_parts.append(f"Required fields: {', '.join(required_fields)}.")
        elif schema_type == "array":
            description_parts.append("Submit the final answer as a list matching the required schema.")
        else:
            description_parts.append("Submit the final answer payload.")
        
        description = " ".join(description_parts)
        
        # Build type hint suggestion
        type_hint = "dict[str, Any]"
        if schema_type == "array":
            type_hint = "list[dict[str, Any]]"
        elif schema_type == "object":
            type_hint = "dict[str, Any]"
        
        # Build validation code suggestions based on schema
        validation_instructions = []
        if schema_type == "object" and properties:
            if required_fields:
                validation_instructions.append(
                    f"# Validate required fields\n"
                    f"if not isinstance(result, dict):\n"
                    f"    raise ValueError(f'result must be a dict, got {{type(result).__name__}}')\n"
                    f"missing_fields = [f for f in {required_fields!r} if f not in result]\n"
                    f"if missing_fields:\n"
                    f"    raise ValueError(f'Missing required fields: {{missing_fields}}')\n"
                )
            # Add type checking for properties if they have type information
            type_checks = []
            for prop_name, prop_schema in properties.items():
                prop_type = prop_schema.get("type")
                if prop_type == "array":
                    type_checks.append(
                        f"if '{prop_name}' in result and not isinstance(result['{prop_name}'], list):\n"
                        f"    raise ValueError(f\"Field '{prop_name}' must be a list, got {{type(result.get('{prop_name}')).__name__}}\")"
                    )
                elif prop_type == "object":
                    type_checks.append(
                        f"if '{prop_name}' in result and not isinstance(result['{prop_name}'], dict):\n"
                        f"    raise ValueError(f\"Field '{prop_name}' must be a dict, got {{type(result.get('{prop_name}')).__name__}}\")"
                    )
                elif prop_type in ["string", "number", "integer", "boolean"]:
                    python_type_map = {
                        "string": "str",
                        "number": "(int, float)",
                        "integer": "int",
                        "boolean": "bool"
                    }
                    python_type = python_type_map.get(prop_type, "Any")
                    if prop_type == "number":
                        type_checks.append(
                            f"if '{prop_name}' in result and not isinstance(result['{prop_name}'], (int, float)):\n"
                            f"    raise ValueError(f\"Field '{prop_name}' must be a number, got {{type(result.get('{prop_name}')).__name__}}\")"
                        )
                    else:
                        type_checks.append(
                            f"if '{prop_name}' in result and not isinstance(result['{prop_name}'], {python_type}):\n"
                            f"    raise ValueError(f\"Field '{prop_name}' must be {prop_type}, got {{type(result.get('{prop_name}')).__name__}}\")"
                        )
            if type_checks:
                validation_instructions.append("\n".join(type_checks))
        elif schema_type == "array":
            validation_instructions.append(
                f"# Validate result is a list\n"
                f"if not isinstance(result, list):\n"
                f"    raise ValueError(f'result must be a list, got {{type(result).__name__}}')"
            )
        
        validation_code_example = "\n    ".join(validation_instructions) if validation_instructions else ""
        
        # Build error context if previous attempts failed
        error_context = ""
        if previous_errors:
            error_context = "\n\nPREVIOUS ATTEMPTS FAILED:\n"
            for err in previous_errors:
                error_context += f"- {err}\n"
            error_context += "\nPlease fix the issues mentioned above and ensure the function is complete and valid.\n"
        
        prompt = (
            "You are generating a Python function that submits and persists a result to a JSON file.\n"
            "Return ONLY JSON with key: submit_result_function.\n\n"
            "REQUIREMENTS:\n"
            "- The function must be decorated with @mcp.tool()\n"
            f"- The function name must be: {function_name}\n"
            f"- The function must accept a single parameter: result: {type_hint}\n"
            "- The function MUST include validation code that checks the result structure matches the schema below\n"
            "- The function must validate required fields exist (if any)\n"
            "- The function must validate field types match the schema (using isinstance checks)\n"
            "- The function must raise ValueError with descriptive messages if validation fails\n"
            "- The function must write the validated result to BASE_DIR / 'submitted_result.json'\n"
            "- The function must use json.dump() with indent=2, ensure_ascii=False, default=str\n"
            "- The function must return the result unchanged (pass-through)\n"
            "- Include type hints for the parameter\n"
            "- The function code must be COMPLETE - do not truncate or leave code unfinished\n"
            f"{error_context}"
            f"Submit result format (JSON Schema):\n{format_json}\n\n"
            f"Function description for @mcp.tool decorator:\n{description}\n\n"
            f"{'Validation code example (you should adapt this based on the actual schema):' if validation_code_example else ''}\n"
            f"{validation_code_example if validation_code_example else ''}\n\n"
            "Output ONLY the Python function code (no markdown, no explanations).\n"
            "The function should include:\n"
            f"1. @mcp.tool(description='{description}')\n"
            f"2. def {function_name}(result: {type_hint}):\n"
            "3. Validation code that checks result structure against the schema\n"
            "4. Code to write validated result to BASE_DIR / 'submitted_result.json'\n"
            "5. Return the result unchanged\n"
        )
        
        self.logger.info(f"LLM call: Generating submit_result function (difficulty {difficulty})")
        raw = self.llm.simple_complete(prompt, temperature=0.3, max_tokens=max_tokens)
        ctx.add_step({"type": "submit_result_function_generated", "difficulty": difficulty, "content": raw})
        
        extracted = self._extract_json(raw) or {}
        function_code = self._sanitize_python_code(extracted.get("submit_result_function", ""))
        
        if not function_code or f"def {function_name}" not in function_code:
            raise RuntimeError(
                f"Failed to generate {function_name} function based on submit_result_format. "
                f"LLM returned: {raw[:500]}"
            )
        
        # Validate that the function name matches
        if f"def {function_name}" not in function_code:
            raise RuntimeError(
                f"Generated function name does not match expected {function_name}. "
                f"Generated code: {function_code[:500]}"
            )
        
        # Remove any import mcp statements - mcp is already defined in tools.py
        lines = function_code.splitlines()
        cleaned_lines = []
        for line in lines:
            stripped = line.strip()
            # Skip import mcp or from mcp imports (mcp is already defined in tools.py)
            if stripped.startswith("import mcp") or stripped.startswith("from mcp"):
                continue
            # Also skip duplicate BASE_DIR definitions
            if stripped.startswith("BASE_DIR =") and "Path(__file__)" in stripped:
                continue
            cleaned_lines.append(line)
        
        function_code = "\n".join(cleaned_lines).strip()
        if function_code and not function_code.endswith("\n"):
            function_code += "\n"
        
        return function_code

    def _save_task_description_immediately(
        self, task_id: str, task_content: str, submit_result_format: dict[str, Any], difficulty: int
    ) -> None:
        """Save task description and submit_result_format immediately after generation."""
        if not hasattr(self, "writer") or self.writer is None:
            return
        
        try:
            from pathlib import Path
            import json
            
            task_dir = self.writer.task_dir(task_id, self.agent_type)
            # Save to runs directory, same as solution and verification
            runs_dir = task_dir / "_sandbox" / "runs" / f"difficulty-{difficulty}"
            runs_dir.mkdir(parents=True, exist_ok=True)
            
            # Save task_content as JSON string (it's a string, not a dict)
            description_target = runs_dir / "task_description.json"
            description_target.write_text(json.dumps(task_content, ensure_ascii=False), encoding="utf-8")
            self.logger.info(f"Task description saved to {description_target} (difficulty: {difficulty})")
            
            # Save submit_result_format as JSON dict
            format_target = runs_dir / "submit_result_format.json"
            format_target.write_text(json.dumps(submit_result_format, ensure_ascii=False, indent=2), encoding="utf-8")
            self.logger.info(f"Submit result format saved to {format_target} (difficulty: {difficulty})")
        except Exception as e:
            self.logger.warning(f"Failed to save task description/format: {e}", exc_info=True)


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
        sandbox_dir: Any | None = None,
    ) -> TaskPackage:
        # Extract file information from data_profile
        file_info_list = self._extract_file_info_from_profile(data_profile, sandbox_dir)

        # Log task generation start
        # Initial task always has difficulty 1 (request.difficulty is the total number of difficulty levels)
        initial_difficulty = 1
        self.logger.info(f"Generating initial task with difficulty {initial_difficulty} for topic: {request.topic}")

        # Step 1: Generate task_content only (without submit_result_format)
        prompt = (
            "You are a task generator.\n"
            "Create ONE verifiable baseline task that will be progressively strengthened in later refinement rounds.\n"
            "Focus on defining a clear, well-scoped initial task aligned with the available data.\n"
            "Return ONLY JSON with keys: task_title, task_content.\n\n"
            "REQUIREMENTS for task_content (MUST be detailed and comprehensive):\n"
            "- Must be at least 400-500 characters long with 5-8 complete sentences.\n"
            "- Must clearly describe: (1) What data to query/retrieve, (2) What operations to perform, "
            "(3) What specific criteria or filters to apply, (4) What information should be extracted, "
            "(5) Any edge cases or special considerations.\n"
            "- Do NOT mention the data source, tool names, or file/module names.\n"
            "- Do NOT describe the output format structure - that will be defined separately.\n"
            "- Must include concrete examples of what fields/values to look for in the data.\n"
            "- Must specify any data quality requirements (e.g., exclude nulls, filter by date ranges).\n"
            "- Base task_content on the provided data file information (titles and summaries).\n"
            "- Write in clear, actionable language that leaves no ambiguity about the task requirements.\n\n"
            f"Topic: {request.topic}\n"
            f"Available data files: {json.dumps(file_info_list, ensure_ascii=False)}\n"
            + "Generate task_title and task_content based on the data files above.\n"
        )

        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        self.logger.info("LLM call: Generating task content (task_title and task_content)")
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
                sandbox_dir=sandbox_dir,
            )

        if not isinstance(extracted, dict):
            raise RuntimeError("Task proposal failed: invalid JSON response.")

        task_title = (extracted.get("task_title") or "").strip()
        task_content = (extracted.get("task_content") or "").strip()
        
        if not task_content or len(task_content) < 200:
            raise RuntimeError("Task proposal failed: task_content too short or missing.")
        
        ctx.add_step({"type": "task_content_generated", "content": {"task_title": task_title, "task_content": task_content}})
        self.writer.record_steps(task_id, self.agent_type, ctx.history)
        
        # Step 2: Generate submit_result_format separately with explicit schema
        format_prompt = (
            "You are defining the output format schema for a task.\n"
            "Based on the task description, design a clear and explicit JSON schema for the answer output.\n"
            "Return ONLY JSON with key: submit_result_format.\n\n"
            "REQUIREMENTS for submit_result_format:\n"
            "- Must be a valid JSON Schema.\n"
            "- Can be a single dict: {\"type\": \"object\", \"properties\": {...}, \"required\": [...]}\n"
            "- Can be a list of dicts: {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {...}}}\n"
            "- MUST explicitly define all keys (field names) in the output structure.\n"
            "- For each key, specify its type (string, number, array, object, etc.) but leave the value open.\n"
            "- Include 'required' array to specify which keys are mandatory.\n"
            "- Each property should have a 'type' and optionally 'description' explaining what the value should contain.\n"
            "- The schema should match what information needs to be extracted based on the task description.\n"
            "- Be specific about the structure: if nested objects, define their properties too.\n\n"
            f"Topic: {request.topic}\n"
            f"Task title: {task_title}\n"
            f"Task content: {task_content}\n"
            f"Available data files: {json.dumps(file_info_list, ensure_ascii=False)}\n"
            + "Design submit_result_format that specifies the exact keys and structure for the answer, "
            + "based on what information should be extracted according to the task content.\n"
        )
        
        self.logger.info("LLM call: Generating submit_result_format schema")
        format_raw = self.llm.simple_complete(format_prompt, temperature=0.35, max_tokens=max_tokens)
        ctx.add_step({"type": "submit_result_format_proposed", "content": format_raw})
        self.writer.record_steps(task_id, self.agent_type, ctx.history)
        
        format_extracted = self._extract_json(format_raw)
        if isinstance(format_extracted, list) and format_extracted and isinstance(format_extracted[0], dict):
            format_extracted = format_extracted[0]
        
        if not isinstance(format_extracted, dict) or "submit_result_format" not in format_extracted:
            # Fallback to default format
            submit_result_format = {
            "type": "object",
            "properties": {"result": {"type": "array", "items": {"type": "object"}}},
            "required": ["result"],
        }
            self.logger.warning("Failed to extract submit_result_format, using default")
        else:
            submit_result_format = format_extracted["submit_result_format"]
        
        # Validate submit_result_format structure
        if not isinstance(submit_result_format, dict):
            submit_result_format = {
                "type": "object",
                "properties": {"result": {"type": "array", "items": {"type": "object"}}},
                "required": ["result"],
            }

        # Initial task always has difficulty 1 (request.difficulty is the total number of difficulty levels)
        ctx.current_difficulty = 1
        
        # Save task_description.json and submit_result_format.json immediately after generation
        self._save_task_description_immediately(task_id, task_content, submit_result_format, ctx.current_difficulty)
        
        # Generate and add submit_result function to tools.py immediately after submit_result_format is created
        submit_result_function_name = f"submit_result_difficulty_{ctx.current_difficulty}"
        if sandbox_dir:
            from pathlib import Path
            tools_path = Path(sandbox_dir) / "tools.py"
            if tools_path.exists():
                try:
                    tools_code = tools_path.read_text(encoding="utf-8")
                    # Check if submit_result function for this difficulty already exists
                    if f"def {submit_result_function_name}" not in tools_code:
                        # Generate submit_result function based on submit_result_format with retry
                        max_tokens = getattr(ctx.request, "max_tokens", 10000)
                        
                        submit_func_code = None
                        last_error = None
                        error_history = []
                        for attempt in range(3):
                            try:
                                submit_func_code = self._generate_submit_result_function(
                                    submit_result_format, ctx.current_difficulty, ctx, max_tokens=max_tokens,
                                    previous_errors=error_history if error_history else None
                                )
                                break
                            except Exception as e:
                                last_error = e
                                error_msg = str(e)
                                error_history.append(f"Attempt {attempt + 1} error: {error_msg}")
                                self.logger.warning(
                                    f"Attempt {attempt + 1}/3 failed to generate {submit_result_function_name}: {e}",
                                    exc_info=True
                                )
                        
                        if submit_func_code:
                            # Append the function to tools.py
                            tools_code = tools_code.rstrip() + "\n\n" + submit_func_code.strip() + "\n"
                            tools_path.write_text(tools_code, encoding="utf-8")
                            self.logger.info(f"Added {submit_result_function_name} function to tools.py after submit_result_format generation")
                        else:
                            raise RuntimeError(
                                f"Failed to generate {submit_result_function_name} after 3 attempts. Last error: {last_error}"
                            ) from last_error
                except Exception as e:
                    self.logger.warning(f"Failed to add {submit_result_function_name} to tools.py: {e}", exc_info=True)
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

        # Add submit_result tool AFTER generating task description and submit_result_format
        # This ensures the tool is only available after the format is defined
        # Use difficulty-specific function name
        from agent_gem.core.task_schema import ToolSpec
        from typing import Any
        
        def submit_result(result: Any) -> Any:
            """Submit the final answer payload."""
            return result
        
        submit_result_spec = ToolSpec.from_function(
            submit_result,
            name=submit_result_function_name,
            description="Submit the final answer payload.",
            meta={"system": True, "difficulty": ctx.current_difficulty},
        )
        if not any(spec.name == submit_result_function_name for spec in tool_specs):
            tool_specs = tool_specs + [submit_result_spec]

        # Agent-authored code (no framework synthesis).
        # Note: task_content and submit_result_format are provided separately to solution and verification
        # Retry only this function if it fails (timeout/error), not the entire _propose_task
        solution_code = None
        verification_code = None
        solution_retry_count = 0
        max_solution_retries = 3
        
        while solution_retry_count < max_solution_retries:
            try:
                solution_code, verification_code = self._generate_agent_solution_and_verification(
                    request=request,
                    ctx=ctx,
                    task_content=task_content,
                    submit_result_format=submit_result_format,
                    tool_specs=tool_specs,
                    records=records,
                    tool_selftest=tool_selftest,
                    submit_result_function_name=submit_result_function_name,
                )
                break  # Success, exit retry loop
            except Exception as e:
                solution_retry_count += 1
                if solution_retry_count >= max_solution_retries:
                    # Final attempt failed, re-raise the exception
                    self.logger.error(f"Failed to generate solution and verification after {max_solution_retries} attempts: {e}")
                    raise
                else:
                    self.logger.warning(
                        f"Failed to generate solution and verification (attempt {solution_retry_count}/{max_solution_retries}), retrying: {e}"
                    )

        ctx.add_step(
            {"type": "solution_and_verification_code", "content": {"solution_code": solution_code, "verification_code": verification_code}}
        )
        pkg = TaskPackage(
            task=TaskDefinition(
                task_id=task_id,
                task_title=task_title,
                task_content=(task_content if len(task_content) >= 10 else f"Solve a task about {request.topic}."),
                submit_result_format=submit_result_format,
                tool_set=tool_specs,
                evaluation_criteria=EvaluationCriteria(),
                difficulty_level=ctx.current_difficulty,
            ),
            agent_type=self.agent_type,
            solution=solution_code,
            verification=verification_code,
            metadata={"topic": request.topic},
        )

        return pkg

    def _refine_task(
        self,
        previous: TaskPackage,
        records: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        ctx: TaskContext,
        target_difficulty: int,
        tool_selftest: dict[str, Any] | None = None,
        data_profile: dict[str, Any] | None = None,
        sandbox_dir: Any | None = None,
    ) -> TaskPackage:
        # Use the same data file abstraction as initial task proposal
        file_info_list: list[dict[str, Any]] = []
        try:
            file_info_list = self._extract_file_info_from_profile(data_profile, sandbox_dir)
        except Exception:
            # If extraction fails, fall back to empty list but still proceed using previous task
            file_info_list = []
        previous_difficulty = previous.task.difficulty_level
        # Update ctx.current_difficulty to target_difficulty for this refinement round
        ctx.current_difficulty = target_difficulty
        # Log task refinement start
        self.logger.info(
            f"Refining task: previous difficulty {previous_difficulty} -> target difficulty {target_difficulty}"
        )
        # Step 1: Generate refined task_content only (without submit_result_format)
        prompt = (
            "You are refining an existing task to increase its difficulty while keeping it verifiable.\n"
            "Your goal is to create a MORE DIFFICULT and MORE DIVERSE task compared to the previous one.\n"
            "Return ONLY JSON with keys: task_content.\n\n"
            "REQUIREMENTS for refined task_content:\n"
            "- Increase the difficulty_level significantly compared to the previous task.\n"
            f"  The previous task had difficulty_level {previous_difficulty}, and you should target difficulty_level {target_difficulty}.\n"
            "- Promote DIVERSITY - you are NOT required to preserve the same goal or data focus as the previous task.\n"
            "  You can explore DIFFERENT data files, DIFFERENT aspects of the topic, or DIFFERENT types of analysis.\n"
            "- You can use DIFFERENT data files from the available set, not necessarily the same ones as the previous task.\n"
            "- Increase difficulty by: (1) Adding more complex operations, (2) Requiring cross-file or multi-step analysis, "
            "(3) Introducing stricter criteria or edge cases, (4) Requiring deeper reasoning or synthesis.\n"
            "- Do NOT mention tool names, data source names, or file/module names in task_content.\n"
            "- Do NOT describe the output format structure - that will be defined separately.\n"
            "- Base task_content on the available data files, using the provided file information (titles and summaries).\n"
            "- You can combine multiple data files, require comparisons, or focus on different aspects than the previous task.\n"
            "- Any new conditions or filters MUST be consistent with what the available data can support.\n"
            "- Write in clear, actionable language that leaves no ambiguity about the refined task requirements.\n\n"
            f"Topic: {ctx.request.topic if ctx and ctx.request else previous.task.task_title}\n"
            f"Available data files: {json.dumps(file_info_list, ensure_ascii=False)}\n"
            f"Previous task (JSON): {json.dumps(previous.as_payload(), ensure_ascii=False)}\n"
        )
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        self.logger.info(f"LLM call: Refining task content (difficulty {ctx.current_difficulty})")
        raw = self.llm.simple_complete(prompt, temperature=0.7, max_tokens=max_tokens)
        extracted = self._extract_json(raw) or {}

        tool_names = [spec.name for spec in tool_specs if spec.name != self._SUBMIT_RESULT_TOOL]
        task_content = self._sanitize_task_content(
            str(extracted.get("task_content") or previous.task.task_content).strip(),
            tool_names + [self._SUBMIT_RESULT_TOOL],
        )
        task_content = self._sanitize_task_content(task_content, tool_names + [self._SUBMIT_RESULT_TOOL])

        if not task_content or len(task_content) < 200:
            task_content = previous.task.task_content or ""
        
        ctx.add_step({"type": "refined_task_content", "content": {"task_content": task_content}})
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)
        
        # Step 2: Generate submit_result_format separately with explicit schema
        format_prompt = (
            "You are defining the output format schema for a refined task.\n"
            "Based on the refined task description, design a clear and explicit JSON schema for the answer output.\n"
            "Return ONLY JSON with key: submit_result_format.\n\n"
            "REQUIREMENTS for submit_result_format:\n"
            "- Must be a valid JSON Schema.\n"
            "- Can be a single dict: {\"type\": \"object\", \"properties\": {...}, \"required\": [...]}\n"
            "- Can be a list of dicts: {\"type\": \"array\", \"items\": {\"type\": \"object\", \"properties\": {...}}}\n"
            "- MUST explicitly define all keys (field names) in the output structure.\n"
            "- For each key, specify its type (string, number, array, object, etc.) but leave the value open.\n"
            "- Include 'required' array to specify which keys are mandatory.\n"
            "- Each property should have a 'type' and optionally 'description' explaining what the value should contain.\n"
            "- The schema should match what information needs to be extracted based on the refined task description.\n"
            "- Be specific about the structure: if nested objects, define their properties too.\n\n"
            f"Topic: {ctx.request.topic if ctx and ctx.request else previous.task.task_title}\n"
            f"Refined task content: {task_content}\n"
            f"Previous submit_result_format: {json.dumps(previous.task.submit_result_format, ensure_ascii=False)}\n"
            f"Available data files: {json.dumps(file_info_list, ensure_ascii=False)}\n"
            + "Design submit_result_format that specifies the exact keys and structure for the answer, "
            + "based on what information should be extracted according to the refined task content.\n"
        )
        
        self.logger.info(f"LLM call: Refining submit_result_format (difficulty {ctx.current_difficulty})")
        format_raw = self.llm.simple_complete(format_prompt, temperature=0.35, max_tokens=max_tokens)
        ctx.add_step({"type": "refined_submit_result_format_proposed", "content": format_raw})
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)
        
        format_extracted = self._extract_json(format_raw)
        if isinstance(format_extracted, list) and format_extracted and isinstance(format_extracted[0], dict):
            format_extracted = format_extracted[0]
        
        if not isinstance(format_extracted, dict) or "submit_result_format" not in format_extracted:
            # Fallback to previous format
            submit_result_format = previous.task.submit_result_format
            self.logger.warning("Failed to extract refined submit_result_format, using previous format")
        else:
            submit_result_format = format_extracted["submit_result_format"]
        
        # Validate submit_result_format structure
        if not isinstance(submit_result_format, dict):
            submit_result_format = previous.task.submit_result_format

        # Save task_description.json and submit_result_format.json immediately after generation
        self._save_task_description_immediately(ctx.task_id, task_content, submit_result_format, ctx.current_difficulty)

        # Generate and add submit_result function to tools.py immediately after submit_result_format is created
        submit_result_function_name = f"submit_result_difficulty_{ctx.current_difficulty}"
        if sandbox_dir:
            from pathlib import Path
            tools_path = Path(sandbox_dir) / "tools.py"
            if tools_path.exists():
                try:
                    tools_code = tools_path.read_text(encoding="utf-8")
                    # Check if submit_result function for this difficulty already exists
                    if f"def {submit_result_function_name}" not in tools_code:
                        # Generate submit_result function based on submit_result_format with retry
                        max_tokens = getattr(ctx.request, "max_tokens", 10000)
                        
                        submit_func_code = None
                        last_error = None
                        error_history = []
                        for attempt in range(3):
                            try:
                                submit_func_code = self._generate_submit_result_function(
                                    submit_result_format, ctx.current_difficulty, ctx, max_tokens=max_tokens,
                                    previous_errors=error_history if error_history else None
                                )
                                break
                            except Exception as e:
                                last_error = e
                                error_msg = str(e)
                                error_history.append(f"Attempt {attempt + 1} error: {error_msg}")
                                self.logger.warning(
                                    f"Attempt {attempt + 1}/3 failed to generate {submit_result_function_name}: {e}",
                                    exc_info=True
                                )
                        
                        if submit_func_code:
                            # Append the function to tools.py
                            tools_code = tools_code.rstrip() + "\n\n" + submit_func_code.strip() + "\n"
                            tools_path.write_text(tools_code, encoding="utf-8")
                            self.logger.info(f"Added {submit_result_function_name} function to tools.py after refined submit_result_format generation")
                        else:
                            raise RuntimeError(
                                f"Failed to generate {submit_result_function_name} after 3 attempts. Last error: {last_error}"
                            ) from last_error
                except Exception as e:
                    self.logger.warning(f"Failed to add {submit_result_function_name} to tools.py: {e}", exc_info=True)

        # Add submit_result tool AFTER generating task description and submit_result_format
        # This ensures the tool is only available after the format is defined
        # Use difficulty-specific function name
        from agent_gem.core.task_schema import ToolSpec
        from typing import Any
        
        def submit_result(result: Any) -> Any:
            """Submit the final answer payload."""
            return result
        
        submit_result_spec = ToolSpec.from_function(
            submit_result,
            name=submit_result_function_name,
            description="Submit the final answer payload.",
            meta={"system": True, "difficulty": ctx.current_difficulty},
        )
        if not any(spec.name == submit_result_function_name for spec in tool_specs):
            tool_specs = tool_specs + [submit_result_spec]

        # Note: task_content and submit_result_format are provided separately to solution and verification
        solution, verification = self._generate_agent_solution_and_verification(
            request=ctx.request,
            ctx=ctx,
            task_content=task_content,
            submit_result_format=submit_result_format,
            tool_specs=tool_specs,
            records=records,
            tool_selftest=tool_selftest if isinstance(tool_selftest, dict) else None,
            previous_solution=previous.solution or "",
            previous_verification=previous.verification or "",
            submit_result_function_name=submit_result_function_name,
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
