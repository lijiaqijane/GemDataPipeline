from __future__ import annotations

import json
import re
import textwrap
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

    def _plan_tool_usage(
        self,
        request: "GenerationRequest",
        tool_specs: list[ToolSpec],
        ctx: TaskContext,
        submit_result_format: dict[str, Any],
        records: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        """Plan which tools to call (and with what queries) for solve(tools)."""
        tool_list = [
            {"name": spec.name, "description": spec.description}
            for spec in tool_specs
        ]
        tool_names = [spec["name"] for spec in tool_list]
        
        # If we have fewer than 2 tools, we can't create a valid plan
        if len(tool_names) < 2:
            ctx.add_step({"type": "tool_plan", "content": "not_enough_tools"})
            return [{"tool": name, "query": request.topic or "task"} for name in tool_names[:2]]
        
        prompt = (
            "You are planning tool usage for a solution function solve(tools).\n"
            "Given the task topic, tools, submit_result_format, and a database sample,\n"
            "decide which tools to call and with what natural-language queries.\n\n"
            "Return ONLY JSON with key `tool_plan` as a list of objects:\n"
            "[{\"tool\": <tool_name>, \"query\": <string>}].\n"
            "CRITICAL REQUIREMENTS:\n"
            "- You MUST choose at least two different tools from the allowed list.\n"
            "- Tools must be from the provided list of names only; do NOT invent new tool names.\n"
            "- Tool names must match EXACTLY (case-sensitive) from the allowed list.\n"
            "- Queries should reflect the task topic and be concrete and different for each tool.\n"
            "- Each query should be a meaningful string (not empty, at least 3 characters).\n\n"
            f"Topic: {request.topic}\n"
            f"Allowed tools (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"Allowed tool names (use these EXACT names): {json.dumps(tool_names, ensure_ascii=False)}\n"
            f"Submit Result Format (JSON): {json.dumps(submit_result_format, ensure_ascii=False)}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
        )
        
        normalized: list[dict[str, str]] = []
        plan_max_tokens = getattr(ctx.request, "max_tokens", 10000)
        for attempt in range(3):
            raw = self.llm.simple_complete(
                prompt, temperature=0.5 + 0.1 * attempt, max_tokens=plan_max_tokens
            )
            ctx.add_step({"type": "tool_plan_raw", "attempt": attempt + 1, "content": raw})
            data = self._extract_json(raw) or {}
            plan = data.get("tool_plan") or []
            seen: set[str] = set()
            if isinstance(plan, list):
                for item in plan:
                    if not isinstance(item, dict):
                        continue
                    name = str(item.get("tool") or "").strip()
                    query = str(item.get("query") or "").strip()
                    if name in tool_names and name not in seen and query and len(query) >= 3:
                        normalized.append({"tool": name, "query": query})
                        seen.add(name)
            if len(normalized) >= 2:
                break
        
        if len(normalized) < 2:
            seen = {item["tool"] for item in normalized}
            for name in tool_names:
                if name not in seen:
                    tool_desc = next(
                        (spec.description for spec in tool_specs if spec.name == name),
                        request.topic or "task"
                    )
                    query = f"{request.topic or 'task'} {tool_desc[:50]}".strip()
                    normalized.append({"tool": name, "query": query})
                    seen.add(name)
                if len(normalized) >= 2:
                    break
        
        ctx.add_step({"type": "tool_plan", "content": normalized})
        return normalized

    def _build_solution_from_plan(
        self,
        request: "GenerationRequest",
        ctx: TaskContext,
        task_content: str,
        submit_result_format: dict[str, Any],
        tool_plan: list[dict[str, str]],
    ) -> str:
        """Build the full solve(tools) function using a fixed tool call header and LLM-combined body."""
        header_lines: list[str] = ["def solve(tools):"]
        var_names: list[str] = []
        for idx, item in enumerate(tool_plan):
            tool_name = item.get("tool", "")
            query = item.get("query", "")
            var_name = f"tool_{idx}_result"
            var_names.append(var_name)
            if isinstance(query, str):
                safe_tool_name = tool_name.replace("'", "\\'")
                header_lines.append(
                    f"    {var_name} = tools['{safe_tool_name}']({repr(query)})"
                )
            else:
                header_lines.append(
                    f"    {var_name} = tools['{tool_name}']({json.dumps(query, ensure_ascii=False)})"
                )
            header_lines.append(f"    if {var_name} is None:")
            header_lines.append(f"        {var_name} = []")
            header_lines.append(f"    elif not isinstance({var_name}, list):")
            header_lines.append(f"        {var_name} = [{var_name}]")
            header_lines.append(f"    if not {var_name}:")
            header_lines.append(f"        {var_name} = [{{}}]")
        header_lines.append(
            "    # Combine tool_*_result variables into the final answer that matches submit_result_format."
        )

        if self._looks_like_record_summary_format(submit_result_format):
            return self._build_structured_solution(
                header_lines=header_lines,
                var_names=var_names,
                request=request,
                submit_result_format=submit_result_format,
            )
        if isinstance(submit_result_format, list):
            return self._build_list_solution(
                header_lines=header_lines,
                var_names=var_names,
                request=request,
                submit_result_format=submit_result_format,
            )

        var_list_str = ", ".join(var_names)
        combo_prompt = (
            "You are writing the body of a Python function solve(tools).\n"
            "CRITICAL: The following variables are ALREADY DEFINED and contain tool call results:\n"
            f"  {var_list_str}\n"
            "You MUST use ALL of these variables in your code. Do NOT ignore them.\n"
            "You MUST construct and return a final answer that strictly matches submit_result_format.\n"
            "STRICT CONSTRAINTS:\n"
            "- Do NOT call tools[...] or tools.name(...) again - the tool calls are already done.\n"
            "- Do NOT define the function header (def solve(tools):).\n"
            "- You MUST reference and use the tool_*_result variables in your code.\n"
            "- Simply write Python statements that process these variables.\n"
            "- End with a single `return <expression>` that matches submit_result_format.\n"
            "- Each tool_*_result is a list of dictionaries returned by the tool.\n"
            "- You can iterate over them, filter them, combine them, etc.\n\n"
            f"Topic: {request.topic}\n"
            f"Task Content:\n{task_content}\n"
            f"Submit Result Format (JSON):\n{json.dumps(submit_result_format, ensure_ascii=False)}\n"
            "Return ONLY one Python code block containing the body statements (no `def` line).\n"
        )
        
        body = None
        body_max_tokens = getattr(ctx.request, "max_tokens", 10000)
        for attempt in range(3):
            raw_body = self.llm.simple_complete(
                combo_prompt, temperature=0.5 + 0.1 * attempt, max_tokens=body_max_tokens
            )
            ctx.add_step({"type": "solution_body_only", "attempt": attempt + 1, "content": raw_body})
            blocks = BaseAgent._extract_code_blocks(raw_body)
            body_source = blocks[0] if blocks else raw_body
            body = BaseAgent._strip_code_fences(body_source)
            body = re.sub(r"^```.*$", "", body, flags=re.MULTILINE)
            uses_tool_result = any(var_name in body for var_name in var_names)
            if uses_tool_result:
                break

        if body and not any(var_name in body for var_name in var_names):
            combine_code = f"    # Combine all tool results\n"
            combine_code += f"    combined_results = []\n"
            for var_name in var_names:
                combine_code += f"    if {var_name}:\n"
                combine_code += f"        combined_results.extend({var_name} if isinstance({var_name}, list) else [{var_name}])\n"
            combine_code += f"    # Process combined_results and return answer matching submit_result_format\n"
            body = combine_code + body

        indented_body_lines = []
        for line in (body or "").splitlines():
            if line.strip():
                indented_body_lines.append(f"    {line}")
            else:
                indented_body_lines.append("")
        solution_code = "\n".join(header_lines + indented_body_lines) + "\n"
        solution_code = re.sub(r"^\s*```.*$", "", solution_code, flags=re.MULTILINE)
        solution_code = solution_code.replace("```", "")
        
        if not any(var_name in solution_code for var_name in var_names):
            fallback_lines = []
            for var_name in var_names:
                fallback_lines.append(f"    if not {var_name}:")
                fallback_lines.append(f"        {var_name} = []")
            fallback_lines.append("    # Combine results")
            fallback_lines.append(f"    result = {var_names[0]} if {var_names[0]} else []")
            if len(var_names) > 1:
                fallback_lines.append(f"    for other_result in {var_names[1:]}:")
                fallback_lines.append("        if other_result:")
                fallback_lines.append("            result.extend(other_result if isinstance(other_result, list) else [other_result])")
            fallback_lines.append("    return {'result': result}")
            solution_code = "\n".join(header_lines + fallback_lines) + "\n"

        valid_solution, validation_err = CodeValidator.validate_solution_code(solution_code)
        if not valid_solution:
            ctx.add_step(
                {
                    "type": "solution_validation_failed",
                    "error": validation_err,
                    "fallback_used": True,
                }
            )
            solution_code = self._build_fallback_solution(
                header_lines=header_lines,
                var_names=var_names,
                submit_result_format=submit_result_format,
            )

        return solution_code

    def _build_fallback_solution(
        self,
        *,
        header_lines: list[str],
        var_names: list[str],
        submit_result_format: dict[str, Any],
    ) -> str:
        if isinstance(submit_result_format, list):
            return self._build_list_solution(
                header_lines=header_lines,
                var_names=var_names,
                request=None,
                submit_result_format=submit_result_format,
            )
        return self._build_structured_solution(
            header_lines=header_lines,
            var_names=var_names,
            request=None,
            submit_result_format=submit_result_format,
        )

    def _looks_like_record_summary_format(self, submit_result_format: dict[str, Any]) -> bool:
        if not isinstance(submit_result_format, dict):
            return False
        keys = set(submit_result_format.keys())
        required = {"fetched_records", "summarized_key_points", "cross_referenced_entities"}
        return required.issubset(keys)

    def _build_list_solution(
        self,
        *,
        header_lines: list[str],
        var_names: list[str],
        request: "GenerationRequest | None",
        submit_result_format: dict[str, Any] | list[Any],
    ) -> str:
        topic = (request.topic if request else None) or "task"
        template = submit_result_format[0] if isinstance(submit_result_format, list) and submit_result_format else {}
        template_keys = list(template.keys()) if isinstance(template, dict) else []
        fallback_keys = template_keys or ["rank", "name", "category", "arrondissement", "price_euro", "rating"]
        max_items = max(3, len(fallback_keys))

        body_lines: list[str] = []
        body_lines.append("    # Deterministic list builder aligned to submit_result_format")
        body_lines.append(f"    _topic = {topic!r}")
        body_lines.append("    combined = []")
        for var in var_names:
            body_lines.append(f"    if isinstance({var}, list):")
            body_lines.append(f"        combined.extend([i for i in {var} if isinstance(i, dict)])")
        body_lines.append("    if not combined:")
        body_lines.append("        combined = [{}]")
        body_lines.append("    ranked = []")
        body_lines.append("    for idx, rec in enumerate(combined, 1):")
        body_lines.append("        score_val = 0.0")
        body_lines.append("        for key in ('rank', 'score', 'popularity_score', 'popularity', 'rating', 'popularity_score_num'):")
        body_lines.append("            val = rec.get(key)")
        body_lines.append("            if isinstance(val, (int, float)):")
        body_lines.append("                score_val = float(val)")
        body_lines.append("                break")
        body_lines.append("        ranked.append((score_val, idx, rec))")
        body_lines.append("    ranked.sort(reverse=True)")
        body_lines.append(f"    max_items = {max_items}")
        body_lines.append("    output = []")
        body_lines.append("    for rank_idx, (_, _, rec) in enumerate(ranked[:max_items], 1):")
        body_lines.append("        item = {}")
        body_lines.append("        name_val = rec.get('name') or rec.get('title') or rec.get('attraction_name') or rec.get('hotel_name') or f'Item {rank_idx}'")
        body_lines.append("        category_val = rec.get('category') or rec.get('type') or rec.get('cuisine') or 'Unknown'")
        body_lines.append("        arr_val = rec.get('arrondissement') or rec.get('arrondissement_number') or rec.get('district') or rec.get('area') or ''")
        body_lines.append("        price_val = rec.get('price_euro')")
        body_lines.append("        if price_val in (None, ''):")
        body_lines.append("            price_val = rec.get('price') or rec.get('price_per_night') or rec.get('cost') or 0")
        body_lines.append("        try:")
        body_lines.append("            price_num = float(price_val)")
        body_lines.append("        except Exception:")
        body_lines.append("            price_num = 0.0")
        body_lines.append("        rating_val = rec.get('rating') or rec.get('score') or rec.get('popularity_score') or rec.get('rank') or 0")
        body_lines.append("        try:")
        body_lines.append("            rating_num = float(rating_val)")
        body_lines.append("        except Exception:")
        body_lines.append("            rating_num = 0.0")
        body_lines.append(f"        for key in {fallback_keys!r}:")
        body_lines.append("            if key == 'rank':")
        body_lines.append("                item[key] = rank_idx")
        body_lines.append("            elif key == 'name':")
        body_lines.append("                item[key] = name_val")
        body_lines.append("            elif key == 'category':")
        body_lines.append("                item[key] = category_val")
        body_lines.append("            elif key == 'arrondissement':")
        body_lines.append("                item[key] = arr_val")
        body_lines.append("            elif key in ('price', 'price_euro', 'price_in_euros'):")
        body_lines.append("                item[key] = price_num")
        body_lines.append("            elif key == 'rating':")
        body_lines.append("                item[key] = rating_num")
        body_lines.append("            else:")
        body_lines.append("                val = rec.get(key)")
        body_lines.append("                item[key] = val if val not in (None, '') else ''")
        body_lines.append("        if 'rank' not in item:")
        body_lines.append("            item['rank'] = rank_idx")
        body_lines.append("        if 'name' not in item:")
        body_lines.append("            item['name'] = name_val")
        body_lines.append("        if 'category' not in item:")
        body_lines.append("            item['category'] = category_val")
        body_lines.append("        if 'arrondissement' not in item:")
        body_lines.append("            item['arrondissement'] = arr_val")
        body_lines.append("        if 'price_euro' not in item:")
        body_lines.append("            item['price_euro'] = price_num")
        body_lines.append("        if 'rating' not in item and 'score' not in item:")
        body_lines.append("            item['rating'] = rating_num")
        body_lines.append("        output.append(item)")
        body_lines.append("    return output")

        return "\n".join(header_lines + body_lines) + "\n"

    def _build_structured_solution(
        self,
        *,
        header_lines: list[str],
        var_names: list[str],
        request: "GenerationRequest | None",
        submit_result_format: dict[str, Any],
    ) -> str:
        topic = (request.topic if request else None) or "task"
        max_records = 5

        body_lines: list[str] = []
        body_lines.append("    # Deterministic structured builder to match submit_result_format")
        body_lines.append(f"    _topic = {topic!r}")
        body_lines.append("    combined = []")
        for var in var_names:
            body_lines.append(f"    if isinstance({var}, list):")
            body_lines.append(f"        combined.extend([i for i in {var} if isinstance(i, dict)])")
        body_lines.append("    fetched = []")
        body_lines.append("    for rec in combined:")
        body_lines.append("        title = str(rec.get('title') or rec.get('name') or rec.get('id') or '').strip()")
        body_lines.append("        raw_summary = rec.get('summary') or rec.get('description') or rec.get('highlights') or ''")
        body_lines.append("        if isinstance(raw_summary, list):")
        body_lines.append("            raw_summary = '; '.join(str(x) for x in raw_summary)")
        body_lines.append("        summary = str(raw_summary).strip()")
        body_lines.append("        url = str(rec.get('url') or rec.get('link') or rec.get('href') or '')")
        body_lines.append("        source = str(rec.get('source') or rec.get('data_type') or rec.get('category') or 'generated')")
        body_lines.append("        fetched.append({")
        body_lines.append("            'title': title or 'Untitled record',")
        body_lines.append("            'summary': summary or f\"Auto-generated summary for {title or 'record'}\",")
        body_lines.append("            'url': url,")
        body_lines.append("            'source': source")
        body_lines.append("        })")
        body_lines.append("    if not fetched:")
        body_lines.append("        fetched = [{")
        body_lines.append("            'title': f\"Placeholder for {_topic}\",")
        body_lines.append("            'summary': f\"Auto-generated summary for {_topic}\",")
        body_lines.append("            'url': '',")
        body_lines.append("            'source': 'generated'")
        body_lines.append("        }]")
        body_lines.append(f"    fetched = fetched[:{max_records}]")
        body_lines.append("    summarized_key_points = '; '.join([r.get('summary','') for r in fetched if r.get('summary')])")
        body_lines.append("    if not summarized_key_points.strip():")
        body_lines.append("        summarized_key_points = f\"Auto-generated summary for {_topic}\"")
        body_lines.append("    related_titles = [r.get('title','') for r in fetched if r.get('title')]")
        body_lines.append("    cross_refs = [{")
        body_lines.append("        'query_used': _topic,")
        body_lines.append("        'related_titles': related_titles")
        body_lines.append("    }]")
        body_lines.append("    answer = {")
        body_lines.append("        'fetched_records': fetched,")
        body_lines.append("        'summarized_key_points': summarized_key_points,")
        body_lines.append("        'cross_referenced_entities': cross_refs")
        body_lines.append("    }")
        body_lines.append("    return answer")

        return "\n".join(header_lines + body_lines) + "\n"

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
        ]
        tool_names = [spec.name for spec in tool_specs]

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
            fallback_title = (request.topic or "Generated Task").strip() or "Generated Task"
            fallback_content = f"Plan and analyze a task in the domain: {request.topic or 'general topic'}."
            extracted = {
                "task_title": fallback_title,
                "task_content": fallback_content,
                "submit_result_format": {
                    "type": "object",
                    "properties": {"result": {"type": "array", "items": {"type": "object"}}},
                    "required": ["result"],
                },
                "difficulty_level": ctx.current_difficulty or request.difficulty or 1,
            }

        task_content = (extracted.get("task_content") or "").strip()
        submit_result_format = extracted.get("submit_result_format") or {
            "type": "object",
            "properties": {"result": {"type": "array", "items": {"type": "object"}}},
            "required": ["result"],
        }

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

        base_prompt = (
            "You are a task solver and verifier. The system must create tasks that are challenging yet automatically verifiable. "
            "Start from the current database context, propose a task, and produce both solution and verification code in Python.\n"
            "Solution expectations: only `def solve(tools):` with inline logic and tool calls already planned.\n"
            "Verification expectations: only `def verify(tools, answer):` with deterministic checks and permissive scoring.\n"
        )

        tool_plan = self._plan_tool_usage(
            request=request,
            tool_specs=tool_specs,
            ctx=ctx,
            submit_result_format=submit_result_format,
            records=records,
        )

        solution_code = self._build_solution_from_plan(
            request=request,
            ctx=ctx,
            task_content=task_content,
            submit_result_format=submit_result_format,
            tool_plan=tool_plan,
        )

        verification_code: Optional[str] = None
        verify_prompt = (
            base_prompt
            + "\n\nNow ONLY generate the verification function `def verify(tools, answer):` "
            "as a single Python code block. Do not include the solution function."
        )
        verify_max_tokens = getattr(ctx.request, "max_tokens", 10000)
        for attempt in range(3):
            raw_verify = self.llm.simple_complete(
                verify_prompt, temperature=0.6 + 0.05 * attempt, max_tokens=verify_max_tokens
            )
            ctx.add_step({"type": "verification_code_only", "attempt": attempt + 1, "content": raw_verify})
            code_blocks = BaseAgent._extract_code_blocks(raw_verify)
            verification_code = None
            for block in code_blocks:
                cleaned = BaseAgent._strip_code_fences(block)
                if "def verify(" in cleaned and "verify(" in cleaned:
                    verification_code = cleaned
                    break

            if not verification_code:
                continue

            ver_ok, ver_err = CodeValidator.validate_verification_code(verification_code)
            if ver_ok:
                break

            ctx.add_step(
                {"type": "verification_validation_failed", "attempt": attempt + 1, "error": ver_err or "unknown error"}
            )

        if not verification_code:
            verification_code = (
                "def verify(tools, answer):\n"
                "    details = []\n"
                "    passed = isinstance(answer, dict)\n"
                "    if not passed:\n"
                "        details.append({'name': 'type', 'passed': False, 'msg': 'answer must be dict'})\n"
                "    score = 1.0 if passed else 0.0\n"
                "    return {'passed': passed, 'score': score, 'details': details}\n"
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

        solution = extracted.get("solution") if isinstance(extracted.get("solution"), str) else previous.solution
        verification = extracted.get("verification") if isinstance(extracted.get("verification"), str) else previous.verification

        pkg = TaskPackage(
            task=TaskDefinition(
                task_id=previous.task.task_id,
                task_title=previous.task.task_title,
                task_content=(task_content if len(task_content) >= 10 else previous.task.task_content),
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
