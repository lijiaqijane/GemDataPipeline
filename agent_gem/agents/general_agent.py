from __future__ import annotations

import builtins
import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from agent_gem.core.task_schema import (
    EvaluationCriteria,
    TaskDefinition,
    TaskPackage,
    ToolSpec,
)
from agent_gem.writer import TaskWriter
from agent_gem.sandbox import SandboxExecutor
from agent_gem.tools import BashTool, JsonRecordsQueryTool, PythonRunnerTool, SearchTool

from .base import BaseAgent, TaskContext

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest

logger = logging.getLogger(__name__)


BUILTIN_FUNCTIONS = set(dir(builtins))


class GeneralAgent(BaseAgent):
    agent_type = "general_agent"
    description = "Automatic environment-synthesis agent that creates diverse, verifiable tasks with a growing toolset."

    _RECORDS_FILENAME = "records.json"

    def _configure_sandbox(self, sandbox: SandboxExecutor):
        sandbox.register_tool(
            BashTool(workdir=sandbox.sandbox_dir, timeout_s=sandbox.timeout_s)
        )
        sandbox.register_tool(SearchTool(cache_path=sandbox.search_cache_path))
        sandbox.register_tool(
            PythonRunnerTool(workdir=sandbox.sandbox_dir, timeout_s=sandbox.timeout_s)
        )
        sandbox.set_tool_call_callback(self._record_tool_call)

    def generate(self, request: GenerationRequest) -> Optional[TaskPackage]:
        if not request.topic:
            request.topic = "general task"

        task_id = str(uuid.uuid4())
        ctx = TaskContext(task_id=task_id, request=request)

        sandbox_dir = Path(self.writer.task_dir(task_id, self.agent_type), "_sandbox")
        sandbox = SandboxExecutor(sandbox_dir=sandbox_dir)
        self._configure_sandbox(sandbox)

        logger.info(
            f"Generating task: {task_id}, topic: {request.topic}, path: {self.writer.task_dir(task_id, self.agent_type)}"
        )

        records = self._seed_database(request.topic, ctx, sandbox)

        task_tool_specs = self._synthesize_task_tools(request.topic, records, ctx)
        self.writer.record_steps(task_id, self.agent_type, ctx.history)
        self._register_task_tools(task_tool_specs, sandbox, ctx)

        package = self._propose_task(task_id, request, records, task_tool_specs, ctx)
        package = self._ensure_substantive_task(task_tool_specs, package, ctx)
        package = self._ensure_valid(request, package, ctx, sandbox)
        self.writer.record_steps(task_id, self.agent_type, ctx.history)

        for round_idx in range(2, max(1, request.max_refine_rounds) + 1):
            target = min(int(request.difficulty), round_idx)
            refined = self._refine_task(
                previous=package,
                records=records,
                tool_specs=task_tool_specs,
                ctx=ctx,
                target_difficulty=target,
            )
            refined = self._ensure_substantive_task(task_tool_specs, refined, ctx=ctx)
            package = self._ensure_valid(request, refined, ctx, sandbox)
            self.writer.record_steps(task_id, self.agent_type, ctx.history)

        if request.persist_result and self.writer is not None:
            self.writer.record_steps(
                task_id,
                self.agent_type,
                [step.to_payload() for step in ctx.history],
                extra={
                    "topic": request.topic,
                    "difficulty": request.difficulty,
                    "records_count": len(records),
                    "task_tools": [spec.name for spec in task_tool_specs],
                },
            )

        return package

    def _record_tool_call(self, record: Any, ctx: TaskContext) -> None:
        try:
            message = {
                "type": "tool_call",
                "tool": getattr(record, "tool", None),
                "input": getattr(record, "tool_input", None),
                "output": getattr(record, "tool_output", None),
                "error": getattr(record, "error", None),
                "duration_s": getattr(record, "duration_s", None),
            }
            ctx.add_step(
                message,
                request_id=f"tool_{getattr(record, 'call_id', '')}",
            )
        except Exception:
            logger.debug("Failed to record tool call step", exc_info=True)

    def _seed_database(
        self, topic: str, ctx: TaskContext, sandbox: SandboxExecutor
    ) -> list[dict[str, Any]]:
        """Collect topic-relevant records and write them into the sandbox database."""
        search_hits: list[dict[str, str]] = []
        result = sandbox.execute(
            "search", f"{topic} structured dataset examples", max_results=5
        )
        if isinstance(result, list):
            search_hits = [row for row in result if isinstance(row, dict)]

        ctx.add_step(
            {
                "type": "seed_database",
                "topic": topic,
                "search_hits": search_hits[:3],
            }
        )

        prompt = (
            "You are a data curation assistant.\n"
            "Create a list of diverse, non-duplicative records for the topic.\n"
            "Return ONLY a JSON array; each item must be an object with fields: title (string), summary (string).\n"
            f"Topic: {topic}\n"
            f"Search hits (JSON): {json.dumps(search_hits, ensure_ascii=False)}"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.4, max_tokens=700)
        extracted = self._extract_json(raw)

        records: list[dict[str, Any]] = []
        if isinstance(extracted, list):
            records = [row for row in extracted if isinstance(row, dict)]
        elif isinstance(extracted, dict):
            records = [extracted]

        records = [
            {
                "title": str(row.get("title") or topic).strip(),
                "summary": str(row.get("summary") or "").strip(),
            }
            for row in records
        ]

        records = [row for row in records if row["summary"] != ""]

        ctx.add_step(
            {
                "type": "seed_database",
                "topic": topic,
                "records": records,
            }
        )
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)

        return records

    def _synthesize_task_tools(
        self, topic: str, records: list[dict[str, Any]], ctx: TaskContext
    ) -> list[ToolSpec]:
        """Generate schema-level tool specs (ToolSpec) for the task toolset."""
        prompt = (
            "Design a task toolset for an RL environment.\n"
            "Return Python code defining 3-5 tools, each decorated with @mcp.tool(...).\n"
            "Constraints:\n"
            "- Tool names must be unique, snake_case, and NOT 'bash'/'search'/'python_runner'.\n"
            "- Each tool MUST accept (query: str, max_results: int = 5).\n"
            "- Each tool MUST return list[dict].\n"
            "- Function bodies must not do any I/O; use 'raise RuntimeError(\"tool spec only\")'.\n"
            "Only output a single Python code block.\n"
            f"Topic: {topic}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
            "\nExample:\n"
            "```python\n"
            '@mcp.tool(name="find_records", description="Find records by keyword.")\n'
            "def find_records(query: str, max_results: int = 5) -> list[dict]:\n"
            "    # Implementation of the tool\n"
            "```\n"
        )

        raw = self.llm.simple_complete(prompt, temperature=0.6, max_tokens=900)
        ctx.add_step({"type": "tool_synthesis", "content": raw})
        specs = self._extract_mcp_tools(raw)

        specs = [
            spec
            for spec in specs
            if spec.name not in {"bash", "search", "python_runner"}
        ]
        ctx.add_step(
            {
                "type": "tool_synthesis",
                "tool_count": len(specs),
                "tools": [spec.model_dump() for spec in specs],
            }
        )
        return specs

    def _register_task_tools(
        self,
        tool_specs: list[ToolSpec],
        sandbox: SandboxExecutor,
        ctx: TaskContext,
    ) -> None:
        records_path = sandbox.sandbox_dir / self._RECORDS_FILENAME
        for spec in tool_specs:
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
            }
        )
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)

    def _propose_task(
        self,
        task_id: str,
        request: GenerationRequest,
        records: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        ctx: TaskContext,
        retry: int = 0,
    ) -> TaskPackage:
        tool_list = [
            {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.parameters,
            }
            for spec in tool_specs
        ]

        prompt = (
            "You are a task generator.\n"
            "Create ONE challenging but automatically verifiable task. Involve concrete task details such that the task"
            " solution can be validated with an verification function. Think ultrahard and be creative, the task should"
            " be challenging enough such that an AI agent cannot easily solve.\n"
            "Return ONLY JSON with keys:\n"
            "task_title: The task name\n"
            "task_content: The detailed, comprehensive task description\n"
            "submit_result_format: The expected structured output of a typed solution schema, it can be a list or a dictionary\n"
            "difficulty_level: Rate in 1-5, how challenging do you consider the current task\n"
            f"Topic: {request.topic}\n"
            f"Tool list (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
            "Example:\n"
            "```json\n"
            "{\n"
            '"task_title": "Trip Planning",\n'
            '"task_content": "I’m planning a three-day trip starting from Hangzhou, and I need help creating an itinerary '
            "from October 1st to October 3rd, 2025. A few important requirements: I don’t want to repeat "
            "any cities, hotels, attractions, or restaurants during the entire trip. Also, please make sure that "
            "every hotel, restaurant, and attraction you recommend is actually located in the city where "
            "I’ll be staying that day. One more thing about the second day - I’m trying to be smart about "
            "my budget. If I end up booking a luxury hotel that costs 800 CNY or more per night, then I "
            "need to be more careful with other expenses: my total spending on both restaurants (lunch "
            "and dinner) should stay under 350 CNY, both restaurants should be rated at least 4.0 stars, "
            "and the afternoon attraction ticket needs to be less than 120 CNY. If the hotel on day 2 is in "
            "the mid-to-high range (500-800 CNY), then I have a bit more flexibility - I just need to make "
            "sure at least one of my restaurant choices is rated 4.0 or higher, and the attraction ticket should "
            "be below 180 CNY. For more affordable hotels (200-500 CNY range), I only need to ensure "
            'that at least one restaurant has a rating of 3.2 or above. Can you help me put together this itinerary?",\n'
            '"submit_result_format": "[\n'
            '{ "time": "2025-10-01", "city": "cite_name", "hotel": "hotel_name", "afternoon_restaurant": "restaurant_name", "afternoon_attraction": "attraction_name", "evening_restaurant": "restaurant_name" },\n'
            '{ "time": "2025-10-02", "city": "cite_name", "hotel": "hotel_name", "afternoon_restaurant": "restaurant_name", "afternoon_attraction": "attraction_name", "evening_restaurant": "restaurant_name" },\n'
            '{ "time": "2025-10-03", "city": "cite_name", "hotel": "hotel_name", "afternoon_restaurant": "restaurant_name", "afternoon_attraction": "attraction_name", "evening_restaurant": "restaurant_name" }\n'
            ']",\n'
            '"difficulty_level": 3\n'
            "}\n"
            "```"
        )

        raw = self.llm.simple_complete(prompt, temperature=0.65, max_tokens=1100)
        ctx.add_step(
            {
                "type": "task_proposed",
                "content": raw,
            }
        )
        self.writer.record_steps(task_id, self.agent_type, ctx.history)

        extracted = self._extract_json(raw)
        if extracted is None and retry < 3:
            logger.info(
                f"Failed to extract valid json from task output. Retry {retry+1}..."
            )
            return self._propose_task(
                task_id, request, records, tool_specs, ctx, retry=retry + 1
            )

        task_content = extracted.get("task_content").strip()

        submit_result_format = extracted.get("submit_result_format")

        ctx.current_difficulty = int(
            extracted.get("difficulty_level") or ctx.current_difficulty
        )
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
        prompt = (
            "You are a task solver and verifier. Based on the task, submit_result_format, and a tool set, \n"
            "create solution and verication code for the task. The solution function is restricted to invoking tool"
            "functions or performing logical computations, and cannot call other functions or directly"
            "access the database. The results produced by the solution function should be validated by the verification"
            " function\n"
            "Return python code blocks for solution code and verification code.\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. solution_code MUST ONLY CALL TOOLS using tool_name1('query') or tool_name2('query').\n"
            "2. Call at least 2 different tools and combine their results into a structured output.\n"
            "3. Do NOT return trivial results like 'list(tools.keys())' or just tool names.\n"
            "4. It can only access data via tools (no direct DB access).\n"
            "5. verification_code must define verify(answer) and return bool.\n"
            "6. The verification must check content/shape, not just type.\n\n"
            f"Topic: {request.topic}\n"
            f"Task Content:\n{task_content}\n"
            f"All tools:\n{json.dumps(tool_list, ensure_ascii=False)}\n"
            f"Submit Result Format:\n{json.dumps(submit_result_format, ensure_ascii=False)}\n"
            "Example solution pattern:\n"
            "```python\n"
            "def solve():\n"
            "    data1 = tool1('query1')\n"
            "    data2 = tool2('query2')\n"
            "    return {'result': data1 + data2}\n"
            "```"
            "Example verify pattern:\n"
            "```python\n"
            "def verify(answer):\n"
            "    # Code to verify if the answer follows submit result format\n"
            "    # Code to verify if the answer is correct\n"
            "    return False\n"
            "```"
        )
        raw_code = self.llm.simple_complete(prompt, temperature=0.65, max_tokens=1100)
        ctx.add_step(
            {
                "type": "solution_and_verification_code",
                "content": raw_code,
            }
        )
        code_blocks = self._extract_code_blocks(raw_code)
        solution_code, verification_code = None, None
        for block in code_blocks:
            if "solve" in block:
                solution_code = block
            if "verify" in block:
                verification_code = block
        ctx.add_step(
            {
                "type": "solution_and_verification_code",
                "content": {
                    "solution_code": solution_code,
                    "verification_code": verification_code,
                },
            }
        )
        pkg = TaskPackage(
            task=TaskDefinition(
                task_id=task_id,
                task_title=request.topic,
                task_content=(
                    task_content
                    if len(task_content) >= 10
                    else f"Solve a task about {request.topic}."
                ),
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
    ) -> TaskPackage:
        tool_list = [
            {"name": spec.name, "description": spec.description} for spec in tool_specs
        ]
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
        raw = self.llm.simple_complete(prompt, temperature=0.7, max_tokens=1200)
        extracted = self._extract_json(raw)
        data: dict[str, Any] = extracted if isinstance(extracted, dict) else {}

        task_content = str(
            data.get("task_content") or previous.task.task_content
        ).strip()
        submit_result_format = data.get(
            "submit_result_format", previous.task.submit_result_format
        )
        if isinstance(submit_result_format, str):
            submit_result_format = {"type": submit_result_format}

        solution = (
            data.get("solution")
            if isinstance(data.get("solution"), str)
            else previous.solution
        )
        verification = (
            data.get("verification")
            if isinstance(data.get("verification"), str)
            else previous.verification
        )

        pkg = TaskPackage(
            task=TaskDefinition(
                task_id=previous.task.task_id,
                task_title=previous.task.task_title,
                task_content=(
                    task_content
                    if len(task_content) >= 10
                    else previous.task.task_content
                ),
                submit_result_format=submit_result_format,
                tool_set=tool_specs,
                evaluation_criteria=previous.task.evaluation_criteria,
                difficulty_level=int(
                    data.get("difficulty_level") or ctx.current_difficulty
                ),
            ),
            solution=solution,
            verification=verification,
            agent_type=self.agent_type,
            metadata={
                **previous.metadata,
                "topic": previous.task.task_title,
                "refined": "true",
            },
        )
        ctx.add_step(
            {
                "type": "task_refined",
                "task_title": pkg.task.task_title,
                "difficulty_level": pkg.task.difficulty_level,
            }
        )
        return pkg

    def _ensure_valid(
        self,
        request: GenerationRequest,
        package: TaskPackage,
        ctx: TaskContext,
        sandbox: SandboxExecutor,
    ) -> TaskPackage:
        if not request.validate or sandbox is None:
            return package

        last_error = ""
        for attempt in range(1, request.max_validation_rounds + 1):
            allowed_tools = {spec.name for spec in package.task.tool_set}
            called_tools = self._extract_tool_calls(package.solution)
            if len(called_tools) < 2:
                last_error = (
                    "solution must call at least 2 different tools; "
                    f"called={sorted(called_tools)} allowed={sorted(allowed_tools)}"
                )
                package = self._repair_package(request, package, ctx, error=last_error)
                continue
            missing_tools = called_tools - allowed_tools
            if missing_tools:
                last_error = (
                    "solution calls tools not in the declared tool_set; "
                    f"missing={sorted(missing_tools)} allowed={sorted(allowed_tools)}"
                )
                package = self._repair_package(request, package, ctx, error=last_error)
                continue

            run = sandbox.run_task(package)
            ctx.add_step(
                {
                    "type": "task_validation",
                    "attempt": attempt,
                    "verified": run.verified,
                    "error": run.error,
                    "verification_error": run.verification_error,
                }
            )
            if run.verified is True:
                return package

            last_error = (
                run.error or run.verification_error or "unknown validation failure"
            )
            package = self._repair_package(request, package, ctx, error=last_error)
            package = self._ensure_substantive_task(package.task.tool_set, package, ctx)

        logger.warning(
            "Task failed validation after %d attempt(s): %s",
            request.max_validation_rounds,
            last_error,
        )
        return package

    def _repair_package(
        self,
        request: GenerationRequest,
        package: TaskPackage,
        ctx: TaskContext,
        *,
        error: str,
    ) -> TaskPackage:
        tool_list = [
            {"name": spec.name, "description": spec.description}
            for spec in package.task.tool_set
        ]
        prompt = (
            "Repair the provided solution/verification so that verify(tools, solve(tools)) returns True.\n"
            "Return ONLY JSON with keys: solution, verification.\n"
            "Constraints:\n"
            "- Keep solve(tools) and verify(tools, answer) signatures unchanged.\n"
            "- Only call tools from the provided tool list.\n"
            "- Do not use file/network access.\n"
            f"Topic: {request.topic}\n"
            f"Tools (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"Task (title/content): {package.task.task_title} / {package.task.task_content}\n"
            f"Current solution:\n{package.solution}\n"
            f"Current verification:\n{package.verification}\n"
            f"Observed error:\n{error}\n"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.55, max_tokens=1100)
        extracted = self._extract_json(raw)
        data: dict[str, Any] = extracted if isinstance(extracted, dict) else {}

        solution = (
            data.get("solution")
            if isinstance(data.get("solution"), str)
            else package.solution
        )
        verification = (
            data.get("verification")
            if isinstance(data.get("verification"), str)
            else package.verification
        )
        repaired = package.copy(
            update={"solution": solution, "verification": verification}
        )
        ctx.add_step({"type": "task_repaired", "error": error})
        return repaired

    def _ensure_substantive_task(
        self,
        tool_specs: list[ToolSpec],
        package: TaskPackage,
        ctx: TaskContext,
    ) -> TaskPackage:
        allowed = {spec.name for spec in tool_specs}
        called = self._extract_tool_calls(package.solution)

        if called and called.issubset(allowed) and len(called) >= 2:
            return package

        ctx.add_step(
            {
                "type": "task_quality_gate",
                "reason": "insufficient_tool_usage",
                "called_tools": sorted(called),
                "allowed_tools": sorted(allowed),
            }
        )
        return package

    @staticmethod
    def _extract_tool_calls(solution_code: str) -> set[str]:
        if not solution_code:
            return set()
        tool_calls = re.findall(
            r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(",
            solution_code,
        )
        called_tools = [fn for fn in tool_calls if fn not in BUILTIN_FUNCTIONS]
        return set(called_tools)

    @staticmethod
    def _extract_code_blocks(raw: str) -> list[str]:
        """
        Extract all code blocks from the given string. Code blocks are assumed to be enclosed in triple backticks.

        Args:
        raw (str): The input string containing potential code blocks.

        Returns:
        list: A list of code blocks (strings) found in the input.
        """

        text = (raw or "").strip()
        if not text:
            return []

        # Regex pattern to match code blocks enclosed by triple backticks (``` ... ```)
        pattern = r"```python(.*?)```"

        # Use re.DOTALL to match across multiple lines and re.search to extract all matches
        code_blocks = re.findall(pattern, raw, re.DOTALL)

        return [code.strip() for code in code_blocks]

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
        code_blocks = GeneralAgent._extract_code_blocks(text)

        mcp_tools: list[ToolSpec] = []

        for code in code_blocks:
            # Regular expression to capture function definitions with @mcp.tool decorators
            mcp_pattern = r"(@mcp\.tool\((.*?)\)\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*\s*\([^\)]*\))\s*(->\s*[a-zA-Z_][a-zA-Z0-9_\[\],]*)?\s*:([\s\S]+?))(?=\n\s*@mcp|\Z)"

            # Match all decorators and the corresponding function definitions
            matches = re.findall(mcp_pattern, code, re.DOTALL)

            for match in matches:
                decorator_params = match[1]
                function_signature = match[2]

                # Generate a ToolSpec from the function string
                # We'll assume that the function string is valid and can be passed to from_function_string
                function_string = match[0]

                try:
                    tool = ToolSpec.from_function_string(function_string)
                    mcp_tools.append(tool)
                except ValueError as e:
                    logger.error(
                        f"Error creating ToolSpec for function {function_signature}: {e}"
                    )

        return mcp_tools


# Helper function to parse the decorator parameters (name="...", description="...", etc.)
def parse_decorator_args(decorator_params: str) -> Dict[str, str]:
    # Regex to match key-value pairs inside the decorator
    param_pattern = r'(\w+)="([^"]*)"'

    params = {}
    matches = re.findall(param_pattern, decorator_params)
    for key, value in matches:
        params[key] = value

    return params
