from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from agent_gem.core.task_schema import EvaluationCriteria, TaskDefinition, TaskPackage, ToolSpec
from agent_gem.database import LocalDatabase
from agent_gem.sandbox import SandboxExecutor
from agent_gem.tools import BashTool, JsonRecordsQueryTool, PythonRunnerTool, SearchTool

from .base import BaseAgent, TaskState

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest

logger = logging.getLogger(__name__)


@dataclass
class GeneralAgentTaskState(TaskState):
    records: list[dict[str, Any]]


class GeneralAgent(BaseAgent):
    agent_type = "general_agent"
    description = "Automatic environment-synthesis agent that creates diverse, verifiable tasks with a growing toolset."

    _RECORDS_FILENAME = "records.json"

    def register_tools(self) -> None:
        if self.sandbox is None:
            return
        self.sandbox.set_tool_call_callback(self._record_tool_call)

    def _configure_sandbox(self, sandbox: SandboxExecutor):
        sandbox.register_tool(BashTool(workdir=sandbox.sandbox_dir, timeout_s=sandbox.timeout_s))
        sandbox.register_tool(SearchTool(cache_path=sandbox.search_cache_path))
        sandbox.register_tool(PythonRunnerTool(workdir=sandbox.sandbox_dir, timeout_s=sandbox.timeout_s))

    def generate(self, request: GenerationRequest) -> Optional[TaskPackage]:
        topic = request.topic or "general knowledge"

        task_id = task_id or str(uuid.uuid4())
        self.task_state = GeneralAgentTaskState(task_id=task_id)
        if self.db is None:
            self.db = LocalDatabase(root=Path(self.taskdb_root))

        sandbox_dir = Path(self.taskdb_root, self.agent_type, f"task-{task_id}", "_sandbox")
        sandbox = SandboxExecutor(sandbox_dir=sandbox_dir)
        self._configure_sandbox(sandbox)

        records = self._seed_database(topic)
        self.db.record_steps(task_id, self.agent_type, self.task_state.steps)

        task_tool_specs = self._synthesize_task_tools(topic, records)
        self._register_task_tools(task_tool_specs)
        self.db.record_steps(task_id, self.agent_type, self.task_state.steps)

        package = self._propose_task(
            request,
            topic,
            records,
            task_tool_specs,
            task_title=request.topic,
            difficulty=1,
        )
        package = self._ensure_substantive_task(task_tool_specs, package)
        package = self._ensure_valid(request, package, topic=topic)

        for round_idx in range(2, max(1, request.max_refine_rounds) + 1):
            target = min(int(request.difficulty), round_idx)
            refined = self._refine_task(
                topic,
                records,
                task_tool_specs,
                previous=package,
                difficulty=target,
            )
            refined = self._ensure_substantive_task(task_tool_specs, refined)
            package = self._ensure_valid(request, refined, topic=topic)

        if request.persist_result and self.db is not None:
            self.db.record_steps(
                package,
                [step.to_payload() for step in self.task_state.steps],
                extra={
                    "topic": topic,
                    "difficulty": request.difficulty,
                    "records_count": len(records),
                    "task_tools": [spec.name for spec in task_tool_specs],
                },
            )

        return package

    def _select_task_title(self, topic: str) -> str:
        prompt = (
            "Generate a short, descriptive task title (4-80 chars). "
            'Return ONLY JSON: {"task_title": "..."}.\n'
            f"Topic: {topic}"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.4, max_tokens=80)
        extracted = self._extract_json(raw)
        title = ""
        if isinstance(extracted, dict):
            title = str(extracted.get("task_title") or "").strip()
        elif isinstance(extracted, str):
            title = extracted.strip()
        if len(title) < 4:
            title = f"{topic.title()} Task"
        return title[:80]

    def _record_tool_call(self, record: Any) -> None:
        try:
            message = {
                "type": "tool_call",
                "tool": getattr(record, "tool", None),
                "input": getattr(record, "tool_input", None),
                "output": getattr(record, "tool_output", None),
                "error": getattr(record, "error", None),
                "duration_s": getattr(record, "duration_s", None),
            }
            self.task_state.add_step(message, request_id=f"tool_{getattr(record, 'call_id', '')}")
        except Exception:
            logger.debug("Failed to record tool call step", exc_info=True)

    def _seed_database(self, topic: str) -> list[dict[str, Any]]:
        """Collect topic-relevant records and write them into the sandbox database."""
        search_hits: list[dict[str, str]] = []
        if self.sandbox is not None:
            result = self.sandbox.execute("search", f"{topic} structured dataset examples", max_results=5)
            if isinstance(result, list):
                search_hits = [row for row in result if isinstance(row, dict)]

        self.task_state.add_step(
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

        self.task_state.add_step(
            {
                "type": "seed_database",
                "topic": topic,
                "records": records,
            }
        )

        return records

    def _synthesize_task_tools(self, topic: str, records: list[dict[str, Any]]) -> list[ToolSpec]:
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
            "def impl(query: str, max_results: int = 5) -> list[dict]:\n"
            "    raise RuntimeError('tool spec only')\n"
            "```\n"
        )

        raw = self.llm.simple_complete(prompt, temperature=0.6, max_tokens=900)
        blocks = self._extract_mcp_tool_blocks(raw)
        specs: list[ToolSpec] = []
        for block in blocks:
            try:
                specs.append(ToolSpec.from_function_string(block))
            except Exception:
                logger.debug("Failed to parse tool spec block", exc_info=True)

        specs = [spec for spec in specs if spec.name not in {"bash", "search", "python_runner"}]
        self.task_state.add_step(
            {
                "type": "tool_synthesis",
                "tool_count": len(specs),
                "tools": [spec.model_dump() for spec in specs],
            }
        )
        return specs

    def _register_task_tools(self, tool_specs: list[ToolSpec]) -> None:
        if self.sandbox is None:
            return
        records_path = self.sandbox.sandbox_dir / self._RECORDS_FILENAME
        for spec in tool_specs:
            self.sandbox.register_tool(
                JsonRecordsQueryTool(
                    name=spec.name,
                    description=spec.description,
                    records_path=records_path,
                )
            )

        self.task_state.add_step(
            {
                "type": "tool_registration",
                "registered_tools": [spec.name for spec in tool_specs],
            }
        )

    def _propose_task(
        self,
        task_id: str,
        request: GenerationRequest,
        topic: str,
        records: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        *,
        task_title: str,
        difficulty: int,
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
            "Create ONE challenging but automatically verifiable task.\n"
            "Return ONLY JSON with keys: task_title, task_content, submit_result_format, difficulty_level, solution, verification.\n"
            "Rules:\n"
            "- solution must define solve(tools) and MUST call at least 2 different tools from the tool list.\n"
            "- verification must define verify(tools, answer) and must check content/shape, not just type.\n"
            "- solve/verify must NOT access files or network; only use the provided tools and pure Python.\n"
            "- Tool calling examples: tools['name']('query') or tools.name('query').\n"
            f"- task_title MUST be exactly: {task_title}\n"
            f"Topic: {topic}\n"
            f"Difficulty_level: {difficulty}\n"
            f"Tool list (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.65, max_tokens=1100)
        extracted = self._extract_json(raw)
        data: dict[str, Any] = extracted if isinstance(extracted, dict) else {}

        task_content = str(data.get("task_content") or f"Solve a task about {topic}.").strip()

        submit_result_format = data.get("submit_result_format", "```json\n```")
        if isinstance(submit_result_format, str):
            submit_result_format = {"type": submit_result_format}

        solution = data.get("solution")
        verification = data.get("verification")
        if not isinstance(solution, str) or len(solution.strip()) < 8:
            first = tool_specs[0].name if tool_specs else "tool_a"
            second = tool_specs[1].name if len(tool_specs) > 1 else first
            solution = (
                "def solve(tools):\n"
                f"    a = tools['{first}']('{topic}')\n"
                f"    b = tools['{second}']('{topic}')\n"
                "    return {'a_count': len(a), 'b_count': len(b), 'a_sample': a[:1], 'b_sample': b[:1]}\n"
            )
        if not isinstance(verification, str) or len(verification.strip()) < 8:
            verification = (
                "def verify(tools, answer):\n"
                "    return isinstance(answer, dict) and 'a_count' in answer and 'b_count' in answer\n"
            )

        pkg = TaskPackage(
            task=TaskDefinition(
                task_id=task_id,
                task_title=task_title if len(task_title) >= 4 else "Task",
                task_content=(task_content if len(task_content) >= 10 else f"Solve a task about {topic}."),
                submit_result_format=submit_result_format,
                tool_set=tool_specs,
                evaluation_criteria=EvaluationCriteria(),
                difficulty_level=int(data.get("difficulty_level") or difficulty),
            ),
            solution=solution,
            verification=verification,
            agent_type=self.agent_type,
            metadata={"topic": topic},
        )

        self.task_state.add_step(
            {
                "type": "task_proposed",
                "task_title": pkg.task.task_title,
                "difficulty_level": pkg.task.difficulty_level,
            }
        )
        return pkg

    def _refine_task(
        self,
        topic: str,
        records: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        *,
        previous: TaskPackage,
        difficulty: int,
    ) -> TaskPackage:
        tool_list = [{"name": spec.name, "description": spec.description} for spec in tool_specs]
        prompt = (
            "Increase the task difficulty while keeping it verifiable.\n"
            "Return ONLY JSON with keys: task_content, submit_result_format, difficulty_level, solution, verification.\n"
            "Keep solve(tools) / verify(tools, answer) signatures unchanged.\n"
            "Do not introduce new tools; only use tools from the provided list.\n"
            f"Target difficulty_level: {difficulty}\n"
            f"Tools (JSON): {json.dumps(tool_list, ensure_ascii=False)}\n"
            f"Database sample (JSON): {json.dumps(records[:5], ensure_ascii=False)}\n"
            f"Previous task (JSON): {json.dumps(previous.as_payload(), ensure_ascii=False)}\n"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.7, max_tokens=1200)
        extracted = self._extract_json(raw)
        data: dict[str, Any] = extracted if isinstance(extracted, dict) else {}

        task_content = str(data.get("task_content") or previous.task.task_content).strip()
        submit_result_format = data.get("submit_result_format", previous.task.submit_result_format)
        if isinstance(submit_result_format, str):
            submit_result_format = {"type": submit_result_format}

        solution = data.get("solution") if isinstance(data.get("solution"), str) else previous.solution
        verification = (
            data.get("verification") if isinstance(data.get("verification"), str) else previous.verification
        )

        pkg = TaskPackage(
            task=TaskDefinition(
                task_id=previous.task.task_id,
                task_title=previous.task.task_title,
                task_content=(task_content if len(task_content) >= 10 else previous.task.task_content),
                submit_result_format=submit_result_format,
                tool_set=tool_specs,
                evaluation_criteria=previous.task.evaluation_criteria,
                difficulty_level=int(data.get("difficulty_level") or difficulty),
            ),
            solution=solution,
            verification=verification,
            agent_type=self.agent_type,
            metadata={**previous.metadata, "topic": topic, "refined": "true"},
        )
        self.task_state.add_step(
            {
                "type": "task_refined",
                "task_title": pkg.task.task_title,
                "difficulty_level": pkg.task.difficulty_level,
            }
        )
        return pkg

    def _ensure_valid(self, request: GenerationRequest, package: TaskPackage, *, topic: str) -> TaskPackage:
        if not request.validate or self.sandbox is None:
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
                package = self._repair_package(request, package, error=last_error)
                continue
            missing_tools = called_tools - allowed_tools
            if missing_tools:
                last_error = (
                    "solution calls tools not in the declared tool_set; "
                    f"missing={sorted(missing_tools)} allowed={sorted(allowed_tools)}"
                )
                package = self._repair_package(request, package, error=last_error)
                continue

            run = self.sandbox.run_task(package)
            self.task_state.add_step(
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

            last_error = run.error or run.verification_error or "unknown validation failure"
            package = self._repair_package(request, package, topic=topic, error=last_error)
            package = self._ensure_substantive_task(package.task.tool_set, package)

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
        *,
        error: str,
    ) -> TaskPackage:
        tool_list = [{"name": spec.name, "description": spec.description} for spec in package.task.tool_set]
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

        solution = data.get("solution") if isinstance(data.get("solution"), str) else package.solution
        verification = (
            data.get("verification") if isinstance(data.get("verification"), str) else package.verification
        )
        repaired = package.copy(update={"solution": solution, "verification": verification})
        self.task_state.add_step({"type": "task_repaired", "error": error[:300]})
        return repaired

    def _ensure_substantive_task(self, tool_specs: list[ToolSpec], package: TaskPackage) -> TaskPackage:
        allowed = {spec.name for spec in tool_specs}
        called = self._extract_tool_calls(package.solution)

        if called and called.issubset(allowed) and len(called) >= 2:
            return package

        self.task_state.add_step(
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
        tool_calls = re.findall(
            r"tools\\[['\"](\\w+)['\"]\\]|tools\\.(\\w+)\\s*\\(",
            solution_code or "",
        )
        called_tools = {name for pair in tool_calls for name in pair if name}
        dict_methods = {
            "keys",
            "values",
            "items",
            "get",
            "pop",
            "update",
            "clear",
            "copy",
        }
        return called_tools - dict_methods

    @staticmethod
    def _extract_mcp_tool_blocks(raw: str) -> list[str]:
        text = (raw or "").strip()
        if not text:
            return []

        # Prefer fenced code blocks if present.
        if "```" in text:
            parts = text.split("```")
            candidates = [part for part in parts if "@mcp.tool" in part]
            if candidates:
                text = max(candidates, key=len)

        indices: list[int] = []
        start = 0
        while True:
            idx = text.find("@mcp.tool", start)
            if idx == -1:
                break
            indices.append(idx)
            start = idx + 1

        blocks: list[str] = []
        for i, idx in enumerate(indices):
            end = indices[i + 1] if i + 1 < len(indices) else len(text)
            block = text[idx:end].strip()
            if "def " in block:
                blocks.append(block)
        return blocks
