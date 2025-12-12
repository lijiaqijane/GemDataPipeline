from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_gem.core.task_schema import EvaluationCriteria, TaskDefinition, TaskPackage, ToolSpec
from agent_gem.core.validation import validate_task_package
from agent_gem.database import LocalDatabase
from agent_gem.tools import BashTool, DockerTool, SandboxFusionTool, SearchTool, ToolRegistry

from .base import AgentRequest, BaseAgent

logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    agent_type = "general_agent"
    description = (
        "Automatic environment-synthesis agent that creates diverse, verifiable tasks with growing toolsets"
    )

    def __init__(
        self,
        llm,
        sandbox: Path,
        use_sandbox_fusion: bool = True,
        use_docker: bool = True,
    ):
        super().__init__(llm)

        sandbox.mkdir(parents=True, exist_ok=True)
        self.sandbox = sandbox
        self.db = LocalDatabase.load(sandbox / "db.json")

        self.registry = ToolRegistry()

        # Configure bash tool (with optional Docker support)
        bash_tool = BashTool(
            workdir=sandbox,
            use_docker=use_docker,
            docker_image=os.getenv("DOCKER_IMAGE", "python:3.11-slim"),
        )
        search_tool = SearchTool()

        # Optionally add Docker tool
        docker_tool = None
        if use_docker:
            docker_tool = DockerTool(
                image=os.getenv("DOCKER_IMAGE", "python:3.11-slim"),
                timeout=int(os.getenv("DOCKER_TIMEOUT", "30")),
                workdir=sandbox,
            )

        # Optionally add SandboxFusion tool if enabled
        sandbox_fusion_tool = None
        if use_sandbox_fusion:
            import os

            base_url = os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080")
            timeout = int(os.getenv("SANDBOX_FUSION_TIMEOUT", "30"))
            default_language = os.getenv("SANDBOX_FUSION_LANGUAGE", "python")
            sandbox_fusion_tool = SandboxFusionTool(
                base_url=base_url, timeout=timeout, default_language=default_language
            )

        self.registry.ensure_defaults(
            bash=bash_tool, search=search_tool, sandbox_fusion=sandbox_fusion_tool
        )

        # Register Docker tool if enabled
        if docker_tool:
            self.registry.register("docker", "Execute code securely in Docker container", docker_tool)

    def seed_database(self, ctx: AgentRequest) -> None:
        """Seed database using the search tool and LLM for the given category."""
        try:
            search_hits = self.registry.tools["search"](f"{ctx.topic} sample data list structured")
        except Exception as exc:  # pragma: no cover - network/API fallback
            logger.warning("Search tool failed, falling back to empty results: %s", exc)
            search_hits = []
        prompt = (
            "You are a data curation assistant. Based on the topic and search hits, "
            "produce 3-5 structured records. Return a JSON array with fields title and summary. Avoid duplicates.\n"
            f"Topic: {ctx.topic}\n"
            f"Search hits (JSON): {json.dumps(search_hits, ensure_ascii=False)}"
        )
        generated = self.llm.chat_completion(prompt, temperature=0.4, max_tokens=400)
        try:
            records = self._extract_json(generated)
        except json.JSONDecodeError:
            records = [{"title": ctx.topic, "summary": generated}]
        if isinstance(records, dict):
            records = [records]
        for row in records:
            self.db.add_record(row)

    def synthesize_tools(self, ctx: AgentRequest, additional_context: str = "") -> None:
        """Ask LLM to generate specialized tools based on the database and register them."""
        context_suffix = f"\nAdditional context: {additional_context}" if additional_context else ""
        prompt = (
            "Generate 2-3 specialized tools for the topic. Return a JSON array with fields name and description. "
            "Tools should rely on existing data or simple logic, not external APIs. "
            "Tools must accept either a single positional string or keyword 'query'; avoid additional kwargs.\n"
            f"Topic: {ctx.topic}\n"
            f"Database examples: {json.dumps(self.db.records[:3], ensure_ascii=False)}{context_suffix}"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.5, max_tokens=400)
        try:
            tools = self._parse_json(raw)
        except json.JSONDecodeError:
            tools = []
        if not isinstance(tools, list):
            tools = [tools]

        for spec in tools:
            name = spec.get("name")
            desc = spec.get("description", "")
            if not name:
                continue

            def make_handler(key: str):
                def base_handler(*args: Any, **kwargs: Any) -> Any:
                    """Flexible lookup handler; tolerates different calling styles from synthesized code."""
                    logger.debug("Tool '%s' called with args=%s, kwargs=%s", key, args, kwargs)
                    candidate: Any = None
                    if args:
                        candidate = args[0]
                    if "query" in kwargs:
                        candidate = kwargs["query"]
                    if candidate is None and kwargs:
                        candidate = " ".join(f"{k}:{v}" for k, v in kwargs.items())
                    if isinstance(candidate, dict):
                        candidate = json.dumps(candidate, ensure_ascii=False)
                    if candidate is None:
                        result = self.db.records
                    elif not isinstance(candidate, str):
                        candidate = str(candidate)
                        result = self.db.query("title", candidate) or [
                            r
                            for r in self.db.records
                            if key in r.get("title", "") or candidate in r.get("summary", "")
                        ]
                    else:
                        result = self.db.query("title", candidate) or [
                            r
                            for r in self.db.records
                            if key in r.get("title", "") or candidate in r.get("summary", "")
                        ]

                    # If tool name suggests it should return a dict (e.g., "checker", "analyzer"), convert to dict
                    if any(word in key.lower() for word in ["checker", "analyzer", "validator", "matcher"]):
                        if isinstance(result, list) and result:
                            # Convert list of records to a structured dict
                            if "component" in key.lower() or "checker" in key.lower():
                                # For component checkers, return dict with present/missing
                                all_text = " ".join(
                                    [r.get("title", "") + " " + r.get("summary", "") for r in result]
                                )
                                query_lower = (candidate or "").lower()
                                present = []
                                missing = []
                                # Simple keyword matching
                                keywords = [
                                    "transportation",
                                    "accommodation",
                                    "activity",
                                    "reservation",
                                    "emergency",
                                    "booking",
                                ]
                                for kw in keywords:
                                    if kw in query_lower:
                                        present.append(f"{kw} details")
                                    else:
                                        missing.append(f"{kw} information")
                                result = (
                                    {"present": present[:3], "missing": missing[:2]}
                                    if missing
                                    else {"present": present[:3], "missing": []}
                                )
                            elif "tool" in key.lower() or "matcher" in key.lower():
                                # For tool matchers, return dict with recommendations
                                result = {
                                    "tools": [r.get("title", "") for r in result[:3]],
                                    "count": len(result),
                                }
                            else:
                                # Generic: return first record as dict or wrap list
                                result = result[0] if result else {}
                    elif isinstance(result, list) and len(result) == 1:
                        # Single result: return as dict
                        result = result[0]

                    logger.debug("Tool '%s' returned %s", key, type(result).__name__)
                    return result

                class GeneratedTool:
                    """Tool wrapper that supports various calling patterns."""

                    def __call__(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
                        return base_handler(*args, **kwargs)

                    def __getattr__(self, _name: str):
                        # Support tool.method() or tool.attribute patterns
                        def wrapper(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
                            return base_handler(*args, **kwargs)

                        return wrapper

                    def __getitem__(self, _name: str):
                        # Support tool['method']() patterns
                        return self.__getattr__(_name)

                    def __setattr__(self, name: str, value: Any) -> None:
                        # Allow setting attributes (some code might try this)
                        object.__setattr__(self, name, value)

                return GeneratedTool()

            self.registry.register(name=name, description=desc, func=make_handler(name))

    def augment_toolset(self, ctx: AgentRequest, bundle: TaskPackage, failure_reason: str) -> bool:
        """Augment the toolset when current tools are insufficient. Returns True if new tools were added."""
        # Analyze if the failure might be due to missing tools
        solution_code = bundle.solution or ""

        called_tools = self._extract_tool_calls(solution_code)
        available_tools = {tool.name for tool in self.registry.tools.values()}
        missing_tools = called_tools - available_tools

        # Check if failure suggests missing functionality
        needs_augmentation = (
            missing_tools
            or "not found" in failure_reason.lower()
            or "missing" in failure_reason.lower()
            or "no attribute" in failure_reason.lower()
            or (len(failure_reason) > 50 and "verification returned False" in failure_reason)
        )

        if not needs_augmentation:
            return False

        logger.info(
            "Augmenting toolset: detected missing tools %s or insufficient functionality",
            missing_tools,
        )

        # Generate additional tools based on task requirements
        additional_context = (
            f"Current task requires tools that are not available. "
            f"Task description: {bundle.description[:200]}. "
            f"Solution code attempts to use: {list(called_tools)}. "
            f"Failure reason: {failure_reason[:200]}. "
            f"Generate 1-2 additional tools that would help solve this task."
        )

        prompt = (
            "Generate 1-2 additional specialized tools to help solve the current task. "
            "Return a JSON array with fields name and description. "
            "Tools should complement existing tools and address the specific needs of the task. "
            "Tools must accept either a single positional string or keyword 'query'.\n"
            f"Topic: {ctx.topic}\n"
            f"Current tools: {json.dumps(self.registry.describe(), ensure_ascii=False)}\n"
            f"Task: {bundle.description[:300]}\n"
            f"Failure context: {failure_reason[:200]}\n"
            f"Database examples: {json.dumps(self.db.records[:3], ensure_ascii=False)}"
        )

        raw = self.llm.chat_completion(prompt, temperature=0.6, max_tokens=400)
        try:
            new_tools = self._extract_json(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse new tools from LLM response")
            return False

        if not isinstance(new_tools, list):
            new_tools = [new_tools]

        added_count = 0
        for spec in new_tools:
            name = spec.get("name")
            desc = spec.get("description", "")
            if not name or name in available_tools:
                continue  # Skip if already exists

            def make_handler(key: str):
                def base_handler(*args: Any, **kwargs: Any) -> Any:
                    """Flexible lookup handler for augmented tools."""
                    logger.debug(
                        "Augmented tool '%s' called with args=%s, kwargs=%s",
                        key,
                        args,
                        kwargs,
                    )
                    candidate: Any = None
                    if args:
                        candidate = args[0]
                    if "query" in kwargs:
                        candidate = kwargs["query"]
                    if candidate is None and kwargs:
                        candidate = " ".join(f"{k}:{v}" for k, v in kwargs.items())
                    if isinstance(candidate, dict):
                        candidate = json.dumps(candidate, ensure_ascii=False)
                    if candidate is None:
                        result = self.db.records
                    elif not isinstance(candidate, str):
                        candidate = str(candidate)
                        result = self.db.query("title", candidate) or [
                            r
                            for r in self.db.records
                            if key in r.get("title", "") or candidate in r.get("summary", "")
                        ]
                    else:
                        result = self.db.query("title", candidate) or [
                            r
                            for r in self.db.records
                            if key in r.get("title", "") or candidate in r.get("summary", "")
                        ]

                    # Smart return format based on tool name
                    if any(word in key.lower() for word in ["checker", "analyzer", "validator", "matcher"]):
                        if isinstance(result, list) and result:
                            if "component" in key.lower() or "checker" in key.lower():
                                query_lower = (candidate or "").lower()
                                present = []
                                missing = []
                                keywords = [
                                    "transportation",
                                    "accommodation",
                                    "activity",
                                    "reservation",
                                    "emergency",
                                    "booking",
                                ]
                                for kw in keywords:
                                    if kw in query_lower:
                                        present.append(f"{kw} details")
                                    else:
                                        missing.append(f"{kw} information")
                                result = (
                                    {"present": present[:3], "missing": missing[:2]}
                                    if missing
                                    else {"present": present[:3], "missing": []}
                                )
                            elif "tool" in key.lower() or "matcher" in key.lower():
                                result = {
                                    "tools": [r.get("title", "") for r in result[:3]],
                                    "count": len(result),
                                }
                            else:
                                result = result[0] if result else {}
                    elif isinstance(result, list) and len(result) == 1:
                        result = result[0]

                    logger.debug("Augmented tool '%s' returned %s", key, type(result).__name__)
                    return result

                class GeneratedTool:
                    def __call__(self, *args: Any, **kwargs: Any) -> Any:
                        return base_handler(*args, **kwargs)

                    def __getattr__(self, _name: str):
                        def wrapper(*args: Any, **kwargs: Any) -> Any:
                            return base_handler(*args, **kwargs)

                        return wrapper

                    def __getitem__(self, _name: str):
                        return self.__getattr__(_name)

                    def __setattr__(self, name: str, value: Any) -> None:
                        object.__setattr__(self, name, value)

                return GeneratedTool()

            self.registry.register(name=name, description=desc, func=make_handler(name))
            added_count += 1
            logger.info("Added new tool: %s - %s", name, desc)

        return added_count > 0

    def propose_task(self, ctx: AgentRequest, difficulty: int = 1) -> TaskPackage:
        """Generate a task with solution and verification code."""
        tool_examples = "\n".join(
            [
                f"- {tool['name']}: Call as tools['{tool['name']}']('query') or tools.{tool['name']}('query')"
                for tool in self.registry.describe()[:3]
            ]
        )
        prompt = (
            "You are a task generator. Based on the tool list and database, create a verifiable task.\n"
            "Return JSON with name, description, solution_code, verification_code.\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. solution_code MUST ACTUALLY CALL TOOLS using tools['name']('query') or tools.name('query').\n"
            "2. Call at least 2 different tools and combine their results into a structured output.\n"
            "3. Do NOT return trivial results like 'list(tools.keys())' or just tool names.\n"
            "4. It can only access data via tools (no direct DB access).\n"
            "5. verification_code must define verify(tools, answer) and return bool.\n"
            "6. The verification must check content/shape, not just type.\n\n"
            f"Category: {ctx.topic}\n"
            f"Tool usage examples:\n{tool_examples}\n"
            f"All tools: {json.dumps(self.registry.describe(), ensure_ascii=False)}\n"
            f"Database samples: {json.dumps(self.db.records[:5], ensure_ascii=False)}\n"
            f"Difficulty: {difficulty}\n\n"
            "Example solution pattern:\n"
            "def solve(tools):\n"
            "    data1 = tools['tool1']('query1')\n"
            "    data2 = tools['tool2']('query2')\n"
            "    return {'result': data1 + data2}\n"
        )
        raw = self.llm.chat_completion(prompt, temperature=0.6, max_tokens=800)
        try:
            parsed = self._extract_json(raw)
        except json.JSONDecodeError:
            parsed = {
                "name": f"{ctx.topic}-task",
                "description": raw[:200],
                "solution_code": "def solve(tools):\n    return list(tools.keys())",
                "verification_code": "def verify(tools, answer):\n    return isinstance(answer, list)",
            }
        return TaskPackage(
            name=parsed.get("name", "generated-task"),
            description=parsed.get("description", ""),
            difficulty=difficulty,
            solution_code=parsed.get("solution_code", ""),
            verification_code=parsed.get("verification_code", ""),
        )

    def refine_task(self, ctx: AgentRequest, prev: TaskPackage) -> TaskPackage:
        """Increase task difficulty while keeping it verifiable."""
        prompt = (
            "Increase the task difficulty while keeping it verifiable. "
            "Input: previous task with solution and verification code. Output: same JSON schema. "
            "Keep solve and verify signatures unchanged. Keep tool calls simple: positional string or keyword 'query' only.\n"
            f"Previous: {json.dumps(prev.__dict__, ensure_ascii=False)}"
        )
        raw = self.llm.chat_completion(prompt, temperature=0.7, max_tokens=800)
        try:
            data = self._extract_json(raw)
        except json.JSONDecodeError:
            data = prev.__dict__ | {"difficulty": prev.difficulty + 1}
        return TaskPackage(
            name=data.get("name", prev.name),
            description=data.get("description", prev.description),
            difficulty=data.get("difficulty", prev.difficulty + 1),
            solution_code=data.get("solution_code", prev.solution_code),
            verification_code=data.get("verification_code", prev.verification_code),
        )

    def repair_bundle(self, ctx: AgentRequest, bundle: TaskPackage, failure_reason: str) -> TaskPackage:
        """Ask LLM to repair the bundle when validation fails."""
        prompt = (
            "The current solution or verification failed. Produce a new JSON with name, description, solution_code, verification_code. "
            "Constraints: solve(tools) must only use tools (no direct DB access); verify(tools, answer) must return bool. "
            "Keep tool calls simple: positional string or keyword 'query' only; avoid extra kwargs.\n"
            f"Failure reason: {failure_reason}\n"
            f"Original task: {json.dumps(bundle.__dict__, ensure_ascii=False)}\n"
            f"Tools: {json.dumps(self.registry.describe(), ensure_ascii=False)}\n"
            f"Database examples: {json.dumps(self.db.records[:5], ensure_ascii=False)}"
        )
        raw = self.llm.chat_completion(prompt, temperature=0.6, max_tokens=800)
        try:
            data = self._extract_json(raw)
        except json.JSONDecodeError:
            logger.warning("LLM repair did not return JSON; keeping original task: %s", raw)
            data = bundle.__dict__
        return TaskPackage(
            name=data.get("name", bundle.name),
            description=data.get("description", bundle.description),
            difficulty=data.get("difficulty", bundle.difficulty),
            solution_code=data.get("solution_code", bundle.solution),
            verification_code=data.get("verification_code", bundle.verification),
            use_docker=bundle.use_docker,  # Preserve use_docker flag
        )

    def ensure_valid(
        self, ctx: AgentRequest, bundle: TaskPackage, fail_soft: bool = False
    ) -> Tuple[TaskPackage, Any]:
        """Execute and verify a bundle; repair via LLM when needed. If fail_soft, return last attempt instead of raising."""
        base_tools = self.registry.as_callable_dict()

        # Ensure the task is not trivial before running executions
        bundle = self._ensure_substantive_task(ctx, bundle, "Initial validation quality gate")

        def fallback(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            # Generic fallback: return all records or filtered by keyword.
            logger.debug("Fallback tool called with args=%s, kwargs=%s", args, kwargs)
            candidate = None
            if args:
                candidate = args[0]
            if "query" in kwargs:
                candidate = kwargs["query"]
            if isinstance(candidate, dict):
                candidate = json.dumps(candidate, ensure_ascii=False)
            if candidate is None:
                result = self.db.records
            else:
                text = candidate if isinstance(candidate, str) else str(candidate)
                result = self.db.query("title", text) or [
                    r for r in self.db.records if text in r.get("title", "") or text in r.get("summary", "")
                ]
            logger.debug("Fallback tool returned %d records", len(result))
            return result

        class ToolProxy(dict):
            """Proxy that supports both dict access and attribute access for tools."""

            def __missing__(self, key: str):
                # Return a callable wrapper that always calls fallback
                class FallbackTool:
                    def __call__(self, *args: Any, **kwargs: Any) -> Any:
                        return fallback(*args, **kwargs)

                    def __getattr__(self, name: str):
                        # Support tool.method() calls
                        return self.__call__

                    def __getitem__(self, name: str):
                        # Support tool['method']() calls
                        return self.__call__

                return FallbackTool()

            def __getattr__(self, key: str):
                # Support tools.tool_name() access
                if key in self:
                    return self[key]
                return self.__missing__(key)

        tools: Dict[str, Any] = ToolProxy(**base_tools)
        last_error = ""
        augmentation_attempted = False

        for attempt in range(self.max_validation_rounds + 1):
            # Refresh tools dict after potential augmentation
            base_tools = self.registry.as_callable_dict()
            tools = ToolProxy(**base_tools)

            try:
                answer = bundle.run_solution(tools)
                valid = bundle.verify(tools, answer)
            except Exception as exc:  # pragma: no cover - runtime defense
                last_error = str(exc)
                logger.warning("Task %s raised during execution: %s", bundle.name, last_error)
                valid = False
            else:
                if not valid:
                    last_error = "verification returned False"

            if valid:
                return bundle, answer

            # Try augmenting toolset if we haven't tried yet and we're past first attempt
            if attempt >= 1 and not augmentation_attempted:
                if self.augment_toolset(ctx, bundle, last_error):
                    augmentation_attempted = True
                    logger.info("Toolset augmented, retrying validation...")
                    continue  # Retry with augmented tools

            bundle = self.repair_bundle(ctx, bundle, last_error or "unknown failure")
            bundle = self._ensure_substantive_task(ctx, bundle, "Post-repair quality gate")

        if fail_soft:
            logger.warning(
                "Task failed validation repeatedly (soft): %s; last error: %s",
                bundle.name,
                last_error,
            )
            return bundle, None
        raise RuntimeError(f"Task failed validation repeatedly: {bundle.name}; last error: {last_error}")

    def _looks_trivial(self, bundle: TaskPackage) -> bool:
        """Heuristic check to reject trivial solution/verifier pairs."""
        sol = bundle.solution or ""
        ver = bundle.verification or ""

        # Check for trivial solution patterns
        for pat in self._trivial_solution_patterns:
            if re.search(pat, sol):
                return True

        # Check if solution actually calls tools
        if not re.search(r"tools\[['\"]\w+['\"]\]|tools\.\w+", sol):
            return True  # No tool calls found

        # Check for trivial verifier patterns
        for pat in self._trivial_verifier_patterns:
            if re.search(pat, ver):
                return True
        if "answer" in ver and "return" in ver and "if" not in ver:
            return True
        return False

    def _ensure_substantive_task(
        self, ctx: AgentRequest, bundle: TaskPackage, reason: str = ""
    ) -> TaskPackage:
        """Repair tasks that are trivial or do not use enough tools."""
        base_reason = reason or "Task too trivial or lacks multiple tool calls"
        for _ in range(3):
            tool_calls = self._extract_tool_calls(bundle.solution)
            if not self._looks_trivial(bundle) and len(tool_calls) >= 2:
                return bundle
            bundle = self.repair_bundle(ctx, bundle, f"{base_reason}; tool_calls={list(tool_calls)}")
        return bundle

    def _persist(self, ctx: AgentRequest, bundles: List[TaskPackage]) -> None:
        """Persist synthesis results to the sandbox for later reproduction."""
        payload = {
            "category": ctx.topic,
            "tooling": self.registry.describe(),
            "records": self.db.records,
            "tasks": [bundle.__dict__ for bundle in bundles],
        }
        target = ctx.sandbox / "tasks.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info("Synthesis artifacts saved to %s", target)

    def generate(
        self,
        category: str,
        sandbox: Path,
        rounds: int = 2,
        validate: bool = True,
        fail_soft: bool = True,
        persist: bool = True,
        use_sandbox_fusion: bool = False,
        use_docker: bool = False,
    ) -> List[TaskPackage]:
        """Main entry point for environment + task synthesis.

        Args:
            category: Task category
            sandbox: Sandbox directory
            rounds: Number of difficulty refinement rounds
            validate: Whether to validate tasks
            fail_soft: Whether to fail softly (warn instead of raise)
            persist: Whether to persist results
            use_sandbox_fusion: Whether to use SandboxFusion for secure code execution
            use_docker: Whether to use Docker for secure code execution
        """
        ctx = self.build_context(
            category,
            sandbox,
            use_sandbox_fusion=use_sandbox_fusion,
            use_docker=use_docker,
        )
        self.seed_database(ctx)
        self.synthesize_tools(ctx)

        bundles: List[TaskPackage] = []
        current = self._ensure_substantive_task(
            ctx, self.propose_task(ctx, difficulty=1), "Initial task quality gate"
        )
        # Set use_docker flag if enabled
        if use_docker:
            current.use_docker = True
        if validate:
            current, _ = self.ensure_valid(ctx, current, fail_soft=fail_soft)
        bundles.append(current)

        for step in range(1, rounds):
            current = self._ensure_substantive_task(
                ctx,
                self.refine_task(ctx, current),
                f"Refined task quality gate (round {step})",
            )
            if use_docker:
                current.use_docker = True
            if validate:
                # Before validating, check if the refined task might need additional tools
                # by analyzing the solution code for tool calls
                called_tools = self._extract_tool_calls(current.solution)
                available_tools = {tool.name for tool in self.registry.tools.values()}
                missing_tools = called_tools - available_tools

                if missing_tools:
                    logger.info("Refined task requires additional tools: %s", missing_tools)
                    # Try to augment toolset proactively
                    self.augment_toolset(ctx, current, f"Task requires tools: {missing_tools}")

                current, _ = self.ensure_valid(ctx, current, fail_soft=fail_soft)
            bundles.append(current)

        if persist:
            self._persist(ctx, bundles)

        return bundles
