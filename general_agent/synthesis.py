from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .database import LocalDatabase
from .llm import LLMClient
from .tools import BashTool, SearchTool, ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class SynthesisContext:
    category: str
    sandbox: Path
    db: LocalDatabase
    registry: ToolRegistry
    llm: LLMClient


@dataclass
class TaskBundle:
    """Bundle of task definition, solution code, and verification code."""

    name: str
    description: str
    difficulty: int
    solution_code: str
    verification_code: str

    def run_solution(self, tools: Dict[str, Any]) -> Any:
        env = self._build_exec_env(tools)
        exec(self.solution_code, env, env)
        if "solve" not in env:
            raise RuntimeError("solution_code must define solve(tools)")
        return env["solve"](tools)

    def verify(self, tools: Dict[str, Any], answer: Any) -> bool:
        env = self._build_exec_env(tools)
        exec(self.verification_code, env, env)
        if "verify" not in env:
            raise RuntimeError("verification_code must define verify(tools, answer)")
        return bool(env["verify"](tools, answer))

    @staticmethod
    def _build_exec_env(tools: Dict[str, Any]) -> Dict[str, Any]:
        safe_builtins = {
            "len": len,
            "range": range,
            "min": min,
            "max": max,
            "sum": sum,
            "any": any,
            "all": all,
            "sorted": sorted,
            "enumerate": enumerate,
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "isinstance": isinstance,
        }
        return {"__builtins__": safe_builtins, "tools": tools}


class EnvironmentSynthesizer:
    """Automated pipeline that synthesizes environments, tools, tasks, and verifiers."""

    def __init__(self, llm: LLMClient, max_validation_rounds: int = 4):
        self.llm = llm
        self.max_validation_rounds = max_validation_rounds
        self._trivial_solution_patterns = [
            r"return\s+list\(tools\.keys\(\)\)",
            r"return\s+tools\.keys\(\)",
        ]
        self._trivial_verifier_patterns = [
            r"return\s+isinstance\(answer,\s*list\)",
            r"return\s+True",
        ]

    @staticmethod
    def _parse_json_response(raw: str) -> Any:
        """Best-effort JSON extraction from raw LLM output."""
        text = raw.strip()
        if not text:
            raise json.JSONDecodeError("Empty response", raw, 0)

        def try_load(candidate: str) -> Any:
            candidate = candidate.strip()
            return json.loads(candidate)

        try:
            return try_load(text)
        except json.JSONDecodeError:
            pass

        fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S)
        if fence:
            candidate = fence.group(1)
            try:
                return try_load(candidate)
            except json.JSONDecodeError:
                pass

        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            try:
                return try_load(candidate)
            except json.JSONDecodeError:
                pass

        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            candidate = text[start : end + 1]
            return try_load(candidate)

        raise json.JSONDecodeError("Unable to extract JSON", raw, 0)

    def build_context(self, category: str, sandbox: Path) -> SynthesisContext:
        sandbox.mkdir(parents=True, exist_ok=True)
        db = LocalDatabase.load(sandbox / "db.json")

        registry = ToolRegistry()
        registry.ensure_defaults(bash=BashTool(workdir=sandbox), search=SearchTool())

        return SynthesisContext(category=category, sandbox=sandbox, db=db, registry=registry, llm=self.llm)

    def seed_database(self, ctx: SynthesisContext) -> None:
        """Seed database using the search tool and LLM for the given category."""
        try:
            search_hits = ctx.registry.tools["search"](
                f"{ctx.category} sample data list structured"
            )
        except Exception as exc:  # pragma: no cover - network/API fallback
            logger.warning("Search tool failed, falling back to empty results: %s", exc)
            search_hits = []
        prompt = (
            "You are a data curation assistant. Based on the topic and search hits, "
            "produce 3-5 structured records. Return a JSON array with fields title and summary. Avoid duplicates.\n"
            f"Topic: {ctx.category}\n"
            f"Search hits (JSON): {json.dumps(search_hits, ensure_ascii=False)}"
        )
        generated = ctx.llm.simple_complete(prompt, temperature=0.4, max_tokens=400)
        try:
            records = self._parse_json_response(generated)
        except json.JSONDecodeError:
            records = [{"title": ctx.category, "summary": generated}]
        if isinstance(records, dict):
            records = [records]
        for row in records:
            ctx.db.add_record(row)

    def synthesize_tools(self, ctx: SynthesisContext) -> None:
        """Ask LLM to generate specialized tools based on the database and register them."""
        prompt = (
            "Generate 2-3 specialized tools for the topic. Return a JSON array with fields name and description. "
            "Tools should rely on existing data or simple logic, not external APIs. "
            "Tools must accept either a single positional string or keyword 'query'; avoid additional kwargs.\n"
            f"Topic: {ctx.category}\n"
            f"Database examples: {json.dumps(ctx.db.records[:3], ensure_ascii=False)}"
        )
        raw = ctx.llm.simple_complete(prompt, temperature=0.5, max_tokens=400)
        try:
            tools = self._parse_json_response(raw)
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
                def base_handler(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
                    """Flexible lookup handler; tolerates different calling styles from synthesized code."""
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
                        return ctx.db.records
                    if not isinstance(candidate, str):
                        candidate = str(candidate)
                    return ctx.db.query("title", candidate) or [
                        r
                        for r in ctx.db.records
                        if key in r.get("title", "") or candidate in r.get("summary", "")
                    ]

                class GeneratedTool:
                    def __call__(self, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
                        return base_handler(*args, **kwargs)

                    def __getattr__(self, _name: str):
                        def wrapper(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
                            return base_handler(*args, **kwargs)

                        return wrapper

                    def __getitem__(self, _name: str):
                        return self.__getattr__(_name)

                return GeneratedTool()

            ctx.registry.register(name=name, description=desc, func=make_handler(name))

    def propose_task(self, ctx: SynthesisContext, difficulty: int = 1) -> TaskBundle:
        """Generate a task with solution and verification code."""
        prompt = (
            "You are a task generator. Based on the tool list and database, create a verifiable task.\n"
            "Return JSON with name, description, solution_code, verification_code.\n"
            "Constraints: solution_code must define solve(tools); it can only access data via tools (no direct DB access). "
            "Keep tool calls simple: single positional string or keyword 'query' only. "
            "verification_code must define verify(tools, answer) and return bool.\n"
            "The solution must produce structured content beyond listing tool names; avoid trivial returns. "
            "The verification must check content/shape, not just type.\n"
            f"Category: {ctx.category}\n"
            f"Tools: {json.dumps(ctx.registry.describe(), ensure_ascii=False)}\n"
            f"Database: {json.dumps(ctx.db.records[:5], ensure_ascii=False)}\n"
            f"Difficulty: {difficulty}"
        )
        raw = ctx.llm.simple_complete(prompt, temperature=0.6, max_tokens=800)
        try:
            parsed = self._parse_json_response(raw)
        except json.JSONDecodeError:
            parsed = {
                "name": f"{ctx.category}-task",
                "description": raw[:200],
                "solution_code": "def solve(tools):\n    return list(tools.keys())",
                "verification_code": "def verify(tools, answer):\n    return isinstance(answer, list)",
            }
        return TaskBundle(
            name=parsed.get("name", "generated-task"),
            description=parsed.get("description", ""),
            difficulty=difficulty,
            solution_code=parsed.get("solution_code", ""),
            verification_code=parsed.get("verification_code", ""),
        )

    def refine_task(self, ctx: SynthesisContext, prev: TaskBundle) -> TaskBundle:
        """Increase task difficulty while keeping it verifiable."""
        prompt = (
            "Increase the task difficulty while keeping it verifiable. "
            "Input: previous task with solution and verification code. Output: same JSON schema. "
            "Keep solve and verify signatures unchanged. Keep tool calls simple: positional string or keyword 'query' only.\n"
            f"Previous: {json.dumps(prev.__dict__, ensure_ascii=False)}"
        )
        raw = ctx.llm.simple_complete(prompt, temperature=0.7, max_tokens=800)
        try:
            data = self._parse_json_response(raw)
        except json.JSONDecodeError:
            data = prev.__dict__ | {"difficulty": prev.difficulty + 1}
        return TaskBundle(
            name=data.get("name", prev.name),
            description=data.get("description", prev.description),
            difficulty=data.get("difficulty", prev.difficulty + 1),
            solution_code=data.get("solution_code", prev.solution_code),
            verification_code=data.get("verification_code", prev.verification_code),
        )

    def repair_bundle(self, ctx: SynthesisContext, bundle: TaskBundle, failure_reason: str) -> TaskBundle:
        """Ask LLM to repair the bundle when validation fails."""
        prompt = (
            "The current solution or verification failed. Produce a new JSON with name, description, solution_code, verification_code. "
            "Constraints: solve(tools) must only use tools (no direct DB access); verify(tools, answer) must return bool. "
            "Keep tool calls simple: positional string or keyword 'query' only; avoid extra kwargs.\n"
            f"Failure reason: {failure_reason}\n"
            f"Original task: {json.dumps(bundle.__dict__, ensure_ascii=False)}\n"
            f"Tools: {json.dumps(ctx.registry.describe(), ensure_ascii=False)}\n"
            f"Database examples: {json.dumps(ctx.db.records[:5], ensure_ascii=False)}"
        )
        raw = ctx.llm.simple_complete(prompt, temperature=0.6, max_tokens=800)
        try:
            data = self._parse_json_response(raw)
        except json.JSONDecodeError:
            logger.warning("LLM repair did not return JSON; keeping original task: %s", raw)
            data = bundle.__dict__
        return TaskBundle(
            name=data.get("name", bundle.name),
            description=data.get("description", bundle.description),
            difficulty=data.get("difficulty", bundle.difficulty),
            solution_code=data.get("solution_code", bundle.solution_code),
            verification_code=data.get("verification_code", bundle.verification_code),
        )

    def ensure_valid(self, ctx: SynthesisContext, bundle: TaskBundle, fail_soft: bool = False) -> Tuple[TaskBundle, Any]:
        """Execute and verify a bundle; repair via LLM when needed. If fail_soft, return last attempt instead of raising."""
        base_tools = ctx.registry.as_callable_dict()

        def fallback(*args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            # Generic fallback: return all records or filtered by keyword.
            candidate = None
            if args:
                candidate = args[0]
            if "query" in kwargs:
                candidate = kwargs["query"]
            if isinstance(candidate, dict):
                candidate = json.dumps(candidate, ensure_ascii=False)
            if candidate is None:
                return ctx.db.records
            text = candidate if isinstance(candidate, str) else str(candidate)
            return ctx.db.query("title", text) or [
                r for r in ctx.db.records if text in r.get("title", "") or text in r.get("summary", "")
            ]

        class ToolProxy(dict):
            def __missing__(self, key: str):
                return fallback

        tools: Dict[str, Any] = ToolProxy(**base_tools)
        last_error = ""

        for attempt in range(self.max_validation_rounds + 1):
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

            bundle = self.repair_bundle(ctx, bundle, last_error or "unknown failure")

        if fail_soft:
            logger.warning(
                "Task failed validation repeatedly (soft): %s; last error: %s",
                bundle.name,
                last_error,
            )
            return bundle, None
        raise RuntimeError(f"Task failed validation repeatedly: {bundle.name}; last error: {last_error}")

    def _looks_trivial(self, bundle: TaskBundle) -> bool:
        """Heuristic check to reject trivial solution/verifier pairs."""
        sol = bundle.solution_code or ""
        ver = bundle.verification_code or ""
        for pat in self._trivial_solution_patterns:
            if re.search(pat, sol):
                return True
        for pat in self._trivial_verifier_patterns:
            if re.search(pat, ver):
                return True
        if "answer" in ver and "return" in ver and "if" not in ver:
            return True
        return False

    def _persist(self, ctx: SynthesisContext, bundles: List[TaskBundle]) -> None:
        """Persist synthesis results to the sandbox for later reproduction."""
        payload = {
            "category": ctx.category,
            "tooling": ctx.registry.describe(),
            "records": ctx.db.records,
            "tasks": [bundle.__dict__ for bundle in bundles],
        }
        target = ctx.sandbox / "tasks.json"
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
        logger.info("Synthesis artifacts saved to %s", target)

    def synthesize(
        self,
        category: str,
        sandbox: Path,
        rounds: int = 2,
        validate: bool = True,
        fail_soft: bool = True,
        persist: bool = True,
    ) -> List[TaskBundle]:
        """Main entry point for environment + task synthesis."""
        ctx = self.build_context(category, sandbox)
        self.seed_database(ctx)
        self.synthesize_tools(ctx)

        bundles: List[TaskBundle] = []
        current = self.propose_task(ctx, difficulty=1)
        if validate:
            current, _ = self.ensure_valid(ctx, current, fail_soft=fail_soft)
        bundles.append(current)

        for step in range(1, rounds):
            current = self.refine_task(ctx, current)
            if validate:
                current, _ = self.ensure_valid(ctx, current, fail_soft=fail_soft)
            bundles.append(current)

        if persist:
            self._persist(ctx, bundles)

        return bundles

