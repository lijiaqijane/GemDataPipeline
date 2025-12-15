from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

from .database import LocalDatabase
from .executor import SandboxFusionExecutor
from .llm import LLMClient
from .tools import BashTool, DockerTool, SearchTool, ToolRegistry

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
    use_docker: bool = False
    use_sandbox_fusion: bool = False

    def run_solution(self, tools: Dict[str, Any], force_local: bool = False) -> Any:
        """Execute solution code.
        
        Args:
            tools: Dictionary of available tools
            force_local: Force local execution even if sandbox mode is enabled (for validation)
        """
        code = self._normalize_code(self.solution_code)
        if not force_local:
            if self.use_sandbox_fusion:
                return self._run_in_sandbox_fusion(code, tools, "solve")
            if self.use_docker:
                return self._run_in_docker(code, tools, "solve")
        
        # Local execution (used for validation or when sandbox is disabled)
        env = self._build_exec_env(tools)
        exec(code, env, env)
        if "solve" not in env:
            raise RuntimeError("solution_code must define solve(tools)")
        return env["solve"](tools)

    def verify(self, tools: Dict[str, Any], answer: Any, force_local: bool = False) -> bool:
        """Execute verification code.
        
        Args:
            tools: Dictionary of available tools
            answer: The answer from run_solution to verify
            force_local: Force local execution even if sandbox mode is enabled (for validation)
        """
        code = self._normalize_code(self.verification_code)
        if not force_local:
            if self.use_sandbox_fusion:
                result = self._run_in_sandbox_fusion(code, tools, "verify", answer)
                return bool(result)
            if self.use_docker:
                result = self._run_in_docker(code, tools, "verify", answer)
                return bool(result)
        
        # Local execution (used for validation or when sandbox is disabled)
        env = self._build_exec_env(tools)
        exec(code, env, env)
        if "verify" not in env:
            raise RuntimeError("verification_code must define verify(tools, answer)")
        return bool(env["verify"](tools, answer))
    
    def _run_in_sandbox_fusion(self, code: str, tools: Dict[str, Any], func_name: str, *args: Any) -> Any:
        """Execute code in SandboxFusion service."""
        import json
        import os
        
        # Serialize tools metadata (names and descriptions only)
        tools_meta = {name: {"name": name} for name in tools.keys()}
        tools_meta_json = json.dumps(tools_meta)
        
        # Create wrapper code for SandboxFusion
        wrapper_code = f"""
import json
import sys

# Simplified tools interface for SandboxFusion
class ToolProxy:
    def __init__(self, tool_names):
        self._names = tool_names
    
    def __getitem__(self, key):
        return self
    
    def __getattr__(self, key):
        return self
    
    def __call__(self, *args, **kwargs):
        return {{"result": "tool_executed", "args": str(args), "kwargs": str(kwargs)}}

tools_meta = json.loads('{tools_meta_json}')
tools = ToolProxy(tools_meta.keys())

{code}

if '{func_name}' == 'solve':
    result = solve(tools)
    print(json.dumps(result, default=str))
elif '{func_name}' == 'verify':
    answer_str = '''{json.dumps(args[0], default=str) if args else 'null'}'''
    try:
        answer = json.loads(answer_str)
    except:
        answer = answer_str
    result = verify(tools, answer)
    print(json.dumps(result, default=str))
"""
        
        executor = SandboxFusionExecutor(
            base_url=os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080"),
            timeout=int(os.getenv("SANDBOX_FUSION_TIMEOUT", "30")),
        )
        
        result = executor(wrapper_code, language="python")
        
        if result.get("return_code", 0) != 0 or result.get("status") == "error":
            raise RuntimeError(f"SandboxFusion execution failed: {result.get('stderr', 'Unknown error')}")
        
        try:
            output = result.get("stdout", "").strip()
            if output:
                # Try to find JSON in output
                start = output.find('{')
                end = output.rfind('}') + 1
                if start >= 0 and end > start:
                    output = output[start:end]
                elif output.startswith('['):
                    end = output.rfind(']') + 1
                    if end > 0:
                        output = output[:end]
                return json.loads(output)
            return None
        except json.JSONDecodeError as e:
            try:
                return eval(output)
            except:
                raise RuntimeError(f"Failed to parse SandboxFusion output: {output[:200]}. Error: {e}")
    
    def _run_in_docker(self, code: str, tools: Dict[str, Any], func_name: str, *args: Any) -> Any:
        """Execute code in Docker container.
        
        Note: Tools cannot be directly passed to Docker, so we create a simplified
        tool interface that uses subprocess to call back to the host.
        """
        import json
        import os
        import base64
        
        # Serialize tools metadata (names and descriptions only)
        tools_meta = {name: {"name": name} for name in tools.keys()}
        tools_meta_json = json.dumps(tools_meta)
        
        # Create a wrapper that provides a mock tools interface
        # In a real implementation, tools would communicate via a shared mechanism
        # For now, we'll create a simplified version that works with the code structure
        wrapper_code = f"""
import json
import sys

# Simplified tools interface - tools are accessed but execution happens locally
class ToolProxy:
    def __init__(self, tool_names):
        self._names = tool_names
    
    def __getitem__(self, key):
        return self
    
    def __getattr__(self, key):
        return self
    
    def __call__(self, *args, **kwargs):
        # Return mock data for tool calls
        return {{"result": "tool_executed", "args": str(args), "kwargs": str(kwargs)}}

tools_meta = json.loads('{tools_meta_json}')
tools = ToolProxy(tools_meta.keys())

{code}

if '{func_name}' == 'solve':
    result = solve(tools)
    print(json.dumps(result, default=str))
elif '{func_name}' == 'verify':
    answer_str = sys.argv[1] if len(sys.argv) > 1 else 'null'
    try:
        answer = json.loads(answer_str)
    except:
        answer = answer_str
    result = verify(tools, answer)
    print(json.dumps(result, default=str))
"""
        
        docker_tool = DockerTool(
            image=os.getenv("DOCKER_IMAGE", "python:3.11-slim"),
            timeout=int(os.getenv("DOCKER_TIMEOUT", "30")),
        )
        
        # For verify, pass answer as JSON string argument
        if func_name == "verify" and args:
            answer_json = json.dumps(args[0], default=str)
            # Escape for shell
            answer_json_escaped = answer_json.replace("'", "'\"'\"'")
            wrapper_code = wrapper_code.replace("sys.argv[1]", f"'{answer_json_escaped}'")
        
        result = docker_tool(code=wrapper_code, language="python")
        
        if result["returncode"] != 0:
            raise RuntimeError(f"Docker execution failed: {result['stderr']}")
        
        try:
            output = result["stdout"].strip()
            # Remove any non-JSON prefix (like print statements)
            if output:
                # Try to find JSON in output
                start = output.find('{')
                end = output.rfind('}') + 1
                if start >= 0 and end > start:
                    output = output[start:end]
                elif output.startswith('['):
                    end = output.rfind(']') + 1
                    if end > 0:
                        output = output[:end]
                return json.loads(output)
            return None
        except json.JSONDecodeError as e:
            # Fallback: try to evaluate as Python literal
            try:
                return eval(output)
            except:
                raise RuntimeError(f"Failed to parse Docker output: {output[:200]}. Error: {e}")

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

    @staticmethod
    def _normalize_code(code: str) -> str:
        """Normalize common lowercase literals to valid Python to avoid trivial runtime errors."""
        if not code:
            return code
        # Replace standalone true/false/null (case-insensitive) with Python literals
        code = re.sub(r"\btrue\b", "True", code, flags=re.IGNORECASE)
        code = re.sub(r"\bfalse\b", "False", code, flags=re.IGNORECASE)
        code = re.sub(r"\bnull\b", "None", code, flags=re.IGNORECASE)
        return code


class EnvironmentSynthesizer:
    """Automated pipeline that synthesizes environments, tools, tasks, and verifiers."""

    def __init__(self, llm: LLMClient, max_validation_rounds: int = 4):
        self.llm = llm
        self.max_validation_rounds = max_validation_rounds
        self._trivial_solution_patterns = [
            r"return\s+list\(tools\.keys\(\)\)",
            r"return\s+tools\.keys\(\)",
            r"return\s+\[.*tools",
            r"return\s+tools",
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

    @staticmethod
    def _extract_tool_calls(solution_code: str) -> set[str]:
        """Extract tool names used in the solution code."""
        tool_calls = re.findall(r"tools\[['\"](\w+)['\"]\]|tools\.(\w+)\s*\(", solution_code or "")
        called_tools = {name for pair in tool_calls for name in pair if name}
        dict_methods = {"keys", "values", "items", "get", "pop", "update", "clear", "copy"}
        return called_tools - dict_methods

    def build_context(self, category: str, sandbox: Path, use_sandbox_fusion: bool = True, use_docker: bool = True) -> SynthesisContext:
        sandbox.mkdir(parents=True, exist_ok=True)
        db = LocalDatabase.load(sandbox / "db.json")

        registry = ToolRegistry()
        
        # Configure bash tool (with optional Docker support)
        import os
        bash_tool = BashTool(
            workdir=sandbox,
            use_docker=use_docker,
            docker_image=os.getenv("DOCKER_IMAGE", "python:3.11-slim")
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
        
        registry.ensure_defaults(bash=bash_tool, search=search_tool)
        
        # Register Docker tool if enabled
        if docker_tool:
            registry.register("docker", "Execute code securely in Docker container", docker_tool)
        
        # Register SandboxFusion executor if enabled (for information only, actual execution is in TaskBundle)
        if use_sandbox_fusion:
            logger.info("SandboxFusion enabled for secure code execution")

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

    def synthesize_tools(self, ctx: SynthesisContext, additional_context: str = "") -> None:
        """Ask LLM to generate specialized tools based on the database and register them."""
        context_suffix = f"\nAdditional context: {additional_context}" if additional_context else ""
        prompt = (
            "Generate 2-3 specialized tools for the topic. Return a JSON array with fields name and description. "
            "Tools should rely on existing data or simple logic, not external APIs. "
            "Tools must accept either a single positional string or keyword 'query'; avoid additional kwargs.\n"
            f"Topic: {ctx.category}\n"
            f"Database examples: {json.dumps(ctx.db.records[:3], ensure_ascii=False)}{context_suffix}"
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
                        result = ctx.db.records
                    elif not isinstance(candidate, str):
                        candidate = str(candidate)
                        result = self.smart_db_query(ctx.db.records, key, candidate)
                    else:
                        result = self.smart_db_query(ctx.db.records, key, candidate)
                    
                    # Simplify return format based on tool type for easier consumption
                    if isinstance(result, list) and result:
                        # Return simplified, consistent format
                        if any(word in key.lower() for word in ["matcher", "finder", "neighborhood"]):
                            # For matchers/finders: return list of names/titles
                            result = [r.get("title", str(r)) for r in result[:5]]
                        elif any(word in key.lower() for word in ["recommendation", "seasonal", "advisor"]):
                            # For recommendations: return summary text from first few matches
                            result = [r.get("summary", r.get("title", str(r))) for r in result[:3]]
                        elif any(word in key.lower() for word in ["categorizer", "attraction", "activity"]):
                            # For categorizers: return structured data
                            result = [{"name": r.get("title", ""), "info": r.get("summary", "")} for r in result[:5]]
                        elif any(word in key.lower() for word in ["checker", "analyzer", "validator"]):
                            # For checkers: return dict with present/missing
                            query_lower = (candidate or "").lower()
                            present = []
                            missing = []
                            keywords = ["transportation", "accommodation", "activity", "reservation", "emergency", "booking"]
                            for kw in keywords:
                                if kw in query_lower:
                                    present.append(f"{kw} details")
                                else:
                                    missing.append(f"{kw} information")
                            result = {"present": present[:3], "missing": missing[:2]}
                        else:
                            # Default: return first few records as list
                            result = result[:5]
                    elif isinstance(result, list) and len(result) == 0:
                        result = []
                    
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

            ctx.registry.register(name=name, description=desc, func=make_handler(name))

    def augment_toolset(self, ctx: SynthesisContext, bundle: TaskBundle, failure_reason: str) -> bool:
        """Augment the toolset when current tools are insufficient. Returns True if new tools were added."""
        # Analyze if the failure might be due to missing tools
        solution_code = bundle.solution_code or ""
        
        called_tools = self._extract_tool_calls(solution_code)
        available_tools = {tool.name for tool in ctx.registry.tools.values()}
        missing_tools = called_tools - available_tools
        
        # Check if failure suggests missing functionality
        needs_augmentation = (
            missing_tools or
            "not found" in failure_reason.lower() or
            "missing" in failure_reason.lower() or
            "no attribute" in failure_reason.lower() or
            (len(failure_reason) > 50 and "verification returned False" in failure_reason)
        )
        
        if not needs_augmentation:
            return False
        
        logger.info("Augmenting toolset: detected missing tools %s or insufficient functionality", missing_tools)
        
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
            f"Topic: {ctx.category}\n"
            f"Current tools: {json.dumps(ctx.registry.describe(), ensure_ascii=False)}\n"
            f"Task: {bundle.description[:300]}\n"
            f"Failure context: {failure_reason[:200]}\n"
            f"Database examples: {json.dumps(ctx.db.records[:3], ensure_ascii=False)}"
        )
        
        raw = ctx.llm.simple_complete(prompt, temperature=0.6, max_tokens=400)
        try:
            new_tools = self._parse_json_response(raw)
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
                    logger.debug("Augmented tool '%s' called with args=%s, kwargs=%s", key, args, kwargs)
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
                        result = ctx.db.records
                    elif not isinstance(candidate, str):
                        candidate = str(candidate)
                        result = self.smart_db_query(ctx.db.records, key, candidate)
                    else:
                        result = self.smart_db_query(ctx.db.records, key, candidate)
                    
                    # Smart return format based on tool name
                    if any(word in key.lower() for word in ["checker", "analyzer", "validator", "matcher"]):
                        if isinstance(result, list) and result:
                            if "component" in key.lower() or "checker" in key.lower():
                                query_lower = (candidate or "").lower()
                                present = []
                                missing = []
                                keywords = ["transportation", "accommodation", "activity", "reservation", "emergency", "booking"]
                                for kw in keywords:
                                    if kw in query_lower:
                                        present.append(f"{kw} details")
                                    else:
                                        missing.append(f"{kw} information")
                                result = {"present": present[:3], "missing": missing[:2]} if missing else {"present": present[:3], "missing": []}
                            elif "tool" in key.lower() or "matcher" in key.lower():
                                result = {"tools": [r.get("title", "") for r in result[:3]], "count": len(result)}
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
            
            ctx.registry.register(name=name, description=desc, func=make_handler(name))
            added_count += 1
            logger.info("Added new tool: %s - %s", name, desc)
        
        return added_count > 0

    def smart_db_query(self, records: List[Dict[str, Any]], tool_key: str, query: str) -> List[Dict[str, Any]]:
        """Enhanced database query with flexible keyword matching."""
        if not query or not isinstance(query, str):
            return records

        query_lower = query.lower().strip()
        tool_lower = tool_key.lower()

        # First try exact database query
        result = self._db_query_exact(records, tool_key, query)
        if result:
            return result

        # Extract keywords from query and tool name
        query_words = set(query_lower.replace('-', ' ').replace('_', ' ').split())
        tool_words = set(tool_lower.replace('-', ' ').replace('_', ' ').split())
        
        # Combine keywords and add semantic expansions
        all_keywords = query_words | tool_words
        
        # Semantic expansions for common travel terms
        expansions = {
            'budget': ['budget', 'affordable', 'cheap', 'money', 'cost', 'price', 'saving'],
            'seasonal': ['seasonal', 'season', 'weather', 'month', 'time', 'when', 'spring', 'summer', 'fall', 'winter'],
            'travel': ['travel', 'trip', 'visit', 'visitor', 'tourist', 'planning', 'itinerary'],
            'planner': ['planner', 'planning', 'plan', 'guide', 'strategies'],
            'accommodation': ['accommodation', 'hotel', 'stay', 'neighborhood', 'arrondissement', 'district'],
            'finder': ['finder', 'find', 'search', 'match', 'recommend', 'guide'],
            'matcher': ['matcher', 'match', 'find', 'recommend', 'suitable'],
            'attraction': ['attraction', 'landmark', 'museum', 'sight', 'cultural', 'experience'],
            'friendly': ['friendly', 'suitable', 'good', 'recommended'],
        }
        
        expanded_keywords = set()
        for word in all_keywords:
            expanded_keywords.add(word)
            for key, synonyms in expansions.items():
                if word in synonyms:
                    expanded_keywords.update(synonyms)
                    break

        # Score and rank records
        scored_records = []
        for record in records:
            title = record.get("title", "").lower()
            summary = record.get("summary", "").lower()
            full_text = title + " " + summary

            # Calculate relevance score
            score = 0
            matched_terms = set()

            for keyword in expanded_keywords:
                if keyword in title:
                    score += 3  # Title matches are most important
                    matched_terms.add(keyword)
                if keyword in summary:
                    score += 2  # Summary matches are important
                    matched_terms.add(keyword)

            if score > 0:
                scored_records.append((score, len(matched_terms), record))

        # Sort by score (descending) and number of matched terms (descending)
        scored_records.sort(key=lambda x: (x[0], x[1]), reverse=True)

        # Return top matches (up to 5 for relevance)
        result = [record for _, _, record in scored_records[:5]]
        
        # If still no results, return all records as fallback
        if not result:
            return records[:5]
        
        return result

    def _db_query_exact(self, records: List[Dict[str, Any]], tool_key: str, query: str) -> List[Dict[str, Any]]:
        """Fallback to exact matching if semantic matching fails."""
        return [
            r for r in records
            if tool_key in r.get("title", "") or query in r.get("summary", "")
        ]

    def propose_task(self, ctx: SynthesisContext, difficulty: int = 1) -> TaskBundle:
        """Generate a task with solution and verification code."""
        tool_examples = "\n".join([
            f"- {tool['name']}: Call as tools['{tool['name']}']('query') or tools.{tool['name']}('query')"
            for tool in ctx.registry.describe()[:3]
        ])
        prompt = (
            "You are a task generator. Based on the tool list and database, create a verifiable task.\n"
            "Return JSON with name, description, solution_code, verification_code.\n"
            "CRITICAL REQUIREMENTS:\n"
            "1. solution_code MUST ACTUALLY CALL TOOLS using tools['name']('query') or tools.name('query').\n"
            "2. Call at least 2 different tools and combine their results into a structured output.\n"
            "3. Do NOT return trivial results like 'list(tools.keys())' or just tool names.\n"
            "4. It can only access data via tools (no direct DB access).\n"
            "5. verification_code must define verify(tools, answer) and return bool.\n"
            "6. The verification must check answer STRUCTURE (keys exist, types correct), NOT exact values.\n"
            "7. IMPORTANT: Tools return LISTS (not strings). Handle them as lists.\n\n"
            "TOOL RETURN FORMATS:\n"
            "- matcher/finder tools: return list of strings (titles)\n"
            "- recommendation/seasonal tools: return list of strings (summaries)\n"
            "- categorizer/attraction tools: return list of dicts with 'name' and 'info' keys\n\n"
            f"Category: {ctx.category}\n"
            f"Tool usage examples:\n{tool_examples}\n"
            f"All tools: {json.dumps(ctx.registry.describe(), ensure_ascii=False)}\n"
            f"Database samples: {json.dumps(ctx.db.records[:5], ensure_ascii=False)}\n"
            f"Difficulty: {difficulty}\n\n"
            "Example solution pattern:\n"
            "def solve(tools):\n"
            "    # Tools return LISTS, not strings!\n"
            "    neighborhoods = tools['neighborhood_matcher']('family')  # Returns list of titles\n"
            "    seasonal = tools['seasonal_recommendation']('April')  # Returns list of summaries\n"
            "    return {'neighborhoods': neighborhoods, 'seasonal': seasonal}\n\n"
            "Example verification pattern:\n"
            "def verify(tools, answer):\n"
            "    # Check structure, not exact values\n"
            "    if not isinstance(answer, dict): return False\n"
            "    if 'neighborhoods' not in answer: return False\n"
            "    if not isinstance(answer['neighborhoods'], list): return False\n"
            "    return len(answer['neighborhoods']) > 0\n"
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
        tool_list = json.dumps(ctx.registry.describe(), ensure_ascii=False)
        
        # Extract tools used in previous task to force using different ones
        prev_tools = self._extract_tool_calls(prev.solution_code)
        all_tools = [t['name'] for t in ctx.registry.describe()]
        unused_tools = [t for t in all_tools if t not in prev_tools and t not in ['bash', 'search']]
        
        prompt = (
            "Create a COMPLETELY DIFFERENT and MORE DIFFICULT task. CRITICAL REQUIREMENTS:\n\n"
            "1. **DIFFERENT NAME**: Must have a new, unique name (not the same as previous)\n"
            "2. **DIFFERENT APPROACH**: Use different tools and different query parameters\n"
            "3. **MORE COMPLEX LOGIC**: Include loops, conditionals, data aggregation\n"
            "4. **MORE TOOLS**: Call at least 3 different tools\n"
            "5. **RICHER OUTPUT**: Return nested data structures with computed values\n\n"
            f"MUST USE THESE TOOLS (not used before): {unused_tools if unused_tools else 'any available'}\n"
            f"ALL available tools: {tool_list}\n"
            f"Database samples: {json.dumps(ctx.db.records[:3], ensure_ascii=False)}\n\n"
            f"PREVIOUS TASK TO IMPROVE (difficulty {prev.difficulty}):\n"
            f"  Name: {prev.name}\n"
            f"  Description: {prev.description}\n"
            f"  Tools used: {list(prev_tools)}\n\n"
            f"NEW TASK REQUIREMENTS (difficulty {prev.difficulty + 1}):\n"
            "- Name: Create a NEW name reflecting the enhanced complexity\n"
            "- solution_code: Use different tools, different parameters, more processing\n"
            "- verification_code: More thorough checks, validate computed values\n\n"
            "Return JSON with: name, description, solution_code, verification_code, difficulty\n"
        )
        
        # Try up to 3 times to get a different task
        for attempt in range(3):
            raw = ctx.llm.simple_complete(prompt, temperature=0.9 + attempt * 0.05, max_tokens=1200)
            try:
                data = self._parse_json_response(raw)
                new_code = data.get("solution_code", "")
                new_name = data.get("name", "")
                new_desc = data.get("description", "")
                
                # Allow acceptance if ANY of (name/code/description/toolset) differs to avoid fallback spam
                name_changed = bool(new_name) and new_name.strip().lower() != prev.name.strip().lower()
                code_changed = bool(new_code) and new_code.strip() != prev.solution_code.strip()
                desc_changed = bool(new_desc) and new_desc.strip() != prev.description.strip()
                
                # Check tool usage difference to encourage diversity
                new_tools = self._extract_tool_calls(new_code)
                prev_tools = self._extract_tool_calls(prev.solution_code)
                tools_changed = bool(new_tools - prev_tools or prev_tools - new_tools)
                
                # Accept if there is meaningful change in name, code, description, or tool usage
                if name_changed or code_changed or desc_changed or tools_changed:
                    return TaskBundle(
                        name=new_name or f"{prev.name} Advanced",
                        description=new_desc or prev.description,
                        difficulty=data.get("difficulty", prev.difficulty + 1),
                        solution_code=new_code or prev.solution_code,
                        verification_code=data.get("verification_code", prev.verification_code),
                    )
            except json.JSONDecodeError:
                continue
        
        # Fallback: manually create a different task (non-warning to avoid log noise)
        logger.info("LLM did not provide a sufficiently different task; using fallback variant")
        return self._create_fallback_refined_task(ctx, prev)
    
    def _create_fallback_refined_task(self, ctx: SynthesisContext, prev: TaskBundle) -> TaskBundle:
        """Create a fallback refined task when LLM fails to generate a different one."""
        all_tools = [t['name'] for t in ctx.registry.describe() if t['name'] not in ['bash', 'search']]
        prev_tools = list(self._extract_tool_calls(prev.solution_code))
        
        # Build solution that uses all available custom tools
        tool_calls = []
        for tool in all_tools[:3]:
            tool_calls.append(f"    result_{tool} = tools['{tool}']('query')")
        
        solution_code = f"""def solve(tools):
    # Collect data from multiple tools
{chr(10).join(tool_calls)}
    
    # Aggregate results
    combined = {{
        'tool_results': {{{', '.join([f"'{t}': result_{t}" for t in all_tools[:3]])}}},
        'total_items': sum(len(r) if isinstance(r, list) else 1 for r in [{', '.join([f'result_{t}' for t in all_tools[:3]])}]),
        'summary': 'Aggregated data from {len(all_tools[:3])} tools'
    }}
    return combined"""
        
        verification_code = """def verify(tools, answer):
    if not isinstance(answer, dict):
        return False
    required = ['tool_results', 'total_items', 'summary']
    for key in required:
        if key not in answer:
            return False
    if not isinstance(answer['tool_results'], dict):
        return False
    if not isinstance(answer['total_items'], int):
        return False
    return len(answer['tool_results']) > 0"""
        
        return TaskBundle(
            name=f"{prev.name} - Multi-Tool Aggregation v{prev.difficulty + 1}",
            description=f"Enhanced version: aggregate data from multiple tools ({', '.join(all_tools[:3])}) and compute summary statistics.",
            difficulty=prev.difficulty + 1,
            solution_code=solution_code,
            verification_code=verification_code,
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
            use_docker=bundle.use_docker,  # Preserve use_docker flag
            use_sandbox_fusion=bundle.use_sandbox_fusion,  # Preserve use_sandbox_fusion flag
        )

    def ensure_valid(self, ctx: SynthesisContext, bundle: TaskBundle, fail_soft: bool = False) -> Tuple[TaskBundle, Any]:
        """Execute and verify a bundle; repair via LLM when needed. If fail_soft, return last attempt instead of raising."""
        base_tools = ctx.registry.as_callable_dict()

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
                result = ctx.db.records
            else:
                text = candidate if isinstance(candidate, str) else str(candidate)
                result = ctx.db.query("title", text) or [
                    r for r in ctx.db.records if text in r.get("title", "") or text in r.get("summary", "")
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
            base_tools = ctx.registry.as_callable_dict()
            tools = ToolProxy(**base_tools)
            
            try:
                # Use local execution for validation to access real tools/database
                answer = bundle.run_solution(tools, force_local=True)
                valid = bundle.verify(tools, answer, force_local=True)
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

    def _looks_trivial(self, bundle: TaskBundle) -> bool:
        """Heuristic check to reject trivial solution/verifier pairs."""
        sol = bundle.solution_code or ""
        ver = bundle.verification_code or ""
        
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

    def _ensure_substantive_task(self, ctx: SynthesisContext, bundle: TaskBundle, reason: str = "") -> TaskBundle:
        """Repair tasks that are trivial or do not use enough tools."""
        base_reason = reason or "Task too trivial or lacks multiple tool calls"
        for _ in range(3):
            tool_calls = self._extract_tool_calls(bundle.solution_code)
            if not self._looks_trivial(bundle) and len(tool_calls) >= 2:
                return bundle
            bundle = self.repair_bundle(
                ctx,
                bundle,
                f"{base_reason}; tool_calls={list(tool_calls)}"
            )
        return bundle

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
        use_sandbox_fusion: bool = True,
        use_docker: bool = False,
    ) -> List[TaskBundle]:
        """Main entry point for environment + task synthesis.
        
        Args:
            category: Task category
            sandbox: Sandbox directory
            rounds: Number of difficulty refinement rounds
            validate: Whether to validate tasks
            fail_soft: Whether to fail softly (warn instead of raise)
            persist: Whether to persist results
            use_sandbox_fusion: Whether to use SandboxFusion for secure code execution (default: True)
            use_docker: Whether to use Docker for secure code execution (default: False)
        
        Note: use_sandbox_fusion and use_docker are mutually exclusive execution modes.
              If both are enabled, use_sandbox_fusion takes priority.
        """
        print(f"\n{'='*60}")
        print(f"🚀 开始任务合成: {category}")
        print(f"{'='*60}")
        
        # Step 1: Build context
        print(f"\n📁 [1/5] 初始化环境...")
        ctx = self.build_context(category, sandbox, use_sandbox_fusion=use_sandbox_fusion, use_docker=use_docker)
        exec_mode = "SandboxFusion" if use_sandbox_fusion else ("Docker" if use_docker else "本地")
        print(f"   ✓ 沙箱目录: {sandbox}")
        print(f"   ✓ 执行模式: {exec_mode}")
        
        # Step 2: Seed database
        print(f"\n📊 [2/5] 生成数据库记录...")
        self.seed_database(ctx)
        print(f"   ✓ 数据库记录数: {len(ctx.db.records)}")
        
        # Step 3: Synthesize tools
        print(f"\n🔧 [3/5] 合成工具集...")
        self.synthesize_tools(ctx)
        tool_names = [t.name for t in ctx.registry.tools.values()]
        print(f"   ✓ 生成工具: {', '.join(tool_names)}")

        bundles: List[TaskBundle] = []
        
        # Step 4: Generate initial task
        print(f"\n📝 [4/5] 生成任务 (共 {rounds} 轮)...")
        print(f"\n   --- 第 1 轮 (难度 1) ---")
        print(f"   ⏳ 生成初始任务...")
        current = self._ensure_substantive_task(ctx, self.propose_task(ctx, difficulty=1), "Initial task quality gate")
        print(f"   ✓ 任务名称: {current.name}")
        
        # Set execution mode flags (SandboxFusion takes priority over Docker)
        if use_sandbox_fusion:
            current.use_sandbox_fusion = True
            current.use_docker = False
        elif use_docker:
            current.use_docker = True
            current.use_sandbox_fusion = False
            
        if validate:
            print(f"   ⏳ 验证任务...")
            try:
                current, answer = self.ensure_valid(ctx, current, fail_soft=fail_soft)
                if answer is not None:
                    print(f"   ✅ 验证通过!")
                else:
                    print(f"   ⚠️  验证失败 (软失败模式)")
            except Exception as e:
                print(f"   ❌ 验证错误: {str(e)[:50]}")
        else:
            print(f"   ⏭️  跳过验证")
        bundles.append(current)

        # Step 5: Refine tasks
        for step in range(1, rounds):
            print(f"\n   --- 第 {step + 1} 轮 (难度 {step + 1}) ---")
            print(f"   ⏳ 生成进阶任务...")
            current = self._ensure_substantive_task(
                ctx,
                self.refine_task(ctx, current),
                f"Refined task quality gate (round {step})"
            )
            print(f"   ✓ 任务名称: {current.name}")
            
            # Set execution mode flags
            if use_sandbox_fusion:
                current.use_sandbox_fusion = True
                current.use_docker = False
            elif use_docker:
                current.use_docker = True
                current.use_sandbox_fusion = False
                
            if validate:
                # Before validating, check if the refined task might need additional tools
                called_tools = self._extract_tool_calls(current.solution_code)
                available_tools = {tool.name for tool in ctx.registry.tools.values()}
                missing_tools = called_tools - available_tools
                
                if missing_tools:
                    print(f"   ⏳ 补充缺失工具: {missing_tools}")
                    self.augment_toolset(ctx, current, f"Task requires tools: {missing_tools}")
                
                print(f"   ⏳ 验证任务...")
                try:
                    current, answer = self.ensure_valid(ctx, current, fail_soft=fail_soft)
                    if answer is not None:
                        print(f"   ✅ 验证通过!")
                    else:
                        print(f"   ⚠️  验证失败 (软失败模式)")
                except Exception as e:
                    print(f"   ❌ 验证错误: {str(e)[:50]}")
            else:
                print(f"   ⏭️  跳过验证")
            bundles.append(current)

        # Final: Persist results
        print(f"\n💾 [5/5] 保存结果...")
        if persist:
            self._persist(ctx, bundles)
            print(f"   ✓ 保存到: {ctx.sandbox / 'tasks.json'}")
        
        print(f"\n{'='*60}")
        print(f"✨ 合成完成! 共生成 {len(bundles)} 个任务")
        for i, b in enumerate(bundles, 1):
            print(f"   [{b.difficulty}] {b.name}")
        print(f"{'='*60}\n")

        return bundles

