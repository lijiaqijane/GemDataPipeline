"""Environment synthesizer main class."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..constraints import CodeValidator, ToolContext
from ..database import LocalDatabase
from ..llm import LLMClient
from ..tools import BashTool, SearchTool, ToolRegistry
from .agents import EnvironmentAgent, ToolAgent, TaskAgent, ValidationAgent
from .context import SynthesisContext
from .task_bundle import TaskBundle
from .utils import parse_json_response

logger = logging.getLogger(__name__)

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
    def _parse_json_response(raw: str, max_retries: int = 3) -> Any:
        """Alias for parse_json_response from utils."""
        return parse_json_response(raw, max_retries)

    @staticmethod
    def _extract_tool_calls(solution_code: str) -> set[str]:
        """Extract tool names used in the solution code using AST analysis."""
        return CodeValidator.extract_tool_calls(solution_code)

    def build_context(self, category: str, sandbox: Path, use_sandbox_fusion: bool = True) -> SynthesisContext:
        sandbox.mkdir(parents=True, exist_ok=True)
        db = LocalDatabase.load(sandbox / "db.json")

        registry = ToolRegistry()
        
        # Configure bash tool (local within sandbox path)
        import os
        bash_tool = BashTool(
            workdir=sandbox,
        )
        search_tool = SearchTool()

        registry.ensure_defaults(bash=bash_tool, search=search_tool)
        
        # Register SandboxFusion executor if enabled (for information only, actual execution is in TaskBundle)
        if use_sandbox_fusion:
            logger.info("SandboxFusion enabled for secure code execution")

        return SynthesisContext(category=category, sandbox=sandbox, db=db, registry=registry, llm=self.llm)

    def seed_database(self, ctx: SynthesisContext) -> None:
        """Step 1: Environment and toolset construction - Generate or retrieve relevant data in a sandbox environment with bash and search tools, and store it in the database.
        
        According to paper design, this step is responsible for:
        1. Using bash and search tools in the sandbox environment
        2. Generating or retrieving relevant data for specific task categories
        3. Storing data in the database
        
        Constraints:
        - Must use search tools to retrieve data from network or knowledge base
        - Can use bash tools to generate mock data
        - All data must be stored in the database
        """
        logger.info(f"Step 1: Starting data generation/retrieval (category: {ctx.category})...")
        
        # Step 1.1: Use search tool to retrieve relevant data
        search_query = f"{ctx.category} sample data list structured information"
        logger.debug(f"Using search tool to retrieve: {search_query}")
        try:
            search_hits = ctx.registry.tools["search"](search_query, max_results=5)
            logger.info(f"Search tool returned {len(search_hits)} results")
        except Exception as exc:  # pragma: no cover - network/API fallback
            logger.warning("Search tool failed, falling back to empty results: %s", exc)
            search_hits = []
        
        # Step 1.2: Optionally use bash tool to generate mock data (if needed)
        # For example: generate test data files, process data, etc.
        bash_commands = []
        if not search_hits:
            # If search has no results, can use bash to generate some basic data
            logger.debug("No search results, considering using bash tool to generate mock data")
            # Can add bash commands here to generate data files, etc.
        
        # Step 1.3: Use LLM to generate structured data based on search results and task category
        # Note: Even if search results are empty, LLM will still generate data based on task category (this is the fallback mechanism)
        data_source = "Search results + task category" if search_hits else "Task category (no search results, using LLM generation)"
        logger.info(f"Step 1.3: Using LLM to generate structured data (data source: {data_source})...")
        
        prompt = (
            "You are a data curation assistant working in a sandbox environment with bash and search tools. "
            "Based on the topic and search hits, produce 5-10 structured records that are relevant to the task category. "
            "Each record should have fields: title, summary, and optionally type, name, or other relevant fields. "
            "Return a JSON array with these records. Avoid duplicates.\n"
            f"Topic/Category: {ctx.category}\n"
            f"Search hits (JSON): {json.dumps(search_hits, ensure_ascii=False)}\n"
            "Make sure the records are diverse and cover different aspects of the topic."
        )
        generated = ctx.llm.simple_complete(prompt, temperature=0.4, max_tokens=600)
        try:
            records = self._parse_json_response(generated)
        except json.JSONDecodeError:
            logger.warning("LLM returned JSON parse failed, using fallback record")
            records = [{"title": ctx.category, "summary": generated[:200]}]
        if isinstance(records, dict):
            records = [records]
        
        logger.info(f"Step 1.3 complete: LLM generated {len(records)} structured records")
        
        # Step 1.4: Store generated data in database
        initial_count = len(ctx.db.records)
        for row in records:
            # Ensure records have basic fields
            if "title" not in row:
                row["title"] = str(row.get("name", ctx.category))
            if "summary" not in row:
                row["summary"] = str(row.get("description", ""))
            ctx.db.add_record(row)
        
        final_count = len(ctx.db.records)
        added_count = final_count - initial_count
        
        logger.info(
            f"Step 1 complete: Generated and stored {added_count} new database records (total {final_count}) to {ctx.db.path}\n"
            f"  Data source: Search tool returned {len(search_hits)} results, LLM generated {len(records)} records based on {'search results and ' if search_hits else ''}task category"
        )

    def _generate_fallback_tools(self, ctx: SynthesisContext) -> List[Dict[str, str]]:
        """Generate fallback tools based on database records and task category.
        
        Use this method to generate basic tools when LLM cannot generate tools.
        """
        category_lower = ctx.category.lower()
        tools = []
        
        # Analyze database records, generate basic query tools
        records = ctx.db.records[:10]
        record_types = set()
        record_keys = set()
        
        for record in records:
            if "type" in record:
                record_types.add(record["type"].lower())
            record_keys.update(record.keys())
        
        # Generate tools based on task category and database structure
        if "travel" in category_lower or "trip" in category_lower or "tour" in category_lower:
            tools.append({
                "name": "get_travel_guides",
                "description": "Get all travel guides from the database. Parameters: query (str, optional) - filter guides by query string"
            })
            tools.append({
                "name": "find_itineraries",
                "description": "Find travel itineraries. Parameters: duration (str, optional) - filter by duration, audience (str, optional) - filter by target audience"
            })
            if "guide" in record_keys or "Guide" in str(records):
                tools.append({
                    "name": "search_guides",
                    "description": "Search for travel guides. Parameters: keyword (str) - search keyword"
                })
            if "itinerary" in str(records).lower() or "Itinerary" in str(records):
                tools.append({
                    "name": "get_itinerary_details",
                    "description": "Get detailed information about an itinerary. Parameters: itinerary_name (str) - name of the itinerary"
                })
        
        # General tools
        if len(records) > 0:
            tools.append({
                "name": "get_all_records",
                "description": "Get all records from the database. Parameters: None"
            })
            tools.append({
                "name": "search_records",
                "description": "Search records by keyword. Parameters: keyword (str) - search keyword to filter records"
            })
        
        # Generate specific tools based on record type field
        if record_types:
            for record_type in list(record_types)[:2]:  # Generate at most 2 type-specific tools
                tool_name = f"get_{record_type.lower().replace(' ', '_')}_records"
                tools.append({
                    "name": tool_name,
                    "description": f"Get all {record_type} records from the database. Parameters: None"
                })
        
        # Ensure at least 2 tools
        if len(tools) < 2:
            tools.extend([
                {
                    "name": "query_database",
                    "description": "Query the database with a search term. Parameters: query (str) - search query"
                },
                {
                    "name": "filter_by_type",
                    "description": "Filter records by type. Parameters: record_type (str) - type to filter by"
                }
            ])
        
        logger.info(f"Generated {len(tools)} fallback tools: {[t['name'] for t in tools]}")
        return tools[:5]  # Return at most 5 tools

    def synthesize_tools(self, ctx: SynthesisContext, additional_context: str = "") -> None:
        """Step 2: Task synthesis - Synthesize a set of task-related tools based on the database, each tool implemented as a function.
        
        According to paper design, this step is responsible for:
        1. Analyzing task requirements based on database
        2. Synthesizing a set of task-related tools, each tool implemented as a function
        3. Tool functions can access database (different from solution function constraints)
        4. Tool functions can call other tool functions
        5. Tool functions must return verifiable results
        """
        context_suffix = f"\nAdditional context: {additional_context}" if additional_context else ""
        
        # Enhanced prompt, requiring generation of at least 3-5 tools
        prompt = (
            "You are a tool synthesis agent. Based on the database records, synthesize 3-5 task-oriented tools. "
            "Each tool should be implemented as a function that can access the database and call other tools. "
            "Return a JSON array with fields name and description. "
            "CRITICAL REQUIREMENTS:\n"
            "1. Generate AT LEAST 3 tools (preferably 4-5 tools) that are relevant to the task category and database records.\n"
            "2. Tools should be diverse and cover different aspects of the task.\n"
            "3. Tools should rely on existing database data or simple logic, not external APIs.\n"
            "4. IMPORTANT: Tools can have multiple parameters (like get_infos_by_hotel(info_keywords: List[str], hotel: str)).\n"
            "5. Tools should accept positional arguments that make sense for the task.\n"
            "6. Examples: get_all_hotels_by_city(city: str), get_infos_by_hotel(info_keywords: List[str], hotel: str), "
            "get_inter_city_transport(from_city: str, to_city: str), find_attractions_by_type(type: str), "
            "get_recommendations_by_criteria(criteria: str).\n"
            "7. Describe the tool's parameters clearly in the description field.\n"
            "8. Tool names should be descriptive and follow naming conventions (get_*, find_*, search_*, filter_*).\n\n"
            f"Topic/Category: {ctx.category}\n"
            f"Database records ({len(ctx.db.records)} total):\n{json.dumps(ctx.db.records[:5], ensure_ascii=False, indent=2)}\n"
            f"Analyze the database structure and generate tools that can query, filter, or process this data.{context_suffix}\n\n"
            "Return ONLY a JSON array, no other text. Example format:\n"
            "[\n"
            '  {"name": "get_travel_guides", "description": "Get all travel guides from the database. Parameters: query (str, optional) - filter by query"},'
            '\n  {"name": "find_itineraries", "description": "Find itineraries matching criteria. Parameters: duration (str, optional) - filter by duration"}'
            "\n]"
        )
        
        # Retry mechanism: try at most 3 times
        max_retries = 3
        tools = []
        for attempt in range(max_retries):
            try:
                raw = ctx.llm.simple_complete(
                    prompt, 
                    temperature=0.6 + attempt * 0.1,  # Gradually increase temperature for diversity
                    max_tokens=800  # Increase token count to ensure enough tools are generated
                )
                parsed = self._parse_json_response(raw)
                logger.debug(f"LLM returned tool definitions (attempt {attempt + 1}/{max_retries}): {raw[:500]}")
                
                if isinstance(parsed, dict):
                    parsed = [parsed]
                elif not isinstance(parsed, list):
                    parsed = []
                
                # Filter out invalid tools
                valid_tools = []
                for tool in parsed:
                    if isinstance(tool, dict) and tool.get("name") and tool.get("description"):
                        # Exclude default tools (bash, search)
                        if tool.get("name") not in ["bash", "search"]:
                            valid_tools.append(tool)
                
                if len(valid_tools) >= 2:  # Need at least 2 valid tools
                    tools = valid_tools
                    logger.info(f"Step 2: Successfully generated {len(tools)} tool definitions (attempt {attempt + 1}/{max_retries})")
                    break
                else:
                    logger.warning(f"Step 2: Insufficient tools generated ({len(valid_tools)} tools, attempt {attempt + 1}/{max_retries}), retrying...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"Tool definition JSON parse failed (attempt {attempt + 1}/{max_retries}): {e}, raw response: {raw[:200]}")
                if attempt == max_retries - 1:
                    tools = []
            except Exception as e:
                logger.error(f"Error during tool generation (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    tools = []
        
        # If all retries failed, generate fallback tools
        if len(tools) < 2:
            logger.warning(f"Step 2: Tool generation failed or insufficient ({len(tools)} tools), using fallback tools")
            tools = self._generate_fallback_tools(ctx)
        
        logger.info(f"Step 2: Will register {len(tools)} custom tools")

        # Create tool context, allowing tools to access database and call other tools
        tool_ctx = ToolContext(db=ctx.db, tools={})

        for spec in tools:
            name = spec.get("name")
            desc = spec.get("description", "")
            if not name:
                continue

            def make_handler(key: str, tool_context: ToolContext):
                def base_handler(*args: Any, **kwargs: Any) -> Any:
                    """Tool function handler: can access database and call other tool functions.
                    
                    According to paper constraints, tool functions can:
                    - Access database (via tool_context.db)
                    - Call other tool functions (via tool_context.tools)
                    - Must return verifiable results
                    
                    Supports multiple parameters, for example:
                    - get_all_hotels_by_city(city: str) - one parameter
                    - get_infos_by_hotel(info_keywords: List[str], hotel: str) - two parameters
                    - get_inter_city_transport(from_city: str, to_city: str) - two parameters
                    """
                    logger.debug("Tool '%s' called with args=%s, kwargs=%s", key, args, kwargs)
                    
                    # Handle multiple parameters case
                    # If tool name contains "by_" or "infos_by", usually needs multiple parameters
                    if "infos_by" in key.lower() or "by_" in key.lower():
                        # For tools like get_infos_by_hotel(info_keywords, hotel)
                        # args[0] is info_keywords (list)
                        # args[1] is hotel (str)
                        if len(args) >= 2:
                            info_keywords = args[0] if isinstance(args[0], list) else [args[0]]
                            entity_name = args[1] if len(args) > 1 else (args[0] if len(args) == 1 else None)
                            
                            # Find corresponding entity from database (tool functions can access database)
                            entity_type = key.split("_by_")[-1] if "_by_" in key else key.split("by_")[-1]
                            entity_records = [r for r in tool_context.get_all_records() 
                                            if entity_type in r.get("type", "").lower() 
                                            and (entity_name in str(r.get("name", "")) or entity_name in str(r.get("title", "")))]
                            
                            if entity_records:
                                entity = entity_records[0]
                                # Return requested fields (verifiable results)
                                result = {k: entity.get(k) for k in info_keywords if k in entity}
                                return result
                            return {}
                    
                    # Handle single parameter case (backward compatible)
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
                        # Tool functions can access database
                        result = tool_context.get_all_records()
                    elif not isinstance(candidate, str):
                        candidate = str(candidate)
                        result = self.smart_db_query(tool_context.get_all_records(), key, candidate)
                    else:
                        result = self.smart_db_query(tool_context.get_all_records(), key, candidate)
                    
                    # According to paper constraints, tool functions must return verifiable results
                    # Return format should be structured (list, dict), not string or None
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
                    elif result is None:
                        # Tool functions must return verifiable results, cannot return None
                        result = []
                    
                    # Ensure returned result is verifiable (list or dict)
                    if not isinstance(result, (list, dict)):
                        logger.warning("Tool '%s' returned non-verifiable result type: %s, converting to list", key, type(result).__name__)
                        result = [result] if result is not None else []
                    
                    logger.debug("Tool '%s' returned %s (verifiable: %s)", key, type(result).__name__, isinstance(result, (list, dict)))
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

            ctx.registry.register(name=name, description=desc, func=make_handler(name, tool_ctx))
            logger.info(f"Step 2: Registered tool '{name}': {desc[:100]}")
        
        # Update tool context, include all registered tools (allow tools to call other tools)
        tool_ctx.tools = ctx.registry.as_callable_dict()
        
        if len(tools) == 0:
            logger.warning("Step 2: No custom tools generated, only default tools (bash, search) available")
        else:
            logger.info(f"Step 2: Successfully registered {len([t for t in ctx.registry.tools.values() if t.name not in ['bash', 'search']])} custom tools")

    def augment_toolset(self, ctx: SynthesisContext, bundle: TaskBundle, failure_reason: str, answer: Any = None) -> bool:
        """Step 3 tool extension: During task difficulty escalation, if existing tools are insufficient to solve, extend the toolset.
        
        According to paper design, during task difficulty escalation, if existing tools are insufficient to solve, the agent will extend the toolset.
        This method implements the tool extension logic, precisely extending tools based on failure analysis results.
        
        Args:
            ctx: Synthesis context
            bundle: Task bundle
            failure_reason: Failure reason
            answer: Solution function output (for analysis)
        
        Returns True if new tools were added.
        """
        # Detailed analysis of failure reason
        failure_analysis = self._analyze_failure(failure_reason, bundle, answer)
        
        # Analyze if tool extension is needed
        solution_code = bundle.solution_code or ""
        called_tools = self._extract_tool_calls(solution_code)
        available_tools = {tool.name for tool in ctx.registry.tools.values()}
        missing_tools = called_tools - available_tools
        
        # Determine if tool extension is needed based on failure type
        needs_augmentation = False
        augmentation_reason = ""
        
        if missing_tools:
            needs_augmentation = True
            augmentation_reason = f"Task code attempted to use non-existent tools: {missing_tools}"
        elif failure_analysis["failure_type"] == "tool_not_found":
            needs_augmentation = True
            augmentation_reason = f"Tool call failed: {failure_analysis['root_cause']}"
        elif failure_analysis["failure_type"] == "verification_failed" and answer is not None:
            # Analyze answer structure to see if specific tools are needed to generate missing fields
            if isinstance(answer, dict):
                missing_fields = self._analyze_missing_fields(bundle, answer, ctx)
                if missing_fields:
                    needs_augmentation = True
                    augmentation_reason = f"Answer missing key fields, may need new tools to generate: {missing_fields}"
            elif isinstance(answer, list) and len(answer) == 0:
                # Empty list may indicate need for data retrieval tools
                needs_augmentation = True
                augmentation_reason = "Answer is empty, may need new data retrieval tools"
        
        if not needs_augmentation:
            logger.debug("Tool extension not needed: failure reason unrelated to tools")
            return False
        
        logger.info("Augmenting toolset: %s", augmentation_reason)
        
        # Generate precise tool extension suggestions based on failure analysis
        tool_suggestion_guidance = self._generate_tool_suggestion_guidance(
            failure_analysis, bundle, missing_tools, answer, ctx
        )
        
        prompt = (
            "Generate 1-2 additional specialized tools to help solve the current task. "
            "Return a JSON array with fields name and description. "
            "Tools should complement existing tools and address the specific needs identified in the failure analysis. "
            "IMPORTANT: Tools can have multiple parameters (like get_infos_by_hotel(info_keywords: List[str], hotel: str)). "
            "Tools should accept positional arguments that make sense for the task. "
            "Describe the tool's parameters clearly in the description field.\n\n"
            f"FAILURE ANALYSIS:\n"
            f"  Type: {failure_analysis['failure_type']}\n"
            f"  Root Cause: {failure_analysis['root_cause']}\n"
            f"  Detailed Analysis: {failure_analysis['detailed_analysis']}\n\n"
            f"TOOL SUGGESTION GUIDANCE:\n{tool_suggestion_guidance}\n\n"
            f"Topic: {ctx.category}\n"
            f"Current tools: {json.dumps(ctx.registry.describe(), ensure_ascii=False, indent=2)}\n"
            f"Task description: {bundle.description[:500]}\n"
            f"Solution code (attempted tools: {list(called_tools)}):\n{bundle.solution_code[:500]}\n"
            f"Missing tools detected: {list(missing_tools) if missing_tools else 'None'}\n"
            f"Database examples: {json.dumps(ctx.db.records[:5], ensure_ascii=False, indent=2)}"
        )
        
        raw = ctx.llm.simple_complete(prompt, temperature=0.6, max_tokens=400)
        try:
            new_tools = self._parse_json_response(raw)
        except json.JSONDecodeError:
            logger.warning("Failed to parse new tools from LLM response")
            return False
        
        if not isinstance(new_tools, list):
            new_tools = [new_tools]
        
        # Create tool context, allowing tools to access database and call other tools
        tool_ctx = ToolContext(db=ctx.db, tools=ctx.registry.as_callable_dict())
        
        added_count = 0
        for spec in new_tools:
            name = spec.get("name")
            desc = spec.get("description", "")
            if not name or name in available_tools:
                continue  # Skip if already exists
            
            def make_handler(key: str, tool_context: ToolContext):
                def base_handler(*args: Any, **kwargs: Any) -> Any:
                    """Tool function handler: can access database and call other tool functions."""
                    logger.debug("Augmented tool '%s' called with args=%s, kwargs=%s", key, args, kwargs)
                    
                    # Handle multiple parameters case (consistent with logic in synthesize_tools)
                    if "infos_by" in key.lower() or "by_" in key.lower():
                        if len(args) >= 2:
                            info_keywords = args[0] if isinstance(args[0], list) else [args[0]]
                            entity_name = args[1] if len(args) > 1 else (args[0] if len(args) == 1 else None)
                            
                            entity_type = key.split("_by_")[-1] if "_by_" in key else key.split("by_")[-1]
                            entity_records = [r for r in tool_context.get_all_records() 
                                            if entity_type in r.get("type", "").lower() 
                                            and (entity_name in str(r.get("name", "")) or entity_name in str(r.get("title", "")))]
                            
                            if entity_records:
                                entity = entity_records[0]
                                result = {k: entity.get(k) for k in info_keywords if k in entity}
                                return result
                            return {}
                    
                    # Handle single parameter case (backward compatible)
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
                        result = tool_context.get_all_records()
                    elif not isinstance(candidate, str):
                        candidate = str(candidate)
                        result = self.smart_db_query(tool_context.get_all_records(), key, candidate)
                    else:
                        result = self.smart_db_query(tool_context.get_all_records(), key, candidate)
                    
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
            
            ctx.registry.register(name=name, description=desc, func=make_handler(name, tool_ctx))
            added_count += 1
            logger.info("Added new tool: %s - %s", name, desc)
        
        # Update tool context, include all registered tools (allow tools to call other tools)
        tool_ctx.tools = ctx.registry.as_callable_dict()
        
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

    def _validate_llm_code_response(self, code: str, response_type: str = "solution") -> tuple[bool, str]:
        """Validate if LLM-generated code complies with paper constraints, using AST analysis instead of regex.
        
        Returns:
            (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Code is empty"
        
        if response_type == "solution":
            return CodeValidator.validate_solution_code(code)
        elif response_type == "verification":
            return CodeValidator.validate_verification_code(code)
        else:
            return False, f"Unknown response type: {response_type}"

    def propose_task(self, ctx: SynthesisContext, difficulty: int = 1) -> TaskBundle:
        """Step 3: Solution generation - Construct simple task, corresponding solution function and verification function.
        
        According to paper design, constraints for this step:
        1. Solution functions can only call tool functions or perform logical calculations
        2. Cannot access database or call other functions
        3. Solution output must pass verification function check
        4. If verification fails, will repeatedly modify solution or verification function
        """
        tool_examples = "\n".join([
            f"- {tool['name']}: Call as tools['{tool['name']}']('query') or tools.{tool['name']}('query')"
            for tool in ctx.registry.describe()[:3]
        ])
        prompt = (
            "You are a task generator following the paper's design. Based on the tool list and database, create a verifiable task.\n"
            "Return JSON with name, description, solution_code, verification_code.\n\n"
            "TASK DESCRIPTION REQUIREMENTS (must be detailed and comprehensive):\n"
            "The 'description' field must include:\n"
            "1. **Task Objective**: Clear statement of what needs to be accomplished\n"
            "2. **Input Requirements**: What information or data is expected as input (if any)\n"
            "3. **Expected Output Format**: Detailed description of the output structure (dict keys, list items, data types)\n"
            "4. **Task Constraints**: Any limitations or special requirements\n"
            "5. **Example Use Case**: A concrete scenario illustrating the task\n"
            "6. **Success Criteria**: How to determine if the task is completed successfully\n"
            "The description should be at least 100 words and provide enough context for understanding the task.\n\n"
            "CRITICAL REQUIREMENTS (Code will be validated by AST analysis, non-compliant code will be rejected):\n"
            "1. solution_code MUST ACTUALLY CALL TOOLS using tools['name']('query') or tools.name('query').\n"
            "2. Call at least 2 different tools and combine their results into a structured output.\n"
            "3. Do NOT return trivial results like 'list(tools.keys())' or just tool names.\n"
            "4. solution_code can ONLY call tool functions or perform logical calculations.\n"
            "5. solution_code CANNOT access database directly (AST will detect db, database, ctx.db variable access).\n"
            "6. solution_code CANNOT define other functions (AST will detect, only solve function is allowed).\n"
            "7. solution_code CANNOT import modules (AST will detect import/from import statements).\n"
            "8. verification_code must define verify(tools, answer) and return bool.\n"
            "9. verification_code CAN access database and all information (unlike solution_code).\n"
            "10. The verification must check answer STRUCTURE (keys exist, types correct), NOT exact values.\n"
            "11. IMPORTANT: Tools return LISTS (not strings). Handle them as lists.\n\n"
            "TOOL PARAMETERS:\n"
            "- Tools can have multiple parameters (like get_infos_by_hotel(info_keywords: List[str], hotel: str))\n"
            "- Tools can have single parameter (like get_all_hotels_by_city(city: str))\n"
            "- Tools can have no parameters (like get_all_cities())\n"
            "- Always pass parameters as positional arguments: tools['tool_name'](arg1, arg2, ...)\n\n"
            "TOOL RETURN FORMATS:\n"
            "- matcher/finder tools: return list of strings (titles)\n"
            "- recommendation/seasonal tools: return list of strings (summaries)\n"
            "- categorizer/attraction tools: return list of dicts with 'name' and 'info' keys\n"
            "- infos_by_* tools: return dict with requested fields\n\n"
            f"Category: {ctx.category}\n"
            f"Difficulty Level: {difficulty}\n"
            f"  - Difficulty 1: Simple tasks requiring basic tool calls\n"
            f"  - Difficulty 2: Moderate tasks requiring multiple tools and data aggregation\n"
            f"  - Difficulty 3+: Complex tasks requiring advanced logic, filtering, and computation\n\n"
            f"Available Tools (with detailed descriptions):\n{json.dumps(ctx.registry.describe(), ensure_ascii=False, indent=2)}\n\n"
            f"Tool Usage Examples:\n{tool_examples}\n\n"
            f"Database Context (sample records):\n{json.dumps(ctx.db.records[:5], ensure_ascii=False, indent=2)}\n"
            f"Total database records: {len(ctx.db.records)}\n\n"
            "Example solution pattern (ONLY tool calls and logic, NO database access, NO other functions):\n"
            "def solve(tools):\n"
            "    # Tools can have multiple parameters!\n"
            "    # Example with single parameter:\n"
            "    hotels = tools['get_all_hotels_by_city']('Paris')  # Returns list\n"
            "    # Example with multiple parameters:\n"
            "    hotel_info = tools['get_infos_by_hotel'](['price_per_night', 'rating'], 'Hotel Name')  # Returns dict\n"
            "    # Can perform logical calculations\n"
            "    return {'hotels': hotels, 'info': hotel_info}\n\n"
            "Example verification pattern (CAN access database and all information):\n"
            "def verify(tools, answer):\n"
            "    # Verification can access database and all information\n"
            "    # Check structure, not exact values\n"
            "    if not isinstance(answer, dict): return False\n"
            "    if 'neighborhoods' not in answer: return False\n"
            "    if not isinstance(answer['neighborhoods'], list): return False\n"
            "    return len(answer['neighborhoods']) > 0\n"
        )
        # LLM call with retry, validate response to prevent hallucinations
        max_retries = 3
        for attempt in range(max_retries):
            raw = ctx.llm.simple_complete(prompt, temperature=0.6, max_tokens=800)
            try:
                parsed = self._parse_json_response(raw)
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"LLM response parsing failed (attempt {attempt + 1}/{max_retries}): {e}")
                    continue
                # Last attempt failed, use fallback
                parsed = {
                    "name": f"{ctx.category}-task",
                    "description": raw[:200],
                    "solution_code": "def solve(tools):\n    return list(tools.keys())",
                    "verification_code": "def verify(tools, answer):\n    return isinstance(answer, list)",
                }
            
            # Validate generated code
            solution_code = parsed.get("solution_code", "")
            verification_code = parsed.get("verification_code", "")
            
            sol_valid, sol_error = self._validate_llm_code_response(solution_code, "solution")
            ver_valid, ver_error = self._validate_llm_code_response(verification_code, "verification")
            
            if sol_valid and ver_valid:
                return TaskBundle(
                    name=parsed.get("name", "generated-task"),
                    description=parsed.get("description", ""),
                    difficulty=difficulty,
                    solution_code=solution_code,
                    verification_code=verification_code,
                )
            else:
                if attempt < max_retries - 1:
                    logger.warning(f"LLM generated invalid code (attempt {attempt + 1}/{max_retries}): solution={sol_error}, verification={ver_error}")
                    # Add error information to prompt, request regeneration
                    prompt += f"\n\nPrevious attempt failed validation:\n- Solution: {sol_error}\n- Verification: {ver_error}\nPlease fix these issues."
                    continue
        
        # All retries failed, return fallback (will be fixed by subsequent _ensure_substantive_task)
        logger.error("All LLM retries failed, using fallback task")
        return TaskBundle(
            name=parsed.get("name", "generated-task"),
            description=parsed.get("description", ""),
            difficulty=difficulty,
            solution_code=solution_code,
            verification_code=verification_code,
        )

    def refine_task(self, ctx: SynthesisContext, prev: TaskBundle) -> TaskBundle:
        """Step 3 task difficulty escalation: Gradually increase task difficulty while maintaining verifiability.
        
        According to paper design, during task difficulty escalation:
        1. Start with simple tasks, after verification passes, increase task complexity
        2. Add constraints, task scale, data volume
        3. If existing tools are insufficient to solve, will extend toolset (handled in ensure_valid)
        """
        tool_list = json.dumps(ctx.registry.describe(), ensure_ascii=False)
        
        # Extract tools used in previous task to force using different ones
        prev_tools = self._extract_tool_calls(prev.solution_code)
        all_tools = [t['name'] for t in ctx.registry.describe()]
        unused_tools = [t for t in all_tools if t not in prev_tools and t not in ['bash', 'search']]
        
        tool_examples = "\n".join([
            f"- {tool['name']}: {tool.get('description', '')}"
            for tool in ctx.registry.describe()[:5]
        ])
        
        prompt = (
            "Create a COMPLETELY DIFFERENT and MORE DIFFICULT task. CRITICAL REQUIREMENTS:\n\n"
            "TASK DESCRIPTION REQUIREMENTS (must be detailed and comprehensive, at least 150 words):\n"
            "The 'description' field must include ALL of the following 6 components:\n"
            "1. **Task Objective**: Clear statement of what needs to be accomplished (more complex than previous task)\n"
            "2. **Input Requirements**: What information or data is expected as input (if any)\n"
            "3. **Expected Output Format**: Detailed description of the output structure (dict keys, list items, data types, nested structures)\n"
            "4. **Task Constraints**: Any limitations or special requirements (must be more complex than previous)\n"
            "5. **Example Use Case**: A concrete scenario illustrating the enhanced complexity of this task\n"
            "6. **Success Criteria**: How to determine if the task is completed successfully (more thorough than previous)\n\n"
            "TASK COMPLEXITY REQUIREMENTS:\n"
            "1. **DIFFERENT NAME**: Must have a new, unique name (not the same as previous)\n"
            "2. **DIFFERENT APPROACH**: Use different tools and different query parameters\n"
            "3. **MORE COMPLEX LOGIC**: Include loops, conditionals, data aggregation, filtering, computation\n"
            "4. **MORE TOOLS**: Call at least 3 different tools (preferably 4-5 tools)\n"
            "5. **RICHER OUTPUT**: Return nested data structures with computed values, statistics, or aggregations\n"
            "6. **ENHANCED VERIFICATION**: Verification code should check more aspects: structure, types, computed values, data relationships\n\n"
            "CODE REQUIREMENTS:\n"
            "1. solution_code can ONLY call tool functions or perform logical calculations\n"
            "2. solution_code CANNOT access database directly\n"
            "3. solution_code CANNOT define other functions (only solve function)\n"
            "4. solution_code CANNOT import modules\n"
            "5. verification_code CAN access database and all information\n"
            "6. verification_code must return bool\n"
            "7. Tools return LISTS or DICTS, handle them appropriately\n\n"
            f"MUST USE THESE TOOLS (not used before): {unused_tools if unused_tools else 'any available tools'}\n"
            f"ALL available tools:\n{json.dumps(ctx.registry.describe(), ensure_ascii=False, indent=2)}\n\n"
            f"Tool Usage Examples:\n{tool_examples}\n\n"
            f"Database Context:\n{json.dumps(ctx.db.records[:5], ensure_ascii=False, indent=2)}\n"
            f"Total database records: {len(ctx.db.records)}\n\n"
            f"PREVIOUS TASK TO IMPROVE (difficulty {prev.difficulty}):\n"
            f"  Name: {prev.name}\n"
            f"  Description: {prev.description[:300]}...\n"
            f"  Tools used: {list(prev_tools)}\n\n"
            f"NEW TASK REQUIREMENTS (difficulty {prev.difficulty + 1}):\n"
            "- Name: Create a NEW name reflecting the enhanced complexity\n"
            "- Description: MUST include all 6 components listed above, be detailed (150+ words)\n"
            "- solution_code: Use different tools, different parameters, more processing, more complex logic\n"
            "- verification_code: More thorough checks, validate computed values, check data relationships\n\n"
            "Return JSON with: name, description, solution_code, verification_code, difficulty\n"
            "The description field MUST be comprehensive and include all 6 required components.\n"
        )
        
        # Try up to 3 times to get a different task
        for attempt in range(3):
            raw = ctx.llm.simple_complete(prompt, temperature=0.7 + attempt * 0.1, max_tokens=1500)
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
                
                # Verify description is detailed enough (at least 100 words)
                desc_word_count = len(new_desc.split()) if new_desc else 0
                desc_is_detailed = desc_word_count >= 100
                
                # Accept if there is meaningful change in name, code, description, or tool usage
                # AND description is detailed enough
                if (name_changed or code_changed or desc_changed or tools_changed) and desc_is_detailed:
                    return TaskBundle(
                        name=new_name or f"{prev.name} Advanced",
                        description=new_desc or prev.description,
                        difficulty=data.get("difficulty", prev.difficulty + 1),
                        solution_code=new_code or prev.solution_code,
                        verification_code=data.get("verification_code", prev.verification_code),
                    )
                elif attempt < 2:  # Not last attempt
                    logger.debug(f"Refined task description too short ({desc_word_count} words) or insufficiently different, retrying...")
                    continue
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
        
        # Create detailed task description (includes 6 core parts)
        detailed_description = (
            f"Task Objective: Aggregate data from multiple tools ({', '.join(all_tools[:3]) if all_tools else 'available tools'}) "
            f"and compute summary statistics. This is an enhanced version of the previous task with increased complexity. "
            f"Input Requirements: No specific input required - the task uses available tools to query and aggregate data. "
            f"Expected Output Format: A dictionary with three keys: 'tool_results' (dict containing results from each tool), "
            f"'total_items' (integer count of total items aggregated), and 'summary' (string description of the aggregation). "
            f"Task Constraints: Must use at least {len(all_tools[:3]) if all_tools else 2} different tools, cannot access database directly, "
            f"cannot define additional functions or import modules. Example Use Case: A data analyst needs to aggregate information "
            f"from multiple sources and compute statistics for reporting. Success Criteria: The solution must successfully call multiple tools, "
            f"aggregate their results into a structured format, compute total item count, and the verification must confirm the output "
            f"has correct structure with all required keys and appropriate data types."
        )
        
        return TaskBundle(
            name=f"{prev.name} - Multi-Tool Aggregation v{prev.difficulty + 1}",
            description=detailed_description,
            difficulty=prev.difficulty + 1,
            solution_code=solution_code,
            verification_code=verification_code,
        )

    def _analyze_failure(self, failure_reason: str, bundle: TaskBundle, answer: Any = None) -> Dict[str, Any]:
        """Detailed analysis of verification failure reasons.
        
        Returns:
            Dictionary containing failure type, reason, and suggested fix
        """
        failure_reason_lower = failure_reason.lower()
        analysis = {
            "failure_type": "unknown",
            "root_cause": failure_reason[:500],
            "detailed_analysis": "",
            "suggested_fix": "",
            "affected_component": "unknown",  # solution, verification, or both
        }
        
        # Classify failure type
        if "verification returned false" in failure_reason_lower:
            analysis["failure_type"] = "verification_failed"
            analysis["affected_component"] = "verification"
            analysis["detailed_analysis"] = (
                "Verification function returned False. Possible reasons:\n"
                "1. Answer data structure does not match verification function expectations\n"
                "2. Verification logic is too strict or incorrect\n"
                "3. Answer missing required fields or incorrect types"
            )
            if answer is not None:
                analysis["detailed_analysis"] += f"\n\nActual answer: {str(answer)[:200]}"
            analysis["suggested_fix"] = (
                "Check if verification function correctly validates answer structure. "
                "Verification should check existence and types of key fields, not exact values. "
                "If answer structure is reasonable, consider relaxing verification conditions."
            )
        
        elif any(keyword in failure_reason_lower for keyword in ["not found", "missing", "keyerror", "attributeerror"]):
            analysis["failure_type"] = "tool_not_found"
            analysis["affected_component"] = "solution"
            # Extract tool name (if possible)
            import re
            tool_match = re.search(r"['\"]([^'\"]+)['\"]", failure_reason)
            tool_name = tool_match.group(1) if tool_match else "unknown"
            analysis["detailed_analysis"] = (
                f"Tool call failed. Tool '{tool_name}' does not exist or is not accessible.\n"
                "Possible reasons:\n"
                "1. Tool name spelling error\n"
                "2. Tool not yet registered\n"
                "3. Tool parameter mismatch"
            )
            analysis["suggested_fix"] = (
                f"Check if tool name is correct. "
                f"Ensure using correct tool name and parameter format."
            )
        
        elif any(keyword in failure_reason_lower for keyword in ["cannot access", "database", "ctx.db", "db."]):
            analysis["failure_type"] = "database_access_violation"
            analysis["affected_component"] = "solution"
            analysis["detailed_analysis"] = (
                "Solution function attempted to directly access database, which violates constraints.\n"
                "According to paper design, solution functions can only access data through tool functions."
            )
            analysis["suggested_fix"] = (
                "Remove all direct database access code. "
                "Use tool functions (e.g., tools['get_all_hotels_by_city']('Paris')) to get data."
            )
        
        elif any(keyword in failure_reason_lower for keyword in ["syntax error", "invalid syntax", "indentation"]):
            analysis["failure_type"] = "syntax_error"
            analysis["affected_component"] = "both"
            analysis["detailed_analysis"] = "Code contains syntax errors."
            analysis["suggested_fix"] = "Fix syntax errors, ensure code complies with Python syntax standards."
        
        elif any(keyword in failure_reason_lower for keyword in ["typeerror", "type error", "cannot convert"]):
            analysis["failure_type"] = "type_error"
            analysis["affected_component"] = "solution"
            analysis["detailed_analysis"] = (
                "Type error: Tool returned data type does not match expectations.\n"
                "Tools usually return list or dict, need to properly handle these types."
            )
            analysis["suggested_fix"] = (
                "Check tool return data types. "
                "Use isinstance() to check types and appropriately handle list and dict."
            )
        
        elif "timeout" in failure_reason_lower or "execution time" in failure_reason_lower:
            analysis["failure_type"] = "timeout"
            analysis["affected_component"] = "solution"
            analysis["detailed_analysis"] = "Code execution timeout. May be due to overly complex logic or infinite loop."
            analysis["suggested_fix"] = "Simplify logic, reduce loop iterations, or optimize algorithm."
        
        else:
            # Default analysis
            analysis["failure_type"] = "runtime_error"
            analysis["affected_component"] = "both"
            analysis["detailed_analysis"] = f"Runtime error: {failure_reason[:300]}"
            analysis["suggested_fix"] = "Check code logic, ensure all variables are defined, all function calls are correct."
        
        return analysis

    def _analyze_missing_fields(self, bundle: TaskBundle, answer: Dict[str, Any], ctx: SynthesisContext) -> List[str]:
        """Analyze potentially missing fields in the answer."""
        # Extract expected fields from task description
        description = bundle.description.lower()
        common_fields = ["name", "title", "description", "summary", "price", "rating", "location", "type", "category"]
        missing = []
        
        for field in common_fields:
            if field in description and field not in answer:
                missing.append(field)
        
        return missing

    def _generate_tool_suggestion_guidance(
        self, 
        failure_analysis: Dict[str, Any], 
        bundle: TaskBundle, 
        missing_tools: set, 
        answer: Any,
        ctx: SynthesisContext
    ) -> str:
        """Generate tool extension suggestions based on failure analysis."""
        guidance_parts = []
        
        if missing_tools:
            guidance_parts.append(
                f"Task code attempted to call the following non-existent tools: {list(missing_tools)}. "
                "Please generate these tools or functionally equivalent alternatives."
            )
        
        if failure_analysis["failure_type"] == "verification_failed":
            guidance_parts.append(
                "Verification failure may be due to missing key fields or incorrect structure in the answer. "
                "Please generate tools that can produce complete answer structures, for example: "
                "- Data aggregation tools (aggregate_data, combine_results) "
                "- Field extraction tools (extract_field, get_details) "
                "- Formatting tools (format_output, structure_data)"
            )
        
        if failure_analysis["failure_type"] == "tool_not_found":
            guidance_parts.append(
                "Tool call failed. Please check if needed: "
                "- Data retrieval tools (get_by_keyword, search_records) "
                "- Filtering tools (filter_by_criteria, match_condition) "
                "- Calculation tools (calculate_statistics, compute_value)"
            )
        
        if answer is not None:
            if isinstance(answer, list) and len(answer) == 0:
                guidance_parts.append(
                    "Answer is empty list, may need data retrieval or query tools to fetch data."
                )
            elif isinstance(answer, dict) and len(answer) == 0:
                guidance_parts.append(
                    "Answer is empty dict, may need data extraction or generation tools to populate data."
                )
        
        return "\n".join(f"- {part}" for part in guidance_parts) if guidance_parts else "Generate appropriate tools based on failure analysis and task requirements."

    @staticmethod
    def _indent_text(text: str, spaces: int = 4) -> str:
        """Add indentation to text."""
        indent = " " * spaces
        return "\n".join(indent + line for line in text.split("\n"))

    def repair_bundle(self, ctx: SynthesisContext, bundle: TaskBundle, failure_reason: str, answer: Any = None) -> TaskBundle:
        """Step 3 iterative optimization: When verification fails, repeatedly modify solution or verification function.
        
        According to paper design, if verification fails, the agent will repeatedly modify solution or verification function.
        This method implements the iterative optimization logic.
        """
        # Detailed analysis of failure reason
        failure_analysis = self._analyze_failure(failure_reason, bundle, answer)
        
        # Build detailed repair guidance
        detailed_guidance = (
            f"FAILURE ANALYSIS:\n"
            f"  Type: {failure_analysis['failure_type']}\n"
            f"  Affected Component: {failure_analysis['affected_component']}\n"
            f"  Root Cause: {failure_analysis['root_cause']}\n"
            f"  Detailed Analysis:\n{self._indent_text(failure_analysis['detailed_analysis'], 4)}\n"
            f"  Suggested Fix: {failure_analysis['suggested_fix']}\n\n"
        )
        
        prompt = (
            "The current solution or verification failed. Following the paper's design, repair the bundle. "
            "Produce a new JSON with name, description, solution_code, verification_code.\n\n"
            f"{detailed_guidance}"
            "CRITICAL CONSTRAINTS (Code will be validated by AST analysis, non-compliant code will be rejected):\n"
            "1. solve(tools) can ONLY call tool functions or perform logical calculations.\n"
            "2. solve(tools) CANNOT access database directly (AST will detect db, database, ctx.db variables).\n"
            "3. solve(tools) CANNOT define other functions (AST will detect, only solve function is allowed).\n"
            "4. solve(tools) CANNOT import modules (AST will detect import/from import statements).\n"
            "5. verify(tools, answer) CAN access database and all information (unlike solution_code).\n"
            "6. verify(tools, answer) must return bool.\n"
            "7. Keep tool calls simple: positional string or keyword 'query' only; avoid extra kwargs.\n"
            "8. Tools return LISTS or DICTS, handle them appropriately.\n\n"
            f"Original failure reason: {failure_reason[:300]}\n"
            f"Original task name: {bundle.name}\n"
            f"Original description: {bundle.description}\n"
            f"Original solution_code:\n{bundle.solution_code}\n"
            f"Original verification_code:\n{bundle.verification_code}\n"
            f"Available tools: {json.dumps(ctx.registry.describe(), ensure_ascii=False)}\n"
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
            use_sandbox_fusion=bundle.use_sandbox_fusion,  # Preserve use_sandbox_fusion flag
        )

    def ensure_valid(self, ctx: SynthesisContext, bundle: TaskBundle, fail_soft: bool = False) -> Tuple[TaskBundle, Any]:
        """Step 3 iterative optimization: Execute and verify tasks, implementing iterative optimization logic from the paper.
        
        According to paper design, this method implements:
        1. Execute solution function and verify results
        2. If verification fails, repeatedly modify solution or verification function (via repair_bundle)
        3. If existing tools are insufficient to solve, extend toolset (via augment_toolset)
        4. During task difficulty escalation, if existing tools are insufficient, will extend toolset
        
        If fail_soft, return last attempt instead of raising.
        
        Note: Since all code execution is performed in SandboxFusion, only need to pass tool name dictionary here.
        _run_in_sandbox_fusion will use tool names to create ToolProxy inside SandboxFusion.
        """
        # Ensure the task is not trivial before running executions
        bundle = self._ensure_substantive_task(ctx, bundle, "Initial validation quality gate")

        # Since code executes in SandboxFusion, only need to pass tool name dictionary
        # _run_in_sandbox_fusion will use these names to create ToolProxy inside SandboxFusion
        tools: Dict[str, Any] = ctx.registry.as_callable_dict()
        last_error = ""
        augmentation_attempted = False

        for attempt in range(self.max_validation_rounds + 1):
            # Refresh tools dict after potential augmentation
            tools = ctx.registry.as_callable_dict()

            try:
                # All execution must happen in SandboxFusion
                if not getattr(bundle, "use_sandbox_fusion", True):
                    raise RuntimeError("Bundle must have use_sandbox_fusion=True for execution")
                # Pass database records to tool execution
                answer = bundle.run_solution(tools, db_records=ctx.db.records)
                valid = bundle.verify(tools, answer, db_records=ctx.db.records)
            except Exception as exc:  # pragma: no cover - runtime defense
                last_error = str(exc)
                logger.warning("Task %s raised during execution: %s", bundle.name, last_error)
                valid = False
            else:
                if not valid:
                    last_error = f"verification returned False. Answer was: {str(answer)[:200] if answer is not None else 'None'}"

            if valid:
                return bundle, answer

            # Try augmenting toolset if we haven't tried yet and we're past first attempt
            if attempt >= 1 and not augmentation_attempted:
                answer_for_analysis = answer if 'answer' in locals() else None
                if self.augment_toolset(ctx, bundle, last_error, answer_for_analysis):
                    augmentation_attempted = True
                    logger.info("Toolset augmented, retrying validation...")
                    continue  # Retry with augmented tools

            bundle = self.repair_bundle(ctx, bundle, last_error or "unknown failure", answer if 'answer' in locals() else None)
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
        """Check if solution function complies with paper constraints: use AST analysis instead of regex.
        
        According to paper design, solution function must:
        1. Only call tool functions or perform logical calculations
        2. Cannot access database
        3. Cannot define other functions (only solve function allowed)
        4. Cannot import modules
        5. Must actually call tools
        """
        sol = bundle.solution_code or ""
        ver = bundle.verification_code or ""
        
        # Use AST analysis to validate solution function
        is_valid, error = CodeValidator.validate_solution_code(sol)
        if not is_valid:
            logger.debug(f"Solution code validation failed: {error}")
            return True  # Does not comply with constraints, treat as trivial
        
        # Check for trivial solution patterns
        for pat in self._trivial_solution_patterns:
            if re.search(pat, sol):
                return True
        
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
        """Persist synthesis results, output format strictly conforms to paper's <environment, tools, task, verifier> quadruple.
        
        According to paper design, generated data format is a quadruple:
        - environment: Environment description (includes category and database records)
        - tools:       Toolset (each tool includes name and description)
        - task:        Task definition (includes name, description, difficulty, solution_code)
        - verifier:    Verifier (includes verification_code)
        
        Output JSON structure:
        {
            "environment": {  # Environment description
                "category": "...",
                "records": [...]  # Database records
            },
            "tools": [  # Toolset
                {"name": "...", "description": "..."},
                ...
            ],
            "tasks": [  # Task list (each task includes task and verifier)
                {
                    "name": "...",
                    "description": "...",
                    "difficulty": 1,
                    "solution_code": "...",  # task part
                    "verification_code": "..."  # verifier part
                },
                ...
            ]
        }
        
        For backward compatibility, still retain original category / tooling / records / tasks fields.
        """
        # Build standard quadruple format: <environment, tools, task, verifier>
        # Each task includes task (solution_code) and verifier (verification_code)
        tasks_with_verifiers = []
        for bundle in bundles:
            task_entry = {
                # Task part: task definition
                "task": {
                    "name": bundle.name,
                    "description": bundle.description,
                    "difficulty": bundle.difficulty,
                    "solution_code": bundle.solution_code,
                },
                # Verifier part: verifier definition
                "verifier": {
                    "verification_code": bundle.verification_code,
                },
                # Retain complete information (backward compatible)
                "name": bundle.name,
                "description": bundle.description,
                "difficulty": bundle.difficulty,
                "solution_code": bundle.solution_code,
                "verification_code": bundle.verification_code,
            }
            tasks_with_verifiers.append(task_entry)
        
        payload = {
            # Standard quadruple format
            "environment": {
                "category": ctx.category,
                "records": ctx.db.records,
                "record_count": len(ctx.db.records),
            },
            "tools": ctx.registry.describe(),
            "tasks": tasks_with_verifiers,
            
            # Compatible fields (backward compatible)
            "category": ctx.category,
            "tooling": ctx.registry.describe(),  # Same as tools
            "records": ctx.db.records,  # Same as environment.records
            
            # Metadata
            "metadata": {
                "version": "1.0",
                "format": "quadruple",  # <environment, tools, task, verifier>
                "task_count": len(bundles),
                "tool_count": len(ctx.registry.tools),
                "generation_timestamp": datetime.now().isoformat(),
            },
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
    ) -> List[TaskBundle]:
        """Main entry point: Perform environment + task synthesis following the three-step process from the paper design.
        
        According to the paper's "specific steps for constructing generated tasks", this method strictly follows three steps:
        
        Step 1: Environment and toolset construction
        - Performed in a sandbox environment with bash and search tools
        - Generate or retrieve relevant data for specific task categories and store in database
        
        Step 2: Task synthesis
        - Based on database, synthesize a set of task-related tools, each tool implemented as a function
        
        Step 3: Solution generation (iterative optimization)
        - Construct simple tasks, corresponding solution functions and verification functions
        - Solution functions can only call tool functions or perform logical calculations, cannot access database
        - If verification fails, repeatedly modify solution or verification function
        - During task difficulty escalation, if existing tools are insufficient, extend toolset
        
        Args:
            category: Task category
            sandbox: Sandbox directory
            rounds: Number of difficulty refinement rounds
            validate: Whether to validate tasks
            fail_soft: Whether to fail softly (warn instead of raise)
            persist: Whether to persist results
            use_sandbox_fusion: Whether to use SandboxFusion for secure code execution (default: True)
        """
        print(f"\n{'='*60}")
        print(f"🚀 Starting task synthesis: {category}")
        print(f"{'='*60}")
        
        # Step 1: Build context
        # Initialize sub-Agents (environment / tools / tasks / validation), forming a general multi-stage Agent pipeline
        env_agent = EnvironmentAgent(self)
        tool_agent = ToolAgent(self)
        task_agent = TaskAgent(self)
        validator = ValidationAgent(self)

        print(f"\n📁 [Step 1] Environment and toolset construction...")
        # Step 1: Generate or retrieve relevant data in sandbox environment with bash and search tools, store in database
        ctx = self.build_context(category, sandbox, use_sandbox_fusion=use_sandbox_fusion)
        exec_mode = "SandboxFusion" if use_sandbox_fusion else "Local"
        print(f"   ✓ Sandbox directory: {sandbox}")
        print(f"   ✓ Execution mode: {exec_mode}")

        # Step 1 (continued): Generate database records
        print(f"\n📊 [Step 1 continued] Generating database records...")
        ctx = env_agent.prepare(category, sandbox, use_sandbox_fusion)
        print(f"   ✓ Database record count: {len(ctx.db.records)}")
        
        # Step 2: Task synthesis - Based on database, synthesize a set of task-related tools
        print(f"\n🔧 [Step 2] Task synthesis - Synthesizing toolset...")
        tool_agent.build_initial_tools(ctx)
        tool_names = [t.name for t in ctx.registry.tools.values()]
        print(f"   ✓ Generated tools: {', '.join(tool_names)}")

        bundles: List[TaskBundle] = []
        
        # Step 3: Solution generation - Construct simple tasks, corresponding solution functions and verification functions
        print(f"\n📝 [Step 3] Solution generation (total {rounds} rounds, iterative optimization)...")
        print(f"\n   --- Round 1 (Difficulty 1) ---")
        print(f"   ⏳ Generating initial task...")
        current = task_agent.propose_initial(ctx, difficulty=1)
        print(f"   ✓ Task name: {current.name}")
        
        # Set execution mode flags
        if use_sandbox_fusion:
            current.use_sandbox_fusion = True
            
        if validate:
            print(f"   ⏳ Validating task...")
            try:
                current, answer = validator.ensure_valid(ctx, current, fail_soft=fail_soft)
                if answer is not None:
                    print(f"   ✅ Validation passed!")
                else:
                    print(f"   ⚠️  Validation failed (soft fail mode)")
            except Exception as e:
                print(f"   ❌ Validation error: {str(e)[:50]}")
        else:
            print(f"   ⏭️  Skipping validation")
        bundles.append(current)

        # Step 3 (continued): Task difficulty escalation - Gradually increase task complexity
        for step in range(1, rounds):
            print(f"\n   --- Round {step + 1} (Difficulty {step + 1}) ---")
            print(f"   ⏳ Generating advanced task (increasing difficulty)...")
            current = task_agent.refine(ctx, current, round_index=step)
            print(f"   ✓ Task name: {current.name}")
            
            # Set execution mode flags
            if use_sandbox_fusion:
                current.use_sandbox_fusion = True
                
            if validate:
                # Step 3 iterative optimization: If existing tools are insufficient, extend toolset
                called_tools = self._extract_tool_calls(current.solution_code)
                available_tools = {tool.name for tool in ctx.registry.tools.values()}
                missing_tools = called_tools - available_tools
                
                if missing_tools:
                    print(f"   ⏳ Insufficient tools, extending toolset: {missing_tools}")
                    tool_agent.maybe_augment(ctx, current, f"Task requires tools: {missing_tools}")
                
                # Step 3 iterative optimization: If verification fails, repeatedly modify solution or verification function
                print(f"   ⏳ Validating task (will iteratively optimize if failed)...")
                try:
                    current, answer = validator.ensure_valid(ctx, current, fail_soft=fail_soft)
                    if answer is not None:
                        print(f"   ✅ Validation passed!")
                    else:
                        print(f"   ⚠️  Validation failed (soft fail mode)")
                except Exception as e:
                    print(f"   ❌ Validation error: {str(e)[:50]}")
            else:
                print(f"   ⏭️  Skipping validation")
            bundles.append(current)

        # Final: Persist results (format: <environment, tools, task, verifier>)
        print(f"\n💾 [Final] Saving results (format: <environment, tools, task, verifier>)...")
        if persist:
            self._persist(ctx, bundles)
            print(f"   ✓ Saved to: {ctx.sandbox / 'tasks.json'}")
        
        print(f"\n{'='*60}")
        print(f"✨ Synthesis complete! Generated {len(bundles)} task(s)")
        for i, b in enumerate(bundles, 1):
            print(f"   [{b.difficulty}] {b.name}")
        print(f"{'='*60}\n")

        return bundles

