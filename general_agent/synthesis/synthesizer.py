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
        """步骤1：环境与工具集构建 - 在包含bash和搜索工具的沙箱环境中，生成或检索相关数据并存储到数据库。
        
        根据论文设计，此步骤负责：
        1. 在沙箱环境中使用bash和搜索工具
        2. 针对特定任务类别，生成或检索相关数据
        3. 将数据存储到数据库中
        
        约束：
        - 必须使用搜索工具从网络或知识库中检索数据
        - 可以使用bash工具生成模拟数据
        - 所有数据必须存储到数据库中
        """
        logger.info(f"步骤1：开始生成/检索数据（类别: {ctx.category}）...")
        
        # 步骤1.1：使用搜索工具检索相关数据
        search_query = f"{ctx.category} sample data list structured information"
        logger.debug(f"使用搜索工具检索: {search_query}")
        try:
            search_hits = ctx.registry.tools["search"](search_query, max_results=5)
            logger.info(f"搜索工具返回 {len(search_hits)} 条结果")
        except Exception as exc:  # pragma: no cover - network/API fallback
            logger.warning("搜索工具失败，回退到空结果: %s", exc)
            search_hits = []
        
        # 步骤1.2：可选使用bash工具生成模拟数据（如果需要）
        # 例如：生成测试数据文件、处理数据等
        bash_commands = []
        if not search_hits:
            # 如果搜索没有结果，可以使用bash生成一些基础数据
            logger.debug("搜索无结果，考虑使用bash工具生成模拟数据")
            # 这里可以添加bash命令来生成数据文件等
        
        # 步骤1.3：使用LLM基于搜索结果和任务类别生成结构化数据
        # 注意：即使搜索结果为空，LLM仍会根据任务类别生成数据（这是fallback机制）
        data_source = "搜索结果 + 任务类别" if search_hits else "任务类别（搜索无结果，使用LLM生成）"
        logger.info(f"步骤1.3：使用LLM生成结构化数据（数据来源: {data_source}）...")
        
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
            logger.warning("LLM返回的JSON解析失败，使用fallback记录")
            records = [{"title": ctx.category, "summary": generated[:200]}]
        if isinstance(records, dict):
            records = [records]
        
        logger.info(f"步骤1.3完成：LLM生成了 {len(records)} 条结构化记录")
        
        # 步骤1.4：将生成的数据存储到数据库
        initial_count = len(ctx.db.records)
        for row in records:
            # 确保记录有基本字段
            if "title" not in row:
                row["title"] = str(row.get("name", ctx.category))
            if "summary" not in row:
                row["summary"] = str(row.get("description", ""))
            ctx.db.add_record(row)
        
        final_count = len(ctx.db.records)
        added_count = final_count - initial_count
        
        logger.info(
            f"步骤1完成：已生成并存储 {added_count} 条新数据库记录（总计 {final_count} 条）到 {ctx.db.path}\n"
            f"  数据来源说明：搜索工具返回 {len(search_hits)} 条结果，LLM基于{'搜索结果和' if search_hits else ''}任务类别生成了 {len(records)} 条记录"
        )

    def _generate_fallback_tools(self, ctx: SynthesisContext) -> List[Dict[str, str]]:
        """生成fallback工具，基于数据库记录和任务类别。
        
        当LLM无法生成工具时，使用这个方法来生成基础工具。
        """
        category_lower = ctx.category.lower()
        tools = []
        
        # 分析数据库记录，生成基础查询工具
        records = ctx.db.records[:10]
        record_types = set()
        record_keys = set()
        
        for record in records:
            if "type" in record:
                record_types.add(record["type"].lower())
            record_keys.update(record.keys())
        
        # 根据任务类别和数据库结构生成工具
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
        
        # 通用工具
        if len(records) > 0:
            tools.append({
                "name": "get_all_records",
                "description": "Get all records from the database. Parameters: None"
            })
            tools.append({
                "name": "search_records",
                "description": "Search records by keyword. Parameters: keyword (str) - search keyword to filter records"
            })
        
        # 根据记录的type字段生成特定工具
        if record_types:
            for record_type in list(record_types)[:2]:  # 最多生成2个类型特定工具
                tool_name = f"get_{record_type.lower().replace(' ', '_')}_records"
                tools.append({
                    "name": tool_name,
                    "description": f"Get all {record_type} records from the database. Parameters: None"
                })
        
        # 确保至少有2个工具
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
        
        logger.info(f"生成了 {len(tools)} 个fallback工具: {[t['name'] for t in tools]}")
        return tools[:5]  # 最多返回5个工具

    def synthesize_tools(self, ctx: SynthesisContext, additional_context: str = "") -> None:
        """步骤2：任务合成 - 基于数据库，合成一组任务相关工具，每个工具实现为一个函数。
        
        根据论文设计，此步骤负责：
        1. 基于数据库分析任务需求
        2. 合成一组任务相关工具，每个工具实现为一个函数
        3. 工具函数可以访问数据库（与解答函数的约束不同）
        4. 工具函数可以调用其他工具函数
        5. 工具函数必须返回可验证的结果
        """
        context_suffix = f"\nAdditional context: {additional_context}" if additional_context else ""
        
        # 增强的prompt，要求生成至少3-5个工具
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
        
        # 重试机制：最多尝试3次
        max_retries = 3
        tools = []
        for attempt in range(max_retries):
            try:
                raw = ctx.llm.simple_complete(
                    prompt, 
                    temperature=0.6 + attempt * 0.1,  # 逐步提高温度增加多样性
                    max_tokens=800  # 增加token数以确保生成足够工具
                )
                parsed = self._parse_json_response(raw)
                logger.debug(f"LLM返回的工具定义（尝试 {attempt + 1}/{max_retries}）: {raw[:500]}")
                
                if isinstance(parsed, dict):
                    parsed = [parsed]
                elif not isinstance(parsed, list):
                    parsed = []
                
                # 过滤掉无效的工具
                valid_tools = []
                for tool in parsed:
                    if isinstance(tool, dict) and tool.get("name") and tool.get("description"):
                        # 排除默认工具（bash, search）
                        if tool.get("name") not in ["bash", "search"]:
                            valid_tools.append(tool)
                
                if len(valid_tools) >= 2:  # 至少需要2个有效工具
                    tools = valid_tools
                    logger.info(f"步骤2：成功生成 {len(tools)} 个工具定义（尝试 {attempt + 1}/{max_retries}）")
                    break
                else:
                    logger.warning(f"步骤2：生成的工具数量不足（{len(valid_tools)} 个，尝试 {attempt + 1}/{max_retries}），继续重试...")
                    
            except json.JSONDecodeError as e:
                logger.warning(f"工具定义JSON解析失败（尝试 {attempt + 1}/{max_retries}）: {e}, 原始响应: {raw[:200]}")
                if attempt == max_retries - 1:
                    tools = []
            except Exception as e:
                logger.error(f"工具生成过程中出错（尝试 {attempt + 1}/{max_retries}）: {e}")
                if attempt == max_retries - 1:
                    tools = []
        
        # 如果所有重试都失败，生成fallback工具
        if len(tools) < 2:
            logger.warning(f"步骤2：工具生成失败或数量不足（{len(tools)} 个），使用fallback工具")
            tools = self._generate_fallback_tools(ctx)
        
        logger.info(f"步骤2：最终将注册 {len(tools)} 个自定义工具")

        # 创建工具上下文，允许工具访问数据库和调用其他工具
        tool_ctx = ToolContext(db=ctx.db, tools={})

        for spec in tools:
            name = spec.get("name")
            desc = spec.get("description", "")
            if not name:
                continue

            def make_handler(key: str, tool_context: ToolContext):
                def base_handler(*args: Any, **kwargs: Any) -> Any:
                    """工具函数处理器：可以访问数据库和调用其他工具函数。
                    
                    根据论文约束，工具函数可以：
                    - 访问数据库（通过 tool_context.db）
                    - 调用其他工具函数（通过 tool_context.tools）
                    - 必须返回可验证的结果
                    
                    支持多个参数，例如：
                    - get_all_hotels_by_city(city: str) - 一个参数
                    - get_infos_by_hotel(info_keywords: List[str], hotel: str) - 两个参数
                    - get_inter_city_transport(from_city: str, to_city: str) - 两个参数
                    """
                    logger.debug("Tool '%s' called with args=%s, kwargs=%s", key, args, kwargs)
                    
                    # 处理多个参数的情况
                    # 如果工具名称包含"by_"或"infos_by"，通常需要多个参数
                    if "infos_by" in key.lower() or "by_" in key.lower():
                        # 对于类似get_infos_by_hotel(info_keywords, hotel)的工具
                        # args[0] 是 info_keywords (list)
                        # args[1] 是 hotel (str)
                        if len(args) >= 2:
                            info_keywords = args[0] if isinstance(args[0], list) else [args[0]]
                            entity_name = args[1] if len(args) > 1 else (args[0] if len(args) == 1 else None)
                            
                            # 从数据库中找到对应的实体（工具函数可以访问数据库）
                            entity_type = key.split("_by_")[-1] if "_by_" in key else key.split("by_")[-1]
                            entity_records = [r for r in tool_context.get_all_records() 
                                            if entity_type in r.get("type", "").lower() 
                                            and (entity_name in str(r.get("name", "")) or entity_name in str(r.get("title", "")))]
                            
                            if entity_records:
                                entity = entity_records[0]
                                # 返回请求的字段（可验证的结果）
                                result = {k: entity.get(k) for k in info_keywords if k in entity}
                                return result
                            return {}
                    
                    # 处理单个参数的情况（向后兼容）
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
                        # 工具函数可以访问数据库
                        result = tool_context.get_all_records()
                    elif not isinstance(candidate, str):
                        candidate = str(candidate)
                        result = self.smart_db_query(tool_context.get_all_records(), key, candidate)
                    else:
                        result = self.smart_db_query(tool_context.get_all_records(), key, candidate)
                    
                    # 根据论文约束，工具函数必须返回可验证的结果
                    # 返回格式应该是结构化的（list, dict），而不是字符串或None
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
                        # 工具函数必须返回可验证的结果，不能返回None
                        result = []
                    
                    # 确保返回的结果是可验证的（list或dict）
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
            logger.info(f"步骤2：注册工具 '{name}': {desc[:100]}")
        
        # 更新工具上下文，包含所有已注册的工具（允许工具调用其他工具）
        tool_ctx.tools = ctx.registry.as_callable_dict()
        
        if len(tools) == 0:
            logger.warning("步骤2：未生成任何自定义工具，只有默认工具（bash, search）可用")
        else:
            logger.info(f"步骤2：成功注册了 {len([t for t in ctx.registry.tools.values() if t.name not in ['bash', 'search']])} 个自定义工具")

    def augment_toolset(self, ctx: SynthesisContext, bundle: TaskBundle, failure_reason: str, answer: Any = None) -> bool:
        """步骤3工具扩展：在任务难度提升过程中，若现有工具不足以求解，扩展工具集。
        
        根据论文设计，在任务难度提升过程中，如果现有工具不足以求解，代理会扩展工具集。
        此方法实现该工具扩展逻辑，根据失败分析结果精准扩展工具。
        
        Args:
            ctx: 合成上下文
            bundle: 任务包
            failure_reason: 失败原因
            answer: 解答函数的输出（用于分析）
        
        Returns True if new tools were added.
        """
        # 详细分析失败原因
        failure_analysis = self._analyze_failure(failure_reason, bundle, answer)
        
        # 分析是否需要工具扩展
        solution_code = bundle.solution_code or ""
        called_tools = self._extract_tool_calls(solution_code)
        available_tools = {tool.name for tool in ctx.registry.tools.values()}
        missing_tools = called_tools - available_tools
        
        # 根据失败类型判断是否需要工具扩展
        needs_augmentation = False
        augmentation_reason = ""
        
        if missing_tools:
            needs_augmentation = True
            augmentation_reason = f"任务代码尝试使用不存在的工具: {missing_tools}"
        elif failure_analysis["failure_type"] == "tool_not_found":
            needs_augmentation = True
            augmentation_reason = f"工具调用失败: {failure_analysis['root_cause']}"
        elif failure_analysis["failure_type"] == "verification_failed" and answer is not None:
            # 分析答案结构，看是否需要特定工具来生成缺失的字段
            if isinstance(answer, dict):
                missing_fields = self._analyze_missing_fields(bundle, answer, ctx)
                if missing_fields:
                    needs_augmentation = True
                    augmentation_reason = f"答案缺少关键字段，可能需要新工具来生成: {missing_fields}"
            elif isinstance(answer, list) and len(answer) == 0:
                # 空列表可能表示需要数据获取工具
                needs_augmentation = True
                augmentation_reason = "答案为空，可能需要新的数据检索工具"
        
        if not needs_augmentation:
            logger.debug("不需要工具扩展：失败原因与工具无关")
            return False
        
        logger.info("Augmenting toolset: %s", augmentation_reason)
        
        # 根据失败分析生成精准的工具扩展建议
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
        
        # 创建工具上下文，允许工具访问数据库和调用其他工具
        tool_ctx = ToolContext(db=ctx.db, tools=ctx.registry.as_callable_dict())
        
        added_count = 0
        for spec in new_tools:
            name = spec.get("name")
            desc = spec.get("description", "")
            if not name or name in available_tools:
                continue  # Skip if already exists
            
            def make_handler(key: str, tool_context: ToolContext):
                def base_handler(*args: Any, **kwargs: Any) -> Any:
                    """工具函数处理器：可以访问数据库和调用其他工具函数。"""
                    logger.debug("Augmented tool '%s' called with args=%s, kwargs=%s", key, args, kwargs)
                    
                    # 处理多个参数的情况（与synthesize_tools中的逻辑一致）
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
                    
                    # 处理单个参数的情况（向后兼容）
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
        
        # 更新工具上下文，包含所有已注册的工具（允许工具调用其他工具）
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
        """验证LLM生成的代码是否符合论文约束，使用AST分析而不是正则表达式。
        
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
        """步骤3：解答生成 - 构造简单任务、对应的解答函数和验证函数。
        
        根据论文设计，此步骤的约束条件：
        1. 解答函数只能调用工具函数或执行逻辑计算
        2. 不能访问数据库或调用其他函数
        3. 解答输出必须通过验证函数检查
        4. 如果验证失败，会反复修改解答或验证函数
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
        # 带重试的LLM调用，验证响应以防止幻觉
        max_retries = 3
        for attempt in range(max_retries):
            raw = ctx.llm.simple_complete(prompt, temperature=0.6, max_tokens=800)
            try:
                parsed = self._parse_json_response(raw)
            except json.JSONDecodeError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"LLM response parsing failed (attempt {attempt + 1}/{max_retries}): {e}")
                    continue
                # 最后一次尝试失败，使用fallback
                parsed = {
                    "name": f"{ctx.category}-task",
                    "description": raw[:200],
                    "solution_code": "def solve(tools):\n    return list(tools.keys())",
                    "verification_code": "def verify(tools, answer):\n    return isinstance(answer, list)",
                }
            
            # 验证生成的代码
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
                    # 在prompt中添加错误信息，要求重新生成
                    prompt += f"\n\nPrevious attempt failed validation:\n- Solution: {sol_error}\n- Verification: {ver_error}\nPlease fix these issues."
                    continue
        
        # 所有重试都失败，返回fallback（会被后续的_ensure_substantive_task修复）
        logger.error("All LLM retries failed, using fallback task")
        return TaskBundle(
            name=parsed.get("name", "generated-task"),
            description=parsed.get("description", ""),
            difficulty=difficulty,
            solution_code=solution_code,
            verification_code=verification_code,
        )

    def refine_task(self, ctx: SynthesisContext, prev: TaskBundle) -> TaskBundle:
        """步骤3任务难度提升：逐步提升任务难度，同时保持可验证性。
        
        根据论文设计，在任务难度提升过程中：
        1. 从简单任务开始验证通过后，增加任务复杂度
        2. 增加约束条件、任务规模、数据量
        3. 如果现有工具不足以求解，会扩展工具集（在ensure_valid中处理）
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
        
        # 创建详细的任务描述（包含6个核心部分）
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
        """详细分析验证失败的原因。
        
        Returns:
            包含失败类型、原因、建议修复方案的字典
        """
        failure_reason_lower = failure_reason.lower()
        analysis = {
            "failure_type": "unknown",
            "root_cause": failure_reason[:500],
            "detailed_analysis": "",
            "suggested_fix": "",
            "affected_component": "unknown",  # solution, verification, or both
        }
        
        # 分类失败类型
        if "verification returned false" in failure_reason_lower:
            analysis["failure_type"] = "verification_failed"
            analysis["affected_component"] = "verification"
            analysis["detailed_analysis"] = (
                "验证函数返回了False。可能的原因：\n"
                "1. 答案的数据结构不符合验证函数的期望\n"
                "2. 验证逻辑过于严格或错误\n"
                "3. 答案缺少必需的字段或类型不正确"
            )
            if answer is not None:
                analysis["detailed_analysis"] += f"\n\n实际答案: {str(answer)[:200]}"
            analysis["suggested_fix"] = (
                "检查验证函数是否正确地验证答案结构。"
                "验证应该检查关键字段的存在和类型，而不是精确值。"
                "如果答案结构合理，考虑放宽验证条件。"
            )
        
        elif any(keyword in failure_reason_lower for keyword in ["not found", "missing", "keyerror", "attributeerror"]):
            analysis["failure_type"] = "tool_not_found"
            analysis["affected_component"] = "solution"
            # 提取工具名称（如果可能）
            import re
            tool_match = re.search(r"['\"]([^'\"]+)['\"]", failure_reason)
            tool_name = tool_match.group(1) if tool_match else "unknown"
            analysis["detailed_analysis"] = (
                f"工具调用失败。工具 '{tool_name}' 不存在或无法访问。\n"
                "可能的原因：\n"
                "1. 工具名称拼写错误\n"
                "2. 工具尚未注册\n"
                "3. 工具参数不匹配"
            )
            analysis["suggested_fix"] = (
                f"检查工具名称是否正确。"
                f"确保使用正确的工具名称和参数格式。"
            )
        
        elif any(keyword in failure_reason_lower for keyword in ["cannot access", "database", "ctx.db", "db."]):
            analysis["failure_type"] = "database_access_violation"
            analysis["affected_component"] = "solution"
            analysis["detailed_analysis"] = (
                "解答函数尝试直接访问数据库，这违反了约束条件。\n"
                "根据论文设计，解答函数只能通过工具函数访问数据。"
            )
            analysis["suggested_fix"] = (
                "移除所有直接数据库访问代码。"
                "使用工具函数（如 tools['get_all_hotels_by_city']('Paris')）来获取数据。"
            )
        
        elif any(keyword in failure_reason_lower for keyword in ["syntax error", "invalid syntax", "indentation"]):
            analysis["failure_type"] = "syntax_error"
            analysis["affected_component"] = "both"
            analysis["detailed_analysis"] = "代码存在语法错误。"
            analysis["suggested_fix"] = "修复语法错误，确保代码符合Python语法规范。"
        
        elif any(keyword in failure_reason_lower for keyword in ["typeerror", "type error", "cannot convert"]):
            analysis["failure_type"] = "type_error"
            analysis["affected_component"] = "solution"
            analysis["detailed_analysis"] = (
                "类型错误：工具返回的数据类型与预期不符。\n"
                "工具通常返回list或dict，需要正确处理这些类型。"
            )
            analysis["suggested_fix"] = (
                "检查工具返回的数据类型。"
                "使用 isinstance() 检查类型，并适当处理list和dict。"
            )
        
        elif "timeout" in failure_reason_lower or "execution time" in failure_reason_lower:
            analysis["failure_type"] = "timeout"
            analysis["affected_component"] = "solution"
            analysis["detailed_analysis"] = "代码执行超时。可能是逻辑过于复杂或存在无限循环。"
            analysis["suggested_fix"] = "简化逻辑，减少循环次数，或优化算法。"
        
        else:
            # 默认分析
            analysis["failure_type"] = "runtime_error"
            analysis["affected_component"] = "both"
            analysis["detailed_analysis"] = f"运行时错误：{failure_reason[:300]}"
            analysis["suggested_fix"] = "检查代码逻辑，确保所有变量都已定义，所有函数调用都正确。"
        
        return analysis

    def _analyze_missing_fields(self, bundle: TaskBundle, answer: Dict[str, Any], ctx: SynthesisContext) -> List[str]:
        """分析答案中可能缺少的字段。"""
        # 从任务描述中提取期望的字段
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
        """根据失败分析生成工具扩展建议。"""
        guidance_parts = []
        
        if missing_tools:
            guidance_parts.append(
                f"任务代码尝试调用以下不存在的工具: {list(missing_tools)}。"
                "请生成这些工具或功能等价的其他工具。"
            )
        
        if failure_analysis["failure_type"] == "verification_failed":
            guidance_parts.append(
                "验证失败可能因为答案缺少关键字段或结构不正确。"
                "请生成能够生成完整答案结构的工具，例如："
                "- 数据聚合工具（aggregate_data, combine_results）"
                "- 字段提取工具（extract_field, get_details）"
                "- 格式化工具（format_output, structure_data）"
            )
        
        if failure_analysis["failure_type"] == "tool_not_found":
            guidance_parts.append(
                "工具调用失败。请检查是否需要："
                "- 数据检索工具（get_by_keyword, search_records）"
                "- 过滤工具（filter_by_criteria, match_condition）"
                "- 计算工具（calculate_statistics, compute_value）"
            )
        
        if answer is not None:
            if isinstance(answer, list) and len(answer) == 0:
                guidance_parts.append(
                    "答案为空列表，可能需要数据获取或查询工具来检索数据。"
                )
            elif isinstance(answer, dict) and len(answer) == 0:
                guidance_parts.append(
                    "答案为空字典，可能需要数据提取或生成工具来填充数据。"
                )
        
        return "\n".join(f"- {part}" for part in guidance_parts) if guidance_parts else "根据失败分析和任务需求生成合适的工具。"

    @staticmethod
    def _indent_text(text: str, spaces: int = 4) -> str:
        """为文本添加缩进。"""
        indent = " " * spaces
        return "\n".join(indent + line for line in text.split("\n"))

    def repair_bundle(self, ctx: SynthesisContext, bundle: TaskBundle, failure_reason: str, answer: Any = None) -> TaskBundle:
        """步骤3迭代优化：当验证失败时，反复修改解答或验证函数。
        
        根据论文设计，如果验证失败，代理会反复修改解答或验证函数。
        此方法实现该迭代优化逻辑。
        """
        # 详细分析失败原因
        failure_analysis = self._analyze_failure(failure_reason, bundle, answer)
        
        # 构建详细的修复提示
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
        """步骤3迭代优化：执行并验证任务，实现论文中的迭代优化逻辑。
        
        根据论文设计，此方法实现：
        1. 执行解答函数并验证结果
        2. 如果验证失败，反复修改解答或验证函数（通过repair_bundle）
        3. 如果现有工具不足以求解，扩展工具集（通过augment_toolset）
        4. 在任务难度提升过程中，若现有工具不足以求解，会扩展工具集
        
        If fail_soft, return last attempt instead of raising.
        
        注意：由于所有代码执行都在SandboxFusion中进行，这里只需要传递工具名称字典即可。
        _run_in_sandbox_fusion会使用工具名称创建SandboxFusion内部的ToolProxy。
        """
        # Ensure the task is not trivial before running executions
        bundle = self._ensure_substantive_task(ctx, bundle, "Initial validation quality gate")

        # 由于代码在SandboxFusion中执行，只需要传递工具名称字典
        # _run_in_sandbox_fusion会使用这些名称创建SandboxFusion内部的ToolProxy
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
                # 传递数据库记录给工具执行
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
        """检查解答函数是否符合论文约束：使用AST分析而不是正则表达式。
        
        根据论文设计，解答函数必须：
        1. 只能调用工具函数或执行逻辑计算
        2. 不能访问数据库
        3. 不能定义其他函数（只能有solve函数）
        4. 不能导入模块
        5. 必须实际调用工具
        """
        sol = bundle.solution_code or ""
        ver = bundle.verification_code or ""
        
        # 使用AST分析验证解答函数
        is_valid, error = CodeValidator.validate_solution_code(sol)
        if not is_valid:
            logger.debug(f"Solution code validation failed: {error}")
            return True  # 不符合约束，视为trivial
        
        # Check for trivial solution patterns (简单模式检查)
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
        """持久化合成结果，输出格式严格符合论文的 <environment, tools, task, verifier> 四元组。
        
        根据论文设计，生成的数据格式为四元组：
        - environment: 环境描述（包含category和数据库records）
        - tools:       工具集（每个工具包含name和description）
        - task:        任务定义（包含name、description、difficulty、solution_code）
        - verifier:    验证器（包含verification_code）
        
        输出JSON结构：
        {
            "environment": {  # 环境描述
                "category": "...",
                "records": [...]  # 数据库记录
            },
            "tools": [  # 工具集
                {"name": "...", "description": "..."},
                ...
            ],
            "tasks": [  # 任务列表（每个任务包含task和verifier）
                {
                    "name": "...",
                    "description": "...",
                    "difficulty": 1,
                    "solution_code": "...",  # task部分
                    "verification_code": "..."  # verifier部分
                },
                ...
            ]
        }
        
        为了兼容旧代码，仍然保留原有的 category / tooling / records / tasks 字段。
        """
        # 构建标准的四元组格式：<environment, tools, task, verifier>
        # 每个任务包含task（solution_code）和verifier（verification_code）
        tasks_with_verifiers = []
        for bundle in bundles:
            task_entry = {
                # Task部分：任务定义
                "task": {
                    "name": bundle.name,
                    "description": bundle.description,
                    "difficulty": bundle.difficulty,
                    "solution_code": bundle.solution_code,
                },
                # Verifier部分：验证器定义
                "verifier": {
                    "verification_code": bundle.verification_code,
                },
                # 保留完整信息（向后兼容）
                "name": bundle.name,
                "description": bundle.description,
                "difficulty": bundle.difficulty,
                "solution_code": bundle.solution_code,
                "verification_code": bundle.verification_code,
            }
            tasks_with_verifiers.append(task_entry)
        
        payload = {
            # 标准四元组格式
            "environment": {
                "category": ctx.category,
                "records": ctx.db.records,
                "record_count": len(ctx.db.records),
            },
            "tools": ctx.registry.describe(),
            "tasks": tasks_with_verifiers,
            
            # 兼容字段（向后兼容）
            "category": ctx.category,
            "tooling": ctx.registry.describe(),  # 与tools相同
            "records": ctx.db.records,  # 与environment.records相同
            
            # 元数据
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
        """主入口：按照论文设计的三步流程进行环境+任务合成。
        
        根据论文的"构造生成任务的具体步骤"，此方法严格遵循三个步骤：
        
        步骤1：环境与工具集构建
        - 在包含bash和搜索工具的沙箱环境中进行
        - 针对特定任务类别，生成或检索相关数据并存储到数据库
        
        步骤2：任务合成
        - 基于数据库，合成一组任务相关工具，每个工具实现为一个函数
        
        步骤3：解答生成（迭代优化）
        - 构造简单任务、对应的解答函数和验证函数
        - 解答函数只能调用工具函数或执行逻辑计算，不能访问数据库
        - 如果验证失败，反复修改解答或验证函数
        - 在任务难度提升过程中，若现有工具不足以求解，扩展工具集
        
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
        print(f"🚀 开始任务合成: {category}")
        print(f"{'='*60}")
        
        # Step 1: Build context
        # 初始化各个子 Agent（环境 / 工具 / 任务 / 校验），形成一个通用的多阶段 Agent 流水线
        env_agent = EnvironmentAgent(self)
        tool_agent = ToolAgent(self)
        task_agent = TaskAgent(self)
        validator = ValidationAgent(self)

        print(f"\n📁 [步骤1] 环境与工具集构建...")
        # 步骤1：在包含bash和搜索工具的沙箱环境中，生成或检索相关数据并存储到数据库
        ctx = self.build_context(category, sandbox, use_sandbox_fusion=use_sandbox_fusion)
        exec_mode = "SandboxFusion" if use_sandbox_fusion else "本地"
        print(f"   ✓ 沙箱目录: {sandbox}")
        print(f"   ✓ 执行模式: {exec_mode}")

        # 步骤1（续）：生成数据库记录
        print(f"\n📊 [步骤1续] 生成数据库记录...")
        ctx = env_agent.prepare(category, sandbox, use_sandbox_fusion)
        print(f"   ✓ 数据库记录数: {len(ctx.db.records)}")
        
        # 步骤2：任务合成 - 基于数据库，合成一组任务相关工具
        print(f"\n🔧 [步骤2] 任务合成 - 合成工具集...")
        tool_agent.build_initial_tools(ctx)
        tool_names = [t.name for t in ctx.registry.tools.values()]
        print(f"   ✓ 生成工具: {', '.join(tool_names)}")

        bundles: List[TaskBundle] = []
        
        # 步骤3：解答生成 - 构造简单任务、对应的解答函数和验证函数
        print(f"\n📝 [步骤3] 解答生成 (共 {rounds} 轮，迭代优化)...")
        print(f"\n   --- 第 1 轮 (难度 1) ---")
        print(f"   ⏳ 生成初始任务...")
        current = task_agent.propose_initial(ctx, difficulty=1)
        print(f"   ✓ 任务名称: {current.name}")
        
        # Set execution mode flags
        if use_sandbox_fusion:
            current.use_sandbox_fusion = True
            
        if validate:
            print(f"   ⏳ 验证任务...")
            try:
                current, answer = validator.ensure_valid(ctx, current, fail_soft=fail_soft)
                if answer is not None:
                    print(f"   ✅ 验证通过!")
                else:
                    print(f"   ⚠️  验证失败 (软失败模式)")
            except Exception as e:
                print(f"   ❌ 验证错误: {str(e)[:50]}")
        else:
            print(f"   ⏭️  跳过验证")
        bundles.append(current)

        # 步骤3（续）：任务难度提升 - 逐步提升任务复杂度
        for step in range(1, rounds):
            print(f"\n   --- 第 {step + 1} 轮 (难度 {step + 1}) ---")
            print(f"   ⏳ 生成进阶任务（提升难度）...")
            current = task_agent.refine(ctx, current, round_index=step)
            print(f"   ✓ 任务名称: {current.name}")
            
            # Set execution mode flags
            if use_sandbox_fusion:
                current.use_sandbox_fusion = True
                
            if validate:
                # 步骤3迭代优化：如果现有工具不足以求解，扩展工具集
                called_tools = self._extract_tool_calls(current.solution_code)
                available_tools = {tool.name for tool in ctx.registry.tools.values()}
                missing_tools = called_tools - available_tools
                
                if missing_tools:
                    print(f"   ⏳ 工具不足，扩展工具集: {missing_tools}")
                    tool_agent.maybe_augment(ctx, current, f"Task requires tools: {missing_tools}")
                
                # 步骤3迭代优化：如果验证失败，反复修改解答或验证函数
                print(f"   ⏳ 验证任务（如失败将迭代优化）...")
                try:
                    current, answer = validator.ensure_valid(ctx, current, fail_soft=fail_soft)
                    if answer is not None:
                        print(f"   ✅ 验证通过!")
                    else:
                        print(f"   ⚠️  验证失败 (软失败模式)")
                except Exception as e:
                    print(f"   ❌ 验证错误: {str(e)[:50]}")
            else:
                print(f"   ⏭️  跳过验证")
            bundles.append(current)

        # 最终：持久化结果（格式：<environment, tools, task, verifier>）
        print(f"\n💾 [最终] 保存结果（格式：<environment, tools, task, verifier>）...")
        if persist:
            self._persist(ctx, bundles)
            print(f"   ✓ 保存到: {ctx.sandbox / 'tasks.json'}")
        
        print(f"\n{'='*60}")
        print(f"✨ 合成完成! 共生成 {len(bundles)} 个任务")
        for i, b in enumerate(bundles, 1):
            print(f"   [{b.difficulty}] {b.name}")
        print(f"{'='*60}\n")

        return bundles

