from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List

import requests

from .executor import SandboxFusionExecutor


@dataclass
class Tool:
    name: str
    description: str
    handler: Callable[..., Any]

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.handler(*args, **kwargs)


@dataclass
class BashTool:
    """Bash tool that executes commands inside SandboxFusion, not on the host."""

    workdir: Path  # kept for compatibility; not used on host anymore
    timeout: int = 20
    executor: SandboxFusionExecutor | None = None

    def __post_init__(self) -> None:
        if self.executor is None:
            self.executor = SandboxFusionExecutor(
                base_url=os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080"),
                timeout=int(os.getenv("SANDBOX_FUSION_TIMEOUT", str(self.timeout))),
            )

    def __call__(self, command: str) -> Dict[str, Any]:
        if self.executor is None:
            return {
                "returncode": -1,
                "stdout": "",
                "stderr": "SandboxFusion executor is not configured",
            }

        # Delegate bash execution to SandboxFusion service
        result = self.executor(command, language="bash")
        return {
            "returncode": result.get("return_code", 0),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }


@dataclass
class KnowledgeBaseTool:
    """知识库检索工具：从本地知识库或预定义数据中检索信息。"""
    
    knowledge_base: List[Dict[str, Any]] = field(default_factory=list)
    
    def __call__(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """从知识库中检索相关信息。
        
        Args:
            query: 查询关键词
            max_results: 最大返回结果数
            
        Returns:
            包含title和url的结果列表
        """
        if not self.knowledge_base:
            return []
        
        query_lower = query.lower()
        scored_results = []
        
        for item in self.knowledge_base:
            title = str(item.get("title", "")).lower()
            content = str(item.get("content", item.get("summary", ""))).lower()
            
            # 计算相关性分数
            score = 0
            query_words = query_lower.split()
            for word in query_words:
                if word in title:
                    score += 3
                if word in content:
                    score += 1
            
            if score > 0:
                scored_results.append((score, item))
        
        # 按分数排序
        scored_results.sort(key=lambda x: x[0], reverse=True)
        
        results: List[Dict[str, str]] = []
        for _, item in scored_results[:max_results]:
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", f"kb://{item.get('id', '')}"),
                "summary": item.get("summary", item.get("content", "")),
            })
        
        return results


@dataclass
class SearchTool:
    """增强的搜索工具：支持多种搜索源（Serper API、知识库等）。"""

    knowledge_base: List[Dict[str, Any]] | None = None
    use_web_search: bool = True
    use_knowledge_base: bool = True
    serper_api_key: str | None = None

    def __post_init__(self) -> None:
        """初始化知识库工具（如果提供）。"""
        if self.knowledge_base is None:
            self.knowledge_base = []
        self.kb_tool = KnowledgeBaseTool(knowledge_base=self.knowledge_base)
        # 从环境变量或默认值获取Serper API key
        if self.serper_api_key is None:
            self.serper_api_key = os.getenv("SERPER_API_KEY", "bcfbc1b1cccf74f1ee580d7f5fe53665eb56f92b")

    def __call__(self, query: str, max_results: int = 5, source: str = "auto") -> List[Dict[str, str]]:
        """搜索相关信息，支持网络搜索和知识库检索。
        
        Args:
            query: 搜索查询
            max_results: 最大返回结果数
            source: 搜索源，"auto"（自动选择）、"web"（仅网络）、"kb"（仅知识库）
            
        Returns:
            包含title和url的结果列表
        """
        all_results: List[Dict[str, str]] = []
        
        # 确定搜索源
        if source == "auto":
            use_web = self.use_web_search
            use_kb = self.use_knowledge_base and len(self.knowledge_base) > 0
        elif source == "web":
            use_web = True
            use_kb = False
        elif source == "kb":
            use_web = False
            use_kb = True
        else:
            use_web = self.use_web_search
            use_kb = self.use_knowledge_base and len(self.knowledge_base) > 0
        
        # 网络搜索
        if use_web:
            try:
                web_results = self._web_search(query, max_results)
                all_results.extend(web_results)
            except Exception as e:
                # 如果网络搜索失败，继续使用知识库
                pass
        
        # 知识库检索
        if use_kb:
            try:
                kb_results = self.kb_tool(query, max_results)
                all_results.extend(kb_results)
            except Exception:
                pass
        
        # 去重（基于title）
        seen_titles = set()
        unique_results = []
        for result in all_results:
            title = result.get("title", "")
            if title and title not in seen_titles:
                seen_titles.add(title)
                unique_results.append(result)
                if len(unique_results) >= max_results:
                    break
        
        return unique_results[:max_results]
    
    def _web_search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """使用Serper API进行网络搜索（Google搜索）。"""
        if not self.serper_api_key:
            raise ValueError("Serper API key is not configured. Set SERPER_API_KEY environment variable.")
        
        url = "https://google.serper.dev/search"
        headers = {
            "X-API-KEY": self.serper_api_key,
            "Content-Type": "application/json"
        }
        payload = {
            "q": query,
            "num": max_results
        }
        
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            # Serper API返回格式: {"organic": [{"title": "...", "link": "...", "snippet": "..."}]}
            organic_results = data.get("organic", [])
            results: List[Dict[str, str]] = []
            
            for item in organic_results[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "summary": item.get("snippet", ""),  # 添加摘要信息
                })
            
            # 如果没有organic结果，尝试使用answerBox或knowledgeGraph
            if not results:
                if "answerBox" in data:
                    answer = data["answerBox"]
                    results.append({
                        "title": answer.get("title", query),
                        "url": answer.get("link", ""),
                        "summary": answer.get("answer", answer.get("snippet", "")),
                    })
                elif "knowledgeGraph" in data:
                    kg = data["knowledgeGraph"]
                    results.append({
                        "title": kg.get("title", query),
                        "url": kg.get("websiteUrl", ""),
                        "summary": kg.get("description", ""),
                    })
            
            return results
            
        except requests.exceptions.RequestException as e:
            # 如果Serper API失败，记录错误但不抛出异常（允许fallback到知识库）
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Serper API搜索失败: {e}")
            return []


@dataclass
class ToolRegistry:
    """Registry that manages tools exposed to synthesis and verification."""

    tools: Dict[str, Tool] = field(default_factory=dict)

    def register(self, name: str, description: str, func: Callable[..., Any]) -> None:
        self.tools[name] = Tool(name=name, description=description, handler=func)

    def ensure_defaults(self, bash: BashTool, search: SearchTool) -> None:
        """Register default tools. Note: SandboxFusion is an execution environment, not a tool."""
        if "bash" not in self.tools:
            self.register("bash", "Execute bash commands inside the sandbox", bash)
        if "search" not in self.tools:
            self.register("search", "Search the web via Serper API (Google) and knowledge base for information", search)

    def as_callable_dict(self) -> Dict[str, Callable[..., Any]]:
        return {name: tool.handler for name, tool in self.tools.items()}

    def describe(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self.tools.values()]

