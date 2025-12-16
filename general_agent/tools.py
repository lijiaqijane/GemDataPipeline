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
    """Knowledge base retrieval tool: retrieves information from local knowledge base or predefined data."""
    
    knowledge_base: List[Dict[str, Any]] = field(default_factory=list)
    
    def __call__(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Retrieve relevant information from the knowledge base.
        
        Args:
            query: Search keywords
            max_results: Maximum number of results to return
            
        Returns:
            List of results containing title and url
        """
        if not self.knowledge_base:
            return []
        
        query_lower = query.lower()
        scored_results = []
        
        for item in self.knowledge_base:
            title = str(item.get("title", "")).lower()
            content = str(item.get("content", item.get("summary", ""))).lower()
            
            # Calculate relevance score
            score = 0
            query_words = query_lower.split()
            for word in query_words:
                if word in title:
                    score += 3
                if word in content:
                    score += 1
            
            if score > 0:
                scored_results.append((score, item))
        
        # Sort by score
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
    """Enhanced search tool: supports multiple search sources (Serper API, knowledge base, etc.)."""

    knowledge_base: List[Dict[str, Any]] | None = None
    use_web_search: bool = True
    use_knowledge_base: bool = True
    serper_api_key: str | None = None

    def __post_init__(self) -> None:
        """Initialize knowledge base tool (if provided)."""
        if self.knowledge_base is None:
            self.knowledge_base = []
        self.kb_tool = KnowledgeBaseTool(knowledge_base=self.knowledge_base)
        # Get Serper API key from environment variable or default value
        if self.serper_api_key is None:
            self.serper_api_key = os.getenv("SERPER_API_KEY", "bcfbc1b1cccf74f1ee580d7f5fe53665eb56f92b")

    def __call__(self, query: str, max_results: int = 5, source: str = "auto") -> List[Dict[str, str]]:
        """Search for relevant information, supporting web search and knowledge base retrieval.
        
        Args:
            query: Search query
            max_results: Maximum number of results to return
            source: Search source, "auto" (auto-select), "web" (web only), "kb" (knowledge base only)
            
        Returns:
            List of results containing title and url
        """
        all_results: List[Dict[str, str]] = []
        
        # Determine search source
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
        
        # Web search
        if use_web:
            try:
                web_results = self._web_search(query, max_results)
                all_results.extend(web_results)
            except Exception as e:
                # If web search fails, continue using knowledge base
                pass
        
        # Knowledge base retrieval
        if use_kb:
            try:
                kb_results = self.kb_tool(query, max_results)
                all_results.extend(kb_results)
            except Exception:
                pass
        
        # Deduplicate (based on title)
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
        """Perform web search using Serper API (Google search)."""
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
            
            # Serper API response format: {"organic": [{"title": "...", "link": "...", "snippet": "..."}]}
            organic_results = data.get("organic", [])
            results: List[Dict[str, str]] = []
            
            for item in organic_results[:max_results]:
                results.append({
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "summary": item.get("snippet", ""),  # Add summary information
                })
            
            # If no organic results, try using answerBox or knowledgeGraph
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
            # If Serper API fails, log error but don't raise exception (allow fallback to knowledge base)
            import logging
            logger = logging.getLogger(__name__)
            logger.warning(f"Serper API search failed: {e}")
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
        return [
            {"name": t.name, "description": t.description} for t in self.tools.values()
        ]
