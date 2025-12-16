from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List
import json
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
class SearchTool:
    """Simple DuckDuckGo search wrapper for sandbox lookups."""

    def __call__(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1}
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        topics = data.get("RelatedTopics", [])[:max_results]
        results: List[Dict[str, str]] = []
        for item in topics:
            if "Text" in item and "FirstURL" in item:
                results.append(
                    {"title": item.get("Text", ""), "url": item.get("FirstURL", "")}
                )
        if not results and data.get("Heading"):
            results.append({"title": data["Heading"], "url": url})
        return results


@dataclass
class SerperSearchTool:
    """Serper API search wrapper for high-quality web search results."""

    api_key: str
    base_url: str = "https://google.serper.dev"

    def __call__(
        self,
        query: str,
        max_results: int = 10,
        search_type: str = "search",
    ) -> Dict[str, Any]:
        """
        Search using Serper API.

        Args:
            query: Search query string
            max_results: Maximum number of results to return
            search_type: Type of search - "search" (default) or "images" or "videos"

        Returns:
            Dictionary containing organic results, answerBox, knowledgeGraph, etc.
        """
        url = f"{self.base_url}/{search_type}"
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }
        payload = json.dumps(
            {
                "q": query,
                "num": max_results,
            }
        )

        response = requests.request("POST", url, headers=headers, data=payload)
        return json.loads(response.text)

    def get_organic_results(
        self, query: str, max_results: int = 10
    ) -> List[Dict[str, Any]]:
        """Get only organic search results."""
        data = self(query, max_results=max_results)
        return data.get("organic", [])[:max_results]

    def get_answer_box(self, query: str) -> Dict[str, Any]:
        """Get answer box if available."""
        data = self(query, max_results=1)
        return data.get("answerBox", {})

    def get_knowledge_graph(self, query: str) -> Dict[str, Any]:
        """Get knowledge graph if available."""
        data = self(query, max_results=1)
        return data.get("knowledgeGraph", {})


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
            self.register("search", "Search the web via DuckDuckGo", search)

    def as_callable_dict(self) -> Dict[str, Callable[..., Any]]:
        return {name: tool.handler for name, tool in self.tools.items()}

    def describe(self) -> List[Dict[str, str]]:
        return [
            {"name": t.name, "description": t.description} for t in self.tools.values()
        ]
