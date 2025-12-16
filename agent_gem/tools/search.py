from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import requests

from agent_gem.core.utils import dump_json
from agent_gem.tools.base import BaseTool, ToolExecutionError


class SearchTool(BaseTool):
    """Search via DuckDuckGo (best-effort; cached under `search_cache.json`)."""

    def __init__(
        self,
        *,
        cache_path: Path,
        timeout_s: int = 10,
        name: str = "search",
        description: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.cache_path = cache_path
        self.timeout_s = timeout_s
        self._cache_lock = threading.Lock()

    def execute(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        query = (query or "").strip()
        if not query:
            return []

        cached = self._load_cache()
        if query in cached:
            return cached[query][:max_results]

        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": 1}
        results: list[dict[str, str]] = []
        try:
            resp = requests.get(url, params=params, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()

            topics = data.get("RelatedTopics", [])[:max_results]
            for item in topics:
                if "Text" in item and "FirstURL" in item:
                    results.append({"title": item.get("Text", ""), "url": item.get("FirstURL", "")})
            if not results and data.get("Heading"):
                results.append({"title": data["Heading"], "url": url})

            cached[query] = results
            self._save_cache(cached)
        except Exception as exc:  # pragma: no cover - network may be restricted
            raise ToolExecutionError("search_failed", results, message=str(exc))

        return results

    def _load_cache(self) -> dict[str, list[dict[str, str]]]:
        with self._cache_lock:
            if not self.cache_path.exists():
                return {}
            try:
                data = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except Exception:
                return {}
            if not isinstance(data, dict):
                return {}
            result: dict[str, list[dict[str, str]]] = {}
            for key, value in data.items():
                if isinstance(key, str) and isinstance(value, list):
                    filtered = [
                        item
                        for item in value
                        if isinstance(item, dict)
                        and isinstance(item.get("title"), str)
                        and isinstance(item.get("url"), str)
                    ]
                    result[key] = filtered
            return result

    def _save_cache(self, cache: dict[str, list[dict[str, str]]]) -> None:
        with self._cache_lock:
            dump_json(self.cache_path, cache)
