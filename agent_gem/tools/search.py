from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any

import requests

from agent_gem.core.utils import dump_json
from agent_gem.tools.base import BaseTool, ToolExecutionError


class SearchTool(BaseTool):
    """Search via Serper API (Google), cached under `search_cache.json`."""

    def __init__(
        self,
        *,
        cache_path: Path,
        timeout_s: int = 10,
        name: str = "search",
        description: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.cache_path = cache_path
        self.timeout_s = timeout_s
        # API key must come from environment or explicit argument; no built-in fallback
        self.api_key = api_key or os.environ.get("SERPER_API_KEY")
        if not self.api_key:
            raise ValueError(
                "SERPER_API_KEY is not set; please export it (e.g., via run.sh) before using SearchTool."
            )
        self._cache_lock = threading.Lock()

    def execute(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        query = (query or "").strip()
        if not query:
            return []

        cached = self._load_cache()
        if query in cached:
            return cached[query][:max_results]

        url = "https://google.serper.dev/search"
        payload = {"q": query}
        headers = {
            "X-API-KEY": self.api_key,
            "Content-Type": "application/json",
        }

        results: list[dict[str, str]] = []
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout_s)
            resp.raise_for_status()
            data = resp.json()

            organic = data.get("organic") or []
            for item in organic[:max_results]:
                if not isinstance(item, dict):
                    continue
                title = str(item.get("title") or "").strip()
                link = str(item.get("link") or item.get("url") or "").strip()
                snippet = str(item.get("snippet") or item.get("description") or "").strip()
                if title or link:
                    results.append({"title": title, "url": link, "summary": snippet})

            if not results and data.get("answerBox"):
                box = data["answerBox"]
                results.append(
                    {
                        "title": str(box.get("title") or query),
                        "url": str(box.get("link") or ""),
                        "summary": str(box.get("answer") or box.get("snippet") or ""),
                    }
                )

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
