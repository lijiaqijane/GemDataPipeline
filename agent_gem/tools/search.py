from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable

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
        bash_runner: Callable[[str, int | None], dict[str, Any]] | None = None,
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
        self.bash_runner = bash_runner
        self._cache_lock = threading.Lock()

    def execute(self, query: str, max_results: int = 5, page: int = 1) -> list[dict[str, str]]:
        if page != 1:
            is_cached = False
        else:
            is_cached = True

        query = (query or "").strip()
        if not query:
            return []

        cached = self._load_cache()
        if is_cached and query in cached:
            return cached[query][:max_results]

        results: list[dict[str, str]] = []
        try:
            if self.bash_runner is not None:
                results = self._execute_in_sandbox(query, max_results)
            else:
                url = "https://google.serper.dev/search"
                payload = {"q": query, "page": page}
                headers = {
                    "X-API-KEY": self.api_key,
                    "Content-Type": "application/json",
                }
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

            results = self._clean_results(results)
            results = self._rerank_with_jina(query, results)[:max_results]
            if is_cached:
                cached[query] = results
                self._save_cache(cached)
        except Exception as exc:  # pragma: no cover - network may be restricted
            raise ToolExecutionError("search_failed", results, message=str(exc))

        return results

    def _clean_results(self, results: list[dict[str, str]]) -> list[dict[str, str]]:
        cleaned: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            url = str(item.get("url") or "").strip()
            summary = str(item.get("summary") or "").strip()
            if not url.startswith(("http://", "https://")):
                continue
            key = (url or title).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            cleaned.append({"title": title, "url": url, "summary": summary})
        return cleaned

    def _rerank_with_jina(self, query: str, results: list[dict[str, str]]) -> list[dict[str, str]]:
        api_key = os.environ.get("JINA_API_KEY", "").strip()
        if not api_key or not results:
            return results
        if self.bash_runner is None:
            return results

        docs = []
        for item in results:
            title = str(item.get("title") or "").strip()
            summary = str(item.get("summary") or "").strip()
            url = str(item.get("url") or "").strip()
            doc = " ".join(part for part in [title, summary, url] if part)
            docs.append(doc or url or title)

        cmd = (
            "python - <<'PY'\n"
            "import json, os, urllib.request\n"
            "query = os.environ.get('RERANK_QUERY', '')\n"
            "docs = json.loads(os.environ.get('RERANK_DOCS', '[]'))\n"
            "api_key = os.environ.get('JINA_API_KEY', '')\n"
            "payload = json.dumps({\n"
            "  'model': 'jina-rerank-v2-base-multilingual',\n"
            "  'query': query,\n"
            "  'documents': docs,\n"
            "  'top_n': min(20, len(docs))\n"
            "}).encode('utf-8')\n"
            "req = urllib.request.Request(\n"
            "  'https://api.jina.ai/v1/rerank',\n"
            "  data=payload,\n"
            "  headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'}\n"
            ")\n"
            "try:\n"
            "  with urllib.request.urlopen(req, timeout=15) as resp:\n"
            "    data = json.loads(resp.read().decode('utf-8', 'replace'))\n"
            "  print(json.dumps({'data': data.get('data', [])}))\n"
            "except Exception as e:\n"
            "  print(json.dumps({'error': str(e)[:100]}))\n"
            "PY"
        )
        env_prefix = (
            f"JINA_API_KEY={api_key} "
            f"RERANK_QUERY={json.dumps(query)} "
            f"RERANK_DOCS={json.dumps(docs)} "
        )
        result = self.bash_runner(env_prefix + cmd, None)
        stdout = (result.get("stdout") or "").strip()
        try:
            payload = json.loads(stdout) if stdout else {}
        except Exception:
            return results
        if not isinstance(payload, dict) or payload.get("error"):
            return results
        data = payload.get("data")
        if not isinstance(data, list):
            return results

        ranked: list[dict[str, str]] = []
        seen: set[int] = set()
        for item in sorted(data, key=lambda x: x.get("relevance_score", 0), reverse=True):
            if not isinstance(item, dict):
                continue
            idx = item.get("index")
            if not isinstance(idx, int):
                continue
            if idx < 0 or idx >= len(results) or idx in seen:
                continue
            ranked.append(results[idx])
            seen.add(idx)
        return ranked or results

    def _execute_in_sandbox(self, query: str, max_results: int) -> list[dict[str, str]]:
        cmd = (
            "python - <<'PY'\n"
            "import json, urllib.request, os\n"
            "query = os.environ.get('SEARCH_QUERY', '')\n"
            "max_results = int(os.environ.get('SEARCH_MAX', '5'))\n"
            "api_key = os.environ.get('SERPER_API_KEY', '')\n"
            "if not api_key:\n"
            "    print(json.dumps({'error': 'missing_api_key'}))\n"
            "    raise SystemExit(0)\n"
            "url = 'https://google.serper.dev/search'\n"
            "payload = json.dumps({'q': query}).encode('utf-8')\n"
            "req = urllib.request.Request(url, data=payload, headers={'X-API-KEY': api_key, 'Content-Type': 'application/json'})\n"
            "try:\n"
            "    with urllib.request.urlopen(req, timeout=10) as resp:\n"
            "        data = json.loads(resp.read().decode('utf-8', 'replace'))\n"
            "    results = []\n"
            "    organic = data.get('organic') or []\n"
            "    for item in organic[:max_results]:\n"
            "        if not isinstance(item, dict):\n"
            "            continue\n"
            "        title = str(item.get('title') or '').strip()\n"
            "        link = str(item.get('link') or item.get('url') or '').strip()\n"
            "        snippet = str(item.get('snippet') or item.get('description') or '').strip()\n"
            "        if title or link:\n"
            "            results.append({'title': title, 'url': link, 'summary': snippet})\n"
            "    if not results and data.get('answerBox'):\n"
            "        box = data['answerBox']\n"
            "        results.append({'title': str(box.get('title') or query), 'url': str(box.get('link') or ''), 'summary': str(box.get('answer') or box.get('snippet') or '')})\n"
            "    print(json.dumps({'results': results}))\n"
            "except Exception as e:\n"
            "    print(json.dumps({'error': str(e)[:100], 'results': []}))\n"
            "PY"
        )

        if self.bash_runner is None:
            raise ToolExecutionError("search_failed", [], message="bash_runner not configured")

        # Pass secrets via env to avoid writing them in the command body.
        env_prefix = (
            f"SERPER_API_KEY={self.api_key} "
            f"SEARCH_QUERY={json.dumps(query)} "
            f"SEARCH_MAX={max_results} "
        )
        result = self.bash_runner(env_prefix + cmd, None)
        stdout = (result.get("stdout") or "").strip()
        try:
            payload = json.loads(stdout) if stdout else {}
        except Exception as exc:
            raise ToolExecutionError("search_failed", [], message=f"invalid output: {exc}")

        if isinstance(payload, dict) and payload.get("error"):
            raise ToolExecutionError(
                "search_failed", payload.get("results", []), message=str(payload.get("error"))
            )

        records = payload.get("results") if isinstance(payload, dict) else None
        if not isinstance(records, list):
            return []
        filtered: list[dict[str, str]] = []
        for item in records[:max_results]:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            link = str(item.get("url") or item.get("link") or "").strip()
            summary = str(item.get("summary") or item.get("snippet") or "").strip()
            if title or link:
                filtered.append({"title": title, "url": link, "summary": summary})
        return filtered

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


class VisitTool(BaseTool):
    """Visit a URL and return the content."""

    def __init__(
        self,
        *,
        timeout_s: int = 100,
        name: str = "visit",
        description: str | None = None,
        api_key: str | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.timeout_s = timeout_s
        self.api_key = api_key or os.environ.get("JINA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "JINA_API_KEY is not set; please export it (e.g., via run.sh) before using VisitTool."
            )

    def execute(self, url: str, goal: str) -> str:
        max_retries = 3
        base_url = "https://r.jina.ai"

        for attempt in range(max_retries):
            headers = {
                "Authorization": f"Bearer {self.api_key}",
            }
            try:
                response = requests.get(f"{base_url}/{url}", headers=headers, timeout=self.timeout_s)
                if response.status_code == 200:
                    webpage_content = response.text
                    return webpage_content
                else:
                    print(response.text)
                    raise ValueError("jina readpage error")
            except requests.exceptions.RequestException as e:
                time.sleep(0.5)
                if attempt == max_retries - 1:
                    return "[visit] Failed to read page."

        return "[visit] Failed to read page."
