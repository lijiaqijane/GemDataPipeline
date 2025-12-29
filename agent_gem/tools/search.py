from __future__ import annotations

import html
import json
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable

import requests

from agent_gem.core.utils import dump_json
from agent_gem.tools.base import BaseTool, ToolExecutionError

logger = logging.getLogger(__name__)


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

    def execute(self, query: str, max_results: int = 5, depth: int = 1) -> list[dict[str, str]]:
        if depth != 1:
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
                payload = {"q": query, "page": depth}
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
    """Visit a URL and return the content using standard HTTP requests."""

    def __init__(
        self,
        *,
        timeout_s: int = 100,
        name: str = "visit",
        description: str | None = None,
        api_key: str | None = None,  # Kept for compatibility, but not used
    ) -> None:
        super().__init__(name=name, description=description)
        self.timeout_s = timeout_s

    def execute(self, url: str, goal: str) -> str:
        """Fetch webpage content using standard HTTP requests."""
        max_retries = 3
        
        # Ensure URL has a scheme
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        }

        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=self.timeout_s, allow_redirects=True)
                if response.status_code == 200:
                    return response.text
                else:
                    return f"[visit] Failed to read page: HTTP {response.status_code}"
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    return f"[visit] Failed to read page: {str(e)}"
                time.sleep(0.5 * (attempt + 1))  # Exponential backoff

        return "[visit] Failed to read page."


class MediaWikiClient:
    """MediaWiki Action API client for Wikipedia content and pageview retrieval."""

    def __init__(
        self,
        endpoint: str = "https://en.wikipedia.org/w/api.php",
        user_agent: str = "MediaWikiClient/1.0 (Educational Purpose)",
    ):
        self.endpoint = endpoint
        self.user_agent = user_agent

    def _request(self, params: dict[str, Any]) -> dict[str, Any]:
        """Send GET request to MediaWiki API."""
        base_params = {
            "format": "json",
            "formatversion": "2",
            "redirects": "1",
        }
        params.update(base_params)
        headers = {"User-Agent": self.user_agent}
        response = requests.get(url=self.endpoint, params=params, headers=headers)
        response.raise_for_status()
        return response.json()

    def _search_pages(self, search_term: str, limit: int = 10) -> list[dict[str, Any]]:
        """Search for pages using MediaWiki search."""
        search_params = {
            "action": "query",
            "list": "search",
            "srsearch": search_term,
            "srlimit": limit,
            "srprop": "title|pageid",
        }
        data = self._request(search_params)
        results = []
        if "query" in data and "search" in data["query"]:
            for item in data["query"]["search"]:
                results.append({"title": item["title"], "pageid": item["pageid"]})
        return results

    def _fetch_pageviews_rest(self, title: str, project: str = "en.wikipedia.org", days: int = 180) -> int | None:
        """Fetch pageviews using Wikimedia Analytics REST API."""
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=days - 1)
        start = start_date.strftime("%Y%m%d")
        end = end_date.strftime("%Y%m%d")
        encoded_title = requests.utils.quote(title.replace(" ", "_"))
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"{project}/all-access/user/{encoded_title}/daily/{start}/{end}"
        )
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                logger.warning(f"Failed to fetch pageviews for {title}: status={response.status_code}")
                return None
            data = response.json()
            total_views = sum(item.get("views", 0) for item in data.get("items", []))
            return int(total_views)
        except Exception as e:
            logger.warning(f"Exception while fetching pageviews for {title}: {e}")
            return None

    def _fetch_page_data(self, page_title: str) -> dict[str, Any] | None:
        """Fetch detailed page data and clean HTML content."""
        parse_params = {
            "action": "parse",
            "page": page_title,
            "prop": "text",
            "formatversion": "2",
            "redirects": "1",
        }
        try:
            data = self._request(parse_params)
            if "error" in data:
                return None
            parse_data = data.get("parse")
            if not parse_data:
                return None
            actual_title = parse_data.get("title")
            page_id = parse_data.get("pageid")
            html_content = parse_data.get("text", "")

            # Clean HTML noise
            noise_patterns = [
                r'<div[^>]*class="[^"]*navbox[^"]*"[^>]*>.*?</div>',
                r'<table[^>]*class="[^"]*infobox[^"]*"[^>]*>.*?</table>',
                r'<table[^>]*class="[^"]*ambox[^"]*"[^>]*>.*?</table>',
                r'<div[^>]*class="[^"]*reflist[^"]*"[^>]*>.*?</div>',
                r'<div[^>]*class="[^"]*printfooter[^"]*"[^>]*>.*?</div>',
                r'<div[^>]*class="[^"]*mw-authority-control[^"]*"[^>]*>.*?</div>',
                r'<div[^>]*id="[^"]*catlinks[^"]*"[^>]*>.*?</div>',
                r'<div[^>]*class="[^"]*portal[^"]*"[^>]*>.*?</div>',
                r'<div[^>]*class="[^"]*hatnote[^"]*"[^>]*>.*?</div>',
                r'<div[^>]*class="[^"]*sidebar[^"]*"[^>]*>.*?</div>',
            ]
            for pattern in noise_patterns:
                html_content = re.sub(pattern, "", html_content, flags=re.DOTALL)

            plain_content = re.sub(r"<[^>]+>", "", html_content)
            plain_content = html.unescape(plain_content)
            plain_content = re.sub(r"\[.*?\]", "", plain_content)

            stop_patterns = [
                r"^See also\s*$",
                r"^References\s*$",
                r"^Notes\s*$",
                r"^External links\s*$",
                r"^Further reading\s*$",
                r"^参见\s*$",
                r"^参考文献\s*$",
                r"^注释\s*$",
                r"^外部链接\s*$",
                r"^相关条目\s*$",
                r"^Category:.*",
                r"^Portal:.*",
                r"^Index of.*",
                r"^vte.*",
            ]
            for pattern in stop_patterns:
                match = re.search(pattern, plain_content, flags=re.IGNORECASE | re.MULTILINE)
                if match:
                    plain_content = plain_content[: match.start()]

            plain_content = re.sub(r"\.mw-parser-output.*\{.*?\}", "", plain_content, flags=re.DOTALL)
            plain_content = re.sub(r"\d+°\d+′[\d.]+″[NSEW].*?/\s*.*?°[NSEW]\s*[\d.]+°[NSEW]", "", plain_content)
            plain_content = re.sub(r"http[s]?://\S+", "", plain_content)
            plain_content = re.sub(r"\n\s*\n", "\n\n", plain_content)
            plain_content = re.sub(r"\n{3,}", "\n\n", plain_content)
            plain_content = re.sub(r" +", " ", plain_content)
            plain_content = plain_content.strip()

            total_pageviews = self._fetch_pageviews_rest(actual_title, days=180)
            return {
                "title": actual_title,
                "pageid": page_id,
                "content": plain_content,
                "pageview": total_pageviews,
            }
        except Exception:
            return None


class MediaWikiTool(BaseTool):
    """Tool for fetching Wikipedia entity data with content and pageviews."""

    def __init__(
        self,
        *,
        endpoint: str = "https://en.wikipedia.org/w/api.php",
        user_agent: str = "MediaWikiClient/1.0 (Educational Purpose)",
        name: str = "mediawiki",
        description: str | None = "Fetch Wikipedia entity data with content and pageviews",
    ):
        super().__init__(name=name, description=description)
        self.client = MediaWikiClient(endpoint=endpoint, user_agent=user_agent)

    def execute(self, search_term: str, limit: int = 1) -> list[dict[str, Any]]:
        """Execute the tool to fetch entity data."""
        return self.fetch_entity_data(search_term, limit)

    def fetch_entity_data(self, entity: Any, limit: int = 1) -> list[dict[str, Any]]:
        """Fetch entity data from Wikipedia."""
        entity_name = entity.name if hasattr(entity, "name") else str(entity)
        search_results = self.client._search_pages(entity_name, limit=limit)
        if not search_results:
            return []
        entity_data_list = []
        for result in search_results:
            page_title = result["title"]
            page_data = self.client._fetch_page_data(page_title)
            if page_data:
                entity_data = {
                    "name": page_data["title"],
                    "domain": entity.domain if entity.domain else "",
                    "description": page_data["content"],
                    "pageview": page_data["pageview"],
                }
                entity_data_list.append(entity_data)
        return entity_data_list
