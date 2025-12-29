from __future__ import annotations

import json
import logging
import os
import re
import shlex
import sys
from urllib.parse import urlparse
from pathlib import Path
from typing import Any

from agent_gem.core.utils import dump_json, slugify
from agent_gem.sandbox import SandboxExecutor

from ..base import BaseAgent, TaskContext

logger = logging.getLogger(__name__)


class DataPipelineMixin:
    """Data ingestion, cleaning, and file materialization helpers."""

    _DATA_FILE_EXTENSIONS = {
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".ndjson",
        ".txt",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".parquet",
    }
    
    # Configuration constants
    # URL fetching limits
    MAX_TABLES_PER_PAGE = 6
    URL_FETCH_TIMEOUT = 8
    URL_FETCH_DEFAULT_TIMEOUT = 10
    DATASET_DOWNLOAD_TIMEOUT = 15
    
    # Content size limits
    MAX_WEBPAGE_BYTES = 2_000_000
    MAX_DATASET_BYTES = 6_000_000
    MAX_CONTENT_PREVIEW = 5000
    
    # Text processing limits
    MIN_TEXT_LINE_LENGTH = 20
    
    # File creation limits
    MAX_METADATA_RECORDS = 10
    MAX_FILENAME_LENGTH = 48
    
    # Search and query limits
    MAX_SEARCH_QUERIES = 10
    MAX_SEARCH_RESULTS_PER_QUERY = 5
    MAX_LLM_TOKENS_FOR_QUERIES = 300
    MAX_LLM_TOKENS_DEFAULT = 10000
    
    # Error message limits
    MAX_ERROR_MESSAGE_LENGTH = 100
    MAX_ERROR_DETAIL_LENGTH = 300
    MAX_ERROR_LIST_LENGTH = 200
    MAX_STDERR_LENGTH = 120
    
    # CSV/File reading limits
    CSV_SAMPLE_BYTES = 4096
    DOWNLOAD_CHUNK_SIZE = 65536
    
    # Content filtering limits
    MAX_FILTERED_LINES = 400
    MAX_LINKS_IN_PAGE = 20
    MAX_TABLES_FOR_INSPECT = 5
    
    # String template constants (for inline Python code)
    DEFAULT_TIMEOUT_STR = "10"
    CHUNK_CHECK_BYTES = 512
    MAX_STDOUT_DETAIL = 500

    def _fetch_url_content(self, sandbox: SandboxExecutor, url: str, timeout: int | None = None) -> dict[str, Any]:
        """Fetch and extract content from a URL via sandbox bash tool."""
        if timeout is None:
            timeout = self.URL_FETCH_DEFAULT_TIMEOUT
        
        result = {
            "url": url,
            "success": False,
            "content": "",
            "tables": [],
            "links": [],
            "error": None,
            "error_detail": None,
        }

        if not url or not url.startswith(("http://", "https://")):
            result["error"] = "Invalid URL"
            return result

        data_exts = sorted(self._DATA_FILE_EXTENSIONS)
        env_vars = (
            f"URL={shlex.quote(url)} "
            f"TIMEOUT={shlex.quote(str(timeout))} "
            f"MAX_BYTES={self.MAX_WEBPAGE_BYTES} "
            f"DATA_EXTS={shlex.quote(','.join(data_exts))} "
        )
        cmd = (
            f"{env_vars} python - <<'PY'\n"
            "import hashlib, html, json, os, pathlib, re, time, urllib.error, urllib.parse, urllib.request\n"
            "from html.parser import HTMLParser\n"
            "# Install cloudscraper if not already installed\n"
            "try:\n"
            "    import cloudscraper\n"
            "except ImportError:\n"
            "    import subprocess, sys\n"
            "    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', 'cloudscraper'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\n"
            "    import cloudscraper\n"
            "url = os.environ.get('URL', '')\n"
            f"timeout = float(os.environ.get('TIMEOUT', '{self.DEFAULT_TIMEOUT_STR}'))\n"
            f"max_bytes = int(os.environ.get('MAX_BYTES', '{self.MAX_WEBPAGE_BYTES}'))\n"
            "exts = [e for e in os.environ.get('DATA_EXTS', '').split(',') if e]\n"
            "errors = []\n"
            "def is_data_file(u: str) -> bool:\n"
            "    path = urllib.parse.urlparse(u).path.lower()\n"
            "    return any(path.endswith(ext) for ext in exts)\n"
            "def extract(text: str, base_url: str) -> tuple[str, list[str], list[list[list[str]]]]:\n"
            "    text = re.sub(r'<script[^>]*>.*?</script>', ' ', text, flags=re.I|re.S)\n"
            "    text = re.sub(r'<style[^>]*>.*?</style>', ' ', text, flags=re.I|re.S)\n"
            "    cleaned = re.sub(r'<[^>]+>', ' ', text)\n"
            "    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]\n"
            "    filtered = []\n"
            "    for ln in lines:\n"
            f"        if len(ln) < {self.MIN_TEXT_LINE_LENGTH}:\n"
            "            continue\n"
            "        if re.search(r'[{};<>]|function\\s*\\(|var\\s+|let\\s+|const\\s+|@keyframes', ln):\n"
            "            continue\n"
            "        if re.search(r'\\b(document|window|navigator|jquery|\\$)\\b', ln, flags=re.I):\n"
            "            continue\n"
            "        filtered.append(ln)\n"
            f"    content = '\\n'.join(filtered[:{self.MAX_FILTERED_LINES}])\n"
            "    hrefs = re.findall(r'href=[\"\\'](.*?)[\"\\']', text, flags=re.I)\n"
            "    links = []\n"
            "    for href in hrefs:\n"
            "        full = urllib.parse.urljoin(base_url, href)\n"
            "        if is_data_file(full):\n"
            "            links.append(full)\n"
            "    tables = []\n"
            "    try:\n"
            "        class TableParser(HTMLParser):\n"
            "            def __init__(self):\n"
            "                super().__init__()\n"
            "                self.tables = []\n"
            "                self._table = []\n"
            "                self._row = []\n"
            "                self._cell = []\n"
            "                self._in_table = False\n"
            "                self._in_row = False\n"
            "                self._in_cell = False\n"
            "            def handle_starttag(self, tag, attrs):\n"
            "                if tag == 'table':\n"
            "                    self._in_table = True\n"
            "                    self._table = []\n"
            "                elif self._in_table and tag == 'tr':\n"
            "                    self._in_row = True\n"
            "                    self._row = []\n"
            "                elif self._in_row and tag in ('td', 'th'):\n"
            "                    self._in_cell = True\n"
            "                    self._cell = []\n"
            "            def handle_endtag(self, tag):\n"
            "                if tag in ('td', 'th') and self._in_cell:\n"
            "                    cell_text = ' '.join(self._cell).strip()\n"
            "                    cell_text = re.sub(r'\\s+', ' ', cell_text)\n"
            "                    self._row.append(cell_text)\n"
            "                    self._in_cell = False\n"
            "                elif tag == 'tr' and self._in_row:\n"
            "                    if self._row and any(c for c in self._row):\n"
            "                        self._table.append(self._row)\n"
            "                    self._in_row = False\n"
            "                elif tag == 'table' and self._in_table:\n"
            "                    if self._table:\n"
            "                        self.tables.append(self._table)\n"
            "                    self._in_table = False\n"
            "            def handle_data(self, data):\n"
            "                if self._in_cell:\n"
            "                    self._cell.append(data)\n"
            "        parser = TableParser()\n"
            "        parser.feed(text)\n"
            f"        tables = parser.tables[:{self.MAX_TABLES_PER_PAGE}]\n"
            "    except Exception:\n"
            "        tables = []\n"
            "    return content, links, tables\n"
            "def fetch_with_cloudscraper(u: str) -> tuple[bool, str, list[str], list[list[list[str]]], str]:\n"
            "    import cloudscraper\n"
            "    scraper = cloudscraper.create_scraper(\n"
            "        browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}\n"
            "    )\n"
            "    resp = scraper.get(u, timeout=timeout)\n"
            "    if resp.status_code == 200:\n"
            "        raw = resp.content[:max_bytes + 1]\n"
            "        if len(raw) > max_bytes:\n"
            "            raw = raw[:max_bytes]\n"
            "        text = raw.decode('utf-8', 'replace')\n"
            "        content, links, tables = extract(text, u)\n"
            f"        return True, content[:{self.MAX_CONTENT_PREVIEW}], links[:{self.MAX_LINKS_IN_PAGE}], tables[:{self.MAX_TABLES_PER_PAGE}], ''\n"
            "    else:\n"
            "        raise ValueError(f'HTTP {resp.status_code}')\n"
            "def fetch_once(u: str, headers: dict[str, str]) -> tuple[bool, str, list[str], list[list[list[str]]], str]:\n"
            "    req = urllib.request.Request(u, headers=headers)\n"
            "    with urllib.request.urlopen(req, timeout=timeout) as resp:\n"
            "        status_code = resp.getcode()\n"
            "        if status_code != 200:\n"
            "            raise urllib.error.HTTPError(u, status_code, 'HTTP Error', resp.headers, None)\n"
            "        raw = resp.read(max_bytes + 1)\n"
            "    if len(raw) > max_bytes:\n"
            "        raw = raw[:max_bytes]\n"
            "    text = raw.decode('utf-8', 'replace')\n"
            "    # Check for Cloudflare/403 indicators\n"
            "    if 'cf-browser-verification' in text.lower() or 'challenge-platform' in text.lower() or 'checking your browser' in text.lower():\n"
            "        raise ValueError('Cloudflare challenge detected')\n"
            "    content, links, tables = extract(text, u)\n"
            f"    return True, content[:{self.MAX_CONTENT_PREVIEW}], links[:{self.MAX_LINKS_IN_PAGE}], tables[:{self.MAX_TABLES_PER_PAGE}], ''\n"
            "def try_fetch(u: str) -> tuple[bool, str, list[str], list[list[list[str]]], str]:\n"
            "    # Use cloudscraper (required)\n"
            "    try:\n"
            "        return fetch_with_cloudscraper(u)\n"
            "    except ImportError as e:\n"
            "        error_msg = 'cloudscraper is required but not installed. Please install: pip install cloudscraper'\n"
            f"        errors.append(error_msg)\n"
            "        raise ImportError(error_msg) from e\n"
            "    except Exception as e:\n"
            f"        errors.append(f'cloudscraper_error: {{str(e)[:{self.MAX_STDERR_LENGTH}]}}')\n"
            "        raise\n"
            "try:\n"
            "    if not url or not url.startswith(('http://', 'https://')):\n"
            "        raise ValueError('Invalid URL')\n"
            "    ok, content, links, tables, err = try_fetch(url)\n"
            "    if ok:\n"
            "        out = {'success': True, 'content': content, 'tables': tables, 'links': links, 'error': None, 'error_detail': None}\n"
            "        print(json.dumps(out, ensure_ascii=False))\n"
            "    else:\n"
            f"        out = {{'success': False, 'content': '', 'tables': [], 'links': [], 'error': err or 'fetch_failed', 'error_detail': '; '.join(errors)[:{self.MAX_ERROR_DETAIL_LENGTH}]}}\n"
            "        print(json.dumps(out, ensure_ascii=False))\n"
            "except Exception as e:\n"
            f"    print(json.dumps({{'success': False, 'content': '', 'tables': [], 'links': [], 'error': str(e)[:{self.MAX_ERROR_MESSAGE_LENGTH}], 'error_detail': '; '.join(errors)[:{self.MAX_ERROR_DETAIL_LENGTH}]}}, ensure_ascii=False))\n"
            "PY"
        )

        tool_result = sandbox.execute_bash(cmd, timeout_s=timeout + 5)
        stdout = (tool_result.get("stdout") or "").strip()
        stderr = (tool_result.get("stderr") or "").strip()
        returncode = tool_result.get("returncode") or tool_result.get("return_code") or 0
        if isinstance(returncode, str) and returncode.isdigit():
            returncode = int(returncode)

        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(stdout) if stdout else {}
        except Exception as exc:
            result["error"] = f"Parse error: {str(exc)[:self.MAX_ERROR_MESSAGE_LENGTH]}"
            if stdout:
                result["error_detail"] = stdout[:self.MAX_STDOUT_DETAIL]
            return result

        if isinstance(parsed, dict):
            result.update(parsed)
        else:
            result["error"] = "Invalid tool output"
            if stdout:
                result["error_detail"] = stdout[:self.MAX_STDOUT_DETAIL]
            return result

        # If tool output says success, require some payload before trusting it.
        if result.get("success"):
            has_payload = bool(result.get("content")) or bool(result.get("links"))
            if has_payload:
                return result

        if returncode and returncode != 0:
            result["error"] = result.get("error") or "Tool error"
            detail = stderr or stdout
            if detail:
                result["error_detail"] = detail[:self.MAX_STDOUT_DETAIL]
            return result

        if result.get("error") and not result.get("error_detail"):
            result["error_detail"] = str(result.get("error"))[:self.MAX_STDOUT_DETAIL]
        return result

    def _is_data_file_url(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        # Exclude framework internal JSON files (Gatsby, Next.js, etc.)
        if path.endswith('.json'):
            # Exclude known framework internal file patterns
            if '/page-data/' in path and 'page-data' in path:
                return False
            if '/manifest' in path and path.endswith('.json'):
                return False
            if path.startswith('/static/') and path.endswith('.json'):
                return False
        return any(path.endswith(ext) for ext in self._DATA_FILE_EXTENSIONS)

    def _download_dataset_file(
        self,
        sandbox: SandboxExecutor,
        url: str,
        dest_dir: Path,
        *,
        max_bytes: int | None = None,
        timeout: int | None = None,
    ) -> Path | None:
        if max_bytes is None:
            env_value = os.getenv("DATASET_MAX_BYTES", "").strip()
            try:
                max_bytes = int(env_value) if env_value else self.MAX_DATASET_BYTES
            except ValueError:
                max_bytes = self.MAX_DATASET_BYTES
        if timeout is None:
            timeout = self.DATASET_DOWNLOAD_TIMEOUT
        url = self._rewrite_data_url(url)
        if not self._is_data_file_url(url):
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(urlparse(url).path).suffix.lower()
        filename_base = slugify(urlparse(url).path) or "dataset"
        filename = f"{filename_base[:self.MAX_FILENAME_LENGTH]}{suffix}"
        target = dest_dir / filename
        if target.exists():
            return target
        try:
            rel_target = target.relative_to(sandbox.sandbox_dir)
        except ValueError:
            logger.warning("Target path outside sandbox; skipping download: %s", target)
            return None

        cmd = (
            f"URL={shlex.quote(url)} "
            f"TARGET={shlex.quote(rel_target.as_posix())} "
            f"MAX_BYTES={shlex.quote(str(max_bytes))} "
            f"TIMEOUT={shlex.quote(str(timeout))} "
            "python - <<'PY'\n"
            "import json, os, pathlib, urllib.request\n"
            "url = os.environ.get('URL', '')\n"
            "target = os.environ.get('TARGET', '')\n"
            f"max_bytes = int(os.environ.get('MAX_BYTES', '{self.MAX_DATASET_BYTES}'))\n"
            f"timeout = float(os.environ.get('TIMEOUT', '{self.DATASET_DOWNLOAD_TIMEOUT}'))\n"
            "errors = []\n"
            "headers_list = [\n"
            "  {'User-Agent': 'Mozilla/5.0', 'Accept': '*/*', 'Accept-Language': 'en-US,en;q=0.9'},\n"
            "  {'User-Agent': 'Mozilla/5.0', 'Accept': 'text/csv,application/json;q=0.9,*/*;q=0.8'},\n"
            "]\n"
            "try:\n"
            "    if not url or not target:\n"
            "        raise ValueError('Missing URL or TARGET')\n"
            "    path = pathlib.Path(target)\n"
            "    path.parent.mkdir(parents=True, exist_ok=True)\n"
            "    if path.exists():\n"
            "        print(json.dumps({'ok': True, 'path': str(path)}))\n"
            "        raise SystemExit(0)\n"
            "    def attempt(headers: dict[str, str]) -> None:\n"
            "        req = urllib.request.Request(url, headers=headers)\n"
            "        with urllib.request.urlopen(req, timeout=timeout) as resp:\n"
            "            content_type = (resp.headers.get('Content-Type') or '').lower()\n"
            "            size = 0\n"
            "            with open(path, 'wb') as f:\n"
            "                while True:\n"
            f"                    chunk = resp.read({self.DOWNLOAD_CHUNK_SIZE})\n"
            "                    if not chunk:\n"
            "                        break\n"
            "                    if size == 0:\n"
            f"                        lower = chunk[:{self.CHUNK_CHECK_BYTES}].lower()\n"
            "                        if 'text/html' in content_type or b'<html' in lower or b'access denied' in lower:\n"
            "                            raise ValueError('HTML/block page')\n"
            "                    size += len(chunk)\n"
            "                    if size > max_bytes:\n"
            "                        raise ValueError('File too large')\n"
            "                    f.write(chunk)\n"
            "            if size == 0:\n"
            "                raise ValueError('Empty file')\n"
            "    ok = False\n"
            "    for headers in headers_list:\n"
            "        try:\n"
            "            attempt(headers)\n"
            "            ok = True\n"
            "            break\n"
            "        except Exception as e:\n"
            f"            errors.append(str(e)[:{self.MAX_STDERR_LENGTH}])\n"
            "            if path.exists():\n"
            "                path.unlink(missing_ok=True)\n"
            "    if not ok:\n"
            f"        raise ValueError('; '.join(errors)[:{self.MAX_ERROR_LIST_LENGTH}])\n"
            "    print(json.dumps({'ok': True, 'path': str(path)}))\n"
            "except Exception as e:\n"
            "    if 'path' in locals() and path.exists():\n"
            "        path.unlink(missing_ok=True)\n"
            f"    print(json.dumps({{'ok': False, 'error': str(e)[:{self.MAX_ERROR_LIST_LENGTH}]}}))\n"
            "PY"
        )

        tool_result = sandbox.execute_bash(cmd, timeout_s=timeout + 5)
        stdout = (tool_result.get("stdout") or "").strip()
        stderr = (tool_result.get("stderr") or "").strip()
        try:
            parsed = json.loads(stdout) if stdout else {}
        except Exception as exc:
            detail = stderr or stdout
            logger.warning("Failed to parse download tool output: %s", detail or exc)
            return None

        if not isinstance(parsed, dict) or not parsed.get("ok"):
            if isinstance(parsed, dict) and parsed.get("error"):
                logger.warning("Dataset download failed: %s", parsed.get("error"))
            return None

        return target

    def _normalize_url(self, url: str, *, rewrite_github: bool = False) -> str:
        """Normalize and optionally rewrite URL.
        
        Args:
            url: URL to normalize
            rewrite_github: If True, rewrite GitHub blob URLs to raw URLs
            
        Returns:
            Normalized URL string
        """
        if not url or not (url := url.strip()):
            return ""
        
        try:
            parsed = urlparse(url)
        except Exception:
            return url
        
        # Rewrite GitHub blob URLs to raw URLs if requested
        if rewrite_github:
            host = parsed.netloc.lower()
            path = parsed.path
            if "github.com" in host and "/blob/" in path:
                raw_path = path.replace("/blob/", "/")
                return f"https://raw.githubusercontent.com{raw_path}"
        
        # Normalize URL components
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        
        # Remove trailing slash (except for root path)
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        
        return f"{scheme}://{netloc}{path}"
    
    def _rewrite_data_url(self, url: str) -> str:
        """Rewrite data URLs (e.g., GitHub blob to raw) (alias for _normalize_url with rewrite_github=True)."""
        return self._normalize_url(url, rewrite_github=True)

    def _generate_search_queries(self, topic: str, *, max_queries: int | None = None) -> list[str]:
        """Generate 10 diverse search queries related to topic, ensuring at least 2 data download queries."""
        if max_queries is None:
            max_queries = self.MAX_SEARCH_QUERIES
        
        prompt = (
            "Generate 10 diverse search queries for the given topic. "
            "Ensure queries are varied and related to the topic. "
            "At least 2 queries should be for downloading data files (e.g., 'topic dataset csv', 'topic data download'). "
            "Output JSON with key 'queries' containing a list of exactly 10 search query strings. "
            "Return ONLY JSON.\n"
            f"Topic: {topic}\n"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.3, max_tokens=self.MAX_LLM_TOKENS_FOR_QUERIES)
        parsed = self._extract_json(raw)

        queries: list[str] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
            queries = [str(x).strip() for x in parsed["queries"] if str(x).strip()]
        elif isinstance(parsed, list):
            queries = [str(x).strip() for x in parsed if str(x).strip()]

        # Ensure we have exactly max_queries queries
        if len(queries) < max_queries:
            # Add data download queries if missing
            data_terms = ["dataset", "data csv", "download data", "data file"]
            count_data_queries = sum(1 for q in queries if any(term in q.lower() for term in data_terms))
            if count_data_queries < 2:
                needed = 2 - count_data_queries
                for i in range(needed):
                    if len(queries) >= max_queries:
                        break
                    queries.append(f"{topic} {data_terms[i % len(data_terms)]}")
            
            # Fill remaining slots with topic variations
            while len(queries) < max_queries:
                queries.append(f"{topic}")
        
        # Deduplicate and limit
        seen: set[str] = set()
        deduped: list[str] = []
        for query in queries:
            key = query.lower()
            if key not in seen:
                seen.add(key)
                deduped.append(query)
                if len(deduped) >= max_queries:
                    break
        
        return deduped[:max_queries]

    def _generate_data_download_queries(self, topic: str, count: int = 5, attempt: int = 1) -> list[str]:
        """Generate queries specifically for data file downloads."""
        attempt_idx = max(1, attempt)
        prompt = (
            f"Attempt {attempt_idx}: Generate {count} search queries specifically for downloading data files related to '{topic}'. "
            "Each query should focus on finding downloadable datasets, CSV files, JSON files, or other data formats. "
            "Output JSON with key 'queries' containing a list of exactly "
            f"{count} search query strings. Return ONLY JSON.\n"
            f"Topic: {topic}\n"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.7, max_tokens=self.MAX_LLM_TOKENS_FOR_QUERIES)
        parsed = self._extract_json(raw)
        
        queries: list[str] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
            queries = [str(x).strip() for x in parsed["queries"] if str(x).strip()]
        elif isinstance(parsed, list):
            queries = [str(x).strip() for x in parsed if str(x).strip()]
        
        # Ensure we have enough queries
        data_terms = ["dataset", "data csv", "download data", "data file", "csv download"]
        while len(queries) < count:
            idx = len(queries) % len(data_terms)
            queries.append(f"{topic} {data_terms[idx]}")
        
        return queries[:count]

    def _seed_database(
        self, topic: str, ctx: TaskContext, sandbox: SandboxExecutor
    ) -> list[dict[str, Any]]:
        """Simplified database seeding: generate queries, fetch URLs, download data or summarize content."""
        # Step 1: Generate 10 diverse queries (at least 2 data download related)
        search_queries = self._generate_search_queries(topic)
        
        # Log search queries
        queries_log_path = sandbox.sandbox_dir / "logs" / "search_queries.jsonl"
        queries_log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(queries_log_path, "a", encoding="utf-8") as f:
                for query in search_queries:
                    log_entry = {
                        "topic": topic,
                        "query": query,
                        "timestamp": ctx.history[-1].timestamp if ctx.history else None,
                    }
                    f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Failed to log search queries: %s", e)
        
        ctx.add_step({
            "type": "search_queries_generated",
            "topic": topic,
            "queries": search_queries,
            "count": len(search_queries),
        })
        
        # Step 2: Get URLs from queries (5 per query), deduplicate
        search_hits: list[dict[str, Any]] = []
        for query in search_queries:
            result = sandbox.execute_search(query, max_results=self.MAX_SEARCH_RESULTS_PER_QUERY)
            if isinstance(result, list):
                for row in result:
                    if isinstance(row, dict):
                        search_hits.append(row)
        
        # Deduplicate URLs
        seen_urls: set[str] = set()
        unique_hits: list[dict[str, Any]] = []
        
        for hit in search_hits:
            url = str(hit.get("url") or hit.get("link") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            unique_hits.append(hit)
        
        # Count data URLs
        data_url_count = sum(1 for hit in unique_hits if self._is_data_file_url(str(hit.get("url") or hit.get("link") or "")))
        logger.info(f"Initial search: {len(unique_hits)} unique URLs, {data_url_count} are data URLs")
        
        # Step 2.5: If data URLs are insufficient, generate additional queries specifically for data files
        # Maximum 3 attempts to find data URLs
        max_data_url_attempts = 3
        data_url_attempt = 0
        
        while data_url_count < 5 and data_url_attempt < max_data_url_attempts:
            data_url_attempt += 1
            logger.info(f"Insufficient data URLs ({data_url_count}/5). Attempt {data_url_attempt}/{max_data_url_attempts}: Generating data-specific queries...")
            additional_data_queries = self._generate_data_download_queries(
                topic, count=5, attempt=data_url_attempt
            )
            
            # Fetch URLs from data-specific queries, but only keep data URLs
            for query in additional_data_queries:
                result = sandbox.execute_search(query, max_results=self.MAX_SEARCH_RESULTS_PER_QUERY)
                if isinstance(result, list):
                    for row in result:
                        if isinstance(row, dict):
                            url = str(row.get("url") or row.get("link") or "").strip()
                            if url and url not in seen_urls and self._is_data_file_url(url):
                                seen_urls.add(url)
                                unique_hits.append(row)
                                data_url_count += 1
                                if data_url_count >= 5:
                                    break
                if data_url_count >= 5:
                    break
            
            logger.info(f"After attempt {data_url_attempt}: {data_url_count} data URLs found")
        
        if data_url_count < 5:
            logger.warning(f"Only found {data_url_count} data URLs after {max_data_url_attempts} attempts (target: 5). Proceeding with available URLs.")
        else:
            logger.info(f"Sufficient data URLs found: {data_url_count} (target: 5)")
        
        # Limit non-data URLs to 20, but keep all data URLs
        data_hits: list[dict[str, Any]] = []
        non_data_hits: list[dict[str, Any]] = []
        
        for hit in unique_hits:
            url = str(hit.get("url") or hit.get("link") or "").strip()
            if self._is_data_file_url(url):
                data_hits.append(hit)
            else:
                non_data_hits.append(hit)
        
        # Keep all data URLs, but limit non-data URLs to 20
        max_non_data_urls = 20
        limited_non_data_hits = non_data_hits[:max_non_data_urls]
        all_hits = data_hits + limited_non_data_hits
        
        if len(non_data_hits) > max_non_data_urls:
            logger.info(f"Limited non-data URLs from {len(non_data_hits)} to {max_non_data_urls} (keeping all {len(data_hits)} data URLs)")
        
        logger.info(f"Final URL list: {len(data_hits)} data URLs + {len(limited_non_data_hits)} non-data URLs = {len(all_hits)} total URLs")
        
        # Step 3: Process URLs - download data files or summarize content to txt (1-5 files per URL)
        data_dir = sandbox.sandbox_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        
        # URL to file mapping (will be saved to disk)
        # Structure: {url: {"files": [file_paths], "error": error_msg or None, "success": bool}}
        url_file_mapping: dict[str, dict[str, Any]] = {}
        
        total_urls = len(all_hits)
        logger.info(f"Processing {total_urls} URLs...")
        processed_count = 0
        success_count = 0
        error_count = 0
        
        for hit in all_hits:
            processed_count += 1
            title = str(hit.get("title") or hit.get("name") or topic).strip()
            url = str(hit.get("url") or hit.get("link") or "").strip()
            
            # Update progress bar on the same line
            if total_urls > 0:
                percentage = (processed_count * 100) // total_urls
                bar_length = 30
                filled = (processed_count * bar_length) // total_urls
                bar = "=" * filled + "-" * (bar_length - filled)
                progress_msg = f"Progress: [{bar}] {processed_count}/{total_urls} ({percentage}%)"
                sys.stdout.write(f"\r{progress_msg}")
                sys.stdout.flush()
            
            if not url:
                error_count += 1
                continue
            
            record = {
                "title": title,
                "summary": str(hit.get("summary") or hit.get("snippet") or "").strip(),
                "url": url,
                "source": "search",
            }
            
            file_saved = False
            error_reasons: list[str] = []
            saved_files: list[str] = []
            
            # If it's a data file, download it
            if self._is_data_file_url(url):
                local_path = self._download_dataset_file(sandbox, url, data_dir)
                if local_path:
                    rel_path = str(local_path.relative_to(sandbox.sandbox_dir))
                    saved_files.append(rel_path)
                    record["downloaded_files"] = [rel_path]
                    record["source"] = "downloaded_dataset"
                    file_saved = True
                    success_count += 1
                else:
                    error_reasons.append("Failed to download data file")
                    error_count += 1
            else:
                # Otherwise, fetch content and summarize to txt files (1-5 files per URL)
                url_data = self._fetch_url_content(sandbox, url, timeout=self.URL_FETCH_TIMEOUT)
                
                if url_data.get("success") and url_data.get("content"):
                    content = url_data["content"]
                    
                    # Check for data file links in the page
                    for link in url_data.get("links", [])[:10]:
                        if self._is_data_file_url(link):
                            local_path = self._download_dataset_file(sandbox, link, data_dir)
                            if local_path:
                                rel_path = str(local_path.relative_to(sandbox.sandbox_dir))
                                saved_files.append(rel_path)
                                record.setdefault("downloaded_files", []).append(rel_path)
                                file_saved = True
                                if len(record.get("downloaded_files", [])) >= 5:
                                    break
                    
                    # Generate summary txt files (1-5 files per URL)
                    max_tokens = getattr(ctx.request, "max_tokens", self.MAX_LLM_TOKENS_DEFAULT)
                    num_files = min(5, max(1, len(content) // 5000))  # 1-5 files based on content length
                    
                    for i in range(num_files):
                        chunk_start = i * (len(content) // num_files)
                        chunk_end = (i + 1) * (len(content) // num_files) if i < num_files - 1 else len(content)
                        chunk = content[chunk_start:chunk_end]
                        
                        summary_prompt = (
                            f"Summarize the following content about '{topic}' into a clear, structured text. "
                            "Focus on key information, facts, and insights. Return plain text only, no JSON.\n\n"
                            f"Content:\n{chunk[:4000]}\n\nSummary:"
                        )
                        
                        try:
                            summary_text = self.llm.simple_complete(summary_prompt, temperature=0.3, max_tokens=max_tokens)
                            
                            # Save to txt file
                            safe_title = slugify(title, max_length=40) or "content"
                            filename = f"{safe_title}_{i+1}.txt" if num_files > 1 else f"{safe_title}.txt"
                            txt_path = data_dir / filename
                            
                            # Avoid overwriting
                            counter = 1
                            while txt_path.exists():
                                filename = f"{safe_title}_{i+1}_{counter}.txt" if num_files > 1 else f"{safe_title}_{counter}.txt"
                                txt_path = data_dir / filename
                                counter += 1
                            
                            txt_path.write_text(summary_text, encoding="utf-8")
                            rel_path = str(txt_path.relative_to(sandbox.sandbox_dir))
                            saved_files.append(rel_path)
                            record.setdefault("downloaded_files", []).append(rel_path)
                            file_saved = True
                            
                        except Exception as e:
                            error_msg = f"Failed to summarize content chunk {i+1}: {str(e)[:100]}"
                            error_reasons.append(error_msg)
                    
                    if file_saved:
                        success_count += 1
                    else:
                        error_count += 1
                        error_reasons.append("Failed to generate summary files")
                else:
                    # Failed to fetch
                    fetch_error = url_data.get("error", "Unknown error")
                    error_reasons.append(f"Failed to fetch content: {fetch_error}")
                    error_count += 1
            
            # Record URL to file mapping
            url_file_mapping[url] = {
                "files": saved_files if file_saved else [],
                "error": "; ".join(error_reasons) if error_reasons else None,
                "success": file_saved,
            }
            
            # Always record, with error information if no file was saved
            if not file_saved and error_reasons:
                record["error"] = "; ".join(error_reasons)
            records.append(record)
        
        # Print newline after progress bar is complete
        if total_urls > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()
        
        # Save URL to file mapping to disk
        mapping_path = sandbox.sandbox_dir / "logs" / "url_file_mapping.json"
        try:
            dump_json(mapping_path, url_file_mapping)
            mapping_rel_path = str(mapping_path.relative_to(sandbox.sandbox_dir))
            logger.info(f"Processing complete: {success_count} succeeded, {error_count} failed out of {processed_count} URLs")
            logger.info(f"URL to file mapping saved to: {mapping_rel_path}")
        except Exception as e:
            logger.warning(f"Failed to save URL to file mapping: {e}")
            logger.info(f"Processing complete: {success_count} succeeded, {error_count} failed out of {processed_count} URLs")
        
        # Save records to db.json
        merged_records = self.writer.merge_records(records)
        self.writer.records = merged_records
        
        task_dir = self.writer.task_dir(ctx.task_id, self.agent_type)
        task_db_path = task_dir / "db.json"
        task_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        payload = {"records": merged_records, "search_queries": search_queries}
        dump_json(task_db_path, payload)
        
        # Final summary
        files_with_errors = [r for r in records if r.get("error")]
        files_success = [r for r in records if r.get("downloaded_files") and not r.get("error")]
        final_data_url_count = sum(1 for hit in all_hits if self._is_data_file_url(str(hit.get("url") or hit.get("link") or "")))
        final_total_url_count = len(all_hits)
        
        logger.info("=" * 70)
        logger.info("Database Seeding Summary")
        logger.info("=" * 70)
        logger.info(f"Topic: {topic}")
        logger.info(f"Total queries generated: {len(search_queries)}")
        logger.info(f"Total unique URLs: {final_total_url_count}")
        logger.info(f"Data URLs: {final_data_url_count}")
        logger.info(f"URLs processed: {processed_count}")
        logger.info(f"URLs with files saved: {len(files_success)}")
        logger.info(f"URLs with errors: {len(files_with_errors)}")
        logger.info(f"URL to file mapping saved to: logs/url_file_mapping.json")
        logger.info("=" * 70)
        
        ctx.add_step({
            "type": "seed_database",
            "topic": topic,
            "queries": search_queries,
            "records_count": len(records),
            "data_files_count": len(files_success),
            "error_count": len(files_with_errors),
        })
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)
        
        return records

    def _load_seeded_records(self, ctx: TaskContext) -> list[dict[str, Any]] | None:
        """Load previously seeded records from task db.json if present."""
        task_dir = self.writer.task_dir(ctx.task_id, self.agent_type)
        db_path = task_dir / "db.json"
        if not db_path.exists():
            return None
        try:
            payload = json.loads(db_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to read seeded db.json for resume: %s", exc)
            return None
        records = None
        search_queries: list[str] = []
        if isinstance(payload, dict):
            records = payload.get("records")
            if isinstance(payload.get("search_queries"), list):
                search_queries = [str(x) for x in payload.get("search_queries") if str(x).strip()]
        elif isinstance(payload, list):
            records = payload
        if not isinstance(records, list):
            return None

        self.writer.records = [r for r in records if isinstance(r, dict)]
        ctx.add_step(
            {
                "type": "seed_database_resumed",
                "topic": ctx.request.topic,
                "records_count": len(self.writer.records),
                "queries": search_queries,
            }
        )
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)
        return self.writer.records

    def _inspect_data_sources(self, sandbox: SandboxExecutor, ctx: TaskContext) -> dict[str, Any]:
        """Enumerate local data artifacts (CSV/JSON/SQLite/TXT/logs) with lightweight schema samples."""
        import csv
        import sqlite3

        base = sandbox.sandbox_dir
        profile: dict[str, Any] = {"csv": [], "json": [], "sqlite": [], "txt": [], "logs": [], "files": []}

        def _safe_rel(path: Path) -> str:
            try:
                return str(path.relative_to(base))
            except Exception:
                return str(path)

        def _truncate(value: Any, limit: int | None = None) -> Any:
            if limit is None:
                limit = self.MAX_ERROR_MESSAGE_LENGTH
            if isinstance(value, str):
                return value if len(value) <= limit else value[:limit] + "..."
            return value

        def _sample_csv(path: Path, max_rows: int = 5) -> dict[str, Any] | None:
            try:
                with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                    sample_text = f.read(self.CSV_SAMPLE_BYTES)
                    f.seek(0)
                    try:
                        dialect = csv.Sniffer().sniff(sample_text) if sample_text else csv.excel
                    except csv.Error:
                        dialect = csv.excel
                    reader = csv.reader(f, dialect)
                    rows = []
                    for row in reader:
                        if not row or all(not str(cell).strip() for cell in row):
                            continue
                        rows.append([str(cell) for cell in row])
                        if len(rows) >= max_rows + 1:
                            break
                    if not rows:
                        return None
                    header = rows[0]
                    data_rows = rows[1:] if len(rows) > 1 else []
                    return {
                        "path": _safe_rel(path),
                        "header": header,
                        "sample_rows": [
                            [_truncate(cell) for cell in row] for row in data_rows[:max_rows]
                        ],
                    }
            except Exception:
                return None

        def _sample_json(path: Path, max_items: int = 5) -> dict[str, Any] | None:
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                return None
            items: list[Any] = []
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, list):
                        items = value
                        break
            samples = []
            for item in items[:max_items]:
                if isinstance(item, dict):
                    samples.append({k: _truncate(item.get(k, "")) for k in list(item.keys())[:self.MAX_METADATA_RECORDS]})
                else:
                    samples.append(_truncate(item))
            return {
                "path": _safe_rel(path),
                "type": "list" if isinstance(payload, list) else "object",
                "sample_items": samples,
            }

        def _sample_sqlite(path: Path, max_rows: int = 5) -> dict[str, Any] | None:
            try:
                conn = sqlite3.connect(path)
            except Exception:
                return None
            try:
                cur = conn.cursor()
                cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
                tables = [row[0] for row in cur.fetchall() if row and row[0]]
                table_samples = []
                for name in tables[:self.MAX_TABLES_FOR_INSPECT]:
                    try:
                        cur.execute(f"PRAGMA table_info('{name}')")
                        columns = [r[1] for r in cur.fetchall() if r and r[1]]
                        include_rows = True
                        if name.lower() in {"pages", "paragraphs"}:
                            include_rows = False
                        if any(col.lower() in {"raw_html", "text"} for col in columns):
                            include_rows = False
                        rows = []
                        row_count = None
                        if include_rows:
                            cur.execute(f"SELECT * FROM '{name}' LIMIT {max_rows}")
                            rows = [
                                [ _truncate(cell) for cell in row ]
                                for row in cur.fetchall()
                            ]
                        cur.execute(f"SELECT COUNT(*) FROM '{name}'")
                        row_count = cur.fetchone()[0]
                        table_samples.append({
                            "table": name,
                            "columns": columns,
                            "rows": rows,
                            "row_count": row_count,
                        })
                    except Exception:
                        continue
                return {"path": _safe_rel(path), "tables": table_samples}
            finally:
                conn.close()

        def _sample_txt(path: Path, max_lines: int = 20) -> dict[str, Any] | None:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    lines = []
                    line_count = 0
                    for idx, line in enumerate(f, 1):
                        line_count = idx
                        if idx > max_lines:
                            break
                        line = line.strip()
                        if line:
                            lines.append(_truncate(line))
                    if not lines:
                        return None
                    return {
                        "path": _safe_rel(path),
                        "preview_lines": lines,
                        "line_count": min(line_count, max_lines),
                    }
            except Exception:
                return None

        # Only scan data/ directory
        data_dir = base / "data"
        if not data_dir.exists():
            return profile
        
        for path in data_dir.rglob("*"):
            if path.is_dir():
                continue
            rel = _safe_rel(path)
            # Skip raw directory files from profile (only include processed data)
            if "raw" in rel.split("/"):
                continue
            if rel.startswith("logs/") or rel.startswith("runs/"):
                if path.suffix.lower() in {".log", ".txt", ".jsonl"}:
                    profile["logs"].append({"path": rel})
                continue
            suffix = path.suffix.lower()
            if suffix in {".csv", ".tsv"}:
                sample = _sample_csv(path)
                if sample:
                    profile["csv"].append(sample)
                continue
            if suffix in {".json", ".jsonl", ".ndjson"}:
                sample = _sample_json(path)
                if sample:
                    profile["json"].append(sample)
                continue
            if suffix in {".db", ".sqlite", ".sqlite3"}:
                sample = _sample_sqlite(path)
                if sample:
                    profile["sqlite"].append(sample)
                continue
            if suffix == ".txt":
                sample = _sample_txt(path)
                if sample:
                    profile["txt"].append(sample)
                continue
            profile["files"].append({"path": rel})

        ctx.add_step({"type": "inspect_data_sources", "summary": {k: len(v) for k, v in profile.items()}})
        return profile
