from __future__ import annotations

import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from agent_gem.core.utils import dump_json, slugify
from agent_gem.sandbox import SandboxExecutor

from ..base import BaseAgent, TaskContext
from .persist import _filter_records_by_existing_files

logger = logging.getLogger(__name__)


class DataPipelineMixin:
    """Data ingestion, cleaning, and file materialization helpers."""

    # Firecrawl configuration
    FIRECRAWL_TIMEOUT = 30  # Reduced from 60 to 30 seconds to prevent long waits
    FIRECRAWL_RETRY_WAIT_S = 1
    FIRECRAWL_SCRAPE_TIMEOUT = 20  # Separate timeout for scrape operations

    # Search and query limits
    MAX_SEARCH_QUERIES = 10
    MAX_SEARCH_RESULTS_PER_QUERY = 5
    MAX_DATA_FILES = 10
    MAX_LLM_TOKENS_FOR_QUERIES = 300

    # File creation limits
    MAX_FILENAME_LENGTH = 48

    # Error message limits
    MAX_ERROR_MESSAGE_LENGTH = 100
    MAX_ERROR_DETAIL_LENGTH = 300

    def _firecrawl_endpoint_urls(self, endpoint: str) -> list[str]:
        base = os.getenv("FIRECRAWL_API_URL", "http://localhost:3002").strip()
        base = base.rstrip("/")
        endpoint = endpoint.lstrip("/")
        if base.endswith("/v2"):
            return [f"{base}/{endpoint}"]
        return [f"{base}/v2/{endpoint}"]

    def _firecrawl_post_url(
        self, url: str, payload: dict[str, Any], *, timeout: int | None = None
    ) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.FIRECRAWL_TIMEOUT) as response:
                raw = response.read()
                body = json.loads(raw.decode("utf-8")) if raw else {}
                return {"ok": True, "status": response.getcode(), "body": body}
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            body: dict[str, Any] = {}
            if raw:
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    body = {"error": raw.decode("utf-8", "replace")[: self.MAX_ERROR_DETAIL_LENGTH]}
            return {
                "ok": False,
                "status": exc.code,
                "body": body,
                "error": str(exc)[: self.MAX_ERROR_MESSAGE_LENGTH],
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": None,
                "body": {},
                "error": str(exc)[: self.MAX_ERROR_MESSAGE_LENGTH],
            }

    def _firecrawl_get_url(self, url: str, *, timeout: int | None = None) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        api_key = os.getenv("FIRECRAWL_API_KEY", "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        request = urllib.request.Request(url, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(request, timeout=timeout or self.FIRECRAWL_TIMEOUT) as response:
                raw = response.read()
                body = json.loads(raw.decode("utf-8")) if raw else {}
                return {"ok": True, "status": response.getcode(), "body": body}
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            body: dict[str, Any] = {}
            if raw:
                try:
                    body = json.loads(raw.decode("utf-8"))
                except Exception:
                    body = {"error": raw.decode("utf-8", "replace")[: self.MAX_ERROR_DETAIL_LENGTH]}
            return {
                "ok": False,
                "status": exc.code,
                "body": body,
                "error": str(exc)[: self.MAX_ERROR_MESSAGE_LENGTH],
            }
        except Exception as exc:
            return {
                "ok": False,
                "status": None,
                "body": {},
                "error": str(exc)[: self.MAX_ERROR_MESSAGE_LENGTH],
            }

    def _firecrawl_request(
        self, endpoint: str, payload: dict[str, Any], *, timeout: int | None = None
    ) -> dict[str, Any]:
        last_response: dict[str, Any] = {"ok": False, "status": None, "body": {}, "error": "request_failed"}
        for url in self._firecrawl_endpoint_urls(endpoint):
            response = self._firecrawl_post_url(url, payload, timeout=timeout)
            last_response = response
            if response.get("status") != 404:
                return response
        return last_response

    def _firecrawl_get(self, endpoint: str, *, timeout: int | None = None) -> dict[str, Any]:
        last_response: dict[str, Any] = {"ok": False, "status": None, "body": {}, "error": "request_failed"}
        for url in self._firecrawl_endpoint_urls(endpoint):
            response = self._firecrawl_get_url(url, timeout=timeout)
            last_response = response
            if response.get("status") != 404:
                return response
        return last_response

    def _firecrawl_response_ok(self, body: Any) -> bool:
        if isinstance(body, dict):
            if body.get("success") is False:
                return False
            if body.get("error"):
                return False
        return True

    def _firecrawl_error_message(self, body: Any) -> str | None:
        if isinstance(body, dict):
            for key in ("error", "message", "detail"):
                value = body.get(key)
                if value:
                    return str(value)[: self.MAX_ERROR_DETAIL_LENGTH]
        return None

    def _firecrawl_payload(self, body: Any) -> Any:
        if isinstance(body, dict) and "data" in body:
            return body.get("data")
        return body

    def _firecrawl_is_forbidden(self, response: dict[str, Any]) -> bool:
        status = response.get("status")
        if status == 403:
            return True
        body = response.get("body")
        if isinstance(body, dict):
            text = json.dumps(body, ensure_ascii=True).lower()
            return "403" in text or "forbidden" in text
        return False

    def _firecrawl_scrape(self, url: str) -> dict[str, Any]:
        # Base payload for most sites
        # Firecrawl supports: "basic", "stealth", "auto" (default: "auto")
        base_payload: dict[str, Any] = {
            "url": url,
            "formats": ["markdown", "html"],
            "onlyMainContent": True,
            "waitFor": int(self.FIRECRAWL_RETRY_WAIT_S * 1000),
            # "proxy": "stealth",
        }

        # Use shorter timeout for scrape to prevent long waits
        response = self._firecrawl_request("scrape", base_payload, timeout=self.FIRECRAWL_SCRAPE_TIMEOUT)
        
        # If 403 Forbidden, retry with different headers and wait time
        if self._firecrawl_is_forbidden(response):
            retry_payload = {
                "url": url,
                "formats": ["markdown", "html"],
                "onlyMainContent": False,
                "headers": {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
                "waitFor": int(self.FIRECRAWL_RETRY_WAIT_S * 1000),
            }
            response = self._firecrawl_request("scrape", retry_payload, timeout=self.FIRECRAWL_SCRAPE_TIMEOUT)

        if not response.get("ok") or not self._firecrawl_response_ok(response.get("body")):
            error = self._firecrawl_error_message(response.get("body")) or response.get("error")
            return {"ok": False, "status": response.get("status"), "error": error or "scrape_failed"}

        body = response.get("body")
        data = self._firecrawl_payload(body)
        if not isinstance(data, dict):
            data = {}
        return {"ok": True, "status": response.get("status"), "data": data}

    def _firecrawl_extract(
        self,
        urls: list[str],
        *,
        schema: dict[str, Any],
        prompt: str,
    ) -> dict[str, Any]:
        payload = {"urls": urls, "schema": schema, "prompt": prompt}
        response = self._firecrawl_request("extract", payload)

        if not response.get("ok") or not self._firecrawl_response_ok(response.get("body")):
            error = self._firecrawl_error_message(response.get("body")) or response.get("error")
            return {"ok": False, "status": response.get("status"), "error": error or "extract_failed"}

        body = response.get("body")
        extract_id = None
        if isinstance(body, dict):
            extract_id = body.get("id")

        if extract_id:
            status = self._firecrawl_wait_extract(str(extract_id))
            if not status.get("ok"):
                return {
                    "ok": False,
                    "status": status.get("status"),
                    "error": status.get("error") or "extract_failed",
                }
            return {"ok": True, "data": status.get("data")}

        payload_data = self._firecrawl_payload(body)
        extracted: Any = None
        if isinstance(payload_data, list) and payload_data:
            first = payload_data[0]
            if isinstance(first, dict) and "data" in first:
                extracted = first.get("data")
            else:
                extracted = first
        elif isinstance(payload_data, dict):
            if isinstance(payload_data.get("data"), dict):
                extracted = payload_data.get("data")
            else:
                extracted = payload_data
        return {"ok": True, "data": extracted}

    def _firecrawl_wait_extract(self, extract_id: str) -> dict[str, Any]:
        deadline = time.time() + max(self.FIRECRAWL_TIMEOUT * 4, 60)
        last_error: str | None = None
        while time.time() < deadline:
            response = self._firecrawl_get(f"extract/{extract_id}")
            if not response.get("ok") or not self._firecrawl_response_ok(response.get("body")):
                last_error = self._firecrawl_error_message(response.get("body")) or response.get("error")
                time.sleep(1.5)
                continue
            body = response.get("body")
            if isinstance(body, dict):
                status = body.get("status")
                if status == "completed":
                    return {"ok": True, "data": body.get("data"), "status": response.get("status")}
                if status == "failed":
                    return {
                        "ok": False,
                        "status": response.get("status"),
                        "error": body.get("error") or "extract_failed",
                    }
            time.sleep(1.5)
        return {"ok": False, "status": None, "error": last_error or "extract_timeout"}

    def _build_content_schema(self) -> dict[str, Any]:
        properties: dict[str, Any] = {
            "title": {"type": "string"},
            "summary": {"type": "string"},
            "content": {"type": "string"},
            "sections": {
                "type": "array",
                "minItems": 10,
                "items": {
                    "type": "object",
                    "properties": {
                        "heading": {"type": "string"},
                        "text": {"type": "string"},
                        "bullets": {
                            "type": "array",
                            "minItems": 4,
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["heading", "text", "bullets"],
                },
            },
        }
        required = ["summary", "content", "sections"]
        return {"type": "object", "properties": properties, "required": required}

    def _unique_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        for idx in range(2, 1000):
            candidate = path.with_name(f"{stem}_{idx}{suffix}")
            if not candidate.exists():
                return candidate
        return path.with_name(f"{stem}_x{suffix}")

    def _write_json_file(self, data_dir: Path, base: str, suffix: str, payload: Any) -> Path:
        filename = f"{base}_{suffix}.json"
        target = self._unique_path(data_dir / filename)
        target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return target

    def _file_base(self, title: str, url: str) -> str:
        base = slugify(title, max_length=self.MAX_FILENAME_LENGTH)
        if base == "task":
            base = slugify(url, max_length=self.MAX_FILENAME_LENGTH)
        return base or "data"

    def _normalize_url(self, url: str) -> str:
        if not url or not (url := url.strip()):
            return ""

        try:
            parsed = urlparse(url)
        except Exception:
            return url

        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        return f"{scheme}://{netloc}{path}"

    def _max_data_files(self) -> int:
        raw = os.getenv("MAX_DATA_FILES", "").strip()
        if raw:
            try:
                value = int(raw)
                if value > 0:
                    return value
            except ValueError:
                pass
        return self.MAX_DATA_FILES

    def _count_data_files(self, data_dir: Path) -> int:
        count = 0
        for path in data_dir.rglob("*"):
            if path.is_dir():
                continue
            if "raw" in path.parts:
                continue
            if path.suffix.lower() in {".json", ".jsonl", ".ndjson"}:
                count += 1
        return count

    def _generate_search_queries(self, topic: str, *, max_queries: int | None = None) -> list[str]:
        """Generate 10 specific search queries related to topic."""
        if max_queries is None:
            max_queries = self.MAX_SEARCH_QUERIES

        prompt = (
            "Generate 10 highly specific search queries for the given topic.\n"
            "Requirements:\n"
            "- Queries must be diverse (distinct angles).\n"
            "- Emphasize topical relevance.\n"
            "- Avoid near-duplicates; each query should introduce new keywords.\n"
            "Output JSON with key 'queries' containing a list of exactly 10 search query strings. "
            "Return ONLY JSON.\n"
            f"Topic: {topic}\n"
        )
        self.logger.info("LLM call: Generating web search queries for data seeding")
        raw = self.llm.simple_complete(prompt, temperature=0.3, max_tokens=self.MAX_LLM_TOKENS_FOR_QUERIES)
        parsed = self._extract_json(raw)

        queries: list[str] = []
        if isinstance(parsed, dict) and isinstance(parsed.get("queries"), list):
            queries = [str(x).strip() for x in parsed["queries"] if str(x).strip()]
        elif isinstance(parsed, list):
            queries = [str(x).strip() for x in parsed if str(x).strip()]

        seen: set[str] = set()
        deduped: list[str] = []
        for query in queries:
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)
            if len(deduped) >= max_queries:
                break

        return deduped

    def _load_search_cache(self, cache_path: Path) -> tuple[list[str], list[dict[str, Any]]]:
        if not cache_path.exists():
            return [], []
        try:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
        except Exception:
            return [], []
        if not isinstance(data, dict):
            return [], []
        queries: list[str] = []
        hits: list[dict[str, Any]] = []
        for query, items in data.items():
            if not isinstance(query, str) or not isinstance(items, list):
                continue
            queries.append(query)
            for item in items:
                if isinstance(item, dict):
                    hits.append(item)
        return queries, hits

    def _process_firecrawl_url(
        self, url: str, title: str, topic: str, data_dir: Path
    ) -> dict[str, Any]:
        scrape_result = self._firecrawl_scrape(url)
        if not scrape_result.get("ok"):
            error_msg = scrape_result.get("error") or "scrape_failed"
            # Reduce log noise: only log important errors, not routine failures
            if "timeout" in error_msg.lower() or "connection" in error_msg.lower():
                logger.warning("Firecrawl scrape failed for %s: %s", url[:50], error_msg)
            else:
                logger.debug("Firecrawl scrape failed for %s: %s", url[:50], error_msg)
            return {"files": [], "error": error_msg}

        base = self._file_base(title, url)
        saved_files: list[Path] = []
        error_notes: list[str] = []
        notes: list[str] = []

        content_schema = self._build_content_schema()
        content_prompt = (
            "Extract the main factual content from the page into the schema fields. "
            "Include a rich 'content' field with multiple paragraphs of the main article/body "
            "(aim for 4-8 paragraphs or 600-1200 words when available). "
            "Also include a 'sections' array with at least 10 items; each item must include a short heading, "
            "a 2-5 sentence text, and 4-6 bullet facts derived from the page. "
            "Avoid navigation, boilerplate, and ads; keep it specific and concrete."
        )
        content_extract = self._firecrawl_extract([url], schema=content_schema, prompt=content_prompt)
        content_payload = content_extract.get("data") if content_extract.get("ok") else None
        if content_extract.get("ok") and isinstance(content_payload, dict):
            sections = content_payload.get("sections")
            if not isinstance(sections, list) or len(sections) < 10:
                error_notes.append("content_extract_empty: missing_sections")
                content_payload = None
        else:
            error_detail = content_extract.get("error")
            if error_detail:
                error_notes.append(f"content_extract_failed: {error_detail}")
            else:
                error_notes.append("content_extract_failed")
            content_payload = None

        if content_payload:
            saved_files.append(self._write_json_file(data_dir, base, "content", content_payload))
        else:
            if not error_notes:
                error_notes.append("content_extract_failed")

        if not saved_files:
            return {
                "files": [],
                "error": "; ".join(error_notes) if error_notes else "no_files_created",
                "notes": notes,
            }

        return {
            "files": saved_files,
            "error": "; ".join(error_notes) if error_notes else None,
            "notes": notes,
        }

    def _seed_database(
        self, topic: str, ctx: TaskContext, sandbox: SandboxExecutor
    ) -> list[dict[str, Any]]:
        """Seed database via Firecrawl: search, scrape, extract."""
        search_cache_path = getattr(sandbox, "search_cache_path", sandbox.sandbox_dir / "search_cache.json")
        cached_queries, cached_hits = self._load_search_cache(search_cache_path)

        search_hits: list[dict[str, Any]] = []
        search_queries: list[str] = []
        if cached_hits:
            search_queries = cached_queries
            search_hits = cached_hits
            ctx.add_step(
                {
                    "type": "search_cache_used",
                    "topic": topic,
                    "cache_path": str(search_cache_path),
                    "queries": search_queries,
                    "count": len(search_hits),
                }
            )
            logger.info("Using cached search results from %s", search_cache_path)
        else:
            search_queries = self._generate_search_queries(topic)

            queries_log_path = sandbox.sandbox_dir / "logs" / "search_queries.jsonl"
            queries_log_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(queries_log_path, "a", encoding="utf-8") as f:
                    for query in search_queries:
                        log_entry = {
                            "topic": topic,
                            "query": query,
                        }
                        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.debug("Failed to log search queries: %s", e)

            ctx.add_step(
                {
                    "type": "search_queries_generated",
                    "topic": topic,
                    "queries": search_queries,
                    "count": len(search_queries),
                }
            )

            for query in search_queries:
                result = sandbox.execute_search(query, max_results=self.MAX_SEARCH_RESULTS_PER_QUERY)
                if isinstance(result, list):
                    for row in result:
                        if isinstance(row, dict):
                            search_hits.append(row)

        seen_urls: set[str] = set()
        unique_hits: list[dict[str, Any]] = []
        for hit in search_hits:
            url = str(hit.get("url") or hit.get("link") or "").strip()
            normalized = self._normalize_url(url)
            if not normalized or normalized in seen_urls:
                continue
            seen_urls.add(normalized)
            unique_hits.append({**hit, "url": url})

        logger.info("Initial search: %s unique URLs", len(unique_hits))

        data_dir = sandbox.sandbox_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)
        records: list[dict[str, Any]] = []
        mapping_path = sandbox.sandbox_dir / "logs" / "url_file_mapping.json"
        url_file_mapping: dict[str, dict[str, Any]] = {}
        max_files = self._max_data_files()
        data_file_count = self._count_data_files(data_dir)
        if data_file_count >= max_files:
            logger.info(
                "Data directory already has %s JSON files (limit %s); skipping URL extraction.",
                data_file_count,
                max_files,
            )

        if search_cache_path.exists() and mapping_path.exists():
            try:
                existing_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            except Exception:
                existing_mapping = {}
            if isinstance(existing_mapping, dict):
                url_file_mapping = existing_mapping
                succeeded: set[str] = set()
                for url, info in existing_mapping.items():
                    if not isinstance(info, dict):
                        continue
                    if info.get("success") and info.get("files"):
                        normalized = self._normalize_url(str(url))
                        if normalized:
                            succeeded.add(normalized)
                if succeeded:
                    original_count = len(unique_hits)
                    unique_hits = [
                        hit
                        for hit in unique_hits
                        if self._normalize_url(str(hit.get("url") or hit.get("link") or "")) not in succeeded
                    ]
                    ctx.add_step(
                        {
                            "type": "url_mapping_resume",
                            "topic": topic,
                            "skipped": original_count - len(unique_hits),
                            "remaining": len(unique_hits),
                            "mapping_path": str(mapping_path),
                        }
                    )
                    logger.info(
                        "Resuming from url_file_mapping; skipped %s already-success URLs",
                        original_count - len(unique_hits),
                    )

        total_urls = len(unique_hits)
        logger.info("Processing %s URLs...", total_urls)
        processed_count = 0
        success_count = 0
        error_count = 0

        for hit in unique_hits:
            if data_file_count >= max_files:
                logger.info("Data file limit reached (%s); stopping URL extraction.", max_files)
                break
            processed_count += 1
            title = str(hit.get("title") or hit.get("name") or topic).strip()
            url = str(hit.get("url") or hit.get("link") or "").strip()

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
                "source": "firecrawl",
            }

            result = self._process_firecrawl_url(url, title, topic, data_dir)
            saved_paths = result.get("files") or []
            error_reason = result.get("error")
            notes = result.get("notes") or []
            saved_files: list[str] = []

            for path in saved_paths:
                try:
                    saved_files.append(str(Path(path).relative_to(sandbox.sandbox_dir)))
                except Exception:
                    saved_files.append(str(path))

            if saved_files:
                record["downloaded_files"] = saved_files
                record["source"] = "firecrawl_extract"
                success_count += 1
                data_file_count += len(saved_files)
            else:
                error_count += 1

            if error_reason:
                record["error"] = error_reason
            if notes:
                record["notes"] = notes

            url_file_mapping[url] = {
                "files": saved_files,
                "error": error_reason,
                "success": bool(saved_files),
                "notes": notes,
            }
            records.append(record)
            try:
                dump_json(mapping_path, url_file_mapping)
            except Exception as e:
                logger.debug("Failed to update URL mapping after %s: %s", url, e)

        if total_urls > 0:
            sys.stdout.write("\n")
            sys.stdout.flush()

        try:
            dump_json(mapping_path, url_file_mapping)
            mapping_rel_path = str(mapping_path.relative_to(sandbox.sandbox_dir))
            logger.info(
                "Processing complete: %s succeeded, %s failed out of %s URLs",
                success_count,
                error_count,
                processed_count,
            )
            logger.info("URL to file mapping saved to: %s", mapping_rel_path)
        except Exception as e:
            logger.warning("Failed to save URL to file mapping: %s", e)
            logger.info(
                "Processing complete: %s succeeded, %s failed out of %s URLs",
                success_count,
                error_count,
                processed_count,
            )

        merged_records = self.writer.merge_records(records)
        if not merged_records:
            synthetic_records: list[dict[str, Any]] = []
            for path in sorted(data_dir.rglob("*.json")):
                if path.is_dir():
                    continue
                rel_path = str(path.relative_to(sandbox.sandbox_dir))
                if "raw" in Path(rel_path).parts:
                    continue
                title = ""
                summary = ""
                url = ""
                try:
                    payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
                    if isinstance(payload, dict):
                        title = str(payload.get("title") or payload.get("heading") or "").strip()
                        summary = str(payload.get("summary") or payload.get("description") or "").strip()
                        url = str(payload.get("url") or payload.get("source") or "").strip()
                except Exception:
                    pass
                if not title:
                    title = path.stem.replace("_", " ").strip()
                synthetic_records.append(
                    {
                        "title": title,
                        "summary": summary,
                        "url": url,
                        "source": "local_json",
                        "downloaded_files": [rel_path],
                    }
                )
            if synthetic_records:
                merged_records = self.writer.merge_records(synthetic_records)
        
        # Filter records to only include those with existing files
        task_dir = self.writer.task_dir(ctx.task_id, self.agent_type)
        sandbox_dir = task_dir / "_sandbox"
        filtered_records = _filter_records_by_existing_files(merged_records, sandbox_dir)
        self.writer.records = filtered_records

        task_db_path = task_dir / "db.json"
        task_db_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {"records": filtered_records, "search_queries": search_queries}
        dump_json(task_db_path, payload)

        files_with_errors = [r for r in records if r.get("error")]
        files_success = [r for r in records if r.get("downloaded_files")]

        logger.info("=" * 70)
        logger.info("Database Seeding Summary")
        logger.info("=" * 70)
        logger.info("Topic: %s", topic)
        logger.info("Total queries generated: %s", len(search_queries))
        logger.info("Total unique URLs: %s", len(unique_hits))
        logger.info("URLs processed: %s", processed_count)
        logger.info("URLs with files saved: %s", len(files_success))
        logger.info("URLs with errors: %s", len(files_with_errors))
        logger.info("URL to file mapping saved to: logs/url_file_mapping.json")
        logger.info("=" * 70)

        ctx.add_step(
            {
                "type": "seed_database",
                "topic": topic,
                "queries": search_queries,
                "records_count": len(records),
                "data_files_count": len(files_success),
                "error_count": len(files_with_errors),
            }
        )
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
        if not records:
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
        """Enumerate local JSON artifacts with lightweight schema samples."""

        base = sandbox.sandbox_dir
        profile: dict[str, Any] = {"json": []}

        def _safe_rel(path: Path) -> str:
            try:
                return str(path.relative_to(base))
            except Exception:
                return str(path)

        def _sample_json(path: Path) -> dict[str, Any] | None:
            """Extract basic metadata from JSON file without reading full content."""
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                return None
            keys_seen: set[str] = set()
            # Extract keys from the structure
            if isinstance(payload, list) and payload:
                # If it's a list, check first item
                first_item = payload[0]
                if isinstance(first_item, dict):
                    keys_seen.update(str(k) for k in first_item.keys())
            elif isinstance(payload, dict):
                # If it's a dict, check for common list keys
                for key in ("records", "items", "data", "rows", "sections", "tables"):
                    value = payload.get(key)
                    if isinstance(value, list) and value:
                        first_item = value[0]
                        if isinstance(first_item, dict):
                            keys_seen.update(str(k) for k in first_item.keys())
                        break
                # If no list found, check top-level keys
                if not keys_seen:
                    keys_seen.update(str(k) for k in payload.keys())
            return {
                "path": _safe_rel(path),
                "type": "list" if isinstance(payload, list) else "object",
                "keys": sorted(keys_seen),
            }

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
            suffix = path.suffix.lower()
            if suffix in {".json", ".jsonl", ".ndjson"}:
                sample = _sample_json(path)
                if sample:
                    profile["json"].append(sample)
                continue
            continue

        ctx.add_step({"type": "inspect_data_sources", "summary": {k: len(v) for k, v in profile.items()}})
        return profile
