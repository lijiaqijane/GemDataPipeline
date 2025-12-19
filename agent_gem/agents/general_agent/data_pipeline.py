from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import uuid
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Any, TYPE_CHECKING

import requests
from bs4 import BeautifulSoup

from agent_gem.core.utils import dump_json, slugify
from agent_gem.sandbox import SandboxExecutor

from ..base import BaseAgent, TaskContext

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest  # noqa: F401

logger = logging.getLogger(__name__)


class DataPipelineMixin:
    """Data ingestion, cleaning, and file materialization helpers."""

    _RECORDS_FILENAME = "records.json"
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

    def _fetch_url_content(self, url: str, timeout: int = 10) -> dict[str, Any]:
        """Fetch and extract content from a URL, handling errors gracefully."""
        result = {
            "url": url,
            "success": False,
            "content": "",
            "tables": [],
            "links": [],
            "error": None,
        }
        
        if not url or not url.startswith(("http://", "https://")):
            result["error"] = "Invalid URL"
            return result
        
        try:
            # Enhanced headers to avoid 403 Forbidden
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "DNT": "1",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
            }
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove script, style, and other non-content tags
            for tag in soup(["script", "style", "nav", "header", "footer", "aside"]):
                tag.decompose()
            
            # Extract text content
            text = soup.get_text(separator='\n', strip=True)
            # Limit content length
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            content = '\n'.join(lines[:500])  # Limit to 500 lines
            
            # Extract tables as structured data
            tables = []
            for table in soup.find_all('table')[:5]:  # Limit to 5 tables
                table_data = []
                for row in table.find_all('tr')[:20]:  # Limit to 20 rows per table
                    cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                    if cells:
                        table_data.append(cells)
                if table_data:
                    tables.append(table_data)
            
            result["success"] = True
            result["content"] = content[:10000]  # Limit to 10k chars
            result["tables"] = tables
            result["links"] = self._extract_dataset_links(soup, base_url=url)
            
        except requests.Timeout:
            result["error"] = "Timeout"
            logger.warning(f"Timeout fetching URL: {url}")
        except requests.RequestException as e:
            result["error"] = f"Request error: {str(e)[:100]}"
            logger.warning(f"Error fetching URL {url}: {e}")
        except Exception as e:
            result["error"] = f"Parse error: {str(e)[:100]}"
            logger.warning(f"Error parsing URL {url}: {e}")
        
        return result

    def _is_data_file_url(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in self._DATA_FILE_EXTENSIONS)

    def _extract_dataset_links(self, soup: BeautifulSoup, *, base_url: str) -> list[str]:
        links: list[str] = []
        for tag in soup.find_all("a", href=True):
            href = (tag.get("href") or "").strip()
            if not href:
                continue
            full = urljoin(base_url, href)
            if self._is_data_file_url(full):
                links.append(full)
        # Deduplicate while preserving order
        seen: set[str] = set()
        deduped: list[str] = []
        for link in links:
            if link not in seen:
                seen.add(link)
                deduped.append(link)
        return deduped

    def _download_dataset_file(
        self,
        url: str,
        dest_dir: Path,
        *,
        max_bytes: int | None = None,
        timeout: int = 15,
    ) -> Path | None:
        if max_bytes is None:
            env_value = os.getenv("DATASET_MAX_BYTES", "").strip()
            try:
                max_bytes = int(env_value) if env_value else 6_000_000
            except ValueError:
                max_bytes = 6_000_000
        if not self._is_data_file_url(url):
            return None
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(urlparse(url).path).suffix.lower()
        filename_base = slugify(urlparse(url).path) or "dataset"
        filename = f"{filename_base[:48]}{suffix}"
        target = dest_dir / filename
        if target.exists():
            return target
        try:
            resp = requests.get(url, stream=True, timeout=timeout)
            resp.raise_for_status()
            size = 0
            with open(target, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    if not chunk:
                        continue
                    size += len(chunk)
                    if size > max_bytes:
                        target.unlink(missing_ok=True)
                        logger.warning("Dataset file too large, skipping: %s", url)
                        return None
                    f.write(chunk)
            return target
        except Exception as exc:
            logger.warning("Failed to download dataset file %s: %s", url, exc)
            return None

    def _sample_dataset_file(self, path: Path, *, max_rows: int = 50) -> list[dict[str, Any]]:
        import csv
        import sqlite3

        suffix = path.suffix.lower()
        samples: list[dict[str, Any]] = []

        def _normalize_headers(headers: list[str]) -> list[str]:
            normalized: list[str] = []
            for idx, name in enumerate(headers):
                clean = re.sub(r"[^\w]+", "_", (name or "").strip().lower()).strip("_")
                normalized.append(clean or f"col_{idx+1}")
            return normalized

        if suffix in {".csv", ".tsv"}:
            with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                sample_text = f.read(4096)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample_text) if sample_text else csv.excel
                except csv.Error:
                    dialect = csv.excel
                if suffix == ".tsv":
                    dialect.delimiter = "\t"
                reader = csv.reader(f, dialect)
                rows: list[list[str]] = []
                for row in reader:
                    if not row or all(not str(cell).strip() for cell in row):
                        continue
                    rows.append([str(cell) for cell in row])
                    if len(rows) >= max_rows + 1:
                        break
                if not rows:
                    return []
                header = _normalize_headers(rows[0])
                data_rows = rows[1:] if any(h for h in header) else rows[:max_rows]
                if data_rows and len(data_rows) > 0:
                    for row in data_rows[:max_rows]:
                        padded = (row + [""] * len(header))[: len(header)]
                        samples.append({header[i]: padded[i] for i in range(len(header))})
                return samples

        if suffix in {".json"}:
            try:
                payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
            except Exception:
                return []
            items: list[Any] = []
            if isinstance(payload, list):
                items = payload
            elif isinstance(payload, dict):
                for value in payload.values():
                    if isinstance(value, list):
                        items = value
                        break
            for item in items[:max_rows]:
                if isinstance(item, dict):
                    samples.append({k: item.get(k, "") for k in item.keys()})
            return samples

        if suffix in {".jsonl", ".ndjson"}:
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if len(samples) >= max_rows:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            item = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(item, dict):
                            samples.append({k: item.get(k, "") for k in item.keys()})
            except Exception:
                return []
            return samples

        if suffix == ".txt":
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    for idx, line in enumerate(f, 1):
                        if len(samples) >= max_rows:
                            break
                        line = line.strip()
                        if not line:
                            continue
                        samples.append({"line_number": idx, "text": line})
            except Exception:
                return []
            return samples

        if suffix in {".db", ".sqlite", ".sqlite3"}:
            try:
                conn = sqlite3.connect(path)
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                )
                tables = [row[0] for row in cursor.fetchall() if row and row[0]]
                if not tables:
                    return []
                table = tables[0]
                cursor.execute(f"PRAGMA table_info({table})")
                cols = [row[1] for row in cursor.fetchall() if row and row[1]]
                if not cols:
                    return []
                cursor.execute(f"SELECT * FROM {table} LIMIT {max_rows}")
                rows = cursor.fetchall()
                for row in rows:
                    samples.append({cols[i]: row[i] for i in range(len(cols))})
                conn.close()
            except Exception:
                return []
            return samples

        if suffix == ".parquet":
            try:
                import pyarrow.parquet as pq
            except Exception:
                logger.warning("Parquet support unavailable (pyarrow not installed).")
                return []
            try:
                table = pq.read_table(path)
                data = table.to_pydict()
                if not data:
                    return []
                keys = list(data.keys())
                for idx in range(min(max_rows, table.num_rows)):
                    row = {k: (data[k][idx] if idx < len(data[k]) else None) for k in keys}
                    samples.append(row)
            except Exception:
                return []
            return samples

        return []

    def _schema_from_samples(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        def _infer_type(value: Any) -> str:
            if isinstance(value, bool):
                return "boolean"
            if isinstance(value, int):
                return "integer"
            if isinstance(value, float):
                return "number"
            return "string"

        fields: dict[str, Any] = {}
        for row in samples:
            if not isinstance(row, dict):
                continue
            for key, value in row.items():
                if key not in fields:
                    fields[key] = {"type": _infer_type(value), "source": "sampled_data"}
        return {"fields": fields, "source": "sampled_data"} if fields else {}

    def _looks_synthetic_samples(self, samples: list[dict[str, Any]]) -> bool:
        if not samples:
            return False
        placeholder = re.compile(r"^(value|sample)_\d+$", re.IGNORECASE)
        total = 0
        hits = 0
        for row in samples[:10]:
            if not isinstance(row, dict):
                continue
            for value in row.values():
                if not isinstance(value, str):
                    continue
                total += 1
                if placeholder.match(value.strip()):
                    hits += 1
        return total > 0 and (hits / total) >= 0.5

    def _clean_content_with_jina(self, text: str, *, topic: str, max_chars: int = 2400) -> str:
        """Filter noisy text via Jina rerank; fallback to safe truncation when disabled or failing."""
        if not text:
            return ""
        api_key = os.getenv("JINA_API_KEY", "").strip()

        # Pre-split into smaller chunks to avoid oversized requests
        chunks: list[str] = []
        for para in re.split(r"\n{2,}", text):
            para = para.strip()
            if len(para) < 32:
                continue
            for seg in textwrap.wrap(para, width=360):
                seg = seg.strip()
                if len(seg) >= 24:
                    chunks.append(seg)
        if not chunks:
            logger.warning("Jina cleaning skipped: no text chunks; using truncation fallback.")
            return text[:max_chars]

        if not api_key:
            logger.warning("Jina cleaning disabled: missing JINA_API_KEY; using truncation fallback.")
            combined = " ".join(chunks[:6])
            return combined[:max_chars]

        top_n = min(12, len(chunks))
        query = (
            f"{topic} dataset schema columns fields csv json table latitude longitude price rating time id name address"
        )
        timeout_s = 15
        env_timeout = os.getenv("JINA_TIMEOUT", "").strip()
        if env_timeout:
            try:
                timeout_s = max(1, int(env_timeout))
            except ValueError:
                timeout_s = 15
        try:
            resp = requests.post(
                "https://api.jina.ai/v1/rerank",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "jina-rerank-v2-base-multilingual",
                    "query": query,
                    "documents": chunks,
                    "top_n": top_n,
                },
                timeout=timeout_s,
            )
            if resp.ok:
                data = resp.json().get("data", [])
                ranked = sorted(
                    data,
                    key=lambda x: x.get("relevance_score", 0),
                    reverse=True,
                )
                selected = [chunks[item.get("index", 0)] for item in ranked[:top_n] if isinstance(item, dict)]
                combined = " ".join(selected)
                return combined[:max_chars] if combined else text[:max_chars]
            else:
                logger.warning(f"Jina cleaning failed: status={resp.status_code} body={resp.text[:200]}")
        except Exception:
            logger.warning("Jina cleaning exception; using truncation fallback.", exc_info=True)

        return text[:max_chars]

    def _extract_table_samples(
        self, tables: list[Any], *, max_rows: int = 50
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Convert parsed HTML tables into structured samples + lightweight schema."""
        samples: list[dict[str, Any]] = []
        schema_fields: dict[str, Any] = {}

        def _normalize_header(cell: Any, idx: int) -> str:
            text = str(cell or "").strip().lower()
            text = re.sub(r"[^\w]+", "_", text)
            text = text.strip("_") or f"col_{idx+1}"
            return text[:48]

        for table in tables:
            if not isinstance(table, list) or not table:
                continue
            header_row = table[0] if isinstance(table[0], list) else []
            has_header = any(str(cell).strip() for cell in header_row) if isinstance(header_row, list) else False
            headers = [_normalize_header(c, idx) for idx, c in enumerate(header_row)] if has_header else []
            data_rows = table[1:] if has_header else table
            if not headers and data_rows and isinstance(data_rows[0], list):
                headers = [f"col_{i+1}" for i in range(len(data_rows[0]))]
            if not headers:
                continue

            for h in headers:
                schema_fields.setdefault(h, {"type": "string", "source": "html_table"})

            for row in data_rows:
                if not isinstance(row, list):
                    continue
                normalized = (row + [""] * len(headers))[: len(headers)]
                samples.append({headers[i]: normalized[i] for i in range(len(headers))})
                if len(samples) >= max_rows:
                    break
            if len(samples) >= max_rows:
                break

        schema = {"fields": schema_fields, "source": "html_table"} if schema_fields else {}
        return samples, schema

    def _merge_record_fields(self, base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
        """Merge two records, preferring already-present rich fields."""
        merged = dict(base)
        for key in [
            "real_data_samples",
            "real_tables",
            "data_schema",
            "clean_content",
            "real_content",
            "fetch_error",
        ]:
            if not merged.get(key) and incoming.get(key):
                merged[key] = incoming[key]
        # Prefer longer/more descriptive summary
        if len(str(incoming.get("summary", ""))) > len(str(merged.get("summary", ""))):
            merged["summary"] = incoming["summary"]
        # Keep source that actually contained data
        if incoming.get("source", "").startswith("real") and not str(merged.get("source", "")).startswith("real"):
            merged["source"] = incoming["source"]
        return merged

    def _record_quality_score(self, record: dict[str, Any]) -> int:
        score = 0
        if record.get("real_data_samples"):
            score += 6
        if record.get("real_tables"):
            score += 3
        if record.get("data_schema"):
            score += 2
        if record.get("clean_content"):
            score += 1
        if record.get("real_content"):
            score += 1
        return score

    def _merge_records(
        self, preferred: list[dict[str, Any]], fallback: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Merge two record lists keyed by URL/title, keeping richer entries."""
        merged: dict[str, dict[str, Any]] = {}
        for rec in preferred + fallback:
            key = (rec.get("url") or rec.get("title") or "").lower()
            if not key:
                key = uuid.uuid4().hex
            existing = merged.get(key)
            if existing:
                merged[key] = self._merge_record_fields(existing, rec)
                if self._record_quality_score(rec) > self._record_quality_score(existing):
                    merged[key] = self._merge_record_fields(rec, merged[key])
            else:
                merged[key] = rec
        return list(merged.values())

    def _dedupe_records_by_title(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Deduplicate records by title while keeping the richest version."""
        deduped: dict[str, dict[str, Any]] = {}
        for row in records:
            key = (row.get("title") or "").lower()
            if not key:
                continue
            existing = deduped.get(key)
            if existing:
                combined = self._merge_record_fields(existing, row)
                if self._record_quality_score(row) > self._record_quality_score(existing):
                    combined = self._merge_record_fields(row, combined)
                deduped[key] = combined
            else:
                deduped[key] = row
        return list(deduped.values())

    def _seed_database(
        self, topic: str, ctx: TaskContext, sandbox: SandboxExecutor
    ) -> list[dict[str, Any]]:
        """Collect topic-relevant records with REAL DATA from URLs and write them into the sandbox database."""
        search_queries = [
            f"{topic} dataset csv filetype:csv",
            f"{topic} dataset json filetype:json",
            f"{topic} open data portal dataset",
        ]
        search_hits: list[dict[str, str]] = []
        for query in search_queries:
            result = sandbox.execute_search(query, max_results=6)
            if isinstance(result, list):
                for row in result:
                    if isinstance(row, dict):
                        search_hits.append(row)

        # Phase 1: Fetch REAL content from URLs
        search_records: list[dict[str, Any]] = []
        url_fetch_count = 0
        url_success_count = 0
        seen_urls: set[str] = set()
        raw_data_dir = sandbox.sandbox_dir / "data" / "raw"
        
        for hit in search_hits:
            title = str(hit.get("title") or hit.get("name") or topic).strip()
            summary = str(
                hit.get("summary")
                or hit.get("snippet")
                or hit.get("description")
                or ""
            ).strip()
            url = str(hit.get("url") or hit.get("link") or "").strip()
            
            if not title or not url or url in seen_urls:
                continue
            seen_urls.add(url)
            if len(seen_urls) > 15:  # Limit to 15 unique URLs to avoid long delays
                break
            
            url_fetch_count += 1

            record = {
                "title": title,
                "summary": summary or title,
                "url": url,
                "source": "search",
            }

            if self._is_data_file_url(url):
                local_path = self._download_dataset_file(url, raw_data_dir)
                if local_path:
                    samples = self._sample_dataset_file(local_path)
                    if samples:
                        record["real_data_samples"] = samples
                        record["data_schema"] = self._schema_from_samples(samples)
                        record["downloaded_files"] = [str(local_path)]
                        record["source"] = "downloaded_dataset"
                        search_records.append(record)
                        continue

            # Fetch real content from URL
            url_data = self._fetch_url_content(url, timeout=8)
            record["source"] = "search_with_real_content" if url_data["success"] else "search"

            if url_data["success"]:
                url_success_count += 1
                real_content = url_data["content"][:5000]
                record["real_content"] = real_content  # Store raw content
                record["clean_content"] = self._clean_content_with_jina(real_content, topic=topic)
                record["real_tables"] = url_data["tables"]  # Store real tables
                if url_data["tables"]:
                    table_samples, table_schema = self._extract_table_samples(url_data["tables"])
                    if table_samples:
                        record["real_data_samples"] = table_samples
                    if table_schema and not record.get("data_schema"):
                        record["data_schema"] = table_schema
                for link in url_data.get("links", [])[:3]:
                    local_path = self._download_dataset_file(link, raw_data_dir)
                    if not local_path:
                        continue
                    samples = self._sample_dataset_file(local_path)
                    if not samples:
                        continue
                    record.setdefault("downloaded_files", []).append(str(local_path))
                    record["real_data_samples"] = samples
                    record["data_schema"] = self._schema_from_samples(samples)
                    record["source"] = "downloaded_dataset"
                    break
            else:
                record["fetch_error"] = url_data["error"]
            
            search_records.append(record)

        ctx.add_step(
            {
                "type": "seed_database_with_real_data",
                "topic": topic,
                "search_queries": search_queries,
                "urls_attempted": url_fetch_count,
                "urls_successful": url_success_count,
                "urls_with_tables": len([r for r in search_records if r.get("real_tables")]),
                "search_hits_preview": search_hits[:5],
            }
        )

        # Phase 2: Extract schemas from REAL content (not just summaries)
        records_with_real_content = [
            r for r in search_records if r.get("real_content") and not r.get("real_data_samples")
        ]
        records_with_only_summary = [r for r in search_records if not r.get("real_content")]
        records_with_tables = [r for r in search_records if r.get("real_data_samples")]
        
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        enriched_records: list[dict[str, Any]] = []
        
        # Strategy A: Extract from real content (preferred)
        if records_with_real_content:
            schema_prompt = (
                "You are a data extraction expert. Analyze the REAL webpage content below and extract actual data structures.\n"
                "For each source, identify:\n"
                "1. ACTUAL data fields found in the content (column names, keys, attribute names)\n"
                "2. ACTUAL sample values extracted from the content\n"
                "3. Data patterns and formats observed\n"
                "4. Any anomalies, missing values, or inconsistencies found\n"
                "5. Estimated data volume\n"
                "Return ONLY a JSON array where each item has: title, summary, url, data_schema (object with fields, samples, patterns, anomalies, volume), real_data_samples (list of dicts).\n"
                f"Topic: {topic}\n\n"
                "Real content from URLs:\n"
                + "\n\n---\n\n".join([
                    f"Title: {r['title']}\nURL: {r['url']}\nContent: {(r.get('clean_content') or r['real_content'])[:1200]}\nTables: {json.dumps(r.get('real_tables', [])[:2], ensure_ascii=False)}"
                    for r in records_with_real_content[:5]
                ])
            )
            raw_schema = self.llm.simple_complete(schema_prompt, temperature=0.2, max_tokens=max_tokens)
            extracted_schema = self._extract_json(raw_schema)

            if isinstance(extracted_schema, list):
                for item in extracted_schema:
                    if isinstance(item, dict) and item.get("title"):
                        enriched_records.append({
                            "title": str(item.get("title", "")).strip(),
                            "summary": str(item.get("summary", "")).strip(),
                            "url": str(item.get("url", "")).strip(),
                            "source": "real_content_extracted",
                            "data_schema": item.get("data_schema", {}),
                            "real_data_samples": item.get("real_data_samples", []),
                        })
        
        # Strategy B: Infer from summaries for URLs that failed (fallback)
        if records_with_only_summary:
            logger.info(f"Using fallback: inferring schemas from {len(records_with_only_summary)} summaries (URLs blocked)")
            fallback_prompt = (
                "You are a data schema analyst. These URLs were blocked (403), but we have their summaries.\n"
                "Infer likely data structures from the descriptions.\n"
                "For each source, infer:\n"
                "1. Likely data fields (based on topic and description)\n"
                "2. Plausible sample values\n"
                "3. Common patterns in this domain\n"
                "4. Typical anomalies\n"
                "Return ONLY a JSON array where each item has: title, summary, url, data_schema (object with fields, samples, patterns, anomalies).\n"
                f"Topic: {topic}\n\n"
                "Summaries from blocked URLs:\n"
                + "\n\n---\n\n".join([
                    f"Title: {r['title']}\nURL: {r['url']}\nSummary: {r['summary']}\nError: {r.get('fetch_error', 'Unknown')}"
                    for r in records_with_only_summary[:5]
                ])
            )
            raw_fallback = self.llm.simple_complete(fallback_prompt, temperature=0.3, max_tokens=max_tokens)
            extracted_fallback = self._extract_json(raw_fallback)

            if isinstance(extracted_fallback, list):
                for item in extracted_fallback:
                    if isinstance(item, dict) and item.get("title"):
                        enriched_records.append({
                            "title": str(item.get("title", "")).strip(),
                            "summary": str(item.get("summary", "")).strip(),
                            "url": str(item.get("url", "")).strip(),
                            "source": "summary_inferred",
                            "data_schema": item.get("data_schema", {}),
                            "real_data_samples": [],  # No real samples from blocked URLs
                        })
        
        primary_records = enriched_records if enriched_records else records_with_real_content or []
        primary_records = primary_records or records_with_tables
        primary_records = primary_records or search_records
        records = self._merge_records(primary_records, search_records)
        records = self._dedupe_records_by_title(records) or [{
            "title": topic, 
            "summary": "Overview of the topic.",
            "source": "fallback",
            "url": "",
            "data_schema": {},
            "real_data_samples": []
        }]

        # Merge new records with existing records in db.json (for multiple runs)
        merged_records = self.writer.merge_records(records)
        self.writer.records = merged_records
        
        # Save merged records to db.json (preserve existing search_hits if any)
        existing_data = {}
        if self.writer.path.exists():
            try:
                existing_data = json.loads(self.writer.path.read_text())
            except Exception:
                pass
        existing_search_hits = existing_data.get("search_hits", [])
        # Merge search_hits (deduplicate by URL)
        existing_urls = {h.get("url", "") for h in existing_search_hits if h.get("url")}
        for hit in search_hits:
            if hit.get("url") and hit.get("url") not in existing_urls:
                existing_search_hits.append(hit)
                existing_urls.add(hit.get("url"))
        
        payload = {"records": merged_records, "search_hits": existing_search_hits}
        
        # Write to global db.json (for multi-task tracking)
        dump_json(self.writer.path, payload)
        
        # Write to sandbox-local records.json (for tool access within this task)
        sandbox_records = sandbox.sandbox_dir / self._RECORDS_FILENAME
        dump_json(sandbox_records, payload)

        ctx.add_step(
            {
                "type": "seed_database_with_real_data",
                "topic": topic,
                "records_count": len(records),
                "records_with_real_samples": len([r for r in records if r.get("real_data_samples")]),
                "records_with_tables": len([r for r in records if r.get("real_tables")]),
                "records_with_schema": len([r for r in records if r.get("data_schema")]),
            }
        )
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)

        return records

    def _infer_content_aware_filename(
        self, record: dict[str, Any], ctx: TaskContext
    ) -> str:
        """Use LLM to infer a descriptive filename based on data content."""
        title = record.get("title", "dataset")
        summary = record.get("summary", "")[:300]
        real_samples = record.get("real_data_samples", [])
        schema = record.get("data_schema", {})
        
        # Extract field names from data
        field_names = []
        if real_samples and isinstance(real_samples, list) and len(real_samples) > 0:
            if isinstance(real_samples[0], dict):
                field_names = list(real_samples[0].keys())[:10]
        # schema may be dict or list; support both
        elif schema:
            schema_obj = schema
            if isinstance(schema, list) and schema:
                schema_obj = schema[0]
            if isinstance(schema_obj, dict):
                fields_obj = schema_obj.get("fields")
                if isinstance(fields_obj, dict):
                    field_names = list(fields_obj.keys())[:10]
                elif isinstance(fields_obj, list):
                    # list can be field names or dicts with "name"
                    collected = []
                    for item in fields_obj:
                        if isinstance(item, str):
                            collected.append(item)
                        elif isinstance(item, dict) and "name" in item:
                            collected.append(str(item["name"]))
                        if len(collected) >= 10:
                            break
                    field_names = collected[:10]
        
        # Build a concise prompt for filename inference
        prompt = (
            "You are a data analyst. Infer a short, descriptive filename (2-3 words) for a dataset.\n"
            "Rules:\n"
            "- Use snake_case (e.g., paris_hotels, weather_data, stock_prices)\n"
            "- Be specific and descriptive\n"
            "- Reflect the data content, not just the source\n"
            "- Maximum 40 characters\n"
            "- Return ONLY the filename without extension\n\n"
            f"Dataset title: {title}\n"
            f"Summary: {summary}\n"
            f"Data fields: {', '.join(field_names) if field_names else 'unknown'}\n\n"
            "Filename (snake_case, no extension):"
        )
        
        try:
            response = self.llm.simple_complete(prompt, temperature=0.2, max_tokens=50)
            filename = response.strip().lower()
            # Clean up the response
            filename = re.sub(r'[^\w\s-]', '', filename)
            filename = re.sub(r'\s+', '_', filename)
            filename = filename[:40]
            
            # Validate it's reasonable
            if len(filename) >= 3 and not filename.startswith('_'):
                return filename
        except Exception as e:
            logger.warning(f"Failed to infer filename with LLM: {e}")
        
        # Fallback to simple title-based name
        safe_name = re.sub(r'[^\w\s-]', '', title)[:50].strip().replace(' ', '_').lower()
        return safe_name or "dataset"

    def _create_data_files_from_records(
        self, records: list[dict[str, Any]], sandbox: SandboxExecutor, ctx: TaskContext
    ) -> None:
        """Create actual data files directly from real data (bypassing LLM generation to avoid long prompts)."""
        import csv
        import os
        import sqlite3
        from datetime import datetime
        from itertools import islice
        
        created_files = []
        data_dir = sandbox.sandbox_dir / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        def _field_names_from_schema(schema_obj: Any) -> list[str]:
            if not schema_obj:
                return []
            if isinstance(schema_obj, dict):
                fields = schema_obj.get("fields")
                if isinstance(fields, dict):
                    return [k for k in fields.keys() if k]
                if isinstance(fields, list):
                    names = []
                    for item in fields:
                        if isinstance(item, str):
                            names.append(item)
                        elif isinstance(item, dict) and item.get("name"):
                            names.append(str(item["name"]))
                    return names
            if isinstance(schema_obj, list):
                names = []
                for item in schema_obj:
                    if isinstance(item, dict):
                        names.extend(_field_names_from_schema(item))
                    elif isinstance(item, str):
                        names.append(item)
                return names
            return []

        def _is_structured(samples: list[Any]) -> bool:
            if not samples:
                return False
            first = samples[0]
            if isinstance(first, dict):
                keys = [k for k in first.keys() if k and k.lower() not in {"note", "notes", "comment"}]
                return len(keys) >= 2
            if isinstance(first, list):
                return len(first) >= 2
            return False
        
        max_files = 6
        env_max_files = os.getenv("MAX_DATA_FILES", "").strip()
        if env_max_files.isdigit():
            max_files = int(env_max_files)
        max_sample_rows = 1000
        env_max_rows = os.getenv("MAX_SAMPLE_ROWS", "").strip()
        if env_max_rows.isdigit():
            max_sample_rows = int(env_max_rows)

        remaining_slots = max_files

        def _record_rank(rec: dict[str, Any]) -> tuple[int, int]:
            samples = rec.get("real_data_samples") or []
            return (1 if samples else 0, len(samples))

        ranked_records = sorted(records, key=_record_rank, reverse=True)

        for i, record in enumerate(ranked_records[: max_files + 4], 1):
            if remaining_slots <= 0:
                break
            before_count = len(created_files)
            # Use LLM to infer content-aware filename
            content_filename = self._infer_content_aware_filename(record, ctx)
            
            real_samples = record.get("real_data_samples", [])
            schema = record.get("data_schema", {})
            if not real_samples and record.get("real_tables"):
                table_samples, table_schema = self._extract_table_samples(record.get("real_tables", []))
                if table_samples:
                    real_samples = table_samples
                    record.setdefault("real_data_samples", table_samples)
                if table_schema and not schema:
                    schema = table_schema
                    record.setdefault("data_schema", table_schema)
            
            if not real_samples and not schema:
                continue
            
            # Strategy 1: If we have real data samples, create CSV directly
            if real_samples and isinstance(real_samples, list) and len(real_samples) > 0:
                if self._looks_synthetic_samples(real_samples):
                    logger.info(
                        "Skipping synthetic samples for %s (placeholder-like values detected)",
                        record.get("title", "unknown"),
                    )
                    real_samples = []
                    record["real_data_samples"] = []
                csv_path = data_dir / f"{content_filename}.csv"
                try:
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        if isinstance(real_samples[0], dict):
                            # List of dicts → CSV
                            fieldnames = list(real_samples[0].keys())
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(list(islice(real_samples, 0, max_sample_rows)))
                            created_files.append(f"{content_filename}.csv")
                        elif isinstance(real_samples[0], list):
                            # List of lists → CSV
                            writer = csv.writer(f)
                            writer.writerows(real_samples[:max_sample_rows])
                            created_files.append(f"{content_filename}.csv")
                except Exception as e:
                    logger.warning(f"Failed to create CSV for {record.get('title', 'unknown')}: {e}")
            
            # Strategy 2: If we have schema with samples, create JSON
            elif schema and schema.get("samples"):
                json_path = data_dir / f"{content_filename}_schema.json"
                try:
                    data_to_save = {
                        "dataset_name": record.get("title", ""),
                        "source_url": record.get("url", ""),
                        "schema": schema,
                        "timestamp": datetime.now().isoformat(),
                    }
                    dump_json(json_path, data_to_save)
                    created_files.append(f"{content_filename}_schema.json")
                except Exception as e:
                    logger.warning(f"Failed to create JSON for {record.get('title', 'unknown')}: {e}")
            if len(created_files) > before_count:
                remaining_slots = max(0, max_files - len(created_files))
        
        # Create a summary SQLite database with metadata
        try:
            db_path = data_dir / "datasets_metadata.db"
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    id INTEGER PRIMARY KEY,
                    name TEXT,
                    source_url TEXT,
                    has_real_data BOOLEAN,
                    record_count INTEGER,
                    created_at TEXT
                )
            """)
            
            for record in records[:10]:
                cursor.execute(
                    "INSERT INTO datasets (name, source_url, has_real_data, record_count, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        record.get("title", ""),
                        record.get("url", ""),
                        bool(record.get("real_data_samples")),
                        len(record.get("real_data_samples", [])),
                        datetime.now().isoformat(),
                    )
                )
            
            conn.commit()
            conn.close()
            created_files.append("datasets_metadata.db")
        except Exception as e:
            logger.warning(f"Failed to create metadata database: {e}")
        
        ctx.add_step({
            "type": "create_data_files_direct",
            "files_created": created_files,
            "count": len(created_files),
        })
        
        logger.info(f"Created {len(created_files)} data files directly from real data: {created_files}")
