from __future__ import annotations

import base64
import json
import logging
import os
import re
import shlex
import textwrap
import uuid
from urllib.parse import urljoin, urlparse
from pathlib import Path
from typing import Any, TYPE_CHECKING

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

    def _raw_pages_enabled(self) -> bool:
        value = os.getenv("USE_RAW_PAGES", "1").strip().lower()
        return value not in {"0", "false", "no"}

    def _fetch_url_content(self, sandbox: SandboxExecutor, url: str, timeout: int = 10) -> dict[str, Any]:
        """Fetch and extract content from a URL via sandbox bash tool."""
        result = {
            "url": url,
            "success": False,
            "content": "",
            "tables": [],
            "links": [],
            "raw_path": "",
            "raw_size": 0,
            "error": None,
            "error_detail": None,
        }

        if not url or not url.startswith(("http://", "https://")):
            result["error"] = "Invalid URL"
            return result

        data_exts = sorted(self._DATA_FILE_EXTENSIONS)
        raw_dir = (sandbox.sandbox_dir / "data" / "raw_pages").relative_to(sandbox.sandbox_dir)
        cmd = (
            f"URL={shlex.quote(url)} "
            f"TIMEOUT={shlex.quote(str(timeout))} "
            "MAX_BYTES=2000000 "
            f"DATA_EXTS={shlex.quote(','.join(data_exts))} "
            f"SAVE_DIR={shlex.quote(raw_dir.as_posix())} "
            f"SAVE_RAW={shlex.quote('1' if self._raw_pages_enabled() else '0')} "
            "python - <<'PY'\n"
            "import hashlib, html, json, os, pathlib, re, urllib.parse, urllib.request\n"
            "from html.parser import HTMLParser\n"
            "url = os.environ.get('URL', '')\n"
            "timeout = float(os.environ.get('TIMEOUT', '10'))\n"
            "max_bytes = int(os.environ.get('MAX_BYTES', '2000000'))\n"
            "exts = [e for e in os.environ.get('DATA_EXTS', '').split(',') if e]\n"
            "save_dir = os.environ.get('SAVE_DIR', 'data/raw_pages')\n"
            "save_raw = os.environ.get('SAVE_RAW', '1')\n"
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
            "        if len(ln) < 20:\n"
            "            continue\n"
            "        if re.search(r'[{};<>]|function\\s*\\(|var\\s+|let\\s+|const\\s+|@keyframes', ln):\n"
            "            continue\n"
            "        if re.search(r'\\b(document|window|navigator|jquery|\\$)\\b', ln, flags=re.I):\n"
            "            continue\n"
            "        filtered.append(ln)\n"
            "    content = '\\n'.join(filtered[:400])\n"
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
            "        tables = parser.tables[:6]\n"
            "    except Exception:\n"
            "        tables = []\n"
            "    return content, links, tables\n"
            "def _safe_save(raw_text: str, url_value: str) -> tuple[str, int]:\n"
            "    try:\n"
            "        if save_raw in {'0', 'false', 'no'}:\n"
            "            return '', 0\n"
            "        base = pathlib.Path(save_dir)\n"
            "        if base.is_absolute():\n"
            "            base = pathlib.Path('data/raw_pages')\n"
            "        base.mkdir(parents=True, exist_ok=True)\n"
            "        slug = hashlib.md5(url_value.encode('utf-8')).hexdigest()[:16]\n"
            "        target = base / f'page_{slug}.html'\n"
            "        target.write_text(raw_text, encoding='utf-8', errors='replace')\n"
            "        return str(target.as_posix()), len(raw_text.encode('utf-8', errors='replace'))\n"
            "    except Exception:\n"
            "        return '', 0\n"
            "def fetch_once(u: str, headers: dict[str, str]) -> tuple[bool, str, list[str], list[list[list[str]]], str, str, int]:\n"
            "    req = urllib.request.Request(u, headers=headers)\n"
            "    with urllib.request.urlopen(req, timeout=timeout) as resp:\n"
            "        raw = resp.read(max_bytes + 1)\n"
            "    if len(raw) > max_bytes:\n"
            "        raw = raw[:max_bytes]\n"
            "    text = raw.decode('utf-8', 'replace')\n"
            "    raw_path, raw_size = _safe_save(text, u)\n"
            "    content, links, tables = extract(text, u)\n"
            "    return True, content[:10000], links[:20], tables[:6], '', raw_path, raw_size\n"
            "def try_fetch(u: str) -> tuple[bool, str, list[str], list[list[list[str]]], str, str, int]:\n"
            "    headers_list = [\n"
            "        {\n"
            "            'User-Agent': 'Mozilla/5.0',\n"
            "            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',\n"
            "            'Accept-Language': 'en-US,en;q=0.9',\n"
            "        },\n"
            "        {\n"
            "            'User-Agent': 'Mozilla/5.0',\n"
            "            'Accept': '*/*',\n"
            "            'Accept-Language': 'en-US,en;q=0.9',\n"
            "        },\n"
            "    ]\n"
            "    for headers in headers_list:\n"
            "        try:\n"
            "            return fetch_once(u, headers)\n"
            "        except Exception as e:\n"
            "            errors.append(str(e)[:120])\n"
            "    return False, '', [], [], errors[-1] if errors else 'fetch_failed', '', 0\n"
            "def jina_proxy(u: str) -> str:\n"
            "    parsed = urllib.parse.urlparse(u)\n"
            "    scheme = 'http' if parsed.scheme == 'http' else 'https'\n"
            "    return f\"https://r.jina.ai/{scheme}://{parsed.netloc}{parsed.path}\"\n"
            "try:\n"
            "    if not url.startswith(('http://', 'https://')):\n"
            "        raise ValueError('Invalid URL')\n"
            "    ok, content, links, tables, err, raw_path, raw_size = try_fetch(url)\n"
            "    if not ok:\n"
            "        proxy = jina_proxy(url)\n"
            "        ok, content, links, tables, err, raw_path, raw_size = try_fetch(proxy)\n"
            "    if ok:\n"
            "        out = {'success': True, 'content': content, 'tables': tables, 'links': links, 'raw_path': raw_path, 'raw_size': raw_size, 'error': None, 'error_detail': None}\n"
            "        print(json.dumps(out, ensure_ascii=False))\n"
            "    else:\n"
            "        out = {'success': False, 'content': '', 'tables': [], 'links': [], 'raw_path': '', 'raw_size': 0, 'error': err or 'fetch_failed', 'error_detail': '; '.join(errors)[:300]}\n"
            "        print(json.dumps(out, ensure_ascii=False))\n"
            "except Exception as e:\n"
            "    print(json.dumps({'success': False, 'content': '', 'tables': [], 'links': [], 'raw_path': '', 'raw_size': 0, 'error': str(e)[:100], 'error_detail': '; '.join(errors)[:300]}, ensure_ascii=False))\n"
            "PY"
        )

        tool_result = sandbox.execute_bash(cmd, timeout_s=timeout + 5)
        stdout = (tool_result.get("stdout") or "").strip()
        stderr = (tool_result.get("stderr") or "").strip()
        returncode = tool_result.get("returncode", 0)
        return_code = tool_result.get("return_code", None)
        if isinstance(returncode, str) and returncode.isdigit():
            returncode = int(returncode)
        if isinstance(return_code, str) and return_code.isdigit():
            return_code = int(return_code)

        parsed: dict[str, Any] = {}
        try:
            parsed = json.loads(stdout) if stdout else {}
        except Exception as exc:
            result["error"] = f"Parse error: {str(exc)[:100]}"
            if stdout:
                result["error_detail"] = stdout[:500]
            return result

        if isinstance(parsed, dict):
            result.update(parsed)
        else:
            result["error"] = "Invalid tool output"
            if stdout:
                result["error_detail"] = stdout[:500]
            return result

        # If tool output says success, require some payload before trusting it.
        if result.get("success"):
            has_payload = bool(result.get("content")) or bool(result.get("links"))
            if has_payload:
                return result

        if (returncode and returncode != 0) or (return_code not in (0, None)):
            result["error"] = result.get("error") or "Tool error"
            detail = stderr or stdout
            if detail:
                result["error_detail"] = detail[:500]
            return result

        if result.get("error") and not result.get("error_detail"):
            result["error_detail"] = str(result.get("error"))[:500]
        return result

    def _is_data_file_url(self, url: str) -> bool:
        path = urlparse(url).path.lower()
        return any(path.endswith(ext) for ext in self._DATA_FILE_EXTENSIONS)

    def _download_dataset_file(
        self,
        sandbox: SandboxExecutor,
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
        url = self._rewrite_data_url(url)
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
            "max_bytes = int(os.environ.get('MAX_BYTES', '6000000'))\n"
            "timeout = float(os.environ.get('TIMEOUT', '15'))\n"
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
            "                    chunk = resp.read(65536)\n"
            "                    if not chunk:\n"
            "                        break\n"
            "                    if size == 0:\n"
            "                        lower = chunk[:512].lower()\n"
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
            "            errors.append(str(e)[:120])\n"
            "            if path.exists():\n"
            "                path.unlink(missing_ok=True)\n"
            "    if not ok:\n"
            "        raise ValueError('; '.join(errors)[:200])\n"
            "    print(json.dumps({'ok': True, 'path': str(path)}))\n"
            "except Exception as e:\n"
            "    if 'path' in locals() and path.exists():\n"
            "        path.unlink(missing_ok=True)\n"
            "    print(json.dumps({'ok': False, 'error': str(e)[:200]}))\n"
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
                    for row in data_rows:
                        if len(row) != len(header):
                            continue
                        padded = (row + [""] * len(header))[: len(header)]
                        samples.append({header[i]: padded[i] for i in range(len(header))})
                        if len(samples) >= max_rows:
                            break
                return self._filter_real_samples(samples)

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
            return self._filter_real_samples(samples)

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
            return self._filter_real_samples(samples)

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
            return self._filter_real_samples(samples)

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
            return self._filter_real_samples(samples)

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
            return self._filter_real_samples(samples)

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

    def _looks_like_asset_value(self, value: str) -> bool:
        v = value.strip().lower()
        if not v:
            return False
        if v.startswith("data:image/"):
            return True
        return bool(re.search(r"\.(png|jpg|jpeg|gif|svg|ico|webp|css|js)(\\?|$)", v))

    def _is_noise_sample(self, sample: dict[str, Any]) -> bool:
        if not sample:
            return True
        keys = {str(k).strip().lower() for k in sample.keys() if k}
        if not keys or len(keys) < 2:
            return True
        if keys.issubset({"src", "sizes", "type", "rel", "href", "integrity", "crossorigin", "density"}):
            return True
        if keys in ({"line_number", "text"}, {"line", "text"}, {"index", "text"}):
            return True
        js_config_markers = {
            "user",
            "userid",
            "currentpath",
            "userproductid",
            "locale",
            "digitseparator",
            "digitdecimal",
            "env",
            "abtestgroup",
        }
        if keys.issubset(js_config_markers) or keys.issubset(js_config_markers | {"timezone", "country"}):
            return True
        values = [v for v in sample.values() if isinstance(v, str)]
        if values:
            asset_hits = sum(1 for v in values if self._looks_like_asset_value(v))
            if asset_hits / max(1, len(values)) >= 0.5:
                return True
        return False

    def _filter_real_samples(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not samples:
            return []
        cleaned = [row for row in samples if isinstance(row, dict) and not self._is_noise_sample(row)]
        return cleaned or []

    def _extract_info_snippets(self, text: str, *, max_snippets: int = 8) -> list[str]:
        """Extract short, high-signal snippets from unstructured text."""
        if not text:
            return []
        candidates: list[str] = []
        for raw in re.split(r"[\n\.]", text):
            line = raw.strip()
            if len(line) < 30 or len(line) > 220:
                continue
            if self._looks_like_code_line(line):
                continue
            candidates.append(line)
        if not candidates:
            return []

        def score(line: str) -> int:
            s = 0
            if re.search(r"\d", line):
                s += 2
            if re.search(r"\b(\d{4}|\d{1,2}[:/]\d{1,2})\b", line):
                s += 1
            if re.search(r"%|\$|€|£|¥", line):
                s += 1
            if re.search(r"\b(api|dataset|download|opening|hours|price|address|phone|email|website)\b", line, re.I):
                s += 1
            if re.search(r"\b[A-Z][a-z]{2,}\b", line):
                s += 1
            return s

        ranked = sorted(candidates, key=score, reverse=True)
        seen: set[str] = set()
        snippets: list[str] = []
        for line in ranked:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            snippets.append(line)
            if len(snippets) >= max_snippets:
                break
        return snippets

    def _looks_like_code_line(self, line: str) -> bool:
        if re.search(r"[{};<>]|function\s*\(|var\s+|let\s+|const\s+", line):
            return True
        if re.search(r"\b(document|window|navigator|jquery|\$)\b", line, re.I):
            return True
        if re.search(r"@keyframes|\.css|\.js|<script|</script", line, re.I):
            return True
        if line.count("{") + line.count("}") >= 2:
            return True
        return False

    def _clean_content_with_jina(
        self,
        text: str,
        *,
        topic: str,
        sandbox: SandboxExecutor | None = None,
        max_chars: int = 2400,
    ) -> str:
        """Filter noisy text via Jina rerank (sandbox), fallback to safe truncation when disabled or failing."""
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

        if sandbox is None:
            logger.warning("Jina cleaning disabled: no sandbox available; using truncation fallback.")
            combined = " ".join(chunks[:6])
            return combined[:max_chars]

        top_n = min(12, len(chunks))
        jina_chunks = chunks[:top_n]
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

        cmd = (
            "python - <<'PY'\n"
            "import base64, json, os, urllib.request\n"
            "query_b64 = os.environ.get('JINA_QUERY_B64', '')\n"
            "docs_b64 = os.environ.get('JINA_DOCS_B64', '')\n"
            "query = base64.b64decode(query_b64.encode('ascii')).decode('utf-8', 'replace') if query_b64 else ''\n"
            "docs_json = base64.b64decode(docs_b64.encode('ascii')).decode('utf-8', 'replace') if docs_b64 else '[]'\n"
            "docs = json.loads(docs_json)\n"
            "api_key = os.environ.get('JINA_API_KEY', '')\n"
            "payload = json.dumps({\n"
            "  'model': 'jina-reranker-v2-base-multilingual',\n"
            "  'query': query,\n"
            "  'documents': docs,\n"
            "  'top_n': min(12, len(docs))\n"
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
            "  print(json.dumps({'error': str(e)[:200]}))\n"
            "PY"
        )

        query_b64 = base64.b64encode(query.encode("utf-8")).decode("ascii")
        docs_b64 = base64.b64encode(json.dumps(jina_chunks).encode("utf-8")).decode("ascii")
        env_prefix = (
            f"JINA_API_KEY={shlex.quote(api_key)} "
            f"JINA_QUERY_B64={shlex.quote(query_b64)} "
            f"JINA_DOCS_B64={shlex.quote(docs_b64)} "
        )
        result = sandbox.execute_bash(env_prefix + cmd, timeout_s=timeout_s + 5)
        stdout = (result.get("stdout") or "").strip()
        stderr = (result.get("stderr") or "").strip()
        try:
            payload = json.loads(stdout) if stdout else {}
        except Exception:
            payload = {}
        if isinstance(payload, dict) and payload.get("data"):
            data = payload.get("data", [])
            ranked = sorted(
                data,
                key=lambda x: x.get("relevance_score", 0),
                reverse=True,
            )
            selected = [chunks[item.get("index", 0)] for item in ranked[:top_n] if isinstance(item, dict)]
            combined = " ".join(selected)
            return combined[:max_chars] if combined else text[:max_chars]
        if isinstance(payload, dict) and payload.get("error"):
            logger.warning("Jina cleaning failed: %s", str(payload.get("error"))[:200])
        else:
            if isinstance(payload, dict) and payload.get("data") == []:
                logger.info("Jina cleaning returned empty rerank list; using truncation fallback.")
            else:
                detail = stderr or stdout
                if detail:
                    logger.warning("Jina cleaning failed: empty response (%s)", detail[:200])
                else:
                    logger.warning("Jina cleaning failed: empty response")

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

        samples = self._filter_real_samples(samples)
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
        if record.get("info_snippets"):
            score += 1
        return score

    def _sort_records(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return sorted(
            records,
            key=lambda r: (self._record_quality_score(r), (r.get("title") or "").lower()),
            reverse=True,
        )

    def _topic_keywords(self, topic: str) -> list[str]:
        stopwords = {
            "the", "and", "for", "with", "from", "this", "that", "these", "those",
            "data", "dataset", "datasets", "open", "portal", "planning",
        }
        tokens = re.split(r"[^\w]+", topic.lower())
        keywords = [t for t in tokens if t and len(t) >= 3 and t not in stopwords]
        return list(dict.fromkeys(keywords))  # preserve order, dedupe

    def _topic_phrase(self, topic: str) -> str:
        return re.sub(r"\s+", " ", topic.strip().lower())

    def _topic_focus_terms(
        self, topic: str, extra_terms: list[str] | None = None
    ) -> list[str]:
        terms: list[str] = []
        terms.extend(self._topic_keywords(topic))
        if extra_terms:
            terms.extend(extra_terms)
        normalized: list[str] = []
        seen: set[str] = set()
        for term in terms:
            cleaned = re.sub(r"\s+", " ", str(term).strip().lower())
            if not cleaned or len(cleaned) < 3:
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            normalized.append(cleaned)
        return normalized

    def _topic_geo_terms(self, topic: str) -> list[str]:
        stop = {
            "travel",
            "planning",
            "trip",
            "tourism",
            "guide",
            "vacation",
            "itinerary",
            "plan",
        }
        keywords = [kw for kw in self._topic_keywords(topic) if kw not in stop]
        return self._topic_focus_terms(topic, keywords)

    def _noise_penalty(self, text: str) -> int:
        if not text:
            return 0
        noise_terms = {
            "login", "sign in", "signin", "sign up", "register", "subscribe",
            "privacy", "terms", "cookie", "policy", "advertisement", "sponsored",
            "newsletter", "contact", "about us", "careers", "jobs", "press",
            "cart", "checkout", "buy", "shop", "order",
            "app store", "google play", "play.google", "itunes",
            "vacation packages", "travel guide", "things to do", "blog", "review site",
            "reddit", "forum", "community",
        }
        lowered = text.lower()
        return sum(1 for term in noise_terms if term in lowered)

    def _data_signal_score(self, text: str, url: str = "") -> int:
        haystack = f"{text} {url}".lower()
        score = 0
        if url and self._is_data_file_url(url):
            score += 4
        signal_terms = {
            "dataset", "data portal", "open data", "api", "registry",
            "statistics", "report", "csv", "json", "xlsx", "sqlite",
            "download", "schema", "table", "metadata",
        }
        score += sum(1 for term in signal_terms if term in haystack)
        return min(score, 8)

    def _canonicalize_url(self, url: str) -> str:
        if not url:
            return ""
        try:
            parsed = urlparse(url)
        except Exception:
            return url.strip()
        scheme = (parsed.scheme or "https").lower()
        netloc = parsed.netloc.lower()
        path = parsed.path or ""
        if path != "/" and path.endswith("/"):
            path = path[:-1]
        return f"{scheme}://{netloc}{path}"

    def _rewrite_data_url(self, url: str) -> str:
        if not url:
            return url
        parsed = urlparse(url)
        host = parsed.netloc.lower()
        path = parsed.path
        if "github.com" in host and "/blob/" in path:
            raw_path = path.replace("/blob/", "/")
            return f"https://raw.githubusercontent.com{raw_path}"
        return url

    def _geo_match_score(self, record: dict[str, Any], geo_terms: list[str]) -> int:
        if not geo_terms:
            return 1
        title = str(record.get("title") or "")
        summary = str(record.get("summary") or "")
        url = str(record.get("url") or "")
        content = str(record.get("clean_content") or record.get("real_content") or "")
        haystack = f"{title} {summary} {content} {url}".lower()
        for term in geo_terms:
            if term and term in haystack:
                return 1
        samples = record.get("real_data_samples") or []
        if isinstance(samples, list):
            for sample in samples[:5]:
                if isinstance(sample, dict):
                    for val in sample.values():
                        if isinstance(val, str) and any(term in val.lower() for term in geo_terms):
                            return 1
        return 0

    def _is_data_like_record(self, record: dict[str, Any]) -> bool:
        if record.get("real_data_samples") or record.get("real_tables"):
            return True
        if record.get("downloaded_files"):
            return True
        url = str(record.get("url") or "")
        if self._is_data_file_url(url):
            return True
        title = str(record.get("title") or "")
        summary = str(record.get("summary") or "")
        signal = self._data_signal_score(f"{title} {summary}", url)
        if record.get("data_schema") and signal >= 3:
            return True
        return signal >= 3

    def _dedupe_records_by_url(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        merged: dict[str, dict[str, Any]] = {}
        for rec in records:
            url = self._canonicalize_url(str(rec.get("url") or ""))
            key = url or (str(rec.get("title") or "").lower().strip() or uuid.uuid4().hex)
            existing = merged.get(key)
            if existing:
                combined = self._merge_record_fields(existing, rec)
                if self._record_quality_score(rec) > self._record_quality_score(existing):
                    combined = self._merge_record_fields(rec, combined)
                merged[key] = combined
            else:
                merged[key] = rec
        return list(merged.values())

    def _record_relevance_score(
        self,
        record: dict[str, Any],
        keywords: list[str],
        topic_phrase: str,
        focus_terms: list[str] | None = None,
        geo_terms: list[str] | None = None,
    ) -> int:
        title = str(record.get("title") or "")
        summary = str(record.get("summary") or "")
        url = str(record.get("url") or "")
        content = str(record.get("clean_content") or record.get("real_content") or "")
        haystack = f"{title} {summary} {content} {url}".lower()
        score = 0
        if topic_phrase and topic_phrase in haystack:
            score += 6
        for kw in keywords:
            if kw in title.lower():
                score += 3
            if kw in summary.lower():
                score += 2
            if kw in content.lower():
                score += 1
            if kw in url.lower():
                score += 1
        if focus_terms:
            matched_focus = False
            for term in focus_terms:
                if term in title.lower():
                    score += 2
                    matched_focus = True
                if term in summary.lower():
                    score += 1
                    matched_focus = True
                if term in url.lower():
                    score += 1
                    matched_focus = True
            if not matched_focus:
                score -= 1
        if any(url.lower().endswith(ext) for ext in self._DATA_FILE_EXTENSIONS):
            score += 2
        if record.get("real_data_samples") or record.get("data_schema"):
            score += 2
        score += self._data_signal_score(haystack, url)
        if geo_terms:
            geo_hit = self._geo_match_score(record, geo_terms)
            if not geo_hit:
                score -= 4
        score -= self._noise_penalty(haystack) * 2
        return score

    def _filter_records_by_relevance(
        self,
        records: list[dict[str, Any]],
        topic: str,
        *,
        focus_terms: list[str] | None = None,
        min_keep: int = 4,
    ) -> list[dict[str, Any]]:
        if not records:
            return records
        focus_terms = focus_terms or []
        keywords = self._topic_keywords(topic)
        topic_phrase = self._topic_phrase(topic)
        geo_terms = self._topic_geo_terms(topic)
        data_like = [rec for rec in records if self._is_data_like_record(rec)]
        if len(data_like) >= min_keep:
            records = data_like
        scored = [
            (
                self._record_relevance_score(
                    record, keywords, topic_phrase, focus_terms, geo_terms
                ),
                record,
            )
            for record in records
        ]
        scored.sort(key=lambda x: x[0], reverse=True)
        threshold = 2 if keywords else 1
        if focus_terms:
            threshold += 1
        filtered = [rec for score, rec in scored if score >= threshold]
        if len(filtered) < min_keep:
            fallback_count = min(len(scored), max(min_keep, len(filtered)))
            filtered = [rec for _, rec in scored[:fallback_count]]
        return filtered

    def _generate_search_queries(self, topic: str, *, max_queries: int = 10) -> list[str]:
        topic_phrase = self._topic_phrase(topic)
        keywords = self._topic_keywords(topic)
        prompt = (
            "You will design search queries for any topic. Output JSON with keys:\n"
            "- entities: list of key entities/objects relevant to the topic\n"
            "- attributes: list of useful fields/attributes for those entities\n"
            "- operations: list of common tasks (compare, rank, map, plan, forecast, summarize, filter)\n"
            "- sources: list of source types (official portal, open data, api, dataset, registry, report)\n"
            "- seed_queries (optional): 3-5 direct queries\n"
            "Keep items short and generic; do NOT mention specific domains unless implied by the topic.\n"
            "Return ONLY JSON.\n"
            f"Topic: {topic}\n"
        )
        raw = self.llm.simple_complete(prompt, temperature=0.2, max_tokens=260)
        parsed = self._extract_json(raw)

        entities: list[str] = []
        attributes: list[str] = []
        operations: list[str] = []
        sources: list[str] = []
        seed_queries: list[str] = []

        if isinstance(parsed, dict):
            if isinstance(parsed.get("entities"), list):
                entities = [str(x).strip() for x in parsed["entities"] if str(x).strip()]
            if isinstance(parsed.get("attributes"), list):
                attributes = [str(x).strip() for x in parsed["attributes"] if str(x).strip()]
            if isinstance(parsed.get("operations"), list):
                operations = [str(x).strip() for x in parsed["operations"] if str(x).strip()]
            if isinstance(parsed.get("sources"), list):
                sources = [str(x).strip() for x in parsed["sources"] if str(x).strip()]
            if isinstance(parsed.get("seed_queries"), list):
                seed_queries = [str(x).strip() for x in parsed["seed_queries"] if str(x).strip()]

        if not sources:
            sources = ["open data portal", "official portal", "api", "dataset", "registry"]

        focus_terms = self._topic_focus_terms(topic, entities + attributes)
        self._last_search_topic = topic
        self._last_search_focus_terms = focus_terms

        expansions: list[str] = []
        for ent in entities[:5] or [topic]:
            for attr in attributes[:5]:
                expansions.append(f"{topic} {ent} {attr} dataset csv")
                expansions.append(f"{topic} {ent} {attr} dataset json")
            for src in sources[:4]:
                expansions.append(f"{topic} {ent} {src}")
            if operations:
                expansions.append(f"{topic} {ent} {operations[0]} data")

        if not expansions:
            expansions = [
                f"{topic} dataset csv",
                f"{topic} dataset json",
                f"{topic} open data portal",
                f"{topic} api data",
                f"{topic} official statistics report",
            ]

        def _ensure_topic(q: str) -> str:
            q = q.strip()
            if not q:
                return q
            q_lower = q.lower()
            if topic_phrase and topic_phrase not in q_lower:
                return f"{topic} {q}".strip()
            return q

        def _query_is_relevant(q: str) -> bool:
            q_lower = q.lower()
            if topic_phrase and topic_phrase in q_lower:
                return True
            return any(kw in q_lower for kw in keywords)

        deduped: list[str] = []
        seen: set[str] = set()
        for item in seed_queries + expansions:
            item = _ensure_topic(item)
            if not item:
                continue
            if keywords and not _query_is_relevant(item):
                continue
            key = item.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(item)
            if len(deduped) >= max_queries:
                break

        if not deduped:
            logger.warning("LLM did not return usable search queries; falling back to topic only.")
            deduped = [
                f"{topic} dataset csv",
                f"{topic} dataset json",
                f"{topic} open data portal",
                f"{topic} official statistics",
                topic,
            ][:max_queries]

        # Prioritize data-centric queries to improve dataset alignment.
        def _data_query_score(q: str) -> int:
            return self._data_signal_score(q, "")

        deduped.sort(key=_data_query_score, reverse=True)

        return deduped

    def _hit_relevance_score(
        self,
        hit: dict[str, Any],
        keywords: list[str],
        topic_phrase: str,
        focus_terms: list[str] | None = None,
        geo_terms: list[str] | None = None,
    ) -> int:
        title = str(hit.get("title") or hit.get("name") or "")
        summary = str(hit.get("summary") or hit.get("snippet") or hit.get("description") or "")
        url = str(hit.get("url") or hit.get("link") or "")
        haystack = f"{title} {summary} {url}".lower()
        score = 0
        if topic_phrase and topic_phrase in haystack:
            score += 6
        for kw in keywords:
            if kw in title.lower():
                score += 3
            if kw in summary.lower():
                score += 2
            if kw in url.lower():
                score += 1
        if focus_terms:
            matched_focus = False
            for term in focus_terms:
                if term in title.lower():
                    score += 2
                    matched_focus = True
                if term in summary.lower():
                    score += 1
                    matched_focus = True
                if term in url.lower():
                    score += 1
                    matched_focus = True
            if not matched_focus:
                score -= 1
        if any(url.lower().endswith(ext) for ext in self._DATA_FILE_EXTENSIONS):
            score += 2
        if re.search(r"\b(dataset|csv|json|sqlite|open data|api)\b", haystack):
            score += 1
        score += self._data_signal_score(haystack, url)
        if geo_terms:
            geo_hit = 0
            for term in geo_terms:
                if term and term in haystack:
                    geo_hit = 1
                    break
            if not geo_hit:
                score -= 3
        score -= self._noise_penalty(haystack) * 2
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
        search_queries = self._generate_search_queries(topic)
        focus_terms = self._topic_focus_terms(
            topic, getattr(self, "_last_search_focus_terms", None)
        )
        search_hits: list[dict[str, str]] = []
        for query in search_queries:
            result = sandbox.execute_search(query, max_results=6)
            if isinstance(result, list):
                for row in result:
                    if isinstance(row, dict):
                        search_hits.append(row)

        keywords = self._topic_keywords(topic)
        topic_phrase = self._topic_phrase(topic)
        geo_terms = self._topic_geo_terms(topic)
        scored_hits = [
            (self._hit_relevance_score(hit, keywords, topic_phrase, focus_terms, geo_terms), hit)
            for hit in search_hits
        ]
        min_hit_score = 2 if len(keywords) >= 2 else 1
        relevant_hits = [hit for score, hit in scored_hits if score >= min_hit_score]
        if relevant_hits and len(relevant_hits) >= 3:
            scored_hits = [
                (self._hit_relevance_score(hit, keywords, topic_phrase, focus_terms, geo_terms), hit)
                for hit in relevant_hits
            ]
        scored_hits.sort(key=lambda x: x[0], reverse=True)
        search_hits = [hit for _, hit in scored_hits]

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
                local_path = self._download_dataset_file(sandbox, url, raw_data_dir)
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
            url_data = self._fetch_url_content(sandbox, url, timeout=8)
            record["source"] = "search_with_real_content" if url_data["success"] else "search"

            if url_data["success"]:
                url_success_count += 1
                if url_data.get("raw_path"):
                    record["raw_html_path"] = url_data.get("raw_path")
                # Prefer direct dataset links if available
                found_dataset = False
                for link in url_data.get("links", [])[:6]:
                    local_path = self._download_dataset_file(sandbox, link, raw_data_dir)
                    if not local_path:
                        continue
                    samples = self._sample_dataset_file(local_path)
                    if not samples:
                        continue
                    record.setdefault("downloaded_files", []).append(str(local_path))
                    record["real_data_samples"] = samples
                    record["data_schema"] = self._schema_from_samples(samples)
                    record["source"] = "downloaded_dataset"
                    found_dataset = True
                    break
                if found_dataset:
                    search_records.append(record)
                    continue

                real_content = url_data["content"][:5000]
                record["real_content"] = real_content  # Store raw content
                record["clean_content"] = self._clean_content_with_jina(
                    real_content, topic=topic, sandbox=sandbox
                )
                info_source = record.get("clean_content") or real_content
                info_snippets = self._extract_info_snippets(info_source)
                if info_snippets:
                    record["info_snippets"] = info_snippets
                record["real_tables"] = url_data["tables"]  # Store real tables
                if url_data["tables"]:
                    table_samples, table_schema = self._extract_table_samples(url_data["tables"])
                    if table_samples:
                        record["real_data_samples"] = table_samples
                    if table_schema and not record.get("data_schema"):
                        record["data_schema"] = table_schema
            else:
                record["fetch_error"] = url_data["error"]
                if url_data.get("error_detail"):
                    record["fetch_error_detail"] = url_data["error_detail"]
            
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
        real_enriched: list[dict[str, Any]] = []
        summary_inferred: list[dict[str, Any]] = []
        
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
                        samples = item.get("real_data_samples", [])
                        if isinstance(samples, list):
                            samples = self._filter_real_samples(samples)
                        real_enriched.append({
                            "title": str(item.get("title", "")).strip(),
                            "summary": str(item.get("summary", "")).strip(),
                            "url": str(item.get("url", "")).strip(),
                            "source": "real_content_extracted",
                            "data_schema": item.get("data_schema", {}),
                            "real_data_samples": samples,
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
                        summary_inferred.append({
                            "title": str(item.get("title", "")).strip(),
                            "summary": str(item.get("summary", "")).strip(),
                            "url": str(item.get("url", "")).strip(),
                            "source": "summary_inferred",
                            "data_schema": item.get("data_schema", {}),
                            "real_data_samples": [],  # No real samples from blocked URLs
                        })
        
        primary_records = real_enriched or records_with_tables or records_with_real_content or []
        primary_records = primary_records or search_records
        records = self._merge_records(primary_records, search_records)
        if summary_inferred and len(records) < 3:
            records = self._merge_records(records, summary_inferred)
        records = self._dedupe_records_by_url(records)
        records = self._dedupe_records_by_title(records)
        records = self._sort_records(records)
        records = self._filter_records_by_relevance(
            records, topic, focus_terms=focus_terms
        )
        if not records:
            logger.warning("No records collected for topic '%s'; database will be empty.", topic)

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
        """Infer a deterministic filename based on title or URL (no LLM)."""
        title = record.get("title") or record.get("url") or "dataset"
        safe_name = slugify(str(title), max_length=40)
        return safe_name or "dataset"

    def _create_data_files_from_records(
        self, records: list[dict[str, Any]], sandbox: SandboxExecutor, ctx: TaskContext
    ) -> None:
        """Create actual data files directly from real data (bypassing LLM generation to avoid long prompts)."""
        import csv
        import os
        import sqlite3
        from datetime import datetime
        import html
        from html.parser import HTMLParser
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
        topic = ctx.request.topic or ""
        if topic:
            keywords = self._topic_keywords(topic)
            topic_phrase = self._topic_phrase(topic)
            focus_terms = self._topic_focus_terms(
                topic, getattr(self, "_last_search_focus_terms", None)
            )
            scored = [
                (
                    self._record_relevance_score(
                        rec, keywords, topic_phrase, focus_terms
                    ),
                    rec,
                )
                for rec in ranked_records
            ]
            scored.sort(key=lambda x: x[0], reverse=True)
            filtered = [rec for score, rec in scored if score >= 2]
            if filtered:
                ranked_records = filtered

        used_filenames: set[str] = set()
        for i, record in enumerate(ranked_records[: max_files + 4], 1):
            if remaining_slots <= 0:
                break
            before_count = len(created_files)
            source = (record.get("source") or "").lower()
            if source == "summary_inferred":
                continue
            # Use deterministic filename to keep downstream tooling stable
            content_filename = self._infer_content_aware_filename(record, ctx)
            if content_filename in used_filenames:
                continue
            
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
                    if csv_path.exists():
                        used_filenames.add(content_filename)
                        continue
                    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                        if isinstance(real_samples[0], dict):
                            # List of dicts → CSV
                            fieldnames = list(real_samples[0].keys())
                            writer = csv.DictWriter(f, fieldnames=fieldnames)
                            writer.writeheader()
                            writer.writerows(list(islice(real_samples, 0, max_sample_rows)))
                            created_files.append(f"{content_filename}.csv")
                            used_filenames.add(content_filename)
                        elif isinstance(real_samples[0], list):
                            # List of lists → CSV
                            writer = csv.writer(f)
                            writer.writerows(real_samples[:max_sample_rows])
                            created_files.append(f"{content_filename}.csv")
                            used_filenames.add(content_filename)
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

        # Create a structured SQLite database with full page paragraphs
        if self._raw_pages_enabled():
            try:
                pages_db_path = data_dir / "web_pages.db"
                conn = sqlite3.connect(pages_db_path)
                cursor = conn.cursor()
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS pages (
                    id INTEGER PRIMARY KEY,
                    url TEXT UNIQUE,
                    title TEXT,
                    fetched_at TEXT,
                    raw_path TEXT,
                    raw_html TEXT
                )
                """)
                cursor.execute("""
                CREATE TABLE IF NOT EXISTS paragraphs (
                    id INTEGER PRIMARY KEY,
                    page_id INTEGER,
                    position INTEGER,
                    text TEXT,
                    FOREIGN KEY(page_id) REFERENCES pages(id)
                )
                """)
                class _ParagraphParser(HTMLParser):
                    def __init__(self) -> None:
                        super().__init__()
                        self._in_p = False
                        self._buf: list[str] = []
                        self.paragraphs: list[str] = []

                    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
                        if tag.lower() == "p":
                            self._in_p = True
                            self._buf = []

                    def handle_endtag(self, tag: str) -> None:
                        if tag.lower() == "p":
                            text = " ".join(self._buf).strip()
                            text = re.sub(r"\s+", " ", text)
                            if text:
                                self.paragraphs.append(text)
                            self._in_p = False
                            self._buf = []

                    def handle_data(self, data: str) -> None:
                        if self._in_p:
                            self._buf.append(data)

                def _extract_paragraphs(raw_html: str) -> list[str]:
                    parser = _ParagraphParser()
                    try:
                        parser.feed(raw_html)
                        parser.close()
                    except Exception:
                        return []
                    if parser.paragraphs:
                        return [html.unescape(p) for p in parser.paragraphs]
                    # Fallback: split cleaned content into paragraphs
                    cleaned = re.sub(r"<[^>]+>", " ", raw_html)
                    chunks = [c.strip() for c in re.split(r"\n\s*\n", cleaned) if c.strip()]
                    fallback = []
                    for chunk in chunks:
                        chunk = re.sub(r"\s+", " ", chunk)
                        if len(chunk) >= 20:
                            fallback.append(html.unescape(chunk))
                    return fallback

                inserted_pages = 0
                inserted_paragraphs = 0
                referenced_raw_files: set[str] = set()
                for record in records:
                    raw_path = record.get("raw_html_path")
                    if not raw_path:
                        continue
                    raw_file = sandbox.sandbox_dir / raw_path
                    if not raw_file.exists():
                        continue
                    try:
                        raw_html = raw_file.read_text(encoding="utf-8", errors="replace")
                    except Exception:
                        continue
                    if not raw_html.strip():
                        continue
                    url = record.get("url", "")
                    title = record.get("title", "")
                    fetched_at = datetime.now().isoformat()

                    cursor.execute(
                        "INSERT OR IGNORE INTO pages (url, title, fetched_at, raw_path, raw_html) VALUES (?, ?, ?, ?, ?)",
                        (url, title, fetched_at, raw_path, raw_html),
                    )
                    referenced_raw_files.add(raw_file.name)
                    cursor.execute("SELECT id FROM pages WHERE url = ?", (url,))
                    row = cursor.fetchone()
                    if not row:
                        continue
                    page_id = int(row[0])
                    paragraphs = _extract_paragraphs(raw_html)
                    if paragraphs:
                        inserted_pages += 1
                    for idx, text in enumerate(paragraphs, 1):
                        cursor.execute(
                            "INSERT INTO paragraphs (page_id, position, text) VALUES (?, ?, ?)",
                            (page_id, idx, text),
                        )
                        inserted_paragraphs += 1

                conn.commit()
                conn.close()
                if inserted_pages:
                    created_files.append("web_pages.db")
                ctx.add_step({
                    "type": "create_web_pages_db",
                    "pages": inserted_pages,
                    "paragraphs": inserted_paragraphs,
                })
                if referenced_raw_files:
                    raw_dir = sandbox.sandbox_dir / "data" / "raw_pages"
                    removed = 0
                    if raw_dir.exists():
                        for path in raw_dir.glob("*.html"):
                            if path.name not in referenced_raw_files:
                                try:
                                    path.unlink()
                                    removed += 1
                                except Exception:
                                    logger.debug("Failed to remove unused raw page: %s", path, exc_info=True)
                    if removed:
                        ctx.add_step({
                            "type": "cleanup_raw_pages",
                            "removed": removed,
                        })
            except Exception as e:
                logger.warning(f"Failed to create web pages database: {e}")
        
        ctx.add_step({
            "type": "create_data_files_direct",
            "files_created": created_files,
            "count": len(created_files),
        })
        
        logger.info(f"Created {len(created_files)} data files directly from real data: {created_files}")

    def _inspect_data_sources(self, sandbox: SandboxExecutor, ctx: TaskContext) -> dict[str, Any]:
        """Enumerate local data artifacts (CSV/JSON/SQLite/logs) with lightweight schema samples."""
        import csv
        import sqlite3

        base = sandbox.sandbox_dir
        profile: dict[str, Any] = {"csv": [], "json": [], "sqlite": [], "logs": [], "files": []}

        def _safe_rel(path: Path) -> str:
            try:
                return str(path.relative_to(base))
            except Exception:
                return str(path)

        def _truncate(value: Any, limit: int = 200) -> Any:
            if isinstance(value, str):
                return value if len(value) <= limit else value[:limit] + "..."
            return value

        def _sample_csv(path: Path, max_rows: int = 5) -> dict[str, Any] | None:
            try:
                with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:
                    sample_text = f.read(4096)
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
                    samples.append({k: _truncate(item.get(k, "")) for k in list(item.keys())[:10]})
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
                for name in tables[:5]:
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

        for path in base.rglob("*"):
            if path.is_dir():
                continue
            rel = _safe_rel(path)
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
            profile["files"].append({"path": rel})

        ctx.add_step({"type": "inspect_data_sources", "summary": {k: len(v) for k, v in profile.items()}})
        return profile
