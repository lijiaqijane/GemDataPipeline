from __future__ import annotations

import json
import logging
import re
import sqlite3
from pathlib import Path
from typing import Any, TYPE_CHECKING

from agent_gem.sandbox import SandboxExecutor

from ..base import BaseAgent, TaskContext

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest  # noqa: F401

logger = logging.getLogger(__name__)


class SetupMixin:
    """Setup and sandbox inspection helpers for GeneralAgent."""

    def _generate_setup_bundle(
        self, topic: str, ctx: TaskContext, sandbox: SandboxExecutor, records: list[dict[str, Any]] | None = None
    ) -> tuple[dict[str, str], dict[str, str]]:
        """Generate and execute setup_env.py; return bundle info and snapshot after setup."""
        if not records:
            raise ValueError("Records must be provided to generate setup_env.py. Cannot proceed without reference data.")
        
        # NEW APPROACH: Lightweight setup_env.py that reads from existing data files
        # Extract only metadata (not full data) to keep prompt short
        datasets_meta = []
        for i, record in enumerate(records[:10], 1):
            dataset_name = record.get("title", f"dataset_{i}")
            safe_name = re.sub(r'[^\w\s-]', '', dataset_name)[:50].strip().replace(' ', '_').lower()
            
            meta = {
                "dataset_name": dataset_name,
                "safe_filename": safe_name,
                "description": record.get("summary", "")[:200],
                "source_url": record.get("url", ""),
                "has_real_data": bool(record.get("real_data_samples")),
                "has_schema": bool(record.get("data_schema")),
                "sample_count": len(record.get("real_data_samples", [])),
            }
            datasets_meta.append(meta)
        
        meta_json = json.dumps(datasets_meta, ensure_ascii=False, indent=2)
        
        base_prompt = (
            "You are a data environment architect. Generate setup_env.py for downstream agent practice.\n\n"
            "CRITICAL ARCHITECTURE CHANGE:\n"
            "- Real data files have ALREADY been created in the sandbox (CSV, JSON, SQLite).\n"
            "- Your setup_env.py should be LIGHTWEIGHT - it validates, augments, or transforms existing files.\n"
            "- DO NOT regenerate data from scratch - read from existing files and enhance them.\n\n"
            "Constraints:\n"
            "- Output exactly one Python code block; filename is setup_env.py and runnable as-is.\n"
            "- Data files live under ./data (CSV/JSON/SQLite); operate there and keep outputs in the same folder.\n"
            "- Read existing data files (e.g., {safe_filename}_real.csv, records.json, datasets_metadata.db).\n"
            "- You may: add anomalies, create derived datasets, add logs, validate data integrity.\n"
            "- You may NOT: ignore existing files and create data from scratch.\n"
            "- No network access; prefer stdlib.\n"
            "- Keep it idempotent (can run multiple times safely).\n"
            f"- Topic: {topic}\n\n"
            "Example tasks for setup_env.py:\n"
            "1. Read {safe_filename}_real.csv and create a corrupted version with anomalies\n"
            "2. Generate synthetic logs based on patterns in existing data\n"
            "3. Create indexes or views in SQLite database\n"
            "4. Add timestamp columns, missing values, or outliers\n"
            "5. Create a summary report of available datasets\n\n"
            f"Existing Data Files Metadata:\n{meta_json}\n"
        )

        setup_code = ""
        exec_result: dict[str, str] = {}
        # Use max_tokens from request, with fallback to 10000
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        for attempt in range(3):
            raw = self.llm.simple_complete(
                base_prompt
                + (f"\nPrevious error: {exec_result.get('stderr','')[:500]}" if attempt else ""),
                temperature=0.45 + 0.1 * attempt,
                max_tokens=max_tokens,
            )
            ctx.add_step(
                {
                    "type": "setup_generation",
                    "attempt": attempt + 1,
                    "content": raw,
                }
            )
            blocks = BaseAgent._extract_code_blocks(raw)
            setup_code = BaseAgent._strip_code_fences(blocks[0] if blocks else raw)

            setup_path = sandbox.sandbox_dir / "setup_env.py"
            setup_path.write_text(setup_code, encoding="utf-8")

            exec_result = sandbox.execute_bash("python setup_env.py")
            ctx.add_step(
                {
                    "type": "setup_execution",
                    "attempt": attempt + 1,
                    "returncode": exec_result.get("returncode"),
                    "stdout": exec_result.get("stdout", "")[:4000],
                    "stderr": exec_result.get("stderr", "")[:4000],
                }
            )
            if exec_result.get("returncode", 1) == 0:
                compile_result = sandbox.execute_bash("python -m py_compile setup_env.py")
                ctx.add_step(
                    {
                        "type": "setup_pycache",
                        "returncode": compile_result.get("returncode"),
                        "stdout": compile_result.get("stdout", "")[:4000],
                        "stderr": compile_result.get("stderr", "")[:4000],
                    }
                )
                break

        if exec_result.get("returncode", 1) != 0:
            raise RuntimeError(
                f"setup_env.py failed after retries: {exec_result.get('stderr','')[:200]}"
            )

        snapshot = sandbox.snapshot_fs()
        return (
            {
                "setup_code": setup_code,
                "returncode": str(exec_result.get("returncode", "")),
                "stdout": exec_result.get("stdout", "")[:1000],
                "stderr": exec_result.get("stderr", "")[:1000],
            },
            snapshot,
        )

    def _inspect_data_sources(self, sandbox: SandboxExecutor, ctx: TaskContext) -> dict[str, Any]:
        """Enumerate local data artifacts (CSV/JSON/SQLite/logs) with lightweight schema samples."""
        base = sandbox.sandbox_dir
        profile: dict[str, Any] = {"csv": [], "json": [], "sqlite": [], "logs": [], "files": []}

        def _limit(obj: Any, max_len: int = 800) -> Any:
            text = json.dumps(obj, ensure_ascii=False)
            if len(text) > max_len:
                return text[:max_len] + "...(truncated)"
            return obj

        for path in base.rglob("*"):
            if path.is_dir():
                continue
            rel = path.relative_to(base).as_posix()
            if rel.startswith("logs/") or rel.startswith("runs/"):
                continue
            suffix = path.suffix.lower()
            profile["files"].append(rel)
            try:
                if suffix == ".csv":
                    import csv

                    with path.open("r", encoding="utf-8", errors="ignore") as f:
                        reader = csv.DictReader(f)
                        rows = []
                        for idx, row in enumerate(reader):
                            if idx >= 3:
                                break
                            rows.append(row)
                        profile["csv"].append({"path": rel, "fields": reader.fieldnames or [], "samples": rows})
                elif suffix in {".json", ".ndjson"}:
                    with path.open("r", encoding="utf-8", errors="ignore") as f:
                        raw = f.read()
                        data = json.loads(raw)
                        if isinstance(data, list) and data:
                            sample = data[:3]
                        elif isinstance(data, dict):
                            sample = {k: data[k] for k in list(data.keys())[:10]}
                        else:
                            sample = data
                        profile["json"].append({"path": rel, "sample": _limit(sample)})
                elif suffix in {".db", ".sqlite"}:
                    conn = sqlite3.connect(path)
                    cur = conn.cursor()
                    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
                    tables = [r[0] for r in cur.fetchall()]
                    table_samples = []
                    for t in tables:
                        cur.execute(f"PRAGMA table_info('{t}')")
                        cols = [r[1] for r in cur.fetchall()]
                        cur.execute(f"SELECT * FROM '{t}' LIMIT 3")
                        rows = cur.fetchall()
                        table_samples.append({"table": t, "columns": cols, "rows": rows})
                    conn.close()
                    profile["sqlite"].append({"path": rel, "tables": table_samples})
                elif suffix == ".log":
                    with path.open("r", encoding="utf-8", errors="ignore") as f:
                        lines: list[str] = []
                        for _ in range(5):
                            line = f.readline()
                            if not line:
                                break
                            lines.append(line.rstrip("\n"))
                    profile["logs"].append({"path": rel, "lines": lines})
            except Exception:
                logger.debug("inspect_data_sources failed for %s", rel, exc_info=True)

        ctx.add_step({"type": "data_profile", "content": _limit(profile, 2000)})
        return profile
