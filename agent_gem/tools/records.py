from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from agent_gem.tools.base import BaseTool, ToolExecutionError


class JsonRecordsQueryTool(BaseTool):
    """Query a JSON list of records stored inside the sandbox."""

    def __init__(
        self,
        *,
        name: str,
        description: str,
        records_path: Path,
    ) -> None:
        super().__init__(name=name, description=description)
        self.records_path = records_path

    def execute(self, query: Any = None, max_results: int = 5) -> list[dict[str, Any]]:
        records = self._load_records()
        if query is None:
            return records[:max_results]

        text = query
        if isinstance(text, dict):
            text = json.dumps(text, ensure_ascii=False)
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text:
            return records[:max_results]

        lowered = text.lower()
        matches: list[dict[str, Any]] = []
        for record in records:
            title = str(record.get("title") or "")
            summary = str(record.get("summary") or "")
            haystack = f"{title}\n{summary}".lower()
            if lowered in haystack:
                matches.append(record)
        return matches[:max_results]

    def _load_records(self) -> list[dict[str, Any]]:
        if not self.records_path.exists():
            return []
        try:
            data = json.loads(self.records_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ToolExecutionError("invalid_records_json", [], message=str(exc))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict) and isinstance(data.get("records"), list):
            return [row for row in data["records"] if isinstance(row, dict)]
        return []
