from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class LocalDatabase:
    """Lightweight JSON storage for scraped or generated data."""

    path: Path
    records: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "LocalDatabase":
        if path.exists():
            try:
                content = json.loads(path.read_text())
            except json.JSONDecodeError:
                content = {}
            records = content.get("records", [])
        else:
            records = []
        return cls(path=path, records=records)

    def add_record(self, record: Dict[str, Any]) -> None:
        title = record.get("title")
        summary = record.get("summary")
        # Deduplicate by title + summary to reduce noise.
        for row in self.records:
            if row.get("title") == title and row.get("summary") == summary:
                return
        self.records.append(record)
        self.save()

    def query(self, key: str, value: Any) -> List[Dict[str, Any]]:
        return [row for row in self.records if row.get(key) == value]

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"records": self.records}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
