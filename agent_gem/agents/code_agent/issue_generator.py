from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from agent_gem.llm import LLMClient
from .prompt import FEATURE_REQUEST_PROMPT


@dataclass
class IssueGeneratorConfig:
    """Configuration for IssueGenerator loaded from YAML."""

    entity_dir: Optional[Path] = None
    prompt_path: Optional[Path] = None
    max_tokens: int = 512
    body_excerpt_limit: int = 1200
    workers: int = 1
    overwrite_existing: bool = False


class IssueGenerator:
    """Generate feature request issues for extracted entities using an LLM."""

    def __init__(
        self,
        llm_client: LLMClient,
        prompt_template: Optional[str] = None,
        max_tokens: int = 512,
        body_excerpt_limit: int = 1200,
        workers: int = 1,
        overwrite_existing: bool = False,
    ) -> None:
        self.llm_client = llm_client
        self.prompt_template = prompt_template or FEATURE_REQUEST_PROMPT
        self.max_tokens = max_tokens
        self.body_excerpt_limit = body_excerpt_limit
        self.workers = max(1, int(workers))
        self.overwrite_existing = bool(overwrite_existing)
        self.logger = logging.getLogger(__name__)

    @classmethod
    def load_config_from_yaml(cls, yaml_path: str | Path) -> "IssueGenerator":
        """
        Load configuration from YAML file and create IssueGenerator instance.

        Args:
            yaml_path: Path to the YAML configuration file

        Returns:
            IssueGenerator instance with configuration loaded

        Raises:
            FileNotFoundError: If the YAML file does not exist
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")

        with open(yaml_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f) or {}

        ig_cfg = config_data.get("issue_generator", {}) or {}

        # Load prompt template
        prompt_path = ig_cfg.get("prompt_path")
        if prompt_path:
            prompt_path = Path(prompt_path)
            if prompt_path.exists():
                prompt_template = prompt_path.read_text(encoding="utf-8")
            else:
                logging.warning(f"Prompt file not found: {prompt_path}, using default")
                prompt_template = FEATURE_REQUEST_PROMPT
        else:
            prompt_template = FEATURE_REQUEST_PROMPT

        # Create LLMClient from environment
        llm_client = LLMClient.from_env()

        return cls(
            llm_client=llm_client,
            prompt_template=prompt_template,
            max_tokens=int(ig_cfg.get("max_tokens", 512)),
            body_excerpt_limit=int(ig_cfg.get("body_excerpt_limit", 1200)),
            workers=int(ig_cfg.get("workers", 1)),
            overwrite_existing=bool(ig_cfg.get("overwrite_existing", False)),
        )

    def generate_issues(self, entity_root: Path) -> List[Path]:
        """Generate issues for all entity.json files under a directory tree."""
        entity_files = sorted(entity_root.rglob("entity.json"))
        issue_paths: List[Path] = []

        if not entity_files:
            self.logger.warning("No entity.json files found under %s", entity_root)
            return issue_paths

        if self.workers > 1:
            with ThreadPoolExecutor(max_workers=self.workers) as executor:
                future_to_file = {
                    executor.submit(self.generate_issue_for_entity, entity_file): entity_file
                    for entity_file in entity_files
                }
                for future in as_completed(future_to_file):
                    entity_file = future_to_file[future]
                    try:
                        issue_paths.append(future.result())
                    except Exception as exc:  # pragma: no cover - defensive
                        self.logger.error(
                            "Failed to generate issue for %s: %s", entity_file, exc
                        )
        else:
            for entity_file in entity_files:
                try:
                    issue_paths.append(self.generate_issue_for_entity(entity_file))
                except Exception as exc:  # pragma: no cover - defensive
                    self.logger.error("Failed to generate issue for %s: %s", entity_file, exc)

        return issue_paths

    def generate_issue_for_entity(self, entity_file: Path) -> Path:
        """Generate a single issue for the provided entity.json file."""
        entity_data = self._load_entity_file(entity_file)
        prompt = self._build_prompt(entity_data)
        issue_path = entity_file.parent / "issue.txt"

        if issue_path.exists() and not self.overwrite_existing:
            self.logger.info(
                "Issue already exists for %s, skipping (overwrite_existing=False)",
                entity_data.get("name"),
            )
            return issue_path

        issue_text = self.llm_client.simple_complete(
            prompt,
            max_tokens=self.max_tokens,
        )

        issue_path.write_text(issue_text, encoding="utf-8")
        self.logger.info("Issue generated for %s -> %s", entity_data.get("name"), issue_path)
        return issue_path

    def _build_prompt(self, entity: Dict[str, Any]) -> str:
        body_content = entity.get("body_content") or entity.get("src_code") or ""
        body_excerpt = body_content.strip()
        if len(body_excerpt) > self.body_excerpt_limit:
            body_excerpt = body_excerpt[: self.body_excerpt_limit] + "... (truncated)"

        docstring = entity.get("docstring") or "No docstring provided."
        signature_content = entity.get("signature_content") or entity.get("signature") or ""

        return self.prompt_template.format(
            repo_name=entity.get("repo_name", "unknown-repo"),
            entity_name=entity.get("name", "unknown-entity"),
            file_path=entity.get("file_path", ""),
            start_line=entity.get("start_line", -1),
            signature_content=signature_content,
            docstring=docstring,
            body_excerpt=body_excerpt,
        )

    def _load_entity_file(self, entity_file: Path) -> Dict[str, Any]:
        with entity_file.open("r", encoding="utf-8") as f:
            return json.load(f)
