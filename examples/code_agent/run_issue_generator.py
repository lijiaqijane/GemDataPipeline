#!/usr/bin/env python3
"""
Issue Generator Example Script

Reads extracted entity folders and generates feature request issues via LLM.

Usage:
    python examples/code_agent/run_issue_generator.py \
        --config examples/config/code_agent.yaml \
        [--entity-dir taskdb/code_agent/extracted_entities] \
        [--verbose]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_gem.agents.code_agent.issue_generator import IssueGenerator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=("%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate feature request issues for extracted entities",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="examples/config/code_agent.yaml",
        help="Path to configuration file (default: examples/config/code_agent.yaml)",
    )
    parser.add_argument(
        "--entity-dir",
        type=str,
        default=None,
        help="Directory containing extracted entities (overrides config)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )

    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)

    try:
        logger.info("Loading IssueGenerator from config %s", args.config)
        generator = IssueGenerator.load_config_from_yaml(args.config)

        # Override entity_dir if provided
        if args.entity_dir:
            entity_dir = Path(args.entity_dir)
        else:
            entity_dir = Path(
                "taskdb/code_agent/extracted_entities"
            )

        if not entity_dir.exists():
            logger.error("Entity directory not found: %s", entity_dir)
            return 1

        logger.info("Generating issues from %s", entity_dir)
        issue_paths = generator.generate_issues(entity_dir)

        summary = {
            "entity_dir": str(entity_dir),
            "issues_generated": len(issue_paths),
            "issue_paths": [str(p) for p in issue_paths],
        }

        summary_path = entity_dir / "issue_generation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Issue generation complete. Summary saved to %s", summary_path)
        return 0

    except Exception as exc:  # pragma: no cover - script entry
        logger.error("Issue generation failed: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
