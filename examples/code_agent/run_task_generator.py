#!/usr/bin/env python3
"""
Task Generator Example Script

Reads extracted entities (entity.json + issue.txt) and generates training tasks.

Usage:
    python examples/code_agent/run_task_generator.py \
        --config examples/config/code_agent.yaml \
        [--entity-dir taskdb/code_agent/extracted_entities] \
        [--output FILE_NAME.json] \
        [--output-dir taskdb/code_agent/tasks] \
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

from agent_gem.agents.code_agent.task_generator import TaskGenerator


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=("%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s"),
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate training tasks from extracted entities",
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
        "--output",
        type=str,
        default=None,
        help="Filename for generated tasks (optional; default uses timestamp)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store generated tasks (overrides config)",
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
        logger.info("Loading TaskGenerator from config %s", args.config)
        generator = TaskGenerator.load_config_from_yaml(args.config)

        entity_dir = Path(args.entity_dir or generator.config.entity_dir)
        if not entity_dir.exists():
            logger.error("Entity directory not found: %s", entity_dir)
            return 1

        if args.output_dir:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            generator.config.save_dir = str(output_dir)
            generator.output_dir = output_dir
        else:
            output_dir = generator.output_dir

        logger.info("Generating tasks from %s", entity_dir)
        tasks = generator.generate_tasks_from_directory(entity_dir)
        if not tasks:
            logger.warning("No tasks generated; check that entities and issues exist")
            return 0

        output_path = generator.save_tasks(tasks, filename=args.output)

        summary = {
            "entity_dir": str(entity_dir),
            "tasks_generated": len(tasks),
            "tasks_file": str(output_path) if output_path else None,
            "output_dir": str(output_dir),
        }
        summary_path = output_dir / "task_generation_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        logger.info("Task generation complete. Summary saved to %s", summary_path)
        return 0

    except Exception as exc:  # pragma: no cover - script entry
        logger.error("Task generation failed: %s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
