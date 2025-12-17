#!/usr/bin/env python3
"""
Example script demonstrating CodeAgent usage for training data generation.

This script shows how to:
1. Initialize CodeAgent with an LLMClient
2. Generate training data from a GitHub repository
3. Access and inspect the generated training data
"""

import logging
from pathlib import Path

import coloredlogs

from agent_gem.llm import LLMClient
from agent_gem.agents import CodeAgent
from agent_gem.generator import GenerationRequest


def setup_logging():
    """Configure logging with colored output."""
    coloredlogs.install(
        level=logging.INFO,
        fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level_styles={
            "debug": {"color": "cyan"},
            "info": {"color": "white"},
            "warning": {"color": "yellow", "bold": True},
            "error": {"color": "red", "bold": True},
        },
    )


def main():
    """Main example function."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Initialize LLM client from environment
    logger.info("Initializing LLMClient from environment...")
    llm = LLMClient.from_env()

    # Initialize CodeAgent
    logger.info("Initializing CodeAgent...")
    agent = CodeAgent(llm, taskdb_root="taskdb")

    # Example 1: Generate from a popular Python repository
    logger.info("\n" + "=" * 60)
    logger.info("Example 1: Flask Repository")
    logger.info("=" * 60)

    request_flask = GenerationRequest(
        agent_type="code_agent",
        topic="https://github.com/pallets/flask",
        num=1,
        difficulty=2,
        validate=True,
    )

    package = agent.generate(request_flask)
    if package:
        logger.info(f"✓ Successfully generated task: {package.task.task_title}")
        logger.info(f"  - Difficulty: {package.task.difficulty_level}")
        logger.info(f"  - Language: {package.metadata.get('language', 'unknown')}")
        logger.info(f"  - Repository: {package.metadata.get('repo_name', 'unknown')}")

        # Show sample of generated content
        logger.info(f"  - Solution preview: {package.solution[:200]}...")
    else:
        logger.error("✗ Failed to generate task from Flask repository")

    # Example 2: Generate with specific difficulty
    logger.info("\n" + "=" * 60)
    logger.info("Example 2: Django Repository (Higher Difficulty)")
    logger.info("=" * 60)

    request_django = GenerationRequest(
        agent_type="code_agent",
        topic="https://github.com/django/django",
        num=1,
        difficulty=4,
        validate=True,
    )

    package = agent.generate(request_django)
    if package:
        logger.info(f"✓ Successfully generated task: {package.task.task_title}")
        logger.info(f"  - Difficulty: {package.task.difficulty_level}")
    else:
        logger.error("✗ Failed to generate task from Django repository")

    # Example 3: Generate from a custom repository
    logger.info("\n" + "=" * 60)
    logger.info("Example 3: Custom Repository")
    logger.info("=" * 60)

    request_custom = GenerationRequest(
        agent_type="code_agent",
        topic="https://github.com/requests/requests",  # Popular HTTP library
        num=1,
        difficulty=2,
        validate=True,
    )

    package = agent.generate(request_custom)
    if package:
        logger.info(f"✓ Successfully generated task: {package.task.task_title}")
    else:
        logger.error("✗ Failed to generate task from requests repository")

    # Example 4: Access training data
    logger.info("\n" + "=" * 60)
    logger.info("Example 4: Accessing Training Data")
    logger.info("=" * 60)

    training_data_dir = Path("taskdb") / "code_agent"
    if training_data_dir.exists():
        tasks = list(training_data_dir.glob("*/training_data.json"))
        logger.info(f"Found {len(tasks)} training data file(s)")

        for task_file in tasks[:3]:  # Show first 3
            logger.info(f"\n  Training data: {task_file.parent.name}")
            try:
                import json

                data = json.loads(task_file.read_text())
                logger.info(f"    - Bug ID: {data['bug_info'].get('bug_id')}")
                logger.info(f"    - Bug Type: {data['bug_info'].get('bug_type')}")
                logger.info(f"    - Valid Tests: {data['metadata'].get('valid_tests_count')}")
            except Exception as e:
                logger.error(f"    Error reading training data: {e}")
    else:
        logger.info(f"No training data directory found at {training_data_dir}")

    logger.info("\n" + "=" * 60)
    logger.info("Example complete!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
