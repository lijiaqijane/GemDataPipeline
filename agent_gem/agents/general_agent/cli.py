from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from agent_gem.agents import GeneralAgent
from agent_gem.generator import GenerationRequest
from agent_gem.llm import LLMClient
from agent_gem.utils import check_sandbox_fusion, validate_environment
from agent_gem.writer import TaskWriter


def add_synthesize_subparser(subparsers: argparse._SubParsersAction) -> None:
    synth_parser = subparsers.add_parser(
        "synthesize", help="Synthesize environment and tasks (compatible with general_agent)"
    )
    synth_parser.add_argument(
        "--category",
        default=None,
        help="Task category, e.g., 'plan a travel itinerary'. If not specified and --num-categories > 1, will extract from task_category.json",
    )
    synth_parser.add_argument(
        "--sandbox",
        default="sandbox/demo",
        help="Sandbox directory to store database and generated outputs",
    )
    synth_parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of difficulty refinement rounds",
    )
    synth_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip auto execution and verification (for debugging)",
    )
    synth_parser.add_argument(
        "--max-validation-rounds",
        type=int,
        default=2,
        help="Maximum repair attempts when validation fails",
    )
    synth_parser.add_argument(
        "--use-sandbox-fusion",
        action="store_true",
        default=True,
        help="Use SandboxFusion for secure code execution (default: enabled)",
    )
    synth_parser.add_argument(
        "--no-sandbox-fusion",
        action="store_false",
        dest="use_sandbox_fusion",
        help="Disable SandboxFusion",
    )
    synth_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )
    synth_parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens for LLM generation",
    )
    synth_parser.add_argument(
        "--num",
        type=int,
        default=1,
        help="Number of tasks to generate per category",
    )
    synth_parser.add_argument(
        "--num-categories",
        type=int,
        default=1,
        help="Number of categories to process (if category is not specified, will extract from task_category.json)",
    )


def _extract_categories_from_json(json_file: str, num_categories: int, start_index: int = 0) -> list[str]:
    """Extract scenarios from task_category.json"""
    import json
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        scenarios = []
        for role in data.get('user_roles', []):
            for scenario in role.get('scenarios', []):
                scenarios.append(scenario.get('scenario', ''))
        if len(scenarios) == 0:
            for role in data.get('domains', []):
                for scenario in role.get('scenarios', []):
                    scenarios.append(scenario.get('scenario', ''))
        # Return requested number of scenarios starting from start_index
        end_index = min(start_index + num_categories, len(scenarios))
        return scenarios[start_index:end_index] if scenarios else []
    except Exception as e:
        logging.error(f"Failed to extract categories from {json_file}: {e}")
        return []


def _generate_category_slug(category: str) -> str:
    """Convert category name to a URL-friendly slug for use in task_id"""
    import re
    # Convert to lowercase, replace spaces and special chars with hyphens
    slug = category.lower()
    slug = re.sub(r'[^\w\s-]', '', slug)  # Remove special characters
    slug = re.sub(r'[-\s]+', '-', slug)  # Replace spaces and multiple hyphens with single hyphen
    slug = slug.strip('-')  # Remove leading/trailing hyphens
    return slug


def handle_synthesize(args: argparse.Namespace) -> None:
    if not args.use_sandbox_fusion:
        logging.error("SandboxFusion is required for synthesis; local sandbox execution is disabled.")
        logging.error("Please enable SandboxFusion or remove --no-sandbox-fusion.")
        sys.exit(1)

    sandbox_url = os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080")
    if not check_sandbox_fusion(sandbox_url):
        logging.error("SandboxFusion service unavailable (%s)", sandbox_url)
        sys.exit(1)

    is_valid, error_msg = validate_environment(use_sandbox_fusion=args.use_sandbox_fusion)
    if not is_valid:
        logging.error("Environment validation failed: %s", error_msg)
        sys.exit(1)

    # Determine categories to process
    num_categories = getattr(args, "num_categories", 1)
    categories = []
    
    if args.category:
        # Use specified category
        categories = [args.category]
    elif num_categories > 1:
        # Extract from JSON
        task_category_file = os.getenv("TASK_CATEGORY_FILE", "task_category_generated.json")
        start_index = int(os.getenv("SCENARIO_INDEX", "0"))
        print(f"start_index: {start_index}")
        categories = _extract_categories_from_json(task_category_file, num_categories, start_index)
        if not categories:
            logging.error(f"Failed to extract categories from {task_category_file}")
            sys.exit(1)
    else:
        logging.error("Either --category must be specified or --num-categories > 1 with task_category.json available")
        sys.exit(1)

    # Setup sandbox directory
    sandbox_path = Path(args.sandbox)
    sandbox_path.mkdir(parents=True, exist_ok=True)

    # Initialize writer and agent (shared across all categories)
    writer = TaskWriter(root=sandbox_path)
    llm = LLMClient.from_env()
    agent = GeneralAgent(llm, taskdb_root=str(sandbox_path))

    num_tasks = getattr(args, "num", 1)
    successful_task_ids = []
    
    # Process each category
    for category_idx, category in enumerate(categories):
        logging.info(f"Processing category {category_idx + 1}/{len(categories)}: {category}")
        
        # Generate category slug for task_id prefix
        category_slug = _generate_category_slug(category)

        # Process each task in the category
        for task_idx in range(num_tasks):
            task_id_prefix = f"{category_slug}-task-{task_idx + 1}"
            logging.info(f"Generating task {task_idx + 1}/{num_tasks} for category: {category} (ID: {task_id_prefix})")

            # Check if tasks.json already exists for this task
            # task_id_prefix is used as the actual task_id in agent.generate()
            task_dir = writer.task_dir(task_id_prefix, "general_agent")
            tasks_json_path = task_dir / "tasks.json"
            if tasks_json_path.exists():
                logging.info(f"Skipping task {task_id_prefix} - tasks.json already exists at {tasks_json_path}")
                successful_task_ids.append(task_id_prefix)
                continue

            # Generate the task package (includes all refinement rounds internally)
            # agent.generate() will automatically handle all difficulty levels and persist results
            request = GenerationRequest(
                agent_type="general_agent",
                topic=category,
                num=1,  # Generate one task at a time
                difficulty=args.rounds,  # Final difficulty level
                validate=not args.no_validate,
                use_sandbox_fusion=args.use_sandbox_fusion,
                max_refine_rounds=args.rounds,  # Number of rounds (initial + refinements)
                max_validation_rounds=args.max_validation_rounds,
                persist_result=True,  # Will persist all difficulty levels via persist_quadruple_format
                max_tokens=getattr(args, "max_tokens", 10000),
                task_id_prefix=task_id_prefix,
            )

            package = agent.generate(request)
            if package:
                successful_task_ids.append(package.task.task_id)
            else:
                logging.warning(f"Failed to generate task package for category: {category}, task: {task_idx + 1}")

    # Read all tasks from persisted tasks.json files for printing summary
    all_tasks_info = []
    for task_id in successful_task_ids:
        task_dir = writer.task_dir(task_id, "general_agent")
        tasks_json_path = task_dir / "tasks.json"
        if tasks_json_path.exists():
            try:
                with open(tasks_json_path, 'r', encoding='utf-8') as f:
                    tasks_data = json.load(f)
                # Extract all tasks from the persisted file
                for task_entry in tasks_data.get("tasks", []):
                    task_info = task_entry.get("task", {})
                    all_tasks_info.append({
                        "difficulty": task_info.get("difficulty", "unknown"),
                        "title": task_info.get("name", "Unknown"),
                        "content": task_info.get("description", "")[:100],
                    })
            except Exception as e:
                logging.debug(f"Failed to read tasks.json for {task_id}: {e}")

    # Print summary of successfully generated tasks (all difficulty levels)
    print(f"Synthesized {len(all_tasks_info)} task(s) across {len(categories)} category/categories:")
    for task_info in all_tasks_info:
        print(f"- [{task_info['difficulty']}] {task_info['title']}: {task_info['content']}")
