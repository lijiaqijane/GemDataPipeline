from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from agent_gem.agents import GeneralAgent
from agent_gem.generator import GenerationRequest
from agent_gem.llm import LLMClient
from agent_gem.utils import check_sandbox_fusion, validate_environment
from agent_gem.writer import TaskWriter
from agent_gem.agents.general_agent.persist import persist_quadruple_format
from .sandbox import GeneralAgentSandboxExecutor


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
        "--merge",
        action="store_true",
        default=False,
        help="Merge tasks.json with existing content",
    )
    synth_parser.add_argument(
        "--no-merge",
        action="store_false",
        dest="merge",
        help="Overwrite tasks.json instead of merging",
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
        task_category_file = os.getenv("TASK_CATEGORY_FILE", "task_category.json")
        start_index = int(os.getenv("SCENARIO_INDEX", "0"))
        categories = _extract_categories_from_json(task_category_file, num_categories, start_index)
        if not categories:
            logging.error("Failed to extract categories from task_category.json")
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
    all_packages = []
    
    # Process each category
    for category_idx, category in enumerate(categories):
        logging.info(f"Processing category {category_idx + 1}/{len(categories)}: {category}")
        
        # Generate category slug for task_id prefix
        category_slug = _generate_category_slug(category)
        category_packages = []  # Packages for current category

        # Process each task in the category
        for task_idx in range(num_tasks):
            task_id_prefix = f"{category_slug}-task-{task_idx + 1}"
            logging.info(f"Generating task {task_idx + 1}/{num_tasks} for category: {category} (ID: {task_id_prefix})")

            # Generate the task package (includes all refinement rounds internally)
            request = GenerationRequest(
                agent_type="general_agent",
                topic=category,
                num=1,  # Generate one task at a time
                difficulty=args.rounds,  # Final difficulty level
                validate=not args.no_validate,
                use_sandbox_fusion=args.use_sandbox_fusion,
                max_refine_rounds=args.rounds,  # Number of rounds (initial + refinements)
                max_validation_rounds=args.max_validation_rounds,
                persist_result=True,
                max_tokens=getattr(args, "max_tokens", 10000),
                task_id_prefix=task_id_prefix,
            )

            package = agent.generate(request)
            if not package:
                logging.warning(f"Failed to generate task package for category: {category}, task: {task_idx + 1}")
                continue

            # Extract records from writer (which loads from db.json)
            records = writer.records

            # Generate per-round packages for quadruple format output
            packages = []

            initial_request = GenerationRequest(
                agent_type="general_agent",
                topic=category,
                num=1,
                difficulty=1,
                validate=not args.no_validate,
                use_sandbox_fusion=args.use_sandbox_fusion,
                max_refine_rounds=1,
                max_validation_rounds=args.max_validation_rounds,
                persist_result=False,
                task_id_prefix=task_id_prefix,  # Use same task_id for all rounds
            )
            initial_package = agent.generate(initial_request)
            if initial_package:
                packages.append(initial_package)
                current_package = initial_package

                for round_idx in range(1, args.rounds):
                    from agent_gem.agents.base import TaskContext

                    refine_request = GenerationRequest(
                        agent_type="general_agent",
                        topic=category,
                        num=1,
                        difficulty=round_idx + 1,
                        validate=not args.no_validate,
                        max_refine_rounds=1,
                        max_validation_rounds=args.max_validation_rounds,
                        persist_result=False,
                        task_id_prefix=task_id_prefix,  # Use same task_id for all rounds
                    )
                    ctx = TaskContext(task_id=current_package.task.task_id, request=refine_request)
                    ctx.current_difficulty = round_idx + 1

                    sandbox_dir = Path(writer.task_dir(current_package.task.task_id, "general_agent"), "_sandbox")
                    sandbox = GeneralAgentSandboxExecutor(sandbox_dir=sandbox_dir)
                    agent._configure_sandbox(sandbox)
                    agent._register_task_tools(current_package.task.tool_set, sandbox, ctx)

                    refined = agent._refine_task(
                        previous=current_package,
                        records=records,
                        tool_specs=current_package.task.tool_set,
                        ctx=ctx,
                        target_difficulty=round_idx + 1,
                    )

                    refined = agent._ensure_substantive_task(current_package.task.tool_set, refined, ctx)
                    if not args.no_validate:
                        refined = agent._ensure_valid(refine_request, refined, ctx, sandbox, records)

                    packages.append(refined)
                    current_package = refined

            category_packages.extend(packages)
        
        # Persist after each category (merge mode for multiple categories)
        persist_quadruple_format(
            writer,
            category=category,
            records=records,
            packages=category_packages,  # Include all packages from all tasks in this category
            output_path=sandbox_path / "tasks.json",
            merge=(getattr(args, "merge", False) or category_idx > 0),  # Merge if multiple categories
        )
        
        all_packages.extend(category_packages)

    print(f"Synthesized {len(all_packages)} task(s) across {len(categories)} category/categories:")
    for pkg in all_packages:
        print(f"- [{pkg.task.difficulty_level}] {pkg.task.task_title}: {pkg.task.task_content[:100]}")
