from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import coloredlogs
import dotenv

from agent_gem.agents import GeneralAgent
from agent_gem.generator import EnvironmentGenerator, GenerationRequest
from agent_gem.llm import LLMClient
from agent_gem.utils import check_sandbox_fusion, validate_environment
from agent_gem.writer import TaskWriter

dotenv.load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_gem", description="Generative agentic environment generator."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Original generate command
    gen_parser = subparsers.add_parser(
        "generate", help="Generate tasks using agent pipelines"
    )
    gen_parser.add_argument(
        "--agent-type",
        default="general_agent",
        choices=[
            "search_agent",
            "code_agent",
            "code_interpreter_agent",
            "general_agent",
        ],
        help="Agent pipeline to invoke.",
    )
    gen_parser.add_argument(
        "--topic",
        default=None,
        help="Optional domain/topic for the generated task; if omitted, the agent will pick one.",
    )
    gen_parser.add_argument("--num", type=int, default=1, help="Number of tasks to generate.")
    gen_parser.add_argument(
        "--difficulty",
        default=3,
        type=int,
        help="Target difficulty level (int).",
    )
    gen_parser.add_argument(
        "--taskdb-root",
        default="taskdb",
        help="Root directory for generated task taskdb.",
    )
    gen_parser.add_argument("--no-validate", action="store_true", help="Skip schema validation guards.")
    gen_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging to surface agent thinking steps.",
    )

    # New synthesize command (compatible with general_agent CLI)
    synth_parser = subparsers.add_parser(
        "synthesize", help="Synthesize environment and tasks (compatible with general_agent)"
    )
    synth_parser.add_argument(
        "--category",
        required=True,
        help="Task category, e.g., 'plan a travel itinerary'",
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
        help="Merge with existing tasks.json instead of overwriting (default: overwrite)",
    )
    synth_parser.add_argument(
        "--no-merge",
        action="store_false",
        dest="merge",
        help="Overwrite existing tasks.json (default behavior)",
    )
    synth_parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Maximum tokens for LLM generation (default: 10000)",
    )

    return parser


def _handle_generate(args: argparse.Namespace) -> None:
    llm = LLMClient.from_env()
    generator = EnvironmentGenerator(llm, taskdb=Path(args.taskdb_root))
    request = GenerationRequest(
        agent_type=args.agent_type,
        topic=args.topic,
        num=args.num,
        difficulty=args.difficulty,
        validate=not args.no_validate,
    )
    packages = generator.generate(request)

    print(f"Generated {len(packages)} task(s) with agent={args.agent_type}:")
    for pkg in packages:
        print(f"- {pkg.task.summary()} @ {pkg.task_path}")


def _handle_synthesize(args: argparse.Namespace) -> None:
    """Handle synthesize command (compatible with general_agent CLI)."""
    # Validate environment before starting
    if args.use_sandbox_fusion:
        sandbox_url = os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080")
        logging.info("Checking SandboxFusion service...")
        if not check_sandbox_fusion(sandbox_url):
            logging.error("SandboxFusion service unavailable (%s)", sandbox_url)
            logging.error("Please start SandboxFusion service first, then retry.")
            sys.exit(1)
        logging.info("SandboxFusion service available")

    is_valid, error_msg = validate_environment(use_sandbox_fusion=args.use_sandbox_fusion)
    if not is_valid:
        logging.error("Environment configuration validation failed: %s", error_msg)
        sys.exit(1)

    # Setup sandbox directory
    sandbox_path = Path(args.sandbox)
    sandbox_path.mkdir(parents=True, exist_ok=True)

    # Initialize components
    llm = LLMClient.from_env()
    writer = TaskWriter(root=sandbox_path)
    agent = GeneralAgent(llm, taskdb_root=str(sandbox_path))

    # Generate tasks
    category = args.category

    logging.info(f"Starting synthesis: category={category}, rounds={args.rounds}")

    # Generate tasks with all refinement rounds
    # GeneralAgent.generate() handles multiple rounds internally
    request = GenerationRequest(
        agent_type="general_agent",
        topic=category,
        num=1,
        difficulty=args.rounds,  # Final difficulty level
        validate=not args.no_validate,
        use_sandbox_fusion=args.use_sandbox_fusion,
        max_refine_rounds=args.rounds,  # Number of rounds (initial + refinements)
        max_validation_rounds=args.max_validation_rounds,
        persist_result=True,
        max_tokens=getattr(args, "max_tokens", 10000),
    )

    # Generate the task package (includes all refinement rounds internally)
    package = agent.generate(request)
    if not package:
        logging.error("Failed to generate task package")
        sys.exit(1)

    # Extract records from writer (which loads from db.json)
    records = writer.records

    # For quadruple format with multiple rounds, we need to generate each round separately
    # Since GeneralAgent.generate() only returns the final package after all refinements,
    # we'll generate packages for each round
    packages = []
    
    # Generate initial task (round 1, difficulty 1)
    initial_request = GenerationRequest(
        agent_type="general_agent",
        topic=category,
        num=1,
        difficulty=1,
        validate=not args.no_validate,
        max_refine_rounds=1,  # Only initial task, no refinement
        max_validation_rounds=args.max_validation_rounds,
        persist_result=False,
    )
    initial_package = agent.generate(initial_request)
    if initial_package:
        packages.append(initial_package)
        current_package = initial_package
        
        # Generate refined tasks for remaining rounds (2 to args.rounds)
        for round_idx in range(1, args.rounds):
            from agent_gem.agents.base import TaskContext
            from agent_gem.sandbox import SandboxExecutor
            
            # Create context for this refinement round
            refine_request = GenerationRequest(
                agent_type="general_agent",
                topic=category,
                num=1,
                difficulty=round_idx + 1,
                validate=not args.no_validate,
                max_refine_rounds=1,
                max_validation_rounds=args.max_validation_rounds,
                persist_result=False,
            )
            ctx = TaskContext(task_id=current_package.task.task_id, request=refine_request)
            ctx.current_difficulty = round_idx + 1
            
            # Setup sandbox for validation
            sandbox_dir = Path(writer.task_dir(current_package.task.task_id, "general_agent"), "_sandbox")
            sandbox = SandboxExecutor(sandbox_dir=sandbox_dir)
            agent._configure_sandbox(sandbox)
            agent._register_task_tools(current_package.task.tool_set, sandbox, ctx)
            
            # Refine the task
            refined = agent._refine_task(
                previous=current_package,
                records=records,
                tool_specs=current_package.task.tool_set,
                ctx=ctx,
                target_difficulty=round_idx + 1,
            )
            
            # Ensure substantive and validate
            refined = agent._ensure_substantive_task(current_package.task.tool_set, refined, ctx)
            if not args.no_validate:
                refined = agent._ensure_valid(refine_request, refined, ctx, sandbox, records)
            
            packages.append(refined)
            current_package = refined

    # Persist in quadruple format
    writer.persist_quadruple_format(
        category=category,
        records=records,
        packages=packages,
        output_path=sandbox_path / "tasks.json",
        merge=getattr(args, "merge", False),  # Default to False (overwrite)
    )

    print(f"Synthesized {len(packages)} task(s):")
    for pkg in packages:
        print(f"- [{pkg.task.difficulty_level}] {pkg.task.task_title}: {pkg.task.task_content[:100]}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to generate command for backward compatibility
    if not args.command:
        args.command = "generate"
        # Re-parse with default command
        args = parser.parse_args(argv)

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    coloredlogs.install(
        level=log_level,
        fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level_styles={
            "debug": {"color": "cyan"},
            "info": {"color": "white"},
            "warning": {"color": "yellow", "bold": True},
            "error": {"color": "red", "bold": True},
            "critical": {"color": "red", "bold": True, "background": "black"},
        },
        field_styles={
            "asctime": {"color": "white"},
            "levelname": {"color": "blue", "bold": True},
            "name": {"color": "green"},
        },
    )

    if args.command == "synthesize":
        _handle_synthesize(args)
    else:
        _handle_generate(args)


if __name__ == "__main__":
    main()
