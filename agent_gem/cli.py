from __future__ import annotations

import argparse
import logging
from pathlib import Path

import coloredlogs
import dotenv

from agent_gem.agents.code_agent.cli import add_code_synthesize_subparser, handle_code_synthesize
from agent_gem.agents.general_agent.cli import add_synthesize_subparser, handle_synthesize
from agent_gem.agents.search_agent.cli import add_search_synthesize_subparser, handle_search_synthesize
from agent_gem.generator import EnvironmentGenerator, GenerationRequest
from agent_gem.llm import LLMClient

dotenv.load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_gem", description="Generative agentic environment generator."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Original generate command
    gen_parser = subparsers.add_parser("generate", help="Generate tasks using agent pipelines")
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

    add_synthesize_subparser(subparsers)

    # add_search_synthesize_subparser(subparsers)

    add_code_synthesize_subparser(subparsers)

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
        handle_synthesize(args)
    elif args.command == "search_synthesize":
        handle_search_synthesize(args)
    elif args.command == "code_synthesize":
        handle_code_synthesize(args)
    else:
        _handle_generate(args)


if __name__ == "__main__":
    main()
