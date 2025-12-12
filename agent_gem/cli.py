from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import coloredlogs
import dotenv

from agent_gem.env_generator import EnvironmentGenerator, GenerationRequest
from agent_gem.llm import LLMClient

dotenv.load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_gem", description="Generative agentic environment generator."
    )
    parser.add_argument(
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
    parser.add_argument(
        "--topic",
        default=None,
        help="Optional domain/topic for the generated task; if omitted, the agent will pick one.",
    )
    parser.add_argument("--num", type=int, default=1, help="Number of tasks to generate.")
    parser.add_argument("--difficulty", default="Medium", help="Target difficulty label.")
    parser.add_argument(
        "--sandbox-root",
        default="sandbox",
        help="Root directory for generated sandboxes.",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip schema validation guards.")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging to surface agent thinking steps.",
    )

    return parser


def _handle_generate(args: argparse.Namespace) -> None:
    llm = LLMClient.from_env()
    generator = EnvironmentGenerator(llm, sandbox_root=Path(args.sandbox_root))
    request = GenerationRequest(
        agent_type=args.agent_type,
        topic=args.topic,
        count=args.num,
        difficulty=args.difficulty,
        sandbox_root=Path(args.sandbox_root),
        validate=not args.no_validate,
    )
    packages = generator.generate(request)

    print(f"Generated {len(packages)} task(s) with agent={args.agent_type}:")
    for pkg in packages:
        print(f"- {pkg.task.summary()} @ {pkg.sandbox_path}")


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args()
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

    _handle_generate(args)


if __name__ == "__main__":
    main()
