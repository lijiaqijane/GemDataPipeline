from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from agent_gem.agents import SearchAgent
from agent_gem.generator import GenerationRequest
from agent_gem.llm import LLMClient

logger = logging.getLogger(__name__)


def add_search_synthesize_subparser(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Add search_synthesize subcommand to argument parser.

    Args:
        subparsers: Argument parser subparsers object
    """
    search_parser = subparsers.add_parser(
        "search_synthesize", help="Generate tasks and answers using search agent"
    )
    search_parser.add_argument(
        "--domain",
        nargs="+",
        required=True,
        help="Domain(s) to search for entities (can specify multiple)",
    )
    search_parser.add_argument(
        "--num_entities_each_domain",
        type=int,
        default=1,
        help="Number of entities to search for each domain",
    )
    search_parser.add_argument(
        "--num_tasks_each_entity",
        type=int,
        default=1,
        help="Number of tasks to generate for each entity",
    )
    search_parser.add_argument(
        "--num_answer_agent",
        type=int,
        default=1,
        help="Number of answer agents to use",
    )
    search_parser.add_argument(
        "--search_depth",
        type=int,
        default=1,
        help="Search depth for each entity",
    )
    search_parser.add_argument(
        "--search_breadth",
        type=int,
        default=1,
        help="Search breadth for each entity",
    )
    search_parser.add_argument(
        "--require_all_incorrect",
        action="store_true",
        default=False,
        help="Require all candidates to be incorrect",
    )
    search_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (JSON format). If not specified, prints to stdout",
    )


def handle_search_synthesize(args: argparse.Namespace) -> None:
    """Handle search_synthesize command execution.

    Args:
        args: Parsed command-line arguments
    """
    try:
        logger.info("Initializing LLM client and search agent")
        llm = LLMClient.from_env()
        search_agent = SearchAgent(llm=llm)

        request = GenerationRequest(
            agent_type="search_agent",
            domain=args.domain,
            num_entities_each_domain=args.num_entities_each_domain,
            num_tasks_each_entity=args.num_tasks_each_entity,
            num_answer_agent=args.num_answer_agent,
            search_depth=args.search_depth,
            search_breadth=args.search_breadth,
            require_all_incorrect=args.require_all_incorrect,
        )

        logger.info("Starting generation process")
        results = search_agent.generate(request)

        # Output results
        if args.output:
            logger.info(f"Writing results to {args.output}")
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Successfully wrote {len(results)} results to {args.output}")
        else:
            # Print to stdout in JSON format for better readability
            print(json.dumps(results, indent=2, ensure_ascii=False))

        logger.info(f"Generation completed. Generated {len(results)} question-answer pairs")

    except Exception as e:
        logger.error(f"Error during search synthesis: {e}", exc_info=True)
        sys.exit(1)
