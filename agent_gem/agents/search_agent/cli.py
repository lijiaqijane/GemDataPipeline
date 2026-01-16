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
        required=False,
        help="Domain(s) to search for entities (can specify multiple)",
    )
    search_parser.add_argument(
        "--num_domains",
        type=int,
        default=10,
        help="Number of domains to search for entities",
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
        "--require_all_incorrect",
        action="store_true",
        default=False,
        help="Require all candidates to be incorrect",
    )
    search_parser.add_argument(
        "--search_depth",
        type=int,
        default=2,
        help="Number of expansion iterations for entity sampling and search depth",
    )
    search_parser.add_argument(
        "--search_breadth",
        type=int,
        default=2,
        help="Number of new entities to extract per entity and search breadth",
    )
    search_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (JSON format). If not specified, prints to stdout",
    )
    search_parser.add_argument(
        "--embedding_path",
        type=str,
        default=None,
        help="Embedding model to use for search agent",
    )
    search_parser.add_argument(
        "--faiss_index_path",
        type=str,
        default=None,
        help="Faiss index to use for search agent",
    )
    search_parser.add_argument(
        "--text_mapping_path",
        type=str,
        default=None,
        help="Text mapping to use for search agent",
    )
    search_parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Maximum number of worker threads for parallel processing (default: 4)",
    )
    search_parser.add_argument(
        "--num_iterations",
        type=int,
        default=1,
        help="Number of iterations for entity sampling and search depth",
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
            num_domains=args.num_domains,
            num_entities_each_domain=args.num_entities_each_domain,
            num_tasks_each_entity=args.num_tasks_each_entity,
            num_iterations=args.num_iterations,
            require_all_incorrect=args.require_all_incorrect,
            search_depth=args.search_depth,
            search_breadth=args.search_breadth,
            embedding_path=args.embedding_path,
            faiss_index_path=args.faiss_index_path,
            text_mapping_path=args.text_mapping_path,
            max_workers=args.max_workers,
        )

        logger.info("Starting generation process")

        if args.output:
            logger.info(f"Results will be incrementally saved to {args.output}")
            results = search_agent.generate(request, output_file=args.output)
            logger.info(f"Successfully saved {len(results)} results to {args.output}")
        else:
            results = search_agent.generate(request)
            print(json.dumps(results, indent=2, ensure_ascii=False))

        logger.info(f"Generation completed. Generated {len(results)} question-answer pairs")

    except Exception as e:
        logger.error(f"Error during search synthesis: {e}", exc_info=True)
        sys.exit(1)
