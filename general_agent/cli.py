from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .llm import LLMClient
from .synthesis import EnvironmentSynthesizer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatic environment and task synthesis agent supporting local vLLM or OpenAI-compatible APIs."
    )
    parser.add_argument(
        "--category",
        required=True,
        help="Task category, e.g., 'plan a travel itinerary'",
    )
    parser.add_argument(
        "--sandbox",
        default="sandbox/demo",
        help="Sandbox directory to store database and generated outputs",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of difficulty refinement rounds",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip auto execution and verification (for debugging)",
    )
    parser.add_argument(
        "--max-validation-rounds",
        type=int,
        default=2,
        help="Maximum repair attempts when validation fails",
    )
    parser.add_argument(
        "--use-sandbox-fusion",
        action="store_true",
        default=True,
        help="Use SandboxFusion for secure code execution (default: enabled)",
    )
    parser.add_argument(
        "--no-sandbox-fusion",
        action="store_false",
        dest="use_sandbox_fusion",
        help="Disable SandboxFusion",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    llm = LLMClient.from_env()
    synthesizer = EnvironmentSynthesizer(
        llm=llm, max_validation_rounds=args.max_validation_rounds
    )

    bundles = synthesizer.synthesize(
        category=args.category,
        sandbox=Path(args.sandbox),
        rounds=args.rounds,
        validate=not args.no_validate,
        use_sandbox_fusion=args.use_sandbox_fusion,
    )

    print(f"Synthesized {len(bundles)} task(s):")
    for bundle in bundles:
        print(f"- [{bundle.difficulty}] {bundle.name}: {bundle.description}")


if __name__ == "__main__":
    main()

