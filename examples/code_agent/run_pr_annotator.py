#!/usr/bin/env python
"""
Example script for running the PR Annotator.

This script demonstrates how to use the PRAnnotator class to annotate PRs
using LLM based on configuration in code_agent.yaml.

Usage:
    python examples/code_agent/run_pr_annotator.py
    
    # With custom config
    python examples/code_agent/run_pr_annotator.py --config path/to/config.yaml
    
    # With verbose logging
    python examples/code_agent/run_pr_annotator.py --verbose
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_gem.agents.code_agent.pr_annotator import PRAnnotator, PRAnnotatorConfig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="PR Annotator - Label PRs using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python examples/code_agent/run_pr_annotator.py
  
  # Run with custom config
  python examples/code_agent/run_pr_annotator.py --config my_config.yaml
  
  # Enable verbose logging
  python examples/code_agent/run_pr_annotator.py --verbose
  
  # Annotate specific file
  python examples/code_agent/run_pr_annotator.py --input-file taskdb/code_agent/prs/repo-prs-valid.jsonl
  
Environment Variables:
  LLM_PROVIDER      LLM provider (deepseek, volcano, openai, vllm)
  VOLCANO_API_KEY   API key for Volcano/Deepseek
  OPENAI_API_KEY    API key for OpenAI
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="examples/config/code_agent.yaml",
        help="Path to configuration YAML file (default: examples/config/code_agent.yaml)",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to input PR JSONL file (overrides config)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Path to directory containing PR JSONL files (overrides config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Path to output directory (overrides config)",
    )
    parser.add_argument(
        "--max-prs",
        type=int,
        help="Maximum number of PRs to annotate (overrides config)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Do not resume from existing annotations",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from: {args.config}")
        annotator = PRAnnotator.load_config_from_yaml(args.config)
        
        # Override config with command line arguments
        if args.input_file:
            annotator.config.input_file = args.input_file
            annotator.config.input_dir = None  # Clear input_dir if input_file is specified
        if args.input_dir:
            annotator.config.input_dir = args.input_dir
        if args.output_dir:
            annotator.config.output_dir = args.output_dir
            annotator.output_dir = Path(args.output_dir)
            annotator.output_dir.mkdir(parents=True, exist_ok=True)
        if args.max_prs:
            annotator.config.max_prs = args.max_prs
        if args.no_resume:
            annotator.config.resume = False
        
        # Display configuration
        logger.info("PR Annotator Configuration:")
        if annotator.config.input_file:
            logger.info(f"  Input file: {annotator.config.input_file}")
        if annotator.config.input_dir:
            logger.info(f"  Input directory: {annotator.config.input_dir}")
        logger.info(f"  Output directory: {annotator.config.output_dir}")
        if annotator.config.max_prs:
            logger.info(f"  Max PRs: {annotator.config.max_prs}")
        logger.info(f"  Resume: {annotator.config.resume}")
        logger.info(f"  LLM temperature: {annotator.config.llm_temperature}")
        logger.info(f"  LLM max tokens: {annotator.config.llm_max_tokens}")
        
        # Annotate PRs
        logger.info("Starting PR annotation...")
        stats = annotator.annotate_all()
        
        # Display statistics
        print("\n" + "=" * 80)
        print("📊 Annotation Statistics")
        print("=" * 80)
        print(f"Total PRs processed: {stats['total_prs']}")
        print(f"Successfully annotated: {stats['annotated_prs']}")
        print(f"Skipped (already annotated): {stats['skipped_prs']}")
        print(f"Failed: {stats['failed_prs']}")
        print("=" * 80)
        
        if stats['total_prs'] == 0:
            logger.warning("No PRs were processed. Check logs for errors.")
            return 1
        
        if stats['annotated_prs'] == 0 and stats['skipped_prs'] == 0:
            logger.warning("No PRs were annotated. Check logs for errors.")
            return 1
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\nAnnotation interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error running PR annotator: {e}", exc_info=args.verbose)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
