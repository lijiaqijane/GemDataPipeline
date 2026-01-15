#!/usr/bin/env python
"""
Example script for running the PR Extractor.

This script demonstrates how to use the PRExtractor class to extract pull request
data from GitHub repositories based on configuration in code_agent.yaml.

Usage:
    python examples/code_agent/run_pr_extractor.py
    
    # With custom config
    python examples/code_agent/run_pr_extractor.py --config path/to/config.yaml
    
    # With verbose logging
    python examples/code_agent/run_pr_extractor.py --verbose
    
    # Override repos from command line
    python examples/code_agent/run_pr_extractor.py --repos scikit-learn/scikit-learn pallets/flask
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_gem.agents.code_agent.pr_extractor import PRExtractor, PRExtractorConfig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GitHub PR Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python examples/code_agent/run_pr_extractor.py
  
  # Run with custom config
  python examples/code_agent/run_pr_extractor.py --config my_config.yaml
  
  # Enable verbose logging
  python examples/code_agent/run_pr_extractor.py --verbose
  
  # Override repositories from command line
  python examples/code_agent/run_pr_extractor.py --repos scikit-learn/scikit-learn pallets/flask
  
  # Extract with max PRs limit
  python examples/code_agent/run_pr_extractor.py --max-pulls 100
  
  # Extract PRs before a cutoff date
  python examples/code_agent/run_pr_extractor.py --cutoff-date 20240101

Environment Variables:
  GITHUB_TOKEN      GitHub personal access token (recommended for higher rate limits)
  GITHUB_TOKENS     Comma-separated list of tokens for parallel processing
        """
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="examples/config/code_agent.yaml",
        help="Path to configuration YAML file (default: examples/config/code_agent.yaml)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG level) logging",
    )
    parser.add_argument(
        "--repos",
        nargs="+",
        help="Override repositories from command line (format: owner/repo)",
    )
    parser.add_argument(
        "--max-pulls",
        type=int,
        help="Override max_pulls from command line",
    )
    parser.add_argument(
        "--cutoff-date",
        type=str,
        help="Override cutoff_date from command line (format: YYYYMMDD)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory from command line",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PR files",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Enable parallel processing with multiple tokens",
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
        extractor = PRExtractor.load_config_from_yaml(args.config)
        
        # Override config with command line arguments if provided
        if args.repos:
            extractor.config.repos = args.repos
            logger.info(f"Overriding repos with: {args.repos}")
        
        if args.max_pulls is not None:
            extractor.config.max_pulls = args.max_pulls
            logger.info(f"Overriding max_pulls with: {args.max_pulls}")
        
        if args.cutoff_date:
            extractor.config.cutoff_date = args.cutoff_date
            logger.info(f"Overriding cutoff_date with: {args.cutoff_date}")
        
        if args.output_dir:
            extractor.config.output_dir = args.output_dir
            extractor.output_dir = Path(args.output_dir)
            extractor.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Overriding output_dir with: {args.output_dir}")
        
        if args.overwrite:
            extractor.config.overwrite_existing = True
            logger.info("Overwriting existing PR files")
        
        if args.parallel:
            extractor.config.use_parallel = True
            logger.info("Parallel processing enabled")
        
        # Validate repos
        if not extractor.config.repos:
            logger.error("No repositories specified. Please provide repos in config file or via --repos argument")
            return 1
        
        logger.info(f"Will extract PRs from {len(extractor.config.repos)} repositories:")
        for repo in extractor.config.repos:
            logger.info(f"  - {repo}")
        
        # Extract PRs
        logger.info("Starting PR extraction...")
        stats = extractor.extract_all()
        
        # Display statistics
        print("\n" + str(stats))
        
        if stats.total_prs_extracted == 0:
            logger.warning("No PRs were extracted. Check logs for errors.")
            return 1
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\nExtraction interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error running PR extractor: {e}", exc_info=args.verbose)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
