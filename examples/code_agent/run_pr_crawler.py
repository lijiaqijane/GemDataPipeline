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
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_gem.agents.code_agent.pr_crawler import PRCrawler, PRCrawlerConfig


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GitHub PR Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python examples/code_agent/run_pr_crawler.py
  
  # Run with custom config
  python examples/code_agent/run_pr_crawler.py --config my_config.yaml
  
  # Enable verbose logging
  python examples/code_agent/run_pr_crawler.py --verbose
  
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
        crawler = PRCrawler.load_config_from_yaml(args.config)
        
        
        logger.info(f"Will crawl PRs from {len(crawler.config.repos)} repositories:")
        for repo in crawler.config.repos:
            logger.info(f"  - {repo}")
        
        # Crawl PRs and filter valid PRs
        logger.info("Starting PR crawling...")
        stats = crawler.crawl_all()
        
        # Display statistics
        print("\n" + str(stats))
        
        if stats.total_prs == 0:
            logger.warning("No PRs were crawled. Check logs for errors.")
            return 1
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\nCrawling interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error running PR crawler: {e}", exc_info=args.verbose)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
