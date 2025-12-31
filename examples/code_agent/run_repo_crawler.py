#!/usr/bin/env python
"""
Example script for running the Repository Crawler.

This script demonstrates how to use the RepoCrawler class to search, filter,
and download GitHub repositories based on configuration in code_agent.yaml.

Usage:
    python examples/run_repo_crawler.py
    
    # With custom config
    python examples/run_repo_crawler.py --config path/to/config.yaml
    
    # With verbose logging
    python examples/run_repo_crawler.py --verbose
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_gem.agents.code_agent.repo_crawler import RepoCrawler


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="GitHub Repository Crawler",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default config
  python examples/run_repo_crawler.py
  
  # Run with custom config
  python examples/run_repo_crawler.py --config my_config.yaml
  
  # Enable verbose logging
  python examples/run_repo_crawler.py --verbose
  
Environment Variables:
  GITHUB_TOKEN    GitHub personal access token (recommended for higher rate limits)
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
        "--search-only",
        action="store_true",
        help="Only search and display results, don't download",
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
        crawler = RepoCrawler.load_config_from_yaml(args.config)
        
        if args.search_only:
            # Search only
            logger.info("Running in search-only mode (no downloads)")
            repos = crawler.search_and_filter_repos()
            
            if repos:
                print("\n" + "="*80)
                print(f"Found {len(repos)} repositories:")
                print("="*80)
                for i, repo in enumerate(repos, 1):
                    print(f"\n{i}. {repo['full_name']}")
                    print(f"   ⭐ Stars: {repo['stars']}")
                    print(f"   📝 Language: {repo['language']}")
                    print(f"   📊 Estimated lines: {repo.get('estimated_lines', 'N/A')}")
                    print(f"   🧪 Has pytest: {repo.get('has_pytest', 'N/A')}")
                    print(f"   🔗 URL: {repo['html_url']}")
                    if repo.get('description'):
                        print(f"   📄 {repo['description'][:100]}...")
            else:
                print("\nNo repositories found matching the criteria.")
        else:
            # Full crawl workflow
            crawler.crawl()
            print("\n✅ Crawl complete! Check the output directory for results.")
        
    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
        return 1
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        return 1
    except KeyboardInterrupt:
        logger.warning("\nCrawl interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Error running crawler: {e}", exc_info=args.verbose)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
