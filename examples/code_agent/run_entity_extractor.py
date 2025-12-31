#!/usr/bin/env python3
"""
Entity Extractor Example Script

This script demonstrates how to extract code entities from repositories crawled by RepoCrawler.
It loads configuration from code_agent.yaml, then processes all repositories in the crawled_repos
directory, extracting code entities (classes, functions, etc.) and saving the results.

Usage:
    python examples/run_entity_extractor.py [--config CONFIG_PATH] [--repo-dir REPO_DIR] [--output-dir OUTPUT_DIR] [--verbose]

Example:
    # Using default configuration from code_agent.yaml
    python examples/run_entity_extractor.py
    
    # With custom configuration and output directory
    python examples/run_entity_extractor.py --config examples/config/code_agent.yaml --output-dir taskdb/extracted_entities
    
    # Verbose output for debugging
    python examples/run_entity_extractor.py --verbose
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_gem.agents.code_agent.entity_extractor import EntityExtractor
import yaml


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format=(
            '%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s'
        )
    )



def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def process_repositories(
    repo_dir: Path,
    entity_extractor: EntityExtractor,
    output_dir: Path,
    verbose: bool = False
) -> dict:
    """
    Process all repositories in the crawled_repos directory.
    
    Args:
        repo_dir: Path to directory containing crawled repositories
        entity_extractor: EntityExtractor instance
        output_dir: Path to save extracted entities
        verbose: Whether to print detailed output
        
    Returns:
        Summary statistics
    """
    logger = logging.getLogger(__name__)
    
    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all repository directories
    if not repo_dir.exists():
        logger.error(f"Repository directory not found: {repo_dir}")
        return {'success': False, 'error': 'repo_dir not found'}
    
    repo_dirs = [d for d in repo_dir.iterdir() if d.is_dir()]
    
    if not repo_dirs:
        logger.warning(f"No repositories found in {repo_dir}")
        return {
            'success': True,
            'total_repos': 0,
            'total_entities': 0,
            'processed_repos': [],
            'errors': []
        }
    
    logger.info(f"Found {len(repo_dirs)} repositories to process")
    
    summary = {
        'extraction_start': datetime.now().isoformat(),
        'total_repos': len(repo_dirs),
        'processed_repos': 0,
        'total_entities': 0,
        'repos': [],
        'errors': []
    }
    
    for idx, repo_path in enumerate(repo_dirs, 1):
        repo_name = repo_path.name
        logger.info(f"[{idx}/{len(repo_dirs)}] Processing: {repo_name}")
        
        try:
            # Extract entities from repository
            result = entity_extractor.extract_entities_from_repo(str(repo_path))
            
            # Save entities to JSON
            output_file = output_dir / f"{repo_name}_entities.json"
            with open(output_file, 'w') as f:
                # Convert CodeEntity objects to dictionaries for JSON serialization
                entities_dict = []
                for entity in result['entities']:
                    entity_data = {
                        'name': entity.name,
                        # 'type': entity.type,
                        'file_path': entity.file_path,
                        'start_line': entity.line_start,
                        'end_line': entity.line_end,
                        'line_count': getattr(entity, 'line_count', entity.line_end - entity.line_start + 1),
                        'docstring': getattr(entity, 'docstring', None),
                    }
                    # Truncate source code to avoid huge JSON files
                    if hasattr(entity, 'source_code') and entity.source_code:
                        entity_data['source_code'] = entity.source_code[:500] + ('...' if len(entity.source_code) > 500 else '')
                    
                    entities_dict.append(entity_data)
                
                output_data = {
                    'repo_path': result['repo_path'],
                    'repo_name': result['repo_name'],
                    'extraction_time': result['extraction_time'],
                    'total_entities': result['total_entities'],
                    'entities': entities_dict,
                    'config': result['config']
                }
                
                json.dump(output_data, f, indent=2)
            
            logger.info(f"  ✓ Extracted {result['total_entities']} entities -> {output_file}")
            
            summary['repos'].append({
                'name': repo_name,
                'path': str(repo_path),
                'total_entities': result['total_entities'],
                'output_file': str(output_file)
            })
            
            summary['processed_repos'] += 1
            summary['total_entities'] += result['total_entities']
            
        except Exception as e:
            logger.error(f"  ✗ Error processing {repo_name}: {e}")
            summary['errors'].append({
                'repo': repo_name,
                'error': str(e)
            })
    
    summary['extraction_end'] = datetime.now().isoformat()
    
    return summary


def save_summary(summary: dict, output_dir: Path) -> None:
    """Save extraction summary to JSON file."""
    summary_file = output_dir / "extraction_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Summary saved to {summary_file}")


def print_summary(summary: dict) -> None:
    """Print extraction summary to console."""
    logger = logging.getLogger(__name__)
    
    logger.info("\n" + "="*70)
    logger.info("ENTITY EXTRACTION SUMMARY")
    logger.info("="*70)
    logger.info(f"Total repositories: {summary['total_repos']}")
    logger.info(f"Processed repositories: {summary['processed_repos']}")
    logger.info(f"Total entities extracted: {summary['total_entities']}")
    
    if summary['repos']:
        logger.info("\nExtracted entities per repository:")
        for repo in summary['repos']:
            logger.info(f"  {repo['name']}: {repo['total_entities']} entities")
    
    if summary['errors']:
        logger.warning(f"\nEncountered {len(summary['errors'])} errors:")
        for error in summary['errors']:
            logger.warning(f"  {error['repo']}: {error['error']}")
    
    logger.info("="*70 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Extract code entities from repositories crawled by RepoCrawler',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='examples/config/code_agent.yaml',
        help='Path to configuration file (default: examples/config/code_agent.yaml)'
    )
    
    parser.add_argument(
        '--repo-dir',
        type=str,
        default=None,
        help='Directory containing crawled repositories (overrides config)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save extracted entities (overrides config)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        logger.info(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        
        # Determine directories
        if args.repo_dir:
            repo_dir = Path(args.repo_dir)
        else:
            # Use repo_crawler output directory as input
            repo_dir = Path(config.get('repo_crawler', {}).get('output', {}).get('save_dir', 'taskdb/code_agent/crawled_repos'))
        
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            output_dir = Path(config.get('entity_extractor', {}).get('output', {}).get('save_dir', 'taskdb/code_agent/extracted_entities'))
        
        logger.info(f"Repository directory: {repo_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Create EntityExtractor from config
        logger.info("Initializing EntityExtractor")
        entity_extractor = EntityExtractor.load_config_from_yaml(args.config)
        
        # Process repositories
        logger.info("Starting entity extraction...")
        summary = process_repositories(
            repo_dir,
            entity_extractor,
            output_dir,
            verbose=args.verbose
        )
        
        # Save and print summary
        save_summary(summary, output_dir)
        print_summary(summary)
        
        logger.info("Entity extraction completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=args.verbose)
        return 1


if __name__ == '__main__':
    sys.exit(main())
