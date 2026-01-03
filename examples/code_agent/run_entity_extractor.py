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
import ast
import json
import logging
import re
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


def _slugify(name: str) -> str:
    """Convert a string to a filesystem-safe, predictable slug."""
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)
    safe = safe.strip("_")
    return safe or "entity"


def process_repositories(
    repo_dir: Path,
    entity_extractor: EntityExtractor,
    output_dir: Path,
    verbose: bool = False,
    overwrite_existing: bool = False,
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

        repo_output_dir = output_dir / repo_name
        if repo_output_dir.exists() and not overwrite_existing:
            logger.info(
                "  ⊙ Skipping %s: output already exists and overwrite_existing=False",
                repo_name,
            )
            summary['repos'].append({
                'name': repo_name,
                'path': str(repo_path),
                'total_entities': None,
                'output_dir': str(repo_output_dir),
                'manifest': str(repo_output_dir / "entities.json"),
                'skipped': True,
            })
            continue
        
        try:
            # Extract entities from repository
            result = entity_extractor.extract_entities_from_repo(str(repo_path))

            # If no entities are found, skip creating output folder/files
            if result['total_entities'] == 0:
                logger.info("  ⊙ No entities found in %s; skipping save", repo_name)
                summary['repos'].append({
                    'name': repo_name,
                    'path': str(repo_path),
                    'total_entities': 0,
                    'output_dir': None,
                    'manifest': None,
                    'skipped': True,
                })
                summary['processed_repos'] += 1
                continue

            repo_output_dir.mkdir(parents=True, exist_ok=True)

            entities_dict = []
            for entity_idx, entity in enumerate(result['entities'], 1):
                try:
                    docstring = ast.get_docstring(entity.node)
                except Exception:
                    docstring = None

                entity_data = {
                    'id': entity_idx,
                    'repo_name': repo_name,
                    'repo_path': str(repo_path),
                    'name': entity.name,
                    'file_path': entity.file_path,
                    'start_line': entity.line_start,
                    'end_line': entity.line_end,
                    'line_count': getattr(entity, 'line_count', entity.line_end - entity.line_start + 1),
                    'docstring': docstring,
                    'src_code': getattr(entity, 'src_code', None),
                    'signature': entity.signature,
                    'signature_content': getattr(entity, 'signature_content', ''),
                    'signature_start_line': getattr(entity, 'signature_start_line', -1),
                    'signature_end_line': getattr(entity, 'signature_end_line', -1),
                    'body_content': getattr(entity, 'body_content', ''),
                    'body_start_line': getattr(entity, 'body_start_line', -1),
                    'body_end_line': getattr(entity, 'body_end_line', -1),
                    'stub': entity.stub,
                    'complexity': entity.complexity,
                }

                entity_dir_name = f"{entity_idx:04d}_{_slugify(entity.name)}"
                entity_dir = repo_output_dir / entity_dir_name
                entity_dir.mkdir(parents=True, exist_ok=True)
                entity_data['entity_dir'] = str(entity_dir)

                with open(entity_dir / "entity.json", 'w', encoding='utf-8') as ef:
                    json.dump(entity_data, ef, indent=2)

                entities_dict.append(entity_data)

            manifest_path = repo_output_dir / "entities.json"
            output_data = {
                'repo_path': result['repo_path'],
                'repo_name': result['repo_name'],
                'extraction_time': result['extraction_time'],
                'total_entities': result['total_entities'],
                'entities': entities_dict,
                'config': result['config']
            }

            with open(manifest_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2)

            logger.info(f"  ✓ Extracted {result['total_entities']} entities -> {repo_output_dir}")

            summary['repos'].append({
                'name': repo_name,
                'path': str(repo_path),
                'total_entities': result['total_entities'],
                'output_dir': str(repo_output_dir),
                'manifest': str(manifest_path),
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
        '--overwrite-existing',
        action='store_true',
        help='Overwrite already extracted entities if output exists'
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

        overwrite_existing = bool(
            args.overwrite_existing or
            config.get('entity_extractor', {}).get('output', {}).get('overwrite_existing', False)
        )
        
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
            verbose=args.verbose,
            overwrite_existing=overwrite_existing,
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
