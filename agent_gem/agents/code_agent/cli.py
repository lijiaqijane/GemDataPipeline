from __future__ import annotations

import json
import argparse
import logging
import os
import sys
from pathlib import Path

from agent_gem.generator import GenerationRequest

def add_code_synthesize_subparser(subparsers: argparse._SubParsersAction) -> None:
    # New batch command for code_agent with triple generation
    code_synth_parser = subparsers.add_parser(
        "code_synthesize", help="Generate code tasks from GitHub repositories"
    )
    code_synth_parser.add_argument(
        "--config",
        "-c",
        default="config/code_agent.yaml",
        help="Path to configuration YAML file (default: config/code_agent.yaml)",
    )
    code_synth_parser.add_argument(
        "--repo",
        help="Specific repository name (e.g., 'numpy/numpy') to generate task from",
    )
    code_synth_parser.add_argument(
        "--file",
        help="Specific file path within the repo (requires --repo)",
    )
    code_synth_parser.add_argument(
        "--function",
        help="Specific function name (requires --repo and --file)",
    )


def handle_code_synthesize(args: argparse.Namespace) -> None:
    """Handle batch command for generating tasks from triples or specific repo/file/function."""
    from agent_gem.agents import CodeAgent, TripleGenerator
    from agent_gem.config import CodeAgentConfig
    
    print("\n" + "=" * 80)
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "Code Agent" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print("=" * 80 + "\n")
    
    # Load configuration
    logging.info(f"Loading configuration from: {args.config}")
    config = CodeAgentConfig.from_file(args.config)
    
    logging.info(f"Configuration:")
    logging.info(f"  taskdb_root: {config.taskdb_root}")
    logging.info(f"  difficulty: {config.difficulty}")
    logging.info(f"  max_tokens: {config.max_tokens}")
    logging.info(f"  num_tasks: {config.num_tasks}")
    logging.info(f"  skip_errors: {config.skip_errors}")
    
    # Determine mode: specific triple or batch from triples
    if args.repo:
        # Mode 1: Generate task from specific (repo, file, function)
        logging.info("\n" + "=" * 80)
        logging.info("Mode: Single task generation from specified repo/file/function")
        logging.info("=" * 80)
        
        if not args.file:
            logging.error("Error: --file is required when --repo is specified")
            sys.exit(1)
        
        if not args.function:
            logging.error("Error: --function is required when --repo and --file are specified")
            sys.exit(1)
        
        # Construct repo URL from repo name (assume GitHub)
        if args.repo.startswith('http'):
            repo_url = args.repo.rstrip('.git')
        else:
            repo_url = f"https://github.com/{args.repo}"
        
        # Create single triple
        triple = {
            'repo_url': repo_url,
            'repo_name': args.repo if '/' in args.repo else f"unknown/{args.repo}",
            'file_path': args.file,
            'function_name': args.function,
            'quality_score': 1.0,  # Manual specification implies high quality
        }
        
        logging.info(f"Target triple:")
        logging.info(f"  Repository: {triple['repo_name']}")
        logging.info(f"  URL: {triple['repo_url']}")
        logging.info(f"  File: {triple['file_path']}")
        logging.info(f"  Function: {triple['function_name']}")
        
        triples = [triple]
        
    else:
        # Mode 2: Generate tasks from triples (load from cache or generate new)
        logging.info("\n" + "=" * 80)
        logging.info("Mode: Batch task generation from triples")
        logging.info("=" * 80)
        
        triples = None
        triples_file = Path(config.cache_dir) / "function_triples.json"
        
        if triples_file.exists():
            # Load from cache
            logging.info(f"Loading cached triples from: {triples_file}")
            with open(triples_file, 'r') as f:
                triples = json.load(f)
            logging.info(f"Loaded {len(triples)} cached triples")
        else:
            # Generate new triples
            logging.info("No cached triples found. Generating new triples from GitHub...")
            
            github_token = os.environ.get("GITHUB_TOKEN")
            if not github_token:
                logging.warning("⚠️  GITHUB_TOKEN not set - API rate limit will be very low (60/hour)")
                logging.warning("   Set token for higher limit (5000/hour): export GITHUB_TOKEN=your_token")
            
            # Get triple generation config
            triple_config = config.get_triple_generation_config()
            
            if triple_config:
                num_triples = triple_config.target_triples
                quality_threshold = triple_config.quality_threshold
                refresh_repos = triple_config.refresh_repos
            else:
                logging.error("Error: triple_generation section not found in config file")
                logging.error("Please add triple_generation configuration to generate triples")
                sys.exit(1)
            
            logging.info(f"Triple generation parameters:")
            logging.info(f"  num_triples: {num_triples}")
            logging.info(f"  quality_threshold: {quality_threshold}")
            logging.info(f"  refresh_repos: {refresh_repos}")
            
            # Initialize TripleGenerator
            import tempfile
            import yaml
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as tmp:
                yaml.dump(config.config['triple_generation'], tmp)
                tmp_config_path = tmp.name
            
            generator = TripleGenerator(
                github_token=github_token,
                cache_dir=config.cache_dir,
                config_path=tmp_config_path
            )
            
            # Clean up temp file
            os.unlink(tmp_config_path)
            
            # Test GitHub connection
            if not generator.test_github_connection():
                logging.error("GitHub API connection failed")
                sys.exit(1)
            
            # Generate triples
            logging.info(f"Generating {num_triples} triples (quality >= {quality_threshold})...")
            triples = generator.generate_triples(
                num_triples=num_triples,
                quality_threshold=quality_threshold,
                refresh_repos=refresh_repos
            )
            
            if not triples:
                logging.error("Failed to generate triples")
                sys.exit(1)
            
            logging.info(f"✅ Generated {len(triples)} triples")
            
            # Display statistics
            repos = set(t['repo_name'] for t in triples)
            avg_score = sum(t['quality_score'] for t in triples) / len(triples)
            logging.info(f"   Repositories: {len(repos)}")
            logging.info(f"   Avg quality: {avg_score:.3f}")
        
        # Limit to num_tasks
        if len(triples) > config.num_tasks:
            logging.info(f"Selecting top {config.num_tasks} highest quality triples")
            triples = triples[:config.num_tasks]
    
    # Initialize CodeAgent
    logging.info("\n" + "=" * 80)
    logging.info("Initializing CodeAgent")
    logging.info("=" * 80)
    
    try:
        agent = CodeAgent(config=config)
        logging.info("✅ CodeAgent initialized")
    except Exception as e:
        logging.error(f"Failed to initialize CodeAgent: {e}")
        sys.exit(1)
    
    # Step 4: Generate tasks
    logging.info("\n" + "=" * 80)
    logging.info(f"Step 4: Generating {len(triples)} tasks")
    logging.info("=" * 80 + "\n")
    
    successful = 0
    failed = 0
    results = []
    
    for i, triple in enumerate(triples, 1):
        logging.info(f"\n{'='*80}")
        logging.info(f"Task {i}/{len(triples)}")
        logging.info(f"{'='*80}")
        logging.info(f"Repository: {triple['repo_name']}")
        logging.info(f"File: {triple['file_path']}")
        logging.info(f"Function: {triple['function_name']}")
        logging.info(f"Quality: {triple['quality_score']:.3f}")
        logging.info(f"{'='*80}\n")
        
        try:
            # Create generation request
            request = GenerationRequest(
                agent_type='code_agent',
                topic=triple['repo_url'],
                difficulty=config.difficulty,
                max_tokens=config.max_tokens,
            )
            
            # Generate task
            package = agent.generate(
                request,
                target_file_path=triple['file_path'],
                target_function_name=triple['function_name']
            )
            
            if package:
                successful += 1
                result = {
                    'success': True,
                    'task_id': package.task.task_id,
                    'triple': triple,
                    'task_dir': package.metadata.get('task_dir', '')
                }
                results.append(result)
                
                logging.info(f"\n✅ Task {i} generated successfully")
                logging.info(f"   Task ID: {package.task.task_id}")
                logging.info(f"   Location: {package.metadata.get('task_dir', 'N/A')}")
            else:
                failed += 1
                result = {
                    'success': False,
                    'triple': triple,
                    'error': 'Generation returned None'
                }
                results.append(result)
                
                logging.warning(f"\n❌ Task {i} generation failed: returned None")
                
                if not config.skip_errors:
                    logging.warning("Stopping batch generation (use --skip-errors to continue)")
                    break
        
        except Exception as e:
            failed += 1
            result = {
                'success': False,
                'triple': triple,
                'error': str(e)
            }
            results.append(result)
            
            logging.error(f"❌ Task {i} generation failed: {e}", exc_info=args.verbose)
            
            if not config.skip_errors:
                logging.warning("Stopping batch generation (use --skip-errors to continue)")
                break
    
    # Step 5: Summary
    logging.info("\n" + "=" * 80)
    logging.info("📊 Batch Generation Complete")
    logging.info("=" * 80)
    logging.info(f"\nStatistics:")
    logging.info(f"  ✅ Successful: {successful}")
    logging.info(f"  ❌ Failed: {failed}")
    logging.info(f"  📊 Total: {successful + failed}")
    if successful + failed > 0:
        logging.info(f"  🎯 Success rate: {successful / (successful + failed) * 100:.1f}%")
    
    
    # Show successful tasks
    if successful > 0:
        logging.info(f"\n✅ Successfully generated tasks:")
        for result in results:
            if result['success']:
                logging.info(f"   - {result['task_id']}: {result['triple']['repo_name']}/{result['triple']['function_name']}")
    
    # Show failed tasks
    if failed > 0:
        logging.info(f"\n❌ Failed tasks:")
        for result in results:
            if not result['success']:
                logging.info(f"   - {result['triple']['repo_name']}/{result['triple']['function_name']}: {result.get('error', 'Unknown')}")
    
    if failed > 0:
        sys.exit(1)
