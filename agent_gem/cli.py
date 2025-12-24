from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import coloredlogs
import dotenv

from agent_gem.agents import GeneralAgent
from agent_gem.generator import EnvironmentGenerator, GenerationRequest
from agent_gem.llm import LLMClient
from agent_gem.utils import check_sandbox_fusion, validate_environment
from agent_gem.writer import TaskWriter

dotenv.load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent_gem", description="Generative agentic environment generator."
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Original generate command
    gen_parser = subparsers.add_parser(
        "generate", help="Generate tasks using agent pipelines"
    )
    gen_parser.add_argument(
        "--agent-type",
        default="general_agent",
        choices=[
            "search_agent",
            "code_agent",
            "code_interpreter_agent",
            "general_agent",
        ],
        help="Agent pipeline to invoke.",
    )
    gen_parser.add_argument(
        "--topic",
        default=None,
        help="Optional domain/topic for the generated task; if omitted, the agent will pick one.",
    )
    gen_parser.add_argument("--num", type=int, default=1, help="Number of tasks to generate.")
    gen_parser.add_argument(
        "--difficulty",
        default=3,
        type=int,
        help="Target difficulty level (int).",
    )
    gen_parser.add_argument(
        "--taskdb-root",
        default="taskdb",
        help="Root directory for generated task taskdb.",
    )
    gen_parser.add_argument("--no-validate", action="store_true", help="Skip schema validation guards.")
    gen_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging to surface agent thinking steps.",
    )

    # New synthesize command (compatible with general_agent CLI)
    synth_parser = subparsers.add_parser(
        "synthesize", help="Synthesize environment and tasks (compatible with general_agent)"
    )
    synth_parser.add_argument(
        "--category",
        required=True,
        help="Task category, e.g., 'plan a travel itinerary'",
    )
    synth_parser.add_argument(
        "--sandbox",
        default="sandbox/demo",
        help="Sandbox directory to store database and generated outputs",
    )
    synth_parser.add_argument(
        "--rounds",
        type=int,
        default=2,
        help="Number of difficulty refinement rounds",
    )
    synth_parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip auto execution and verification (for debugging)",
    )
    synth_parser.add_argument(
        "--max-validation-rounds",
        type=int,
        default=2,
        help="Maximum repair attempts when validation fails",
    )
    synth_parser.add_argument(
        "--use-sandbox-fusion",
        action="store_true",
        default=True,
        help="Use SandboxFusion for secure code execution (default: enabled)",
    )
    synth_parser.add_argument(
        "--no-sandbox-fusion",
        action="store_false",
        dest="use_sandbox_fusion",
        help="Disable SandboxFusion",
    )
    synth_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    # New batch command for code_agent with triple generation
    batch_parser = subparsers.add_parser(
        "batch", help="Batch generate tasks from GitHub repositories"
    )
    batch_parser.add_argument(
        "--config",
        "-c",
        default="config/code_agent.yaml",
        help="Path to configuration YAML file (default: config/code_agent.yaml)",
    )
    batch_parser.add_argument(
        "--repo",
        help="Specific repository name (e.g., 'numpy/numpy') to generate task from",
    )
    batch_parser.add_argument(
        "--file",
        help="Specific file path within the repo (requires --repo)",
    )
    batch_parser.add_argument(
        "--function",
        help="Specific function name (requires --repo and --file)",
    )
    batch_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser


def _handle_generate(args: argparse.Namespace) -> None:
    llm = LLMClient.from_env()
    generator = EnvironmentGenerator(llm, taskdb=Path(args.taskdb_root))
    request = GenerationRequest(
        agent_type=args.agent_type,
        topic=args.topic,
        num=args.num,
        difficulty=args.difficulty,
        validate=not args.no_validate,
    )
    packages = generator.generate(request)

    print(f"Generated {len(packages)} task(s) with agent={args.agent_type}:")
    for pkg in packages:
        print(f"- {pkg.task.summary()} @ {pkg.task_path}")


def _handle_synthesize(args: argparse.Namespace) -> None:
    """Handle synthesize command (compatible with general_agent CLI)."""
    # Validate environment before starting
    if args.use_sandbox_fusion:
        sandbox_url = os.getenv("SANDBOX_FUSION_URL", "http://localhost:8080")
        logging.info("Checking SandboxFusion service...")
        if not check_sandbox_fusion(sandbox_url):
            logging.error("SandboxFusion service unavailable (%s)", sandbox_url)
            logging.error("Please start SandboxFusion service first, then retry.")
            sys.exit(1)
        logging.info("SandboxFusion service available")

    is_valid, error_msg = validate_environment(use_sandbox_fusion=args.use_sandbox_fusion)
    if not is_valid:
        logging.error("Environment configuration validation failed: %s", error_msg)
        sys.exit(1)

    # Setup sandbox directory
    sandbox_path = Path(args.sandbox)
    sandbox_path.mkdir(parents=True, exist_ok=True)

    # Initialize components
    llm = LLMClient.from_env()
    writer = TaskWriter(root=sandbox_path)
    agent = GeneralAgent(llm, taskdb_root=str(sandbox_path))

    # Generate tasks
    category = args.category

    logging.info(f"Starting synthesis: category={category}, rounds={args.rounds}")

    # Generate tasks with all refinement rounds
    # GeneralAgent.generate() handles multiple rounds internally
    request = GenerationRequest(
        agent_type="general_agent",
        topic=category,
        num=1,
        difficulty=args.rounds,  # Final difficulty level
        validate=not args.no_validate,
        max_refine_rounds=args.rounds,  # Number of rounds (initial + refinements)
        max_validation_rounds=args.max_validation_rounds,
        persist_result=True,
    )

    # Generate the task package (includes all refinement rounds internally)
    package = agent.generate(request)
    if not package:
        logging.error("Failed to generate task package")
        sys.exit(1)

    # Extract records from writer (which loads from db.json)
    records = writer.records

    # For quadruple format with multiple rounds, we need to generate each round separately
    # Since GeneralAgent.generate() only returns the final package after all refinements,
    # we'll generate packages for each round
    packages = []
    
    # Generate initial task (round 1, difficulty 1)
    initial_request = GenerationRequest(
        agent_type="general_agent",
        topic=category,
        num=1,
        difficulty=1,
        validate=not args.no_validate,
        max_refine_rounds=1,  # Only initial task, no refinement
        max_validation_rounds=args.max_validation_rounds,
        persist_result=False,
    )
    initial_package = agent.generate(initial_request)
    if initial_package:
        packages.append(initial_package)
        current_package = initial_package
        
        # Generate refined tasks for remaining rounds (2 to args.rounds)
        for round_idx in range(1, args.rounds):
            from agent_gem.agents.base import TaskContext
            from agent_gem.sandbox import SandboxExecutor
            
            # Create context for this refinement round
            refine_request = GenerationRequest(
                agent_type="general_agent",
                topic=category,
                num=1,
                difficulty=round_idx + 1,
                validate=not args.no_validate,
                max_refine_rounds=1,
                max_validation_rounds=args.max_validation_rounds,
                persist_result=False,
            )
            ctx = TaskContext(task_id=current_package.task.task_id, request=refine_request)
            ctx.current_difficulty = round_idx + 1
            
            # Setup sandbox for validation
            sandbox_dir = Path(writer.task_dir(current_package.task.task_id, "general_agent"), "_sandbox")
            sandbox = SandboxExecutor(sandbox_dir=sandbox_dir)
            agent._configure_sandbox(sandbox)
            agent._register_task_tools(current_package.task.tool_set, sandbox, ctx)
            
            # Refine the task
            refined = agent._refine_task(
                previous=current_package,
                records=records,
                tool_specs=current_package.task.tool_set,
                ctx=ctx,
                target_difficulty=round_idx + 1,
            )
            
            # Ensure substantive and validate
            refined = agent._ensure_substantive_task(current_package.task.tool_set, refined, ctx)
            if not args.no_validate:
                refined = agent._ensure_valid(refine_request, refined, ctx, sandbox)
            
            packages.append(refined)
            current_package = refined

    # Persist in quadruple format
    writer.persist_quadruple_format(
        category=category,
        records=records,
        packages=packages,
        output_path=sandbox_path / "tasks.json",
    )

    print(f"Synthesized {len(packages)} task(s):")
    for pkg in packages:
        print(f"- [{pkg.task.difficulty_level}] {pkg.task.task_title}: {pkg.task.task_content[:100]}")


def _handle_batch(args: argparse.Namespace) -> None:
    """Handle batch command for generating tasks from triples or specific repo/file/function."""
    from agent_gem.agents import CodeAgent, TripleGenerator
    from agent_gem.config import CodeAgentConfig
    
    print("\n" + "=" * 80)
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "批量任务生成 - Code Agent" + " " * 23 + "║")
    print("╚" + "=" * 78 + "╝")
    print("=" * 80 + "\n")
    
    # Load configuration
    logging.info(f"Loading configuration from: {args.config}")
    config = CodeAgentConfig.from_file(args.config)
    
    logging.info(f"Configuration:")
    logging.info(f"  taskdb_root: {config.taskdb_root}")
    logging.info(f"  difficulty: {config.difficulty}")
    logging.info(f"  max_tokens: {config.max_tokens}")
    logging.info(f"  temperature: {config.temperature}")
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
                temperature=config.temperature,
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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Default to generate command for backward compatibility
    if not args.command:
        args.command = "generate"
        # Re-parse with default command
        args = parser.parse_args(argv)

    log_level = logging.DEBUG if getattr(args, "verbose", False) else logging.INFO
    coloredlogs.install(
        level=log_level,
        fmt="[%(asctime)s][%(levelname)s][%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level_styles={
            "debug": {"color": "cyan"},
            "info": {"color": "white"},
            "warning": {"color": "yellow", "bold": True},
            "error": {"color": "red", "bold": True},
            "critical": {"color": "red", "bold": True, "background": "black"},
        },
        field_styles={
            "asctime": {"color": "white"},
            "levelname": {"color": "blue", "bold": True},
            "name": {"color": "green"},
        },
    )

    if args.command == "synthesize":
        _handle_synthesize(args)
    elif args.command == "batch":
        _handle_batch(args)
    else:
        _handle_generate(args)


if __name__ == "__main__":
    main()
