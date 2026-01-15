"""
Test script for TaskExecutor.

This script demonstrates how to use the TaskExecutor to run tasks.
"""

import argparse
import logging
import sys
import yaml
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent_gem.agents.code_agent.task_executor import TaskExecutor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Run task executor for SWE-bench tasks"
    )
    
    # Configuration file
    parser.add_argument(
        "--config",
        type=str,
        default="examples/config/code_agent.yaml",
        help="Path to configuration file"
    )
    
    # Task selection
    parser.add_argument(
        "--task-id",
        type=str,
        default=None,
        help="Single task ID to process"
    )
    parser.add_argument(
        "--task-list-file",
        type=str,
        default=None,
        help="Path to file containing task IDs (one per line)"
    )
    parser.add_argument(
        "--task-batch",
        type=int,
        default=-1,
        help="Batch size (number of tasks per batch)"
    )
    parser.add_argument(
        "--batch-index",
        type=int,
        default=-1,
        help="Batch index (1-based, e.g., 1 for first batch)"
    )
    
    # Paths (optional - can be read from config file)
    parser.add_argument(
        "--tasks-map",
        type=str,
        default=None,
        help="Path to tasks map file (JSON or JSONL). If not provided, reads from config file."
    )
    parser.add_argument(
        "--setup-dir",
        type=str,
        default=None,
        help="Directory where repositories should be cloned. If not provided, reads from config file."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output results. If not provided, reads from config file."
    )
    
    # Execution settings
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="Number of parallel processes (default: from config file)"
    )
    parser.add_argument(
        "--max-iteration-num",
        type=int,
        default=None,
        help="Maximum number of iterations (default: from config file)"
    )
    
    # Agent flags
    parser.add_argument(
        "--disable-context-retrieval",
        action="store_true",
        help="Disable context retrieval agent"
    )
    parser.add_argument(
        "--using-ubuntu-only",
        action="store_true",
        help="Use Ubuntu-only base images"
    )
    
    args = parser.parse_args()
    
    try:
        # Resolve config file path to absolute path
        config_path = Path(args.config)
        if not config_path.is_absolute():
            # If relative path, try relative to current working directory first
            # If not found, try relative to script's parent directory (for default path)
            if not config_path.exists():
                script_relative_path = Path(__file__).parent.parent / config_path
                if script_relative_path.exists():
                    config_path = script_relative_path
        config_path = config_path.resolve()
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load configuration
        config = TaskExecutor.load_config_from_yaml(str(config_path))
        
        # Override with command-line arguments
        if args.tasks_map:
            config.tasks_map_file = args.tasks_map
        if args.setup_dir:
            config.setup_dir = args.setup_dir
        if args.output_dir:
            config.output_dir = args.output_dir
        if args.task_id:
            config.task_id = args.task_id
        if args.task_list_file:
            config.task_list_file = args.task_list_file
        if args.task_batch != -1:
            config.task_batch = args.task_batch
        if args.batch_index != -1:
            config.batch_index = args.batch_index
        if args.max_iteration_num is not None:
            config.max_iteration_num = args.max_iteration_num
        
        # Get num_processes from config or command line
        if args.num_processes is not None:
            num_processes = args.num_processes
        else:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            task_execution_config = config_dict.get('task_execution', {})
            num_processes = task_execution_config.get('num_processes', 1)
        
        # Override agent flags
        if args.disable_context_retrieval:
            config.disable_context_retrieval = True
        if args.using_ubuntu_only:
            config.using_ubuntu_only = True
        
        # Validate required parameters
        if not config.tasks_map_file:
            raise ValueError("tasks_map_file is required (set in config file or --tasks-map)")
        if not config.setup_dir:
            raise ValueError("setup_dir is required (set in config file or --setup-dir)")
        if not config.output_dir:
            raise ValueError("output_dir is required (set in config file or --output-dir)")
        
        # Create task executor
        executor = TaskExecutor(
            output_dir=config.output_dir,
            max_iteration_num=config.max_iteration_num,
            disable_context_retrieval=config.disable_context_retrieval,
            using_ubuntu_only=config.using_ubuntu_only,
            tasks_map_file=config.tasks_map_file,
            setup_dir=config.setup_dir,
            use_cache=config.use_cache,
            cache_name_format=config.cache_name_format,
            task_dir_name_format=config.task_dir_name_format,
            timestamp_format=config.timestamp_format,
        )
        
        # Setup and run tasks
        logger.info("Setting up and running tasks...")
        executor.setup_and_run_tasks(
            task_id=config.task_id,
            task_list_file=config.task_list_file,
            task_batch=config.task_batch,
            batch_index=config.batch_index,
            num_processes=num_processes,
            organize_output=False,
        )
        
        logger.info("Task execution completed!")
        
    except Exception as e:
        logger.exception(f"Error during task execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
