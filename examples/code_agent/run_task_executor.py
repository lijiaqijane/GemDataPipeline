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
        
        # Load configuration and create task executor
        executor = TaskExecutor.load_config_from_yaml(str(config_path))
        
        # Setup and run tasks
        logger.info("Setting up and running tasks...")
        executor.setup_and_run_tasks()
        
        logger.info("Task execution completed!")
        
    except Exception as e:
        logger.exception(f"Error during task execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
