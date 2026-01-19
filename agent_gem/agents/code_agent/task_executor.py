"""
Task Executor for code_agent module.

This module provides the TaskExecutor to manage task execution flow,
adapted from app.main.py.
"""

from __future__ import annotations

import os
import shutil
import json
import multiprocessing
import logging
import subprocess
import contextlib
import signal
import sys
import time
import random
from datetime import datetime
from os.path import join as pjoin
from pathlib import Path
from collections.abc import Mapping, Sequence, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

import docker
import yaml

from agent_gem.agents.code_agent.environment_setup_utils import Task, SweTask, TaskAdapter
from agent_gem.agents.code_agent.environment_setup_agent import EnvironmentSetupAgent
from agent_gem.agents.code_agent.environment_setup_utils.utils_adapter import (
    create_dir_if_not_exists,
    clone_repo_and_checkout,
)

from .raw_tasks import RawSweTask

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def cd(newdir: str):
    """Context manager for changing the current working directory."""
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)


def run_command(cmd: List[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a command in the shell."""
    try:
        cp = subprocess.run(cmd, check=True, **kwargs)
        return cp
    except subprocess.CalledProcessError as e:
        logger.error(f"Error running command: {cmd}, {e}")
        raise e


def clone_repo(clone_link: str, cloned_dir: str) -> None:
    """Clone a repository to the specified directory."""
    dest_dir = os.path.dirname(cloned_dir)
    cloned_name = os.path.basename(cloned_dir)
    clone_cmd = ["git", "clone", clone_link, cloned_name]
    create_dir_if_not_exists(dest_dir)
    with cd(dest_dir):
        run_command(clone_cmd)


def clone_repo_for_cache(clone_link: str, cloned_dir: str) -> None:
    """Clone a repository for cache (without checkout)."""
    if clone_link.endswith('.git'):
        clone_repo(clone_link, cloned_dir)
    else:
        # If it's a local path, copy it
        if os.path.isdir(cloned_dir):
            shutil.rmtree(cloned_dir)
        shutil.copytree(clone_link, cloned_dir)


@dataclass
class TaskExecutorConfig:
    """Configuration for TaskExecutor."""
    
    # Task execution parameters
    output_dir: str
    max_iteration_num: int = 15
    disable_context_retrieval: bool = False
    using_ubuntu_only: bool = False
    num_processes: int = 1
    shuffle_tasks: bool = False  # Whether to shuffle task order randomly
    llm_temperature: float = 0.0  # Temperature for LLM calls during task execution
    # Environment setup parameters
    setup_dir: Optional[str] = None
    use_cache: bool = True
    cache_name_format: str = "{repo}_cache"
    task_dir_name_format: str = "{task_id}_{timestamp}"
    timestamp_format: str = "%Y-%m-%d_%H-%M-%S"
    
    # PR data source
    # Option 1: Specify a single prs-valid.jsonl file
    prs_valid_file: Optional[str] = None
    # Option 2: Specify a directory containing prs-valid.jsonl files
    prs_valid_dir: Optional[str] = None
    
    # PR annotation filtering
    pr_annotations_dir: Optional[str] = None  # Directory containing pr annotation files
    filter_criteria: Optional[Dict[str, Any]] = None  # Filter criteria for PR annotations
    # Example filter_criteria:
    # {
    #     "pr_category": ["bug_fix"],  # Only bug fixes
    #     "issue_difficulty": ["easy", "medium"],  # Only easy or medium
    #     "issue_description_reasonable": True,  # Must be reasonable
    #     "gold_patch_solves_issue": True,  # Must solve issue
    #     "test_patch_designed_for_issue": True,  # Must have proper test
    #     "num_gold_files_changed_max": 10,  # Max files in gold patch
    #     "num_test_files_changed_max": 5,  # Max files in test patch
    # }
    
    # Task selection (mutually exclusive)
    task_id: Optional[str] = None
    task_list_file: Optional[str] = None
    task_batch: int = -1
    batch_index: int = -1


class TaskExecutor:
    """
    Task executor for managing task execution flow.
    
    This class handles both environment setup and task execution.
    """
    
    def __init__(
        self,
        output_dir: str,
        max_iteration_num: int = 15,
        disable_context_retrieval: bool = False,
        using_ubuntu_only: bool = False,
        # Environment setup parameters
        setup_dir: str | None = None,
        use_cache: bool = True,
        cache_name_format: str = "{repo}_cache",
        task_dir_name_format: str = "{task_id}_{timestamp}",
        timestamp_format: str = "%Y-%m-%d_%H-%M-%S",
        # PR data source
        prs_valid_file: str | None = None,
        prs_valid_dir: str | None = None,
        # PR annotation filtering
        pr_annotations_dir: str | None = None,
        filter_criteria: Dict[str, Any] | None = None,
        num_processes: int = 1,
        shuffle_tasks: bool = False,
        llm_temperature: float = 0.0,
        agent_llm_configs: Dict[str, Dict[str, float | int]] | None = None,
    ):
        """
        Initialize the task executor.
        
        Args:
            output_dir: Output directory for results
            max_iteration_num: Maximum number of iterations
            disable_context_retrieval: Whether to disable context retrieval
            using_ubuntu_only: Whether to use Ubuntu-only base images
            setup_dir: Directory for repository cache and task directories
            use_cache: Whether to use existing cache
            cache_name_format: Format string for cache directory names
            task_dir_name_format: Format string for task directory names
            timestamp_format: Format string for timestamps
            prs_valid_file: Path to a single prs-valid.jsonl file
            prs_valid_dir: Directory containing prs-valid.jsonl files
            pr_annotations_dir: Directory containing pr annotation files
            filter_criteria: Filter criteria for PR annotations
            num_processes: Number of parallel processes
            shuffle_tasks: Whether to shuffle task order randomly
            llm_temperature: Temperature for LLM calls during task execution (deprecated, use agent_llm_configs)
            agent_llm_configs: Dict mapping agent names to their LLM configs
                              Format: {"agent_name": {"temperature": 0.2, "max_tokens": 4096}}
        """
        self.output_dir = output_dir
        self.max_iteration_num = max_iteration_num
        self.disable_context_retrieval = disable_context_retrieval
        self.using_ubuntu_only = using_ubuntu_only
        self.num_processes = num_processes
        self.shuffle_tasks = shuffle_tasks
        self.llm_temperature = llm_temperature
        self.agent_llm_configs = agent_llm_configs or {}
        # Environment setup parameters
        self.setup_dir = setup_dir
        self.use_cache = use_cache
        self.cache_name_format = cache_name_format
        self.task_dir_name_format = task_dir_name_format
        self.timestamp_format = timestamp_format
        
        # PR data source
        self.prs_valid_file = prs_valid_file
        self.prs_valid_dir = prs_valid_dir
        self.pr_annotations_dir = pr_annotations_dir
        self.filter_criteria = filter_criteria or {}
        
        # Shutdown control for graceful interruption
        self._shutdown_event = multiprocessing.Event()
        self._executor = None  # Track executor for cleanup
    
    def __getstate__(self):
        """Customize pickling to exclude non-serializable objects."""
        state = self.__dict__.copy()
        # Remove _shutdown_event and _executor before pickling
        # These objects cannot be pickled and are not needed in child processes
        state['_shutdown_event'] = None
        state['_executor'] = None
        return state
    
    def __setstate__(self, state):
        """Restore state after unpickling."""
        self.__dict__.update(state)
        # Recreate _shutdown_event in child process (won't be shared with parent, but that's OK)
        # Child processes don't need to check shutdown event - they'll be terminated by parent
        if '_shutdown_event' not in state or state['_shutdown_event'] is None:
            self._shutdown_event = multiprocessing.Event()
        if '_executor' not in state:
            self._executor = None
    
    @classmethod
    def load_config_from_yaml(cls, config_path: str) -> "TaskExecutor":
        """
        Load configuration from YAML file and create TaskExecutor instance.
        
        Args:
            config_path: Path to the YAML configuration file
        
        Returns:
            TaskExecutor instance with loaded configuration
        """
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
        
        task_execution_config = config_dict.get('task_execution', {})
        
        # Extract cache config
        cache_config = task_execution_config.get('cache', {})
        
        # Extract task_dir config
        task_dir_config = task_execution_config.get('task_dir', {})
        
        # Extract agents config
        agents_config = task_execution_config.get('agents', {})
        
        # Extract PR data source config
        prs_valid_file = task_execution_config.get('prs_valid_file')
        prs_valid_dir = task_execution_config.get('prs_valid_dir')
        pr_annotations_dir = task_execution_config.get('pr_annotations_dir')
        filter_criteria = task_execution_config.get('filter_criteria')
        
        # Support backward compatibility: check environment_setup if not in task_execution
        env_setup_config = config_dict.get('environment_setup', {})
        if env_setup_config:
            # Fallback to environment_setup for backward compatibility
            cache_config = env_setup_config.get('cache', cache_config)
            task_dir_config = env_setup_config.get('task_dir', task_dir_config)
        
        # Extract agent-specific LLM configurations
        agent_llm_configs = {}
        default_temperature = agents_config.get('llm_temperature', 0.0)
        default_max_tokens = agents_config.get('llm_max_tokens')
        
        # Agent names in the system
        agent_names = [
            'context_retrieval_agent',
            'write_docker_agent',
            'write_eval_script_agent',
            'test_analysis_agent'
        ]
        
        for agent_name in agent_names:
            agent_config = agents_config.get(agent_name, {})
            agent_llm_config = {}
            
            # Get temperature: agent-specific > default
            if 'temperature' in agent_config and agent_config['temperature'] is not None:
                agent_llm_config['temperature'] = agent_config['temperature']
            elif default_temperature is not None:
                agent_llm_config['temperature'] = default_temperature
            
            # Get max_tokens: agent-specific > default
            if 'max_tokens' in agent_config and agent_config['max_tokens'] is not None:
                agent_llm_config['max_tokens'] = agent_config['max_tokens']
            elif default_max_tokens is not None:
                agent_llm_config['max_tokens'] = default_max_tokens
            
            if agent_llm_config:
                agent_llm_configs[agent_name] = agent_llm_config
        
        # Create and return TaskExecutor instance
        return cls(
            output_dir=task_execution_config.get('output_dir', ''),
            max_iteration_num=task_execution_config.get('max_iteration_num', 15),
            disable_context_retrieval=agents_config.get('disable_context_retrieval', False),
            using_ubuntu_only=agents_config.get('using_ubuntu_only', False),
            setup_dir=task_execution_config.get('setup_dir') or env_setup_config.get('setup_dir'),
            use_cache=cache_config.get('use_cache', True),
            cache_name_format=cache_config.get('cache_name_format', '{repo}_cache'),
            task_dir_name_format=task_dir_config.get('name_format', '{task_id}_{timestamp}'),
            timestamp_format=task_dir_config.get('timestamp_format', '%Y-%m-%d_%H-%M-%S'),
            prs_valid_file=prs_valid_file,
            prs_valid_dir=prs_valid_dir,
            pr_annotations_dir=pr_annotations_dir,
            filter_criteria=filter_criteria,
            num_processes=task_execution_config.get('num_processes', 1),
            shuffle_tasks=task_execution_config.get('shuffle_tasks', False),
            llm_temperature=default_temperature,
            agent_llm_configs=agent_llm_configs if agent_llm_configs else None,
        )

    
    def load_prs_from_file_or_dir(self) -> Dict[str, Any]:
        """
        Load PR data from prs-valid.jsonl file(s).
        
        Can load from:
        1. A single specified file (prs_valid_file)
        2. All *-prs-valid.jsonl files in a directory (prs_valid_dir)
        
        Returns:
            Dictionary mapping instance_id to PR data
        """
        all_prs = {}
        
        # Load from single file
        if self.prs_valid_file:
            file_path = Path(self.prs_valid_file)
            if not file_path.exists():
                raise FileNotFoundError(f"PR valid file not found: {self.prs_valid_file}")
            
            logger.info(f"Loading PRs from file: {self.prs_valid_file}")
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        pr_data = json.loads(line)
                        instance_id = pr_data.get('instance_id')
                        if instance_id:
                            all_prs[instance_id] = pr_data
        
        # Load from directory
        elif self.prs_valid_dir:
            dir_path = Path(self.prs_valid_dir)
            if not dir_path.exists() or not dir_path.is_dir():
                raise FileNotFoundError(f"PR valid directory not found: {self.prs_valid_dir}")
            
            # Find all *-prs-valid.jsonl files
            pr_files = list(dir_path.glob("*-prs-valid.jsonl"))
            if not pr_files:
                raise ValueError(f"No *-prs-valid.jsonl files found in {self.prs_valid_dir}")
            
            logger.info(f"Loading PRs from {len(pr_files)} files in directory: {self.prs_valid_dir}")
            for pr_file in pr_files:
                with open(pr_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            pr_data = json.loads(line)
                            instance_id = pr_data.get('instance_id')
                            if instance_id:
                                if instance_id in all_prs:
                                    logger.warning(f"Duplicate instance_id {instance_id} found, keeping first occurrence")
                                else:
                                    all_prs[instance_id] = pr_data
        
        else:
            raise ValueError("Either prs_valid_file or prs_valid_dir must be specified")
        
        logger.info(f"Loaded {len(all_prs)} PRs")
        return all_prs
    
    def load_pr_annotations(self) -> Dict[str, Any]:
        """
        Load PR annotations from annotation files.
        
        Returns:
            Dictionary mapping instance_id to annotation data
        """
        if not self.pr_annotations_dir:
            return {}
        
        annotations = {}
        dir_path = Path(self.pr_annotations_dir)
        
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"PR annotations directory not found: {self.pr_annotations_dir}")
            return {}
        
        # Find all *-prs-annotated.jsonl files
        annotation_files = list(dir_path.glob("*-prs-annotated.jsonl"))
        
        if not annotation_files:
            logger.warning(f"No *-prs-annotated.jsonl files found in {self.pr_annotations_dir}")
            return {}
        
        logger.info(f"Loading PR annotations from {len(annotation_files)} files in directory: {self.pr_annotations_dir}")
        for annotation_file in annotation_files:
            with open(annotation_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        annotation_data = json.loads(line)
                        instance_id = annotation_data.get('instance_id')
                        if instance_id:
                            if instance_id in annotations:
                                logger.warning(f"Duplicate annotation for instance_id {instance_id}, keeping first occurrence")
                            else:
                                annotations[instance_id] = annotation_data
        
        logger.info(f"Loaded {len(annotations)} PR annotations")
        return annotations
    
    def filter_prs_by_annotations(
        self,
        prs: Dict[str, Any],
        annotations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Filter PRs based on annotation criteria.
        
        Args:
            prs: Dictionary of PR data (instance_id -> PR data)
            annotations: Dictionary of annotation data (instance_id -> annotation)
        
        Returns:
            Filtered dictionary of PR data
        """
        if not self.filter_criteria:
            logger.info("No filter criteria specified, returning all PRs")
            return prs
        
        filtered_prs = {}
        criteria = self.filter_criteria
        
        logger.info(f"Filtering PRs with criteria: {criteria}")
        
        for instance_id, pr_data in prs.items():
            annotation = annotations.get(instance_id)
            
            if not annotation:
                # If annotation is required but not found, skip
                if criteria.get('require_annotation', False):
                    continue
                # Otherwise, include PRs without annotations
                filtered_prs[instance_id] = pr_data
                continue
            
            # Check each criterion
            match = True
            
            # PR category filter
            if 'pr_category' in criteria:
                allowed_categories = criteria['pr_category']
                if not isinstance(allowed_categories, list):
                    allowed_categories = [allowed_categories]
                if annotation.get('pr_category') not in allowed_categories:
                    match = False
            
            # Issue difficulty filter
            if match and 'issue_difficulty' in criteria:
                allowed_difficulties = criteria['issue_difficulty']
                if not isinstance(allowed_difficulties, list):
                    allowed_difficulties = [allowed_difficulties]
                if annotation.get('issue_difficulty') not in allowed_difficulties:
                    match = False
            
            # Boolean filters
            for field in ['issue_description_reasonable', 'gold_patch_solves_issue', 'test_patch_designed_for_issue']:
                if match and field in criteria:
                    expected_value = criteria[field]
                    if annotation.get(field) != expected_value:
                        match = False
            
            # File count filters
            if match and 'num_gold_files_changed_max' in criteria:
                max_files = criteria['num_gold_files_changed_max']
                if annotation.get('num_gold_files_changed', 0) > max_files:
                    match = False
            
            if match and 'num_test_files_changed_max' in criteria:
                max_files = criteria['num_test_files_changed_max']
                if annotation.get('num_test_files_changed', 0) > max_files:
                    match = False
            
            if match and 'num_gold_files_changed_min' in criteria:
                min_files = criteria['num_gold_files_changed_min']
                if annotation.get('num_gold_files_changed', 0) < min_files:
                    match = False
            
            if match and 'num_test_files_changed_min' in criteria:
                min_files = criteria['num_test_files_changed_min']
                if annotation.get('num_test_files_changed', 0) < min_files:
                    match = False
            
            if match:
                filtered_prs[instance_id] = pr_data
        
        logger.info(f"Filtered {len(prs)} PRs down to {len(filtered_prs)} PRs")
        return filtered_prs
    
    def parse_task_list_file(self, task_list_file: str) -> List[str]:
        """
        Parse the task list file.
        
        The file should contain one task/instance id per line.
        
        Args:
            task_list_file: Path to the task list file
        
        Returns:
            List of task IDs
        """
        with open(task_list_file, 'r', encoding='utf-8') as f:
            task_ids = f.readlines()
        return [x.strip() for x in task_ids if x.strip()]
    
    def setup_repo_cache(self, repo_name: str, setup_dir: str) -> str:
        """
        Set up repository cache directory.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            setup_dir: Root directory for setup
        
        Returns:
            Path to the repository cache directory
        """
        cache_name = self.cache_name_format.format(repo=repo_name)
        repo_cache_dir = pjoin(setup_dir, cache_name)
        
        if not os.path.isdir(repo_cache_dir):
            logger.info(f"Cloning repository {repo_name} to cache: {repo_cache_dir}")
            github_link = f"https://github.com/{repo_name}.git"
            clone_repo_for_cache(github_link, repo_cache_dir)
        else:
            if self.use_cache:
                logger.info(f"Cache already exists: {repo_cache_dir}, skipping clone.")
            else:
                logger.info(f"Removing existing cache: {repo_cache_dir}")
                shutil.rmtree(repo_cache_dir)
                github_link = f"https://github.com/{repo_name}.git"
                clone_repo_for_cache(github_link, repo_cache_dir)
        
        return repo_cache_dir
    
    def setup_task_directory(self, task_id: str, setup_dir: str) -> str:
        """
        Create task-specific working directory.
        
        Args:
            task_id: Task identifier
            setup_dir: Root directory for setup
        
        Returns:
            Path to the task directory
        """
        task_start_time_s = datetime.now().strftime(self.timestamp_format)
        task_repo_name = self.task_dir_name_format.format(
            task_id=task_id,
            timestamp=task_start_time_s
        )
        task_repo_dir = pjoin(setup_dir, task_repo_name)
        create_dir_if_not_exists(task_repo_dir)
        
        return task_repo_dir
    
    def setup_and_run_tasks(
        self,
        organize_output: bool = False,
    ) -> None:
        """
        Load tasks from config and run them (setup happens during execution).
        
        Args:
            organize_output: Whether to organize output
        """
        if not self.setup_dir:
            raise ValueError("setup_dir is required")
        
        # Determine data source
        if self.prs_valid_file or self.prs_valid_dir:
            # Load from PR valid files
            tasks_map = self.load_prs_from_file_or_dir()
            
            # Load annotations and filter if needed
            if self.pr_annotations_dir:
                annotations = self.load_pr_annotations()
                tasks_map = self.filter_prs_by_annotations(tasks_map, annotations)
        else:
            raise ValueError("Either prs_valid_file, or prs_valid_dir must be specified")
        
        all_task_ids = list(tasks_map.keys())
        
        if len(all_task_ids) == 0:
            raise ValueError("No task ids to run.")
        
        # Create lightweight task info objects (no setup yet - setup happens during execution)
        task_infos = []
        for task_id in all_task_ids:
            task_info = tasks_map[task_id]
            repo_name = task_info.get('repo')
            if not repo_name:
                logger.warning(f"Task {task_id} missing 'repo' field, skipping")
                continue
            task_infos.append((task_id, task_info))
        
        # Shuffle tasks if requested
        if self.shuffle_tasks:
            logger.info("Shuffling task order randomly...")
            random.shuffle(task_infos)
            logger.info("Task order has been shuffled")
        
        logger.info(f"Loaded {len(task_infos)} tasks (setup will happen during execution)")
        
        # Group tasks (simple grouping by index)
        task_groups = {}
        for idx, (task_id, task_info) in enumerate(task_infos):
            task_groups[str(idx)] = [(task_id, task_info)]
        
        # Run tasks (setup happens inside run_task_groups_with_lazy_setup)
        self.run_task_groups_with_lazy_setup(task_groups, self.num_processes, organize_output)
    
    def run_task_groups_with_lazy_setup(
        self,
        task_groups: Mapping[str, Sequence[tuple[str, dict]]],
        num_processes: int = 1,
        organize_output: bool = False,
    ) -> None:
        """
        Run task groups with lazy setup (setup happens during execution).
        
        Args:
            task_groups: Dictionary of task groups, each item is (task_id, task_info) tuple
            num_processes: Number of parallel processes
            organize_output: Whether to organize output
        """
        from itertools import chain
        all_tasks = list(chain.from_iterable(task_groups.values()))
        num_tasks = len(all_tasks)
        
        logger.info(f"Total number of tasks: {num_tasks}")
        logger.info(f"Total number of processes: {num_processes}")
        logger.info(f"Task group info: (number of groups: {len(task_groups)})")
        for key, tasks in task_groups.items():
            logger.info(f"\t{key}: {len(tasks)} tasks")
        
        if num_processes == 1:
            logger.info("Running in single process mode with lazy setup.")
            self.run_tasks_serial(all_tasks)
            logger.info("Finished all tasks sequentially.")
        else:
            self.run_task_groups_parallel_with_lazy_setup(task_groups, num_processes)
        
        if organize_output:
            # post-process completed experiments to get input file to SWE-bench
            logger.info("Post-processing completed experiment results.")
            # swe_input_file = self.organize_and_form_input(self.output_dir)
            # logger.info(f"SWE-Bench input file created: {swe_input_file}")
    
    def run_task_groups(
        self,
        task_groups: Mapping[str, Sequence],
        num_processes: int = 1,
        organize_output: bool = False,
    ) -> None:
        """
        Run task groups in parallel or serial (legacy method for backward compatibility).
        
        Args:
            task_groups: Dictionary of task groups (RawSweTask objects)
            num_processes: Number of parallel processes
            organize_output: Whether to organize output
        """
        from itertools import chain
        all_tasks = list(chain.from_iterable(task_groups.values()))
        num_tasks = len(all_tasks)
        
        logger.info(f"Total number of tasks: {num_tasks}")
        logger.info(f"Total number of processes: {num_processes}")
        logger.info(f"Task group info: (number of groups: {len(task_groups)})")
        for key, tasks in task_groups.items():
            logger.info(f"\t{key}: {len(tasks)} tasks")
        
        if num_processes == 1:
            logger.info("Running in single process mode.")
            for task in all_tasks:
                self.run_task_in_subprocess_legacy(task)
            logger.info("Finished all tasks sequentially.")
        else:
            self.run_task_groups_parallel(task_groups, num_processes)
        
        if organize_output:
            # post-process completed experiments to get input file to SWE-bench
            logger.info("Post-processing completed experiment results.")
            # swe_input_file = self.organize_and_form_input(self.output_dir)
            # logger.info(f"SWE-Bench input file created: {swe_input_file}")
    
    def run_tasks_serial(self, tasks: list) -> None:
        """
        Run tasks serially with lazy setup.
        
        Args:
            tasks: List of (task_id, task_info) tuples
        """
        for task_id, task_info in tasks:
            self.run_raw_task_with_setup(task_id, task_info)
    
    def run_task_groups_parallel_with_lazy_setup(
        self,
        task_groups: Mapping[str, Sequence[tuple[str, dict]]],
        num_processes: int,
    ) -> None:
        """Run task groups in parallel with lazy setup."""
        num_task_groups = len(task_groups)
        num_processes = min(num_processes, num_task_groups)
        
        task_group_ids_items = sorted(
            task_groups.items(), key=lambda x: len(x[1]), reverse=True
        )
        logger.info(f"Sorted task groups: {[x[0] for x in task_group_ids_items]}")
        
        # Set up signal handler for graceful shutdown
        future_to_gid = {}  # Will be populated in the try block
        
        def signal_handler(signum, frame):
            logger.warning("Received interrupt signal. Shutting down gracefully...")
            self._shutdown_event.set()
            if self._executor:
                # Cancel all pending futures
                try:
                    # Try to cancel futures if executor supports it
                    for future in list(future_to_gid.keys()):
                        if not future.done():
                            future.cancel()
                except Exception:
                    pass
                # Shutdown executor
                try:
                    # Python 3.9+ supports cancel_futures
                    if sys.version_info >= (3, 9):
                        self._executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        self._executor.shutdown(wait=False)
                except Exception:
                    pass
            sys.exit(1)
        
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                self._executor = executor
                future_to_gid = {
                    executor.submit(_safe_run_group_with_lazy_setup, gid, tasks, self): gid
                    for gid, tasks in task_group_ids_items
                }
                
                try:
                    for future in as_completed(future_to_gid):
                        if self._shutdown_event.is_set():
                            logger.warning("Shutdown requested. Cancelling remaining tasks...")
                            break
                        
                        gid = future_to_gid[future]
                        try:
                            future.result(timeout=0.1)  # Use timeout to allow checking shutdown event
                            logger.info(f"Task group {gid} finished successfully.")
                        except KeyboardInterrupt:
                            logger.warning("KeyboardInterrupt received. Shutting down...")
                            self._shutdown_event.set()
                            # Cancel remaining futures
                            for f in future_to_gid:
                                if not f.done():
                                    f.cancel()
                            raise
                        except Exception as e:
                            if not self._shutdown_event.is_set():
                                logger.error(f"Task group {gid} failed: {e!r}")
                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt in as_completed loop. Shutting down...")
                    self._shutdown_event.set()
                    # Cancel all remaining futures
                    for future in future_to_gid:
                        if not future.done():
                            future.cancel()
                    # Wait a bit for cancellation
                    time.sleep(1)
                    raise
        finally:
            # Restore original signal handlers
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            self._executor = None
        
        if not self._shutdown_event.is_set():
            logger.info("All task groups have been processed.")
        else:
            logger.warning("Task execution was interrupted.")
    
    def run_task_groups_parallel(
        self,
        task_groups: Mapping[str, Sequence],
        num_processes: int,
    ) -> None:
        """Run task groups in parallel (legacy method for backward compatibility)."""
        num_task_groups = len(task_groups)
        num_processes = min(num_processes, num_task_groups)
        
        task_group_ids_items = sorted(
            task_groups.items(), key=lambda x: len(x[1]), reverse=True
        )
        logger.info(f"Sorted task groups: {[x[0] for x in task_group_ids_items]}")
        
        # Set up signal handler for graceful shutdown
        future_to_gid = {}  # Will be populated in the try block
        
        def signal_handler(signum, frame):
            logger.warning("Received interrupt signal. Shutting down gracefully...")
            self._shutdown_event.set()
            if self._executor:
                # Cancel all pending futures
                try:
                    for future in list(future_to_gid.keys()):
                        if not future.done():
                            future.cancel()
                except Exception:
                    pass
                # Shutdown executor
                try:
                    if sys.version_info >= (3, 9):
                        self._executor.shutdown(wait=False, cancel_futures=True)
                    else:
                        self._executor.shutdown(wait=False)
                except Exception:
                    pass
            sys.exit(1)
        
        original_sigint = signal.signal(signal.SIGINT, signal_handler)
        original_sigterm = signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            with ProcessPoolExecutor(max_workers=num_processes) as executor:
                self._executor = executor
                future_to_gid = {
                    executor.submit(_safe_run_group, gid, tasks, self): gid
                    for gid, tasks in task_group_ids_items
                }
                
                try:
                    for future in as_completed(future_to_gid):
                        if self._shutdown_event.is_set():
                            logger.warning("Shutdown requested. Cancelling remaining tasks...")
                            break
                        
                        gid = future_to_gid[future]
                        try:
                            future.result(timeout=0.1)
                            logger.info(f"Task group {gid} finished successfully.")
                        except KeyboardInterrupt:
                            logger.warning("KeyboardInterrupt received. Shutting down...")
                            self._shutdown_event.set()
                            for f in future_to_gid:
                                if not f.done():
                                    f.cancel()
                            raise
                        except Exception as e:
                            if not self._shutdown_event.is_set():
                                logger.error(f"Task group {gid} failed: {e!r}")
                except KeyboardInterrupt:
                    logger.warning("KeyboardInterrupt in as_completed loop. Shutting down...")
                    self._shutdown_event.set()
                    for future in future_to_gid:
                        if not future.done():
                            future.cancel()
                    time.sleep(1)
                    raise
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)
            self._executor = None
        
        if not self._shutdown_event.is_set():
            logger.info("All task groups have been processed.")
        else:
            logger.warning("Task execution was interrupted.")
    
    def run_task_in_subprocess_legacy(self, task: RawSweTask, timeout_seconds: int = 5400) -> None:
        """
        Run a task in a subprocess with timeout control (legacy method).
        
        Args:
            task: Task to run (RawSweTask object)
            timeout_seconds: Timeout in seconds
        """
        task_id = task.task_id
        
        p = multiprocessing.Process(target=self.run_raw_task, args=(task,))
        p.start()
        p.join(timeout=timeout_seconds)
        if p.is_alive():
            logger.error(f"[TIMEOUT] Task {task_id} exceeded {timeout_seconds}s. Killing it...")
            # Create status.json before terminating to mark task as incomplete
            task_output_dir = pjoin(self.output_dir, f"{task_id}")
            status_file = pjoin(task_output_dir, "status.json")
            try:
                create_dir_if_not_exists(task_output_dir)
                with open(status_file, 'w', encoding='utf-8') as f:
                    json.dump({"is_finish": False, "timeout": True}, f)
                logger.info(f"Created status.json for timed-out task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to create status.json for timed-out task {task_id}: {e}")
            
            p.terminate()
            p.join()
    
    def run_task_in_subprocess(self, task_id: str, task_info: dict, timeout_seconds: int = 5400) -> None:
        """
        Run a task in a subprocess with timeout control and lazy setup.
        
        Args:
            task_id: Task identifier
            task_info: Task information dictionary
            timeout_seconds: Timeout in seconds
        """
        # Check shutdown event before starting (only if available and set)
        # In child processes, this will be a new Event that won't be shared with parent
        # So we can check it, but it won't reflect parent's shutdown state
        # The parent will terminate child processes by cancelling futures
        shutdown_event = getattr(self, '_shutdown_event', None)
        if shutdown_event and shutdown_event.is_set():
            logger.warning(f"Skipping task {task_id} due to shutdown request")
            return
        
        p = multiprocessing.Process(target=self.run_raw_task_with_setup, args=(task_id, task_info))
        p.start()
        
        # Use a loop with timeout to periodically check shutdown event
        elapsed = 0
        check_interval = 1.0  # Check every second
        
        while p.is_alive() and elapsed < timeout_seconds:
            # In child process, shutdown_event won't reflect parent's state
            # But we can still check it for local shutdown (though unlikely)
            if shutdown_event and shutdown_event.is_set():
                logger.warning(f"Shutdown requested. Terminating task {task_id}...")
                # Create status.json before terminating to mark task as incomplete
                task_output_dir = pjoin(self.output_dir, f"{task_id}")
                status_file = pjoin(task_output_dir, "status.json")
                try:
                    create_dir_if_not_exists(task_output_dir)
                    with open(status_file, 'w', encoding='utf-8') as f:
                        json.dump({"is_finish": False, "shutdown": True}, f)
                    logger.info(f"Created status.json for shutdown task {task_id}")
                except Exception as e:
                    logger.warning(f"Failed to create status.json for shutdown task {task_id}: {e}")
                
                p.terminate()
                p.join(timeout=5)
                if p.is_alive():
                    logger.warning(f"Task {task_id} did not terminate gracefully. Killing...")
                    p.kill()
                    p.join()
                return
            
            time.sleep(check_interval)
            elapsed += check_interval
        
        if p.is_alive():
            logger.error(f"[TIMEOUT] Task {task_id} exceeded {timeout_seconds}s. Killing it...")
            # Create status.json before terminating to mark task as incomplete
            task_output_dir = pjoin(self.output_dir, f"{task_id}")
            status_file = pjoin(task_output_dir, "status.json")
            try:
                create_dir_if_not_exists(task_output_dir)
                with open(status_file, 'w', encoding='utf-8') as f:
                    json.dump({"is_finish": False, "timeout": True}, f)
                logger.info(f"Created status.json for timed-out task {task_id}")
            except Exception as e:
                logger.warning(f"Failed to create status.json for timed-out task {task_id}: {e}")
            
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join()
        else:
            p.join()  # Wait for clean exit
    
    def run_raw_task_with_setup(
        self,
        task_id: str,
        task_info: dict,
        print_callback: Callable[[dict], None] | None = None,
    ) -> bool:
        """
        Setup and run a single task (setup happens just before execution).
        
        Args:
            task_id: Task identifier
            task_info: Task information dictionary
            print_callback: Optional callback for printing progress
        
        Returns:
            Whether the task completed successfully
        """
        repo_name = task_info.get('repo')
        if not repo_name:
            logger.error(f"Task {task_id} missing 'repo' field")
            return False
        
        # Setup environment (happens just before running)
        try:
            logger.info(f"Setting up environment for task {task_id}...")
            repo_cache_dir = self.setup_repo_cache(repo_name, self.setup_dir)
            task_repo_dir = self.setup_task_directory(task_id, self.setup_dir)
            
            setup_info = {
                'repo_path': task_repo_dir,
                'repo_cache_path': repo_cache_dir,
            }
            
            # Create RawSweTask now that setup is complete
            task = RawSweTask(task_id, setup_info, task_info, None)
            
        except Exception as e:
            logger.error(f"Failed to set up environment for task {task_id}: {e}")
            return False
        
        # Now run the task using existing run_raw_task logic
        return self.run_raw_task(task, print_callback)
    
    def run_raw_task(
        self,
        task: RawSweTask,
        print_callback: Callable[[dict], None] | None = None,
    ) -> bool:
        """
        High-level entry for running one task.
        
        Args:
            task: The task instance to run (RawSweTask object)
            print_callback: Optional callback for printing progress
        
        Returns:
            Whether the task completed successfully
        """
        task_id = task.task_id
        
        start_time_s = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        task_output_dir = pjoin(self.output_dir, f"{task_id}")
        
        status_file = pjoin(task_output_dir, "status.json")
        if os.path.exists(status_file):
            logger.info(f"Status file already exists for task {task_id}, skipping execution")
            return True
        elif os.path.exists(task_output_dir):
            try:
                shutil.rmtree(task_output_dir)
                logger.info(f"Cleared existing task directory {task_output_dir}")
            except Exception as e:
                logger.error(f"Error clearing task directory {task_output_dir}: {e}")
                return False
        
        create_dir_if_not_exists(task_output_dir)
        
        # Dump task metadata
        task.dump_meta_data(task_output_dir)
        
        logger.info(f"============= Running task {task_id} =============")
        
        run_ok = False
        try:
            run_ok = self.do_inference(task.to_task(), task_output_dir, print_callback)
            
            if run_ok:
                run_status_message = f"Task {task_id} completed successfully."
            else:
                run_status_message = f"Task {task_id} failed without exception."
        except Exception as e:
            logger.exception(e)
            run_status_message = f"Task {task_id} failed with exception: {e}."
        
        logger.info(run_status_message)
        return run_ok
    
    def cleanup_dangling_images(self, client: docker.DockerClient) -> None:
        """
        Remove all Docker images with <none> tag (dangling images).
        
        Args:
            client: Docker client instance
        """
        try:
            # Get all images including dangling ones
            all_images = client.images.list(all=True)
            removed_count = 0
            
            for image in all_images:
                # Check if image has no tags or only <none> tags
                # Dangling images have empty tags list or tags like '<none>:<none>'
                is_dangling = (
                    not image.tags or 
                    all(tag.startswith('<none>') or tag == '<none>:<none>' for tag in image.tags)
                )
                
                if is_dangling:
                    try:
                        client.images.remove(image.id, force=True)
                        removed_count += 1
                        logger.debug(f"Removed dangling image: {image.id[:12]}")
                    except docker.errors.ImageNotFound:
                        # Image already removed, skip
                        pass
                    except docker.errors.APIError as e:
                        # Some images may be in use (e.g., used by containers), skip them
                        logger.debug(f"Could not remove image {image.id[:12]}: {e}")
                        pass
                    except Exception as e:
                        logger.warning(f"Unexpected error removing image {image.id[:12]}: {e}")
                        pass
            
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} dangling image(s)")
        except Exception as e:
            logger.warning(f"Error cleaning up dangling images: {e}")

    def do_inference(
        self,
        python_task: Task,
        task_output_dir: str,
        print_callback: Callable[[dict], None] | None = None,
    ) -> bool:
        """
        Execute inference using AgentsManager.
        
        Args:
            python_task: Task instance
            task_output_dir: Output directory
            print_callback: Optional callback for printing progress
        
        Returns:
            Whether the inference completed successfully
        """
        client = docker.from_env()
        create_dir_if_not_exists(task_output_dir)
        
        commit_hash = python_task.commit
        clone_repo_and_checkout(
            python_task.repo_cache_path,
            commit_hash,
            python_task.project_path
        )
        
        start_time = datetime.now()
        
        # Prepare agent-specific LLM configurations
        agent_configs = self.agent_llm_configs if hasattr(self, 'agent_llm_configs') else None
        
        try:
            environment_setup_agent = EnvironmentSetupAgent(
                python_task,
                task_output_dir,
                client,
                start_time,
                self.max_iteration_num,
                disable_context_retrieval=self.disable_context_retrieval,
                using_ubuntu_only=self.using_ubuntu_only,
                agent_configs=agent_configs,
            )
            environment_setup_agent.run_workflow()
            run_ok = True
            end_time = datetime.now()
            self.dump_cost(start_time, end_time, task_output_dir, python_task.project_path)
        finally:
            if hasattr(python_task, 'remove_project'):
                python_task.remove_project()
            # Clean up dangling images before closing client
            try:
                self.cleanup_dangling_images(client)
            except Exception as e:
                logger.warning(f"Failed to cleanup dangling images: {e}")
            if client:
                client.close()
        
        return run_ok
    
    def dump_cost(
        self,
        start_time: datetime,
        end_time: datetime,
        task_output_dir: str,
        project_path: str,
    ) -> None:
        """
        Dump cost statistics.
        
        Args:
            start_time: Start time
            end_time: End time
            task_output_dir: Output directory
            project_path: Project path
        """
        try:
            from agent_gem.agents.code_agent.environment_setup_utils import (
                cd,
                get_current_commit_hash,
            )
            with cd(project_path):
                commit_hash = get_current_commit_hash()
        except Exception:
            commit_hash = "unknown"
        
        try:
            from agent_gem.agents.code_agent.environment_setup_utils import get_model_adapter
            model_adapter = get_model_adapter()
            model_stats = model_adapter.get_overall_exec_stats()
        except Exception:
            model_stats = {
                "model": "unknown",
                "total_cost": 0.0,
                "total_tokens": 0,
            }
        
        stats = {
            "commit": commit_hash,
            "start_epoch": start_time.timestamp(),
            "end_epoch": end_time.timestamp(),
            "elapsed_seconds": (end_time - start_time).total_seconds(),
        }
        stats.update(model_stats)
        
        with open(pjoin(task_output_dir, "cost.json"), "w") as f:
            json.dump(stats, f, indent=4)


def _safe_run_group_with_lazy_setup(gid: str, tasks: Sequence[tuple[str, dict]], executor: TaskExecutor) -> None:
    """
    Wrapper to run one task group inside a child process with lazy setup.
    
    Args:
        gid: Group ID
        tasks: Tasks in the group, each is (task_id, task_info) tuple
        executor: Task executor instance
    """
    try:
        run_task_group_with_lazy_setup(gid, tasks, executor)
    except Exception as e:
        raise RuntimeError(f"Group {gid} execution failed: {e!r}") from e


def run_task_group_with_lazy_setup(task_group_id: str, task_group_items: list, executor: TaskExecutor) -> None:
    """
    Run all tasks in a task group sequentially with lazy setup.
    
    Args:
        task_group_id: Group ID
        task_group_items: Tasks in the group, each is (task_id, task_info) tuple
        executor: Task executor instance
    """
    logger.info(
        f"Starting process for task group {task_group_id}. "
        f"Number of tasks: {len(task_group_items)}."
    )
    for task_id, task_info in task_group_items:
        executor.run_task_in_subprocess(task_id, task_info)
        logger.info(f"Task {task_id} completed.")
    
    logger.info(f"Finished task group {task_group_id}.")


def _safe_run_group(gid: str, tasks: Sequence, executor: TaskExecutor) -> None:
    """
    Wrapper to run one task group inside a child process (legacy method).
    
    Args:
        gid: Group ID
        tasks: Tasks in the group (RawSweTask objects)
        executor: Task executor instance
    """
    try:
        run_task_group(gid, tasks, executor)
    except Exception as e:
        raise RuntimeError(f"Group {gid} execution failed: {e!r}") from e


def run_task_group(task_group_id: str, task_group_items: list, executor: TaskExecutor) -> None:
    """
    Run all tasks in a task group sequentially (legacy method).
    
    Args:
        task_group_id: Group ID
        task_group_items: Tasks in the group (RawSweTask objects)
        executor: Task executor instance
    """
    logger.info(
        f"Starting process for task group {task_group_id}. "
        f"Number of tasks: {len(task_group_items)}."
    )
    for task in task_group_items:
        executor.run_task_in_subprocess_legacy(task)
        task_id = task.task_id
        logger.info(f"Task {task_id} completed.")
    
    logger.info(f"Finished task group {task_group_id}.")
