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
from datetime import datetime
from os.path import join as pjoin
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
    
    # Environment setup parameters
    tasks_map_file: Optional[str] = None
    setup_dir: Optional[str] = None
    use_cache: bool = True
    cache_name_format: str = "{repo}_cache"
    task_dir_name_format: str = "{task_id}_{timestamp}"
    timestamp_format: str = "%Y-%m-%d_%H-%M-%S"
    
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
        tasks_map_file: str | None = None,
        setup_dir: str | None = None,
        use_cache: bool = True,
        cache_name_format: str = "{repo}_cache",
        task_dir_name_format: str = "{task_id}_{timestamp}",
        timestamp_format: str = "%Y-%m-%d_%H-%M-%S",
    ):
        """
        Initialize the task executor.
        
        Args:
            output_dir: Output directory for results
            max_iteration_num: Maximum number of iterations
            disable_context_retrieval: Whether to disable context retrieval
            using_ubuntu_only: Whether to use Ubuntu-only base images
            tasks_map_file: Path to tasks map file (JSON or JSONL)
            setup_dir: Directory for repository cache and task directories
            use_cache: Whether to use existing cache
            cache_name_format: Format string for cache directory names
            task_dir_name_format: Format string for task directory names
            timestamp_format: Format string for timestamps
        """
        self.output_dir = output_dir
        self.max_iteration_num = max_iteration_num
        self.disable_context_retrieval = disable_context_retrieval
        self.using_ubuntu_only = using_ubuntu_only
        
        # Environment setup parameters
        self.tasks_map_file = tasks_map_file
        self.setup_dir = setup_dir
        self.use_cache = use_cache
        self.cache_name_format = cache_name_format
        self.task_dir_name_format = task_dir_name_format
        self.timestamp_format = timestamp_format
    
    @classmethod
    def load_config_from_yaml(cls, config_path: str) -> TaskExecutorConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the YAML configuration file
        
        Returns:
            TaskExecutorConfig object
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
        
        # Support backward compatibility: check environment_setup if not in task_execution
        env_setup_config = config_dict.get('environment_setup', {})
        if env_setup_config and not task_execution_config.get('tasks_map_file'):
            # Fallback to environment_setup for backward compatibility
            cache_config = env_setup_config.get('cache', cache_config)
            task_dir_config = env_setup_config.get('task_dir', task_dir_config)
        
        return TaskExecutorConfig(
            output_dir=task_execution_config.get('output_dir', ''),
            max_iteration_num=task_execution_config.get('max_iteration_num', 15),
            disable_context_retrieval=agents_config.get('disable_context_retrieval', False),
            using_ubuntu_only=agents_config.get('using_ubuntu_only', False),
            tasks_map_file=task_execution_config.get('tasks_map_file') or env_setup_config.get('tasks_map_file'),
            setup_dir=task_execution_config.get('setup_dir') or env_setup_config.get('setup_dir'),
            use_cache=cache_config.get('use_cache', True),
            cache_name_format=cache_config.get('cache_name_format', '{repo}_cache'),
            task_dir_name_format=task_dir_config.get('name_format', '{task_id}_{timestamp}'),
            timestamp_format=task_dir_config.get('timestamp_format', '%Y-%m-%d_%H-%M-%S'),
            task_id=task_execution_config.get('task_id') or env_setup_config.get('task_id'),
            task_list_file=task_execution_config.get('task_list_file') or env_setup_config.get('task_list_file'),
            task_batch=task_execution_config.get('task_batch', env_setup_config.get('task_batch', -1)),
            batch_index=task_execution_config.get('batch_index', env_setup_config.get('batch_index', -1)),
        )
    
    def load_tasks_map(self, tasks_map_file: str) -> Dict[str, Any]:
        """
        Load a .jsonl or .json file and return a dict: {instance_id: instance_dict}.
        
        Args:
            tasks_map_file: Path to the tasks map file (JSON or JSONL format)
        
        Returns:
            Dictionary mapping instance_id to instance_dict
        """
        if tasks_map_file.endswith('.jsonl'):
            with open(tasks_map_file, 'r', encoding='utf-8') as f:
                instances = [json.loads(line) for line in f if line.strip()]
        else:
            with open(tasks_map_file, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                if isinstance(obj, dict):
                    return obj
                elif isinstance(obj, list):
                    instances = obj
                else:
                    raise ValueError(
                        f"Unsupported JSON structure in file: {tasks_map_file}"
                    )
        
        return {
            inst["instance_id"]: inst
            for inst in instances
            if "instance_id" in inst
        }
    
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
        task_id: Optional[str] = None,
        task_list_file: Optional[str] = None,
        task_batch: int = -1,
        batch_index: int = -1,
        num_processes: int = 1,
        organize_output: bool = False,
    ) -> None:
        """
        Load tasks from config, set up environments, and run them.
        
        Args:
            task_id: Single task ID to process
            task_list_file: Path to file containing task IDs (one per line)
            task_batch: Batch size (number of tasks per batch)
            batch_index: Batch index (1-based, e.g., 1 for first batch)
            num_processes: Number of parallel processes
            organize_output: Whether to organize output
        """
        if not self.tasks_map_file:
            raise ValueError("tasks_map_file is required")
        if not self.setup_dir:
            raise ValueError("setup_dir is required")
        
        # Validate task selection parameters
        selection_count = sum([
            task_id is not None,
            task_list_file is not None,
            batch_index > 0 and task_batch > 0
        ])
        if selection_count > 1:
            raise ValueError(
                "Only one of task_id, task_list_file, or (task_batch, batch_index) "
                "can be specified"
            )
        
        # Load tasks map
        tasks_map = self.load_tasks_map(self.tasks_map_file)
        all_task_ids = []
        tasks_map_key_list = list(tasks_map.keys())
        
        # Determine which tasks to process
        if task_list_file is not None:
            all_task_ids = self.parse_task_list_file(task_list_file)
        elif task_id is not None:
            all_task_ids = [task_id]
        elif batch_index > 0 and task_batch > 0:
            total = len(tasks_map_key_list)
            num_batches = (total + task_batch - 1) // task_batch
            
            if batch_index < 1 or batch_index > num_batches:
                raise ValueError(
                    f"batch_index {batch_index} out of range "
                    f"(should be 1 ~ {num_batches})"
                )
            
            start = (batch_index - 1) * task_batch
            end = min(batch_index * task_batch, total)
            all_task_ids = tasks_map_key_list[start:end]
        else:
            all_task_ids = list(tasks_map.keys())
        
        if len(all_task_ids) == 0:
            raise ValueError("No task ids to run.")
        
        # Check if all task ids are in the tasks map
        missing_task_ids = [x for x in all_task_ids if x not in tasks_map]
        if missing_task_ids:
            for task_id in sorted(missing_task_ids):
                logger.warning(f"Skipping task {task_id} which was not found in tasks map.")
            all_task_ids = [x for x in all_task_ids if x not in missing_task_ids]
        
        all_task_ids = sorted(all_task_ids)
        
        # Set up environment for each task and create RawSweTask objects
        all_tasks = []
        for task_id in all_task_ids:
            try:
                task_info = tasks_map[task_id]
                repo_name = task_info.get('repo')
                if not repo_name:
                    raise ValueError(f"Task {task_id} missing 'repo' field in task_info")
                
                repo_cache_dir = self.setup_repo_cache(repo_name, self.setup_dir)
                task_repo_dir = self.setup_task_directory(task_id, self.setup_dir)
                
                setup_info = {
                    'repo_path': task_repo_dir,
                    'repo_cache_path': repo_cache_dir,
                }

                task = RawSweTask(task_id, setup_info, task_info, None)
                all_tasks.append(task)
            except Exception as e:
                logger.error(f"Failed to set up environment for task {task_id}: {e}")
                raise
        
        logger.info(f"Setup {len(all_tasks)} tasks")
        
        # Group tasks (simple grouping by index)
        task_groups = {}
        for idx, task in enumerate(all_tasks):
            task_groups[str(idx)] = [task]
        
        # Run tasks
        self.run_task_groups(task_groups, num_processes, organize_output)
    
    def run_task_groups(
        self,
        task_groups: Mapping[str, Sequence],
        num_processes: int = 1,
        organize_output: bool = False,
    ) -> None:
        """
        Run task groups in parallel or serial.
        
        Args:
            task_groups: Dictionary of task groups
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
            self.run_tasks_serial(all_tasks)
            logger.info("Finished all tasks sequentially.")
        else:
            self.run_task_groups_parallel(task_groups, num_processes)
        
        if organize_output:
            # post-process completed experiments to get input file to SWE-bench
            logger.info("Post-processing completed experiment results.")
            # swe_input_file = self.organize_and_form_input(self.output_dir)
            # logger.info(f"SWE-Bench input file created: {swe_input_file}")
    
    def run_tasks_serial(self, tasks: list) -> None:
        """Run tasks serially."""
        for task in tasks:
            self.run_task_in_subprocess(task)
    
    def run_task_groups_parallel(
        self,
        task_groups: Mapping[str, Sequence],
        num_processes: int,
    ) -> None:
        """Run task groups in parallel."""
        num_task_groups = len(task_groups)
        num_processes = min(num_processes, num_task_groups)
        
        task_group_ids_items = sorted(
            task_groups.items(), key=lambda x: len(x[1]), reverse=True
        )
        logger.info(f"Sorted task groups: {[x[0] for x in task_group_ids_items]}")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            future_to_gid = {
                executor.submit(_safe_run_group, gid, tasks, self): gid
                for gid, tasks in task_group_ids_items
            }
            
            for future in as_completed(future_to_gid):
                gid = future_to_gid[future]
                try:
                    future.result()
                    logger.info(f"Task group {gid} finished successfully.")
                except Exception as e:
                    logger.error(f"Task group {gid} failed: {e!r}")
        
        logger.info("All task groups have been processed.")
    
    def run_task_in_subprocess(self, task: RawSweTask, timeout_seconds: int = 5400) -> None:
        """
        Run a task in a subprocess with timeout control.
        
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
            p.terminate()
            p.join()
    
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
        
        try:
            environment_setup_agent = EnvironmentSetupAgent(
                python_task,
                task_output_dir,
                client,
                start_time,
                self.max_iteration_num,
                disable_context_retrieval=self.disable_context_retrieval,
                using_ubuntu_only=self.using_ubuntu_only,
            )
            environment_setup_agent.run_workflow()
            run_ok = True
            end_time = datetime.now()
            self.dump_cost(start_time, end_time, task_output_dir, python_task.project_path)
        finally:
            if hasattr(python_task, 'remove_project'):
                python_task.remove_project()
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


def _safe_run_group(gid: str, tasks: Sequence, executor: TaskExecutor) -> None:
    """
    Wrapper to run one task group inside a child process.
    
    Args:
        gid: Group ID
        tasks: Tasks in the group
        executor: Task executor instance
    """
    try:
        run_task_group(gid, tasks, executor)
    except Exception as e:
        raise RuntimeError(f"Group {gid} execution failed: {e!r}") from e


def run_task_group(task_group_id: str, task_group_items: list, executor: TaskExecutor) -> None:
    """
    Run all tasks in a task group sequentially.
    
    Args:
        task_group_id: Group ID
        task_group_items: Tasks in the group
        executor: Task executor instance
    """
    logger.info(
        f"Starting process for task group {task_group_id}. "
        f"Number of tasks: {len(task_group_items)}."
    )
    for task in task_group_items:
        executor.run_task_in_subprocess(task)
        task_id = task.task_id
        logger.info(f"Task {task_id} completed.")
    
    logger.info(f"Finished task group {task_group_id}.")
