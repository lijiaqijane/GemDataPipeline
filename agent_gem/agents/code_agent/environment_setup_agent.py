"""
Agents Manager for code_agent module.

This module provides the EnvironmentSetupAgent to orchestrate all agents' workflow,
adapted from app.agents.agents_manager.
"""

from __future__ import annotations

import os
import re
import json
import random
import logging
from datetime import datetime
from os.path import join as pjoin
from copy import deepcopy
from typing import Any

import docker
from packaging import version
from filelock import FileLock

from agent_gem.agents.code_agent.environment_setup_utils import Task
from agent_gem.agents.code_agent.environment_setup_utils.context_retrieval_agent import ContextRetrievalAgent
from agent_gem.agents.code_agent.environment_setup_utils.write_dockerfile_agent import WriteDockerfileAgent
from agent_gem.agents.code_agent.environment_setup_utils.write_eval_script_agent import WriteEvalScriptAgent
from agent_gem.agents.code_agent.environment_setup_utils.test_analysis_agent import TestAnalysisAgent

from agent_gem.agents.code_agent.environment_setup_utils import get_model_adapter

logger = logging.getLogger(__name__)

DIFF_MODIFIED_FILE_REGEX = r"--- a/(.*)"
DIFF_DEVNULL_REGEX = r"--- /dev/null\n\+\+\+ b/(.*)"

class EnvironmentSetupAgent:
    """
    Simple manager to orchestrate LLM-based agents.
    """
    
    def __init__(
        self,
        task: Task,
        output_dir: str,
        client: docker.DockerClient,
        start_time: datetime,
        max_iteration_num: int,
        disable_context_retrieval: bool = False,
        using_ubuntu_only: bool = False,
    ):
        """
        Initialize the environment setup agent.
        
        Args:
            task: Task instance
            output_dir: Output directory for results
            client: Docker client
            start_time: Start time for cost tracking
            max_iteration_num: Maximum number of iterations
            disable_context_retrieval: Whether to disable context retrieval
            using_ubuntu_only: Whether to use Ubuntu-only base images
        """
        self.task = task
        self.output_dir = os.path.abspath(output_dir)
        self.run_count = 0
        self.client = client
        self.max_iteration_num = max_iteration_num
        self.start_time = start_time
        
        self.test_files = self.get_test_files()
        self.repo_basic_info = self.get_repository_basic_info()
        self.workflow_finish_status = False
        
        # Initialize agents
        self.agents_dict = {
            "write_docker_agent": WriteDockerfileAgent(
                task, output_dir, self.repo_basic_info, using_ubuntu_only
            ),
            "write_eval_script_agent": WriteEvalScriptAgent(
                task, output_dir, self.repo_basic_info
            ),
            "test_analysis_agent": TestAnalysisAgent(
                task, output_dir, self.repo_basic_info, client
            ),
            "context_retrieval_agent": ContextRetrievalAgent(
                task, output_dir, self.repo_basic_info
            ),
        }
        self.set_agent_status('all', False)
        self.disable_context_retrieval = disable_context_retrieval
        
        if disable_context_retrieval:
            self.set_agent_status("context_retrieval_agent", True)
        
        self.agents_dict['test_analysis_agent'].disable_context_retrieval = disable_context_retrieval
        
    
    def set_agent_status(self, agent_name: str, status: bool):
        """Set the status of an agent to control if it's active or inactive."""
        if agent_name == 'all':
            for agent_key, agent_value in self.agents_dict.items():
                agent_value.finish_status = status
        elif agent_name in self.agents_dict:
            agent = self.agents_dict[agent_name]
            agent.finish_status = status
        else:
            logger.error(f"Agent {agent_name} not found!")
    
    def get_agent_status(self, agent_name: str) -> bool:
        """Get the current status of an agent."""
        if agent_name in self.agents_dict:
            return self.agents_dict[agent_name].finish_status
        else:
            logger.error(f"Agent {agent_name} not found!")
            return False
    
    def set_agents_iteration_num(self, iteration_num: int) -> None:
        """Set the iteration number for all agents."""
        for agent_key, agent_value in self.agents_dict.items():
            agent_value.iteration_num = iteration_num
    
    def get_test_files(self) -> list[str]:
        """
        Extract modified/deleted files via '--- a/...' and newly added files via '/dev/null' pattern.
        Returns combined list in patch order (no dedup).
        """
        patch = self.task.test_patch
        old_paths = re.findall(DIFF_MODIFIED_FILE_REGEX, patch)
        new_paths = re.findall(DIFF_DEVNULL_REGEX, patch)
        return old_paths + new_paths
    
    def get_repository_basic_info(self) -> str:
        """Get basic repository information."""
        version_str = f"Version: {self.task.version}\n" if self.task.version else ""
        return (
            f"Target repository name: {self.task.repo_name}\n"
            f"Commit SHA: {self.task.commit}\n"
            + version_str
            + "Target test files:\n"
            + "\n".join(self.test_files)
            + "\n"
        )
    
    def dump_cost(self):
        """Dump cost statistics."""
        start_time = self.start_time
        end_time = datetime.now()
        task_output_dir = self.output_dir
        project_path = self.task.project_path
        
        model_adapter = get_model_adapter()
        model_stats = model_adapter.get_overall_exec_stats()
        
        stats = {
            "start_epoch": start_time.timestamp(),
            "end_epoch": end_time.timestamp(),
            "elapsed_seconds": (end_time - start_time).total_seconds(),
        }
        stats.update(model_stats)
        
        with open(pjoin(task_output_dir, "cost.json"), "w") as f:
            json.dump(stats, f, indent=4)
    
    
    def run_workflow(self) -> None:
        """Run the main workflow coordinating all agents."""
        for iteration_num in range(self.max_iteration_num):
            self.set_agents_iteration_num(iteration_num)
            
            if self.disable_context_retrieval and iteration_num == 0:
                readme_content = self.agents_dict['context_retrieval_agent'].browse_readme()
                if readme_content:
                    self.agents_dict['write_eval_script_agent'].add_user_message(readme_content)
                    self.agents_dict['write_docker_agent'].add_user_message(readme_content)
            
            # Step 1: Context Retrieval
            if not self.get_agent_status("context_retrieval_agent"):
                collected_information, summary, success = (
                    self.agents_dict['context_retrieval_agent'].run_task()
                )
                self.dump_cost()
                if collected_information is not None:
                    self.set_agent_status("context_retrieval_agent", True)
                    self.agents_dict['write_eval_script_agent'].add_user_message(collected_information)
                    self.agents_dict['write_docker_agent'].add_user_message(collected_information)
            
            # Step 2: Write Dockerfile
            if (self.get_agent_status("context_retrieval_agent") and
                not self.get_agent_status("write_docker_agent")):
                _, _, success = self.agents_dict['write_docker_agent'].run_task()
                self.dump_cost()
                if success:
                    self.set_agent_status("write_docker_agent", True)
            
            # Step 3: Write Eval Script
            if (self.get_agent_status("context_retrieval_agent") and
                self.get_agent_status("write_docker_agent") and
                not self.get_agent_status("write_eval_script_agent")):
                self.agents_dict['write_eval_script_agent'].dockerfile = (
                    self.agents_dict['write_docker_agent'].get_latest_dockerfile()
                )
                _, _, success = self.agents_dict['write_eval_script_agent'].run_task()
                self.dump_cost()
                if success:
                    self.set_agent_status("write_eval_script_agent", True)
            
            # Step 4: Test Analysis
            if (self.get_agent_status("context_retrieval_agent") and
                self.get_agent_status("write_docker_agent") and
                self.get_agent_status("write_eval_script_agent")):
                dockerfile = self.agents_dict['write_docker_agent'].get_latest_dockerfile()
                eval_script_skeleton = (
                    self.agents_dict['write_eval_script_agent'].get_latest_eval_script_skeleton()
                )
                eval_script = self.agents_dict['write_eval_script_agent'].get_latest_eval_script()
                self.agents_dict['test_analysis_agent'].dockerfile = dockerfile
                self.agents_dict['test_analysis_agent'].eval_script_skeleton = eval_script_skeleton
                self.agents_dict['test_analysis_agent'].eval_script = eval_script
                
                analysis, _, success = (
                    self.agents_dict['test_analysis_agent'].run_task(self.disable_context_retrieval)
                )
                self.dump_cost()
                
                if isinstance(analysis, str):
                    try:
                        analysis = json.loads(analysis)
                    except Exception:
                        analysis = {}
                else:
                    analysis = {}
                
                is_finish = analysis.get("is_finish", None)
                if is_finish:
                    self.workflow_finish_status = True
                    break
                
                # Handle feedback from test analysis
                guidance_for_context_retrieval_agent = analysis.get(
                    "guidance_for_context_retrieval_agent", None
                )
                if guidance_for_context_retrieval_agent:
                    prefix_prompt = (
                        "After setting up dockerfile and running tests, the test log analysis agent "
                        "find that there is other context information need to collect. Here is his analysis:\n"
                    )
                    self.set_agent_status("context_retrieval_agent", False)
                    self.agents_dict['context_retrieval_agent'].add_user_message(
                        f'{prefix_prompt}{guidance_for_context_retrieval_agent}\n\n'
                    )
                
                guidance_for_write_dockerfile_agent = analysis.get(
                    "guidance_for_write_dockerfile_agent", None
                )
                if guidance_for_write_dockerfile_agent:
                    prefix_prompt = (
                        'After setting up dockerfile and running tests, the test log analysis agent '
                        'find that there is a problem with dockefile. Here is his analysis:\n'
                    )
                    self.set_agent_status("write_docker_agent", False)
                    self.agents_dict['write_docker_agent'].add_user_message(
                        f'{prefix_prompt}{guidance_for_write_dockerfile_agent}\n\n'
                    )
                
                guidance_for_write_eval_script_agent = analysis.get(
                    "guidance_for_write_eval_script_agent", None
                )
                if guidance_for_write_eval_script_agent:
                    prefix_prompt = (
                        'After setting up dockerfile and running tests, the test log analysis agent '
                        'find that there is a problem with eval script. Here is his analysis:\n'
                    )
                    self.set_agent_status("write_eval_script_agent", False)
                    self.agents_dict['write_eval_script_agent'].add_user_message(
                        f'{prefix_prompt}{guidance_for_write_eval_script_agent}\n\n'
                    )
        else:
            logger.info("Exceed largest number of tries..")
        
        # Save final results
        dockerfile_content = self.agents_dict['write_docker_agent'].get_latest_dockerfile()
        eval_script_content = self.agents_dict['write_eval_script_agent'].get_latest_eval_script()
        eval_script_skeleton_content = (
            self.agents_dict['write_eval_script_agent'].get_latest_eval_script_skeleton()
        )
        
        if dockerfile_content and eval_script_content:
            with open(os.path.join(self.output_dir, "Dockerfile"), "w") as dockerfile_f:
                dockerfile_f.write(dockerfile_content)
            with open(os.path.join(self.output_dir, "eval.sh"), "w") as eval_script_f:
                eval_script_f.write(eval_script_content)
        
        with open(os.path.join(self.output_dir, "status.json"), "w") as status_file_f:
            json.dump({"is_finish": self.workflow_finish_status}, status_file_f)
    
