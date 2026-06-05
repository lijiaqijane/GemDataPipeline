"""
Write Dockerfile Agent.

This module provides the WriteDockerfileAgent for generating Dockerfiles,
adapted from app.agents.write_dockerfile_agent.write_dockerfile_agent.
"""

from __future__ import annotations

import os
import shutil
import logging
from os.path import join as pjoin

from ..base_agent import BaseAgent
from ..task_adapter import Task
from . import write_dockerfile_utils

logger = logging.getLogger(__name__)


class WriteDockerfileAgent(BaseAgent):
    """
    LLM-based agent for creating or modifying a Dockerfile via direct chat.
    Manages its own create/modify logic, output directories, and retry behavior.
    """
    
    api_functions: list[str] = []
    
    def __init__(
        self,
        task: Task,
        output_dir: str,
        repo_basic_info: str,
        using_ubuntu_only: bool = False,
    ):
        """
        Initialize the write Dockerfile agent.
        
        Args:
            task: Task instance
            output_dir: Output directory for agent results
            repo_basic_info: Basic repository information
            using_ubuntu_only: Whether to use Ubuntu-only base images
        """
        super().__init__(agent_id="WriteDockerfileAgent")
        self.task = task
        self.output_dir = os.path.abspath(output_dir)
        self.run_count = 0
        self.reference_setup = None
        self.repo_basic_info = repo_basic_info
        self.using_ubuntu_only = using_ubuntu_only
        self.init_msg_thread()
    
    def init_msg_thread(self) -> None:
        """Initialize the message thread with system and user prompts."""
        self.msg_thread = self.msg_thread.__class__()  # Create new thread
        self.add_system_message(write_dockerfile_utils.get_system_prompt_dockerfile())
        self.add_user_message(self.repo_basic_info)
    
    
    def run_task(self, print_callback=None) -> tuple[str, str, bool]:
        """
        Create or modify a Dockerfile based on the given message_thread context.
        
        Args:
            print_callback: Optional callback for printing progress
        
        Returns:
            Tuple of (output, summary, success)
        """
        logger.info(f"Iteration ROUND {self.iteration_num}: Dockerfile Generation")
        prev_dir = self.get_latest_write_dockerfile_output_dir()
        prev_file = os.path.join(prev_dir, 'Dockerfile')
        self.run_count += 1
        curr_dir = self.get_latest_write_dockerfile_output_dir()
        os.makedirs(curr_dir, exist_ok=True)
        
        # Inject either modify or init prompt
        if os.path.exists(prev_file):
            modify_prompt = write_dockerfile_utils.get_user_prompt_modify_dockerfile()
            prev_content = self._read_file(prev_file)
            self.add_user_message(f"Previous dockerfile:\n{prev_content}\n")
            self.add_user_message(modify_prompt)
        else:
            if self.using_ubuntu_only:
                self.add_user_message(
                    write_dockerfile_utils.get_user_prompt_init_dockerfile_using_ubuntu_only()
                )
            else:
                self.add_user_message(
                    write_dockerfile_utils.get_user_prompt_init_dockerfile()
                )
        
        # Delegate to the retryable writer
        task_output = write_dockerfile_utils.write_dockerfile_with_retries(
            self.msg_thread,
            curr_dir,
            self.task,
            print_callback=print_callback
        )
        
        # Post-process: validate or fallback copy
        dockerfile_path = os.path.join(curr_dir, 'Dockerfile')
        if not os.path.isfile(dockerfile_path):
            # Fallback: copy previous
            if os.path.exists(prev_file):
                shutil.copy(prev_file, dockerfile_path)
            summary = "Dockerfile generation failed."
            is_ok = False
        else:
            summary = "Dockerfile created/updated successfully."
            is_ok = True
        
        dockerfile_output_dir = self.get_latest_write_dockerfile_output_dir()
        conversation_file = pjoin(dockerfile_output_dir, "conversation.json")
        self.msg_thread.save_to_file(conversation_file)
        
        return task_output, summary, is_ok
    
    def _read_file(self, path: str) -> str:
        """Read file content."""
        try:
            with open(path, 'r') as f:
                return f.read()
        except Exception:
            return ""
    
    def get_latest_write_dockerfile_output_dir(self) -> str:
        """Return the directory of the most recent Dockerfile outputs."""
        return os.path.join(self.output_dir, f"write_dockerfile_agent_{self.run_count}")
    
    def get_latest_dockerfile(self) -> str:
        """Read and return contents of the latest generated Dockerfile."""
        path = os.path.join(self.get_latest_write_dockerfile_output_dir(), 'Dockerfile')
        try:
            with open(path, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to read latest Dockerfile at {path}: {e}")
            return ""
