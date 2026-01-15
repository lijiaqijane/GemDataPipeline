"""
Test Analysis Agent.

This module provides the TestAnalysisAgent for analyzing test results,
adapted from app.agents.test_analysis_agent.test_analysis_agent.
"""

from __future__ import annotations

import os
import json
import re
import traceback
import logging
from pathlib import Path
from os.path import join as pjoin

import docker

from ..base_agent import BaseAgent
from ..task_adapter import Task
from ..message_thread import FunctionCallIntent
from . import test_analysis_utils
from .docker_utils import (
    build_container,
    cleanup_container,
    remove_image,
    copy_to_container,
    exec_run_with_timeout,
    BuildImageError,
    EvaluationError,
)

logger = logging.getLogger(__name__)

MAX_LINE_NUM = 600
ansi_escape = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")


class TestAnalysisAgent(BaseAgent):
    """
    Agent responsible for analyzing test execution results.
    """
    
    api_functions = ["setup_docker_and_run_test"]
    
    def __init__(
        self,
        task: Task,
        output_dir: str,
        repo_basic_info: str,
        client: docker.DockerClient,
    ):
        """
        Initialize the test analysis agent.
        
        Args:
            task: Task instance
            output_dir: Output directory for agent results
            repo_basic_info: Basic repository information
            client: Docker client
        """
        super().__init__(agent_id="TestAnalysisAgent")
        self.task = task
        self.output_dir = os.path.abspath(output_dir)
        self.analysis_count = 0
        self.run_test_num = 0
        self.setup_dockerfile_num = 0
        self.repo_basic_info = repo_basic_info
        self.task_id = task.task_id.lower()
        self.client = client
        self.test_analysis_dir = os.path.join(self.output_dir, "test_analysis_agent")
        self.eval_script_skeleton = None
        self.dockerfile = None
        self.eval_script = None
        self.timeout = 3600
        self.disable_context_retrieval = False
    
    def init_msg_thread(self) -> None:
        """Initialize the message thread with system and user prompts."""
        self.msg_thread = self.msg_thread.__class__()  # Create new thread
        
        if self.disable_context_retrieval:
            system_prompt = test_analysis_utils.SYSTEM_PROMPT_WITHOUT_CONTEXT_RETRIEVAL
        else:
            system_prompt = test_analysis_utils.SYSTEM_PROMPT
        
        self.add_system_message(system_prompt)
        self.add_user_message(self.repo_basic_info)
        self.add_user_message(f'The current dockerfile used to setup environment:\n{self.dockerfile}')
        self.add_user_message(
            f'The current eval script (omit test patch to decrease length) used to run tests:\n{self.eval_script_skeleton}'
        )
    
    def get_latest_test_analysis_output_dir(self):
        output_dir = f'{self.test_analysis_dir}_{self.analysis_count}'
        return output_dir

    def get_latest_test_log(self) -> str:
        """Read the latest test_output.txt produced by run_test."""
        test_dir = self.get_latest_test_analysis_output_dir()
        path = os.path.join(test_dir, "test_output.txt")
        try:
            return Path(path).read_text()
        except FileNotFoundError:
            return ""
    
    def get_test_log_with_line_numbers(self) -> str:
        test_log = self.get_latest_test_log()
        lines = test_log.splitlines()
        
       
        width = len(str(len(lines)))
        full_formatted = [f"{i + 1:>{width}}   {line}" for i, line in enumerate(lines)]
        
        if len(full_formatted) <= MAX_LINE_NUM:
            log_body = "\n".join(full_formatted)
            return f'Test log:\n{log_body}\n\n'

        
        head_size = MAX_LINE_NUM // 2
        tail_size = MAX_LINE_NUM - head_size
        
        head = full_formatted[:head_size]
        tail = full_formatted[-tail_size:]
        
       
        omission = " " * width + "   [..., {} lines omitted ...]".format(
            len(full_formatted) - head_size - tail_size)
        
        truncated_log = "\n".join(head + [omission] + tail)
        
        return f'Test log (showing first {head_size} & last {tail_size} lines):\n{truncated_log}\n\n'
    
    def run_task(self, disable_context_retrieval: bool = False, print_callback=None) -> tuple[str, str, bool]:
        """
        Execute the test analysis task.
        
        Args:
            disable_context_retrieval: Whether to disable context retrieval
            print_callback: Optional callback for printing progress
        
        Returns:
            Tuple of (output, summary, success)
        """
        self.init_msg_thread()
        logger.info(
            f"Task {self.task.task_id} Iteration ROUND {self.iteration_num} "
            f"Try to setup docker and run tests"
        )
        
        self.analysis_count += 1
        test_log_output_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(test_log_output_dir, exist_ok=True)
        
        intent = FunctionCallIntent("setup_docker_and_run_test", {}, None)
        tool_output, _, success = self.dispatch_intent(intent)
        
        build_image_status = False
        if 'Image built successfully!' not in tool_output:
            logger.info('Build Image Failure!')
            error_msg = (
                f'We can not run tests successfully, cause we encounter some errors '
                f'when building dockerfile. As follows:\n{tool_output}\n\n'
            )
            self.add_user_message(error_msg)
        elif success:
            build_image_status = True
        
        # Analyze test results
        if build_image_status and success:
            test_log = self.get_test_log_with_line_numbers()
            self.add_user_message(test_log)
        
        analysis = test_analysis_utils.run_with_retries(
            self.msg_thread,
            disable_context_retrieval=disable_context_retrieval,
            print_callback=print_callback
        )
        
        task_output = analysis
        analysis_file = Path(f"{self.get_latest_test_analysis_output_dir()}/analysis.json")
        
        to_save = {}
        if isinstance(analysis, dict):
            to_save = analysis
        elif isinstance(analysis, str):
            try:
                to_save = json.loads(analysis)
            except Exception:
                to_save = {}
        else:
            to_save = {}
        
        if task_output is None:
            summary = "The tool returned nothing. The main agent probably did not provide enough clues."
            success = False
        else:
            summary = "The tool returned the analysis in json format generated by another agent."
            success = True
        
        with analysis_file.open("w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        
        conversation_file = pjoin(test_log_output_dir, "conversation.json")
        self.msg_thread.save_to_file(conversation_file)
        
        return task_output, summary, success
    
    def run_task_without_run_test(self, print_callback=None) -> tuple[str, str, bool]:
        """
        Run task without actually running tests (for ablation study).
        
        Args:
            print_callback: Optional callback for printing progress
        
        Returns:
            Tuple of (output, summary, success)
        """
        self.init_msg_thread()
        logger.info(
            f"Task {self.task.task_id} Iteration ROUND {self.iteration_num} "
            f"Try to analyze the test log"
        )
        
        self.analysis_count += 1
        test_log_output_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(test_log_output_dir, exist_ok=True)
        
        success = False
        analysis = test_analysis_utils.run_with_retries(
            self.msg_thread,
            print_callback=print_callback
        )
        task_output = analysis
        analysis_file = Path(f"{self.get_latest_test_analysis_output_dir()}/analysis.json")
        
        to_save = {}
        if isinstance(analysis, dict):
            to_save = analysis
        elif isinstance(analysis, str):
            try:
                to_save = json.loads(analysis)
            except Exception:
                to_save = {}
        else:
            to_save = {}
        
        if task_output is None:
            summary = "The tool returned nothing."
            success = False
        else:
            summary = "The tool returned the analysis in json format."
            success = True
        
        with analysis_file.open("w", encoding="utf-8") as f:
            json.dump(to_save, f, ensure_ascii=False, indent=2)
        
        conversation_file = pjoin(test_log_output_dir, "conversation.json")
        self.msg_thread.save_to_file(conversation_file)
        
        return task_output, summary, success
    
    def build_docker_image(
        self,
        dockerfile,
        cur_build_image_dir,
        task_id,
        image_name,
        build_image_logger,
        client
    ):
        """Build Docker image with detailed logging and error handling."""
    
        
        build_image_logger.info(
            f"Building image {task_id}\n"
            f"Using dockerfile:\n{dockerfile}\n"
        )

    

        if self.setup_dockerfile_num > 1:
            # prev_image_name = f"{task_id}:latest_{setup_dockerfile_num - 1}"
            prev_image_name = f"{self.task_id}-dockerfile{self.setup_dockerfile_num-1}:latest"
            try:
                client.images.remove(prev_image_name, force=True)
                build_image_logger.info(f"Deleted previous image: {prev_image_name}")

            except docker.errors.ImageNotFound:
                build_image_logger.info(f"Do not find previous image, images list is clean.")
            except Exception as e: 
                build_image_logger.error(f"Failed to delete previous image {prev_image_name}: {str(e)}")
        
        

        dockerfile_path = f'{cur_build_image_dir}/Dockerfile'
        with open(dockerfile_path, "w") as f:
            f.write(dockerfile)

        
        command_output = []  
        capturing = False   
        response = client.api.build(
            path=cur_build_image_dir,
            tag=image_name,
            rm=True,
            forcerm=True,
            decode=True,
            platform="linux/x86_64",
            nocache=True,
        )

        buffer = ""

       
        for chunk in response:
            if "stream" in chunk:
              
                buffer += ansi_escape.sub("", chunk["stream"]).replace("\r\n", "\n").replace("\r", "\n")
                
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if not line.strip():
                        continue

                    
                    if line.startswith("Step "):
                        last_command = line
                        command_output = [line]
                        capturing = True
                    elif capturing:
                        command_output.append(line)

                 
                    build_image_logger.info(line)

            elif "errorDetail" in chunk and capturing:
               
                if buffer.strip():
                    command_output.append(buffer.strip())
                    build_image_logger.info(buffer.strip())
                    buffer = ""

             
                error_msg = ansi_escape.sub("", chunk["errorDetail"]["message"])
                build_image_logger.error(f"Error: {error_msg}")
                command_output.append(f"Error: {error_msg}")

             
                raise docker.errors.BuildError(error_msg, build_log=command_output)

      
        if buffer.strip():
            build_image_logger.info(buffer.strip())

        build_image_logger.info("Image built successfully!")


    def setup_docker_and_run_test(
        self
    ) -> tuple[str, str, bool]:
        # building docker image first
       
        dockerfile = self.dockerfile
        
        eval_script = self.eval_script
        tool_output = ""
        summary = ""
        success = False
        self.setup_dockerfile_num += 1
        cur_build_image_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(cur_build_image_dir, exist_ok=True)
        build_image_logger = test_analysis_utils.setup_logger(self.task_id, Path(f'{cur_build_image_dir}/build_image.log'))
        # image_name = f"{self.task_id}:latest_{self.setup_dockerfile_num}"
        image_name = f"{self.task_id}-dockerfile{self.setup_dockerfile_num}:latest"
       
        try:
            self.build_docker_image(dockerfile,
                                    cur_build_image_dir,
                                   
                                    self.task_id, 
                                    image_name,
                                    build_image_logger,
                                    self.client) 
            tool_output += "Image built successfully!\n"
            summary += f"Docker image {image_name} built successfully.\n"
        except docker.errors.BuildError as e:
           
            build_log = e.build_log
            if len(build_log) > MAX_LINE_NUM:
                half = MAX_LINE_NUM // 2
                skipped = len(build_log) - MAX_LINE_NUM
                build_log = (
                    build_log[:half]
                    + [f"...skipped {skipped} lines..."]
                    + build_log[-half:]
                )
            tool_output += "\n".join(build_log)
            build_image_logger.error(e)
            summary += f"Failed to build Docker image."
            success = False
            return tool_output, summary, success
        except Exception as e:
          
            build_image_logger.error(f"Unexpected error: {str(e)}")
            tool_output += f'{str(e)}\n'
            summary += f"Unexpected error when building images."
            success = False
            return tool_output, summary, success
        finally:
            test_analysis_utils.close_logger(build_image_logger)

        test_output, test_summary, test_success = self.run_test(eval_script)
        tool_output += test_output
        summary += test_summary
        success = test_success

        return tool_output, summary, success

    # def setup_docker_and_run_test(self) -> tuple[str, str, bool]:
    #     """
    #     Setup Docker and run tests.
        
    #     Returns:
    #         Tuple of (output, summary, success)
    #     """
    #     dockerfile = self.dockerfile
    #     eval_script = self.eval_script
    #     tool_output = ""
    #     summary = ""
    #     success = False
    #     self.setup_dockerfile_num += 1
    #     cur_build_image_dir = self.get_latest_test_analysis_output_dir()
    #     os.makedirs(cur_build_image_dir, exist_ok=True)
        
    #     image_name = f"{self.task_id}-dockerfile{self.setup_dockerfile_num}:latest"
        
    #     try:
    #         # Build Docker image
    #         build_container(
    #             self.client,
    #             dockerfile,
    #             Path(cur_build_image_dir),
    #             image_name,
    #             logger=logger,
    #         )
    #         tool_output += "Image built successfully!\n"
    #         summary += f"Docker image {image_name} built successfully.\n"
    #     except BuildImageError as e:
    #         tool_output += f"Failed to build Docker image: {str(e)}\n"
    #         summary += "Failed to build Docker image."
    #         success = False
    #         return tool_output, summary, success
    #     except Exception as e:
    #         tool_output += f'Unexpected error: {str(e)}\n'
    #         summary += "Unexpected error when building images."
    #         success = False
    #         return tool_output, summary, success
        
    #     # Run tests
    #     test_output, test_summary, test_success = self.run_test(eval_script, image_name)
    #     tool_output += test_output
    #     summary += test_summary
    #     success = test_success
        
    #     return tool_output, summary, success
    
    def run_test(self, eval_script: str) -> (str, str, bool):
        tool_output = ""
        summary = ""
        success = False
        patch = self.task.patch
        self.run_test_num += 1
        self.reset_tool_sequence()
        cur_test_dir = self.get_latest_test_analysis_output_dir()
        os.makedirs(cur_test_dir, exist_ok=True)
        run_test_logger = test_analysis_utils.setup_logger(self.task_id, Path(f'{cur_test_dir}/run_test.log'))
        # test_image_name = f"{self.task_id}:latest_{self.setup_dockerfile_num}"
        test_image_name = f"{self.task_id}-dockerfile{self.setup_dockerfile_num}:latest"
        # test_container_name =  f"{self.task_id}:test_{self.run_test_num}"
        test_container_name = f"{self.task_id}-test{self.run_test_num}"
        instance_id = self.task_id
        container = None
        test_output_path = f'{cur_test_dir}/test_output.txt'
        try:
            container = build_container(self.client,test_image_name,test_container_name,instance_id,run_test_logger)

            container.start()
            run_test_logger.info(f"Container for {instance_id} started: {container.id}")
            tool_output += f"Container {container.id} started.\n"
            summary += "Container started.\n"
            # Copy model prediction as patch file to container
            patch_file = Path(f"{cur_test_dir}/patch.diff")
            patch_file.write_text(patch or "")
            run_test_logger.info(
                f"Intermediate patch for {instance_id} written to {patch_file}, now applying to container..."
            )
            copy_to_container(container, patch_file, Path("/tmp/patch.diff"))

        
            # Attempt to apply patch to container
            val = container.exec_run(
                "git apply -p1 -v /tmp/patch.diff",
                workdir="/testbed",
                user="root",
            )
            exit_code = val.exit_code
            output = val.output.decode("utf-8", errors="replace")

            if exit_code != 0:
                run_test_logger.info("Failed to apply patch to container, trying again...")
                run_test_logger.error(f"git apply returned exit_code={exit_code}. Output:\n{output}")
                # try "patch --batch --fuzz=5 -p1 -i {patch_path}" to try again
                val = container.exec_run(
                    "patch --batch --fuzz=5 -p1 -i /tmp/patch.diff",
                    workdir="/testbed",
                    user="root",
                )
                if val.exit_code != 0:
                    run_test_logger.info(f"Apply patch fail:\n{val.output.decode('utf-8')}")
                    raise EvaluationError(
                        instance_id,
                        f"Apply patch fail:\n{val.output.decode('utf-8')}. Check if you apply patch in incorrect directories.",
                        run_test_logger,
                    )
                else:
                    run_test_logger.info(f"Apply patch success:\n{val.output.decode('utf-8')}")
            else:
                run_test_logger.info(f"Apply patch success:\n{val.output.decode('utf-8')}")
            tool_output += "Patch applied successfully.\n"
            summary += "Patch applied.\n"
                    # Get git diff before running eval script
            git_diff_output_before = (
                container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
            )
            run_test_logger.info(f"Git diff before:\n{git_diff_output_before}")

            eval_file = Path(f"{self.get_latest_test_analysis_output_dir()}/eval.sh")
            eval_file.write_text(eval_script)
            run_test_logger.info(
                f"Eval script for {instance_id} written to {patch_file}, now applying to container..."
            )
            copy_to_container(container, eval_file, Path("/eval.sh"))

            # Run eval script, write output to logs
            result = exec_run_with_timeout(container, "/bin/bash /eval.sh", timeout=self.timeout)
            test_output = result.decode("utf-8")
            
            with open(test_output_path, "w") as f:
                f.write(test_output)
            run_test_logger.info(f"Test output for {instance_id} written to {test_output_path}")

            # Get git diff after running eval script
            git_diff_output_after = (
                container.exec_run("git diff", workdir="/testbed").output.decode("utf-8").strip()
            )

            # Check if git diff changed after running eval script
            run_test_logger.info(f"Git diff after:\n{git_diff_output_after}")
            if git_diff_output_after != git_diff_output_before:
                run_test_logger.info(f"Git diff changed after running eval script")
                tool_output += "Note: Git diff changed after test execution.\n"
                summary += "Git diff changed.\n"

        except EvaluationError as e:
            error_msg = (f"EvaluationError {instance_id}: {e}\n"
                        f"{traceback.format_exc()}\n"
                        f"Check ({run_test_logger.log_file}) for more information.")
            run_test_logger.info(error_msg)
            tool_output += error_msg + "\n"
            summary += "Evaluation error occurred.\n"
            success = False
           
        except Exception as e:
            error_msg = (f"Error in evaluating model for {instance_id}: {e}\n"
                        f"{traceback.format_exc()}\n"
                        f"Check ({run_test_logger.log_file}) for more information.")
            run_test_logger.info(error_msg)
            tool_output += error_msg + "\n"
            summary += "Unexpected error occurred.\n"
            success = False
        else:
            if not os.path.exists(test_output_path):
                tool_output += "Do not generate test_output.txt. Please check the correctness of dockerfile and eval script.\n"
                summary += 'Fail to obtain test results.'
                success = False
            else:
                tool_output += f"Find test_output.txt! Waiting for analysis. "
                summary += 'Obtain test results successfully.'
                success = True

        finally:
           
            # Remove instance container + image, close logger
            cleanup_container(self.client, container,run_test_logger)
            
            remove_image(self.client, test_image_name, run_test_logger)
            test_analysis_utils.close_logger(run_test_logger)
        self.dump_tool_sequence(self.get_latest_test_analysis_output_dir())
        return tool_output, summary, success
