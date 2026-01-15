"""
Test Analysis Utilities.

This module provides utilities for test analysis, adapted from
app.agents.test_analysis_agent.test_analysis_utils.
"""

from __future__ import annotations

import re
import json
import logging
from typing import Any
from collections.abc import Callable

from pathlib import Path
import threading

logger_lock = threading.Lock()

from ..message_thread import MessageThread
from ..model_adapter import get_model_adapter

try:
    from app.post_process import ExtractStatus, is_valid_json
except ImportError:
    ExtractStatus = type('ExtractStatus', (), {
        'IS_VALID_JSON': 'IS_VALID_JSON',
        'NOT_VALID_JSON': 'NOT_VALID_JSON',
    })()
    def is_valid_json(json_str: str):
        try:
            data = json.loads(json_str)
            return ExtractStatus.IS_VALID_JSON, data
        except json.JSONDecodeError:
            return ExtractStatus.NOT_VALID_JSON, None

logger = logging.getLogger(__name__)


def setup_logger(instance_id: str, log_file: Path, mode="w"):
    """
    This logger is used for logging the build process of images and containers.
    It writes logs to the log file.
    """
    with logger_lock:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        new_logger = logging.getLogger(f"{instance_id}.{log_file.name}")
        handler = logging.FileHandler(log_file, mode=mode)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        new_logger.addHandler(handler)
        new_logger.setLevel(logging.INFO)
        new_logger.propagate = False
        setattr(new_logger, "log_file", log_file)
        return new_logger

def close_logger(new_logger):
    # To avoid too many open files
    with logger_lock:
        for handler in new_logger.handlers:
            handler.close()
            new_logger.removeHandler(handler)


SYSTEM_PROMPT = """You are an expert in analyzing logs of test execution script and dockerfile construction.  
Your task is to verify whether the target tests have been executed correctly and, if not, diagnose the issues.
Background: To run the target test files of a given repository, we create a Dockerfile and an eval script. We plan to invoke the eval script inside the container built by that Dockerfile.

You will:
1. Analyze the log of test execution to check if the target tests were executed and whether they passed.
2. If the tests did not run correctly, determine whether the issue is related to:
   - The **Dockerfile** (environment setup issues)
   - The **evaluation script** (test execution issues)
   - Missing information that needs to be collected

3. Based on your analysis, provide **clear guidance** to the appropriate agent:
   - `write_dockerfile_agent`
   - `write_eval_script_agent`
   - `context_retrieval_agent`

Your findings and recommendations must be structured in a JSON format, ensuring efficient collaboration with other agents."""

SYSTEM_PROMPT_WITHOUT_CONTEXT_RETRIEVAL = """You are an expert in analyzing logs of test execution script and dockerfile construction.  
Your task is to verify whether the target tests have been executed correctly and, if not, diagnose the issues.
Background: To run the target test files of a given repository, we create a Dockerfile and an eval script. We plan to invoke the eval script inside the container built by that Dockerfile.

Your task is : 

1. Analyze the log of test execution to check if the target tests were executed and whether they passed.

2. If the tests did not run correctly, determine whether the issue is related to:
   - The **Dockerfile** (environment setup issues)
   - The **evaluation script** (test execution issues)
   
3. Based on your analysis, provide **clear guidance** to the appropriate agent:
   - `write_dockerfile_agent`
   - `write_eval_script_agent`
   
Your findings and recommendations must be structured in a JSON format, ensuring efficient collaboration with other agents."""


ANALYZE_PROMPT = """
Given the test log and the target tests, analyze the results and determine the next steps. But if the dockerfile is not built successfully, you should analyze what issues happen.

### **Step 1: Verify Test Execution**
- Identify which test files were added or modified by the eval script.
- Confirm that those tests were actually executed (they appear in the test log).
- Check their pass/fail status:
   - If all tests passed, report success.
- Ensure there is at least some test output in the log:
   - If no test output is found, set `is_finish = false` and include an instruction for write_eval_script_agent to revise the eval script so that tests actually run.

### **Step 2: Identify Problems**
- If the tests failed due to **environment setup issues**, analyze whether the problem comes from:
  - The **Dockerfile** (e.g., incorrect dependencies, wrong OS, missing configurations).
  - The **evaluation script** (e.g., incorrect test commands, wrong paths, missing environment activation).
- Sometimes, tests may fail due to incorrect versions of specific dependencies. Be sure to check the versions of critical dependencies to ensure compatibility.
- If there are missing dependencies or unknown errors, consider whether additional context retrieval is required.
- Tests should not be run in the Dockerfile**; skip tests during environment setup and run them in the evaluation script.
- Note that the eval script MUST catch exit code after running tests, and echo "OMNIGRIL_EXIT_CODE=$rc". This is important for judge whether tests are run successfully.

### **Step 3: Plan Corrective Actions**
- If a fix is needed in the **Dockerfile**, provide guidance to `write_dockerfile_agent` on how to fix it, always include the original error message and a brief description of what is missing or suspected to be the cause.
- If a fix is needed in the **evaluation script**, provide guidance to `write_eval_script_agent` on how to fix it, always include the original error message and a brief description of what is missing or suspected to be the cause.
- If more information from the target repository is needed, provide guidance to `context_retrieval_agent` on what to collect. Here are some instructions:
    1. Always include the original error message and a brief description of what is missing or suspected to be the cause.
    2. Clearly specify what information or files should be searched for. For environment or dependency issues, recommend files such as requirements.txt, environment.yml, Dockerfile, setup.py, pyproject.toml, etc. For test or evaluation issues, suggest looking for files such as  eval*.sh, pytest.ini, .github/workflows/*, etc. 
    3. Additionally, encourage reviewing documentation files like README.md, CONTRIBUTING.md, or any docs in the root or docs/ directory for relevant setup or testing instructions (Contributing file often contains some testing instruction).

### **Output Example**
Provide your answer in JSON format:
```json
{
    "is_finish": true/false,  # If tests passed and everything is correct, set this to true.
    "guidance_for_write_dockerfile_agent": "<Provide detailed guidance if modifications are needed>",
    "guidance_for_write_eval_script_agent": "<Provide detailed guidance if modifications are needed>",
    "guidance_for_context_retrieval_agent": "<Specify what additional information from the target repository is needed, if any>",
}
```

**Important Notes:**
- If `is_finish` is `true`, all guidance fields can be empty.
- Be specific in your guidance, providing detailed steps for the necessary fixes. Only provide guidance to the relevant agent based on the actual issue. For any agent not called, its guidance field must be empty. 
- Calling context_retrieval_agent is expensive. Only suggest using it when there is clearly missing information that is necessary to fix the Dockerfile or evaluation script. Be precise and specific in what to retrieve (e.g., particular files or configuration scripts) to avoid repeated or vague searches.
- Provide detailed error information to tell agent what errors happen.
"""

ANALYZE_PROMPT_WITHOUT_CONTEXT_RETRIEVAL = """
Given the test log and the target tests, analyze the results and determine the next steps. But if the dockerfile is not built successfully, you should analyze what issues happen.

### **Step 1: Verify Test Execution**
- Identify which test files were added or modified by the eval script.
- Confirm that those tests were actually executed (they appear in the test log).
- Check their pass/fail status:
   - If all tests passed, report success.
- Ensure there is at least some test output in the log:
   - If no test output is found, set `is_finish = false` and include an instruction for write_eval_script_agent to revise the eval script so that tests actually run.

### **Step 2: Identify Problems**
- If the tests failed due to **environment setup issues**, analyze whether the problem comes from:
  - The **Dockerfile** (e.g., incorrect dependencies, wrong OS, missing configurations).
  - The **evaluation script** (e.g., incorrect test commands, wrong paths, missing environment activation).
- Sometimes, tests may fail due to incorrect versions of specific dependencies. Be sure to check the versions of critical dependencies to ensure compatibility.
- If there are missing dependencies or unknown errors, consider whether additional context retrieval is required.
- Tests should not be run in the Dockerfile**; skip tests during environment setup and run them in the evaluation script.
- Note that the eval script MUST catch exit code after running tests, and echo "OMNIGRIL_EXIT_CODE=$rc". This is important for judge whether tests are run successfully.

### **Step 3: Plan Corrective Actions**
- If a fix is needed in the **Dockerfile**, provide guidance to `write_dockerfile_agent` on how to fix it, always include the original error message and a brief description of what is missing or suspected to be the cause.
- If a fix is needed in the **evaluation script**, provide guidance to `write_eval_script_agent` on how to fix it, always include the original error message and a brief description of what is missing or suspected to be the cause.

### **Output Example**
Provide your answer in JSON format:
```json
{
    "is_finish": true/false,  # If tests passed and everything is correct, set this to true.
    "guidance_for_write_dockerfile_agent": "<Provide detailed guidance if modifications are needed>",
    "guidance_for_write_eval_script_agent": "<Provide detailed guidance if modifications are needed>",
}
```

**Important Notes:**
- If `is_finish` is `true`, all guidance fields can be empty.
- Be specific in your guidance, providing detailed steps for the necessary fixes. Only provide guidance to the relevant agent based on the actual issue. For any agent not called, its guidance field must be empty. 
- Provide detailed error information to tell agent what errors happen.
"""


def run_with_retries( msg_thread: MessageThread, disable_context_retrieval=False,retries=3,print_callback: Callable[[dict], None] | None = None) -> tuple[str | None, list[MessageThread]]:
   
    for idx in range(1, retries + 1):
        logger.debug(
            "Trying to analyze the test log. Try {} of {}.", idx, retries
        )

        res_text = run(msg_thread,disable_context_retrieval)
        res_text = extract_json_from_response(res_text)
        # res_text = msg_threads.append(new_thread)
        res_text = res_text.lstrip('```json').rstrip('```')
        logger.debug(res_text)
        extract_status, data = is_valid_json(res_text)

        if extract_status != ExtractStatus.IS_VALID_JSON:
            logger.debug("Invalid json. Will retry.")
            continue

        valid, diagnosis = is_valid_response(data,disable_context_retrieval)
        if not valid:
            logger.debug(f"{diagnosis}. Will retry.")
            continue

        logger.debug("Extracted a valid json")
        return res_text
    return None


def run(msg_thread: MessageThread,disable_context_retrieval:bool) -> tuple[str, MessageThread]:
    """
    Run the agent to extract issue to json format.
    """

    if disable_context_retrieval:
        msg_thread.add_user(ANALYZE_PROMPT_WITHOUT_CONTEXT_RETRIEVAL)
    else:
        msg_thread.add_user(ANALYZE_PROMPT)
    res_text, *_ = get_model_adapter().call(
        msg_thread.to_msg(), response_format="json_object"
    )

    msg_thread.add_model(res_text, [])  # no tools

    return res_text


def is_valid_response(data: Any,disable_context_retrieval:bool) -> tuple[bool, str]:
    if not isinstance(data, dict):
        return False, "Json is not a dict"

    if not data.get("is_finish"):
        terminate = data.get("is_finish")
        if terminate is None:
            return False, "'is_finish' parameter is missing"

        if not isinstance(terminate, bool):
            return False, "'is_finish' parameter must be a boolean (true/false)"
    if disable_context_retrieval:
        key_list = ['guidance_for_write_dockerfile_agent',
            'guidance_for_write_eval_script_agent']
    else:
        key_list = ['guidance_for_write_dockerfile_agent',
            'guidance_for_write_eval_script_agent',
            'guidance_for_context_retrieval_agent']
    for key in key_list:
        if not data.get(key):
            terminate = data.get(key)
            if terminate is None:
                return False, f"'{key}' parameter is missing"

            if not isinstance(terminate, str):
                return False, "'{key}' parameter must be a string"

        
    return True, "OK"

def extract_json_from_response(res_text: str):
    """
    Extarct json result from the LLM response
    """
    json_extracted = None

    
    json_matches = re.findall(r"```json([\s\S]*?)```", res_text, re.IGNORECASE)
    if json_matches:
        json_extracted = json_matches[0].strip()

   
    if not json_extracted:
        json_code_blocks = re.findall(r"```([\s\S]*?)```", res_text, re.IGNORECASE)
        for content in json_code_blocks:
            clean_content = content.strip()
           
            try:
                json.loads(clean_content)  
                json_extracted = clean_content
                break
            except json.JSONDecodeError:
                continue  

    return json_extracted if json_extracted else res_text  