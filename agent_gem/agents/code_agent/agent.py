from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from agent_gem.core.task_schema import TaskPackage, TaskDefinition
from agent_gem.llm import LLMClient
from agent_gem.sandbox.executor import CodeExecutor
from agent_gem.sandbox.manager import SandboxManager

from ..base import BaseAgent

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.config import CodeAgentConfig
    from agent_gem.generator import GenerationRequest

logger = logging.getLogger(__name__)


class CodeAgent(BaseAgent):
    """
    Code Agent for automatic training data generation from GitHub repositories.
    
    Workflow:
    1. Start SandboxFusion service
    2. Clone/analyze repository in sandbox
    3. Extract dependencies and configure environment in sandbox
    4. Generate synthetic bugs (via LLM on host)
    5. Create GitHub issues and PR descriptions (via LLM on host)
    6. Generate test cases (via LLM on host)
    7. Validate tests in sandbox
    8. Package as training data
    9. Save to host filesystem
    10. Stop SandboxFusion service
    """

    agent_type = "code_agent"
    description = "Synthesizes code issue + patch validation tasks from real GitHub repositories (executed in SandboxFusion)"

    def __init__(self, config: "CodeAgentConfig") -> None:
        """
        Initialize CodeAgent from configuration.
        
        Args:
            config: CodeAgentConfig instance with all settings
        """
        from agent_gem.config import CodeAgentConfig
        
        # Store configuration
        self.config = config
        
        # Extract configuration values
        self.taskdb_root = config.taskdb_root
        self.max_tokens = config.max_tokens
        self.auto_save = config.auto_save
        self.save_logs = config.save_logs
        
        # Wrap LLM's chat_completion to log responses
        llm = LLMClient.from_env()
        super().__init__(llm, self.taskdb_root)
        
        original_chat_completion = llm.chat_completion
        llm_call_counter = [0]  # Use list to allow mutation in nested function
        
        def wrapped_chat_completion(*args, **kwargs):
            llm_call_counter[0] += 1
            response = original_chat_completion(*args, **kwargs)
            
            # Log raw LLM response if log_dir is set
            if hasattr(self, 'log_dir') and self.log_dir and self.save_logs:
                log_file = self.log_dir / f"llm_response_{llm_call_counter[0]:02d}.txt"
                with open(log_file, "w") as f:
                    f.write(f"LLM Call #{llm_call_counter[0]}\n")
                    f.write(f"Arguments: {args}\n")
                    f.write(f"Keyword Arguments: {kwargs}\n")
                    f.write(f"\n{'='*80}\nResponse:\n{'='*80}\n")
                    f.write(response)
            
            return response
        
        llm.chat_completion = wrapped_chat_completion
        
        self.task_generator = FeatureRequestGenerator(llm, config=config)
        
        # Log output directory
        self.log_dir: Optional[Path] = None
        
        # SandboxFusion integration
        self.sandbox_manager = SandboxManager(
            root=Path(self.taskdb_root),
            sandbox_url=config.sandbox_url,
            use_docker_runner=config.use_docker_runner,
            silent=config.silent
        )
        self.code_executor: Optional[CodeExecutor] = None

    def generate(self, request: GenerationRequest, target_file_path: str, target_function_name: Optional[str] = None) -> Optional[TaskPackage]:
        """
        Generate a feature request task from a GitHub repository (executed in sandbox).
        
        Args:
            request: GenerationRequest with repo info in topic field
            target_file_path: Specific file path to delete function from.
            target_function_name: Optional specific function name to delete.
                                If None, LLM will select a suitable function automatically.
                                If provided, must exist in target_file_path.
            
        Returns:
            TaskPackage with function implementation task
        """
        # Extract repo URL from topic
        repo_url = request.topic
        
        logger.info(f"[CodeAgent] Task mode: feature_request")
        
        task_id = str(uuid.uuid4())
        
        # Initialize output directory for logging (under taskdb_root)
        if self.save_logs:
            task_dir = Path(self.taskdb_root) / f"{self.agent_type}/task-{task_id}"
            self.log_dir = task_dir / "logs"
            self.log_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[CodeAgent] Logging output to: {self.log_dir}")

        logger.info(
            f"[CodeAgent] Starting task generation using SandboxFusion: {repo_url}"
        )

        try:
            # Step 0: Start SandboxFusion
            logger.info("[CodeAgent] Starting SandboxFusion service...")
            with self.sandbox_manager.managed_executor() as executor:
                self.code_executor = CodeExecutor(executor)
                
                # Step 1: Clone and analyze repository in sandbox
                logger.info("[CodeAgent] Cloning repository in sandbox...")
                if not self._clone_repo_in_sandbox(repo_url):
                    logger.error("[CodeAgent] Failed to clone repository")
                    return None
                
                # Step 2: Analyze repository in sandbox
                logger.info("[CodeAgent] Analyzing repository in sandbox...")
                metadata = self._analyze_repo_in_sandbox(repo_url)
                if not metadata:
                    logger.error("[CodeAgent] Failed to analyze repository")
                    return None
                
                # Save repository metadata
                self._save_log("01_repo_metadata", asdict(metadata))

                # Step 3: Select file for task generation (specified or intelligent selection)
                logger.info(f"[CodeAgent] Using specified target file: {target_file_path}")
 
                # Read it directly from sandbox
                logger.info(f"[CodeAgent] Reading specified file from sandbox...")
                code = self.code_executor.read_file(f"/workspace/repo/{target_file_path}")
                
                if not code:
                    logger.error(f"[CodeAgent] Failed to read specified file: {target_file_path}")
                    return None
                logger.info(f"[CodeAgent] Using specified file: {target_file_path}")
                self._save_log("02_target_file_code", code)

                # Step 4: Generate feature request + issue + test
                logger.info(f"[CodeAgent] Generating feature request for specified function: {target_function_name}")
                result = self.task_generator.generate_feature_request_with_issue_and_tests(
                    metadata, code, target_file_path,
                    max_tokens=self.max_tokens,
                    code_executor=self.code_executor,
                    max_retries=5,
                    target_function_name=target_function_name
                )
                
                if not result:
                    logger.warning("[CodeAgent] Failed to generate feature request task")
                    return None
                
                deletion, issue, test_info = result
                
                # Save generated components
                self._save_log("03_feature_request", asdict(deletion))
                self._save_log("04_issue", asdict(issue))
                self._save_log("05_test_script", asdict(test_info))
            
                # Feature request mode: No PR generation needed
                logger.info("="*80)
                logger.info("[CodeAgent] Feature request task completed")
                logger.info("[CodeAgent] The deletion patch + issue + tests define the implementation task")
                logger.info("="*80)
                
                # Generate final task package
                logger.info("[CodeAgent] Generating final task package...")
                
                # Create TaskDefinition from the generated issue and deletion
                task_definition = TaskDefinition(
                    task_id=task_id,
                    task_title=issue.issue_title,
                    task_content=issue.issue_body,
                    difficulty_level=request.difficulty,
                )
                
                # Create solution text with only feature request information
                solution_text = f"""# Function Implementation Solution

## Task Description
Implement the deleted function to restore the repository functionality.

## Deleted Function
File: {deletion.affected_file}
Function: {deletion.function_name}
Signature: {deletion.function_signature}
Location: Lines {deletion.line_start}-{deletion.line_end}

## Function Docstring
{deletion.function_docstring}

## Implementation Hints
{deletion.implementation_hints}

## Original Function Code (Reference)
```python
{deletion.original_function_code}
```

## Deletion Patch
```diff
{deletion.deletion_patch_content}
```

## How to Apply Solution
1. Review the function signature and docstring above
2. Implement the function following the hints
3. The original code is provided as reference
4. Apply the deletion patch in reverse to restore the function
5. Run the verification tests to confirm correctness
"""
                
                # Create TaskPackage object
                task_package = TaskPackage(
                    id=task_id,
                    task=task_definition,
                    solution=solution_text,
                    verification=test_info.verification_command,
                    agent_type="code_agent",
                    metadata={
                        "repo_url": metadata.repo_url,
                        "repo_name": metadata.repo_name,
                        "language": metadata.language,
                        "test_framework": metadata.test_framework,
                        "dependencies": ",".join(metadata.dependencies) if metadata.dependencies else "",
                        "file_path": deletion.affected_file,
                        "function_name": deletion.function_name,
                        "task_mode": "feature_request",
                        # Test information (stored separately)
                        "test_file_path": test_info.test_file_path or "",
                        "test_code": test_info.test_code,
                        "test_patch_content": test_info.test_patch_content,
                        "test_description": test_info.test_description,
                        "test_fuzz": str(test_info.test_fuzz) if test_info.test_fuzz is not None else "",
                    },
                )
                
                # Create combined output for logging
                final_output = {
                    "task_id": task_id,
                    "task_mode": "feature_request",
                    "deletion": asdict(deletion),
                    "issue": asdict(issue),
                    "test": asdict(test_info),
                    "metadata": asdict(metadata),
                    "status": "completed"
                }
                
                self._save_log("06_final_output", final_output)
                
                logger.info(f"[CodeAgent] Task generation completed successfully")
                logger.info(f"[CodeAgent] Output directory: {task_dir}")
                
                # Auto-save task if enabled
                if self.auto_save:
                    logger.info(f"[CodeAgent] Auto-saving task...")
                    save_success = self.save_task_to_disk(task_package, task_dir)
                    
                    if save_success:
                        logger.info(f"[CodeAgent] Task auto-saved to: {task_dir}")
                        # Update package metadata with save location
                        task_package.metadata["task_saved"] = True
                        task_package.metadata["task_dir"] = str(task_dir)
                    else:
                        logger.warning(f"[CodeAgent] Task auto-save failed")
                        task_package.metadata["task_saved"] = False
                else:
                    logger.info(f"[CodeAgent] Auto-save disabled, task not saved")
                    task_package.metadata["task_saved"] = False
                
                return task_package

        except Exception as e:
            logger.error(f"[CodeAgent] Task generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def save_task_to_disk(self, package: TaskPackage, task_dir: Path) -> bool:
        """Save task package to disk with all necessary files.
        
        Args:
            package: TaskPackage to save
            task_dir: Directory to save task to
            
        Returns:
            bool: True if successful, False otherwise
        """
        import subprocess
        
        try:
            # Create task directory
            task_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[CodeAgent] Saving task to: {task_dir}")
            
            # Clone repo on host
            repo_url = package.metadata.get("repo_url")
            if repo_url:
                repo_dir = task_dir / "repo"
                logger.info(f"[CodeAgent] Cloning repository to host: {repo_url}")
                
                try:
                    # Git clone with shallow clone to save space
                    clone_cmd = [
                        "git", "clone", 
                        "--depth", "1",
                        "--single-branch",
                        repo_url, 
                        str(repo_dir)
                    ]
                    
                    result = subprocess.run(
                        clone_cmd,
                        capture_output=True,
                        text=True,
                        timeout=300
                    )
                    
                    if result.returncode == 0:
                        logger.info(f"[CodeAgent] Successfully cloned repo to: {repo_dir}")
                        
                        # Get repo statistics
                        import os
                        py_files = sum(1 for root, dirs, files in os.walk(repo_dir) 
                                     for f in files if f.endswith('.py'))
                        total_files = sum(1 for root, dirs, files in os.walk(repo_dir) 
                                        for f in files)
                        
                        # Save repo info
                        repo_info = {
                            "cloned": True,
                            "repo_path": str(repo_dir),
                            "total_files": total_files,
                            "python_files": py_files,
                            "clone_command": " ".join(clone_cmd)
                        }
                        
                        repo_info_file = task_dir / "repo_info.json"
                        with open(repo_info_file, 'w', encoding='utf-8') as f:
                            json.dump(repo_info, f, indent=2, ensure_ascii=False)
                        
                        logger.info(f"[CodeAgent] Repo stats: {total_files} files, {py_files} Python files")
                        package.metadata["repo_cloned"] = True
                        package.metadata["repo_path"] = str(repo_dir)
                        package.metadata["repo_total_files"] = total_files
                        package.metadata["repo_python_files"] = py_files
                    else:
                        logger.error(f"[CodeAgent] Failed to clone repo: {result.stderr}")
                        package.metadata["repo_cloned"] = False
                        package.metadata["clone_error"] = result.stderr
                        
                except subprocess.TimeoutExpired:
                    logger.error("[CodeAgent] Git clone timeout")
                    package.metadata["repo_cloned"] = False
                    package.metadata["clone_error"] = "Timeout"
                except Exception as e:
                    logger.error(f"[CodeAgent] Failed to clone repo: {e}")
                    package.metadata["repo_cloned"] = False
                    package.metadata["clone_error"] = str(e)
            
            # Use TaskWriter to persist in standard format
            from agent_gem.writer import TaskWriter
            writer = TaskWriter(root=Path(self.taskdb_root))
            writer.persist([package])
            logger.info(f"[CodeAgent] Task persisted using TaskWriter")
            
            return True
            
        except Exception as e:
            logger.error(f"[CodeAgent] Failed to save task: {e}", exc_info=True)
            return False

    def _save_log(self, name: str, content: Any) -> None:
        """Save log content to output directory."""
        if not self.save_logs or not self.log_dir:
            return
        
        if isinstance(content, (dict, list)):
            filepath = self.log_dir / f"{name}.json"
            with open(filepath, "w") as f:
                # Convert sets to lists for JSON serialization
                json.dump(content, f, indent=2, ensure_ascii=False, default=lambda x: list(x) if isinstance(x, set) else str(x))
        else:
            filepath = self.log_dir / f"{name}.txt"
            with open(filepath, "w") as f:
                f.write(str(content))
        
        logger.info(f"[CodeAgent] Saved log: {filepath}")

    def _extract_source_files(self, metadata: RepoMetadata) -> Dict[str, str]:
        """Extract main source files from the repository metadata."""
        source_codes: Dict[str, str] = {}
        
        # In a real scenario, this would read from the cloned repo
        # For now, we create a placeholder that will be filled during sandbox execution
        logger.info(f"Found {len(metadata.main_files)} main files")
        
        # Return a dictionary that will be populated when files are available
        return {"main.py": "# Main source code placeholder"}


    def _clone_repo_in_sandbox(self, repo_url: str, max_retries: int = 3) -> bool:
        """Clone repository in sandbox with retry mechanism."""
        if not self.code_executor:
            return False
        dest_path = "/workspace/repo"
        cmd = f"rm -rf {dest_path} && git clone --depth 1 {repo_url} {dest_path}"
        cloned = False
        
        for _ in range(max_retries):
            clone_result = self.code_executor.run_command(cmd, timeout_s=300)
            if self.code_executor._is_success(clone_result):
                cloned = True
                break
        
        if cloned and self.save_logs and self.log_dir:
            # Save repo directory listing
            try:
                dir_listing = self.code_executor.run_command(
                    "find /workspace/repo -type f -name '*.py' -o -name '*.json' -o -name '*.txt' -o -name '*.md' | head -100",
                )
                
                if self.code_executor._is_success(dir_listing):
                    self._save_log("00_repo_directory_listing", self.code_executor._extract_stdout(dir_listing))
            except Exception as e:
                logger.warning(f"Failed to save repo directory listing: {e}")
        
        return cloned

    def _analyze_repo_in_sandbox(self, repo_url: str) -> Optional[RepoMetadata]:
        """Analyze repository in sandbox."""
        if not self.code_executor:
            return None
        
        try:
            # Detect language
            language = self.code_executor.detect_language("/workspace/repo")
            
            # Extract dependencies
            dependencies = self.code_executor.extract_dependencies(
                "/workspace/repo", language
            )
            
            # For now, use placeholder metadata - in a real scenario,
            # this would be extracted from the repo in the sandbox
            
            return RepoMetadata(
                repo_url=repo_url,
                repo_name=repo_url.split("/")[-1],
                language=language,
                dependencies=dependencies,
                main_files=["main.py"],  # Placeholder
                test_framework="pytest" if language == "python" else "jest",
                build_system="setuptools" if language == "python" else "npm",
                entry_points=["main.py"],  # Placeholder
            )
        except Exception as e:
            logger.error(f"Failed to analyze repo in sandbox: {e}")
            return None
