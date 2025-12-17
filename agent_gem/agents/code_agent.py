from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from agent_gem.core.task_schema import TaskPackage, ToolSpec, EvaluationCriteria, TaskDefinition
from agent_gem.core.utils import dump_json
from agent_gem.sandbox import SandboxExecutor
from agent_gem.sandbox.executor import SandboxFusionExecutor
from agent_gem.tools import BashTool, PythonRunnerTool

from .base import BaseAgent, TaskContext
from .code_repo_analyzer import RepositoryAnalyzer, DependencyConfigurer, RepoMetadata
from .code_bug_generator import BugGenerator, IssuePRGenerator, BugInfo, IssuePRInfo
from .code_test_validator import TestCaseGenerator, TestValidator, TestValidationResult
from .sandbox_integration import SandboxManager, CodeExecutor

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest

logger = logging.getLogger(__name__)


# Track LLM call counter for logging
_llm_call_count = {}


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

    def __init__(self, llm, taskdb_root: str = "taskdb", sandbox_url: Optional[str] = None, save_logs: bool = True, 
                 max_tokens: int = 2000, temperature: float = 0.7) -> None:
        super().__init__(llm, taskdb_root)
        
        # Store LLM generation parameters
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Wrap LLM's chat_completion to log responses
        original_chat_completion = llm.chat_completion
        llm_call_counter = [0]  # Use list to allow mutation in nested function
        
        def wrapped_chat_completion(*args, **kwargs):
            llm_call_counter[0] += 1
            response = original_chat_completion(*args, **kwargs)
            
            # Log raw LLM response if output_dir is set
            if hasattr(self, 'output_dir') and self.output_dir and self.save_logs:
                log_file = self.output_dir / f"llm_response_{llm_call_counter[0]:02d}.txt"
                with open(log_file, "w") as f:
                    f.write(f"LLM Call #{llm_call_counter[0]}\n")
                    f.write(f"Arguments: {args}\n")
                    f.write(f"Keyword Arguments: {kwargs}\n")
                    f.write(f"\n{'='*80}\nResponse:\n{'='*80}\n")
                    f.write(response)
            
            return response
        
        llm.chat_completion = wrapped_chat_completion
        
        self.repo_analyzer = RepositoryAnalyzer(llm)
        self.bug_generator = BugGenerator(llm)
        self.issue_pr_generator = IssuePRGenerator(llm)
        self.test_case_generator = TestCaseGenerator(llm)
        self.test_validator = TestValidator(timeout_s=30)
        self.dep_configurer = DependencyConfigurer(llm)
        
        # Log output directory
        self.save_logs = save_logs
        self.output_dir: Optional[Path] = None
        
        # SandboxFusion integration
        self.sandbox_manager = SandboxManager(
            sandbox_url=sandbox_url,
            use_docker_runner=True,
            silent=False
        )
        self.sandbox_executor: Optional[SandboxFusionExecutor] = None
        self.code_executor: Optional[CodeExecutor] = None

    def generate(self, request: GenerationRequest) -> Optional[TaskPackage]:
        """
        Generate a training task from a GitHub repository (executed in sandbox).
        
        Args:
            request: GenerationRequest with repo info in topic field
            
        Returns:
            TaskPackage with code repair task
        """
        # Extract repo URL from topic
        repo_url = request.topic or "https://github.com/pallets/flask"
        
        task_id = str(uuid.uuid4())
        ctx = TaskContext(task_id=task_id, request=request)
        
        # Initialize output directory for logging
        if self.save_logs:
            self.output_dir = Path("output") / task_id
            self.output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"[CodeAgent] Logging output to: {self.output_dir}")

        logger.info(
            f"[CodeAgent] Starting task generation in SandboxFusion: {repo_url}"
        )

        try:
            # Step 0: Start SandboxFusion
            logger.info("[CodeAgent] Starting SandboxFusion service...")
            with self.sandbox_manager.managed_executor() as executor:
                self.sandbox_executor = executor
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
                
                ctx.add_step({
                    "step": "repo_analysis",
                    "metadata": asdict(metadata)
                })

                # Step 3: Extract source files in sandbox
                logger.info("[CodeAgent] Extracting source files...")
                source_codes = self._extract_source_files_in_sandbox(metadata)
                if not source_codes:
                    logger.warning("[CodeAgent] No source files found")
                    return None
                
                # Save extracted source files
                self._save_log("02_extracted_source_files", source_codes)

                # Step 4: Generate bug (on host via LLM)
                logger.info("[CodeAgent] Generating synthetic bug...")
                import random
                file_path, code = random.choice(list(source_codes.items()))
                bug = self.bug_generator.generate_bug(
                    metadata, code, file_path,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                if not bug:
                    logger.warning("[CodeAgent] Failed to generate bug")
                    return None
                
                # Save bug generation
                self._save_log("03_generated_bug", asdict(bug))

                ctx.add_step({
                    "step": "bug_generation",
                    "bug": asdict(bug)
                })
                
                # Step 4.5: Apply bug to repository in sandbox
                logger.info("[CodeAgent] Applying bug to repository in sandbox...")
                buggy_applied = self._apply_bug_to_repo(bug, metadata)
                if not buggy_applied:
                    logger.warning("[CodeAgent] Failed to apply bug to repo, continuing anyway")
                else:
                    logger.info("[CodeAgent] Bug successfully applied to repo")

                # Step 5: Generate issue and PR (on host via LLM)
                logger.info("[CodeAgent] Generating issue and PR...")
                issue_pr = self.issue_pr_generator.generate_issue_and_pr(
                    bug, metadata,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature * 0.9  # Slightly lower temperature for structured output
                )
                
                # Save issue/PR response (whether successful or not)
                if issue_pr:
                    self._save_log("04_issue_pr", asdict(issue_pr))
                
                if not issue_pr:
                    logger.warning("[CodeAgent] Failed to generate issue/PR, using fallback")
                    # Create a basic fallback Issue/PR
                    from agent_gem.agents.code_bug_generator import IssuePRInfo
                    issue_pr = IssuePRInfo(
                        issue_title=f"Fix: {bug.bug_title}",
                        issue_body=f"## Bug Description\n\n{bug.bug_description}\n\n## Affected Code\n\n```python\n{bug.buggy_code}\n```",
                        issue_labels=["bug", bug.bug_type],
                        pr_title=f"Fix: {bug.bug_title}",
                        pr_description=f"This PR fixes a {bug.bug_type} bug in {bug.affected_file}.",
                        pr_changes_summary="Applied fix to resolve the issue",
                        fixed_code=bug.buggy_code,  # Use buggy code as placeholder
                        test_additions="# TODO: Add test cases",
                    )

                ctx.add_step({
                    "step": "issue_pr_generation",
                    "issue_pr": asdict(issue_pr)
                })
                
                # Step 5.5: Apply fix to repository in sandbox
                logger.info("[CodeAgent] Applying fix to repository in sandbox...")
                fixed_applied = self._apply_fix_to_repo(issue_pr, bug, metadata)
                if not fixed_applied:
                    logger.warning("[CodeAgent] Failed to apply fix to repo, continuing anyway")
                else:
                    logger.info("[CodeAgent] Fix successfully applied to repo")
                    # Get real git diff
                    git_diff = self._get_repo_diff()
                    if git_diff:
                        logger.info(f"[CodeAgent] Generated git diff ({len(git_diff)} chars)")
                        ctx.add_step({
                            "step": "git_diff_generated",
                            "diff_size": len(git_diff)
                        })

                # Step 6: Generate test cases (on host via LLM)
                logger.info("[CodeAgent] Generating test cases...")
                tests = self.test_case_generator.generate_multiple_test_cases(
                    bug, metadata, bug.buggy_code, issue_pr.fixed_code, num_tests=2,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature
                )
                
                if not tests:
                    logger.warning("[CodeAgent] Failed to generate test cases")
                    return None
                
                # Save generated test cases
                self._save_log("05_test_cases", [asdict(t) for t in tests])

                ctx.add_step({
                    "step": "test_generation",
                    "test_count": len(tests),
                    "tests": [{"test_id": t.test_id, "test_name": t.test_name} for t in tests]
                })

                # Step 7: Validate tests in sandbox
                logger.info("[CodeAgent] Validating tests in sandbox...")
                validation_results = self._validate_tests_in_sandbox(
                    tests, bug.buggy_code, issue_pr.fixed_code, metadata
                )
                
                # Save validation results
                self._save_log("06_test_validation_results", [asdict(r) for r in validation_results])
                
                valid_tests = [r for r in validation_results if r.is_valid_test]
                if not valid_tests:
                    logger.warning("[CodeAgent] No valid tests generated, using all tests anyway")
                    # Use all tests even if validation failed
                    valid_tests = validation_results

                ctx.add_step({
                    "step": "test_validation",
                    "valid_tests": len(valid_tests),
                    "total_tests": len(tests),
                    "results": [
                        {
                            "test_id": r.test_id,
                            "valid": r.is_valid_test,
                            "fails_on_buggy": r.fails_on_buggy,
                            "passes_on_fixed": r.passes_on_fixed
                        }
                        for r in validation_results
                    ]
                })

                # Step 8: Generate environment setup commands
                logger.info("[CodeAgent] Generating environment setup commands...")
                setup_info = self.dep_configurer.generate_setup_commands(metadata)
                ctx.add_step({
                    "step": "environment_setup",
                    "setup_info": setup_info
                })
                
                # Step 8.5: Get final git diff for the task
                logger.info("[CodeAgent] Extracting final git diff...")
                git_diff = self._get_repo_diff()
                if git_diff:
                    self._save_log("08_git_diff", git_diff)
                    logger.info(f"[CodeAgent] Git diff saved ({len(git_diff)} characters)")

                # Step 9: Create task package
                logger.info("[CodeAgent] Creating task package...")
                task_package = self._create_task_package(
                    task_id, request, metadata, bug, issue_pr, valid_tests, 
                    validation_results, setup_info, ctx, git_diff
                )
                
                # Save final task package
                if task_package:
                    task_dict = {
                        "task_id": task_package.id,
                        "task": task_package.task.model_dump() if task_package.task else None,
                        "task_package_version": "1.0"
                    }
                    self._save_log("07_final_task_package", task_dict)

                # Step 10: Persist training data to host
                logger.info("[CodeAgent] Persisting training data to host...")
                self._persist_training_data(
                    task_id, metadata, bug, issue_pr, tests, validation_results, setup_info, git_diff
                )

                # Persist context
                self.writer.record_steps(task_id, self.agent_type, ctx.history)

                logger.info(f"[CodeAgent] Task generation completed: {task_id}")
                return task_package

        except Exception as e:
            logger.error(f"[CodeAgent] Task generation failed: {e}", exc_info=True)
            ctx.add_step({
                "step": "error",
                "error": str(e)
            })
            return None
        finally:
            self.sandbox_executor = None
            self.code_executor = None
            logger.info("[CodeAgent] SandboxFusion service stopped")

    def _save_log(self, name: str, content: Any) -> None:
        """Save log content to output directory."""
        if not self.save_logs or not self.output_dir:
            return
        
        if isinstance(content, (dict, list)):
            filepath = self.output_dir / f"{name}.json"
            with open(filepath, "w") as f:
                json.dump(content, f, indent=2, ensure_ascii=False)
        else:
            filepath = self.output_dir / f"{name}.txt"
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

    def _create_task_package(
        self,
        task_id: str,
        request: GenerationRequest,
        metadata: RepoMetadata,
        bug: BugInfo,
        issue_pr: IssuePRInfo,
        valid_tests: List[Any],
        validation_results: List[TestValidationResult],
        setup_info: Dict[str, Any],
        ctx: TaskContext,
        git_diff: str = "",
    ) -> TaskPackage:
        """Create a TaskPackage from all generated components."""
        
        # Build task content
        task_content = f"""
# Code Repository Bug Fix Task

## Repository
- **Name**: {metadata.repo_name}
- **Language**: {metadata.language}
- **Test Framework**: {metadata.test_framework}

## Issue Report
### {issue_pr.issue_title}

{issue_pr.issue_body}

**Labels**: {', '.join(issue_pr.issue_labels)}

## Expected Fix (Pull Request)
### {issue_pr.pr_title}

{issue_pr.pr_description}

## Environment Setup
### Dependencies
- {chr(10).join(['- ' + dep for dep in metadata.dependencies[:10]])}

### Setup Commands
- {chr(10).join(['- ' + cmd for cmd in setup_info.get('install_deps', [])])}

## What You Need To Do
1. Understand the bug described in the issue
2. Apply the provided fix to the repository
3. Run the test suite to verify the fix works
4. Ensure all tests pass

## Bug Information
- **Type**: {bug.bug_type}
- **Severity**: {bug.severity}
- **Affected File**: {bug.affected_file}
- **Affected Function**: {bug.affected_function}
"""

        # Build solution with the fixed code and test additions
        solution = f"""
# Solution

## Git Diff (Buggy → Fixed)
```diff
{git_diff if git_diff else "# Git diff not available"}
```

## Fixed Code
```{metadata.language}
{issue_pr.fixed_code}
```

## Test Code
```{metadata.language}
{issue_pr.test_additions}
```

## Verification Steps
1. Checkout buggy branch: git checkout buggy
2. Verify tests fail on buggy version
3. Checkout fixed branch: git checkout fixed
4. Verify tests pass on fixed version
5. Review the git diff for code changes
"""

        # Build verification function
        verification = f"""
def verify(tools, answer):
    '''
    Verify that the fix is correct by:
    1. Checking that test cases pass
    2. Verifying the solution matches expected output
    '''
    # Basic verification - in sandbox this would run actual tests
    return isinstance(answer, dict) and answer.get('tests_passed', False)
"""

        # Evaluation criteria
        evaluation_criteria = EvaluationCriteria(
            success_definition="Tests pass on the fixed code",
            key_metrics=["test_pass_rate", "code_quality"],
            failure_modes=["Tests fail", "Syntax errors"],
        )

        # Create tool specs
        tools = self._default_tools()

        # Create task definition
        task = TaskDefinition(
            task_title=f"Fix: {issue_pr.issue_title}",
            task_content=task_content,
            submit_result_format={"type": "pytest output"},
            tool_set=tools,
            evaluation_criteria=evaluation_criteria,
            difficulty_level=request.difficulty,
        )

        # Metadata for the task
        metadata_dict = {
            "repo_url": str(metadata.repo_url),
            "repo_name": str(metadata.repo_name),
            "language": str(metadata.language),
            "test_framework": str(metadata.test_framework),
            "bug_id": str(bug.bug_id),
            "valid_tests_count": str(len(valid_tests)),
            "setup_commands": json.dumps(setup_info) if isinstance(setup_info, dict) else str(setup_info),
            "source": "github_repository",
            "has_git_diff": str(bool(git_diff)),
            "git_diff_size": str(len(git_diff)) if git_diff else "0",
        }

        return TaskPackage(
            task=task,
            solution=solution,
            verification=verification,
            agent_type=self.agent_type,
            metadata=metadata_dict,
        )

    def _default_tools(self) -> List[ToolSpec]:
        """Define tools available for code tasks."""
        def git(command: str) -> str:
            """Inspect repository state and apply patches."""
            raise RuntimeError("tool spec only")

        def tests(target: str = "all") -> dict[str, object]:
            """Run the project's test suite."""
            raise RuntimeError("tool spec only")

        def bash(command: str) -> dict[str, object]:
            """Execute shell commands in the sandbox."""
            raise RuntimeError("tool spec only")

        return [
            ToolSpec.from_function(git, name="git"),
            ToolSpec.from_function(tests, name="tests"),
            ToolSpec.from_function(bash, name="bash"),
        ]

    def _build_prompt(self, request: GenerationRequest) -> str:
        """Legacy method - not used in this implementation."""
        return (
            "You are the Code Agent creating repository repair tasks. "
            f"Generate exactly 1 task for topic '{request.topic or 'a codebase you choose'}' that mimics issue/PR workflows."
        )

    def _clone_repo_in_sandbox(self, repo_url: str) -> bool:
        """Clone repository in sandbox."""
        if not self.code_executor:
            return False
        
        result = self.code_executor.clone_repo(repo_url, "/workspace/repo")
        
        if result and self.save_logs and self.output_dir:
            # Save repo directory listing
            try:
                dir_listing = self.code_executor.run_code(
                    "find /workspace/repo -type f -name '*.py' -o -name '*.json' -o -name '*.txt' -o -name '*.md' | head -100",
                    language="bash"
                )
                if dir_listing and dir_listing.get("stdout"):
                    self._save_log("00_repo_directory_listing", dir_listing["stdout"])
            except Exception as e:
                logger.warning(f"Failed to save repo directory listing: {e}")
        
        return result

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

    def _extract_source_files_in_sandbox(self, metadata: RepoMetadata) -> Dict[str, str]:
        """Extract source files in sandbox."""
        if not self.code_executor:
            return {}
        
        try:
            # List files from sandbox
            files = self.code_executor.list_files("/workspace/repo")
            
            # Read source files
            source_codes: Dict[str, str] = {}
            for file_path in files[:5]:  # Limit to 5 files
                if file_path.endswith((".py", ".js", ".ts", ".java")):
                    content = self.code_executor.read_file(f"/workspace/repo/{file_path}")
                    if content and len(content) < 10000:  # Limit file size
                        source_codes[file_path] = content
            
            if not source_codes:
                # Fallback: create placeholder
                source_codes["main.py"] = "# Main source code placeholder"
            
            return source_codes
        except Exception as e:
            logger.error(f"Failed to extract source files: {e}")
            return {"main.py": "# Main source code placeholder"}

    def _validate_tests_in_sandbox(
        self,
        tests: List[Any],
        buggy_code: str,
        fixed_code: str,
        metadata: RepoMetadata,
    ) -> List[TestValidationResult]:
        """Validate tests in sandbox using real repository versions."""
        if not self.code_executor:
            # Fallback to local validation if no sandbox
            logger.warning("[CodeAgent] No sandbox executor, using local validation")
            return self.test_validator.validate_multiple(
                tests, buggy_code, fixed_code, metadata
            )
        
        # Validate tests in the actual repository with buggy and fixed versions
        validation_results = []
        
        for test in tests:
            try:
                # Test on buggy version
                logger.info(f"[CodeAgent] Testing {test.test_name} on buggy version...")
                self.code_executor.checkout_git_branch("buggy")
                
                # Write test file to repo
                test_file_path = f"/workspace/repo/test_{test.test_id}.py"
                self.code_executor.apply_code_to_file(
                    f"test_{test.test_id}.py", 
                    test.test_code
                )
                
                # Run test
                result_buggy = self.code_executor.run_tests_in_repo(
                    f"python -m pytest test_{test.test_id}.py -v"
                )
                buggy_passed = result_buggy.get("exit_code", 1) == 0
                
                # Test on fixed version
                logger.info(f"[CodeAgent] Testing {test.test_name} on fixed version...")
                self.code_executor.checkout_git_branch("fixed")
                
                # Write test file to repo (same test)
                self.code_executor.apply_code_to_file(
                    f"test_{test.test_id}.py", 
                    test.test_code
                )
                
                # Run test
                result_fixed = self.code_executor.run_tests_in_repo(
                    f"python -m pytest test_{test.test_id}.py -v"
                )
                fixed_passed = result_fixed.get("exit_code", 1) == 0
                
                # Determine if test is valid
                is_valid = (not buggy_passed) and fixed_passed
                
                validation_results.append(TestValidationResult(
                    test_id=test.test_id,
                    test_name=test.test_name,
                    fails_on_buggy=not buggy_passed,
                    passes_on_fixed=fixed_passed,
                    error_message_buggy=result_buggy.get("stderr") if not buggy_passed else None,
                    error_message_fixed=result_fixed.get("stderr") if not fixed_passed else None,
                    is_valid_test=is_valid,
                ))
                
                logger.info(
                    f"[CodeAgent] Test {test.test_name}: "
                    f"buggy={'FAIL' if not buggy_passed else 'PASS'}, "
                    f"fixed={'PASS' if fixed_passed else 'FAIL'}, "
                    f"valid={is_valid}"
                )
                
            except Exception as e:
                logger.error(f"[CodeAgent] Error validating test {test.test_name}: {e}")
                validation_results.append(TestValidationResult(
                    test_id=test.test_id,
                    test_name=test.test_name,
                    fails_on_buggy=False,
                    passes_on_fixed=False,
                    error_message_buggy=str(e),
                    error_message_fixed=str(e),
                    is_valid_test=False,
                ))
        
        return validation_results
    
    def _apply_bug_to_repo(self, bug: BugInfo, metadata: RepoMetadata) -> bool:
        """
        Apply the generated bug to the repository in sandbox.
        
        Creates a 'buggy' branch with the buggy code applied.
        
        Args:
            bug: BugInfo containing the buggy code
            metadata: Repository metadata
            
        Returns:
            True if successful
        """
        if not self.code_executor:
            logger.warning("[CodeAgent] No sandbox executor available")
            return False
        
        try:
            # Create buggy branch
            logger.info("[CodeAgent] Creating buggy branch...")
            self.code_executor.create_git_branch("buggy")
            
            # Apply buggy code to the affected file
            logger.info(f"[CodeAgent] Applying buggy code to {bug.affected_file}...")
            success = self.code_executor.apply_code_to_file(
                bug.affected_file,
                bug.buggy_code
            )
            
            if not success:
                logger.error("[CodeAgent] Failed to write buggy code to file")
                return False
            
            # Commit the changes
            logger.info("[CodeAgent] Committing buggy version...")
            commit_success = self.code_executor.git_commit(
                message=f"Bug: {bug.bug_title}",
                file_paths=[bug.affected_file]
            )
            
            if commit_success:
                logger.info("[CodeAgent] Buggy version committed successfully")
            else:
                logger.warning("[CodeAgent] Git commit failed, but file was modified")
            
            return True
            
        except Exception as e:
            logger.error(f"[CodeAgent] Error applying bug to repo: {e}")
            return False
    
    def _apply_fix_to_repo(self, issue_pr: IssuePRInfo, bug: BugInfo, metadata: RepoMetadata) -> bool:
        """
        Apply the fix to the repository in sandbox.
        
        Creates a 'fixed' branch from 'buggy' branch with the fix applied.
        
        Args:
            issue_pr: IssuePRInfo containing the fixed code
            bug: BugInfo for context
            metadata: Repository metadata
            
        Returns:
            True if successful
        """
        if not self.code_executor:
            logger.warning("[CodeAgent] No sandbox executor available")
            return False
        
        try:
            # Checkout buggy branch first to base fix on it
            logger.info("[CodeAgent] Checking out buggy branch...")
            self.code_executor.checkout_git_branch("buggy")
            
            # Create fixed branch from buggy
            logger.info("[CodeAgent] Creating fixed branch...")
            self.code_executor.create_git_branch("fixed")
            
            # Apply fixed code
            logger.info(f"[CodeAgent] Applying fixed code to {bug.affected_file}...")
            success = self.code_executor.apply_code_to_file(
                bug.affected_file,
                issue_pr.fixed_code
            )
            
            if not success:
                logger.error("[CodeAgent] Failed to write fixed code to file")
                return False
            
            # Commit the fix
            logger.info("[CodeAgent] Committing fixed version...")
            commit_success = self.code_executor.git_commit(
                message=f"Fix: {issue_pr.pr_title}",
                file_paths=[bug.affected_file]
            )
            
            if commit_success:
                logger.info("[CodeAgent] Fixed version committed successfully")
            else:
                logger.warning("[CodeAgent] Git commit failed, but file was modified")
            
            return True
            
        except Exception as e:
            logger.error(f"[CodeAgent] Error applying fix to repo: {e}")
            return False
    
    def _get_repo_diff(self) -> str:
        """
        Get the git diff between buggy and fixed branches.
        
        Returns:
            Git diff as string
        """
        if not self.code_executor:
            return ""
        
        try:
            diff = self.code_executor.get_git_diff("buggy", "fixed")
            return diff
        except Exception as e:
            logger.error(f"[CodeAgent] Error getting git diff: {e}")
            return ""

    def _persist_training_data(
        self,
        task_id: str,
        metadata: RepoMetadata,
        bug: BugInfo,
        issue_pr: IssuePRInfo,
        tests: List[Any],
        validation_results: List[TestValidationResult],
        setup_info: Dict[str, Any],
        git_diff: str = "",
    ) -> None:
        """Persist training data in structured format."""
        # Convert data structures to dictionaries
        repo_metadata = asdict(metadata)
        bug_info = asdict(bug)
        issue_pr_info = asdict(issue_pr)
        
        test_cases = [
            {
                "test_id": t.test_id,
                "test_name": t.test_name,
                "test_code": t.test_code,
                "test_framework": t.test_framework,
                "description": t.description,
                "assertion_description": t.assertion_description,
            }
            for t in tests
        ]
        
        validation_result_dicts = [
            {
                "test_id": r.test_id,
                "test_name": r.test_name,
                "fails_on_buggy": r.fails_on_buggy,
                "passes_on_fixed": r.passes_on_fixed,
                "error_message_buggy": r.error_message_buggy,
                "error_message_fixed": r.error_message_fixed,
                "is_valid_test": r.is_valid_test,
            }
            for r in validation_results
        ]
        
        # Use writer to persist
        self.writer.persist_code_training_data(
            task_id=task_id,
            repo_metadata=repo_metadata,
            bug_info=bug_info,
            issue_pr_info=issue_pr_info,
            test_cases=test_cases,
            validation_results=validation_result_dicts,
              setup_info=setup_info,
              git_diff=git_diff,
        )
