
import re
import json
import logging
from dataclasses import dataclass
from typing import List, Optional

from agent_gem.llm import LLMClient

logger = logging.getLogger(__name__)

@dataclass
class RepoMetadata:
    """Metadata about a repository."""
    repo_url: str
    repo_name: str
    language: str
    dependencies: List[str]
    main_files: List[str]
    test_framework: str
    build_system: str
    entry_points: List[str]

@dataclass
class BugInfo:
    """Information about a synthetic bug."""

    bug_id: str
    bug_title: str
    bug_description: str
    bug_type: str  # "logic", "performance", "security", "memory", "type"
    severity: str  # "low", "medium", "high", "critical"
    affected_file: str
    affected_function: str
    bug_location: str  # line range (e.g., "line 10-15")
    original_code_snippet: str  # Code before bug injection
    buggy_code_snippet: str  # Only the buggy part
    patch_content: str  # Unified diff patch format
    line_start: int  # Starting line number where bug is injected
    line_end: int  # Ending line number where bug is injected


@dataclass
class IssueInfo:
    """GitHub issue information for a bug."""

    issue_title: str
    issue_body: str
    issue_labels: List[str]
    root_cause: str  # Explanation of the root cause
    reproduction_steps: str  # How to reproduce the bug


@dataclass
class TestInfo:
    """Test script information for a bug."""

    test_code: str  # Test code that should catch the bug
    test_patch_content: str  # Unified diff patch format for adding tests
    test_description: str  # Description of what the tests check
    test_file_path: Optional[str] = None  # Path to test file (extracted from patch)
    verification_command: Optional[str] = None  # Command to run the tests
    test_fuzz: Optional[int] = None  # Fuzz value used when applying test patch


@dataclass
class FunctionDeletionInfo:
    """Information about a deleted function for implementation task."""

    deletion_id: str
    function_name: str
    function_signature: str
    function_docstring: str  # What the function should do
    affected_file: str
    deletion_location: str  # line range (e.g., "line 10-25")
    original_function_code: str  # Complete function code before deletion
    deletion_patch_content: str  # Unified diff patch format for deletion
    line_start: int  # Starting line number of function
    line_end: int  # Ending line number of function
    implementation_hints: str  # Hints for implementing the function


def extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    response = response.strip()
    
    # Remove markdown code blocks
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    
    # Try to find JSON object boundaries
    if not response.startswith("{"):
        # Look for the first { character
        match = re.search(r'\{', response)
        if match:
            response = response[match.start():]
    
    if not response.endswith("}"):
        # Look for the last } character
        match = re.search(r'\}[^}]*$', response)
        if match:
            response = response[:match.end()]
    
    return response


class FeatureRequestGenerator:
    """Generate synthetic bugs based on repository analysis."""

    def __init__(self, llm: LLMClient, config=None):
        self.llm = llm
        # Store test generation config if provided
        if config:
            self.target_test_count = getattr(config, 'target_test_count', 4)
            self.min_test_count = getattr(config, 'min_test_count', 3)
            self.max_test_count = getattr(config, 'max_test_count', 6)
        else:
            # Default values
            self.target_test_count = 4
            self.min_test_count = 3
            self.max_test_count = 6


    def _validate_patch(self, patch_content: str, code_executor, fuzz: int = 3) -> tuple[bool, str]:
        """
        Validate that a patch can be applied using patch --dry-run.
        
        Args:
            patch_content: Unified diff patch content
            code_executor: CodeExecutor instance with access to repository
            fuzz: Fuzz factor for patch matching (default: 3, higher = more tolerant)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        import uuid
        
        try:
            # Write patch to temporary file
            patch_uuid = uuid.uuid4()
            patch_path = f"/workspace/tmp/validate_{patch_uuid}.patch"
            
            # Ensure patch ends with newline
            if not patch_content.endswith('\n'):
                patch_content += '\n'
            
            # Log a brief preview to aid debugging
            logger.debug(f"[ValidatePatch] Preparing patch (length={len(patch_content)})")
            logger.debug(f"[ValidatePatch] Patch preview (first 500 chars):\n{patch_content[:500]}")
            
            # Also log the middle section to see the actual deletion
            if len(patch_content) > 1000:
                logger.debug(f"[ValidatePatch] Patch middle (chars 500-1000):\n{patch_content[500:1000]}")

            # Write patch file using bash heredoc
            write_cmd = f"""mkdir -p /workspace/tmp && cat > {patch_path} << 'EOF'
{patch_content}EOF"""
            
            write_result = code_executor.run_command(write_cmd)
            if not code_executor._is_success(write_result):
                return False, f"Failed to write patch file."
            
            # Validate patch with --dry-run and configurable fuzz
            validate_cmd = f"cd /workspace/repo && patch -p1 --dry-run --fuzz={fuzz} --verbose < {patch_path}"
            validate_result = code_executor.run_command(validate_cmd)
            
            # Clean up temporary patch file
            code_executor.run_command(f"rm -f {patch_path}")
            
            stdout = code_executor._extract_stdout(validate_result)

            if code_executor._is_success(validate_result):
                logger.debug(f"[ValidatePatch] Validation passed (fuzz={fuzz})")
                if stdout:
                    logger.debug(f"[ValidatePatch] stdout (first 200): {stdout[:200]}")
                return True, ""
            else:
                error_msg = stdout or f"No output from patch --dry-run (fuzz={fuzz})."
                logger.debug(f"[ValidatePatch] Failed (fuzz={fuzz}): {error_msg}")
                return False, f"{error_msg} Command: patch -p1 --dry-run --fuzz={fuzz} --verbose < {patch_path}"
                
        except Exception as e:
            logger.error(f"Exception during patch validation: {e}")
            return False, str(e)

    
    def generate_feature_request_with_issue_and_tests(
        self,
        metadata: RepoMetadata,
        source_code: str,
        file_path: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        code_executor=None,
        max_retries: int = 3,
        target_function_name: Optional[str] = None,
    ) -> Optional[tuple[FunctionDeletionInfo, IssueInfo, TestInfo]]:
        """
        Generate feature request task: delete a function and ask to implement it.
        
        Stage 1: Delete a function and generate issue
        Stage 2: Generate test code for the missing function
        
        Args:
            metadata: Repository metadata
            source_code: Source code content
            file_path: Path to the file in the repository
            max_tokens: Max tokens for LLM
            temperature: LLM temperature
            code_executor: Optional CodeExecutor for validation
            max_retries: Maximum number of retry attempts per stage
            
        Returns:
            Tuple of (FunctionDeletionInfo, IssueInfo, TestInfo) or None if generation fails
        """
        # ========== STAGE 1: Generate and validate feature request + issue ==========
        logger.info("[FeatureRequest - Stage 1] Selecting and deleting function...")
        
        deletion, issue = None, None
        for attempt in range(max_retries):
            try:
                logger.info(f"[Stage 1 - Attempt {attempt+1}/{max_retries}] Generating feature request + issue...")
                
                # Generate feature request + issue
                deletion, issue = self._generate_feature_request_and_issue(
                    metadata, source_code, file_path, max_tokens, temperature, code_executor,
                    target_function_name=target_function_name
                )
                
                if not deletion or not issue:
                    logger.warning(f"[Stage 1 - Attempt {attempt+1}/{max_retries}] Failed to generate deletion/issue")
                    continue
                
                # Validate deletion patch if code_executor is provided
                if code_executor and deletion.deletion_patch_content:
                    logger.info(f"[Stage 1 - Attempt {attempt+1}/{max_retries}] Validating deletion patch...")
                    
                    # Reset repository to clean state before validation
                    logger.debug("Resetting repository to clean state...")
                    code_executor.run_command("cd /workspace/repo && git checkout main 2>/dev/null || git checkout master 2>/dev/null || true")
                    code_executor.run_command("cd /workspace/repo && git reset --hard HEAD")
                    code_executor.run_command("cd /workspace/repo && git clean -fd")
                    
                    # Verify the target function still exists in the file
                    check_cmd = f"cd /workspace/repo && grep -n 'def {deletion.function_name}' {deletion.affected_file} | head -5"
                    check_result = code_executor.run_command(check_cmd)
                    if code_executor._is_success(check_result):
                        grep_output = code_executor._extract_stdout(check_result)
                        logger.debug(f"Function '{deletion.function_name}' found in file: {grep_output[:200]}")
                    else:
                        logger.warning(f"Function '{deletion.function_name}' NOT found in {deletion.affected_file} after reset!")
                        logger.warning("This may indicate the file was modified in a previous attempt")
                    
                    fuzz = min(3 + attempt, 10)
                    valid, error = self._validate_patch(deletion.deletion_patch_content, code_executor, fuzz=fuzz)
                    
                    if not valid:
                        logger.warning(f"[Stage 1 - Attempt {attempt+1}/{max_retries}] Deletion patch validation failed: {error}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            logger.error("[Stage 1] Failed to generate valid deletion patch after all retries")
                            return None
                    
                    logger.info(f"✓ [Stage 1] Deletion patch validated successfully (fuzz={fuzz})")
                else:
                    logger.info("[Stage 1] No validation (no code_executor provided)")
                
                # Stage 1 complete
                logger.info(f"✓ [Stage 1] Feature request + issue generated: {deletion.function_name}")
                break
                
            except Exception as e:
                logger.warning(f"[Stage 1 - Attempt {attempt+1}/{max_retries}] Error: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error("[Stage 1] Failed after all retries")
                    return None
        
        if not deletion or not issue:
            logger.error("[Stage 1] Failed to generate feature request + issue")
            return None
        
        # ========== STAGE 2: Generate tests for the missing function ==========
        logger.info("[FeatureRequest - Stage 2] Generating tests for deleted function...")
        
        test = None
        feedback_history = []  # Track feedback from failed attempts
        
        for attempt in range(max_retries):
            try:
                logger.info(f"[Stage 2 - Attempt {attempt+1}/{max_retries}] Generating test code...")
                
                # Generate tests based on function specification (with feedback from previous attempts)
                test = self._generate_test_for_deleted_function(
                    deletion, metadata, source_code, file_path, max_tokens, temperature,
                    feedback_history=feedback_history if feedback_history else None
                )
                
                if not test:
                    logger.warning(f"[Stage 2 - Attempt {attempt+1}/{max_retries}] Failed to generate test")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error("[Stage 2] Failed to generate test after all retries")
                        return None
                
                # Extract test file path and build verification command
                test.test_file_path = self._extract_test_file_path(test.test_patch_content)
                test.verification_command = self._build_test_command(metadata.test_framework, test.test_file_path)
                logger.info(f"Test file: {test.test_file_path}")
                logger.info(f"Verification command: {test.verification_command}")
                
                if not test:
                    logger.warning(f"[Stage 2 - Attempt {attempt+1}/{max_retries}] Failed to generate test")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        logger.error("[Stage 2] Failed to generate test after all retries")
                        return None
                
                # Validate test patch can be applied
                if code_executor and test.test_patch_content:
                    logger.info(f"[Stage 2 - Attempt {attempt+1}/{max_retries}] Validating test patch...")
                    
                    # Reset repository to clean state before validation
                    code_executor.run_command("cd /workspace/repo && git checkout main 2>/dev/null || git checkout master 2>/dev/null || true")
                    code_executor.run_command("cd /workspace/repo && git reset --hard HEAD")
                    code_executor.run_command("cd /workspace/repo && git clean -fd")
                    logger.debug("Repository reset to clean state for test patch validation")
                    
                    # For new file patches, ensure target directory exists and file doesn't exist
                    test_file_path = self._extract_test_file_path(test.test_patch_content)
                    if test_file_path and test.test_patch_content.startswith('--- /dev/null'):
                        # This is a new file patch
                        logger.debug(f"Detected new file patch for: {test_file_path}")
                        
                        # Ensure parent directory exists
                        test_dir = '/workspace/repo/' + '/'.join(test_file_path.split('/')[:-1])
                        if test_dir != '/workspace/repo/':
                            mkdir_result = code_executor.run_command(f"mkdir -p {test_dir}")
                            logger.debug(f"Created directory: {test_dir}")
                        
                        # Remove file if it exists (from previous attempts)
                        rm_result = code_executor.run_command(f"rm -f /workspace/repo/{test_file_path}")
                        logger.debug(f"Removed existing file (if any): {test_file_path}")
                    
                    fuzz = min(3 + attempt, 10)
                    test_valid, test_error = self._validate_patch(test.test_patch_content, code_executor, fuzz=fuzz)
                    
                    if not test_valid:
                        logger.warning(f"[Stage 2 - Attempt {attempt+1}/{max_retries}] Test patch validation failed: {test_error}")
                        if attempt < max_retries - 1:
                            continue
                        else:
                            logger.error("[Stage 2] Failed to generate valid test patch after all retries")
                            return None
                    
                    logger.info(f"✓ [Stage 2] Test patch validated successfully (fuzz={fuzz})")
                    
                    # Record the successful fuzz value
                    test.test_fuzz = fuzz
                    logger.info(f"Recorded test patch fuzz value: {fuzz}")
                    
                    # Validate that tests PASS on clean code (with the function present)
                    logger.info(f"[Stage 2 - Attempt {attempt+1}/{max_retries}] Verifying tests pass on clean code...")
                    test_passes, test_error_output = self._validate_test_on_clean_code(
                        test, code_executor, metadata, fuzz=fuzz
                    )
                    
                    if not test_passes:
                        logger.warning(f"[Stage 2 - Attempt {attempt+1}/{max_retries}] Tests failed on clean code")
                        
                        # Collect feedback for next attempt
                        feedback = {
                            "attempt": attempt + 1,
                            "reason": "Test execution failed on clean code",
                            "error_output": test_error_output[:1000] if test_error_output else "No error output captured",
                            "test_file": test.test_file_path,
                        }
                        feedback_history.append(feedback)
                        
                        if attempt < max_retries - 1:
                            logger.info(f"Regenerating test with feedback from attempt {attempt+1}...")
                            logger.debug(f"Feedback: {feedback['reason']} - {feedback['error_output'][:200]}")
                            continue
                        else:
                            logger.error("[Stage 2] Tests don't pass on clean code after all retries")
                            return None
                    else:
                        logger.info("✓ [Stage 2] Tests pass on clean code")
                    
                    break
                else:
                    logger.info("[Stage 2] No test validation (no code_executor provided)")
                    break
                
            except Exception as e:
                logger.warning(f"[Stage 2 - Attempt {attempt+1}/{max_retries}] Error: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    logger.error("[Stage 2] Failed after all retries")
                    return None
        
        if not test:
            logger.error("[Stage 2] Failed to generate test")
            return None
        
        logger.info(
            f"✓ Successfully generated feature request task: {deletion.function_name} "
            f"in {file_path} at lines {deletion.line_start}-{deletion.line_end}"
        )
        return (deletion, issue, test)
    
    
    def _generate_feature_request_and_issue(
        self,
        metadata: RepoMetadata,
        source_code: str,
        file_path: str,
        max_tokens: int = 2000,
        temperature: float = 0.7,
        code_executor=None,
        target_function_name: Optional[str] = None,
    ) -> tuple[Optional[FunctionDeletionInfo], Optional[IssueInfo]]:
        """Generate feature request and issue (Stage 1)."""
        
        # If target function is specified, find it directly
        if target_function_name:
            logger.info(f"[FeatureRequest] Searching for specified function: {target_function_name}")
            return self._generate_for_specific_function(
                target_function_name, metadata, source_code, file_path, code_executor
            )
        
        # Otherwise, use LLM to select a function
        lines = source_code.split('\n')
        numbered_code = '\n'.join([f"{i+1:4d} | {line}" for i, line in enumerate(lines)])

        prompt = f"""You are a task generator. SELECT a complete function from the code for implementation task generation.

Repository: {metadata.repo_name} | Language: {metadata.language} | File: {file_path}

Source code:
```{metadata.language}
{numbered_code}
```

TASK: 
1. Select a meaningful, self-contained function (not __init__ or trivial helpers)
2. The function should be implementable based on its signature and docstring
3. Identify the EXACT line numbers where the function starts and ends

Respond with JSON:
{{
    "deletion": {{
        "deletion_id": "DEL-XXXX",
        "function_name": "function_name",
        "function_signature": "def function_name(args):",
        "function_docstring": "What the function should do",
        "affected_file": "{file_path}",
        "line_start": 10,
        "line_end": 25,
        "implementation_hints": "Hints about edge cases, return types, etc."
    }},
    "issue": {{
        "issue_title": "Implement function_name",
        "issue_body": "Detailed description of what to implement and requirements",
        "issue_labels": ["enhancement", "feature"],
        "root_cause": "Function is not implemented",
        "reproduction_steps": "Not applicable for implementation tasks"
    }}
}}

IMPORTANT: Only identify the function. Do NOT generate any patch content.
Generate only valid JSON."""

        try:
            response = self.llm.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            response = extract_json_from_response(response)
            data = json.loads(response)

            # Parse deletion info
            del_data = data.get("deletion", {})
            line_start = del_data.get("line_start", -1)
            line_end = del_data.get("line_end", -1)
            
            # Validate line numbers
            lines = source_code.split('\n')
            if line_start < 1 or line_end > len(lines) or line_start > line_end:
                logger.warning(f"Invalid line numbers: {line_start}-{line_end} (total lines: {len(lines)})")
                return (None, None)
            
            # Extract original function code from source
            original_function_code = '\n'.join(lines[line_start-1:line_end])
            
            # Build deletion patch using git diff (more reliable than manual construction)
            deletion_patch_content = ""

            logger.info("Generating deletion patch using git diff in sandbox...")
            deletion_patch_content = self._build_deletion_patch_with_git(
                code_executor, file_path, line_start, line_end,
                function_name=del_data.get("function_name", "unknown")
            )
            
            if not deletion_patch_content:
                logger.warning(f"Failed to build deletion patch for lines {line_start}-{line_end}")
                return (None, None)
            
            deletion = FunctionDeletionInfo(
                deletion_id=del_data.get("deletion_id", "DEL-0000"),
                function_name=del_data.get("function_name", "unknown"),
                function_signature=del_data.get("function_signature", ""),
                function_docstring=del_data.get("function_docstring", ""),
                affected_file=file_path,
                deletion_location=f"line {line_start}-{line_end}",
                original_function_code=original_function_code,
                deletion_patch_content=deletion_patch_content,
                line_start=line_start,
                line_end=line_end,
                implementation_hints=del_data.get("implementation_hints", "")
            )

            # Parse issue info
            issue_data = data.get("issue", {})
            issue = IssueInfo(
                issue_title=issue_data.get("issue_title", f"Implement: {deletion.function_name}"),
                issue_body=issue_data.get("issue_body", deletion.function_docstring),
                issue_labels=issue_data.get("issue_labels", ["enhancement", "feature"]),
                root_cause=issue_data.get("root_cause", "Function not implemented"),
                reproduction_steps=issue_data.get("reproduction_steps", "N/A")
            )
            
            return (deletion, issue)

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to parse deletion/issue response: {e}")
            return (None, None)
    
    def _generate_for_specific_function(
        self,
        function_name: str,
        metadata: RepoMetadata,
        source_code: str,
        file_path: str,
        code_executor,
    ) -> tuple[Optional[FunctionDeletionInfo], Optional[IssueInfo]]:
        """
        Generate feature request for a specific function by name.
        
        Args:
            function_name: Name of the function to delete
            metadata: Repository metadata
            source_code: Source code content
            file_path: Path to the file
            code_executor: Code executor for validation
            
        Returns:
            Tuple of (FunctionDeletionInfo, IssueInfo) or (None, None) if function not found
        """
        import ast
        
        logger.info(f"[SpecificFunction] Analyzing code to find function: {function_name}")
        
        try:
            # Parse the source code with AST
            tree = ast.parse(source_code)
            lines = source_code.split('\n')
            
            # Find the target function
            target_func = None
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name == function_name:
                    target_func = node
                    break
            
            if not target_func:
                logger.error(f"[SpecificFunction] Function '{function_name}' not found in {file_path}")
                return (None, None)
            
            # Extract function information
            line_start = target_func.lineno
            line_end = target_func.end_lineno
            
            logger.info(f"[SpecificFunction] Found function '{function_name}' at lines {line_start}-{line_end}")
            
            # Extract function signature
            args_list = []
            for arg in target_func.args.args:
                args_list.append(arg.arg)
            function_signature = f"def {function_name}({', '.join(args_list)}):"
            
            # Extract docstring
            docstring = ast.get_docstring(target_func) or "No docstring provided"
            
            # Extract original function code
            original_function_code = '\n'.join(lines[line_start-1:line_end])
            
            # Build deletion patch using git diff
            logger.info("[SpecificFunction] Generating deletion patch using git diff...")
            deletion_patch_content = self._build_deletion_patch_with_git(
                code_executor, file_path, line_start, line_end, function_name
            )
            
            if not deletion_patch_content:
                logger.warning(f"[SpecificFunction] Failed to build deletion patch for {function_name}")
                return (None, None)
            
            # Use LLM to generate implementation hints and issue description
            logger.info("[SpecificFunction] Generating implementation hints and issue...")
            hints_prompt = f"""Given this function, provide implementation hints and create a GitHub issue.

Function: {function_name}
Signature: {function_signature}
Docstring: {docstring}

Code:
```{metadata.language}
{original_function_code}
```

Generate JSON with:
{{
    "implementation_hints": "Detailed hints about how to implement this function, including edge cases, return types, algorithms, etc.",
    "issue_title": "Brief title for the implementation task",
    "issue_body": "Detailed description of what needs to be implemented and requirements"
}}"""

            response = self.llm.chat_completion(
                [{"role": "user", "content": hints_prompt}],
                temperature=0.7,
                max_tokens=2000,
            )
            
            response = extract_json_from_response(response)
            hints_data = json.loads(response)
            
            # Create FunctionDeletionInfo
            deletion = FunctionDeletionInfo(
                deletion_id=f"DEL-{function_name[:8].upper()}",
                function_name=function_name,
                function_signature=function_signature,
                function_docstring=docstring,
                affected_file=file_path,
                deletion_location=f"line {line_start}-{line_end}",
                original_function_code=original_function_code,
                deletion_patch_content=deletion_patch_content,
                line_start=line_start,
                line_end=line_end,
                implementation_hints=hints_data.get("implementation_hints", "Implement according to the function signature and docstring.")
            )
            
            # Create IssueInfo
            issue = IssueInfo(
                issue_title=hints_data.get("issue_title", f"Implement {function_name}"),
                issue_body=hints_data.get("issue_body", f"Implement the function {function_name} according to its specification."),
                issue_labels=["enhancement", "feature"],
                root_cause="Function not implemented",
                reproduction_steps="N/A"
            )
            
            logger.info(f"✓ [SpecificFunction] Successfully generated task for {function_name}")
            return (deletion, issue)
            
        except SyntaxError as e:
            logger.error(f"[SpecificFunction] Failed to parse source code: {e}")
            return (None, None)
        except Exception as e:
            logger.error(f"[SpecificFunction] Error generating task for {function_name}: {e}")
            return (None, None)
    
    def _build_deletion_patch_with_git(
        self,
        code_executor,
        file_path: str,
        line_start: int,
        line_end: int,
        function_name: str
    ) -> str:
        """
        Build deletion patch by actually deleting lines in sandbox and using git diff.
        
        This is more reliable than manual patch construction as it ensures the patch
        matches the actual file content.
        
        Args:
            code_executor: CodeExecutor instance
            file_path: Path to the file in repository
            line_start: Starting line number (1-indexed)
            line_end: Ending line number (1-indexed, inclusive)
            function_name: Function name for validation
            
        Returns:
            Git diff output as patch string
        """
        try:
            # Reset to clean state
            code_executor.run_command("cd /workspace/repo && git checkout main 2>/dev/null || git checkout master 2>/dev/null || true")
            code_executor.run_command("cd /workspace/repo && git reset --hard HEAD")
            code_executor.run_command("cd /workspace/repo && git clean -fd")
            
            # Verify function exists
            check_cmd = f"cd /workspace/repo && sed -n '{line_start},{line_end}p' {file_path} | head -20"
            check_result = code_executor.run_command(check_cmd)
            if not code_executor._is_success(check_result):
                logger.error(f"Failed to read lines {line_start}-{line_end} from {file_path}")
                return ""
            
            extracted_text = code_executor._extract_stdout(check_result)
            if f"def {function_name}" not in extracted_text:
                logger.error(f"Function '{function_name}' not found in specified lines")
                logger.debug(f"Extracted text: {extracted_text[:200]}")
                return ""
            
            logger.debug(f"✓ Verified function '{function_name}' exists at lines {line_start}-{line_end}")
            
            # Delete the lines using sed
            delete_cmd = f"cd /workspace/repo && sed -i '{line_start},{line_end}d' {file_path}"
            delete_result = code_executor.run_command(delete_cmd)
            if not code_executor._is_success(delete_result):
                logger.error(f"Failed to delete lines from {file_path}")
                return ""
            
            logger.debug(f"✓ Deleted lines {line_start}-{line_end} from {file_path}")
            
            # Generate git diff
            diff_cmd = f"cd /workspace/repo && git diff {file_path}"
            diff_result = code_executor.run_command(diff_cmd)
            if not code_executor._is_success(diff_result):
                logger.error("Failed to generate git diff")
                return ""
            
            patch_content = code_executor._extract_stdout(diff_result)
            
            # Validate patch has essential components (not just length)
            if not patch_content:
                logger.error("Git diff output is empty")
                return ""
            
            # Check for essential patch headers
            if '---' not in patch_content or '+++' not in patch_content:
                logger.error(f"Git diff output missing headers (length={len(patch_content)})")
                logger.debug(f"Patch content: {patch_content[:200]}")
                return ""
            
            # Check for at least one hunk (@@)
            if '@@' not in patch_content:
                logger.error(f"Git diff output missing hunk markers (length={len(patch_content)})")
                logger.debug(f"Patch content: {patch_content[:200]}")
                return ""
            
            logger.info(f"✓ Generated deletion patch using git diff ({len(patch_content)} bytes)")
            
            # Reset again to clean state for validation
            code_executor.run_command("cd /workspace/repo && git checkout main 2>/dev/null || git checkout master 2>/dev/null || true")
            code_executor.run_command("cd /workspace/repo && git reset --hard HEAD")
            code_executor.run_command("cd /workspace/repo && git clean -fd")
            
            return patch_content
            
        except Exception as e:
            logger.error(f"Failed to build deletion patch with git: {e}")
            # Reset on error
            try:
                code_executor.run_command("cd /workspace/repo && git reset --hard HEAD")
                code_executor.run_command("cd /workspace/repo && git clean -fd")
            except:
                pass
            return ""
    
    
    def _generate_test_for_deleted_function(
        self,
        deletion: FunctionDeletionInfo,
        metadata: RepoMetadata,
        source_code: str,
        file_path: str,
        max_tokens: int = 4000,
        temperature: float = 0.7,
        feedback_history: Optional[List[dict]] = None,
    ) -> Optional[TestInfo]:
        """Generate test code for a deleted function (Stage 2)."""
        module_path = file_path.replace('/', '.').replace('.py', '')

        # Build feedback section if we have previous failures
        feedback_section = ""
        if feedback_history:
            feedback_section = "\n\n**PREVIOUS ATTEMPTS FAILED:**\n"
            for i, feedback in enumerate(feedback_history, 1):
                feedback_section += f"\nAttempt {feedback['attempt']}:\n"
                feedback_section += f"- Reason: {feedback['reason']}\n"
                feedback_section += f"- Error output:\n```\n{feedback['error_output']}\n```\n"
            
            feedback_section += "\n**IMPORTANT**: Learn from the errors above. Fix the issues in your generated tests.\n"
            feedback_section += "Common issues to avoid:\n"
            feedback_section += "1. Import errors - ensure all imports are correct and modules exist\n"
            feedback_section += "2. Assertion errors - verify test expectations match actual function behavior\n"
            feedback_section += "3. Syntax errors - ensure valid Python syntax\n"
            feedback_section += "4. Missing fixtures or setup - add any necessary test setup\n"

        # Get test count configuration (default values if not in config)
        target_count = getattr(self, 'target_test_count', 4)
        min_count = getattr(self, 'min_test_count', 3)
        max_count = getattr(self, 'max_test_count', 6)

        prompt = f"""You are a test generator. Given a function specification, generate test code that this function can successfully execute.

Repository: {metadata.repo_name}
Language: {metadata.language}
File: {file_path}
Module Import Path: {module_path}
Test Framework: {metadata.test_framework}

Function to test:
- Name: {deletion.function_name}
- Signature: {deletion.function_signature}
- Purpose: {deletion.function_docstring}
- Implementation hints: {deletion.implementation_hints}

**Original Function Code (for reference):**
```{metadata.language}
{deletion.original_function_code}
```
{feedback_section}

TASK: Generate {target_count} test functions (between {min_count}-{max_count}) that validate the function works correctly.

**Test Coverage Requirements:**
1. Basic functionality ({min(2, target_count//2)} tests): Test the main use cases
2. Edge cases (1-2 tests): Boundary conditions, empty inputs, special values
3. Error handling (0-1 tests): Invalid inputs if the function has error handling

**Quality over Quantity:**
- Focus on the MOST critical and representative test scenarios
- Each test should verify a distinct aspect of the function
- Avoid redundant or trivial tests
- Prefer {target_count} tests, but ensure quality coverage

The tests should validate:
- Normal cases (based on the function logic above)
- Edge cases (boundary conditions, empty inputs, etc.)
- Error handling (if the function has try/except blocks)
- Return values (match what the original function returns)

CRITICAL JSON FORMAT REQUIREMENTS:
- Use \\n for newlines in strings (NOT actual newlines)
- Escape all quotes inside strings with \\"
- Keep the entire response as valid JSON

CRITICAL PATCH FORMAT REQUIREMENTS:
- test_patch_content MUST start with "--- /dev/null\\n+++ b/tests/test_{deletion.function_name}.py"
- Each line in the patch must be prefixed with + for new content
- Use @@ -0,0 +1,N @@ where N is the number of new lines
- Example: "--- /dev/null\\n+++ b/tests/test_example.py\\n@@ -0,0 +1,5 @@\\n+line1\\n+line2\\n+line3\\n+line4\\n+line5"

Respond with JSON (properly escaped):
{{
    "test_code": "import pytest\\nfrom {module_path} import {deletion.function_name}\\n\\ndef test_basic():\\n    result = {deletion.function_name}()\\n    assert result is not None",
    "test_patch_content": "--- /dev/null\\n+++ b/tests/test_{deletion.function_name}.py\\n@@ -0,0 +1,6 @@\\n+import pytest\\n+from {module_path} import {deletion.function_name}\\n+\\n+def test_basic():\\n+    result = {deletion.function_name}()\\n+    assert result is not None",
    "test_description": "Basic test for {deletion.function_name}"
}}"""

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.chat_completion(
                    [{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                
                # Try to extract and parse JSON
                response = extract_json_from_response(response)
                
                # Try lenient parsing first
                try:
                    data = json.loads(response)
                except json.JSONDecodeError as e:
                    # Try to fix common issues
                    logger.debug(f"[Attempt {attempt+1}/{max_retries}] JSON parse error: {e}")
                    logger.debug(f"Response preview: {response[:500]}")
                    
                    # Try to extract using regex as fallback
                    test_code_match = re.search(r'"test_code"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', response)
                    patch_match = re.search(r'"test_patch_content"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', response)
                    desc_match = re.search(r'"test_description"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', response)
                    
                    if test_code_match and patch_match:
                        test_code = test_code_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        patch_content = patch_match.group(1).replace('\\n', '\n').replace('\\"', '"')
                        description = desc_match.group(1).replace('\\n', '\n').replace('\\"', '"') if desc_match else ""
                        
                        data = {
                            "test_code": test_code,
                            "test_patch_content": patch_content,
                            "test_description": description
                        }
                        logger.info(f"✓ Extracted test using regex fallback")
                    else:
                        if attempt < max_retries - 1:
                            logger.warning(f"[Attempt {attempt+1}/{max_retries}] Failed to parse, retrying...")
                            continue
                        else:
                            raise

                test = TestInfo(
                    test_code=data.get("test_code", ""),
                    test_patch_content=data.get("test_patch_content", ""),
                    test_description=data.get("test_description", "")
                )
                
                # Validate that we got actual content
                if not test.test_code or not test.test_patch_content:
                    logger.warning(f"[Attempt {attempt+1}/{max_retries}] Empty test content, retrying...")
                    if attempt < max_retries - 1:
                        continue
                    else:
                        return None
                
                return test

            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"[Attempt {attempt+1}/{max_retries}] Failed to parse test response: {e}")
                if attempt < max_retries - 1:
                    logger.info("Retrying with adjusted prompt...")
                    continue
                else:
                    logger.error(f"Failed to generate test after {max_retries} attempts")
                    return None
        
        return None
    
    
    def _build_test_command(self, test_framework: str, test_file_path: Optional[str]) -> str:
        """
        Build test command based on framework and test file path.
        
        Args:
            test_framework: Test framework name (pytest, unittest, etc.)
            test_file_path: Path to test file (optional)
            
        Returns:
            Test command string
        """
        framework = (test_framework or "pytest").lower()
        
        if framework in ("pytest", "py.test"):
            if test_file_path:
                return f"cd /workspace/repo && set -o pipefail && pytest {test_file_path} -xvs 2>&1 | head -100"
            else:
                return "cd /workspace/repo && set -o pipefail && pytest -xvs 2>&1 | head -100"
        elif framework in ("unittest", "nose"):
            if test_file_path:
                # Convert file path to module path for unittest
                module_path = test_file_path.replace('/', '.').replace('.py', '')
                return f"cd /workspace/repo && set -o pipefail && python -m unittest {module_path} -v 2>&1 | head -100"
            else:
                return "cd /workspace/repo && set -o pipefail && python -m unittest -v 2>&1 | head -100"
        else:
            # Default to pytest
            if test_file_path:
                return f"cd /workspace/repo && set -o pipefail && pytest {test_file_path} -xvs 2>&1 | head -100"
            else:
                return "cd /workspace/repo && set -o pipefail && pytest -xvs 2>&1 | head -100"
    
    def _install_test_dependencies(
        self,
        code_executor,
        metadata: RepoMetadata
    ) -> None:
        """
        Install project dependencies before running tests.
        
        Args:
            code_executor: Code executor instance
            metadata: Repository metadata with dependency information
        """
        try:
            logger.info("Installing project dependencies...")
            
            # Strategy: Try multiple installation methods in order of preference
            install_cmds = []
            
            # 1. Check for common dependency files and add appropriate commands
            file_checks = {
                "environment.yml": "cd /workspace/repo && conda env update -f environment.yml 2>&1 | tail -20",
                "environment.yaml": "cd /workspace/repo && conda env update -f environment.yaml 2>&1 | tail -20",
                "requirements.txt": "cd /workspace/repo && pip install -r requirements.txt 2>&1 | tail -20",
                "setup.py": "cd /workspace/repo && pip install -e . 2>&1 | tail -20",
                "pyproject.toml": "cd /workspace/repo && pip install . 2>&1 | tail -20",
            }
            
            for file_name, install_cmd in file_checks.items():
                check_result = code_executor.run_command(f"test -f /workspace/repo/{file_name} && echo 'exists'")
                if code_executor._is_success(check_result):
                    stdout = code_executor._extract_stdout(check_result)
                    if 'exists' in stdout:
                        install_cmds.append(install_cmd)
                        logger.debug(f"Found {file_name}, will try: {install_cmd}")
            
            # 2. If we have extracted dependencies but no dependency files, install them directly
            if metadata.dependencies and not install_cmds:
                # Filter out common packages that might cause issues
                deps_to_install = [
                    dep for dep in metadata.dependencies 
                    if dep and not dep.startswith(('#', '-', 'python'))
                ]
                if deps_to_install:
                    # Install in batches to avoid command line length limits
                    batch_size = 20
                    for i in range(0, len(deps_to_install), batch_size):
                        batch = deps_to_install[i:i+batch_size]
                        deps_str = ' '.join(batch)
                        install_cmds.append(f"cd /workspace/repo && pip install {deps_str} 2>&1 | tail -20")
                        logger.debug(f"Will install batch {i//batch_size + 1}: {len(batch)} packages")
            
            # 3. Try each installation command until one succeeds
            for cmd in install_cmds:
                logger.info(f"Trying: {cmd[:80]}...")
                result = code_executor.run_command(cmd, timeout_s=180)
                if result.get("status") == "Success" or result.get("exit_code", 1) == 0:
                    logger.info(f"✓ Dependencies installed successfully")
                    return
                else:
                    stdout = code_executor._extract_stdout(result)
                    logger.debug(f"Installation attempt failed: {stdout[-200:]}")
            
            # If all fail, log warning but continue (tests might still work)
            logger.warning("Could not install dependencies, but continuing...")
            
        except Exception as e:
            logger.warning(f"Error installing dependencies: {e}. Continuing anyway...")
    
    def _validate_test_on_clean_code(
        self,
        test: TestInfo,
        code_executor,
        metadata: RepoMetadata,
        fuzz: int = 5
    ) -> tuple[bool, Optional[str]]:
        """
        Validate that tests pass on clean code (for feature_request mode).
        
        This ensures the generated tests are correct and can run successfully
        when the function is present.
        
        Args:
            test: TestInfo with test patch
            code_executor: Code executor instance
            metadata: Repository metadata
            fuzz: Fuzz value to use when applying patch (should match validation fuzz)
            
        Returns:
            Tuple of (success: bool, error_output: Optional[str])
            - success: True if tests pass on clean code, False otherwise
            - error_output: Error message/output if tests failed, None if passed
        """
        try:
            import uuid
            
            # Create a temporary branch for validation
            logger.info("Creating temporary branch for test validation on clean code...")
            code_executor.run_command("cd /workspace/repo && git checkout main 2>/dev/null || git checkout master 2>/dev/null || true")
            code_executor.run_command("cd /workspace/repo && git reset --hard HEAD")
            code_executor.run_command("cd /workspace/repo && git clean -fd")
            code_executor.run_command("cd /workspace/repo && git branch -D temp_clean_test 2>/dev/null || true")
            code_executor.run_command("cd /workspace/repo && git checkout -b temp_clean_test")
            
            # For new file patches, ensure target directory exists and file doesn't exist
            test_file_path = self._extract_test_file_path(test.test_patch_content)
            if test_file_path and test.test_patch_content.startswith('--- /dev/null'):
                logger.debug(f"Preparing for new test file: {test_file_path}")
                
                # Ensure parent directory exists
                test_dir = '/workspace/repo/' + '/'.join(test_file_path.split('/')[:-1])
                if test_dir != '/workspace/repo/':
                    code_executor.run_command(f"mkdir -p {test_dir}")
                    logger.debug(f"Created directory: {test_dir}")
                
                # Remove file if it exists
                code_executor.run_command(f"rm -f /workspace/repo/{test_file_path}")
                logger.debug(f"Removed existing file (if any): {test_file_path}")
            
            # Apply test patch
            test_patch_uuid = uuid.uuid4()
            test_patch_path = f"/workspace/tmp/test_clean_{test_patch_uuid}.patch"
            
            write_cmd = f"""mkdir -p /workspace/tmp && cat > {test_patch_path} << 'EOF'
{test.test_patch_content}
EOF"""
            write_result = code_executor.run_command(write_cmd)
            if not code_executor._is_success(write_result):
                logger.error(f"✗ Failed to write test patch file for clean code validation")
                return (False, "Failed to write test patch file")
            
            # Use the same fuzz value that was successful during validation
            apply_cmd = f"cd /workspace/repo && patch -p1 --fuzz={fuzz} --verbose < {test_patch_path} 2>&1"
            logger.info(f"Applying test patch with fuzz={fuzz} (same as validation)")
            apply_result = code_executor.run_command(apply_cmd)
            
            if not code_executor._is_success(apply_result):
                stdout = code_executor._extract_stdout(apply_result)
                logger.warning(f"Failed to apply test patch on clean code")
                logger.warning(f"Patch output: {stdout[:500]}")
                self._cleanup_test_validation(code_executor, test_patch_path)
                return (False, f"Failed to apply test patch: {stdout[:500]}")
            
            logger.info("✓ Test patch applied to clean code")
            
            # Install dependencies
            logger.info("Installing test dependencies...")
            self._install_test_dependencies(code_executor, metadata)
              
            # Build test command
            test_file_path = self._extract_test_file_path(test.test_patch_content)
            test_cmd = self._build_test_command(metadata.test_framework, test_file_path)
            logger.info(f"Running tests on clean code: {test_cmd}")

            # Run tests
            test_result = code_executor.run_command(test_cmd)
 
            # Clean up
            self._cleanup_test_validation(code_executor, test_patch_path)
            
            if code_executor._is_success(test_result):
                logger.info("✓ Tests PASSED on clean code")
                return (True, None)
            else:
                stdout = code_executor._extract_stdout(test_result)
                logger.warning(f"✗ Tests FAILED on clean code")
                logger.warning(f"Output: {stdout[-500:]}")
                return (False, stdout)
                
        except Exception as e:
            logger.error(f"Exception during test validation on clean code: {e}")
            return (False, f"Exception: {str(e)}")
    
    def _extract_test_file_path(self, patch_content: str) -> Optional[str]:
        """
        Extract test file path from unified diff patch content.
        
        Args:
            patch_content: Unified diff patch content
            
        Returns:
            Test file path (e.g., "tests/test_main.py") or None if not found
        """
        try:
            # Look for "+++ b/path/to/file" pattern in patch (first line only)
            match = re.search(r'^\+\+\+ b/([^\r\n]+)', patch_content, re.MULTILINE)
            if match:
                # Take only the path token (stop at whitespace or diff markers)
                path_line = match.group(1).strip()
                file_path = path_line.split()[0].strip()
                logger.debug(f"Extracted test file path from patch: {file_path}")
                return file_path
            else:
                logger.warning("Could not extract test file path from patch")
                return None
        except Exception as e:
            logger.warning(f"Error extracting test file path: {e}")
            return None
    
    def _cleanup_test_validation(
        self,
        code_executor,
        test_patch_path: str
    ) -> None:
        """
        Clean up temporary files and git state after test validation.
        
        Args:
            code_executor: Code executor instance
            test_patch_path: Path to test patch file
        """
        try:
            # Remove any working tree changes on the temp branch so new files don't linger
            code_executor.run_command("cd /workspace/repo && git reset --hard HEAD")
            code_executor.run_command("cd /workspace/repo && git clean -fd")

            code_executor.run_command(f"rm -f {test_patch_path}")

            code_executor.run_command("cd /workspace/repo && git checkout main 2>/dev/null || git checkout master 2>/dev/null || true")
            # Clean again after switching back to main/master to ensure no stray files remain
            code_executor.run_command("cd /workspace/repo && git reset --hard HEAD")
            code_executor.run_command("cd /workspace/repo && git clean -fd")
            code_executor.run_command("cd /workspace/repo && git branch -D temp_test_branch 2>/dev/null || true")
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
