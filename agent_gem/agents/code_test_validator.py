"""Generate and validate test cases for code bug fixes."""

from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_gem.llm import LLMClient
from agent_gem.agents.code_repo_analyzer import RepoMetadata
from agent_gem.agents.code_bug_generator import BugInfo, IssuePRInfo, extract_json_from_response

logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    """A single test case."""

    test_id: str
    test_name: str
    test_code: str
    test_framework: str
    description: str
    assertion_description: str


@dataclass
class TestValidationResult:
    """Result of running tests against buggy and fixed code."""

    test_id: str
    test_name: str
    fails_on_buggy: bool  # Should fail on buggy code
    passes_on_fixed: bool  # Should pass on fixed code
    error_message_buggy: Optional[str] = None
    error_message_fixed: Optional[str] = None
    is_valid_test: bool = False  # True if test behaves as expected


class TestCaseGenerator:
    """Generate test cases for bug fixes."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_test_case(
        self,
        bug: BugInfo,
        metadata: RepoMetadata,
        buggy_code: str,
        fixed_code: str,
        max_tokens: int = 1500,
        temperature: float = 0.6,
    ) -> Optional[TestCase]:
        """
        Generate a test case that should fail on buggy code and pass on fixed code.
        
        Args:
            bug: BugInfo object
            metadata: Repository metadata
            buggy_code: The buggy code
            fixed_code: The fixed code
            
        Returns:
            TestCase object
        """
        prompt = f"""Generate a unit test that will FAIL on the buggy code but PASS on the fixed code.

Language: {metadata.language}
Test Framework: {metadata.test_framework}
Bug Description: {bug.bug_description}
Buggy Code:
```{metadata.language}
{buggy_code}
```

Fixed Code:
```{metadata.language}
{fixed_code}
```

Generate a JSON response with:
{{
    "test_name": "test_function_name",
    "test_code": "Complete test code that imports the function and tests it",
    "assertion_description": "What the test is checking for",
    "explanation": "Why this test will fail on buggy code"
}}

The test code must:
1. Be valid {metadata.language} code for {metadata.test_framework}
2. Import or reference the function being tested
3. Have clear assertions that would fail on the buggy version
4. Be self-contained and runnable

Return only valid JSON."""

        try:
            response = self.llm.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract JSON from response
            response = extract_json_from_response(response)
            data = json.loads(response)

            test = TestCase(
                test_id=f"test_{bug.bug_id}",
                test_name=data.get("test_name", f"test_{bug.bug_id}"),
                test_code=data.get("test_code", ""),
                test_framework=metadata.test_framework,
                description=data.get("assertion_description", ""),
                assertion_description=data.get("assertion_description", ""),
            )

            logger.info(f"Generated test case: {test.test_id}")
            return test

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to generate test case: {e}")
            return None

    def generate_multiple_test_cases(
        self,
        bug: BugInfo,
        metadata: RepoMetadata,
        buggy_code: str,
        fixed_code: str,
        num_tests: int = 2,
        max_tokens: int = 1500,
        temperature: float = 0.6,
    ) -> List[TestCase]:
        """Generate multiple test cases for a single bug."""
        tests: List[TestCase] = []

        for i in range(num_tests):
            test = self.generate_test_case(
                bug, metadata, buggy_code, fixed_code,
                max_tokens=max_tokens,
                temperature=temperature
            )
            if test:
                tests.append(test)

        return tests


class TestValidator:
    """Validate test cases by running them against buggy and fixed code."""

    def __init__(self, timeout_s: int = 30):
        self.timeout_s = timeout_s

    def validate_test(
        self,
        test: TestCase,
        buggy_code: str,
        fixed_code: str,
        metadata: RepoMetadata,
    ) -> TestValidationResult:
        """
        Run a test against both buggy and fixed code.
        
        Returns:
            TestValidationResult with validation outcome
        """
        # Create temp directory for test execution
        temp_dir = Path(tempfile.mkdtemp(prefix="test_validation_"))

        try:
            result_buggy = self._run_test(
                test, buggy_code, temp_dir, "buggy", metadata
            )
            result_fixed = self._run_test(
                test, fixed_code, temp_dir, "fixed", metadata
            )

            # Test is valid if it fails on buggy and passes on fixed
            is_valid = (not result_buggy[0]) and result_fixed[0]

            return TestValidationResult(
                test_id=test.test_id,
                test_name=test.test_name,
                fails_on_buggy=not result_buggy[0],
                passes_on_fixed=result_fixed[0],
                error_message_buggy=result_buggy[1],
                error_message_fixed=result_fixed[1],
                is_valid_test=is_valid,
            )

        finally:
            # Cleanup
            import shutil

            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_test(
        self,
        test: TestCase,
        code: str,
        work_dir: Path,
        variant: str,
        metadata: RepoMetadata,
    ) -> Tuple[bool, Optional[str]]:
        """
        Run a single test variant.
        
        Returns:
            (success: bool, error_message: Optional[str])
        """
        if metadata.language == "python":
            return self._run_pytest(test, code, work_dir, variant, metadata)
        elif metadata.language in ("javascript", "typescript"):
            return self._run_jest(test, code, work_dir, variant, metadata)
        else:
            logger.warning(f"Test validation not supported for {metadata.language}")
            return (False, "Language not supported")

    def _run_pytest(
        self,
        test: TestCase,
        code: str,
        work_dir: Path,
        variant: str,
        metadata: RepoMetadata,
    ) -> Tuple[bool, Optional[str]]:
        """Run test with pytest."""
        # Create main module with the code
        main_file = work_dir / f"code_{variant}.py"
        main_file.write_text(code)

        # Create test file
        test_file = work_dir / f"test_{variant}.py"
        test_code = self._adapt_test_for_pytest(test.test_code, metadata)
        test_file.write_text(test_code)

        try:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "pytest",
                    str(test_file),
                    "-v",
                    "--tb=short",
                ],
                cwd=str(work_dir),
                capture_output=True,
                timeout=self.timeout_s,
                text=True,
            )

            success = result.returncode == 0
            error_msg = result.stdout + result.stderr if not success else None

            return (success, error_msg)

        except subprocess.TimeoutExpired:
            return (False, "Test execution timed out")
        except Exception as e:
            return (False, str(e))

    def _run_jest(
        self,
        test: TestCase,
        code: str,
        work_dir: Path,
        variant: str,
        metadata: RepoMetadata,
    ) -> Tuple[bool, Optional[str]]:
        """Run test with jest."""
        # Similar to pytest but for JavaScript
        test_file = work_dir / f"test_{variant}.js"
        test_code = self._adapt_test_for_jest(test.test_code, code, metadata)
        test_file.write_text(test_code)

        try:
            result = subprocess.run(
                ["npm", "test", "--", str(test_file)],
                cwd=str(work_dir),
                capture_output=True,
                timeout=self.timeout_s,
                text=True,
            )

            success = result.returncode == 0
            error_msg = result.stdout + result.stderr if not success else None

            return (success, error_msg)

        except subprocess.TimeoutExpired:
            return (False, "Test execution timed out")
        except Exception as e:
            return (False, str(e))

    def _adapt_test_for_pytest(self, test_code: str, metadata: RepoMetadata) -> str:
        """Adapt test code for pytest execution."""
        # Ensure it has proper imports and structure
        if "import pytest" not in test_code and "from pytest import" not in test_code:
            test_code = "import sys\nfrom pathlib import Path\n" + test_code

        # Ensure functions are properly formatted for pytest
        if "def test_" not in test_code:
            # Wrap in a test function if not already wrapped
            test_code = "def test_generated():\n" + "\n".join(
                "    " + line for line in test_code.split("\n")
            )

        return test_code

    def _adapt_test_for_jest(self, test_code: str, source_code: str, metadata: RepoMetadata) -> str:
        """Adapt test code for jest execution."""
        # For JavaScript/TypeScript, include source code and test
        adapted = f"""
{source_code}

{test_code}
"""
        return adapted

    def validate_multiple(
        self,
        tests: List[TestCase],
        buggy_code: str,
        fixed_code: str,
        metadata: RepoMetadata,
    ) -> List[TestValidationResult]:
        """Validate multiple test cases."""
        results: List[TestValidationResult] = []

        for test in tests:
            result = self.validate_test(test, buggy_code, fixed_code, metadata)
            results.append(result)

            logger.info(
                f"Test validation: {test.test_id} - "
                f"fails_on_buggy={result.fails_on_buggy}, "
                f"passes_on_fixed={result.passes_on_fixed}"
            )

        valid_count = sum(1 for r in results if r.is_valid_test)
        logger.info(f"Valid tests: {valid_count}/{len(results)}")

        return results
