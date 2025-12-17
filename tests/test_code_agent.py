"""
Unit tests for CodeAgent components.

Tests the repository analysis, bug generation, test case generation, and validation.
"""

import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import tempfile
import json

from agent_gem.agents.code_repo_analyzer import RepositoryAnalyzer, RepoMetadata, DependencyConfigurer
from agent_gem.agents.code_bug_generator import BugGenerator, BugInfo, IssuePRGenerator
from agent_gem.agents.code_test_validator import TestCaseGenerator, TestValidator, TestCase
from agent_gem.llm import LLMClient


class TestRepositoryAnalyzer(unittest.TestCase):
    """Test cases for RepositoryAnalyzer."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=LLMClient)
        self.analyzer = RepositoryAnalyzer(self.mock_llm)

    def test_detect_language_python(self):
        """Test Python language detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create Python files
            (tmppath / "main.py").write_text("print('hello')")
            (tmppath / "utils.py").write_text("def foo(): pass")

            detected = self.analyzer._detect_language(tmppath)
            self.assertEqual(detected, "python")

    def test_detect_language_javascript(self):
        """Test JavaScript language detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create JavaScript files
            (tmppath / "index.js").write_text("console.log('hello');")
            (tmppath / "app.js").write_text("const x = 1;")

            detected = self.analyzer._detect_language(tmppath)
            self.assertEqual(detected, "javascript")

    def test_extract_python_dependencies_requirements_txt(self):
        """Test dependency extraction from requirements.txt."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "requirements.txt").write_text(
                "numpy>=1.0\npandas==1.2.0\n# comment\nrequests"
            )

            deps = self.analyzer._extract_dependencies(tmppath, "python")
            self.assertIn("numpy>=1.0", deps)
            self.assertIn("pandas==1.2.0", deps)
            self.assertIn("requests", deps)
            self.assertNotIn("# comment", deps)

    def test_detect_test_framework_pytest(self):
        """Test pytest framework detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "requirements.txt").write_text("pytest>=6.0\n")
            (tmppath / "test_main.py").write_text("def test_foo(): pass")

            framework = self.analyzer._detect_test_framework(tmppath, "python")
            self.assertEqual(framework, "pytest")

    def test_detect_build_system(self):
        """Test build system detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            (tmppath / "setup.py").write_text("from setuptools import setup\n")

            system = self.analyzer._detect_build_system(tmppath)
            self.assertEqual(system, "setuptools")


class TestBugGenerator(unittest.TestCase):
    """Test cases for BugGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=LLMClient)
        self.generator = BugGenerator(self.mock_llm)

    def test_generate_bug_success(self):
        """Test successful bug generation."""
        # Mock LLM response
        self.mock_llm.chat_completion.return_value = json.dumps({
            "bug_id": "BUG-001",
            "bug_title": "Off-by-one error in loop",
            "bug_description": "Loop iterates one too many times",
            "bug_type": "off-by-one",
            "severity": "high",
            "affected_function": "process_items",
            "bug_location": "line 10-15",
            "buggy_code": "for i in range(len(items) + 1):",
        })

        metadata = RepoMetadata(
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            language="python",
            dependencies=["requests"],
            main_files=["main.py"],
            test_framework="pytest",
            build_system="setuptools",
            entry_points=["main.py"],
        )

        bug = self.generator.generate_bug(
            metadata,
            "for i in range(len(items) + 1):\n    print(items[i])",
            "main.py",
        )

        self.assertIsNotNone(bug)
        self.assertEqual(bug.bug_id, "BUG-001")
        self.assertEqual(bug.bug_type, "off-by-one")
        self.assertEqual(bug.severity, "high")

    def test_generate_bug_invalid_json(self):
        """Test bug generation with invalid JSON response."""
        self.mock_llm.chat_completion.return_value = "Invalid JSON {]"

        metadata = RepoMetadata(
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            language="python",
            dependencies=[],
            main_files=[],
            test_framework="pytest",
            build_system="setuptools",
            entry_points=[],
        )

        bug = self.generator.generate_bug(metadata, "code", "file.py")
        self.assertIsNone(bug)


class TestIssuePRGenerator(unittest.TestCase):
    """Test cases for IssuePRGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=LLMClient)
        self.generator = IssuePRGenerator(self.mock_llm)

    def test_generate_issue_and_pr(self):
        """Test successful issue and PR generation."""
        self.mock_llm.chat_completion.return_value = json.dumps({
            "issue_title": "Fix off-by-one error in loop",
            "issue_body": "## Description\nLoop iterates one too many times",
            "issue_labels": ["bug", "high-priority"],
            "pr_title": "Fix: off-by-one error",
            "pr_description": "Fixes #123",
            "pr_changes_summary": "Changed loop range",
            "fixed_code": "for i in range(len(items)):",
            "test_additions": "def test_loop_range(): assert len(list(range(len(items)))) == len(items)",
        })

        bug = BugInfo(
            bug_id="BUG-001",
            bug_title="Off-by-one error",
            bug_description="Loop iterates one too many times",
            bug_type="off-by-one",
            severity="high",
            affected_file="main.py",
            affected_function="process",
            bug_location="10-15",
            buggy_code="for i in range(len(items) + 1):",
        )

        metadata = RepoMetadata(
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            language="python",
            dependencies=[],
            main_files=[],
            test_framework="pytest",
            build_system="setuptools",
            entry_points=[],
        )

        issue_pr = self.generator.generate_issue_and_pr(bug, metadata)

        self.assertIsNotNone(issue_pr)
        self.assertEqual(issue_pr.issue_title, "Fix off-by-one error in loop")
        self.assertIn("bug", issue_pr.issue_labels)
        self.assertEqual(issue_pr.fixed_code, "for i in range(len(items)):")


class TestDependencyConfigurer(unittest.TestCase):
    """Test cases for DependencyConfigurer."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_llm = MagicMock(spec=LLMClient)
        self.configurer = DependencyConfigurer(self.mock_llm)

    def test_generate_setup_commands_python(self):
        """Test Python setup command generation."""
        metadata = RepoMetadata(
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            language="python",
            dependencies=["requests", "numpy"],
            main_files=["main.py"],
            test_framework="pytest",
            build_system="setuptools",
            entry_points=["main.py"],
        )

        setup_info = self.configurer.generate_setup_commands(metadata)

        self.assertIn("install_deps", setup_info)
        self.assertIn("setup_env", setup_info)
        self.assertIn("test_commands", setup_info)
        self.assertTrue(any("pip install" in cmd for cmd in setup_info["install_deps"]))
        self.assertTrue(any("pytest" in cmd for cmd in setup_info["test_commands"]))

    def test_generate_setup_commands_javascript(self):
        """Test JavaScript setup command generation."""
        metadata = RepoMetadata(
            repo_url="https://github.com/test/repo",
            repo_name="test-repo",
            language="javascript",
            dependencies=["express", "lodash"],
            main_files=["index.js"],
            test_framework="jest",
            build_system="npm",
            entry_points=["index.js"],
        )

        setup_info = self.configurer.generate_setup_commands(metadata)

        self.assertIn("npm install", setup_info["install_deps"][0])
        self.assertIn("npm test", setup_info["test_commands"][0])


class TestTestCaseValidator(unittest.TestCase):
    """Test cases for TestValidator."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = TestValidator(timeout_s=5)

    def test_adapt_test_for_pytest(self):
        """Test pytest adaptation."""
        test_code = "assert 1 == 1"
        adapted = self.validator._adapt_test_for_pytest(test_code, None)
        self.assertIn("def test_", adapted)

    def test_adapt_test_already_wrapped(self):
        """Test that already wrapped tests are not double-wrapped."""
        test_code = "def test_example():\n    assert 1 == 1"
        adapted = self.validator._adapt_test_for_pytest(test_code, None)
        # Should not have extra indentation
        self.assertIn("def test_example", adapted)


if __name__ == "__main__":
    unittest.main()
