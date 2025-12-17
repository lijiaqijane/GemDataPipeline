"""Analyze GitHub repositories for CodeAgent training data generation."""

from __future__ import annotations

import json
import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_gem.llm import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class RepoMetadata:
    """Metadata extracted from a repository."""

    repo_url: str
    repo_name: str
    language: str
    dependencies: List[str]
    main_files: List[str]
    test_framework: str
    build_system: str
    entry_points: List[str]


class RepositoryAnalyzer:
    """Analyze a GitHub repository to extract metadata and dependencies."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze_repo(
        self, repo_url: str, local_path: Optional[Path] = None
    ) -> RepoMetadata:
        """
        Analyze a GitHub repository.
        
        Args:
            repo_url: Full GitHub URL or owner/repo format
            local_path: Optional path to pre-cloned repo, otherwise will clone
            
        Returns:
            RepoMetadata with extracted information
        """
        # Normalize repo URL
        if not repo_url.startswith("http"):
            repo_url = f"https://github.com/{repo_url}"

        # Clone or use existing repo
        if local_path is None:
            local_path = Path(tempfile.mkdtemp(prefix="repo_analysis_"))
            logger.info(f"Cloning repository from {repo_url} to {local_path}")
            self._clone_repo(repo_url, local_path)
        else:
            local_path = Path(local_path)

        # Extract metadata
        repo_name = local_path.name
        language = self._detect_language(local_path)
        dependencies = self._extract_dependencies(local_path, language)
        main_files = self._find_main_files(local_path, language)
        test_framework = self._detect_test_framework(local_path, language)
        build_system = self._detect_build_system(local_path)
        entry_points = self._find_entry_points(local_path, language)

        return RepoMetadata(
            repo_url=repo_url,
            repo_name=repo_name,
            language=language,
            dependencies=dependencies,
            main_files=main_files,
            test_framework=test_framework,
            build_system=build_system,
            entry_points=entry_points,
        )

    def _clone_repo(self, repo_url: str, target_path: Path) -> None:
        """Clone a GitHub repository."""
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo_url, str(target_path)],
                check=True,
                capture_output=True,
                timeout=120,
            )
            logger.info(f"Successfully cloned repository to {target_path}")
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to clone repository: {e.stderr.decode()}")
        except subprocess.TimeoutExpired:
            raise RuntimeError("Repository clone timed out")

    def _detect_language(self, repo_path: Path) -> str:
        """Detect primary programming language of the repository."""
        file_extensions = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".go": "go",
            ".rs": "rust",
            ".cpp": "cpp",
            ".c": "c",
            ".cs": "csharp",
            ".rb": "ruby",
        }

        file_counts: Dict[str, int] = {}
        for ext, lang in file_extensions.items():
            count = len(list(repo_path.rglob(f"*{ext}")))
            if count > 0:
                file_counts[lang] = count

        if not file_counts:
            return "unknown"

        primary_language = max(file_counts.items(), key=lambda x: x[1])[0]
        logger.info(f"Detected primary language: {primary_language}")
        return primary_language

    def _extract_dependencies(self, repo_path: Path, language: str) -> List[str]:
        """Extract dependencies based on language and package manager."""
        dependencies: List[str] = []

        if language == "python":
            # Check for requirements.txt
            req_file = repo_path / "requirements.txt"
            if req_file.exists():
                dependencies.extend(
                    [
                        line.strip()
                        for line in req_file.read_text().splitlines()
                        if line.strip() and not line.startswith("#")
                    ]
                )

            # Check for setup.py or setup.cfg
            setup_py = repo_path / "setup.py"
            if setup_py.exists():
                content = setup_py.read_text()
                # Simple regex to extract install_requires
                match = re.search(r'install_requires\s*=\s*\[(.*?)\]', content, re.DOTALL)
                if match:
                    deps_str = match.group(1)
                    deps = re.findall(r'"([^"]+)"|\'([^\']+)\'', deps_str)
                    dependencies.extend([d[0] or d[1] for d in deps])

            # Check for pyproject.toml
            pyproject = repo_path / "pyproject.toml"
            if pyproject.exists():
                content = pyproject.read_text()
                # Extract dependencies section
                match = re.search(
                    r'dependencies\s*=\s*\[(.*?)\]', content, re.DOTALL
                )
                if match:
                    deps_str = match.group(1)
                    deps = re.findall(r'"([^"]+)"|\'([^\']+)\'', deps_str)
                    dependencies.extend([d[0] or d[1] for d in deps])

        elif language == "javascript" or language == "typescript":
            # Check package.json
            pkg_json = repo_path / "package.json"
            if pkg_json.exists():
                try:
                    pkg_data = json.loads(pkg_json.read_text())
                    dependencies.extend(
                        list(pkg_data.get("dependencies", {}).keys())
                    )
                    dependencies.extend(
                        list(pkg_data.get("devDependencies", {}).keys())
                    )
                except json.JSONDecodeError:
                    pass

        logger.info(f"Extracted {len(dependencies)} dependencies")
        return list(set(dependencies))  # Remove duplicates

    def _find_main_files(self, repo_path: Path, language: str) -> List[str]:
        """Find main source files in the repository."""
        main_files: List[str] = []

        if language == "python":
            # Look for main.py, __main__.py, or entry point in setup.py
            patterns = ["main.py", "__main__.py", "app.py", "cli.py"]
            for pattern in patterns:
                matches = list(repo_path.rglob(pattern))
                main_files.extend([str(m.relative_to(repo_path)) for m in matches])

            # Also look for files in src/ directory
            src_dir = repo_path / "src"
            if src_dir.exists():
                py_files = list(src_dir.rglob("*.py"))
                main_files.extend(
                    [str(f.relative_to(repo_path)) for f in py_files[:5]]
                )

        elif language == "javascript" or language == "typescript":
            patterns = ["index.js", "index.ts", "main.js", "main.ts", "app.js"]
            for pattern in patterns:
                matches = list(repo_path.rglob(pattern))
                main_files.extend([str(m.relative_to(repo_path)) for m in matches])

        return main_files[:10]  # Limit to 10 files

    def _detect_test_framework(self, repo_path: Path, language: str) -> str:
        """Detect test framework used in the repository."""
        if language == "python":
            test_frameworks = ["pytest", "unittest", "nose2", "hypothesis"]
            req_file = repo_path / "requirements.txt"
            if req_file.exists():
                content = req_file.read_text().lower()
                for framework in test_frameworks:
                    if framework in content:
                        return framework

            # Check for test files
            test_files = list(repo_path.rglob("test_*.py")) + list(
                repo_path.rglob("*_test.py")
            )
            if test_files:
                # Default to pytest for Python
                return "pytest"

            return "pytest"

        elif language == "javascript" or language == "typescript":
            return "jest"

        return "unknown"

    def _detect_build_system(self, repo_path: Path) -> str:
        """Detect the build system used in the repository."""
        build_indicators = {
            "Makefile": "make",
            "tox.ini": "tox",
            "setup.py": "setuptools",
            "setup.cfg": "setuptools",
            "pyproject.toml": "poetry/setuptools",
            "Cargo.toml": "cargo",
            "build.gradle": "gradle",
            "pom.xml": "maven",
            "package.json": "npm/yarn",
        }

        for filename, system in build_indicators.items():
            if (repo_path / filename).exists():
                return system

        return "unknown"

    def _find_entry_points(self, repo_path: Path, language: str) -> List[str]:
        """Find entry points (main functions/modules) in the repository."""
        entry_points: List[str] = []

        if language == "python":
            # Check main.py
            main_py = repo_path / "main.py"
            if main_py.exists():
                entry_points.append("main.py")

            # Check __main__.py
            for main_module in repo_path.rglob("__main__.py"):
                entry_points.append(str(main_module.relative_to(repo_path)))

        return entry_points


class DependencyConfigurer:
    """Configure dependencies in a sandbox environment."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_setup_commands(
        self, metadata: RepoMetadata
    ) -> Dict[str, Any]:
        """
        Generate commands to set up the repository environment.
        
        Returns:
            Dict with 'install_deps', 'setup_env', and 'test_commands'
        """
        commands: Dict[str, Any] = {
            "install_deps": [],
            "setup_env": [],
            "test_commands": [],
        }

        if metadata.language == "python":
            # Generate pip install commands
            if metadata.dependencies:
                # Group by main packages
                deps_str = " ".join(metadata.dependencies[:20])  # Limit to 20
                commands["install_deps"] = [
                    f"pip install --upgrade pip",
                    f"pip install {deps_str}",
                ]

            # Setup env commands
            commands["setup_env"] = [
                "cd /workspace/repo",
                "python -m pip list",
            ]

            # Test commands
            if "pytest" in metadata.test_framework:
                commands["test_commands"] = [
                    "cd /workspace/repo",
                    "pytest -v",
                    "pytest --tb=short",
                ]
            elif "unittest" in metadata.test_framework:
                commands["test_commands"] = [
                    "cd /workspace/repo",
                    "python -m unittest discover -v",
                ]

        elif metadata.language in ("javascript", "typescript"):
            commands["install_deps"] = [
                "npm install",
            ]
            commands["setup_env"] = [
                "npm list",
            ]
            commands["test_commands"] = [
                "npm test",
            ]

        return commands
