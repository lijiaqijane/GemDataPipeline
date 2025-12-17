"""SandboxFusion integration for CodeAgent.

Manages the lifecycle of SandboxFusionExecutor for code generation tasks.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Generator, Optional

from agent_gem.sandbox.executor import SandboxFusionExecutor, DockerAPIRunner

logger = logging.getLogger(__name__)


class SandboxManager:
    """Manages SandboxFusion container lifecycle for CodeAgent."""

    def __init__(
        self,
        sandbox_url: Optional[str] = None,
        docker_image: Optional[str] = None,
        docker_cmd: Optional[str] = None,
        use_docker_runner: bool = True,
        use_china_mirror: bool = True,
        silent: bool = False,
    ) -> None:
        """
        Initialize SandboxManager.

        Args:
            sandbox_url: URL of SandboxFusion service (default from SANDBOX_URL env)
            docker_image: Custom Docker image (default from SANDBOX_IMAGE env)
            docker_cmd: Custom Docker run command (default from SANDBOX_CMD env)
            use_docker_runner: Whether to automatically manage Docker container
            use_china_mirror: Whether to use China mirror for docker image (ignored if docker_image is set)
            silent: Whether to suppress logging
        """
        self.sandbox_url = sandbox_url or os.getenv(
            "SANDBOX_URL", "http://localhost:8080"
        )
        self.docker_image = docker_image or os.getenv("SANDBOX_IMAGE")
        self.docker_cmd = docker_cmd or os.getenv("SANDBOX_CMD")
        self.use_docker_runner = use_docker_runner
        self.use_china_mirror = use_china_mirror
        self.silent = silent
        
        self.docker_runner: Optional[DockerAPIRunner] = None
        self.executor: Optional[SandboxFusionExecutor] = None

    @contextmanager
    def managed_executor(self) -> Generator[SandboxFusionExecutor, None, None]:
        """
        Context manager for managed SandboxFusion executor lifecycle.

        Yields:
            SandboxFusionExecutor instance
        """
        try:
            self.start()
            yield self.executor
        finally:
            self.stop()

    def start(self) -> SandboxFusionExecutor:
        """
        Start SandboxFusion service and executor.

        Returns:
            SandboxFusionExecutor instance
        """
        if self.executor is not None:
            logger.warning("SandboxFusion executor already started")
            return self.executor

        # Start Docker container if needed
        if self.use_docker_runner:
            self._start_docker()
            time.sleep(2)  # Wait for container to be ready
            
            # Update sandbox_url if docker runner is using a different port
            if self.docker_runner and self.docker_runner.port:
                from urllib.parse import urlparse, urlunparse
                parsed = urlparse(self.sandbox_url)
                # Only update port if hostname is localhost/127.0.0.1
                if parsed.hostname in ["localhost", "127.0.0.1", "0.0.0.0"]:
                    new_netloc = f"{parsed.hostname}:{self.docker_runner.port}"
                    self.sandbox_url = urlunparse(parsed._replace(netloc=new_netloc))
                    if not self.silent:
                        logger.info(f"Updated SandboxFusion URL to: {self.sandbox_url}")

        # Create executor
        self.executor = SandboxFusionExecutor(
            base_url=self.sandbox_url,
            timeout_s=120,
            default_language="python",
        )

        if not self.silent:
            logger.info(f"SandboxFusion executor started: {self.sandbox_url}")

        return self.executor

    def stop(self) -> None:
        """Stop SandboxFusion service and executor."""
        if self.executor is not None:
            self.executor = None
            if not self.silent:
                logger.info("SandboxFusion executor stopped")

        if self.docker_runner is not None:
            self._stop_docker()

    def _start_docker(self) -> None:
        """Start Docker container running SandboxFusion."""
        if self.docker_runner is not None:
            logger.warning("Docker runner already started")
            return

        if not self.silent:
            logger.info("Starting SandboxFusion Docker container")

        self.docker_runner = DockerAPIRunner(
            docker_image=self.docker_image,
            docker_cmd=self.docker_cmd,
            use_china_mirror=self.use_china_mirror,
            silent=self.silent
        )

        if not self.docker_runner.start():
            raise RuntimeError("Failed to start SandboxFusion Docker container")

        # Wait for container to be ready
        try:
            self.docker_runner.wait_ready(max_wait_time=60)
            if not self.silent:
                logger.info(
                    f"SandboxFusion container ready on port {self.docker_runner.port}"
                )
        except RuntimeError as e:
            self.docker_runner.stop()
            self.docker_runner = None
            raise e

    def _stop_docker(self) -> None:
        """Stop Docker container running SandboxFusion."""
        if self.docker_runner is None:
            return

        if not self.silent:
            logger.info("Stopping SandboxFusion Docker container")

        if not self.docker_runner.stop():
            logger.warning("Failed to stop SandboxFusion Docker container cleanly")

        self.docker_runner = None


class CodeExecutor:
    """Executes code operations in SandboxFusion."""

    def __init__(self, executor: SandboxFusionExecutor) -> None:
        """
        Initialize CodeExecutor.

        Args:
            executor: SandboxFusionExecutor instance
        """
        self.executor = executor

    def clone_repo(self, repo_url: str, target_path: str = "/workspace/repo") -> bool:
        """
        Clone a git repository in the sandbox.

        Args:
            repo_url: URL of the repository
            target_path: Target path in sandbox

        Returns:
            True if successful, False otherwise
        """
        cmd = f"cd /workspace && git clone --depth 1 {repo_url} {target_path.split('/')[-1]}"
        result = self.executor.run_code(cmd, language="bash")

        return result.get("status") == "success" or result.get("return_code") == 0

    def detect_language(self, repo_path: str = "/workspace/repo") -> str:
        """
        Detect programming language of repository.

        Args:
            repo_path: Path to repository in sandbox

        Returns:
            Detected language
        """
        code = f"""
import os
from pathlib import Path

repo = Path("{repo_path}")
file_extensions = {{\
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
}}

file_counts = {{}}
for ext, lang in file_extensions.items():
    count = len(list(repo.rglob(f"*{{ext}}")))
    if count > 0:
        file_counts[lang] = count

if file_counts:
    primary = max(file_counts.items(), key=lambda x: x[1])[0]
    print(primary)
else:
    print("unknown")
"""
        result = self.executor.run_code(code, language="python")
        output = result.get("stdout", "").strip()
        return output.split("\n")[-1] if output else "unknown"

    def extract_dependencies(
        self, repo_path: str = "/workspace/repo", language: str = "python"
    ) -> list[str]:
        """
        Extract dependencies from repository.

        Args:
            repo_path: Path to repository in sandbox
            language: Programming language

        Returns:
            List of dependencies
        """
        if language == "python":
            code = f"""
import re
from pathlib import Path

repo = Path("{repo_path}")
dependencies = []

# Check requirements.txt
req_file = repo / "requirements.txt"
if req_file.exists():
    dependencies.extend([
        line.strip()
        for line in req_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ])

# Check setup.py
setup_py = repo / "setup.py"
if setup_py.exists():
    content = setup_py.read_text()
    match = re.search(r'install_requires\\s*=\\s*\\[(.*?)\\]', content, re.DOTALL)
    if match:
        deps_str = match.group(1)
        deps = re.findall(r'"([^"]+)"|'([^']+)'', deps_str)
        dependencies.extend([d[0] or d[1] for d in deps])

# Check pyproject.toml
pyproject = repo / "pyproject.toml"
if pyproject.exists():
    content = pyproject.read_text()
    match = re.search(r'dependencies\\s*=\\s*\\[(.*?)\\]', content, re.DOTALL)
    if match:
        deps_str = match.group(1)
        deps = re.findall(r'"([^"]+)"|'([^']+)'', deps_str)
        dependencies.extend([d[0] or d[1] for d in deps])

import json
print(json.dumps(list(set(dependencies))))
"""
            result = self.executor.run_code(code, language="python")
            output = result.get("stdout", "").strip()
            try:
                return json.loads(output)
            except (json.JSONDecodeError, IndexError):
                return []

        elif language in ("javascript", "typescript"):
            code = f"""
import json
from pathlib import Path

repo = Path("{repo_path}")
pkg_json = repo / "package.json"
dependencies = []

if pkg_json.exists():
    import json as js
    try:
        pkg_data = js.loads(pkg_json.read_text())
        dependencies.extend(list(pkg_data.get("dependencies", {{}}).keys()))
        dependencies.extend(list(pkg_data.get("devDependencies", {{}}).keys()))
    except:
        pass

print(json.dumps(list(set(dependencies))))
"""
            result = self.executor.run_code(code, language="python")
            output = result.get("stdout", "").strip()
            try:
                return json.loads(output)
            except (json.JSONDecodeError, IndexError):
                return []

        return []

    def install_dependencies(self, dependencies: list[str], language: str = "python") -> bool:
        """
        Install dependencies in sandbox.

        Args:
            dependencies: List of dependencies to install
            language: Programming language

        Returns:
            True if successful, False otherwise
        """
        if language == "python":
            if not dependencies:
                return True
            
            deps_str = " ".join(dependencies[:20])  # Limit to 20
            cmd = f"pip install {deps_str}"
            result = self.executor.run_code(cmd, language="bash")
            return result.get("status") == "success" or result.get("return_code") == 0

        elif language in ("javascript", "typescript"):
            cmd = "npm install"
            result = self.executor.run_code(cmd, language="bash")
            return result.get("status") == "success" or result.get("return_code") == 0

        return False

    def run_command(
        self, command: str, working_dir: str = "/workspace/repo"
    ) -> Dict[str, Any]:
        """
        Run a shell command in sandbox.

        Args:
            command: Command to run
            working_dir: Working directory for command

        Returns:
            Command execution result
        """
        full_cmd = f"cd {working_dir} && {command}"
        return self.executor.run_code(full_cmd, language="bash")

    def run_python_code(self, code: str, working_dir: str = "/workspace/repo") -> Dict[str, Any]:
        """
        Run Python code in sandbox.

        Args:
            code: Python code to execute
            working_dir: Working directory

        Returns:
            Execution result
        """
        setup_code = f"""
import os
os.chdir('{working_dir}')
"""
        full_code = setup_code + "\n" + code
        return self.executor.run_code(full_code, language="python")

    def read_file(self, file_path: str) -> Optional[str]:
        """
        Read a file from sandbox.

        Args:
            file_path: Path to file in sandbox

        Returns:
            File contents or None if error
        """
        code = f"""
from pathlib import Path
try:
    print(Path("{file_path}").read_text())
except Exception as e:
    print(f"Error: {{e}}")
"""
        result = self.executor.run_code(code, language="python")
        output = result.get("stdout", "").strip()
        return output if output and not output.startswith("Error") else None

    def write_file(self, file_path: str, content: str) -> bool:
        """
        Write a file to sandbox.

        Args:
            file_path: Path to file in sandbox
            content: File contents

        Returns:
            True if successful, False otherwise
        """
        # Escape content for Python string
        safe_content = content.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
        
        code = f"""
from pathlib import Path
Path("{file_path}").parent.mkdir(parents=True, exist_ok=True)
Path("{file_path}").write_text('''{content}''')
print("OK")
"""
        result = self.executor.run_code(code, language="python")
        return "OK" in result.get("stdout", "")

    def list_files(self, directory: str) -> list[str]:
        """
        List files in sandbox directory.

        Args:
            directory: Directory path in sandbox

        Returns:
            List of file paths
        """
        code = f"""
from pathlib import Path
import json
directory = Path("{directory}")
if directory.exists():
    files = [str(f.relative_to(directory)) for f in directory.rglob("*") if f.is_file()][:20]
    print(json.dumps(files))
else:
    print(json.dumps([]))
"""
        result = self.executor.run_code(code, language="python")
        output = result.get("stdout", "").strip()
        try:
            return json.loads(output)
        except (json.JSONDecodeError, IndexError):
            return []
    def run_code(self, code: str, language: str = "python", timeout_s: int = 30) -> Dict[str, Any]:
        """
        Run code in sandbox.

        Args:
            code: Code to execute
            language: Programming language (python, javascript, bash, etc.)
            timeout_s: Timeout in seconds

        Returns:
            Execution result with stdout/stderr
        """
        if language == "python":
            return self.run_python_code(code)
        else:
            return self.run_command(code)

    def git_command(self, args: list[str], cwd: str = "/workspace/repo") -> Dict[str, Any]:
        """
        Execute git command in sandbox.
        
        Args:
            args: Git command arguments (e.g., ["add", "file.py"])
            cwd: Working directory for git command
            
        Returns:
            Execution result with stdout/stderr
        """
        cmd = f"cd {cwd} && git " + " ".join(args)
        return self.run_command(cmd)
    
    def apply_code_to_file(self, file_path: str, code: str, repo_path: str = "/workspace/repo") -> bool:
        """
        Apply code changes to a file in the repo.
        
        Args:
            file_path: Relative path to file in repo (e.g., "main.py")
            code: New code content
            repo_path: Base path of the repository in sandbox
            
        Returns:
            True if successful, False otherwise
        """
        full_path = f"{repo_path}/{file_path}"
        
        # Write file using bash to avoid Python string escaping issues
        code_escaped = code.replace("'", "'\"'\"'")  # Escape single quotes for bash
        cmd = f"cat > {full_path} << 'EOF'\n{code}\nEOF"
        
        result = self.run_command(cmd)
        return result.get("exit_code", 1) == 0
    
    def create_git_branch(self, branch_name: str, repo_path: str = "/workspace/repo") -> bool:
        """
        Create and checkout a new git branch.
        
        Args:
            branch_name: Name of the branch to create
            repo_path: Path to the repository
            
        Returns:
            True if successful
        """
        result = self.git_command(["checkout", "-b", branch_name], cwd=repo_path)
        return result.get("exit_code", 1) == 0
    
    def checkout_git_branch(self, branch_name: str, repo_path: str = "/workspace/repo") -> bool:
        """
        Checkout an existing git branch.
        
        Args:
            branch_name: Name of the branch to checkout
            repo_path: Path to the repository
            
        Returns:
            True if successful
        """
        result = self.git_command(["checkout", branch_name], cwd=repo_path)
        return result.get("exit_code", 1) == 0
    
    def git_commit(self, message: str, file_paths: list[str] = None, repo_path: str = "/workspace/repo") -> bool:
        """
        Commit changes to git.
        
        Args:
            message: Commit message
            file_paths: List of files to commit (None = commit all changes)
            repo_path: Path to the repository
            
        Returns:
            True if successful
        """
        # Configure git user if not already configured
        self.git_command(["config", "user.email", "codeagent@example.com"], cwd=repo_path)
        self.git_command(["config", "user.name", "CodeAgent"], cwd=repo_path)
        
        # Add files
        if file_paths:
            for fp in file_paths:
                self.git_command(["add", fp], cwd=repo_path)
        else:
            self.git_command(["add", "-A"], cwd=repo_path)
        
        # Commit
        result = self.git_command(["commit", "-m", f'"{message}"'], cwd=repo_path)
        return result.get("exit_code", 1) == 0
    
    def get_git_diff(self, branch1: str, branch2: str, repo_path: str = "/workspace/repo") -> str:
        """
        Get git diff between two branches.
        
        Args:
            branch1: First branch (e.g., "buggy")
            branch2: Second branch (e.g., "fixed")
            repo_path: Path to the repository
            
        Returns:
            Git diff output
        """
        result = self.git_command(["diff", branch1, branch2], cwd=repo_path)
        return result.get("stdout", "")
    
    def run_tests_in_repo(self, test_command: str = "pytest -v", repo_path: str = "/workspace/repo") -> Dict[str, Any]:
        """
        Run tests in the repository.
        
        Args:
            test_command: Command to run tests
            repo_path: Path to the repository
            
        Returns:
            Test execution result
        """
        cmd = f"cd {repo_path} && {test_command}"
        return self.run_command(cmd)