"""
PR Extractor Module

This module provides functionality to extract pull request data from GitHub repositories,
and save them in JSONL format.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Dict

from .repo import Repo, extract_patches, extract_problem_statement_and_hints

import yaml
from fastcore.xtras import obj2dict

logger = logging.getLogger(__name__)


@dataclass
class ExtractionStats:
    """Statistics from PR extraction and task generation."""
    
    total_repos: int = 0
    total_prs_extracted: int = 0
    total_task_instances: int = 0
    total_task_instances_with_tests: int = 0
    pr_files: List[Path] = field(default_factory=list)
    task_files: List[Path] = field(default_factory=list)  # Files with tests only
    task_files_all: List[Path] = field(default_factory=list)  # Files with all tasks (including no tests)
    repo_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)  # repo_name -> {prs, tasks, tasks_with_tests}
    
    def __str__(self) -> str:
        """Format statistics as a readable string."""
        lines = [
            "=" * 80,
            "📊 Extraction Statistics",
            "=" * 80,
            f"Repositories processed: {self.total_repos}",
            f"Total PRs extracted: {self.total_prs_extracted}",
            f"Total task instances: {self.total_task_instances}",
            f"Task instances with tests: {self.total_task_instances_with_tests}",
            "",
            "Per Repository:",
        ]
        
        for repo_name, stats in self.repo_stats.items():
            lines.append(f"  {repo_name}:")
            lines.append(f"    - PRs: {stats.get('prs', 0)}")
            lines.append(f"    - Tasks: {stats.get('tasks', 0)}")
            lines.append(f"    - Tasks with tests: {stats.get('tasks_with_tests', 0)}")
        
        lines.extend([
            "",
            "Output Files:",
            f"  PR files: {len(self.pr_files)}",
            f"  Task files (with tests only): {len(self.task_files)}",
            f"  Task files (all, including no tests): {len(self.task_files_all)}",
            "",
            "Note:",
            "  - Task files with '.jsonl' extension contain only tasks with tests",
            "  - Task files with '.all.jsonl' extension contain all valid tasks (including those without tests)",
            "=" * 80,
        ])
        
        return "\n".join(lines)


@dataclass
class PRExtractorConfig:
    """Configuration for PR extraction settings."""
    
    # Repository settings
    repos: List[str] = field(default_factory=list)
    # Optional path to RepoCrawler summary.json; when provided and
    # 'repos' is empty, the repo list will be populated from this file.
    repos_from_repo_crawler_summary: Optional[str] = None
    
    # Output settings
    output_dir: str = "taskdb/code_agent/prs"
    overwrite_existing: bool = False
    
    # Task instance generation settings
    generate_tasks: bool = True  # Whether to generate task instances from PRs
    task_output_dir: Optional[str] = None  # Directory to save task instances (defaults to output_dir/../tasks)
    
    # PR filtering settings
    max_pulls: Optional[int] = None  # Maximum number of PRs to extract per repo
    cutoff_date: Optional[str] = None  # Cutoff date in format YYYYMMDD
    
    # GitHub API settings
    token: Optional[str] = None  # GitHub token (reads from GITHUB_TOKEN env var if not set)
    tokens: Optional[List[str]] = None  # Multiple tokens for parallelization (reads from GITHUB_TOKENS env var if not set)
    
    # PR query settings
    state: str = "closed"  # PR state: "open", "closed", or "all"
    sort: str = "created"  # Sort field: "created", "updated", "popularity"
    direction: str = "desc"  # Sort direction: "asc" or "desc"
    
    # Parallelization settings
    use_parallel: bool = False  # Whether to use parallel processing with multiple tokens


class PRExtractor:
    """
    Pull Request Extractor for GitHub repositories.
    
    This class extracts pull request data from given repositories and saves them
    in JSONL format.
    
    Example:
        >>> config = PRExtractorConfig(repos=["scikit-learn/scikit-learn"])
        >>> extractor = PRExtractor(config)
        >>> extractor.extract_all()
        
        # Or load from YAML
        >>> extractor = PRExtractor.load_config_from_yaml("config/code_agent.yaml")
        >>> extractor.extract_all()
    """
    
    def __init__(self, config: PRExtractorConfig):
        """
        Initialize the PR extractor.
        
        Args:
            config: PR extraction configuration
        """
        self.config = config
        
        # Set GitHub token from config or environment
        self.github_token = config.token or os.environ.get("GITHUB_TOKEN")
        
        # Get multiple tokens for parallelization if available
        if config.tokens:
            self.github_tokens = config.tokens
        else:
            tokens_env = os.environ.get("GITHUB_TOKENS")
            if tokens_env:
                self.github_tokens = [t.strip() for t in tokens_env.split(",")]
            else:
                self.github_tokens = [self.github_token] if self.github_token else []
        
        # Prepare output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare task output directory
        if config.task_output_dir:
            self.task_output_dir = Path(config.task_output_dir)
        else:
            # Default to ../tasks relative to prs directory
            self.task_output_dir = self.output_dir.parent / "tasks"
        self.task_output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[PRExtractor] Initialized with output directory: {self.output_dir}")
        if config.generate_tasks:
            logger.info(f"[PRExtractor] Task instances will be saved to: {self.task_output_dir}")
        logger.info(f"[PRExtractor] Will extract PRs from {len(config.repos)} repositories")
        if self.github_token or self.github_tokens:
            logger.info("[PRExtractor] Using authenticated GitHub API")
        else:
            logger.warning("[PRExtractor] No GitHub token found, using unauthenticated API (lower rate limits)")
    
    @classmethod
    def load_config_from_yaml(cls, yaml_path: str | Path) -> "PRExtractor":
        """
        Load configuration from YAML file and create PRExtractor instance.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            PRExtractor instance with loaded configuration
            
        Raises:
            FileNotFoundError: If the YAML file does not exist
            ValueError: If the configuration is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if 'pr_extractor' not in config_data:
            logger.warning("No 'pr_extractor' section in config, using defaults")
            return cls(PRExtractorConfig())
        
        pr_config = config_data['pr_extractor']
        
        # Parse repository list
        repos = pr_config.get('repos', [])
        if isinstance(repos, str):
            # Handle comma-separated string
            repos = [r.strip() for r in repos.split(',')]

        # Optionally populate repos from RepoCrawler summary.json
        repos_from_summary = pr_config.get('repos_from_repo_crawler_summary')
        if (not repos) and repos_from_summary:
            summary_path = Path(repos_from_summary)
            try:
                if summary_path.is_dir():
                    summary_path = summary_path / "summary.json"
                if summary_path.exists():
                    with summary_path.open("r", encoding="utf-8") as f:
                        summary_data = json.load(f)
                    repo_entries = summary_data.get("repositories", [])
                    repos = [
                        entry["full_name"]
                        for entry in repo_entries
                        if isinstance(entry, dict) and "full_name" in entry
                    ]
                    logger.info(
                        "[PRExtractor] Loaded %d repositories from RepoCrawler summary: %s",
                        len(repos),
                        summary_path,
                    )
                else:
                    logger.warning(
                        "[PRExtractor] repos_from_repo_crawler_summary path does not exist: %s",
                        summary_path,
                    )
            except Exception as e:
                logger.error(
                    "[PRExtractor] Failed to load repos from RepoCrawler summary %s: %s",
                    summary_path,
                    e,
                )
        
        # Parse output settings
        output = pr_config.get('output', {})
        output_dir = output.get('save_dir', 'taskdb/code_agent/prs')
        overwrite_existing = output.get('overwrite_existing', False)
        
        # Parse PR filtering settings
        max_pulls = pr_config.get('max_pulls')
        cutoff_date = pr_config.get('cutoff_date')
        
        # Parse GitHub API settings
        github = pr_config.get('github', {})
        token = github.get('token')
        tokens = github.get('tokens')
        if tokens and isinstance(tokens, str):
            tokens = [t.strip() for t in tokens.split(',')]
        
        # Parse PR query settings
        state = pr_config.get('state', 'closed')
        sort = pr_config.get('sort', 'created')
        direction = pr_config.get('direction', 'desc')
        
        # Parse parallelization settings
        use_parallel = pr_config.get('use_parallel', False)
        
        # Parse task generation settings
        generate_tasks = pr_config.get('generate_tasks', True)
        task_output_dir = pr_config.get('task_output_dir')
        
        config = PRExtractorConfig(
            repos=repos,
            repos_from_repo_crawler_summary=repos_from_summary,
            output_dir=output_dir,
            overwrite_existing=overwrite_existing,
            max_pulls=max_pulls,
            cutoff_date=cutoff_date,
            token=token,
            tokens=tokens,
            state=state,
            sort=sort,
            direction=direction,
            use_parallel=use_parallel,
            generate_tasks=generate_tasks,
            task_output_dir=task_output_dir,
        )
        
        return cls(config)
    
    def extract_prs_from_repo(
        self,
        repo_name: str,
        output_path: Optional[Path] = None,
        token: Optional[str] = None,
    ) -> tuple[Path, int]:
        """
        Extract all PRs from a single repository.
        
        Args:
            repo_name: Repository name in format "owner/repo"
            output_path: Optional output file path (defaults to output_dir/{repo}-prs.jsonl)
            token: Optional GitHub token (uses instance token if not provided)
            
        Returns:
            Path to the output file
            
        Raises:
            ValueError: If repo_name is not in correct format
        """
        if '/' not in repo_name:
            raise ValueError(f"Repository name must be in format 'owner/repo', got: {repo_name}")
        
        owner, repo = repo_name.split("/", 1)
        repo = repo.strip()
        
        # Determine output path
        if output_path is None:
            repo_slug = repo_name.replace("/", "-")
            filename = f"{repo_slug}-prs.jsonl"
            if self.config.cutoff_date:
                filename = filename.replace(".jsonl", f"-{self.config.cutoff_date}.jsonl")
            output_path = self.output_dir / filename
        
        # Check if file already exists
        if output_path.exists() and not self.config.overwrite_existing:
            logger.info(f"📁 PR data for {repo_name} already exists at {output_path}, skipping...")
            # Count existing PRs
            pr_count = sum(1 for _ in open(output_path))
            return output_path, pr_count
        
        # Use provided token or instance token
        use_token = token or self.github_token
        
        logger.info(f"Extracting PRs from {repo_name}...")
        logger.info(f"Will save to {output_path}")
        
        # Create Repo object
        repo_obj = Repo(owner, repo, token=use_token)
        
        # Convert cutoff_date to datetime format if provided
        cutoff_datetime = None
        if self.config.cutoff_date:
            cutoff_datetime = datetime.strptime(
                self.config.cutoff_date, "%Y%m%d"
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
        
        # Extract PRs
        pr_count = 0
        with open(output_path, "w") as f:
            for i_pull, pull in enumerate(repo_obj.get_all_pulls(
                state=self.config.state,
                sort=self.config.sort,
                direction=self.config.direction,
            )):
                # Extract resolved issues
                setattr(pull, "resolved_issues", repo_obj.extract_resolved_issues(pull))
                
                # Write PR to file
                print(json.dumps(obj2dict(pull)), end="\n", flush=True, file=f)
                pr_count += 1
                
                # Check max_pulls limit
                if self.config.max_pulls is not None and pr_count >= self.config.max_pulls:
                    logger.info(f"Reached max_pulls limit ({self.config.max_pulls}) for {repo_name}")
                    break
                
                # Check cutoff_date
                if cutoff_datetime and pull.created_at < cutoff_datetime:
                    logger.info(f"Reached cutoff_date ({self.config.cutoff_date}) for {repo_name}")
                    break
        
        logger.info(f"✅ Successfully extracted {pr_count} PRs from {repo_name} to {output_path}")
        return output_path, pr_count
    
    def build_task_instances_from_pr_file(
        self,
        pr_file: Path,
        task_output_path: Optional[Path] = None,
        token: Optional[str] = None,
    ) -> tuple[Path, Path, int, int]:
        """
        Create task instances from a PR JSONL file.
        
        Args:
            pr_file: Path to PR JSONL file
            task_output_path: Optional output file path (defaults to task_output_dir/{repo}-task-instances.jsonl)
            token: Optional GitHub token (uses instance token if not provided)
            
        Returns:
            Path to the task instances output file
        """
        if not pr_file.exists():
            raise FileNotFoundError(f"PR file not found: {pr_file}")
        
        # Determine output path
        if task_output_path is None:
            # Extract repo name from PR filename (e.g., "pandas-prs.jsonl" -> "pandas")
            repo_name = pr_file.stem.replace("-prs", "").replace(".jsonl", "")
            if self.config.cutoff_date:
                repo_name = repo_name.replace(f"-{self.config.cutoff_date}", "")
            filename = f"{repo_name}-task-instances.jsonl"
            task_output_path = self.task_output_dir / filename
        
        # Check if file already exists
        all_output = task_output_path.parent / f"{task_output_path.stem}.all.jsonl"
        if all_output.exists() and not self.config.overwrite_existing:
            logger.info(f"📁 Task instances for {pr_file.name} already exist at {all_output}, skipping...")
            # Count existing tasks
            total_tasks = sum(1 for _ in open(all_output))
            tasks_with_tests = sum(1 for _ in open(task_output_path)) if task_output_path.exists() else 0
            return task_output_path, all_output, total_tasks, tasks_with_tests
        
        # Use provided token or instance token
        use_token = token or self.github_token
        
        logger.info(f"Building task instances from {pr_file.name}...")
        logger.info(f"Will save to {task_output_path}")
        
        def load_repo(repo_name: str) -> Repo:
            """Return repo object for a given repo name."""
            owner, repo = repo_name.split("/")
            return Repo(owner, repo, token=use_token)
        
        repos = {}
        completed = 0
        with_tests = 0
        total_instances = 0
        seen_prs = set()
        
        # Continue where we left off if output file already exists
        if all_output.exists():
            with open(all_output) as f:
                for line in f:
                    pr = json.loads(line)
                    if "instance_id" not in pr:
                        pr["instance_id"] = (
                            pr["repo"] + "-" + str(pr["pull_number"])
                        ).replace("/", "__")
                    instance_id = pr["instance_id"]
                    seen_prs.add(instance_id)
                    if self._is_valid_instance(pr):
                        completed += 1
                        if self._has_test_patch(pr):
                            with_tests += 1
        
        logger.info(f"Will skip {len(seen_prs)} pull requests that have already been inspected")
        
        # Write to .all file for all PRs
        write_mode_all = "w" if not all_output.exists() else "a"
        with open(all_output, write_mode_all) as all_output_file:
            # Write to output file for PRs with test suites
            write_mode = "w" if not task_output_path.exists() else "a"
            with open(task_output_path, write_mode) as output_file:
                for ix, line in enumerate(open(pr_file)):
                    total_instances += 1
                    pull = json.loads(line)
                    
                    if ix % 100 == 0:
                        repo_full_name = pull.get("base", {}).get("repo", {}).get("full_name", "unknown")
                        logger.info(
                            f"[{repo_full_name}] (Up to {ix} checked) "
                            f"{completed} valid, {with_tests} with tests."
                        )
                    
                    # Construct instance fields
                    repo_full_name = pull.get("base", {}).get("repo", {}).get("full_name")
                    if not repo_full_name:
                        # Fallback: try to get from pull directly
                        repo_full_name = pull.get("head", {}).get("repo", {}).get("full_name")
                    if not repo_full_name:
                        logger.warning(f"Could not determine repo name from PR {pull.get('number')}, skipping")
                        continue
                    
                    instance_id = (repo_full_name + "-" + str(pull["number"])).replace("/", "__")
                    if instance_id in seen_prs:
                        seen_prs -= {instance_id}
                        continue
                    
                    if not self._is_valid_pull(pull):
                        # Throw out invalid PRs
                        continue
                    
                    # Create task instance
                    if repo_full_name not in repos:
                        repos[repo_full_name] = load_repo(repo_full_name)
                    repo = repos[repo_full_name]
                    
                    instance = self._create_instance(repo, pull)
                    if self._is_valid_instance(instance):
                        # If valid, write to .all output file
                        print(json.dumps(instance), end="\n", flush=True, file=all_output_file)
                        completed += 1
                        if self._has_test_patch(instance):
                            # If has test suite, write to output file
                            print(json.dumps(instance), end="\n", flush=True, file=output_file)
                            with_tests += 1
        
        logger.info(
            f"[{', '.join(repos.keys())}] Total instances: {total_instances}, completed: {completed}, with tests: {with_tests}"
        )
        logger.info(f"✅ Successfully generated task instances:")
        logger.info(f"   - With tests only: {task_output_path}")
        logger.info(f"   - All tasks (including no tests): {all_output}")
        return task_output_path, all_output, completed, with_tests
    
    def _create_instance(self, repo: Repo, pull: dict) -> dict:
        """
        Create a single task instance from a pull request.
        
        Args:
            repo: Repo object
            pull: PR dictionary object from GitHub
            
        Returns:
            Task instance dictionary
        """
        patch, test_patch = extract_patches(pull, repo)
        problem_statement, hints = extract_problem_statement_and_hints(pull, repo)
        
        repo_full_name = pull.get("base", {}).get("repo", {}).get("full_name")
        if not repo_full_name:
            repo_full_name = repo.repo.full_name
        
        return {
            "repo": repo_full_name,
            "pull_number": pull["number"],
            "instance_id": (repo_full_name + "-" + str(pull["number"])).replace("/", "__"),
            "issue_numbers": pull.get("resolved_issues", []),
            "base_commit": pull.get("base", {}).get("sha", ""),
            "patch": patch,
            "test_patch": test_patch,
            "problem_statement": problem_statement,
            "hints_text": hints,
            "created_at": pull.get("created_at", ""),
        }
    
    def _is_valid_pull(self, pull: dict) -> bool:
        """
        Check whether PR has an associated issue and is merged.
        
        Args:
            pull: PR dictionary object
            
        Returns:
            bool: whether PR is valid
        """
        if pull.get("merged_at") is None:
            return False
        if "resolved_issues" not in pull or len(pull["resolved_issues"]) < 1:
            return False
        return True
    
    def _is_valid_instance(self, instance: dict) -> bool:
        """
        Check whether task instance has all required fields.
        
        Args:
            instance: Task instance dictionary
            
        Returns:
            bool: whether task instance is valid
        """
        if instance.get("patch") is None or instance.get("patch") == "":
            return False
        if instance.get("problem_statement") is None or instance.get("problem_statement") == "":
            return False
        return True
    
    def _has_test_patch(self, instance: dict) -> bool:
        """
        Check whether task instance has a test suite.
        
        Args:
            instance: Task instance dictionary
            
        Returns:
            bool: whether task instance has a test suite
        """
        test_patch = instance.get("test_patch")
        if test_patch is None or test_patch.strip() == "":
            return False
        return True
    
    def extract_all(self) -> ExtractionStats:
        """
        Extract PRs from all configured repositories and optionally generate task instances.
        
        If use_parallel is True and multiple tokens are available, this will
        parallelize the extraction across tokens.
        
        Returns:
            ExtractionStats object with extraction statistics
        """
        stats = ExtractionStats()
        
        if not self.config.repos:
            logger.warning("No repositories configured for extraction")
            return stats
        
        stats.total_repos = len(self.config.repos)
        
        # If parallel extraction is enabled and multiple tokens are available,
        # keep the existing two-phase behavior (extract first, then build).
        if self.config.use_parallel and len(self.github_tokens) > 1:
            pr_results = self._extract_all_parallel()
            
            # Collect PR statistics
            for repo_name, (pr_path, pr_count) in pr_results.items():
                stats.total_prs_extracted += pr_count
                stats.pr_files.append(pr_path)
                stats.repo_stats[repo_name] = {
                    "prs": pr_count,
                    "tasks": 0,
                    "tasks_with_tests": 0,
                }
            
            # Generate task instances if enabled
            if self.config.generate_tasks:
                logger.info("Generating task instances from extracted PRs...")
                for repo_name, (pr_path, _) in pr_results.items():
                    try:
                        (
                            task_path,
                            task_path_all,
                            total_tasks,
                            tasks_with_tests,
                        ) = self.build_task_instances_from_pr_file(pr_path)
                        stats.total_task_instances += total_tasks
                        stats.total_task_instances_with_tests += tasks_with_tests
                        stats.task_files.append(task_path)  # Files with tests only
                        stats.task_files_all.append(
                            task_path_all
                        )  # Files with all tasks
                        
                        # Update repo stats
                        if repo_name in stats.repo_stats:
                            stats.repo_stats[repo_name]["tasks"] = total_tasks
                            stats.repo_stats[repo_name]["tasks_with_tests"] = (
                                tasks_with_tests
                            )
                    except Exception as e:
                        logger.error(
                            "Error generating task instances from %s: %s",
                            pr_path,
                            e,
                            exc_info=True,
                        )
                        continue
        else:
            # Sequential mode: after finishing PR extraction for each repo,
            # immediately build its task instances (if enabled).
            logger.info(
                "[PRExtractor] Running in sequential mode: extracting PRs and "
                "building task instances per repository."
            )
            
            token = self.github_token or (
                self.github_tokens[0] if self.github_tokens else None
            )
            
            for repo_name in self.config.repos:
                try:
                    repo_name = repo_name.strip().strip(",").strip()
                    pr_path, pr_count = self.extract_prs_from_repo(
                        repo_name, token=token
                    )
                    
                    # Update PR statistics
                    stats.total_prs_extracted += pr_count
                    stats.pr_files.append(pr_path)
                    stats.repo_stats[repo_name] = {
                        "prs": pr_count,
                        "tasks": 0,
                        "tasks_with_tests": 0,
                    }
                    
                    # Immediately build task instances for this repo if enabled
                    if self.config.generate_tasks:
                        (
                            task_path,
                            task_path_all,
                            total_tasks,
                            tasks_with_tests,
                        ) = self.build_task_instances_from_pr_file(pr_path)
                        
                        stats.total_task_instances += total_tasks
                        stats.total_task_instances_with_tests += tasks_with_tests
                        stats.task_files.append(task_path)
                        stats.task_files_all.append(task_path_all)
                        
                        stats.repo_stats[repo_name]["tasks"] = total_tasks
                        stats.repo_stats[repo_name]["tasks_with_tests"] = (
                            tasks_with_tests
                        )
                except Exception as e:
                    logger.error(
                        "Error extracting PRs or generating tasks from %s: %s",
                        repo_name,
                        e,
                        exc_info=True,
                    )
                    continue
        
        return stats
    
    def _extract_all_sequential(self) -> Dict[str, tuple[Path, int]]:
        """Extract PRs sequentially from all repositories."""
        results = {}
        token = self.github_token or (self.github_tokens[0] if self.github_tokens else None)
        
        for repo_name in self.config.repos:
            try:
                repo_name = repo_name.strip().strip(",").strip()
                output_path, pr_count = self.extract_prs_from_repo(repo_name, token=token)
                results[repo_name] = (output_path, pr_count)
            except Exception as e:
                logger.error(f"Error extracting PRs from {repo_name}: {e}", exc_info=True)
                continue
        
        return results
    
    def _extract_all_parallel(self) -> Dict[str, tuple[Path, int]]:
        """Extract PRs in parallel using multiple tokens."""
        from multiprocessing import Pool
        
        def split_repos(repos: List[str], n: int) -> List[List[str]]:
            """Split repository list into n approximately equal sublists."""
            avg_length = len(repos) // n
            remainder = len(repos) % n
            result, start = [], 0
            
            for i in range(n):
                length = avg_length + 1 if i < remainder else avg_length
                sublist = repos[start : start + length]
                result.append(sublist)
                start += length
            
            return result
        
        # Split repos across tokens
        repo_lists = split_repos(self.config.repos, len(self.github_tokens))
        
        # Create data for parallel processing
        data_pooled = [
            {
                "repos": repo_list,
                "extractor_config": self.config,
                "output_dir": str(self.output_dir),
                "token": token,
            }
            for repo_list, token in zip(repo_lists, self.github_tokens)
        ]
        
        # Process in parallel
        with Pool(len(self.github_tokens)) as p:
            worker_results = p.map(_extract_repos_worker, data_pooled)
        
        # Flatten results
        results = {}
        for worker_result in worker_results:
            results.update(worker_result)
        
        return results


def _extract_repos_worker(data: dict) -> Dict[str, tuple[Path, int]]:
    """
    Worker function for parallel PR extraction.
    
    Args:
        data: Dictionary containing repos, config, output_dir, and token
        
    Returns:
        Dictionary mapping repo_name to (output_path, pr_count)
    """
    repos = data["repos"]
    config = data["extractor_config"]
    output_dir = Path(data["output_dir"])
    token = data["token"]
    
    # Create a temporary extractor instance for this worker
    worker_config = PRExtractorConfig(
        repos=repos,
        repos_from_repo_crawler_summary=config.repos_from_repo_crawler_summary,
        output_dir=str(output_dir),
        overwrite_existing=config.overwrite_existing,
        max_pulls=config.max_pulls,
        cutoff_date=config.cutoff_date,
        token=token,
        state=config.state,
        sort=config.sort,
        direction=config.direction,
        use_parallel=False,  # Don't parallelize within worker
        generate_tasks=config.generate_tasks,
        task_output_dir=config.task_output_dir,
    )
    
    extractor = PRExtractor(worker_config)
    return extractor._extract_all_sequential()
