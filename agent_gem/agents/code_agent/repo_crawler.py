"""
Repository Crawler Module

This module provides functionality to search, filter, and download GitHub repositories
based on various criteria such as programming language, stars, size, lines of code,
update time, topics, and presence of test files.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

logger = logging.getLogger(__name__)


@dataclass
class RepoFilterConfig:
    """Configuration for repository filtering criteria."""
    
    languages: List[str] = field(default_factory=lambda: ["Python"])
    stars_min: int = 100
    stars_max: int = 10000
    size_max: int = 51200  # KB
    line_count_min: int = 500
    line_count_max: int = 50000
    updated_within_days: int = 730
    topics: List[str] = field(default_factory=list)
    exclude_keywords: List[str] = field(default_factory=list)
    exclude_topics: List[str] = field(default_factory=list)
    require_pytest: bool = True
    require_main_package: bool = False  # Require repo to have main package folder
    max_repos: int = 50
    max_pages: int = 10  # Maximum pages to search per query


@dataclass
class RepoOutputConfig:
    """Configuration for repository output settings."""
    
    save_dir: str = "crawled_repos"
    save_metadata: bool = True
    clone_codebase: bool = True
    shallow_clone: bool = True
    save_readme: bool = True


@dataclass
class GitHubConfig:
    """Configuration for GitHub API settings."""
    
    token: Optional[str] = None
    timeout: int = 30
    sleep_between_requests: float = 1.0


class RepoCrawler:
    """
    Repository crawler for searching and downloading GitHub repositories.
    
    This class provides functionality to:
    - Search GitHub repositories based on multiple criteria
    - Filter repositories by language, stars, size, topics, etc.
    - Check for presence of pytest test files
    - Estimate code line count
    - Clone repository codebase
    - Save repository metadata
    
    Example:
        >>> config = RepoCrawler.load_config_from_yaml("config/code_agent.yaml")
        >>> crawler = RepoCrawler(config)
        >>> repos = crawler.search_and_filter_repos()
        >>> crawler.download_repos(repos)
    """
    
    def __init__(
        self,
        filter_config: RepoFilterConfig,
        output_config: RepoOutputConfig,
        github_config: GitHubConfig,
    ):
        """
        Initialize the repository crawler.
        
        Args:
            filter_config: Repository filtering criteria
            output_config: Output settings for saving repositories
            github_config: GitHub API configuration
        """
        self.filter_config = filter_config
        self.output_config = output_config
        self.github_config = github_config
        
        # Set GitHub token from config or environment
        self.github_token = github_config.token or os.environ.get("GITHUB_TOKEN")
        
        # Prepare output directory
        self.output_dir = Path(output_config.save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[RepoCrawler] Initialized with output directory: {self.output_dir}")
        if self.github_token:
            logger.info("[RepoCrawler] Using authenticated GitHub API")
        else:
            logger.warning("[RepoCrawler] No GitHub token found, using unauthenticated API (lower rate limits)")
    
    @classmethod
    def load_config_from_yaml(cls, yaml_path: str | Path) -> "RepoCrawler":
        """
        Load configuration from YAML file and create RepoCrawler instance.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            RepoCrawler instance with loaded configuration
            
        Raises:
            FileNotFoundError: If the YAML file does not exist
            ValueError: If the configuration is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if 'repo_crawler' not in config:
            raise ValueError("Configuration file must contain 'repo_crawler' section")
        
        crawler_config = config['repo_crawler']
        
        # Parse filter config
        filters = crawler_config.get('filters', {})
        filter_config = RepoFilterConfig(
            languages=filters.get('languages', ['Python']),
            stars_min=filters.get('stars', {}).get('min', 100),
            stars_max=filters.get('stars', {}).get('max', 10000),
            size_max=filters.get('size', {}).get('max', 51200),
            line_count_min=filters.get('line_count', {}).get('min', 500),
            line_count_max=filters.get('line_count', {}).get('max', 50000),
            updated_within_days=filters.get('updated_within_days', 730),
            topics=filters.get('topics', []),
            exclude_keywords=filters.get('exclude_keywords', []),
            exclude_topics=filters.get('exclude_topics', []),
            require_pytest=filters.get('require_pytest', True),
            require_main_package=filters.get('require_main_package', False),
            max_repos=filters.get('max_repos', 50),
            max_pages=filters.get('max_pages', 10),
        )
        
        # Parse output config
        output = crawler_config.get('output', {})
        output_config = RepoOutputConfig(
            save_dir=output.get('save_dir', 'crawled_repos'),
            save_metadata=output.get('save_metadata', True),
            clone_codebase=output.get('clone_codebase', True),
            shallow_clone=output.get('shallow_clone', True),
            save_readme=output.get('save_readme', True),
        )
        
        # Parse GitHub config
        github = crawler_config.get('github', {})
        github_config = GitHubConfig(
            token=github.get('token'),
            timeout=github.get('timeout', 30),
            sleep_between_requests=github.get('sleep_between_requests', 1.0),
        )
        
        return cls(filter_config, output_config, github_config)
    
    def _get_headers(self) -> Dict[str, str]:
        """Get HTTP headers for GitHub API requests."""
        headers = {
            'Accept': 'application/vnd.github.v3+json',
        }
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        return headers
    
    def _check_has_main_package(self, repo_full_name: str) -> bool:
        """
        Check if a repository has a main package folder (folder with same name as repo).
        
        Args:
            repo_full_name: Full repository name (owner/repo)
            
        Returns:
            True if main package folder is found, False otherwise
        """
        try:
            # Extract repo name from full_name (e.g., "numpy/numpy" -> "numpy")
            repo_name = repo_full_name.split('/')[-1]
            
            # Get repository contents
            url = f"https://api.github.com/repos/{repo_full_name}/contents"
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.github_config.timeout,
            )
            
            if response.status_code != 200:
                logger.debug(f"[RepoCrawler] Could not check main package for {repo_full_name}: {response.status_code}")
                return False
            
            contents = response.json()
            
            # Look for a directory with the same name as the repo
            for item in contents:
                if isinstance(item, dict):
                    if item.get('type') == 'dir' and item.get('name', '').lower() == repo_name.lower():
                        logger.debug(f"[RepoCrawler] Found main package in {repo_full_name}: {item.get('name')}")
                        return True
            
            logger.debug(f"[RepoCrawler] No main package found in {repo_full_name} (looking for '{repo_name}' folder)")
            return False
            
        except Exception as e:
            logger.debug(f"[RepoCrawler] Error checking main package for {repo_full_name}: {e}")
            return False
    
    def _check_has_pytest(self, repo_full_name: str) -> bool:
        """
        Check if a repository has pytest test files.
        
        Args:
            repo_full_name: Full repository name (owner/repo)
            
        Returns:
            True if pytest tests are found, False otherwise
        """
        try:
            # Search for test files in the repository
            url = f"https://api.github.com/repos/{repo_full_name}/contents"
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.github_config.timeout,
            )
            
            if response.status_code != 200:
                logger.debug(f"[RepoCrawler] Could not check tests for {repo_full_name}: {response.status_code}")
                return False
            
            # Check for common test directories and files
            contents = response.json()
            test_indicators = [
                'tests',
                'test',
                'pytest.ini',
                'conftest.py',
                'tox.ini',
                'setup.cfg',
            ]
            
            for item in contents:
                if isinstance(item, dict):
                    name = item.get('name', '').lower()
                    if any(indicator in name for indicator in test_indicators):
                        logger.debug(f"[RepoCrawler] Found pytest indicator in {repo_full_name}: {name}")
                        return True
            
            # Also check if there are test_*.py or *_test.py files
            for item in contents:
                if isinstance(item, dict):
                    name = item.get('name', '')
                    if name.startswith('test_') and name.endswith('.py'):
                        return True
                    if name.endswith('_test.py'):
                        return True
            
            return False
            
        except Exception as e:
            logger.debug(f"[RepoCrawler] Error checking pytest for {repo_full_name}: {e}")
            return False
    
    def _estimate_line_count(self, repo_full_name: str, language: str) -> Optional[int]:
        """
        Estimate the number of code lines in a repository.
        
        This uses the GitHub API to get language statistics.
        
        Args:
            repo_full_name: Full repository name (owner/repo)
            language: Primary programming language
            
        Returns:
            Estimated line count or None if unavailable
        """
        try:
            url = f"https://api.github.com/repos/{repo_full_name}/languages"
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.github_config.timeout,
            )
            
            if response.status_code != 200:
                return None
            
            languages = response.json()
            # Get bytes for the specified language
            bytes_count = languages.get(language, 0)
            
            # Rough estimation: 1 line ≈ 40-50 bytes (including whitespace and comments)
            # This is a rough approximation
            estimated_lines = bytes_count // 45
            
            return estimated_lines
            
        except Exception as e:
            logger.debug(f"[RepoCrawler] Error estimating line count for {repo_full_name}: {e}")
            return None
    
    def _is_excluded_by_keywords(self, repo_info: Dict[str, Any]) -> bool:
        """
        Check if repository should be excluded based on keywords.
        
        Args:
            repo_info: Repository metadata
            
        Returns:
            True if repository should be excluded
        """
        text_to_check = (
            f"{repo_info.get('name', '')} "
            f"{repo_info.get('description', '')} "
            f"{' '.join(repo_info.get('topics', []))}"
        ).lower()
        
        for keyword in self.filter_config.exclude_keywords:
            if keyword.lower() in text_to_check:
                logger.debug(f"[RepoCrawler] Excluding {repo_info['full_name']} due to keyword: {keyword}")
                return True
        
        repo_topics = [t.lower() for t in repo_info.get('topics', [])]
        for topic in self.filter_config.exclude_topics:
            if topic.lower() in repo_topics:
                logger.debug(f"[RepoCrawler] Excluding {repo_info['full_name']} due to topic: {topic}")
                return True
        
        return False
    
    def search_and_filter_repos(self) -> List[Dict[str, Any]]:
        """
        Search GitHub for repositories matching the filter criteria.
        
        Supports pagination to fetch results across multiple pages.
        For each query (language + topic combination), the crawler will:
        1. Fetch up to 30 results per page
        2. Automatically move to the next page if more results exist
        3. Apply filters to each result (keywords, pytest, line count, etc.)
        4. Continue until reaching max_repos or no more results
        
        Returns:
            List of filtered repository metadata dictionaries
        """
        logger.info("[RepoCrawler] Starting repository search...")
        
        repos = []
        seen_repos = set()
        
        # Calculate date threshold
        days_ago = (datetime.now() - timedelta(days=self.filter_config.updated_within_days)).strftime("%Y-%m-%d")
        
        # Build search queries for each language and topic combination
        queries = []
        for language in self.filter_config.languages:
            if self.filter_config.topics:
                for topic in self.filter_config.topics:
                    query = (
                        f"language:{language} "
                        f"stars:{self.filter_config.stars_min}..{self.filter_config.stars_max} "
                        f"size:<{self.filter_config.size_max} "
                        f"pushed:>{days_ago} "
                        f"topic:{topic}"
                    )
                    queries.append(query)
            else:
                query = (
                    f"language:{language} "
                    f"stars:{self.filter_config.stars_min}..{self.filter_config.stars_max} "
                    f"size:<{self.filter_config.size_max} "
                    f"pushed:>{days_ago}"
                )
                queries.append(query)
        
        logger.info(f"[RepoCrawler] Searching with {len(queries)} query combinations...")
        
        # Execute searches
        for i, query in enumerate(queries):
            if len(repos) >= self.filter_config.max_repos:
                break
            
            logger.info(f"[RepoCrawler] Query {i+1}/{len(queries)}: {query}")
            
            # Pagination: fetch multiple pages for each query
            page = 1
            total_results_this_query = 0
            pages_fetched = 0
            
            while True:
                if len(repos) >= self.filter_config.max_repos:
                    break
                
                # Check if we've reached max pages limit
                if pages_fetched >= self.filter_config.max_pages:
                    logger.info(f"[RepoCrawler] Reached max pages limit ({self.filter_config.max_pages} pages)")
                    break
                
                try:
                    url = "https://api.github.com/search/repositories"
                    params = {
                        'q': query,
                        'sort': 'stars',
                        'order': 'desc',
                        'per_page': 30,  # Max per page for GitHub API
                        'page': page,
                    }
                    
                    response = requests.get(
                        url,
                        params=params,
                        headers=self._get_headers(),
                        timeout=self.github_config.timeout,
                    )
                    
                    # Handle rate limiting
                    if response.status_code == 403:
                        rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
                        rate_limit_reset = response.headers.get('X-RateLimit-Reset', 'unknown')
                        logger.error(f"[RepoCrawler] GitHub API rate limit exceeded")
                        logger.error(f"  Remaining: {rate_limit_remaining}")
                        logger.error(f"  Reset time: {rate_limit_reset}")
                        break
                    
                    # Handle authentication errors
                    if response.status_code == 401:
                        logger.error("[RepoCrawler] GitHub API authentication failed")
                        logger.error("Please check your GITHUB_TOKEN")
                        break
                    
                    if response.status_code != 200:
                        logger.warning(f"[RepoCrawler] GitHub API error: {response.status_code}")
                        break
                    
                    data = response.json()
                    items = data.get('items', [])
                    
                    # If no items, we've reached the end of results
                    if not items:
                        logger.debug(f"[RepoCrawler] No more results for query, total found: {total_results_this_query}")
                        break
                    
                    # Get total count from this query (only on first page)
                    if page == 1:
                        total_results_this_query = data.get('total_count', 0)
                        logger.info(f"[RepoCrawler] Total results available for this query: {total_results_this_query}")
                    
                    # Process each repository on this page
                    for item in items:
                        if len(repos) >= self.filter_config.max_repos:
                            break
                        
                        repo_full_name = item['full_name']
                        
                        # Skip duplicates
                        if repo_full_name in seen_repos:
                            continue
                        
                        # Extract basic info
                        repo_info = {
                            'full_name': repo_full_name,
                            'name': item['name'],
                            'owner': item['owner']['login'],
                            'html_url': item['html_url'],
                            'clone_url': item['clone_url'],
                            'description': item.get('description', ''),
                            'stars': item['stargazers_count'],
                            'size': item['size'],
                            'language': item['language'],
                            'updated_at': item['updated_at'],
                            'topics': item.get('topics', []),
                        }
                        
                        # Apply exclusion filters
                        if self._is_excluded_by_keywords(repo_info):
                            continue
                        
                        # Check for pytest tests if required
                        if self.filter_config.require_pytest:
                            has_pytest = self._check_has_pytest(repo_full_name)
                            repo_info['has_pytest'] = has_pytest
                            if not has_pytest:
                                logger.debug(f"[RepoCrawler] Skipping {repo_full_name}: no pytest tests found")
                                continue
                        else:
                            repo_info['has_pytest'] = self._check_has_pytest(repo_full_name)
                        
                        # Check for main package if required
                        if self.filter_config.require_main_package:
                            has_main_package = self._check_has_main_package(repo_full_name)
                            repo_info['has_main_package'] = has_main_package
                            if not has_main_package:
                                logger.debug(f"[RepoCrawler] Skipping {repo_full_name}: no main package folder found")
                                continue
                        else:
                            repo_info['has_main_package'] = self._check_has_main_package(repo_full_name)
                        
                        # Estimate line count
                        line_count = self._estimate_line_count(repo_full_name, item['language'] or 'Python')
                        repo_info['estimated_lines'] = line_count
                        
                        # Filter by line count if available
                        if line_count is not None:
                            if line_count < self.filter_config.line_count_min:
                                logger.debug(f"[RepoCrawler] Skipping {repo_full_name}: too few lines ({line_count})")
                                continue
                            if line_count > self.filter_config.line_count_max:
                                logger.debug(f"[RepoCrawler] Skipping {repo_full_name}: too many lines ({line_count})")
                                continue
                        
                        # Add to results
                        repos.append(repo_info)
                        seen_repos.add(repo_full_name)
                        
                        logger.info(
                            f"[RepoCrawler] ✓ Added: {repo_full_name} "
                            f"({repo_info['stars']} stars, ~{line_count or '?'} lines)"
                        )
                    
                    # If we got fewer items than per_page, we've reached the end
                    if len(items) < 30:
                        logger.debug(f"[RepoCrawler] Reached last page of results (got {len(items)} items)")
                        break
                    
                    # Move to next page
                    page += 1
                    pages_fetched += 1
                    
                    # Rate limiting: sleep between requests
                    time.sleep(self.github_config.sleep_between_requests)
                    
                except Exception as e:
                    logger.warning(f"[RepoCrawler] Error during search on page {page}: {e}")
                    break
        
        logger.info(f"[RepoCrawler] Search complete. Found {len(repos)} repositories.")
        return repos
    
    def download_repos(self, repos: List[Dict[str, Any]]) -> None:
        """
        Download repositories and save metadata.
        
        Args:
            repos: List of repository metadata dictionaries
        """
        logger.info(f"[RepoCrawler] Starting download of {len(repos)} repositories...")
        
        for i, repo in enumerate(repos):
            logger.info(f"[RepoCrawler] [{i+1}/{len(repos)}] Processing {repo['full_name']}...")
            
            # Create repo-specific directory
            repo_dir = self.output_dir / repo['owner'] / repo['name']
            repo_dir.mkdir(parents=True, exist_ok=True)
            
            # Save metadata
            if self.output_config.save_metadata:
                metadata_path = repo_dir / "metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(repo, f, indent=2)
                logger.info(f"  ✓ Saved metadata to {metadata_path}")
            
            # Clone repository
            if self.output_config.clone_codebase:
                code_dir = repo_dir / "repo"
                if code_dir.exists():
                    logger.info(f"  ⊙ Code already exists at {code_dir}")
                else:
                    success = self._clone_repo(repo['clone_url'], code_dir)
                    if success:
                        logger.info(f"  ✓ Cloned code to {code_dir}")
                    else:
                        logger.warning(f"  ✗ Failed to clone {repo['full_name']}")
            
            # Save README
            if self.output_config.save_readme:
                readme_path = repo_dir / "README.md"
                success = self._download_readme(repo['full_name'], readme_path)
                if success:
                    logger.info(f"  ✓ Saved README to {readme_path}")
        
        logger.info(f"[RepoCrawler] Download complete. Repositories saved to {self.output_dir}")
        
        # Save summary
        summary_path = self.output_dir / "summary.json"
        summary = {
            'total_repos': len(repos),
            'crawled_at': datetime.now().isoformat(),
            'filter_config': {
                'languages': self.filter_config.languages,
                'stars_range': f"{self.filter_config.stars_min}-{self.filter_config.stars_max}",
                'max_size_kb': self.filter_config.size_max,
                'line_count_range': f"{self.filter_config.line_count_min}-{self.filter_config.line_count_max}",
                'updated_within_days': self.filter_config.updated_within_days,
                'topics': self.filter_config.topics,
                'require_pytest': self.filter_config.require_pytest,
            },
            'repositories': [
                {
                    'full_name': r['full_name'],
                    'stars': r['stars'],
                    'language': r['language'],
                    'estimated_lines': r.get('estimated_lines'),
                    'has_pytest': r.get('has_pytest'),
                    'html_url': r['html_url'],
                }
                for r in repos
            ],
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"[RepoCrawler] Saved summary to {summary_path}")
    
    def _clone_repo(self, clone_url: str, target_dir: Path) -> bool:
        """
        Clone a repository using git.
        
        Args:
            clone_url: Git clone URL
            target_dir: Target directory for cloning
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = ['git', 'clone']
            if self.output_config.shallow_clone:
                cmd.extend(['--depth', '1'])
            cmd.extend([clone_url, str(target_dir)])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes timeout
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.warning(f"[RepoCrawler] Error cloning repository: {e}")
            return False
    
    def _download_readme(self, repo_full_name: str, target_path: Path) -> bool:
        """
        Download repository README file.
        
        Args:
            repo_full_name: Full repository name (owner/repo)
            target_path: Target path for README file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to get README via API
            url = f"https://api.github.com/repos/{repo_full_name}/readme"
            response = requests.get(
                url,
                headers=self._get_headers(),
                timeout=self.github_config.timeout,
            )
            
            if response.status_code != 200:
                return False
            
            data = response.json()
            
            # Get the download URL
            download_url = data.get('download_url')
            if not download_url:
                return False
            
            # Download the content
            content_response = requests.get(download_url, timeout=self.github_config.timeout)
            if content_response.status_code != 200:
                return False
            
            # Save to file
            with open(target_path, 'wb') as f:
                f.write(content_response.content)
            
            return True
            
        except Exception as e:
            logger.debug(f"[RepoCrawler] Error downloading README for {repo_full_name}: {e}")
            return False
    
    def crawl(self) -> None:
        """
        Execute the complete crawling workflow:
        1. Search and filter repositories
        2. Download repositories and metadata
        """
        logger.info("[RepoCrawler] Starting crawl workflow...")
        
        # Search and filter
        repos = self.search_and_filter_repos()
        
        if not repos:
            logger.warning("[RepoCrawler] No repositories found matching criteria")
            return
        
        # Download
        self.download_repos(repos)
        
        logger.info("[RepoCrawler] Crawl workflow complete!")


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GitHub Repository Crawler")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/config/code_agent.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    )
    
    # Load configuration and run crawler
    try:
        crawler = RepoCrawler.load_config_from_yaml(args.config)
        crawler.crawl()
    except Exception as e:
        logger.error(f"Error running crawler: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
