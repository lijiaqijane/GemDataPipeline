"""
Triple Generator for automated task generation.

This module provides TripleGenerator class that generates (repo, file, function) triples
by searching GitHub for suitable repositories and extracting high-quality functions.
"""

from __future__ import annotations

import ast
import json
import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_gem.config import CodeAgentConfig

logger = logging.getLogger(__name__)

class _TripleConfigWrapper:
    """Wrapper to simplify accessing triple_generation config from CodeAgentConfig."""
    
    def __init__(self, code_agent_config: CodeAgentConfig):
        self._config = code_agent_config
        self._triple_config = code_agent_config.config.get('triple_generation', {})
    
    @property
    def cache_dir(self) -> str:
        return self._triple_config.get('output', {}).get('cache_dir', 'triple_cache')
    
    @property
    def target_triples(self) -> int:
        return self._triple_config.get('output', {}).get('target_triples', 100)
    
    @property
    def quality_threshold(self) -> float:
        return self._triple_config.get('output', {}).get('quality_threshold', 0.5)
    
    @property
    def refresh_repos(self) -> bool:
        return self._triple_config.get('output', {}).get('refresh_repos', False)
    
    @property
    def max_repos(self) -> int:
        return self._triple_config.get('repo_filter', {}).get('max_repos', 50)
    
    @property
    def repo_updated_within_days(self) -> int:
        return self._triple_config.get('repo_filter', {}).get('updated_within_days', 730)
    
    @property
    def repo_stars_min(self) -> int:
        return self._triple_config.get('repo_filter', {}).get('stars', {}).get('min', 100)
    
    @property
    def repo_stars_max(self) -> int:
        return self._triple_config.get('repo_filter', {}).get('stars', {}).get('max', 10000)
    
    @property
    def repo_size_max(self) -> int:
        return self._triple_config.get('repo_filter', {}).get('size', {}).get('max', 10240)
    
    @property
    def repo_topics(self) -> List[str]:
        return self._triple_config.get('repo_filter', {}).get('topics', ['library', 'tool', 'cli'])
    
    @property
    def repo_exclude_keywords(self) -> List[str]:
        return self._triple_config.get('repo_filter', {}).get('exclude_keywords', [])
    
    @property
    def repo_exclude_topics(self) -> List[str]:
        return self._triple_config.get('repo_filter', {}).get('exclude_topics', [])
    
    @property
    def max_files_per_repo(self) -> int:
        return self._triple_config.get('file_filter', {}).get('max_files_per_repo', 10)
    
    @property
    def file_exclude_dirs(self) -> List[str]:
        return self._triple_config.get('file_filter', {}).get('exclude_directories', ['tests', 'docs'])
    
    @property
    def file_exclude_files(self) -> List[str]:
        return self._triple_config.get('file_filter', {}).get('exclude_files', ['__init__.py'])
    
    @property
    def focus_main_package(self) -> bool:
        return self._triple_config.get('file_filter', {}).get('focus_main_package', True)
    
    @property
    def file_scoring(self) -> Dict[str, float]:
        return self._triple_config.get('file_filter', {}).get('scoring', {})
    
    @property
    def file_line_count_min(self) -> int:
        return self._triple_config.get('file_filter', {}).get('line_count', {}).get('min', 50)
    
    @property
    def file_line_count_max(self) -> int:
        return self._triple_config.get('file_filter', {}).get('line_count', {}).get('max', 2000)
    
    @property
    def file_line_count_optimal_range(self) -> tuple:
        lc = self._triple_config.get('file_filter', {}).get('line_count', {})
        return (lc.get('optimal_min', 100), lc.get('optimal_max', 500))
    
    @property
    def file_min_functions(self) -> int:
        return self._triple_config.get('file_filter', {}).get('min_functions', 2)
    
    @property
    def file_utility_keywords(self) -> List[str]:
        return self._triple_config.get('file_filter', {}).get('utility_keywords', ['util', 'helper'])
    
    @property
    def function_skip_private(self) -> bool:
        return self._triple_config.get('function_filter', {}).get('skip_private', True)
    
    @property
    def function_min_quality_score(self) -> float:
        return self._triple_config.get('function_filter', {}).get('min_quality_score', 0.3)
    
    @property
    def function_scoring(self) -> Dict[str, float]:
        return self._triple_config.get('function_filter', {}).get('scoring', {})
    
    @property
    def function_line_count_min(self) -> int:
        return self._triple_config.get('function_filter', {}).get('line_count', {}).get('min', 5)
    
    @property
    def function_line_count_max(self) -> int:
        return self._triple_config.get('function_filter', {}).get('line_count', {}).get('max', 200)
    
    @property
    def function_line_count_optimal_range(self) -> tuple:
        lc = self._triple_config.get('function_filter', {}).get('line_count', {})
        return (lc.get('optimal_min', 10), lc.get('optimal_max', 100))
    
    @property
    def function_line_count_good_range(self) -> tuple:
        # Good range is between min and optimal_min, and between optimal_max and max
        opt_min, opt_max = self.function_line_count_optimal_range
        return (self.function_line_count_min, opt_min), (opt_max, self.function_line_count_max)
    
    @property
    def function_max_arguments(self) -> int:
        return self._triple_config.get('function_filter', {}).get('max_arguments', 5)

class TripleGenerator:
    """
    Generate (repo, file, function) triples for automated task generation.
    
    Three-stage filtering process:
    1. Repo filtering: Find high-quality, lightweight Python repos from GitHub
    2. File scoring: Select the best files from each repo
    3. Function extraction: Extract suitable functions from selected files
    
    Features:
    - Caches all intermediate results to avoid redundant work
    - Filters out GPU-heavy and resource-intensive repos
    - Uses scoring algorithms to prioritize high-quality candidates
    - Supports configuration via YAML files
    """
    
    def __init__(
        self, 
        github_token: Optional[str] = None, 
        cache_dir: Optional[str] = None,
        config_path: Optional[str | Path] = None
    ):
        """
        Initialize TripleGenerator.
        
        Args:
            github_token: GitHub personal access token (optional, but recommended for higher rate limits)
            cache_dir: Directory to store cached data (overrides config if provided)
            config_path: Path to YAML configuration file (optional, uses default config if not provided)
        """
        # Load configuration
        if config_path:
            code_agent_config = CodeAgentConfig.from_file(config_path)
            logger.info(f"[TripleGenerator] Loaded configuration from {config_path}")
        else:
            code_agent_config = CodeAgentConfig.default()
            logger.info("[TripleGenerator] Using default configuration")
        
        # Wrap config for easier property access
        self.config = _TripleConfigWrapper(code_agent_config)
        
        # Setup cache directory (command line arg overrides config)
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path(self.config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize GitHub API client
        self.github_token = github_token or os.environ.get("GITHUB_TOKEN")
        if not self.github_token:
            logger.warning("[TripleGenerator] No GitHub token provided. API rate limit will be low (60 requests/hour).")
            logger.warning("[TripleGenerator] Set GITHUB_TOKEN environment variable for higher limits (5000 requests/hour).")
        
        # Cache files
        self.repos_cache_file = self.cache_dir / "repos_metadata.json"
        self.triples_cache_file = self.cache_dir / "function_triples.json"
        
        logger.info(f"[TripleGenerator] Initialized with cache dir: {self.cache_dir}")
    
    def test_github_connection(self) -> bool:
        """
        Test GitHub API connection and authentication.
        
        Returns:
            True if connection is successful, False otherwise
        """
        import requests
        
        logger.info("[TripleGenerator] Testing GitHub API connection...")
        
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
            logger.info("[TripleGenerator] Using provided GitHub token")
        else:
            logger.info("[TripleGenerator] No token provided (unauthenticated mode)")
        
        try:
            # Test with rate limit endpoint (doesn't count against rate limit)
            url = "https://api.github.com/rate_limit"
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                core_limit = data['resources']['core']
                search_limit = data['resources']['search']
                
                logger.info("✅ GitHub API connection successful!")
                logger.info(f"  Core API: {core_limit['remaining']}/{core_limit['limit']} remaining")
                logger.info(f"  Search API: {search_limit['remaining']}/{search_limit['limit']} remaining")
                
                if self.github_token:
                    logger.info("  Authentication: ✅ Token valid")
                else:
                    logger.info("  Authentication: ⚠️  Unauthenticated (low rate limits)")
                    logger.info("  Recommend setting GITHUB_TOKEN for higher limits")
                
                return True
            
            elif response.status_code == 401:
                logger.error("❌ GitHub API authentication failed (401 Unauthorized)")
                logger.error("Your GitHub token is invalid or expired.")
                logger.error("\nTo fix:")
                logger.error("  1. Go to https://github.com/settings/tokens")
                logger.error("  2. Generate new token (classic) with 'public_repo' scope")
                logger.error("  3. Set: export GITHUB_TOKEN=your_token_here")
                return False
            
            elif response.status_code == 403:
                logger.error("❌ GitHub API rate limit exceeded or access forbidden")
                logger.error(f"  Response: {response.text[:200]}")
                return False
            
            else:
                logger.error(f"❌ GitHub API error: {response.status_code}")
                logger.error(f"  Response: {response.text[:200]}")
                return False
        
        except requests.exceptions.Timeout:
            logger.error("❌ GitHub API connection timeout")
            logger.error("  Check your internet connection")
            return False
        
        except requests.exceptions.ConnectionError:
            logger.error("❌ Cannot connect to GitHub API")
            logger.error("  Check your internet connection and proxy settings")
            return False
        
        except Exception as e:
            logger.error(f"❌ Unexpected error testing GitHub connection: {e}")
            return False
    
    def generate_triples(
        self, 
        num_triples: Optional[int] = None, 
        quality_threshold: Optional[float] = None,
        refresh_repos: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """
        Generate (repo, file, function) triples.
        
        Args:
            num_triples: Target number of triples to generate (uses config if None)
            quality_threshold: Minimum quality score (0-1) for triples (uses config if None)
            refresh_repos: If True, fetch fresh repos from GitHub instead of using cache (uses config if None)
            
        Returns:
            List of triple dictionaries with metadata
        """
        # Use config values if not provided
        num_triples = num_triples if num_triples is not None else self.config.target_triples
        quality_threshold = quality_threshold if quality_threshold is not None else self.config.quality_threshold
        refresh_repos = refresh_repos if refresh_repos is not None else self.config.refresh_repos
        
        logger.info(f"[TripleGenerator] Starting triple generation (target: {num_triples}, threshold: {quality_threshold})")
        
        # Stage 1: Get candidate repos
        repos = self._fetch_or_load_repos(limit=self.config.max_repos, refresh=refresh_repos)
        logger.info(f"[TripleGenerator] Stage 1 complete: {len(repos)} candidate repos")
        
        # Stage 2 & 3: Analyze repos and extract triples
        all_triples = []
        for i, repo_info in enumerate(repos):
            try:
                logger.info(f"[TripleGenerator] Processing repo {i+1}/{len(repos)}: {repo_info['full_name']}")
                
                # Check if we already have enough triples
                if len(all_triples) >= num_triples:
                    logger.info(f"[TripleGenerator] Reached target of {num_triples} triples")
                    break
                
                # Extract triples from this repo
                repo_triples = self._extract_repo_triples(repo_info)
                
                # Filter by quality threshold
                high_quality = [t for t in repo_triples if t.get('quality_score', 0) >= quality_threshold]
                
                if high_quality:
                    logger.info(f"[TripleGenerator] Found {len(high_quality)} high-quality triples from {repo_info['full_name']}")
                    all_triples.extend(high_quality)
                else:
                    logger.debug(f"[TripleGenerator] No high-quality triples from {repo_info['full_name']}")
                    
            except Exception as e:
                logger.warning(f"[TripleGenerator] Failed to process {repo_info['full_name']}: {e}")
                continue
        
        # Sort by quality score
        all_triples.sort(key=lambda x: x.get('quality_score', 0), reverse=True)
        
        # Return top N
        result = all_triples[:num_triples]
        
        # Cache the results
        self._save_triples_cache(result)
        
        logger.info(f"[TripleGenerator] Generated {len(result)} triples")
        return result
    
    def _fetch_or_load_repos(self, limit: int = 50, refresh: bool = False) -> List[Dict[str, Any]]:
        """
        Fetch candidate repos from GitHub or load from cache.
        
        Args:
            limit: Maximum number of repos to fetch
            refresh: If True, fetch fresh data instead of using cache
            
        Returns:
            List of repo metadata dictionaries
        """
        # Check cache first (only use if not empty and not refreshing)
        if not refresh and self.repos_cache_file.exists():
            logger.info("[TripleGenerator] Loading repos from cache...")
            try:
                with open(self.repos_cache_file, 'r') as f:
                    repos = json.load(f)
                
                # If cache is empty, fall through to fetch from GitHub
                if not repos:
                    logger.warning("[TripleGenerator] Cache is empty, will fetch from GitHub instead")
                else:
                    logger.info(f"[TripleGenerator] Loaded {len(repos)} repos from cache")
                    return repos
            except Exception as e:
                logger.warning(f"[TripleGenerator] Failed to load cache: {e}")
        
        # Fetch from GitHub
        logger.info("[TripleGenerator] Fetching repos from GitHub...")
        repos = self._search_github_repos(limit=limit)
        
        # Save to cache only if we got results
        if repos:
            try:
                with open(self.repos_cache_file, 'w') as f:
                    json.dump(repos, f, indent=2)
                logger.info(f"[TripleGenerator] Cached {len(repos)} repos")
            except Exception as e:
                logger.warning(f"[TripleGenerator] Failed to save cache: {e}")
        else:
            logger.warning("[TripleGenerator] No repos fetched, cache not updated")
        
        return repos
    
    def _search_github_repos(self, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Search GitHub for suitable Python repositories.
        
        Criteria:
        - Language: Python
        - Stars: 100-10000 (quality but not too complex)
        - Size: < 50MB (avoid large repos)
        - Recently updated (within 2 years)
        - Has tests
        - Excludes heavy ML/DL frameworks
        
        Args:
            limit: Maximum number of repos to return
            
        Returns:
            List of repo metadata dictionaries
        """
        import requests
        from datetime import datetime, timedelta
        
        # Build search query using config
        days_ago = (datetime.now() - timedelta(days=self.config.repo_updated_within_days)).strftime("%Y-%m-%d")
        stars_range = f"{self.config.repo_stars_min}..{self.config.repo_stars_max}"
        size_limit = f"<{self.config.repo_size_max}"
        
        queries = []
        for topic in self.config.repo_topics:
            query = f"language:python stars:{stars_range} size:{size_limit} pushed:>{days_ago} topic:{topic}"
            queries.append(query)
        
        repos = []
        seen_repos = set()
        
        headers = {}
        if self.github_token:
            headers['Authorization'] = f'token {self.github_token}'
        
        for query in queries:
            if len(repos) >= limit:
                break
            
            try:
                # GitHub Search API
                url = "https://api.github.com/search/repositories"
                params = {
                    'q': query,
                    'sort': 'stars',
                    'order': 'desc',
                    'per_page': min(30, limit - len(repos))
                }
                
                response = requests.get(url, params=params, headers=headers, timeout=30)
                
                # Handle authentication errors
                if response.status_code == 401:
                    logger.error("[TripleGenerator] GitHub API authentication failed (401 Unauthorized)")
                    if self.github_token:
                        logger.error("Your GitHub token is invalid or expired.")
                        logger.error("Please check:")
                        logger.error("  1. Token is correct (no extra spaces)")
                        logger.error("  2. Token has not been revoked")
                        logger.error("  3. Token has 'public_repo' or 'repo' scope")
                    else:
                        logger.error("No GitHub token provided, but API returned 401.")
                        logger.error("This is unusual. Try providing a valid token:")
                        logger.error("  export GITHUB_TOKEN=your_personal_access_token")
                    logger.error("\nTo create a new token:")
                    logger.error("  1. Go to https://github.com/settings/tokens")
                    logger.error("  2. Click 'Generate new token (classic)'")
                    logger.error("  3. Select 'public_repo' scope")
                    logger.error("  4. Copy and set: export GITHUB_TOKEN=<your_token>")
                    break
                
                # Handle rate limiting
                if response.status_code == 403:
                    rate_limit_remaining = response.headers.get('X-RateLimit-Remaining', 'unknown')
                    rate_limit_reset = response.headers.get('X-RateLimit-Reset', 'unknown')
                    logger.error(f"[TripleGenerator] GitHub API rate limit exceeded")
                    logger.error(f"  Remaining: {rate_limit_remaining}")
                    logger.error(f"  Reset time: {rate_limit_reset}")
                    if not self.github_token:
                        logger.error("  You're using unauthenticated API (60 requests/hour)")
                        logger.error("  Set GITHUB_TOKEN to get 5000 requests/hour")
                    break
                
                # Handle other errors
                if response.status_code != 200:
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('message', 'Unknown error')
                        logger.warning(f"[TripleGenerator] GitHub API error {response.status_code}: {error_msg}")
                    except:
                        logger.warning(f"[TripleGenerator] GitHub API error: {response.status_code}")
                    continue
                
                data = response.json()
                
                for item in data.get('items', []):
                    repo_full_name = item['full_name']
                    
                    # Skip duplicates
                    if repo_full_name in seen_repos:
                        continue
                    
                    # Filter out heavy dependencies
                    if self._is_heavy_repo(item):
                        logger.debug(f"[TripleGenerator] Skipping heavy repo: {repo_full_name}")
                        continue
                    
                    # Extract metadata
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
                        'has_tests': self._check_has_tests(item),
                    }
                    
                    repos.append(repo_info)
                    seen_repos.add(repo_full_name)
                    
                    logger.debug(f"[TripleGenerator] Added repo: {repo_full_name} ({repo_info['stars']} stars)")
                    
                    if len(repos) >= limit:
                        break
                
                # Rate limiting: sleep between requests
                import time
                time.sleep(1)
                
            except Exception as e:
                logger.warning(f"[TripleGenerator] Error searching GitHub: {e}")
                continue
        
        logger.info(f"[TripleGenerator] Found {len(repos)} candidate repos")
        return repos
    
    def _is_heavy_repo(self, repo_item: Dict[str, Any]) -> bool:
        """
        Check if repo is likely to be resource-heavy.
        
        Args:
            repo_item: GitHub API repo item
            
        Returns:
            True if repo appears to be heavy
        """
        # Check name and description for heavy keywords (from config)
        text = f"{repo_item['name']} {repo_item.get('description', '')}".lower()
        
        if any(keyword in text for keyword in self.config.repo_exclude_keywords):
            return True
        
        # Check topics (from config)
        topics = [t.lower() for t in repo_item.get('topics', [])]
        exclude_topics_set = set(self.config.repo_exclude_topics)
        
        if any(topic in exclude_topics_set for topic in topics):
            return True
        
        return False
    
    def _check_has_tests(self, repo_item: Dict[str, Any]) -> bool:
        """
        Check if repo likely has tests (heuristic based on common patterns).
        
        Args:
            repo_item: GitHub API repo item
            
        Returns:
            True if repo likely has tests
        """
        # This is a heuristic - we can't know for sure without cloning
        # Common indicators: topics, description, name patterns
        topics = [t.lower() for t in repo_item.get('topics', [])]
        test_topics = {'testing', 'pytest', 'unittest', 'test'}
        
        return any(topic in test_topics for topic in topics)
    
    def _extract_repo_triples(self, repo_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extract (file, function) pairs from a repository.
        
        Args:
            repo_info: Repository metadata
            
        Returns:
            List of triple dictionaries
        """
        repo_cache_dir = self.cache_dir / "repos" / repo_info['full_name'].replace('/', '_')
        triples_cache = repo_cache_dir / "triples.json"
        
        # Check cache
        if triples_cache.exists():
            try:
                with open(triples_cache, 'r') as f:
                    triples = json.load(f)
                logger.debug(f"[TripleGenerator] Loaded {len(triples)} triples from cache for {repo_info['full_name']}")
                return triples
            except Exception as e:
                logger.warning(f"[TripleGenerator] Failed to load triples cache: {e}")
        
        # Clone repo (shallow)
        repo_path = self._clone_repo_shallow(repo_info)
        if not repo_path:
            return []
        
        try:
            # Stage 2: Score files
            file_scores = self._score_repo_files(repo_path, repo_info)
            
            # Select top files (from config)
            top_files = sorted(file_scores.items(), key=lambda x: x[1], reverse=True)[:self.config.max_files_per_repo]
            
            logger.debug(f"[TripleGenerator] Selected {len(top_files)} top files from {repo_info['full_name']}")
            
            # Stage 3: Extract functions
            triples = []
            for file_path, file_score in top_files:
                functions = self._extract_file_functions(repo_path, file_path, repo_info)
                triples.extend(functions)
            
            # Save to cache
            repo_cache_dir.mkdir(parents=True, exist_ok=True)
            with open(triples_cache, 'w') as f:
                json.dump(triples, f, indent=2)
            
            logger.debug(f"[TripleGenerator] Extracted {len(triples)} triples from {repo_info['full_name']}")
            
            return triples
            
        finally:
            # Cleanup: remove cloned repo
            self._cleanup_repo(repo_path)
    
    def _clone_repo_shallow(self, repo_info: Dict[str, Any]) -> Optional[Path]:
        """
        Clone a repository (shallow, depth=1) to temporary location.
        
        Args:
            repo_info: Repository metadata
            
        Returns:
            Path to cloned repo or None if failed
        """
        temp_dir = Path(tempfile.mkdtemp(prefix="triple_gen_"))
        repo_path = temp_dir / repo_info['name']
        
        try:
            clone_cmd = [
                'git', 'clone',
                '--depth', '1',
                '--single-branch',
                repo_info['clone_url'],
                str(repo_path)
            ]
            
            result = subprocess.run(
                clone_cmd,
                capture_output=True,
                text=True,
                timeout=120
            )
            
            if result.returncode == 0:
                logger.debug(f"[TripleGenerator] Cloned {repo_info['full_name']} to {repo_path}")
                return repo_path
            else:
                logger.warning(f"[TripleGenerator] Failed to clone {repo_info['full_name']}: {result.stderr}")
                return None
                
        except Exception as e:
            logger.warning(f"[TripleGenerator] Error cloning {repo_info['full_name']}: {e}")
            return None
    
    def _cleanup_repo(self, repo_path: Optional[Path]) -> None:
        """Remove cloned repository."""
        if repo_path and repo_path.exists():
            try:
                import shutil
                shutil.rmtree(repo_path.parent)
                logger.debug(f"[TripleGenerator] Cleaned up {repo_path}")
            except Exception as e:
                logger.warning(f"[TripleGenerator] Failed to cleanup {repo_path}: {e}")
    
    def _identify_main_package(self, repo_path: Path, repo_info: Dict[str, Any]) -> Optional[str]:
        """
        Identify the main package directory in the repository.
        
        For example, in numpy repo, this should return 'numpy'.
        
        Args:
            repo_path: Path to cloned repository
            repo_info: Repository metadata
            
        Returns:
            Main package directory name or None if not found
        """
        # Strategy 1: Look for a directory with __init__.py that matches repo name
        repo_name = repo_info['name'].lower().replace('-', '_').replace('.', '_')
        potential_package = repo_path / repo_name
        if potential_package.is_dir() and (potential_package / '__init__.py').exists():
            logger.debug(f"[TripleGenerator] Found main package: {repo_name} (matches repo name)")
            return repo_name
        
        # Strategy 2: Find all top-level directories with __init__.py
        candidates = []
        try:
            for item in repo_path.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Skip common non-package directories (from config)
                    if item.name in set(self.config.file_exclude_dirs):
                        continue
                    
                    # Check if it's a Python package
                    if (item / '__init__.py').exists():
                        # Count Python files in this directory
                        py_files = list(item.rglob('*.py'))
                        candidates.append((item.name, len(py_files)))
            
            # Choose the package with the most Python files
            if candidates:
                main_package = max(candidates, key=lambda x: x[1])[0]
                logger.debug(f"[TripleGenerator] Found main package: {main_package} ({len(candidates)} candidates)")
                return main_package
        
        except Exception as e:
            logger.warning(f"[TripleGenerator] Error identifying main package: {e}")
        
        logger.debug(f"[TripleGenerator] No main package identified for {repo_info['name']}")
        return None
    
    def _score_repo_files(self, repo_path: Path, repo_info: Dict[str, Any]) -> Dict[str, float]:
        """
        Score all Python files in the repository, focusing on the main package.
        
        Args:
            repo_path: Path to cloned repository
            repo_info: Repository metadata
            
        Returns:
            Dictionary mapping file paths to scores
        """
        file_scores = {}
        
        # Identify the main package directory (if configured)
        if self.config.focus_main_package:
            main_package = self._identify_main_package(repo_path, repo_info)
            
            if main_package:
                logger.info(f"[TripleGenerator] Focusing on main package: {main_package}")
                search_root = repo_path / main_package
            else:
                logger.info(f"[TripleGenerator] No main package found, scanning entire repo")
                search_root = repo_path
        else:
            logger.info(f"[TripleGenerator] Scanning entire repo (focus_main_package=False)")
            search_root = repo_path
        
        # Find all Python files
        try:
            for py_file in search_root.rglob("*.py"):
                # Skip hidden directories and common excludes (from config)
                if any(part.startswith('.') for part in py_file.parts):
                    continue
                exclude_dirs = set(self.config.file_exclude_dirs)
                if any(part in exclude_dirs for part in py_file.parts):
                    continue
                
                try:
                    # Read file content
                    with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                        code = f.read()
                    
                    # Get relative path (relative to repo root, not search root)
                    rel_path = str(py_file.relative_to(repo_path))
                    
                    # Score the file
                    score = self._score_file(rel_path, code)
                    
                    if score > 0:
                        file_scores[rel_path] = score
                        
                except Exception as e:
                    logger.debug(f"[TripleGenerator] Failed to score {py_file}: {e}")
                    continue
        
        except Exception as e:
            logger.warning(f"[TripleGenerator] Error scanning repo files: {e}")
        
        return file_scores
    
    def _score_file(self, file_path: str, code: str) -> float:
        """
        Score a file's suitability for task generation.
        
        Args:
            file_path: Relative file path
            code: File content
            
        Returns:
            Quality score (0-100+)
        """
        score = 0.0
        
        # Get scoring configuration
        scoring = self.config.file_scoring
        
        # Filter out unsuitable files (from config)
        file_name = file_path.split('/')[-1]
        if any(pattern in file_name for pattern in self.config.file_exclude_files):
            return 0.0
        
        # Check file size (from config)
        lines = code.split('\n')
        line_count = len(lines)
        
        # Hard limits from config
        if line_count < self.config.file_line_count_min:
            return 0.0
        elif line_count > self.config.file_line_count_max:
            return 0.0
        
        # Score based on optimal range (from config)
        optimal_min, optimal_max = self.config.file_line_count_optimal_range
        if optimal_min <= line_count <= optimal_max:
            score += scoring['line_count_optimal']
        elif self.config.file_line_count_min <= line_count < optimal_min or \
             optimal_max < line_count <= self.config.file_line_count_max:
            score += scoring['line_count_good']
        else:
            score += scoring['line_count_acceptable']
        
        # Count function definitions
        function_pattern = r'^\s*def\s+\w+\s*\('
        functions = re.findall(function_pattern, code, re.MULTILINE)
        function_count = len(functions)
        
        # Score functions (from config)
        if function_count < self.config.file_min_functions:
            return 0.0  # Not enough functions
        
        score += min(function_count * scoring['function_count'], scoring['function_count_max'])
        
        # Count docstrings (from config)
        docstring_pattern = r'\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\''
        docstrings = re.findall(docstring_pattern, code)
        docstring_count = len(docstrings)
        score += min(docstring_count * scoring['docstring'], scoring['docstring_max'])
        
        # Check for test coverage indicators (from config)
        if 'tests/' not in file_path and 'test_' not in file_name:
            score += scoring['not_test']
        
        # Prefer files in main package (from config)
        path_parts = file_path.split('/')
        if len(path_parts) <= 2:
            score += scoring['main_package']
        
        # Penalize utility/helper files (from config)
        if any(keyword in file_path.lower() for keyword in self.config.file_utility_keywords):
            score += scoring['utility_penalty']  # Note: this should be negative in config
        
        # Check code complexity (from config)
        has_classes = bool(re.search(r'^\s*class\s+\w+', code, re.MULTILINE))
        has_decorators = bool(re.search(r'^\s*@\w+', code, re.MULTILINE))
        has_imports = bool(re.search(r'^\s*(?:from|import)\s+', code, re.MULTILINE))
        
        complexity_score = sum([has_classes * 5, has_decorators * 3, has_imports * 2])
        score += min(complexity_score, scoring['complexity'])
        
        return score
    
    def _extract_file_functions(
        self, 
        repo_path: Path, 
        file_path: str, 
        repo_info: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Extract and score functions from a file.
        
        Args:
            repo_path: Path to repository
            file_path: Relative path to file
            repo_info: Repository metadata
            
        Returns:
            List of function triples with metadata
        """
        triples = []
        
        try:
            # Read file
            full_path = repo_path / file_path
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                code = f.read()
            
            # Parse with AST
            tree = ast.parse(code)
            
            # Extract functions
            for node in ast.walk(tree):
                if not isinstance(node, ast.FunctionDef):
                    continue
                
                # Skip private functions (from config)
                if self.config.function_skip_private and node.name.startswith('_'):
                    continue
                
                # Score the function
                func_score = self._score_function(node, code)
                
                # Skip low-quality functions (from config)
                if func_score < self.config.function_min_quality_score:
                    continue
                
                # Extract metadata
                func_info = {
                    'repo_url': repo_info['clone_url'].replace('.git', ''),
                    'repo_name': repo_info['full_name'],
                    'repo_stars': repo_info['stars'],
                    'file_path': file_path,
                    'function_name': node.name,
                    'line_start': node.lineno,
                    'line_end': node.end_lineno,
                    'quality_score': func_score,
                    'has_docstring': ast.get_docstring(node) is not None,
                    'has_type_hints': self._has_type_hints(node),
                    'line_count': node.end_lineno - node.lineno + 1,
                }
                
                triples.append(func_info)
        
        except SyntaxError as e:
            logger.debug(f"[TripleGenerator] Syntax error in {file_path}: {e}")
        except Exception as e:
            logger.debug(f"[TripleGenerator] Error extracting functions from {file_path}: {e}")
        
        return triples
    
    def _score_function(self, func_node: ast.FunctionDef, full_code: str) -> float:
        """
        Score a function's suitability for task generation.
        
        Args:
            func_node: AST FunctionDef node
            full_code: Full file content
            
        Returns:
            Quality score (0-1)
        """
        score = 0.0
        
        # Line count (prefer medium-sized functions) - from config
        line_count = func_node.end_lineno - func_node.lineno + 1
        scoring = self.config.function_scoring
        
        if line_count < self.config.function_line_count_min:
            return 0.0  # Too trivial
        elif line_count > self.config.function_line_count_max:
            return 0.0  # Too complex
        
        # Score based on line count range (from config)
        optimal_min, optimal_max = self.config.function_line_count_optimal_range
        good_min, good_max = self.config.function_line_count_good_range
        
        if optimal_min <= line_count <= optimal_max:
            score += scoring['line_count_optimal']
        elif good_min <= line_count < optimal_min or optimal_max < line_count <= good_max:
            score += scoring['line_count_good']
        else:
            score += scoring['line_count_acceptable']
        
        # Has docstring
        if ast.get_docstring(func_node):
            score += scoring['has_docstring']
        
        # Has type hints
        if self._has_type_hints(func_node):
            score += scoring['has_type_hints']
        
        # Has complexity (if/loop statements)
        has_control_flow = any(
            isinstance(node, (ast.If, ast.For, ast.While, ast.Try))
            for node in ast.walk(func_node)
        )
        if has_control_flow:
            score += scoring['has_control_flow']
        
        # Has return statement
        has_return = any(
            isinstance(node, ast.Return) and node.value is not None
            for node in ast.walk(func_node)
        )
        if has_return:
            score += scoring['has_return']
        
        # Not too many arguments (from config)
        arg_count = len(func_node.args.args)
        if arg_count <= self.config.function_max_arguments:
            score += scoring['arg_count_ok']
        
        return min(score, 1.0)
    
    def _has_type_hints(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has type hints."""
        # Check return annotation
        if func_node.returns:
            return True
        
        # Check argument annotations
        for arg in func_node.args.args:
            if arg.annotation:
                return True
        
        return False
    
    def _save_triples_cache(self, triples: List[Dict[str, Any]]) -> None:
        """Save generated triples to cache."""
        try:
            with open(self.triples_cache_file, 'w') as f:
                json.dump(triples, f, indent=2)
            logger.info(f"[TripleGenerator] Saved {len(triples)} triples to cache")
        except Exception as e:
            logger.warning(f"[TripleGenerator] Failed to save triples cache: {e}")
    
    def load_cached_triples(self) -> Optional[List[Dict[str, Any]]]:
        """Load previously generated triples from cache."""
        if not self.triples_cache_file.exists():
            return None
        
        try:
            with open(self.triples_cache_file, 'r') as f:
                triples = json.load(f)
            logger.info(f"[TripleGenerator] Loaded {len(triples)} triples from cache")
            return triples
        except Exception as e:
            logger.warning(f"[TripleGenerator] Failed to load triples cache: {e}")
            return None
