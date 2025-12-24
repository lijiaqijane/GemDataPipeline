from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str
    base_url: str
    model: str
    api_key: str | None
    timeout: float = 120.0
    max_retries: int = 3

    @classmethod
    def from_env(cls) -> "LLMConfig":
        provider = os.getenv("LLM_PROVIDER", "deepseek").lower()
        if provider not in {"deepseek", "volcano", "openai", "vllm"}:
            raise RuntimeError(f"Unknown LLM provider {provider}")

        # Deepseek via Volcano Ark (OpenAI-compatible endpoint)
        # We treat both "deepseek" and "volcano" providers as using VOLCANO_* env vars,
        # and allow DEEPSEEK_* overrides for flexibility.
        if provider in {"deepseek", "volcano"}:
            base_url = os.getenv(
                "VOLCANO_BASE_URL",
                "https://ark.cn-beijing.volces.com/api/v3",
            )
            model = os.getenv("VOLCANO_MODEL", "deepseek-v3-2-251201")
            api_key = os.getenv("VOLCANO_API_KEY")

            # Optional overrides using DEEPSEEK_* if provided
            base_url = os.getenv("DEEPSEEK_BASE_URL", base_url)
            model = os.getenv("DEEPSEEK_MODEL", model)
            api_key = (
                os.getenv("DEEPSEEK_API_KEY")
                or os.getenv("DEEPSEEK_API")
                or api_key
            )
        elif provider == "openai":
            base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
            api_key = os.getenv("OPENAI_API_KEY")
        else:
            base_url = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
            model = os.getenv("VLLM_MODEL", "local-model")
            api_key = os.getenv("VLLM_API_KEY")

        timeout = float(os.getenv("LLM_TIMEOUT", "120"))
        max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
        return cls(
            provider=provider,
            base_url=base_url,
            model=model,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
        )


class CodeAgentConfig:
    """
    Unified configuration for CodeAgent pipeline (triple generation + task generation).
    
    This class provides a unified interface for configuring both:
    1. Triple generation (repo/file/function filtering)
    2. Task generation (CodeAgent parameters)
    """
    
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initialize configuration from dictionary.
        
        Args:
            config_dict: Configuration dictionary loaded from YAML
        """
        self.config = config_dict
        self._validate()
    
    @classmethod
    def from_file(cls, config_path: str | Path) -> CodeAgentConfig:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to YAML configuration file
            
        Returns:
            CodeAgentConfig instance
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If config is invalid
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        logger.info(f"[Config] Loading CodeAgent configuration from {config_path}")
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
            
            if not config_dict:
                raise ValueError("Configuration file is empty")
            
            return cls(config_dict)
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
        except Exception as e:
            raise ValueError(f"Error loading configuration: {e}")
    
    @classmethod
    def default(cls) -> CodeAgentConfig:
        """
        Create default configuration.
        
        Returns:
            CodeAgentConfig with default values
        """
        default_config = {
            'triple_generation': {
                'repo_filter': {
                    'stars': {'min': 100, 'max': 10000},
                    'size': {'max': 10240},
                    'updated_within_days': 730,
                    'topics': ['library', 'tool', 'cli'],
                    'exclude_keywords': ['tensorflow', 'pytorch', 'gpu'],
                    'exclude_topics': ['machine-learning', 'deep-learning'],
                    'max_repos': 50,
                },
                'file_filter': {
                    'focus_main_package': True,
                    'exclude_directories': ['tests', 'docs', 'examples'],
                    'exclude_files': ['__init__.py', 'setup.py'],
                    'line_count': {'min': 50, 'max': 2000, 'optimal_min': 100, 'optimal_max': 500},
                    'min_functions': 2,
                    'scoring': {
                        'line_count_optimal': 20,
                        'function_count': 10,
                        'function_count_max': 50,
                        'docstring': 5,
                        'not_test': 15,
                        'utility_penalty': -10,
                    },
                    'utility_keywords': ['util', 'helper'],
                    'max_files_per_repo': 10,
                },
                'function_filter': {
                    'skip_private': True,
                    'line_count': {'min': 5, 'max': 200, 'optimal_min': 10, 'optimal_max': 100},
                    'max_arguments': 5,
                    'min_quality_score': 0.3,
                    'scoring': {
                        'line_count_optimal': 0.3,
                        'has_docstring': 0.2,
                        'has_type_hints': 0.1,
                        'has_control_flow': 0.2,
                        'has_return': 0.1,
                        'arg_count_ok': 0.1,
                    },
                },
                'output': {
                    'target_triples': 100,
                    'quality_threshold': 0.5,
                    'cache_dir': 'triple_cache',
                    'refresh_repos': False,
                },
            },
            'task_generation': {
                'taskdb_root': 'batch_taskdb',
                'difficulty': 2,
                'max_tokens': 2000,
                'temperature': 0.7,
                'save_logs': True,
                'auto_save': True,
            },
            'batch_processing': {
                'num_tasks': 10,
                'skip_errors': False,
                'max_retries': 5,
            },
            'sandbox': {
                'sandbox_url': None,  # Auto-detect from env
                'use_docker_runner': True,
                'silent': False,
            }
        }
        
        return cls(default_config)
    
    def _validate(self) -> None:
        """
        Validate configuration structure and values.
        
        Raises:
            ValueError: If configuration is invalid
        """
        required_sections = ['task_generation', 'batch_processing']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required section: {section}")
        
        # Validate task_generation
        task_gen = self.config['task_generation']
        if 'difficulty' in task_gen:
            if not 1 <= task_gen['difficulty'] <= 3:
                raise ValueError("task_generation.difficulty must be between 1 and 3")
        if 'max_tokens' in task_gen:
            if task_gen['max_tokens'] <= 0:
                raise ValueError("task_generation.max_tokens must be > 0")
        if 'temperature' in task_gen:
            if not 0 <= task_gen['temperature'] <= 2:
                raise ValueError("task_generation.temperature must be between 0 and 2")
        
        # Validate batch_processing
        batch = self.config['batch_processing']
        if 'num_tasks' in batch:
            if batch['num_tasks'] <= 0:
                raise ValueError("batch_processing.num_tasks must be > 0")
        if 'max_retries' in batch:
            if batch['max_retries'] < 0:
                raise ValueError("batch_processing.max_retries must be >= 0")
        
        # Validate triple_generation if present
        if 'triple_generation' in self.config:
            triple_gen = self.config['triple_generation']
            
            # Validate repo_filter
            if 'repo_filter' in triple_gen:
                repo = triple_gen['repo_filter']
                if 'stars' in repo:
                    if repo['stars'].get('min', 0) > repo['stars'].get('max', float('inf')):
                        raise ValueError("repo_filter.stars.min must be <= max")
            
            # Validate file_filter
            if 'file_filter' in triple_gen:
                file_f = triple_gen['file_filter']
                if 'line_count' in file_f:
                    lc = file_f['line_count']
                    if lc.get('min', 0) > lc.get('max', float('inf')):
                        raise ValueError("file_filter.line_count.min must be <= max")
            
            # Validate function_filter
            if 'function_filter' in triple_gen:
                func_f = triple_gen['function_filter']
                if 'min_quality_score' in func_f:
                    score = func_f['min_quality_score']
                    if not 0 <= score <= 1:
                        raise ValueError("function_filter.min_quality_score must be between 0 and 1")
        
        logger.info("[Config] CodeAgent configuration validated successfully")
    
    # Convenience property accessors
    
    @property
    def taskdb_root(self) -> str:
        return self.config['task_generation'].get('taskdb_root', 'batch_taskdb')
    
    @property
    def difficulty(self) -> int:
        return self.config['task_generation'].get('difficulty', 2)
    
    @property
    def max_tokens(self) -> int:
        return self.config['task_generation'].get('max_tokens', 2000)
    
    @property
    def temperature(self) -> float:
        return self.config['task_generation'].get('temperature', 0.7)
    
    @property
    def save_logs(self) -> bool:
        return self.config['task_generation'].get('save_logs', True)
    
    @property
    def auto_save(self) -> bool:
        return self.config['task_generation'].get('auto_save', True)
    
    @property
    def target_test_count(self) -> int:
        """Get target number of test functions to generate."""
        return self.config['task_generation'].get('test_generation', {}).get('target_test_count', 4)
    
    @property
    def min_test_count(self) -> int:
        """Get minimum number of test functions."""
        return self.config['task_generation'].get('test_generation', {}).get('min_test_count', 3)
    
    @property
    def max_test_count(self) -> int:
        """Get maximum number of test functions."""
        return self.config['task_generation'].get('test_generation', {}).get('max_test_count', 6)
    
    @property
    def num_tasks(self) -> int:
        return self.config['batch_processing'].get('num_tasks', 10)
    
    @property
    def skip_errors(self) -> bool:
        return self.config['batch_processing'].get('skip_errors', False)
    
    @property
    def max_retries(self) -> int:
        return self.config['batch_processing'].get('max_retries', 5)
    
    @property
    def sandbox_url(self) -> Optional[str]:
        return self.config.get('sandbox', {}).get('sandbox_url')
    
    @property
    def use_docker_runner(self) -> bool:
        return self.config.get('sandbox', {}).get('use_docker_runner', True)
    
    @property
    def silent(self) -> bool:
        return self.config.get('sandbox', {}).get('silent', False)
    
    # Triple generation property accessors
    
    @property
    def has_triple_generation(self) -> bool:
        """Check if triple generation config is present."""
        return 'triple_generation' in self.config
    
    @property
    def cache_dir(self) -> str:
        """Get cache directory for triples."""
        return self.config.get('triple_generation', {}).get('output', {}).get('cache_dir', 'triple_cache')
    
    @property
    def target_triples(self) -> int:
        """Get target number of triples to generate."""
        return self.config.get('triple_generation', {}).get('output', {}).get('target_triples', 100)
    
    @property
    def quality_threshold(self) -> float:
        """Get quality threshold for triples."""
        return self.config.get('triple_generation', {}).get('output', {}).get('quality_threshold', 0.5)
    
    @property
    def refresh_repos(self) -> bool:
        """Get whether to refresh repository cache."""
        return self.config.get('triple_generation', {}).get('output', {}).get('refresh_repos', False)
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path.
        
        Args:
            key_path: Dot-separated path (e.g., 'task_generation.difficulty')
            default: Default value if key not found
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __repr__(self) -> str:
        return f"CodeAgentConfig(num_tasks={self.num_tasks}, difficulty={self.difficulty})"
