"""
PR Annotator Module

This module provides functionality to annotate PRs using LLM, including:
- PR category (bug fix or feature request)
- Issue difficulty (easy, medium, hard)
- Whether issue description is reasonable
- Whether gold patch solves the core issue
- Whether test patch is designed for the issue
- Whether the code requires GPU resources to run
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from agent_gem.llm import LLMClient

# Try to import tiktoken for accurate token counting
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

logger = logging.getLogger(__name__)


@dataclass
class PRAnnotation:
    """Annotation result for a single PR."""
    
    instance_id: str
    pr_category: str  # "bug_fix" or "feature_request"
    issue_difficulty: str  # "easy", "medium", or "hard"
    issue_description_reasonable: bool
    gold_patch_solves_issue: bool
    test_patch_designed_for_issue: bool
    requires_gpu: bool  # Whether the code requires GPU resources to run
    num_gold_files_changed: int  # Number of files changed in gold patch
    num_test_files_changed: int  # Number of files changed in test patch
    reasoning: Optional[str] = None  # Optional reasoning from LLM


@dataclass
class PRAnnotatorConfig:
    """Configuration for PR annotation settings."""
    
    # Input settings
    input_file: Optional[str] = None  # Path to valid PR JSONL file
    input_dir: Optional[str] = None  # Directory containing valid PR files
    
    # Output settings
    output_dir: str = "taskdb/code_agent/pr_annotations"
    output_file: Optional[str] = None  # Optional: specific output file name
    
    # Processing settings
    max_prs: Optional[int] = None  # Maximum number of PRs to annotate (None = all)
    resume: bool = True  # Resume from existing annotations
    
    # LLM settings
    llm_temperature: float = 0.3  # Lower temperature for more consistent annotations
    llm_max_tokens: int = 2048  # Max tokens for LLM response
    max_prompt_length: Optional[int] = None  # Maximum prompt length in tokens (None = no limit)
    
    # Retry settings
    max_retries: int = 3  # Maximum number of retries for failed LLM annotations
    retry_delay: float = 1.0  # Delay in seconds between retries


class PRAnnotator:
    """
    PR Annotator for labeling PRs using LLM.
    
    This class reads valid PR files and uses LLM to annotate each PR with:
    - PR category (bug fix or feature request)
    - Issue difficulty (easy, medium, hard)
    - Whether issue description is reasonable
    - Whether gold patch solves the core issue
    - Whether test patch is designed for the issue
    - Whether the code requires GPU resources to run
    
    Example:
        >>> config = PRAnnotatorConfig(input_file="taskdb/code_agent/prs/repo-prs-valid.jsonl")
        >>> annotator = PRAnnotator(config)
        >>> annotator.annotate_all()
        
        # Or load from YAML
        >>> annotator = PRAnnotator.load_config_from_yaml("config/code_agent.yaml")
        >>> annotator.annotate_all()
    """
    
    def __init__(self, config: PRAnnotatorConfig):
        """
        Initialize the PR annotator.
        
        Args:
            config: PR annotation configuration
        """
        self.config = config
        
        # Initialize LLM client
        self.llm = LLMClient.from_env()
        
        # Initialize tokenizer for token counting
        self.tokenizer = None
        if TIKTOKEN_AVAILABLE and config.max_prompt_length is not None:
            try:
                # Try to get encoding for the model (default to cl100k_base for GPT-4)
                model_name = self.llm.config.model.lower()
                if "gpt-4" in model_name or "gpt-3.5" in model_name:
                    encoding_name = "cl100k_base"
                elif "gpt-3" in model_name:
                    encoding_name = "p50k_base"
                else:
                    # Default to cl100k_base for most modern models
                    encoding_name = "cl100k_base"
                
                self.tokenizer = tiktoken.get_encoding(encoding_name)
                logger.info(f"[PRAnnotator] Using tiktoken encoding: {encoding_name}")
            except Exception as e:
                logger.warning(f"[PRAnnotator] Failed to initialize tiktoken: {e}. Will use character-based estimation.")
                self.tokenizer = None
        
        # Prepare output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[PRAnnotator] Initialized with output directory: {self.output_dir}")
    
    @classmethod
    def load_config_from_yaml(cls, yaml_path: str | Path) -> "PRAnnotator":
        """
        Load configuration from YAML file and create PRAnnotator instance.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            PRAnnotator instance with loaded configuration
            
        Raises:
            FileNotFoundError: If the YAML file does not exist
            ValueError: If the configuration is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if 'pr_annotator' not in config_data:
            logger.warning("No 'pr_annotator' section in config, using defaults")
            return cls(PRAnnotatorConfig())
        
        annotator_config = config_data['pr_annotator']
        
        # Parse input settings
        input_file = annotator_config.get('input_file')
        input_dir = annotator_config.get('input_dir')
        
        # Parse output settings
        output = annotator_config.get('output', {})
        output_dir = output.get('save_dir', 'taskdb/code_agent/pr_annotations')
        output_file = output.get('file_name')
        
        # Parse processing settings
        max_prs = annotator_config.get('max_prs')
        resume = annotator_config.get('resume', True)
        
        # Parse LLM settings
        llm = annotator_config.get('llm', {})
        llm_temperature = llm.get('temperature', 0.3)
        llm_max_tokens = llm.get('max_tokens', 2048)
        max_prompt_length = llm.get('max_prompt_length')
        
        # Parse retry settings
        max_retries = annotator_config.get('max_retries', 3)
        retry_delay = annotator_config.get('retry_delay', 1.0)
        
        config = PRAnnotatorConfig(
            input_file=input_file,
            input_dir=input_dir,
            output_dir=output_dir,
            output_file=output_file,
            max_prs=max_prs,
            resume=resume,
            llm_temperature=llm_temperature,
            llm_max_tokens=llm_max_tokens,
            max_prompt_length=max_prompt_length,
            max_retries=max_retries,
            retry_delay=retry_delay,
        )
        
        return cls(config)
    
    def _create_annotation_prompt(self, pr_data: dict) -> str:
        """
        Create a prompt for LLM to annotate a PR.
        
        Args:
            pr_data: PR data dictionary
            
        Returns:
            Prompt string for LLM
        """
        problem_statement = pr_data.get('problem_statement', '')
        patch = pr_data.get('patch', '')
        test_patch = pr_data.get('test_patch', '')
        repo = pr_data.get('repo', '')
        pull_number = pr_data.get('pull_number', '')
        
        prompt = f"""You are an expert code reviewer. Please analyze the following pull request and provide annotations.

Repository: {repo}
Pull Request Number: {pull_number}

Problem Statement:
{problem_statement}

Gold Patch (the fix):
```diff
{patch}
```

Test Patch (the test):
```diff
{test_patch}
```

Please provide annotations in the following JSON format:
{{
    "pr_category": "bug_fix" or "feature_request",
    "issue_difficulty": "easy", "medium", or "hard",
    "issue_description_reasonable": true or false,
    "gold_patch_solves_issue": true or false,
    "test_patch_designed_for_issue": true or false,
    "requires_gpu": true or false,
    "reasoning": "Brief explanation of your annotations"
}}

Guidelines:
1. PR category: "bug_fix" if the PR fixes a bug, "feature_request" if it adds new functionality
2. Issue difficulty: 
   - "easy": Simple fix, minimal code changes, straightforward logic
   - "medium": Moderate complexity, requires understanding of codebase, some refactoring
   - "hard": Complex fix, requires deep understanding, significant changes or architectural decisions
3. Issue description reasonable: true if the problem statement clearly describes the issue, false if it's vague or unclear
4. Gold patch solves issue: true if the patch directly addresses the core problem described in the issue
5. Test patch designed for issue: true if the test patch specifically tests the issue being fixed
6. Requires GPU: true if the code to solve this issue requires GPU resources (e.g., deep learning models, CUDA operations, GPU-accelerated computations, neural network training/inference). Consider both the gold patch and test patch. false if the code can run on CPU only.

Please respond with ONLY the JSON object, no additional text."""
        
        return prompt
    
    def _parse_llm_response(self, response: str) -> Optional[Dict]:
        """
        Parse LLM response to extract annotation data.
        
        Args:
            response: LLM response string
            
        Returns:
            Dictionary with annotation data, or None if parsing fails
        """
        try:
            # Try to extract JSON from response
            # Remove markdown code blocks if present
            response = response.strip()
            if response.startswith("```"):
                # Remove code block markers (handle ```json or ```)
                lines = response.split('\n')
                # Remove first line if it's a code block marker
                if lines[0].strip().startswith('```'):
                    lines = lines[1:]
                # Remove last line if it's a code block marker
                if lines and lines[-1].strip().startswith('```'):
                    lines = lines[:-1]
                response = '\n'.join(lines)
            
            # Find JSON object
            start_idx = response.find('{')
            end_idx = response.rfind('}')
            if start_idx == -1 or end_idx == -1:
                logger.warning("No JSON object found in LLM response")
                return None
            
            json_str = response[start_idx:end_idx+1]
            data = json.loads(json_str)
            
            # Validate required fields
            required_fields = [
                'pr_category', 'issue_difficulty', 'issue_description_reasonable',
                'gold_patch_solves_issue', 'test_patch_designed_for_issue', 'requires_gpu'
            ]
            for field in required_fields:
                if field not in data:
                    logger.warning(f"Missing required field '{field}' in LLM response")
                    return None
            
            # Validate values
            if data['pr_category'] not in ['bug_fix', 'feature_request']:
                logger.warning(f"Invalid pr_category: {data['pr_category']}")
                return None
            
            if data['issue_difficulty'] not in ['easy', 'medium', 'hard']:
                logger.warning(f"Invalid issue_difficulty: {data['issue_difficulty']}")
                return None
            
            if not isinstance(data['issue_description_reasonable'], bool):
                logger.warning(f"issue_description_reasonable must be boolean")
                return None
            
            if not isinstance(data['gold_patch_solves_issue'], bool):
                logger.warning(f"gold_patch_solves_issue must be boolean")
                return None
            
            if not isinstance(data['test_patch_designed_for_issue'], bool):
                logger.warning(f"test_patch_designed_for_issue must be boolean")
                return None
            
            if not isinstance(data['requires_gpu'], bool):
                logger.warning(f"requires_gpu must be boolean")
                return None
            
            return data
            
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON from LLM response: {e}")
            return None
        except Exception as e:
            logger.warning(f"Error parsing LLM response: {e}")
            return None
    
    def _count_files_in_patch(self, patch: str) -> int:
        """
        Count the number of files changed in a patch.
        
        Args:
            patch: Patch string in git diff format
            
        Returns:
            Number of unique files changed
        """
        if not patch or not patch.strip():
            return 0
        
        # Pattern to match file paths in diff format
        # Matches: diff --git a/path b/path
        file_pattern = r'^diff --git a/(.+?) b/(.+?)$'
        
        files = set()
        
        for line in patch.split('\n'):
            match = re.match(file_pattern, line)
            if match:
                # Extract file path (use the 'b' path as it's the final path)
                file_path = match.group(2).strip()
                if file_path and file_path != '/dev/null':
                    files.add(file_path)
        
        return len(files)
    
    def _count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a text string.
        
        Args:
            text: Text string to count tokens for
            
        Returns:
            Number of tokens
        """
        if self.tokenizer:
            # Use tiktoken for accurate token counting
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimation (approximately 4 characters per token)
            return len(text) // 4
    
    def annotate_pr(self, pr_data: dict) -> Optional[PRAnnotation]:
        """
        Annotate a single PR using LLM with retry mechanism.
        
        Args:
            pr_data: PR data dictionary
            
        Returns:
            PRAnnotation object, or None if annotation fails after all retries
        """
        instance_id = pr_data.get('instance_id')
        if not instance_id:
            logger.warning("PR data missing instance_id, skipping")
            return None
        
        # Count files changed (automatically, no LLM needed)
        gold_patch = pr_data.get('patch', '')
        test_patch = pr_data.get('test_patch', '')
        num_gold_files_changed = self._count_files_in_patch(gold_patch)
        num_test_files_changed = self._count_files_in_patch(test_patch)
        
        # Create prompt
        prompt = self._create_annotation_prompt(pr_data)
        
        # Check prompt token count before calling LLM
        if self.config.max_prompt_length is not None:
            prompt_tokens = self._count_tokens(prompt)
            if prompt_tokens > self.config.max_prompt_length:
                logger.warning(
                    f"Prompt token count ({prompt_tokens}) exceeds maximum ({self.config.max_prompt_length} tokens) "
                    f"for PR {instance_id}. Skipping annotation."
                )
                return None
        
        # Retry mechanism
        last_exception = None
        for attempt in range(self.config.max_retries + 1):
            try:
                # Call LLM
                messages = [
                    {"role": "system", "content": "You are an expert code reviewer. Provide accurate annotations in JSON format."},
                    {"role": "user", "content": prompt}
                ]
                
                if attempt > 0:
                    logger.info(f"Retrying annotation for {instance_id} (attempt {attempt + 1}/{self.config.max_retries + 1})")
                    time.sleep(self.config.retry_delay)
                
                response = self.llm.chat_completion(
                    messages=messages,
                    temperature=self.config.llm_temperature,
                    max_tokens=self.config.llm_max_tokens,
                )
                
                # Parse response
                annotation_data = self._parse_llm_response(response)
                if not annotation_data:
                    if attempt < self.config.max_retries:
                        logger.warning(f"Failed to parse annotation for {instance_id}, will retry")
                        continue
                    else:
                        logger.warning(f"Failed to parse annotation for {instance_id} after {self.config.max_retries + 1} attempts")
                        return None
                
                # Create annotation object
                annotation = PRAnnotation(
                    instance_id=instance_id,
                    pr_category=annotation_data['pr_category'],
                    issue_difficulty=annotation_data['issue_difficulty'],
                    issue_description_reasonable=annotation_data['issue_description_reasonable'],
                    gold_patch_solves_issue=annotation_data['gold_patch_solves_issue'],
                    test_patch_designed_for_issue=annotation_data['test_patch_designed_for_issue'],
                    requires_gpu=annotation_data['requires_gpu'],
                    num_gold_files_changed=num_gold_files_changed,
                    num_test_files_changed=num_test_files_changed,
                    reasoning=annotation_data.get('reasoning'),
                )
                
                return annotation
                
            except Exception as e:
                last_exception = e
                error_str = str(e)
                
                # Check if it's an input length exceeded error - don't retry for this
                if any(keyword in error_str for keyword in [
                    "exceeds the maximum length",
                    "InvalidParameter",
                    "Input length",
                    "maximum length"
                ]):
                    logger.warning(
                        f"Input length exceeded for PR {instance_id}. "
                        f"Error: {error_str}. Skipping (will not retry)."
                    )
                    return None
                
                if attempt < self.config.max_retries:
                    logger.warning(f"Error annotating PR {instance_id} (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}")
                    continue
                else:
                    logger.error(f"Error annotating PR {instance_id} after {self.config.max_retries + 1} attempts: {e}", exc_info=True)
        
        return None
    
    def _get_input_files(self) -> List[Path]:
        """
        Get list of input PR files to process.
        
        Returns:
            List of Path objects for input files
        """
        input_files = []
        
        if self.config.input_file:
            input_file = Path(self.config.input_file)
            if input_file.exists():
                input_files.append(input_file)
                return input_files
            else:
                logger.warning(f"Input file not found: {input_file}")
            
        
        if self.config.input_dir:
            input_dir = Path(self.config.input_dir)
            if input_dir.exists() and input_dir.is_dir():
                # Find all *-prs-valid.jsonl files
                for file in input_dir.glob("*-prs-valid.jsonl"):
                    input_files.append(file)
            else:
                logger.warning(f"Input directory not found: {input_dir}")
        
        if not input_files:
            raise ValueError("No input files found. Please specify input_file or input_dir in config.")
        
        return input_files
    
    def _get_output_file(self, input_file: Path) -> Path:
        """
        Get output file path for annotations.
        
        Args:
            input_file: Input PR file path
            
        Returns:
            Output file path
        """
        if self.config.output_file:
            return self.output_dir / self.config.output_file
        
        # Generate output filename from input filename
        input_name = input_file.stem  # e.g., "repo-prs-valid"
        output_name = input_name.replace("-prs-valid", "-prs-annotated") + ".jsonl"
        return self.output_dir / output_name
    
    def _load_existing_annotations(self, output_file: Path) -> set:
        """
        Load existing annotations from output file.
        
        Args:
            output_file: Path to output file
            
        Returns:
            Set of instance_ids that have already been annotated
        """
        annotated_ids = set()
        
        if output_file.exists() and self.config.resume:
            try:
                with open(output_file, 'r') as f:
                    for line in f:
                        if line.strip():
                            annotation = json.loads(line)
                            annotated_ids.add(annotation.get('instance_id'))
                logger.info(f"Loaded {len(annotated_ids)} existing annotations from {output_file}")
            except Exception as e:
                logger.warning(f"Failed to load existing annotations: {e}")
        
        return annotated_ids
    
    def annotate_all(self) -> Dict[str, int]:
        """
        Annotate all PRs from input files.
        
        Returns:
            Dictionary with statistics: total_prs, annotated_prs, failed_prs
        """
        input_files = self._get_input_files()
        
        stats = {
            'total_prs': 0,
            'annotated_prs': 0,
            'failed_prs': 0,
            'skipped_prs': 0,
        }
        
        for input_file in input_files:
            logger.info(f"Processing input file: {input_file}")
            output_file = self._get_output_file(input_file)
            
            # Load existing annotations
            annotated_ids = self._load_existing_annotations(output_file)
            
            # Open output file in append mode if resuming, otherwise write mode
            mode = 'a' if (output_file.exists() and self.config.resume) else 'w'
            
            with open(output_file, mode) as out_f:
                with open(input_file, 'r') as in_f:
                    for line_num, line in enumerate(in_f, 1):
                        if line.strip():
                            try:
                                pr_data = json.loads(line)
                                instance_id = pr_data.get('instance_id')
                                
                                if not instance_id:
                                    logger.warning(f"PR at line {line_num} missing instance_id, skipping")
                                    stats['failed_prs'] += 1
                                    continue
                                
                                stats['total_prs'] += 1
                                
                                # Check if already annotated
                                if instance_id in annotated_ids:
                                    logger.debug(f"Skipping already annotated PR: {instance_id}")
                                    stats['skipped_prs'] += 1
                                    continue
                                
                                # Check max_prs limit
                                if self.config.max_prs and stats['annotated_prs'] >= self.config.max_prs:
                                    logger.info(f"Reached max_prs limit ({self.config.max_prs})")
                                    break
                                
                                # Annotate PR
                                logger.info(f"Annotating PR {instance_id} ({stats['annotated_prs'] + 1}/{stats['total_prs']})")
                                annotation = self.annotate_pr(pr_data)
                                
                                if annotation:
                                    # Convert annotation to dict and save
                                    annotation_dict = {
                                        'instance_id': annotation.instance_id,
                                        'pr_category': annotation.pr_category,
                                        'issue_difficulty': annotation.issue_difficulty,
                                        'issue_description_reasonable': annotation.issue_description_reasonable,
                                        'gold_patch_solves_issue': annotation.gold_patch_solves_issue,
                                        'test_patch_designed_for_issue': annotation.test_patch_designed_for_issue,
                                        'requires_gpu': annotation.requires_gpu,
                                        'num_gold_files_changed': annotation.num_gold_files_changed,
                                        'num_test_files_changed': annotation.num_test_files_changed,
                                        'reasoning': annotation.reasoning,
                                    }
                                    print(json.dumps(annotation_dict), file=out_f, flush=True)
                                    stats['annotated_prs'] += 1
                                else:
                                    stats['failed_prs'] += 1
                                
                            except json.JSONDecodeError as e:
                                logger.warning(f"Failed to parse PR at line {line_num}: {e}")
                                stats['failed_prs'] += 1
                            except Exception as e:
                                logger.error(f"Error processing PR at line {line_num}: {e}", exc_info=True)
                                stats['failed_prs'] += 1
            
            logger.info(f"Completed processing {input_file}")
            logger.info(f"  - Total PRs: {stats['total_prs']}")
            logger.info(f"  - Annotated: {stats['annotated_prs']}")
            logger.info(f"  - Skipped: {stats['skipped_prs']}")
            logger.info(f"  - Failed: {stats['failed_prs']}")
            logger.info(f"  - Output saved to: {output_file}")
        
        return stats
