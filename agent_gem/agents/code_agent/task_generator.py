"""
Task Generator Module

This module provides functionality to generate training tasks for agents based on
extracted code entities. Each task includes the entity's file (in stub form), an issue
description, and the ground truth patch to implement the entity.
"""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class TaskGeneratorConfig:
    """Configuration for task generation."""
    
    # Input settings
    entity_dir: str = "taskdb/code_agent/extracted_entities"
    entity_filename: str = "entity.json"
    issue_filename: str = "issue.txt"
    
    # Output settings
    save_dir: str = "taskdb/code_agent/tasks"
    save_format: str = "json"  # json or jsonl
    
    # Task content settings
    include_docstring: bool = True  # Include docstring in stub
    include_type_hints: bool = True  # Include type hints in stub
    include_decorators: bool = True  # Include decorators in stub
    
    # Issue generation settings
    issue_template: str = "default"  # default, detailed, or minimal
    include_context: bool = True  # Include surrounding code context
    context_lines: int = 3  # Number of context lines before/after entity
    
    # Filtering settings
    min_entity_lines: int = 5  # Minimum entity size for task generation
    max_entity_lines: int = 200  # Maximum entity size for task generation
    exclude_simple_entities: bool = True  # Exclude entities with complexity < 2
    
    # Metadata settings
    include_metadata: bool = True
    include_entity_properties: bool = True


@dataclass
class Task:
    """A training task for code generation."""
    
    task_id: str
    prompt: str
    issue: str
    entity_info: Dict[str, Any]
    ground_truth: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            'task_id': self.task_id,
            'prompt': self.prompt,
            'issue': self.issue,
            'entity_info': self.entity_info,
            'ground_truth': self.ground_truth,
            'metadata': self.metadata,
        }


class TaskGenerator:
    """
    Generate training tasks from code entities.
    
    This class converts extracted entities into training tasks for agents.
    Each task includes:
    - A stub version of the entity's file
    - An issue description requesting implementation
    - Ground truth patch from stub to full implementation
    
    Example:
        >>> config = TaskGenerator.load_config_from_yaml("config/code_agent.yaml")
        >>> generator = TaskGenerator(config)
        >>> tasks = generator.generate_tasks_from_entities(entities)
        >>> generator.save_tasks(tasks)
    """
    
    def __init__(self, config: Optional[TaskGeneratorConfig] = None):
        """
        Initialize the task generator.
        
        Args:
            config: TaskGeneratorConfig instance. If None, uses default config.
        """
        self.config = config or TaskGeneratorConfig()
        
        # Prepare output directory
        self.output_dir = Path(self.config.save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"[TaskGenerator] Initialized with output directory: {self.output_dir}")
    
    @classmethod
    def load_config_from_yaml(cls, yaml_path: str | Path) -> "TaskGenerator":
        """
        Load configuration from YAML file and create TaskGenerator instance.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            TaskGenerator instance with loaded configuration
            
        Raises:
            FileNotFoundError: If the YAML file does not exist
            ValueError: If the configuration is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if 'task_generator' not in config_data:
            logger.warning("No 'task_generator' section in config, using defaults")
            return cls()
        
        task_gen_config = config_data['task_generator']
        
        # Parse input settings
        input_cfg = task_gen_config.get('input', {})
        
        # Parse output settings
        output = task_gen_config.get('output', {})
        
        # Parse task content settings
        task_content = task_gen_config.get('task_content', {})
        
        # Parse issue generation settings
        issue_gen = task_gen_config.get('issue_generation', {})
        
        # Parse filtering settings
        filters = task_gen_config.get('filters', {})
        
        # Parse metadata settings
        metadata = task_gen_config.get('metadata', {})
        
        config = TaskGeneratorConfig(
            # Output
            save_dir=output.get('save_dir', 'taskdb/code_agent/tasks'),
            save_format=output.get('save_format', 'json'),
            # Task content
            include_docstring=task_content.get('include_docstring', True),
            include_type_hints=task_content.get('include_type_hints', True),
            include_decorators=task_content.get('include_decorators', True),
            # Issue generation
            issue_template=issue_gen.get('template', 'default'),
            include_context=issue_gen.get('include_context', True),
            context_lines=issue_gen.get('context_lines', 3),
            # Filters
            min_entity_lines=filters.get('min_entity_lines', 5),
            max_entity_lines=filters.get('max_entity_lines', 200),
            exclude_simple_entities=filters.get('exclude_simple_entities', True),
            # Metadata
            include_metadata=metadata.get('include_metadata', True),
            include_entity_properties=metadata.get('include_entity_properties', True),
            # Input
            entity_dir=input_cfg.get('entity_dir', 'taskdb/code_agent/extracted_entities'),
            entity_filename=input_cfg.get('entity_filename', 'entity.json'),
            issue_filename=input_cfg.get('issue_filename', 'issue.txt'),
        )
        
        return cls(config)
    
    def _extract_line_span(self, entity_data: Dict[str, Any]) -> tuple[int, int]:
        """Get start/end lines from entity metadata."""
        line_start = entity_data.get('start_line') or entity_data.get('line_start')
        line_end = entity_data.get('end_line') or entity_data.get('line_end')
        if line_start is None or line_end is None:
            raise ValueError("Entity data missing start_line/end_line")
        return int(line_start), int(line_end)
    
    def _get_file_content_with_stub(self, entity_data: Dict[str, Any], file_content: str) -> str:
        """Replace the entity span in file_content with the provided stub code."""
        line_start, line_end = self._extract_line_span(entity_data)
        stub_code = entity_data.get('stub')
        if not stub_code:
            raise ValueError("Entity data missing stub code")

        lines = file_content.splitlines(keepends=True)

        # Calculate 0-indexed line numbers
        start_idx = line_start - 1
        end_idx = line_end - 1

        # Preserve indentation of original entity
        if lines and start_idx < len(lines):
            original_indent = len(lines[start_idx]) - len(lines[start_idx].lstrip())
            stub_lines = stub_code.splitlines(keepends=True)

            indented_stub_lines = []
            for line in stub_lines:
                if line.strip():
                    current_indent = len(line) - len(line.lstrip())
                    indent_diff = original_indent - current_indent
                    if indent_diff > 0:
                        line = ' ' * indent_diff + line
                indented_stub_lines.append(line)
            stub_code = ''.join(indented_stub_lines)

        new_lines = lines[:start_idx] + [stub_code + '\n'] + lines[end_idx + 1:]
        return ''.join(new_lines)
    
    def _generate_code_patch(self, original: str, modified: str) -> str:
        """
        Generate a unified diff patch between original and modified content.
        
        Args:
            original: Original content (with stub)
            modified: Modified content (with full implementation)
            
        Returns:
            Unified diff patch as string
        """
        original_lines = original.splitlines(keepends=True)
        modified_lines = modified.splitlines(keepends=True)
        
        diff = difflib.unified_diff(
            original_lines,
            modified_lines,
            fromfile='a/file.py',
            tofile='b/file.py',
            lineterm='',
        )
        
        return ''.join(diff)
    
    def _generate_prompt(self, issue: str, stub_file: str) -> str:
        """Generate prompt text using the provided issue and stubbed file content."""
        prompt_parts = []
        
        prompt_parts.append("# Code Implementation Task\n\n")
        prompt_parts.append("You are given a Python file with an incomplete implementation. ")
        prompt_parts.append("Your task is to implement the missing functionality as described in the issue below.\n\n")
        
        prompt_parts.append("## Issue\n\n")
        prompt_parts.append(issue)
        prompt_parts.append("\n\n")
        
        prompt_parts.append("## Current File Content\n\n")
        prompt_parts.append("```python\n")
        prompt_parts.append(stub_file)
        prompt_parts.append("\n```\n\n")
        
        prompt_parts.append("## Task\n\n")
        prompt_parts.append("Generate a code patch that implements the missing functionality. ")
        prompt_parts.append("The patch should be applicable to the file shown above.\n")
        
        return ''.join(prompt_parts)
    
    def _generate_task_from_record(
        self,
        entity_data: Dict[str, Any],
        issue_text: str,
        file_content: str,
        task_id: Optional[str] = None,
    ) -> Task:
        """Build a Task from entity metadata, issue text, and file content."""
        line_start, line_end = self._extract_line_span(entity_data)

        if task_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            task_id = f"task_{entity_data.get('name', 'entity')}_{timestamp}"

        stub_file = self._get_file_content_with_stub(entity_data, file_content)
        patch = self._generate_code_patch(stub_file, file_content)

        entity_info = {
            'name': entity_data.get('name'),
            'repo_name': entity_data.get('repo_name'),
            'file_path': entity_data.get('file_path'),
            'line_start': line_start,
            'line_end': line_end,
            'type': entity_data.get('type') or entity_data.get('entity_type') or 'unknown',
        }

        if self.config.include_entity_properties:
            if 'properties' in entity_data:
                entity_info['properties'] = entity_data['properties']
            if 'complexity' in entity_data:
                entity_info['complexity'] = entity_data.get('complexity')

        ground_truth = {
            'patch': patch,
            'original_code': entity_data.get('src_code') or entity_data.get('body_content'),
            'stub_code': entity_data.get('stub'),
            'file_with_stub': stub_file,
            'file_original': file_content,
        }

        metadata = None
        if self.config.include_metadata:
            metadata = {
                'generated_at': datetime.now().isoformat(),
                'generator_version': '1.0.0',
                'entity_lines': line_end - line_start + 1,
                'config': {
                    'issue_template': self.config.issue_template,
                    'include_context': self.config.include_context,
                },
            }

        prompt = self._generate_prompt(issue_text, stub_file)

        return Task(
            task_id=task_id,
            prompt=prompt,
            issue=issue_text,
            entity_info=entity_info,
            ground_truth=ground_truth,
            metadata=metadata,
        )
    
    def _load_entities_with_issues(self, entity_root: Path) -> List[Dict[str, Any]]:
        """Load entity metadata and corresponding issue text from disk."""
        records: List[Dict[str, Any]] = []
        entity_files = sorted(entity_root.rglob(self.config.entity_filename))

        if not entity_files:
            logger.warning("[TaskGenerator] No entity files found under %s", entity_root)
            return records

        for entity_file in entity_files:
            try:
                issue_path = entity_file.parent / self.config.issue_filename
                if not issue_path.exists():
                    logger.warning(
                        "[TaskGenerator] Missing issue file for %s (expected %s)",
                        entity_file,
                        issue_path,
                    )
                    continue

                with entity_file.open('r', encoding='utf-8') as f:
                    entity_data = json.load(f)

                issue_text = issue_path.read_text(encoding='utf-8')
                records.append({'entity': entity_data, 'issue': issue_text})
            except Exception as exc:
                logger.warning(
                    "[TaskGenerator] Failed to load entity/issue for %s: %s",
                    entity_file,
                    exc,
                )

        logger.info(
            "[TaskGenerator] Loaded %d entity/issue pairs from %s", len(records), entity_root
        )
        return records

    def _generate_tasks_from_records(
        self,
        records: List[Dict[str, Any]],
        file_cache: Optional[Dict[str, str]] = None,
    ) -> List[Task]:
        tasks: List[Task] = []
        cache: Dict[str, str] = file_cache or {}

        logger.info("[TaskGenerator] Generating tasks from %d entity/issue pairs", len(records))

        for record in records:
            entity_data = record.get('entity', {})
            issue_text = record.get('issue', '')

            file_path = entity_data.get('file_path')
            if not file_path:
                logger.warning("[TaskGenerator] Entity missing file_path, skipping")
                continue

            try:
                if file_path not in cache:
                    if Path(file_path).exists():
                        cache[file_path] = Path(file_path).read_text(encoding='utf-8')
                    else:
                        logger.warning(
                            "[TaskGenerator] File not found for entity %s: %s",
                            entity_data.get('name'),
                            file_path,
                        )
                        continue

                file_content = cache[file_path]
                task = self._generate_task_from_record(entity_data, issue_text, file_content)
                tasks.append(task)
            except Exception as exc:
                logger.warning(
                    "[TaskGenerator] Error generating task for %s: %s",
                    entity_data.get('name'),
                    exc,
                )

        logger.info("[TaskGenerator] Generated %d tasks", len(tasks))
        return tasks

    def generate_tasks_from_directory(self, entity_root: str | Path) -> List[Task]:
        """Load entities/issues from a directory and generate tasks for all of them."""
        records = self._load_entities_with_issues(Path(entity_root))
        return self._generate_tasks_from_records(records)
    
    def save_tasks(
        self,
        tasks: List[Task],
        filename: Optional[str] = None,
    ) -> Path:
        """
        Save tasks to file.
        
        Args:
            tasks: List of Task objects to save
            filename: Optional filename. If not provided, generates one based on timestamp.
            
        Returns:
            Path to saved file
        """
        if not tasks:
            logger.warning("[TaskGenerator] No tasks to save")
            return None
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            ext = 'jsonl' if self.config.save_format == 'jsonl' else 'json'
            filename = f"tasks_{timestamp}.{ext}"
        
        output_path = self.output_dir / filename
        
        logger.info(f"[TaskGenerator] Saving {len(tasks)} tasks to {output_path}")
        
        # Convert tasks to dict
        tasks_data = [task.to_dict() for task in tasks]
        
        # Save based on format
        if self.config.save_format == 'jsonl':
            with open(output_path, 'w', encoding='utf-8') as f:
                for task_data in tasks_data:
                    f.write(json.dumps(task_data, ensure_ascii=False) + '\n')
        else:  # json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(tasks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[TaskGenerator] Successfully saved tasks to {output_path}")
        
        # Save summary
        summary_path = self.output_dir / f"{filename.rsplit('.', 1)[0]}_summary.json"
        summary = {
            'total_tasks': len(tasks),
            'generated_at': datetime.now().isoformat(),
            'output_file': str(output_path),
            'task_ids': [task.task_id for task in tasks],
            'entity_types': {},
        }
        
        # Count entity types
        for task in tasks:
            entity_type = task.entity_info.get('type', 'unknown')
            summary['entity_types'][entity_type] = summary['entity_types'].get(entity_type, 0) + 1
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"[TaskGenerator] Saved summary to {summary_path}")
        
        return output_path
    
    def load_tasks(self, filepath: str | Path) -> List[Task]:
        """
        Load tasks from file.
        
        Args:
            filepath: Path to tasks file
            
        Returns:
            List of Task objects
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Tasks file not found: {filepath}")
        
        logger.info(f"[TaskGenerator] Loading tasks from {filepath}")
        
        tasks = []
        
        if filepath.suffix == '.jsonl':
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    task_data = json.loads(line)
                    tasks.append(Task(**task_data))
        else:  # json
            with open(filepath, 'r', encoding='utf-8') as f:
                tasks_data = json.load(f)
                for task_data in tasks_data:
                    tasks.append(Task(**task_data))
        
        logger.info(f"[TaskGenerator] Loaded {len(tasks)} tasks")
        return tasks


def main():
    """Main entry point for command-line usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Task Generator for Code Agent")
    parser.add_argument(
        "--config",
        type=str,
        default="examples/config/code_agent.yaml",
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--entity-dir",
        type=str,
        help="Directory containing extracted entities and generated issues",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output filename (optional)",
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
    
    # Load configuration
    try:
        generator = TaskGenerator.load_config_from_yaml(args.config)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}", exc_info=True)
        return 1
    
    entity_dir = Path(args.entity_dir or generator.config.entity_dir)
    if not entity_dir.exists():
        logger.error("Entity directory not found: %s", entity_dir)
        return 1

    try:
        tasks = generator.generate_tasks_from_directory(entity_dir)
        if not tasks:
            logger.warning("No tasks generated")
            return 0
        generator.save_tasks(tasks, filename=args.output)
    except Exception as e:
        logger.error(f"Error generating tasks: {e}", exc_info=True)
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
