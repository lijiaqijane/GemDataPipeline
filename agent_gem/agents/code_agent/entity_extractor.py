"""
Entity Extractor Module

This module provides functionality to extract code entities (classes, functions, etc.)
from source code files in GitHub repositories. Supports multiple programming languages
and provides configuration for filtering and extraction.
"""

from __future__ import annotations

import json
import logging
import os
import re
import ast
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .entity import CodeEntity, PythonEntity

logger = logging.getLogger(__name__)


@dataclass
class EntityExtractorConfig:
    """Configuration for entity extraction."""
    
    languages: List[str] = field(default_factory=lambda: ["Python"])
    max_entities_per_file: int = -1  # -1 means no limit
    max_entities_per_repo: int = 1000  # Maximum total entities per repository
    dirs_exclude: List[str] = field(default_factory=list)  # Directories to exclude
    dirs_include: List[str] = field(default_factory=list)  # Empty means include all
    exclude_test_files: bool = True  # Exclude test files and test directories
    exclude_hidden_files: bool = True  # Exclude files starting with .
    min_entity_line_count: int = 1  # Minimum lines for an entity
    max_entity_line_count: int = 500  # Maximum lines for an entity
    min_file_line_count: int = 1  # Minimum lines for a file to be processed
    max_file_line_count: int = -1  # Maximum lines for a file to be processed, -1 means no limit
    
    # Entity type filters
    include_functions: bool = True  # Include function entities
    include_classes: bool = True  # Include class entities
    
    # Entity name filters
    exclude_entity_names: List[str] = field(default_factory=list)  # Entity names to exclude (exact match)
    exclude_entity_patterns: List[str] = field(default_factory=list)  # Entity name patterns to exclude (regex)
    
    # Documentation requirement
    require_docstring: bool = False  # Require entities to have docstrings
    
    # Control flow requirements (empty list = no requirement)
    require_control_flows: List[str] = field(default_factory=list)  # e.g., ["has_if", "has_loop"]
    exclude_control_flows: List[str] = field(default_factory=list)  # e.g., ["has_exception"]
    
    # Code properties requirements
    require_properties: List[str] = field(default_factory=list)  # e.g., ["has_return", "has_function_call"]
    exclude_properties: List[str] = field(default_factory=list)  # e.g., ["has_decorator"]
    
    # Complexity filters
    min_complexity: int = -1  # Minimum cyclomatic complexity (-1 = no limit)
    max_complexity: int = -1  # Maximum cyclomatic complexity (-1 = no limit)


class EntityExtractor:
    """
    Extract code entities (functions, classes) from source code files.
    
    This class walks through repository directories and extracts code entities
    based on configurable criteria. Supports Python and other languages.
    
    Example:
        >>> config = EntityExtractorConfig()
        >>> extractor = EntityExtractor(config)
        >>> entities = extractor.extract_entities_from_repo("/path/to/repo")
        >>> extractor.save_entities(entities, "/path/to/output.json")
    """
    
    def __init__(self, config: Optional[EntityExtractorConfig] = None):
        """
        Initialize the entity extractor.
        
        Args:
            config: EntityExtractorConfig instance. If None, uses default config.
        """
        self.config = config or EntityExtractorConfig()
        
        # Language-to-method mapping
        self.get_entities_from_file = {
            ".py": self.get_entities_from_file_py,
        }
        self.exts: List[str] = list(self.get_entities_from_file.keys())

    
    @classmethod
    def load_config_from_yaml(cls, yaml_path: str | Path) -> "EntityExtractor":
        """
        Load configuration from YAML file and create EntityExtractor instance.
        
        Args:
            yaml_path: Path to the YAML configuration file
            
        Returns:
            EntityExtractor instance with loaded configuration
            
        Raises:
            FileNotFoundError: If the YAML file does not exist
            ValueError: If the configuration is invalid
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        if 'entity_extractor' not in config_data:
            logger.warning("No 'entity_extractor' section in config, using defaults")
            return cls()
        
        extractor_config = config_data['entity_extractor']
        
        config = EntityExtractorConfig(
            languages=extractor_config.get('languages', ['Python']),
            max_entities_per_file=extractor_config.get('max_entities_per_file', -1),
            max_entities_per_repo=extractor_config.get('max_entities_per_repo', 1000),
            dirs_exclude=extractor_config.get('dirs_exclude', [
                "test", "tests", "spec", "specs", "docs", "examples", "__pycache__", ".git"
            ]),
            dirs_include=extractor_config.get('dirs_include', []),
            exclude_test_files=extractor_config.get('exclude_test_files', True),
            exclude_hidden_files=extractor_config.get('exclude_hidden_files', True),
            min_entity_line_count=extractor_config.get('min_entity_line_count', 1),
            max_entity_line_count=extractor_config.get('max_entity_line_count', 500),
            min_file_line_count=extractor_config.get('min_file_line_count', 1),
            max_file_line_count=extractor_config.get('max_file_line_count', -1),
            # Entity type filters
            include_functions=extractor_config.get('include_functions', True),
            include_classes=extractor_config.get('include_classes', True),
            # Entity name filters
            exclude_entity_names=extractor_config.get('exclude_entity_names', []),
            exclude_entity_patterns=extractor_config.get('exclude_entity_patterns', []),
            # Documentation requirement
            require_docstring=extractor_config.get('require_docstring', False),
            # Control flow requirements
            require_control_flows=extractor_config.get('require_control_flows', []),
            exclude_control_flows=extractor_config.get('exclude_control_flows', []),
            # Code properties requirements
            require_properties=extractor_config.get('require_properties', []),
            exclude_properties=extractor_config.get('exclude_properties', []),
            # Complexity filters
            min_complexity=extractor_config.get('min_complexity', -1),
            max_complexity=extractor_config.get('max_complexity', -1),
        )
        
        return cls(config)
    
    def extract_entities(
        self,
        dir_path: str,
    ) -> List[CodeEntity]:
        """
        Extract entities (functions, classes, etc.) from all files in a directory.
        
        Args:
            dir_path: Path to the directory to scan
            
        Returns:
            List of CodeEntity objects containing entity information
        """
        entities = []
        file_count = 0
        error_count = 0
        
        logger.info(f"[EntityExtractor] Starting extraction from: {dir_path}")
        logger.info(f"[EntityExtractor] Config: max_entities_per_file={self.config.max_entities_per_file}, "
                   f"max_entities_per_repo={self.config.max_entities_per_repo}")
        logger.info(f"[EntityExtractor] Exclude dirs: {self.config.dirs_exclude}")
        if self.config.dirs_include:
            logger.info(f"[EntityExtractor] Include dirs: {self.config.dirs_include}")
        
        for root, dirs, files in os.walk(dir_path):
            # Skip excluded directories
            dirs[:] = [d for d in dirs if not self._should_skip_dir(d, root)]
            
            for file in files:
                # Check if total entities limit reached
                if self.config.max_entities_per_repo != -1 and len(entities) >= self.config.max_entities_per_repo:
                    logger.info(f"[EntityExtractor] Reached max entities per repo limit ({self.config.max_entities_per_repo})")
                    logger.info(f"[EntityExtractor] Extraction complete. Found {len(entities)} entities from {file_count} files")
                    return entities
                
                file_path = os.path.join(root, file)
                
                # Skip hidden files
                if self.config.exclude_hidden_files and os.path.basename(file).startswith('.'):
                    continue
                
                # Skip test files
                if self.config.exclude_test_files and self._is_test_path(root, file):
                    continue
                
                # Check file extension
                file_ext = Path(file_path).suffix
                if file_ext not in self.exts:
                    continue
                
                # Check if file is readable and count lines
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                        file_line_count = len(file_content.splitlines())
                    
                    # Check minimum file line count
                    if file_line_count < self.config.min_file_line_count:
                        logger.debug(f"[EntityExtractor] Skipping file {file_path}: {file_line_count} lines < min {self.config.min_file_line_count}")
                        continue
                    
                    # Check maximum file line count
                    if self.config.max_file_line_count != -1 and file_line_count > self.config.max_file_line_count:
                        logger.debug(f"[EntityExtractor] Skipping file {file_path}: {file_line_count} lines > max {self.config.max_file_line_count}")
                        continue
                        
                except Exception as e:
                    logger.debug(f"[EntityExtractor] Skipping unreadable file {file_path}: {e}")
                    error_count += 1
                    continue
                
                # Extract entities from file
                try:
                    file_entities_before = len(entities)
                    self.get_entities_from_file[file_ext](
                        entities,
                        file_path,
                        self.config.max_entities_per_file
                    )
                    file_entities_added = len(entities) - file_entities_before
                    if file_entities_added > 0:
                        logger.debug(f"[EntityExtractor] {file_path}: {file_entities_added} entities")
                    file_count += 1
                except Exception as e:
                    logger.warning(f"[EntityExtractor] Error extracting from {file_path}: {e}")
                    error_count += 1
                    continue
        
        logger.info(f"[EntityExtractor] Extraction complete. Found {len(entities)} entities from {file_count} files "
                   f"({error_count} errors)")
        return entities
    
    def extract_entities_from_repo(self, repo_path: str) -> Dict[str, Any]:
        """
        Extract entities from an entire repository.
        
        Args:
            repo_path: Path to the repository root
            
        Returns:
            Dictionary containing extraction metadata and entities
        """
        repo_path = Path(repo_path)
        entities = self.extract_entities(str(repo_path))
        
        return {
            'repo_path': str(repo_path),
            'repo_name': repo_path.name,
            'extraction_time': datetime.now().isoformat(),
            'total_entities': len(entities),
            'entities': entities,
            'config': {
                'max_entities_per_file': self.config.max_entities_per_file,
                'max_entities_per_repo': self.config.max_entities_per_repo,
                'exclude_test_files': self.config.exclude_test_files,
            }
        }
    
    def _should_skip_dir(self, dir_name: str, full_path: str) -> bool:
        """Check if a directory should be skipped."""
        # Check exclude list
        if any(excluded in full_path for excluded in self.config.dirs_exclude):
            return True
        
        # Check include list (if specified, skip dirs not in include list)
        if self.config.dirs_include:
            if not any(included in full_path for included in self.config.dirs_include):
                return True
        
        # Skip hidden directories
        if self.config.exclude_hidden_files and dir_name.startswith('.'):
            return True
        
        return False
    
    def _is_test_path(self, root: str, file: str) -> bool:
        """Check whether the file path corresponds to a testing related file"""
        if len(self.exts) > 1 and not any([file.endswith(ext) for ext in self.exts]):
            return False
        if file.lower().startswith("test") or file.rsplit(".", 1)[0].endswith("test"):
            return True
        dirs = root.split("/")
        if any([x in dirs for x in ["tests", "test", "specs"]]):
            return True
        return False
    
    def _should_include_entity(self, entity: CodeEntity) -> bool:
        """Check if an entity should be included based on filtering criteria."""
        from .entity import CodeProperty
        
        # Check entity type filters
        if isinstance(entity.node, ast.FunctionDef) and not self.config.include_functions:
            logger.debug(f"[EntityExtractor] Excluding function: {entity.name}")
            return False
        if isinstance(entity.node, ast.ClassDef) and not self.config.include_classes:
            logger.debug(f"[EntityExtractor] Excluding class: {entity.name}")
            return False
        
        # Check entity name exclusions (exact match)
        if entity.name in self.config.exclude_entity_names:
            logger.debug(f"[EntityExtractor] Excluding entity by name: {entity.name}")
            return False
        
        # Check entity name pattern exclusions (regex)
        for pattern in self.config.exclude_entity_patterns:
            if re.match(pattern, entity.name):
                logger.debug(f"[EntityExtractor] Excluding entity by pattern '{pattern}': {entity.name}")
                return False
        
        # Check docstring requirement
        if self.config.require_docstring:
            if not self._has_docstring(entity.node):
                logger.debug(f"[EntityExtractor] Excluding entity without docstring: {entity.name}")
                return False
        
        # Check line count
        entity_line_count = entity.line_end - entity.line_start + 1
        if entity_line_count < self.config.min_entity_line_count:
            logger.debug(f"[EntityExtractor] Excluding entity {entity.name}: {entity_line_count} lines < min {self.config.min_entity_line_count}")
            return False
        if entity_line_count > self.config.max_entity_line_count:
            logger.debug(f"[EntityExtractor] Excluding entity {entity.name}: {entity_line_count} lines > max {self.config.max_entity_line_count}")
            return False
        
        # Check required control flows
        for required_flow in self.config.require_control_flows:
            try:
                prop = CodeProperty(required_flow)
                if prop not in entity._tags:
                    logger.debug(f"[EntityExtractor] Excluding entity {entity.name}: missing required control flow '{required_flow}'")
                    return False
            except ValueError:
                logger.warning(f"[EntityExtractor] Invalid control flow property: {required_flow}")
        
        # Check excluded control flows
        for excluded_flow in self.config.exclude_control_flows:
            try:
                prop = CodeProperty(excluded_flow)
                if prop in entity._tags:
                    logger.debug(f"[EntityExtractor] Excluding entity {entity.name}: has excluded control flow '{excluded_flow}'")
                    return False
            except ValueError:
                logger.warning(f"[EntityExtractor] Invalid control flow property: {excluded_flow}")
        
        # Check required properties
        for required_prop in self.config.require_properties:
            try:
                prop = CodeProperty(required_prop)
                if prop not in entity._tags:
                    logger.debug(f"[EntityExtractor] Excluding entity {entity.name}: missing required property '{required_prop}'")
                    return False
            except ValueError:
                logger.warning(f"[EntityExtractor] Invalid code property: {required_prop}")
        
        # Check excluded properties
        for excluded_prop in self.config.exclude_properties:
            try:
                prop = CodeProperty(excluded_prop)
                if prop in entity._tags:
                    logger.debug(f"[EntityExtractor] Excluding entity {entity.name}: has excluded property '{excluded_prop}'")
                    return False
            except ValueError:
                logger.warning(f"[EntityExtractor] Invalid code property: {excluded_prop}")
        
        # Check complexity
        if self.config.min_complexity != -1 and entity.complexity < self.config.min_complexity:
            logger.debug(f"[EntityExtractor] Excluding entity {entity.name}: complexity {entity.complexity} < min {self.config.min_complexity}")
            return False
        if self.config.max_complexity != -1 and entity.complexity > self.config.max_complexity:
            logger.debug(f"[EntityExtractor] Excluding entity {entity.name}: complexity {entity.complexity} > max {self.config.max_complexity}")
            return False
        
        return True
    
    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if an AST node has a docstring."""
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return False
        if not node.body:
            return False
        first_stmt = node.body[0]
        return (
            isinstance(first_stmt, ast.Expr)
            and isinstance(first_stmt.value, ast.Constant)
            and isinstance(first_stmt.value.value, str)
        )
    
    def get_entities_from_file_py(
            self,
            entities: list[PythonEntity],
            file_path: str,
            max_entities: int = -1,
        ) -> None:
        try:
            file_content = open(file_path, "r", encoding="utf8").read()
            tree = ast.parse(file_content, filename=file_path)
        except SyntaxError:
            return

        for node in ast.walk(tree):
            if not any([isinstance(node, x) for x in (ast.ClassDef, ast.FunctionDef)]):
                continue
            
            entity = self._build_entity(node, file_content, file_path)
            
            # Apply entity filters
            if not self._should_include_entity(entity):
                continue
            
            entities.append(entity)
            if max_entities != -1 and len(entities) >= max_entities:
                return
    

    def _build_entity(self, node: ast.AST, file_content: str, file_path: str) -> PythonEntity:
        """Turns an AST node into a PythonEntity object."""
        start_line = node.lineno  # type: ignore[attr-defined]
        end_line = (
            node.end_lineno if hasattr(node, "end_lineno") else None  # type: ignore[attr-defined]
        )

        if end_line is None:
            # Calculate end line manually if not available (older Python versions)
            end_line = (
                start_line
                + len(
                    ast.get_source_segment(file_content, node).splitlines()  # type: ignore[attr-defined]
                )
                - 1
            )

        src_code = ast.get_source_segment(file_content, node)

        # Get the line content for the source definition
        source_line = file_content.splitlines()[start_line - 1]
        leading_whitespace = len(source_line) - len(source_line.lstrip())

        # Determine the number of spaces per tab
        indent_size = 4  # Default fallback
        if "\t" in file_content:
            indent_size = source_line.expandtabs().index(source_line.lstrip())

        # Calculate indentation level
        indent_level = leading_whitespace // indent_size if leading_whitespace > 0 else 0

        # Remove indentation from source source code
        assert src_code is not None
        lines = src_code.splitlines()
        dedented_src_code = [lines[0]]
        for line in lines[1:]:
            # Strip leading spaces equal to indent_level * indent_size
            dedented_src_code.append(line[indent_level * indent_size :])
        src_code = "\n".join(dedented_src_code)

        return PythonEntity(
            file_path=file_path,
            indent_level=indent_level,
            indent_size=indent_size,
            line_end=end_line,
            line_start=start_line,
            node=node,
            src_code=src_code,
        )


def main():
    """Main entry point for command-line usage."""
    import argparse




if __name__ == "__main__":
    exit(main())
