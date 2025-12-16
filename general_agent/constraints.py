"""Code constraint system: enforces constraints through interfaces and AST analysis, not through prompts.

According to paper requirements:
1. Solution functions can only call tool functions or perform logical calculations, cannot directly access database
2. Tool functions can access database and call other tool functions
3. Verification functions can access database and all information
"""

from __future__ import annotations

import ast
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Protocol

from .database import LocalDatabase

logger = logging.getLogger(__name__)


class ToolCallable(Protocol):
    """Tool function protocol: can be called by solution functions and verification functions."""
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Call tool function."""
        ...


@dataclass
class SolutionContext:
    """Execution context for solution functions: can only access tools, cannot access database.
    
    According to paper constraints, solution functions can only:
    - Call tool functions
    - Perform logical calculations (conditionals, loops, etc.)
    
    Solution functions cannot:
    - Directly access database
    - Call other non-tool functions
    """
    
    tools: Dict[str, ToolCallable]
    
    def get_tool(self, name: str) -> ToolCallable:
        """Get tool function."""
        if name not in self.tools:
            raise AttributeError(f"Tool '{name}' not available in solution context. Available: {list(self.tools.keys())}")
        return self.tools[name]
    
    def __getitem__(self, name: str) -> ToolCallable:
        """Support tools['name'] syntax."""
        return self.get_tool(name)
    
    def __getattr__(self, name: str) -> ToolCallable:
        """Support tools.name syntax."""
        return self.get_tool(name)
    
    def __contains__(self, name: str) -> bool:
        """Check if tool exists."""
        return name in self.tools
    
    def keys(self):
        """Return all tool names."""
        return self.tools.keys()


@dataclass
class ToolContext:
    """Execution context for tool functions: can access database and call other tool functions.
    
    According to paper constraints, tool functions can:
    - Access database
    - Call other tool functions
    - Must return verifiable results
    """
    
    db: LocalDatabase
    tools: Dict[str, ToolCallable]
    
    def get_tool(self, name: str) -> ToolCallable:
        """Get other tool function."""
        if name not in self.tools:
            raise AttributeError(f"Tool '{name}' not available. Available: {list(self.tools.keys())}")
        return self.tools[name]
    
    def query_db(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """Query database."""
        return self.db.query(key, value)
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        """Get all database records."""
        return self.db.records


@dataclass
class VerificationContext:
    """Execution context for verification functions: can access database and all information.
    
    According to paper constraints, verification functions can:
    - Access database
    - Access all tools
    - Access all information
    """
    
    db: LocalDatabase
    tools: Dict[str, ToolCallable]
    answer: Any
    
    def get_tool(self, name: str) -> ToolCallable:
        """Get tool function."""
        if name not in self.tools:
            raise AttributeError(f"Tool '{name}' not available. Available: {list(self.tools.keys())}")
        return self.tools[name]
    
    def query_db(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """Query database."""
        return self.db.query(key, value)
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        """Get all database records."""
        return self.db.records


class CodeValidator:
    """Validates code compliance with constraints using AST analysis."""
    
    @staticmethod
    def validate_solution_code(code: str) -> tuple[bool, str]:
        """Validate if solution function code complies with constraints.
        
        Constraints:
        1. Must define solve(tools) function
        2. Cannot import modules
        3. Cannot define other functions (only solve function allowed)
        4. Cannot directly access database (checked via variable names)
        5. Must call tools
        
        Returns:
            (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Code is empty"
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check 1: Must define solve function
        has_solve = False
        solve_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "solve":
                has_solve = True
                solve_node = node
                break
        
        if not has_solve:
            return False, "Missing 'def solve(tools)' function definition"
        
        # Check 2: Cannot import modules
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Build string representation of import statement (compatible with Python 3.8+)
                if isinstance(node, ast.Import):
                    module_names = ", ".join([alias.name for alias in node.names])
                    import_str = f"import {module_names}"
                else:  # ImportFrom
                    module = node.module or ""
                    names = ", ".join([alias.name for alias in node.names]) if node.names else "*"
                    import_str = f"from {module} import {names}"
                return False, f"Solution code cannot import modules: {import_str}"
        
        # Check 3: Cannot define other functions (only solve function allowed)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name != "solve":
                return False, f"Solution code can only define solve function, cannot define other functions: {node.name}"
        
        # Check 4: Cannot directly access database (checked via variable names and attribute access)
        # Use NodeVisitor to correctly check node context
        class DatabaseAccessChecker(ast.NodeVisitor):
            def __init__(self):
                self.violations = []
                self.forbidden_names = {"db", "database", "ctx"}
            
            def visit_Name(self, node: ast.Name):
                if node.id in self.forbidden_names:
                    # Check if used in attribute access, subscript, or call
                    # Since AST has no parent, we check surrounding context
                    # If name appears on left side of assignment, might be definition, not access
                    # Simplified handling: if name appears in expression, treat as access
                    self.violations.append(f"Solution code cannot directly access database variable: {node.id}")
                self.generic_visit(node)
            
            def visit_Attribute(self, node: ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id in self.forbidden_names:
                    if node.attr in {"db", "database", "records"}:
                        self.violations.append(f"Solution code cannot directly access database: {node.value.id}.{node.attr}")
                self.generic_visit(node)
        
        checker = DatabaseAccessChecker()
        checker.visit(tree)
        if checker.violations:
            return False, checker.violations[0]
        
        # Check 5: Must call tools (in solve function)
        if solve_node:
            has_tool_call = False
            for node in ast.walk(solve_node):
                # Check tools['name'] or tools.name or tools.name() calls
                if isinstance(node, ast.Subscript):
                    if isinstance(node.value, ast.Name) and node.value.id == "tools":
                        has_tool_call = True
                        break
                elif isinstance(node, ast.Attribute):
                    if isinstance(node.value, ast.Name) and node.value.id == "tools":
                        has_tool_call = True
                        break
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, (ast.Attribute, ast.Subscript)):
                        func_obj = node.func
                        if isinstance(func_obj, ast.Attribute):
                            if isinstance(func_obj.value, ast.Name) and func_obj.value.id == "tools":
                                has_tool_call = True
                                break
                        elif isinstance(func_obj, ast.Subscript):
                            if isinstance(func_obj.value, ast.Name) and func_obj.value.id == "tools":
                                has_tool_call = True
                                break
            
            if not has_tool_call:
                return False, "Solution code must call at least one tool function"
        
        return True, ""
    
    @staticmethod
    def validate_verification_code(code: str) -> tuple[bool, str]:
        """Validate if verification function code complies with constraints.
        
        Constraints:
        1. Must define verify(tools, answer) function
        2. Can import modules (if needed)
        3. Can define helper functions
        4. Can access database (via tools or direct access)
        
        Returns:
            (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Code is empty"
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # Check 1: Must define verify function
        has_verify = False
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "verify":
                has_verify = True
                break
        
        if not has_verify:
            return False, "Missing 'def verify(tools, answer)' function definition"
        
        return True, ""
    
    @staticmethod
    def extract_tool_calls(code: str) -> set[str]:
        """Extract all tool call names from code."""
        tool_calls = set()
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return tool_calls
        
        for node in ast.walk(tree):
            # Check tools['name'] calls
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == "tools":
                    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                        tool_calls.add(node.slice.value)
                    elif isinstance(node.slice, ast.Str):  # Python < 3.8
                        tool_calls.add(node.slice.s)
            
            # Check tools.name calls
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "tools":
                    tool_calls.add(node.attr)
            
            # Check tools['name']() or tools.name() calls
            elif isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Attribute):
                    if isinstance(func.value, ast.Name) and func.value.id == "tools":
                        tool_calls.add(func.attr)
                elif isinstance(func, ast.Subscript):
                    if isinstance(func.value, ast.Name) and func.value.id == "tools":
                        if isinstance(func.slice, ast.Constant) and isinstance(func.slice.value, str):
                            tool_calls.add(func.slice.value)
                        elif isinstance(func.slice, ast.Str):  # Python < 3.8
                            tool_calls.add(func.slice.s)
        
        # Remove dict methods
        dict_methods = {"keys", "values", "items", "get", "pop", "update", "clear", "copy"}
        tool_calls -= dict_methods
        
        return tool_calls

