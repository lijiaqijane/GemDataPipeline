from __future__ import annotations

import ast
import astor

from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

TODO_REWRITE = "TODO: Implement this function"

class CodeProperty(Enum):
    # Core entity types
    IS_FUNCTION = "is_function"
    IS_CLASS = "is_class"

    # Control flow
    HAS_EXCEPTION = "has_exception"
    HAS_IF = "has_if"
    HAS_IF_ELSE = "has_if_else"
    HAS_LOOP = "has_loop"
    HAS_SWITCH = "has_switch"  # Added for switch statements

    # Operations
    HAS_ARITHMETIC = "has_arithmetic"
    HAS_ASSIGNMENT = "has_assignment"
    HAS_DECORATOR = "has_decorator"
    HAS_FUNCTION_CALL = "has_function_call"
    HAS_IMPORT = "has_import"
    HAS_LAMBDA = "has_lambda"
    HAS_LIST_COMPREHENSION = "has_list_comprehension"
    HAS_LIST_INDEXING = "has_list_indexing"
    HAS_OFF_BY_ONE = "has_off_by_one"
    HAS_PARENT = "has_parent"
    HAS_RETURN = "has_return"
    HAS_WRAPPER = "has_wrapper"

    # Operations by type
    HAS_BINARY_OP = "has_binary_op"
    HAS_BOOL_OP = "has_bool_op"
    HAS_UNARY_OP = "has_unary_op"

class CodeEntityMeta(type):
    def __new__(mcs, name, bases, namespace):
        # Create properties for all enum values
        for prop in CodeProperty:
            namespace[prop.value] = property(lambda self, p=prop: p in self._tags)
        return super().__new__(mcs, name, bases, namespace)

@dataclass
class CodeEntity(metaclass=CodeEntityMeta):
    """Data class to hold information about a code entity (e.g. function, class)."""

    file_path: str
    indent_level: int
    indent_size: int
    line_end: int
    line_start: int
    node: Any
    src_code: Any

    def __post_init__(self):
        self._tags: set[CodeProperty] = set()
        self._analyze_properties()

    def _analyze_properties(self):
        """To be implemented by language-specific classes"""
        pass

    @property
    def complexity(self) -> int:
        """Get the complexity of the code entity."""
        return -1  # Default value = no notion of complexity implemented

    @property
    def ext(self) -> str:
        if isinstance(self.file_path, Path):
            self.file_path = str(self.file_path)
        return self.file_path.rsplit(".", 1)[-1].lower()

    @property
    @abstractmethod
    def name(self) -> str:
        """Get the name of the code entity."""
        pass

    @property
    @abstractmethod
    def signature(self) -> str:
        """Get the signature of the code entity."""
        pass

    @property
    @abstractmethod
    def stub(self) -> str:
        """Get stub (code with implementation removed) for the code entity."""
        pass

@dataclass
class PythonEntity(CodeEntity):
    signature_content: str = ""
    signature_start_line: int = -1
    signature_end_line: int = -1
    body_content: str = ""
    body_start_line: int = -1
    body_end_line: int = -1

    def _analyze_properties(self):
        node = self.node

        # Core entity types
        if isinstance(node, ast.FunctionDef):
            self._tags.add(CodeProperty.IS_FUNCTION)
        elif isinstance(node, ast.ClassDef):
            self._tags.add(CodeProperty.IS_CLASS)

        # Control flow
        if any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_LOOP)
        if any(isinstance(n, ast.If) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_IF)
            if any(n.orelse for n in ast.walk(node) if isinstance(n, ast.If)):
                self._tags.add(CodeProperty.HAS_IF_ELSE)
        if any(isinstance(n, ast.Try) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_EXCEPTION)

        # Operations
        if any(isinstance(n, ast.Subscript) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_LIST_INDEXING)
        if any(isinstance(n, ast.Call) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_FUNCTION_CALL)
        if any(isinstance(n, ast.Return) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_RETURN)
        if any(isinstance(n, ast.ListComp) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_LIST_COMPREHENSION)
        if any(isinstance(n, (ast.Import, ast.ImportFrom)) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_IMPORT)
        if any(isinstance(n, ast.Assign) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_ASSIGNMENT)
        if any(isinstance(n, ast.Lambda) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_LAMBDA)
        if any(isinstance(n, (ast.BinOp, ast.UnaryOp)) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_ARITHMETIC)
        if any(
            isinstance(n, ast.FunctionDef) and n.decorator_list for n in ast.walk(node)
        ):
            self._tags.add(CodeProperty.HAS_DECORATOR)
        if any(isinstance(n, (ast.Try, ast.With)) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_WRAPPER)
        if any(isinstance(n, ast.ClassDef) and n.bases for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_PARENT)

        # Operations by type
        if any(isinstance(n, ast.BinOp) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_BINARY_OP)
        if any(isinstance(n, ast.BoolOp) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_BOOL_OP)
        if any(isinstance(n, ast.UnaryOp) for n in ast.walk(node)):
            self._tags.add(CodeProperty.HAS_UNARY_OP)

        # Special cases
        if any(
            isinstance(n, ast.Compare)
            and len(n.ops) == 1
            and n.ops[0].__class__.__name__ in ["Lt", "Gt", "LtE", "GtE"]
            for n in ast.walk(node)
        ):
            self._tags.add(CodeProperty.HAS_OFF_BY_ONE)

    @property
    def complexity(self) -> int:
        """
        Simple way of calculating the complexity of a function.
        Complexity starts at 1 and increases for each decision point:
        - if/elif/else statements
        - for/while loops
        - and/or operators
        - except clauses
        - boolean operators
        """
        complexity = 1  # Base complexity

        for n in ast.walk(self.node):
            # Decision points
            if isinstance(n, (ast.If, ast.While, ast.For)):
                complexity += 1
            # Boolean operators
            elif isinstance(n, ast.BoolOp):
                complexity += len(n.values) - 1
            # Exception handling
            elif isinstance(n, ast.Try):
                complexity += len(n.handlers)
            # Comparison operators
            elif isinstance(n, ast.Compare):
                complexity += len(n.ops)

        return complexity

    @property
    def name(self):
        return self.node.name

    @property
    def signature(self):
        if isinstance(self.node, ast.ClassDef):
            return f"class {self.node.name}:"
        elif isinstance(self.node, ast.FunctionDef):
            args = [ast.unparse(arg) for arg in self.node.args.args]
            args_str = ", ".join(args)
            return f"def {self.node.name}({args_str})"

    @property
    def stub(self):
        src_code = self.src_code
        tree = ast.parse(src_code)

        class FunctionBodyStripper(ast.NodeTransformer):
            def visit_FunctionDef(self, node):
                # Keep the original arguments and decorator list
                new_node = ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=[],  # Empty body initially
                    decorator_list=node.decorator_list,
                    returns=node.returns,
                    type_params=getattr(node, "type_params", None),  # For Python 3.12+
                )

                # Add docstring if it exists
                if (
                    node.body
                    and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                ):
                    new_node.body.append(node.body[0])

                # Add a comment indicating to implement this function
                new_node.body.append(ast.Expr(ast.Constant(TODO_REWRITE)))

                # Add a 'pass' statement after the docstring
                new_node.body.append(ast.Pass())

                return new_node

        stripped_tree = FunctionBodyStripper().visit(tree)
        ast.fix_missing_locations(stripped_tree)
        return astor.to_source(stripped_tree).strip()