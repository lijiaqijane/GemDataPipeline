from __future__ import annotations

import ast
import builtins
import re
from typing import Iterable, Set

from .task_schema import TaskPackage


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
                if isinstance(node, ast.Import):
                    module_names = ", ".join([alias.name for alias in node.names])
                    import_str = f"import {module_names}"
                else:
                    module = node.module or ""
                    names = ", ".join([alias.name for alias in node.names]) if node.names else "*"
                    import_str = f"from {module} import {names}"
                return False, f"Solution code cannot import modules: {import_str}"

        # Check 3: Cannot define other functions (only solve function allowed)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name != "solve":
                return False, f"Solution code can only define solve function, cannot define other functions: {node.name}"

        # Check 4: Cannot directly access database (checked via variable names and attribute access)
        class DatabaseAccessChecker(ast.NodeVisitor):
            def __init__(self):
                self.violations = []
                self.forbidden_names = {"db", "database", "ctx"}

            def visit_Name(self, node: ast.Name):
                if node.id in self.forbidden_names:
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

        # Check 5: Only allow calling tools[...] / tools.name(...) or a small whitelist of pure builtins.
        allowed_builtins = {
            "len",
            "range",
            "min",
            "max",
            "sum",
            "any",
            "all",
            "sorted",
            "enumerate",
            "bool",
            "int",
            "float",
            "str",
            "list",
            "dict",
            "set",
            "isinstance",
            "hasattr",
            "getattr",
            "type",
            "zip",
            "map",
            "filter",
            "reversed",
            "iter",
            "next",
        }

        class CallChecker(ast.NodeVisitor):
            def __init__(self) -> None:
                self.violations: list[str] = []
                self.has_tool_call = False

            def visit_Lambda(self, node: ast.Lambda) -> None:
                self.violations.append("Solution code cannot define lambda expressions")

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                # tools['name'](...) or tools.name(...)
                if isinstance(func, (ast.Attribute, ast.Subscript)):
                    target = func
                    if isinstance(target, ast.Attribute):
                        if isinstance(target.value, ast.Name) and target.value.id == "tools":
                            self.has_tool_call = True
                            self.generic_visit(node)
                            return
                    elif isinstance(target, ast.Subscript):
                        if isinstance(target.value, ast.Name) and target.value.id == "tools":
                            self.has_tool_call = True
                            self.generic_visit(node)
                            return

                    # Any other attribute/subscript call is disallowed (e.g., obj.method(), arr[idx]())
                    self.violations.append(
                        "Solution code may only call tools[...] or tools.name(...); "
                        "other method or indexed calls are not allowed"
                    )
                    self.generic_visit(node)
                    return

                # Direct name calls – allow only a safe builtin whitelist
                if isinstance(func, ast.Name):
                    name = func.id
                    if name not in allowed_builtins:
                        self.violations.append(
                            f"Solution code cannot call function '{name}' directly; "
                            "only tools[...] / tools.name(...) and simple builtins are allowed"
                        )
                self.generic_visit(node)

        if solve_node:
            call_checker = CallChecker()
            call_checker.visit(solve_node)
            if call_checker.violations:
                return False, call_checker.violations[0]
            if not call_checker.has_tool_call:
                return False, "Solution code must call at least one tool function via tools[...] or tools.name(...)"

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
        verify_node: ast.FunctionDef | None = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "verify":
                verify_node = node
                break

        if verify_node is None:
            return False, "Missing 'def verify(tools, answer)' function definition"

        # Check 2: avoid trivial always-true verifier
        def _returns_constant_true(fn: ast.FunctionDef) -> bool:
            returns = [
                n
                for n in ast.walk(fn)
                if isinstance(n, ast.Return)
                and isinstance(n.value, ast.Constant)
                and n.value.value is True
            ]
            # If every return is literal True and no control flow, treat as trivial
            return bool(returns) and all(
                isinstance(n, (ast.Return, ast.Expr)) for n in fn.body
            )

        if _returns_constant_true(verify_node):
            return (
                False,
                "Verification code must perform checks; trivial 'return True' detected",
            )

        return True, ""

    @staticmethod
    def extract_tool_calls(code: str) -> set[str]:
        """Extract all tool call names from code."""
        tool_calls = set()

        if not code or not isinstance(code, str):
            return tool_calls

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
                    elif isinstance(node.slice, ast.Str):
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
                        elif isinstance(func.slice, ast.Str):
                            tool_calls.add(func.slice.s)

        # Remove dict methods
        dict_methods = {"keys", "values", "items", "get", "pop", "update", "clear", "copy"}
        tool_calls -= dict_methods

        return tool_calls


def validate_task_package(package: TaskPackage) -> TaskPackage:
    """Lightweight validation guardrails for generated tasks."""
    task = package.task
    if not task.tool_set:
        package.validated = False
        package.validation_reason = f"Task '{task.task_title}' must declare at least one tool."

    if len(task.task_content.split()) < 8:
        package.validated = False
        package.validation_reason = f"Task '{task.task_title}' lacks sufficient detail."

    if not _looks_runnable(package.solution):
        package.validated = False
        package.validation_reason = f"Solution for '{task.task_title}' is not runnable."

    if "verify" not in package.verification:
        package.validated = False
        package.validation_reason = f"Verification for '{task.task_title}' must define a check."

    return package


def _looks_runnable(code: str) -> bool:
    banned: Set[str] = {"rm -rf", "shutdown", ":(){:|:&};:"}
    lowered = code.lower()
    return not any(token in lowered for token in banned)
