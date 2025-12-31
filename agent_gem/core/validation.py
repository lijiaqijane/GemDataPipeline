from __future__ import annotations

import ast
import builtins
import re
from typing import Iterable, Set

from .task_schema import TaskPackage


class CodeValidator:
    """Validates code compliance with constraints using AST analysis."""

    _SAFE_BUILTINS: Set[str] = {
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "iter",
        "isinstance",
        "len",
        "list",
        "max",
        "min",
        "next",
        "range",
        "round",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    }
    _SAFE_IMPORTS: Set[str] = {"re", "math"}

    @staticmethod
    def validate_solution_code(code: str) -> tuple[bool, str]:
        """Validate if solution function code complies with constraints.

        Constraints:
        1. Must define solve(tools) function
        2. Can only import safe modules (pure logic helpers)
        3. Cannot define other functions (only solve function allowed)
        4. Cannot define classes
        5. Cannot directly access database (checked via variable names)
        6. Must call tools
        7. Only tool calls + safe builtin calls (no file/network access)

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

        allowed_import_names: Set[str] = set()
        # Check 2: Only allow safe module imports
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.Import):
                    module_names = []
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        module_names.append(alias.name)
                        if top not in CodeValidator._SAFE_IMPORTS:
                            return (
                                False,
                                f"Solution code cannot import modules: import {alias.name}; "
                                f"allowed={sorted(CodeValidator._SAFE_IMPORTS)}",
                            )
                        allowed_import_names.add(alias.asname or top)
                    import_str = f"import {', '.join(module_names)}"
                else:
                    module = node.module or ""
                    if node.level and node.level > 0:
                        return (
                            False,
                            f"Solution code cannot use relative imports: from .{module} import ...",
                        )
                    top = module.split(".")[0] if module else ""
                    if top not in CodeValidator._SAFE_IMPORTS:
                        names = ", ".join([alias.name for alias in node.names]) if node.names else "*"
                        import_str = f"from {module} import {names}"
                        return (
                            False,
                            f"Solution code cannot import modules: {import_str}; "
                            f"allowed={sorted(CodeValidator._SAFE_IMPORTS)}",
                        )
                    for alias in node.names or []:
                        allowed_import_names.add(alias.asname or alias.name)

        # Check 3: Cannot define other functions (only solve function allowed)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name != "solve":
                return False, f"Solution code can only define solve function, cannot define other functions: {node.name}"

        # Check 4: Cannot define classes
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                return False, f"Solution code cannot define classes: {node.name}"

        # Check 5: Cannot directly access database (checked via variable names and attribute access)
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

        class CallChecker(ast.NodeVisitor):
            """Ensure tool calls + safe builtins only (no file/network access helpers)."""

            def __init__(self) -> None:
                self.has_tool_call = False
                self.has_lambda = False
                self.invalid_call: str | None = None

            def visit_Lambda(self, node: ast.Lambda) -> None:
                # Lambda functions are allowed as they are safe
                self.generic_visit(node)

            def visit_Call(self, node: ast.Call) -> None:
                func = node.func
                if isinstance(func, ast.Name) and func.id == "__import__":
                    raise StopIteration("Solution code cannot use __import__")
                # tools['name'](...) or tools.name(...)
                if isinstance(func, (ast.Attribute, ast.Subscript)):
                    target = func
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name) and target.value.id == "tools":
                        self.has_tool_call = True
                    elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == "tools":
                        self.has_tool_call = True
                    elif isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
                        if target.value.id in {"__builtins__", "builtins"}:
                            self.invalid_call = "builtins_subscript"
                elif isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id in {"__builtins__", "builtins"}:
                        self.invalid_call = f"{func.value.id}.{func.attr}"
                elif isinstance(func, ast.Name):
                    if func.id == "tools":
                        # Direct tools(...) should be treated as a tool call as well.
                        self.has_tool_call = True
                    elif func.id not in CodeValidator._SAFE_BUILTINS and func.id not in allowed_import_names:
                        self.invalid_call = func.id

                if self.invalid_call:
                    return
                self.generic_visit(node)

        if solve_node:
            call_checker = CallChecker()
            try:
                call_checker.visit(solve_node)
            except StopIteration as e:
                return False, str(e)
            # Lambda functions are now allowed
            if call_checker.invalid_call:
                return False, f"Solution code cannot call '{call_checker.invalid_call}'; only tools and safe builtins are allowed"
            if not call_checker.has_tool_call:
                return False, "Solution code must call at least one tool function via tools[...] or tools.name(...)"

            def _is_stringish(expr: ast.AST) -> bool:
                if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
                    return True
                if isinstance(expr, ast.JoinedStr):
                    return True
                if isinstance(expr, ast.Call):
                    func = expr.func
                    if isinstance(func, ast.Name) and func.id in {"str", "repr"}:
                        return True
                    if isinstance(func, ast.Attribute):
                        if isinstance(func.value, ast.Name) and func.value.id == "json" and func.attr in {"dumps", "dump"}:
                            return True
                return False

            def _is_submit_result_call(node: ast.Call) -> bool:
                func = node.func
                if isinstance(func, ast.Attribute):
                    return isinstance(func.value, ast.Name) and func.value.id == "tools" and func.attr == "submit_result"
                if isinstance(func, ast.Subscript):
                    if isinstance(func.value, ast.Name) and func.value.id == "tools":
                        if isinstance(func.slice, ast.Constant) and func.slice.value == "submit_result":
                            return True
                        if isinstance(func.slice, ast.Str) and func.slice.s == "submit_result":
                            return True
                return False

            class SubmitResultArgChecker(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.invalid_reason: str | None = None

                def visit_Call(self, node: ast.Call) -> None:
                    if self.invalid_reason:
                        return
                    if _is_submit_result_call(node):
                        if not node.args and not node.keywords:
                            self.invalid_reason = "submit_result must receive the answer dict (no empty call)"
                            return
                        for expr in list(node.args) + [kw.value for kw in node.keywords]:
                            if _is_stringish(expr):
                                self.invalid_reason = (
                                    "submit_result must receive a dict answer; do not use str/repr/json dumps or f-strings"
                                )
                                return
                    self.generic_visit(node)

            class AnswerAssignmentChecker(ast.NodeVisitor):
                def __init__(self) -> None:
                    self.invalid_reason: str | None = None

                def visit_Assign(self, node: ast.Assign) -> None:
                    if self.invalid_reason:
                        return
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "answer":
                            if _is_stringish(node.value):
                                self.invalid_reason = "answer must be a dict, not a stringified value"
                                return
                    self.generic_visit(node)

                def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                    if self.invalid_reason:
                        return
                    if isinstance(node.target, ast.Name) and node.target.id == "answer":
                        if node.value and _is_stringish(node.value):
                            self.invalid_reason = "answer must be a dict, not a stringified value"
                            return
                    self.generic_visit(node)

            submit_checker = SubmitResultArgChecker()
            submit_checker.visit(solve_node)
            if submit_checker.invalid_reason:
                return False, submit_checker.invalid_reason

            answer_checker = AnswerAssignmentChecker()
            answer_checker.visit(solve_node)
            if answer_checker.invalid_reason:
                return False, answer_checker.invalid_reason

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
