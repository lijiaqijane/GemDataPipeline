"""代码约束系统：通过接口和AST分析强制约束，而不是通过prompt。

根据论文要求：
1. 解答函数只能调用工具函数或执行逻辑计算，不能直接访问数据库
2. 工具函数可以访问数据库和调用其他工具函数
3. 验证函数可以访问数据库和所有信息
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
    """工具函数协议：可以被解答函数和验证函数调用。"""
    
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """调用工具函数。"""
        ...


@dataclass
class SolutionContext:
    """解答函数的执行上下文：只能访问工具，不能访问数据库。
    
    根据论文约束，解答函数只能：
    - 调用工具函数
    - 执行逻辑计算（条件判断、循环等）
    
    解答函数不能：
    - 直接访问数据库
    - 调用其他非工具函数
    """
    
    tools: Dict[str, ToolCallable]
    
    def get_tool(self, name: str) -> ToolCallable:
        """获取工具函数。"""
        if name not in self.tools:
            raise AttributeError(f"Tool '{name}' not available in solution context. Available: {list(self.tools.keys())}")
        return self.tools[name]
    
    def __getitem__(self, name: str) -> ToolCallable:
        """支持 tools['name'] 语法。"""
        return self.get_tool(name)
    
    def __getattr__(self, name: str) -> ToolCallable:
        """支持 tools.name 语法。"""
        return self.get_tool(name)
    
    def __contains__(self, name: str) -> bool:
        """检查工具是否存在。"""
        return name in self.tools
    
    def keys(self):
        """返回所有工具名称。"""
        return self.tools.keys()


@dataclass
class ToolContext:
    """工具函数的执行上下文：可以访问数据库和调用其他工具函数。
    
    根据论文约束，工具函数可以：
    - 访问数据库
    - 调用其他工具函数
    - 必须返回可验证的结果
    """
    
    db: LocalDatabase
    tools: Dict[str, ToolCallable]
    
    def get_tool(self, name: str) -> ToolCallable:
        """获取其他工具函数。"""
        if name not in self.tools:
            raise AttributeError(f"Tool '{name}' not available. Available: {list(self.tools.keys())}")
        return self.tools[name]
    
    def query_db(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """查询数据库。"""
        return self.db.query(key, value)
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        """获取所有数据库记录。"""
        return self.db.records


@dataclass
class VerificationContext:
    """验证函数的执行上下文：可以访问数据库和所有信息。
    
    根据论文约束，验证函数可以：
    - 访问数据库
    - 访问所有工具
    - 访问所有信息
    """
    
    db: LocalDatabase
    tools: Dict[str, ToolCallable]
    answer: Any
    
    def get_tool(self, name: str) -> ToolCallable:
        """获取工具函数。"""
        if name not in self.tools:
            raise AttributeError(f"Tool '{name}' not available. Available: {list(self.tools.keys())}")
        return self.tools[name]
    
    def query_db(self, key: str, value: Any) -> List[Dict[str, Any]]:
        """查询数据库。"""
        return self.db.query(key, value)
    
    def get_all_records(self) -> List[Dict[str, Any]]:
        """获取所有数据库记录。"""
        return self.db.records


class CodeValidator:
    """使用AST分析验证代码是否符合约束。"""
    
    @staticmethod
    def validate_solution_code(code: str) -> tuple[bool, str]:
        """验证解答函数代码是否符合约束。
        
        约束：
        1. 必须定义 solve(tools) 函数
        2. 不能导入模块
        3. 不能定义其他函数（只能有solve函数）
        4. 不能直接访问数据库（通过变量名检查）
        5. 必须调用工具
        
        Returns:
            (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Code is empty"
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # 检查1: 必须定义 solve 函数
        has_solve = False
        solve_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "solve":
                has_solve = True
                solve_node = node
                break
        
        if not has_solve:
            return False, "Missing 'def solve(tools)' function definition"
        
        # 检查2: 不能导入模块
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # 构建导入语句的字符串表示（兼容 Python 3.8+）
                if isinstance(node, ast.Import):
                    module_names = ", ".join([alias.name for alias in node.names])
                    import_str = f"import {module_names}"
                else:  # ImportFrom
                    module = node.module or ""
                    names = ", ".join([alias.name for alias in node.names]) if node.names else "*"
                    import_str = f"from {module} import {names}"
                return False, f"Solution code cannot import modules: {import_str}"
        
        # 检查3: 不能定义其他函数（只能有solve函数）
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name != "solve":
                return False, f"Solution code can only define solve function, cannot define other functions: {node.name}"
        
        # 检查4: 不能直接访问数据库（通过变量名和属性访问检查）
        # 使用 NodeVisitor 来正确检查节点上下文
        class DatabaseAccessChecker(ast.NodeVisitor):
            def __init__(self):
                self.violations = []
                self.forbidden_names = {"db", "database", "ctx"}
            
            def visit_Name(self, node: ast.Name):
                if node.id in self.forbidden_names:
                    # 检查是否在属性访问、下标或调用中使用
                    # 由于AST没有parent，我们检查周围的上下文
                    # 如果名称出现在赋值左侧，可能是定义，不是访问
                    # 这里简化处理：如果名称出现在表达式中，视为访问
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
        
        # 检查5: 必须调用工具（在solve函数中）
        if solve_node:
            has_tool_call = False
            for node in ast.walk(solve_node):
                # 检查 tools['name'] 或 tools.name 或 tools.name() 调用
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
        """验证验证函数代码是否符合约束。
        
        约束：
        1. 必须定义 verify(tools, answer) 函数
        2. 可以导入模块（如果需要）
        3. 可以定义辅助函数
        4. 可以访问数据库（通过tools或直接访问）
        
        Returns:
            (is_valid, error_message)
        """
        if not code or not code.strip():
            return False, "Code is empty"
        
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"Syntax error: {e}"
        
        # 检查1: 必须定义 verify 函数
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
        """从代码中提取所有工具调用名称。"""
        tool_calls = set()
        
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return tool_calls
        
        for node in ast.walk(tree):
            # 检查 tools['name'] 调用
            if isinstance(node, ast.Subscript):
                if isinstance(node.value, ast.Name) and node.value.id == "tools":
                    if isinstance(node.slice, ast.Constant) and isinstance(node.slice.value, str):
                        tool_calls.add(node.slice.value)
                    elif isinstance(node.slice, ast.Str):  # Python < 3.8
                        tool_calls.add(node.slice.s)
            
            # 检查 tools.name 调用
            elif isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name) and node.value.id == "tools":
                    tool_calls.add(node.attr)
            
            # 检查 tools['name']() 或 tools.name() 调用
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
        
        # 移除字典方法
        dict_methods = {"keys", "values", "items", "get", "pop", "update", "clear", "copy"}
        tool_calls -= dict_methods
        
        return tool_calls

