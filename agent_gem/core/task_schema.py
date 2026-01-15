from __future__ import annotations

import ast
import base64
import builtins
import functools
import inspect
import json
import os
import textwrap
import typing
import uuid
from datetime import datetime, timezone
from functools import cached_property
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional

from mcp.server.fastmcp.utilities.context_injection import find_context_parameter
from mcp.server.fastmcp.utilities.func_metadata import FuncMetadata, func_metadata
from mcp.shared.tool_name_validation import validate_and_warn_tool_name
from pydantic import BaseModel, Field, field_validator, model_serializer
from pydantic_core import core_schema


def _is_async_callable(obj: Any) -> bool:
    while isinstance(obj, functools.partial):  # pragma: no cover
        obj = obj.func

    return inspect.iscoroutinefunction(obj) or (
        callable(obj) and inspect.iscoroutinefunction(getattr(obj, "__call__", None))
    )


class TaskStep(BaseModel):
    parentUuid: uuid.UUID | None
    sessionId: uuid.UUID | None
    message: dict[str, Any]
    requestId: str
    taskId: str
    uuid: uuid.UUID
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    )

    def to_payload(self) -> dict[str, Any]:
        return {
            "parentUuid": str(self.parentUuid) if self.parentUuid else None,
            "sessionId": str(self.sessionId) if self.sessionId else None,
            "taskId": str(self.taskId) if self.taskId else None,
            "message": self.message,
            "requestId": self.requestId,
            "uuid": str(self.uuid),
            "timestamp": self.timestamp,
        }


class ToolSpec(BaseModel):
    """
    Spec for generated tool set
    Adapted from
    https://github.com/modelcontextprotocol/python-sdk/blob/main/src/mcp/server/fastmcp/tools/base.py
    """

    fn: Callable[..., Any] = Field(exclude=True)
    name: str = Field(description="Name of the tool")
    title: str | None = Field(None, description="Human-readable title of the tool")
    description: str = Field(description="Description of what the tool does")
    parameters: dict[str, Any] = Field(description="JSON schema for tool parameters")
    fn_metadata: FuncMetadata = Field(
        description="Metadata about the function including a pydantic model for tool arguments"
    )
    is_async: bool = Field(description="Whether the tool is async")
    meta: dict[str, Any] | None = Field(default=None, description="Optional metadata for this tool")

    @cached_property
    def output_schema(self) -> dict[str, Any] | None:
        return self.fn_metadata.output_schema

    @model_serializer(mode="plain")
    def _serialize(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "parameters": self.parameters,
            "output_schema": self.output_schema,
            "is_async": self.is_async,
            "meta": self.meta,
        }

    @classmethod
    def from_function(
        cls,
        fn: Callable[..., Any],
        name: str | None = None,
        title: str | None = None,
        description: str | None = None,
        context_kwarg: str | None = None,
        meta: dict[str, Any] | None = None,
        structured_output: bool | None = None,
    ) -> "ToolSpec":
        """Create a Tool from a function."""
        func_name = name or fn.__name__

        validate_and_warn_tool_name(func_name)

        if func_name == "<lambda>":
            raise ValueError("You must provide a name for lambda functions")

        func_doc = description or fn.__doc__ or ""
        is_async = _is_async_callable(fn)

        if context_kwarg is None:  # pragma: no branch
            context_kwarg = find_context_parameter(fn)

        func_arg_metadata = func_metadata(
            fn,
            skip_names=[context_kwarg] if context_kwarg is not None else [],
            structured_output=structured_output,
        )
        parameters = func_arg_metadata.arg_model.model_json_schema(by_alias=True)

        return cls(
            fn=fn,
            name=func_name,
            title=title,
            description=func_doc,
            parameters=parameters,
            fn_metadata=func_arg_metadata,
            is_async=is_async,
            meta=meta,
        )

    @classmethod
    def from_function_string(
        cls,
        function_string: str,
        *,
        extra_globals: dict[str, Any] | None = None,
        filename: str = "<tool>",
    ) -> "ToolSpec":
        """Create a ToolSpec from a Python function string.

        The function body is ignored; only the signature, return annotation, and
        `*.tool(...)` decorator arguments are used to build the ToolSpec.
        """

        source = textwrap.dedent(function_string).strip()
        if not source:
            raise ValueError("function_string must be a non-empty string")

        module = ast.parse(source, filename=filename, mode="exec")

        target_fn: ast.FunctionDef | ast.AsyncFunctionDef | None = None
        decorator_call: ast.Call | None = None
        for node in module.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            call = _extract_tool_decorator_call(node)
            if call is None:
                continue
            target_fn = node
            decorator_call = call
            break

        if target_fn is None:
            raise ValueError("No function decorated with `*.tool(...)` was found")

        decorator_kwargs = _parse_tool_decorator_kwargs(decorator_call)

        name = decorator_kwargs.pop("name", None)
        title = decorator_kwargs.pop("title", None)
        description = decorator_kwargs.pop("description", None)
        structured_output = decorator_kwargs.pop("structured_output", None)
        meta = decorator_kwargs.pop("meta", None)

        if name is not None and not isinstance(name, str):
            raise TypeError("tool decorator `name` must be a string")
        if title is not None and not isinstance(title, str):
            raise TypeError("tool decorator `title` must be a string")
        if description is not None and not isinstance(description, str):
            raise TypeError("tool decorator `description` must be a string")
        if structured_output is not None and not isinstance(structured_output, bool):
            raise TypeError("tool decorator `structured_output` must be a bool")
        if meta is not None and not isinstance(meta, dict):
            raise TypeError("tool decorator `meta` must be a dict")

        if decorator_kwargs:
            meta = dict(meta or {})
            meta.setdefault("_decorator_extras", {}).update(decorator_kwargs)

        sanitized_fn = _sanitize_function_def(target_fn)
        sanitized_module = ast.Module(body=[sanitized_fn], type_ignores=[])
        ast.fix_missing_locations(sanitized_module)

        safe_globals: dict[str, Any] = {"__builtins__": {}}
        safe_globals.update(vars(typing))
        safe_globals.update(
            {
                "typing": typing,
                "str": builtins.str,
                "int": builtins.int,
                "float": builtins.float,
                "bool": builtins.bool,
                "bytes": builtins.bytes,
                "dict": builtins.dict,
                "list": builtins.list,
                "set": builtins.set,
                "tuple": builtins.tuple,
            }
        )
        try:
            from mcp.server.fastmcp.server import Context as MCPContext

            safe_globals.setdefault("Context", MCPContext)
        except Exception:
            pass
        if extra_globals:
            safe_globals.update(extra_globals)

        _inject_annotation_placeholders(sanitized_fn, safe_globals)
        exec(
            compile(
                sanitized_module,
                filename=filename,
                mode="exec",
                flags=annotations.compiler_flag,
                dont_inherit=True,
            ),
            safe_globals,
        )

        fn_obj = safe_globals.get(target_fn.name)
        if not callable(fn_obj):
            raise ValueError("Failed to define function from function_string")

        return cls.from_function(
            fn=fn_obj,
            name=name,
            title=title,
            description=description,
            meta=meta,
            structured_output=structured_output,
        )


def _extract_tool_decorator_call(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.Call | None:
    for decorator in node.decorator_list:
        if not isinstance(decorator, ast.Call):
            continue
        func = decorator.func
        if isinstance(func, ast.Attribute) and func.attr == "tool":
            return decorator
        if isinstance(func, ast.Name) and func.id == "tool":
            return decorator
    return None


def _literal_eval_or_none(expr: ast.expr) -> Any | None:
    try:
        return ast.literal_eval(expr)
    except (ValueError, SyntaxError):
        return None


def _parse_tool_decorator_kwargs(decorator_call: ast.Call) -> dict[str, Any]:
    result: dict[str, Any] = {}

    positional = list(decorator_call.args)
    if positional:
        first = _literal_eval_or_none(positional[0])
        if first is not None:
            result["name"] = first

    for kw in decorator_call.keywords:
        if kw.arg is None:
            continue
        value = _literal_eval_or_none(kw.value)
        if value is None:
            continue
        result[kw.arg] = value

    return result


class _AnnotationPlaceholder:
    @classmethod
    def __class_getitem__(cls, _item: object) -> type["_AnnotationPlaceholder"]:
        return cls

    @classmethod
    def __get_pydantic_core_schema__(cls, _source: Any, _handler: Any) -> core_schema.CoreSchema:
        return core_schema.any_schema()


def _inject_annotation_placeholders(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    globals_dict: dict[str, Any],
) -> None:
    def iter_annotation_exprs() -> List[ast.expr]:
        exprs: List[ast.expr] = []
        for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
            if arg.annotation is not None:
                exprs.append(arg.annotation)
        if node.args.vararg is not None and node.args.vararg.annotation is not None:
            exprs.append(node.args.vararg.annotation)
        if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
            exprs.append(node.args.kwarg.annotation)
        if node.returns is not None:
            exprs.append(node.returns)
        return exprs

    for expr in iter_annotation_exprs():
        for subnode in ast.walk(expr):
            if isinstance(subnode, ast.Name):
                globals_dict.setdefault(subnode.id, _AnnotationPlaceholder)
            elif isinstance(subnode, ast.Attribute):
                chain: List[str] = []
                cursor: ast.AST | None = subnode
                while isinstance(cursor, ast.Attribute):
                    chain.append(cursor.attr)
                    cursor = cursor.value
                if not isinstance(cursor, ast.Name):
                    continue
                root = cursor.id
                chain = list(reversed(chain))
                _ensure_attr_chain(globals_dict, root, chain)


def _ensure_attr_chain(
    globals_dict: dict[str, Any],
    root: str,
    attrs: List[str],
) -> None:
    if not attrs:
        globals_dict.setdefault(root, _AnnotationPlaceholder)
        return

    root_obj = globals_dict.get(root)
    if root_obj is None:
        root_obj = SimpleNamespace()
        globals_dict[root] = root_obj
    elif root_obj is not _AnnotationPlaceholder and not isinstance(root_obj, SimpleNamespace):
        current = root_obj
        for attr in attrs:
            if hasattr(current, attr):
                current = getattr(current, attr)
                continue
            return
        return

    if root_obj is _AnnotationPlaceholder:
        root_obj = SimpleNamespace()
        globals_dict[root] = root_obj

    current = root_obj
    for attr in attrs[:-1]:
        next_obj = getattr(current, attr, None)
        if not isinstance(next_obj, SimpleNamespace):
            next_obj = SimpleNamespace()
            setattr(current, attr, next_obj)
        current = next_obj
    setattr(current, attrs[-1], _AnnotationPlaceholder)


def _sanitize_function_def(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> ast.FunctionDef | ast.AsyncFunctionDef:
    docstring = ast.get_docstring(node, clean=False)

    def sanitize_defaults(defaults: List[ast.expr]) -> List[ast.expr]:
        sanitized: List[ast.expr] = []
        for default in defaults:
            if _literal_eval_or_none(default) is None:
                sanitized.append(ast.Constant(value=None))
            else:
                sanitized.append(default)
        return sanitized

    def is_safe_annotation_expr(expr: ast.AST) -> bool:
        disallowed = (
            ast.Call,
            ast.Lambda,
            ast.Await,
            ast.Yield,
            ast.YieldFrom,
            ast.ListComp,
            ast.SetComp,
            ast.DictComp,
            ast.GeneratorExp,
            ast.NamedExpr,
        )
        return not any(isinstance(sub, disallowed) for sub in ast.walk(expr))

    def sanitize_annotation(expr: ast.expr | None) -> ast.expr | None:
        if expr is None:
            return None
        if not is_safe_annotation_expr(expr):
            return ast.Name(id="Any", ctx=ast.Load())
        return expr

    new_node = ast.fix_missing_locations(
        ast.copy_location(
            (
                ast.AsyncFunctionDef(
                    name=node.name,
                    args=node.args,
                    body=[],
                    decorator_list=[],
                    returns=sanitize_annotation(node.returns),
                    type_comment=getattr(node, "type_comment", None),
                )
                if isinstance(node, ast.AsyncFunctionDef)
                else ast.FunctionDef(
                    name=node.name,
                    args=node.args,
                    body=[],
                    decorator_list=[],
                    returns=sanitize_annotation(node.returns),
                    type_comment=getattr(node, "type_comment", None),
                )
            ),
            node,
        )
    )

    new_node.args.defaults = sanitize_defaults(list(node.args.defaults))
    new_node.args.kw_defaults = [
        (
            None
            if default is None
            else (default if _literal_eval_or_none(default) is not None else ast.Constant(value=None))
        )
        for default in node.args.kw_defaults
    ]

    for arg in list(new_node.args.posonlyargs) + list(new_node.args.args) + list(new_node.args.kwonlyargs):
        arg.annotation = sanitize_annotation(arg.annotation)
    if new_node.args.vararg is not None:
        new_node.args.vararg.annotation = sanitize_annotation(new_node.args.vararg.annotation)
    if new_node.args.kwarg is not None:
        new_node.args.kwarg.annotation = sanitize_annotation(new_node.args.kwarg.annotation)

    new_body: List[ast.stmt] = []
    if docstring is not None:
        new_body.append(ast.Expr(value=ast.Constant(value=docstring)))
    new_body.append(ast.Pass())
    new_node.body = new_body

    return new_node


class EvaluationCriteria(BaseModel):
    correctness: float = Field(0.6, ge=0.0, le=1.0)
    diversity: float = Field(0.6, ge=0.0, le=1.0)
    complexity: float = Field(0.6, ge=0.0, le=1.0)
    solution_verifiability: float = Field(0.6, ge=0.0, le=1.0)

    @field_validator("*", mode="before")
    @classmethod
    def default_to_float(cls, value: float) -> float:
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                return 0.0
        return value


class TaskDefinition(BaseModel):
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_title: str = Field(..., min_length=4)
    task_content: str = Field(..., min_length=10)
    submit_result_format: Any = None
    tool_set: List[ToolSpec] = Field(default_factory=list)
    evaluation_criteria: EvaluationCriteria = Field(default_factory=EvaluationCriteria)
    difficulty_level: int = Field(default=1)

    def summary(self) -> str:
        # Build tool schemas string separately to avoid complex expressions in f-strings.
        tool_schemas = "\n".join(
            [
                json.dumps(tool.output_schema, sort_keys=True, indent=2)
                for tool in self.tool_set
            ]
        )
        return (
            f"# {self.task_title} \n\n {self.task_content} \n\n"
            f"## Submit Result Format\n {self.submit_result_format}\n\n"
            f"## Tool Set \n{tool_schemas}\n\n"
        )


class TaskPackage(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task: TaskDefinition
    solution: str | None = None
    verification: str | None = None
    agent_type: str = Field(..., min_length=3)
    metadata: Dict[str, str] = Field(default_factory=dict)
    task_path: Optional[str] = None
    use_docker: bool = False
    validated: bool | None = None
    validation_reason: str | None = None

    def as_payload(self) -> Dict[str, object]:
        return {
            "agent_type": self.agent_type,
            "task": self.task.model_dump(),
            "solution": self.solution,
            "verification": self.verification,
            "metadata": self.metadata,
            "task_path": self.task_path,
        }

    def run_solution(self, tools: Dict[str, Any]) -> Any:
        env = self._build_exec_env(tools, allow_imports=False)
        try:
            exec(self.solution, env, env)
        except (SyntaxError, IndentationError):
            normalized = self._normalize_solution_indentation(self.solution or "")
            if normalized and normalized != self.solution:
                exec(normalized, env, env)
            else:
                raise
        if "solve" not in env:
            raise RuntimeError("solution_code must define solve(tools)")
        return env["solve"](tools)

    def verify(self, tools: Dict[str, Any], answer: Any) -> bool:
        verified, score, details, message = self.verify_with_meta(tools, answer)
        if verified is None:
            raise RuntimeError(message or "verification did not return a boolean-like result")
        return bool(verified)

    def verify_with_meta(
        self, tools: Dict[str, Any], answer: Any
    ) -> tuple[bool | None, float | None, Any, str | None]:
        """Run verification and return richer metadata (bool/score/details/message)."""
        try:
            env = self._build_exec_env(tools, allow_imports=True)
            exec(self.verification, env, env)
            if "verify" not in env:
                raise RuntimeError("verification_code must define verify(tools, answer)")
            raw_output = env["verify"](tools, answer)
        except Exception as exc:
            # Surface verifier execution failures without crashing the caller.
            return None, None, None, f"verification execution failed: {exc}"

        verified, score, details, message = self._normalize_verification_output(raw_output)

        # If verify explicitly returned False/None, surface a clearer message for downstream logging.
        if verified is False and message is None:
            message = "verification returned False"
        if verified is None and message is None:
            message = f"verification returned unsupported type: {type(raw_output).__name__}"

        return verified, score, details, message

    @staticmethod
    def _coerce_score(value: Any) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    @classmethod
    def _normalize_verification_output(
        cls, output: Any
    ) -> tuple[bool | None, float | None, Any, str | None]:
        """Accept common verification return patterns and normalize them.

        Supported patterns:
        - bool
        - dict with keys like passed/success/ok/result and optional score/details/message/error
        - tuple/list like (bool, score, details)
        """
        verified: bool | None = None
        score: float | None = None
        details: Any = None
        message: str | None = None

        if isinstance(output, dict):
            for key in ("passed", "success", "ok", "result"):
                if key in output:
                    verified = bool(output.get(key))
                    break
            score = cls._coerce_score(output.get("score"))
            details = output.get("details") or output
            message = output.get("message") or output.get("error")
        elif isinstance(output, (list, tuple)) and output:
            if isinstance(output[0], bool):
                verified = output[0]
            if len(output) > 1:
                score = cls._coerce_score(output[1])
                if score is None and isinstance(output[1], str):
                    message = output[1]
            if len(output) > 2:
                details = output[2]
            if len(output) > 3 and message is None and isinstance(output[3], str):
                message = output[3]
        elif isinstance(output, bool):
            verified = output
        else:
            details = output

        return verified, score, details, message

    @staticmethod
    def _normalize_solution_indentation(code: str) -> str:
        """Best-effort fix for unindented solve() bodies."""
        if not code or not isinstance(code, str):
            return code
        lines = code.splitlines()
        try:
            def_idx = next(i for i, line in enumerate(lines) if line.strip().startswith("def solve"))
        except StopIteration:
            return code
        fixed = lines[: def_idx + 1]
        for line in lines[def_idx + 1 :]:
            if not line.strip():
                fixed.append(line)
                continue
            leading = len(line) - len(line.lstrip(" "))
            if leading < 4:
                fixed.append("    " + line.lstrip())
            else:
                fixed.append(line)
        return "\n".join(fixed) + ("\n" if code.endswith("\n") else "")

    def _build_exec_env(self, tools: Dict[str, Any], *, allow_imports: bool = False) -> Dict[str, Any]:
        """Build a safe execution environment for LLM-generated code.
        
        This environment restricts access to Python builtins to a safe subset
        commonly used in data processing and validation tasks. The selection
        includes basic type checks, iterations, and object introspection
        functions that LLMs frequently generate. For verifiers we optionally
        allow imports; for solutions we keep imports disabled to enforce
        tool-only access.
        """
        safe_builtins = {
            # Basic types and conversions
            "bool": bool,
            "int": int,
            "float": float,
            "str": str,
            "list": list,
            "dict": dict,
            "tuple": tuple,
            "set": set,
            "type": type,
            
            # Type checking
            "isinstance": isinstance,
            "issubclass": issubclass,
            
            # Object introspection (commonly used by LLMs for validation)
            "hasattr": hasattr,
            "getattr": getattr,
            "setattr": setattr,
            "dir": dir,
            "callable": callable,
            "vars": vars,
            
            # Collections and iteration
            "len": len,
            "range": range,
            "enumerate": enumerate,
            "zip": zip,
            "reversed": reversed,
            "iter": iter,
            "next": next,
            "slice": slice,
            
            # Functional operations
            "map": map,
            "filter": filter,
            "sorted": sorted,
            "sum": sum,
            "min": min,
            "max": max,
            "any": any,
            "all": all,
            
            # Basic operations
            "abs": abs,
            "round": round,
            "ord": ord,
            "chr": chr,
            "bin": bin,
            "hex": hex,
            "oct": oct,
        }
        if allow_imports:
            # Allow imports only when explicitly enabled (verifier path).
            safe_builtins["__import__"] = __import__

        # Pre-import a few safe standard modules commonly used in verifiers.
        env = {
            "__builtins__": safe_builtins,
            "tools": tools,
        }
        if allow_imports:
            try:
                import json, re, math, statistics  # type: ignore

                env.update({"json": json, "re": re, "math": math, "statistics": statistics})
            except Exception:
                pass
        return env
