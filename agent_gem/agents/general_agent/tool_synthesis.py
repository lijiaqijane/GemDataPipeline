from __future__ import annotations

import ast
import importlib.util
import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from agent_gem.core.task_schema import ToolSpec
from agent_gem.sandbox import SandboxExecutor
from agent_gem.tools import CallableTool

from ..base import BaseAgent, TaskContext

logger = logging.getLogger(__name__)


class ToolSynthesisMixin:
    """Tool generation, compilation, and registration helpers."""
    _SUBMIT_RESULT_TOOL = "submit_result"
    _SUBMITTED_RESULT_FILE = "submitted_result.json"
    _MAX_REGEN_ATTEMPTS = 5
    _MAX_FIELD_INVENTORY_SIZE = 2600
    _MAX_FIELD_VALUES = 5
    _FILTERED_TOOL_NAMES = {"bash", "search", "python_runner"}

    @staticmethod
    def _tool_code_uses_mcp_tool(code: str) -> bool:
        """Detect whether code uses @mcp.tool decorators (which require a FastMCP instance)."""
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for dec in node.decorator_list:
                # @mcp.tool
                if isinstance(dec, ast.Attribute) and isinstance(dec.value, ast.Name):
                    if dec.value.id == "mcp" and dec.attr == "tool":
                        return True
                # @mcp.tool(...)
                if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) and isinstance(dec.func.value, ast.Name):
                    if dec.func.value.id == "mcp" and dec.func.attr == "tool":
                        return True
        return False

    @staticmethod
    def _tool_code_has_mcp_binding(code: str) -> bool:
        """Detect a module-scope binding for a FastMCP instance named `mcp`."""
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        def _assigns_mcp(nodes: list[ast.stmt]) -> bool:
            for stmt in nodes:
                if isinstance(stmt, ast.Assign):
                    for target in stmt.targets:
                        if isinstance(target, ast.Name) and target.id == "mcp":
                            return True
                if isinstance(stmt, ast.AnnAssign):
                    if isinstance(stmt.target, ast.Name) and stmt.target.id == "mcp":
                        return True
            return False

        if _assigns_mcp(tree.body):
            return True

        for node in tree.body:
            if isinstance(node, ast.Try):
                # allow patterns like:
                # try: mcp = FastMCP(...)
                # except: mcp = ...
                if _assigns_mcp(node.body):
                    return True
                for handler in node.handlers:
                    if _assigns_mcp(handler.body):
                        return True
        return False

    @staticmethod
    def _tool_code_reads_data(code: str) -> bool:
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Name) and func.id == "open":
                return True
            if isinstance(func, ast.Attribute):
                attr = func.attr
                if attr in {"read_text", "read_bytes", "read_csv", "read_json", "connect", "open"}:
                    return True
                if isinstance(func.value, ast.Name):
                    base = func.value.id
                    if base == "csv" and attr == "DictReader":
                        return True
                    if base == "json" and attr == "load":
                        return True
                    if base == "sqlite3" and attr == "connect":
                        return True
                    if base in {"pd", "pandas"} and attr in {"read_csv", "read_json"}:
                        return True
        return False

    @staticmethod
    def _tool_code_embeds_dataset(code: str) -> bool:
        """Detect large embedded list-of-dict literals (likely sample data leakage)."""
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                value = node.value
            elif isinstance(node, ast.AnnAssign):
                value = node.value
            else:
                continue
            if isinstance(value, (ast.List, ast.Tuple)):
                dict_items = [elt for elt in value.elts if isinstance(elt, ast.Dict)]
                if len(dict_items) >= 3:
                    for dct in dict_items:
                        key_count = sum(1 for k in dct.keys if k is not None)
                        if key_count >= 2:
                            return True
        return False

    @staticmethod
    def _tool_code_returns_raw_csv_rows(code: str) -> bool:
        """Detect appending raw csv.DictReader rows without remapping keys."""
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        reader_vars: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            func = value.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "csv"
                and func.attr == "DictReader"
            ):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        reader_vars.add(target.id)
        if not reader_vars:
            return False
        for node in ast.walk(tree):
            if isinstance(node, ast.For) and isinstance(node.iter, ast.Name) and node.iter.id in reader_vars:
                if isinstance(node.target, ast.Name):
                    row_var = node.target.id
                else:
                    continue
                for child in ast.walk(node):
                    if not isinstance(child, ast.Call):
                        continue
                    func = child.func
                    if isinstance(func, ast.Attribute) and func.attr == "append" and child.args:
                        arg = child.args[0]
                        if isinstance(arg, ast.Name) and arg.id == row_var:
                            return True
        return False

    @staticmethod
    def _tool_code_uses_path_replace(code: str) -> bool:
        """Detect string replace() called on Path-derived loop variables."""
        try:
            tree = ast.parse(code)
        except Exception:
            return False
        path_containers: set[str] = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue
            target = node.targets[0].id
            value = node.value
            if isinstance(value, ast.List):
                for elt in value.elts:
                    if isinstance(elt, ast.BinOp) and isinstance(elt.op, ast.Div):
                        path_containers.add(target)
                        break
            elif isinstance(value, ast.Dict):
                for elt in value.values:
                    if isinstance(elt, ast.BinOp) and isinstance(elt.op, ast.Div):
                        path_containers.add(target)
                        break
        if not path_containers:
            return False
        for node in ast.walk(tree):
            if not isinstance(node, ast.For):
                continue
            iter_node = node.iter
            iter_name = None
            if isinstance(iter_node, ast.Name) and iter_node.id in path_containers:
                iter_name = iter_node.id
            elif (
                isinstance(iter_node, ast.Call)
                and isinstance(iter_node.func, ast.Attribute)
                and isinstance(iter_node.func.value, ast.Name)
                and iter_node.func.value.id in path_containers
                and iter_node.func.attr == "values"
            ):
                iter_name = iter_node.func.value.id
            if not iter_name or not isinstance(node.target, ast.Name):
                continue
            loop_var = node.target.id
            for child in ast.walk(node):
                if not isinstance(child, ast.Call):
                    continue
                func = child.func
                if (
                    isinstance(func, ast.Attribute)
                    and func.attr == "replace"
                    and isinstance(func.value, ast.Name)
                    and func.value.id == loop_var
                ):
                    return True
        return False

    @classmethod
    def _truncate_value(cls, value: Any, limit: int = 60) -> str:
        text = str(value).strip()
        if len(text) > limit:
            return text[:limit] + "..."
        return text

    @classmethod
    def _collect_sample_values(cls, columns: list[str], rows: list[list[Any]]) -> dict[str, list[str]]:
        values: dict[str, list[str]] = {col: [] for col in columns}
        for row in rows:
            if not isinstance(row, list):
                continue
            for idx, col in enumerate(columns):
                if idx >= len(row):
                    continue
                raw = row[idx]
                if raw is None:
                    continue
                text = cls._truncate_value(raw)
                if not text:
                    continue
                bucket = values.get(col)
                if bucket is None:
                    continue
                if text not in bucket:
                    bucket.append(text)
                if len(bucket) >= cls._MAX_FIELD_VALUES:
                    continue
        # Drop empty buckets
        return {k: v for k, v in values.items() if v}

    @classmethod
    def _collect_sample_values_from_dicts(cls, items: list[dict[str, Any]]) -> dict[str, list[str]]:
        values: dict[str, list[str]] = {}
        for item in items:
            if not isinstance(item, dict):
                continue
            for key, raw in item.items():
                if raw is None or isinstance(raw, (dict, list)):
                    continue
                text = cls._truncate_value(raw)
                if not text:
                    continue
                bucket = values.setdefault(str(key), [])
                if text not in bucket:
                    bucket.append(text)
                if len(bucket) >= cls._MAX_FIELD_VALUES:
                    continue
        return values

    def _build_field_inventory(self, data_profile: dict[str, Any]) -> dict[str, Any]:
        """Build a compact field/value inventory to guide tool parameter design."""
        inventory: dict[str, Any] = {"csv": [], "json": [], "sqlite": [], "txt": []}
        for entry in data_profile.get("csv", []):
            header = entry.get("header") if isinstance(entry.get("header"), list) else []
            rows = entry.get("sample_rows") if isinstance(entry.get("sample_rows"), list) else []
            sample_values = self._collect_sample_values([str(h) for h in header], rows)
            inventory["csv"].append(
                {
                    "path": entry.get("path"),
                    "columns": header,
                    "sample_values": sample_values,
                }
            )
        for entry in data_profile.get("json", []):
            items = entry.get("sample_items") if isinstance(entry.get("sample_items"), list) else []
            dict_items = [item for item in items if isinstance(item, dict)]
            sample_values = self._collect_sample_values_from_dicts(dict_items)
            keys = sorted({str(k) for item in dict_items for k in item.keys()})
            inventory["json"].append(
                {
                    "path": entry.get("path"),
                    "keys": keys,
                    "sample_values": sample_values,
                }
            )
        for entry in data_profile.get("sqlite", []):
            tables = entry.get("tables") if isinstance(entry.get("tables"), list) else []
            table_summaries = []
            for table in tables:
                if not isinstance(table, dict):
                    continue
                columns = table.get("columns") if isinstance(table.get("columns"), list) else []
                rows = table.get("rows") if isinstance(table.get("rows"), list) else []
                sample_values = self._collect_sample_values([str(c) for c in columns], rows)
                table_summaries.append(
                    {
                        "table": table.get("table"),
                        "columns": columns,
                        "sample_values": sample_values,
                    }
                )
            inventory["sqlite"].append(
                {
                    "path": entry.get("path"),
                    "tables": table_summaries,
                }
            )
        for entry in data_profile.get("txt", []):
            inventory["txt"].append(
                {
                    "path": entry.get("path"),
                    "line_count": entry.get("line_count"),
                }
            )
        return inventory

    @staticmethod
    def _extract_tool_output_keys(code: str) -> dict[str, list[str]]:
        """Extract literal dict keys used inside each tool function."""
        try:
            tree = ast.parse(code)
        except Exception:
            return {}
        keys_by_func: dict[str, set[str]] = {}
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            func_keys: set[str] = set()
            for child in ast.walk(node):
                if not isinstance(child, ast.Dict):
                    continue
                for key in child.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        func_keys.add(key.value)
            if func_keys:
                keys_by_func[node.name] = func_keys
        return {name: sorted(keys) for name, keys in keys_by_func.items()}

    @classmethod
    def _validate_tool_code(cls, code: str) -> tuple[bool, list[str]]:
        """Return (ok, reasons). ok means code reads from local data sources."""
        reasons: list[str] = []
        uses_mcp = cls._tool_code_uses_mcp_tool(code)
        has_mcp = cls._tool_code_has_mcp_binding(code)
        if uses_mcp and not has_mcp:
            reasons.append("missing_mcp_instance")
            return False, reasons
        if re.search(r"\bDATA_DIR\s*/\s*BASE_DIR\b", code) or re.search(r"\bBASE_DIR\s*/\s*DATA_DIR\b", code):
            reasons.append("invalid_path_join_base_dir")
            return False, reasons
        if re.search(r"\bBASE_DIR\s*/\s*BASE_DIR\b", code):
            reasons.append("duplicated_base_dir_in_path")
            return False, reasons
        if cls._tool_code_returns_raw_csv_rows(code):
            reasons.append("raw_csv_rows_returned")
            return False, reasons
        if cls._tool_code_uses_path_replace(code):
            reasons.append("path_replace_on_path")
            return False, reasons
        reads_data = cls._tool_code_reads_data(code)
        embeds_data = cls._tool_code_embeds_dataset(code)
        if not reads_data:
            reasons.append("no_data_io_detected")
            if embeds_data:
                reasons.append("embedded_dataset_literal")
            return False, reasons
        if embeds_data:
            reasons.append("embedded_dataset_literal")
        return True, reasons

    @staticmethod
    def _sanitize_tool_decorators(code: str) -> str:
        """Normalize tool decorators and drop non-literal args to avoid import-time errors."""
        try:
            tree = ast.parse(code)
        except Exception:
            return code
        changed = False

        def is_tool_decorator(dec: ast.AST) -> bool:
            if isinstance(dec, ast.Call):
                func = dec.func
                if isinstance(func, ast.Attribute) and func.attr == "tool":
                    return True
                if isinstance(func, ast.Name) and func.id == "tool":
                    return True
                return False
            if isinstance(dec, ast.Attribute) and dec.attr == "tool":
                return True
            if isinstance(dec, ast.Name) and dec.id == "tool":
                return True
            return False

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for idx, dec in enumerate(node.decorator_list):
                if not is_tool_decorator(dec):
                    continue
                if isinstance(dec, (ast.Attribute, ast.Name)):
                    node.decorator_list[idx] = ast.Call(func=dec, args=[], keywords=[])
                    dec = node.decorator_list[idx]
                    changed = True
                if not isinstance(dec, ast.Call):
                    continue
                if dec.args:
                    first = dec.args[0]
                    if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
                        dec.args = []
                        changed = True
                new_keywords: list[ast.keyword] = []
                for kw in dec.keywords:
                    if kw.arg in {"description", "name", "title"}:
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, str):
                            new_keywords.append(kw)
                        else:
                            changed = True
                    elif kw.arg == "structured_output":
                        if isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, bool):
                            new_keywords.append(kw)
                        else:
                            changed = True
                    else:
                        new_keywords.append(kw)
                dec.keywords = new_keywords

        if not changed:
            return code
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    @staticmethod
    def _normalize_typeddict_imports(code: str) -> str:
        """Force TypedDict to come from typing_extensions for Pydantic compatibility."""
        try:
            tree = ast.parse(code)
        except Exception:
            return code

        typing_aliases: set[str] = set()
        has_typing_extensions = False
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "typing":
                        typing_aliases.add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module == "typing_extensions":
                if any(alias.name == "TypedDict" for alias in node.names):
                    has_typing_extensions = True

        class Transformer(ast.NodeTransformer):
            def __init__(self) -> None:
                self.uses_typeddict = False

            def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.AST | None:
                if node.module == "typing":
                    new_names = []
                    removed = False
                    for alias in node.names:
                        if alias.name == "TypedDict":
                            removed = True
                            self.uses_typeddict = True
                        else:
                            new_names.append(alias)
                    if removed:
                        if not new_names:
                            return None
                        node.names = new_names
                    return node
                return node

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id == "TypedDict":
                    self.uses_typeddict = True
                return node

            def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
                node = self.generic_visit(node)
                if isinstance(node, ast.Attribute) and node.attr == "TypedDict":
                    if isinstance(node.value, ast.Name) and node.value.id in typing_aliases | {"typing"}:
                        self.uses_typeddict = True
                        return ast.copy_location(ast.Name(id="TypedDict", ctx=node.ctx), node)
                return node

        transformer = Transformer()
        tree = transformer.visit(tree)
        if tree is None:
            return code

        if transformer.uses_typeddict and not has_typing_extensions:
            import_node = ast.ImportFrom(
                module="typing_extensions",
                names=[ast.alias(name="TypedDict", asname=None)],
                level=0,
            )
            insert_idx = 0
            if tree.body:
                if (
                    isinstance(tree.body[0], ast.Expr)
                    and isinstance(tree.body[0].value, ast.Constant)
                    and isinstance(tree.body[0].value.value, str)
                ):
                    insert_idx = 1
                while (
                    insert_idx < len(tree.body)
                    and isinstance(tree.body[insert_idx], ast.ImportFrom)
                    and tree.body[insert_idx].module == "__future__"
                ):
                    insert_idx += 1
            tree.body.insert(insert_idx, import_node)

        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    @staticmethod
    def _ensure_submit_result_raw(code: str) -> str:
        """Force submit_result to persist and return raw result without wrappers."""
        try:
            tree = ast.parse(code)
        except Exception:
            return code
        submit_node = None
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "submit_result":
                submit_node = node
                break
        canonical_body = ast.parse(
            "\n".join(
                [
                    "output_path = BASE_DIR / \"submitted_result.json\"",
                    "with open(output_path, 'w', encoding='utf-8') as f:",
                    "    json.dump(result, f, indent=2, ensure_ascii=False, default=str)",
                    "return result",
                ]
            )
        ).body
        if submit_node is None:
            submit_src = (
                "@mcp.tool(description=\"Submit and persist a result to submitted_result.json\")\n"
                "def submit_result(result):\n"
                "    output_path = BASE_DIR / \"submitted_result.json\"\n"
                "    with open(output_path, 'w', encoding='utf-8') as f:\n"
                "        json.dump(result, f, indent=2, ensure_ascii=False, default=str)\n"
                "    return result\n"
            )
            submit_tree = ast.parse(submit_src)
            tree.body.extend(submit_tree.body)
        else:
            submit_node.body = canonical_body
        return ast.unparse(tree)

    @staticmethod
    def _summarize_import_error(stdout: str, stderr: str) -> str:
        """Return a compact import failure message to avoid echoing bad code."""
        text = f"{stdout}\n{stderr}"
        if "description" in text and "PosixPath" in text and "string" in text:
            return "import_failed: tool_description_not_string"
        if "Tool" in text and "validation error" in text and "description" in text:
            return "import_failed: tool_description_invalid"
        return "import_failed: tool_import_error"

    @staticmethod
    def _collect_annotation_names(code: str) -> set[str]:
        """Collect typing names used in annotations to ensure typing imports."""
        try:
            tree = ast.parse(code)
        except Exception:
            return set()
        wanted = {
            "List",
            "Dict",
            "Optional",
            "Any",
            "Literal",
            "Set",
            "Tuple",
            "Union",
            "Iterable",
            "Mapping",
            "Sequence",
        }
        found: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                ann_nodes: list[ast.AST] = []
                for arg in list(node.args.posonlyargs) + list(node.args.args) + list(node.args.kwonlyargs):
                    if arg.annotation is not None:
                        ann_nodes.append(arg.annotation)
                if node.args.vararg is not None and node.args.vararg.annotation is not None:
                    ann_nodes.append(node.args.vararg.annotation)
                if node.args.kwarg is not None and node.args.kwarg.annotation is not None:
                    ann_nodes.append(node.args.kwarg.annotation)
                if node.returns is not None:
                    ann_nodes.append(node.returns)
                for ann in ann_nodes:
                    for sub in ast.walk(ann):
                        if isinstance(sub, ast.Name) and sub.id in wanted:
                            found.add(sub.id)
                        elif isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name):
                            if sub.value.id == "typing" and sub.attr in wanted:
                                found.add(sub.attr)
        return found

    @staticmethod
    def _validate_tool_output_schema(specs: list[ToolSpec]) -> list[str]:
        """Validate output shapes based on tool naming conventions."""
        errors: list[str] = []
        for spec in specs:
            if spec.name == ToolSynthesisMixin._SUBMIT_RESULT_TOOL:
                continue
            schema = spec.output_schema
            if not isinstance(schema, dict):
                errors.append(f"tool_output_schema_missing:{spec.name}")
                continue
            schema_type = schema.get("type")
            if schema_type == "array":
                if not spec.name.startswith(("list_", "get_", "search_")):
                    errors.append(f"tool_output_array_disallowed:{spec.name}:{schema_type}")
            elif schema_type != "object":
                errors.append(f"tool_output_type_invalid:{spec.name}:{schema_type}")
        return errors

    @staticmethod
    def _validate_tool_specs_parameters(specs: list[ToolSpec]) -> list[str]:
        """Reject generic free-text parameters unless enumerated."""
        errors: list[str] = []
        banned_names = {"query", "q", "text", "search", "keyword", "term"}
        for spec in specs:
            if spec.name == ToolSynthesisMixin._SUBMIT_RESULT_TOOL:
                continue
            params = spec.parameters or {}
            props = params.get("properties") if isinstance(params, dict) else None
            if not isinstance(props, dict):
                continue
            for param_name, schema in props.items():
                if param_name not in banned_names:
                    continue
                enum_values = None
                if isinstance(schema, dict):
                    if isinstance(schema.get("enum"), list):
                        enum_values = schema.get("enum")
                    elif isinstance(schema.get("oneOf"), list):
                        enum_values = [
                            item.get("const")
                            for item in schema["oneOf"]
                            if isinstance(item, dict) and "const" in item
                        ]
                if not enum_values:
                    errors.append(f"free_text_param_disallowed:{spec.name}.{param_name}")
        return errors

    def _ensure_mcp_installed(self, sandbox: SandboxExecutor) -> tuple[bool, str]:
        """Ensure mcp package (FastMCP) is available in the sandbox."""
        # Check if FastMCP is importable
        import_check = sandbox.execute_bash(
            'python -c "from mcp.server.fastmcp import FastMCP; print(\'MCP_IMPORT_OK\')" 2>&1'
        )
        import_output = (import_check.get("stdout", "") or "") + (import_check.get("stderr", "") or "")
        import_success = import_check.get("returncode") == 0 or "MCP_IMPORT_OK" in import_output
        
        if not import_success:
            # mcp is not available, install it via pip
            install_result = sandbox.execute_bash("pip install mcp 2>&1")
            # Verify installation
            verify_import = sandbox.execute_bash(
                'python -c "from mcp.server.fastmcp import FastMCP; print(\'MCP_IMPORT_OK\')" 2>&1'
            )
            verify_output = (verify_import.get("stdout", "") or "") + (verify_import.get("stderr", "") or "")
            verify_success = verify_import.get("returncode") == 0 or "MCP_IMPORT_OK" in verify_output
            
            if not verify_success:
                error_msg = verify_import.get("stderr", "") or verify_import.get("stdout", "")
                if not error_msg:
                    error_msg = install_result.get("stderr", "") or install_result.get("stdout", "") or "Unknown installation error"
                return False, f"Failed to install or import mcp: {error_msg[:500]}"

        return True, ""


    @staticmethod
    def _sanitize_llm_code(raw: str) -> str:
        """Best-effort cleanup for LLM outputs that omit code fences."""
        text = (raw or "").strip()
        if not text:
            return ""
        # If code blocks exist, join them.
        blocks = BaseAgent._extract_code_blocks(text)
        if blocks:
            return "\n\n".join(BaseAgent._strip_code_fences(block) for block in blocks).strip()
        # Otherwise, strip leading prose until a likely code line.
        lines = text.splitlines()
        start = 0
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith(("import ", "from ", "@", "def ", "class ")):
                start = i
                break
        cleaned = "\n".join(lines[start:]).strip()
        return cleaned.replace("```", "").strip()

    def _submit_result_spec(self) -> ToolSpec:
        def submit_result(result: Any) -> Any:
            """Submit the final answer payload from solve()."""
            return result

        return ToolSpec.from_function(
            submit_result,
            name=self._SUBMIT_RESULT_TOOL,
            description="Submit the final answer payload.",
            meta={"system": True},
        )

    def _ensure_submit_result_tool(self, tool_specs: list[ToolSpec]) -> list[ToolSpec]:
        if any(spec.name == self._SUBMIT_RESULT_TOOL for spec in tool_specs):
            return tool_specs
        return tool_specs + [self._submit_result_spec()]

    def _iter_data_entries(self, data_profile: dict[str, Any]) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        for kind in ("csv", "json", "sqlite", "txt"):
            items = data_profile.get(kind, [])
            if not isinstance(items, list):
                continue
            for entry in items:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                if isinstance(path, str) and path.strip():
                    entries.append({"kind": kind, "path": path, "entry": entry})
        log_items = data_profile.get("logs", [])
        if isinstance(log_items, list):
            for entry in log_items:
                if not isinstance(entry, dict):
                    continue
                path = entry.get("path")
                if isinstance(path, str) and path.strip():
                    entries.append({"kind": "txt", "path": path, "entry": {"path": path}})
        entries.sort(key=lambda item: item["path"])
        return entries

    def _prepare_entry_states(self, entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
        entry_states: list[dict[str, Any]] = []
        used_tool_names: set[str] = set()
        for entry in entries:
            prefix = self._tool_prefix_from_path(entry["path"])
            list_name, main_name = self._unique_tool_names(prefix, used_tool_names)
            used_tool_names.update({list_name, main_name})
            entry_states.append(
                {
                    "kind": entry["kind"],
                    "path": entry["path"],
                    "entry": entry["entry"],
                    "list_tool_name": list_name,
                    "main_tool_name": main_name,
                    "fragment": None,
                    "specs": None,
                    "imports": None,
                }
            )
        return entry_states

    @staticmethod
    def _fragment_defines_symbol(code: str, name: str) -> bool:
        if not code or not name:
            return False
        pattern = rf"^\s*(def|class)\s+{re.escape(name)}\b|^\s*{re.escape(name)}\s*="
        return re.search(pattern, code, re.MULTILINE) is not None

    def _paths_for_validation_reasons(
        self, reasons: list[str], entry_states: list[dict[str, Any]]
    ) -> set[str]:
        paths: set[str] = set()
        for reason in reasons:
            if reason == "raw_csv_rows_returned":
                for state in entry_states:
                    fragment = state.get("fragment") or ""
                    if fragment and self._tool_code_returns_raw_csv_rows(fragment):
                        paths.add(state["path"])
            elif reason == "no_data_io_detected":
                for state in entry_states:
                    fragment = state.get("fragment") or ""
                    if fragment and not self._tool_code_reads_data(fragment):
                        paths.add(state["path"])
            elif reason == "embedded_dataset_literal":
                for state in entry_states:
                    fragment = state.get("fragment") or ""
                    if fragment and self._tool_code_embeds_dataset(fragment):
                        paths.add(state["path"])
            elif reason == "path_replace_on_path":
                for state in entry_states:
                    fragment = state.get("fragment") or ""
                    if fragment and self._tool_code_uses_path_replace(fragment):
                        paths.add(state["path"])
            elif reason == "invalid_path_join_base_dir":
                for state in entry_states:
                    fragment = state.get("fragment") or ""
                    if fragment and (
                        re.search(r"\bDATA_DIR\s*/\s*BASE_DIR\b", fragment)
                        or re.search(r"\bBASE_DIR\s*/\s*DATA_DIR\b", fragment)
                    ):
                        paths.add(state["path"])
            elif reason == "duplicated_base_dir_in_path":
                for state in entry_states:
                    fragment = state.get("fragment") or ""
                    if fragment and re.search(r"\bBASE_DIR\s*/\s*BASE_DIR\b", fragment):
                        paths.add(state["path"])
        return paths

    def _target_regen_paths(
        self, message: str, entry_states: list[dict[str, Any]]
    ) -> set[str]:
        paths: set[str] = set()
        if not message:
            return paths

        for missing_path in re.findall(r"path_missing:([A-Za-z0-9_./\\-]+)", message):
            for state in entry_states:
                if state["path"] == missing_path:
                    paths.add(state["path"])

        tool_names: set[str] = set()
        tool_names.update(re.findall(r"free_text_param_disallowed:([A-Za-z0-9_]+)\.", message))
        tool_names.update(re.findall(r"tool_output_[^:]+:([A-Za-z0-9_]+)", message))
        tool_names.update(re.findall(r"in ([A-Za-z0-9_]+):", message))
        if "tools.py missing required tool defs" in message or "tools.py global name collisions" in message:
            tool_names.update(re.findall(r"'([A-Za-z0-9_]+)'", message))

        if "tools.py invalid (not data-driven)" in message:
            reasons = re.findall(r"'([^']+)'", message)
            paths.update(self._paths_for_validation_reasons(reasons, entry_states))

        tool_name_to_path = {}
        for state in entry_states:
            tool_name_to_path[state["list_tool_name"]] = state["path"]
            tool_name_to_path[state["main_tool_name"]] = state["path"]

        for name in tool_names:
            path = tool_name_to_path.get(name)
            if path:
                paths.add(path)

        if "tools.py global name collisions" in message:
            for name in tool_names:
                for state in entry_states:
                    fragment = state.get("fragment") or ""
                    if self._fragment_defines_symbol(fragment, name):
                        paths.add(state["path"])

        return paths

    @staticmethod
    def _tool_prefix_from_path(path: str) -> str:
        parts = Path(path).with_suffix("").as_posix().split("/")
        if parts and parts[0] == "data":
            parts = parts[1:]
        if not parts:
            parts = ["data"]
        parts = parts[-2:] if len(parts) >= 2 else parts
        prefix = "_".join(parts)
        prefix = re.sub(r"[^a-zA-Z0-9_]", "_", prefix)
        prefix = re.sub(r"_+", "_", prefix).strip("_").lower()
        if not prefix:
            prefix = "data"
        if prefix[0].isdigit():
            prefix = f"data_{prefix}"
        return prefix

    @staticmethod
    def _typed_dict_class_names(code: str) -> set[str]:
        try:
            tree = ast.parse(code)
        except Exception:
            return set()
        names: set[str] = set()
        for node in tree.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "TypedDict":
                    names.add(node.name)
                elif isinstance(base, ast.Attribute) and base.attr == "TypedDict":
                    names.add(node.name)
        return names

    @staticmethod
    def _annotation_uses_names(node: ast.AST | None, names: set[str]) -> bool:
        if node is None or not names:
            return False
        for child in ast.walk(node):
            if isinstance(child, ast.Name) and child.id in names:
                return True
            if isinstance(child, ast.Constant) and isinstance(child.value, str) and child.value in names:
                return True
        return False

    @staticmethod
    def _namespace_for_path(path: str) -> str:
        base = ToolSynthesisMixin._tool_prefix_from_path(path)
        digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:8]
        return f"{base}_{digest}"

    @staticmethod
    def _unique_tool_names(prefix: str, used: set[str]) -> tuple[str, str]:
        base_list = f"list_{prefix}_options"
        base_tool = f"get_{prefix}_records"
        list_name = base_list
        tool_name = base_tool
        suffix = 2
        while list_name in used or tool_name in used:
            list_name = f"{base_list}_{suffix}"
            tool_name = f"{base_tool}_{suffix}"
            suffix += 1
        return list_name, tool_name

    @staticmethod
    def _collect_module_imports(code: str) -> list[str]:
        try:
            tree = ast.parse(code)
        except Exception:
            return []
        imports: list[str] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                if isinstance(node, ast.ImportFrom) and node.module == "__future__":
                    continue
                imports.append(ast.unparse(node))
        return imports

    @staticmethod
    def _string_literals_in_code(code: str) -> set[str]:
        try:
            tree = ast.parse(code)
        except Exception:
            return set()
        literals: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                if node.value:
                    literals.add(node.value)
            elif isinstance(node, ast.JoinedStr):
                parts = []
                for value in node.values:
                    if isinstance(value, ast.Constant) and isinstance(value.value, str):
                        parts.append(value.value)
                if parts:
                    literals.add("".join(parts))
        return literals

    @classmethod
    def _path_literals_ok(cls, code: str, rel_path: str) -> bool:
        literals = cls._string_literals_in_code(code)
        if not literals:
            return False
        normalized = rel_path.replace("\\", "/")
        parts = [part for part in normalized.split("/") if part]
        if normalized in literals:
            return True
        return all(part in literals for part in parts)

    @staticmethod
    def _extract_decorated_tool_functions(code: str) -> tuple[list[dict[str, Any]], ast.Module | None]:
        normalized = re.sub(r"@mcp\.tool\s*(?=\n)", "@mcp.tool()", code)
        try:
            tree = ast.parse(normalized)
        except Exception:
            return [], None
        lines = normalized.splitlines()
        funcs: list[dict[str, Any]] = []

        def is_tool_decorator(dec: ast.AST) -> bool:
            if isinstance(dec, ast.Call):
                func = dec.func
                if isinstance(func, ast.Attribute) and func.attr == "tool":
                    return True
                if isinstance(func, ast.Name) and func.id == "tool":
                    return True
            if isinstance(dec, ast.Attribute) and dec.attr == "tool":
                return True
            if isinstance(dec, ast.Name) and dec.id == "tool":
                return True
            return False

        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if not any(is_tool_decorator(dec) for dec in node.decorator_list):
                continue
            decorator_lines = [dec.lineno for dec in node.decorator_list if getattr(dec, "lineno", None)]
            start_line = min(decorator_lines) if decorator_lines else node.lineno or 1
            end_line = node.end_lineno or node.lineno or start_line
            block = "\n".join(lines[start_line - 1 : end_line]).strip()
            funcs.append({"name": node.name, "node": node, "block": block})
        return funcs, tree

    @staticmethod
    def _function_body_has_logic(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        body = list(node.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        for stmt in body:
            if isinstance(stmt, ast.Pass):
                continue
            if isinstance(stmt, ast.Raise):
                continue
            if (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is Ellipsis
            ):
                continue
            return True
        return False

    @staticmethod
    def _demote_extra_tool_decorators(code: str, keep_names: set[str]) -> str:
        try:
            tree = ast.parse(code)
        except Exception:
            return code

        def is_tool_decorator(dec: ast.AST) -> bool:
            if isinstance(dec, ast.Call):
                func = dec.func
                if isinstance(func, ast.Attribute) and func.attr == "tool":
                    return True
                if isinstance(func, ast.Name) and func.id == "tool":
                    return True
            if isinstance(dec, ast.Attribute) and dec.attr == "tool":
                return True
            if isinstance(dec, ast.Name) and dec.id == "tool":
                return True
            return False

        changed = False
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            if node.name in keep_names:
                continue
            if not node.decorator_list:
                continue
            new_decorators = [dec for dec in node.decorator_list if not is_tool_decorator(dec)]
            if len(new_decorators) != len(node.decorator_list):
                node.decorator_list = new_decorators
                changed = True

        if not changed:
            return code
        return ast.unparse(tree)

    @staticmethod
    def _strip_module_imports_and_globals(code: str) -> str:
        try:
            tree = ast.parse(code)
        except Exception:
            return code

        def _is_docstring(stmt: ast.stmt) -> bool:
            return (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )

        def _is_random_seed(stmt: ast.stmt) -> bool:
            if not isinstance(stmt, ast.Expr) or not isinstance(stmt.value, ast.Call):
                return False
            call = stmt.value
            func = call.func
            return (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "random"
                and func.attr == "seed"
            )

        def _is_main_guard(stmt: ast.stmt) -> bool:
            if not isinstance(stmt, ast.If):
                return False
            test = stmt.test
            if not isinstance(test, ast.Compare):
                return False
            if not isinstance(test.left, ast.Name) or test.left.id != "__name__":
                return False
            if len(test.ops) != 1 or not isinstance(test.ops[0], ast.Eq):
                return False
            if len(test.comparators) != 1:
                return False
            comp = test.comparators[0]
            return isinstance(comp, ast.Constant) and comp.value == "__main__"

        stripped: list[ast.stmt] = []
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            if isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                elif isinstance(node, ast.AnnAssign):
                    targets = [node.target]
                names = {
                    t.id
                    for t in targets
                    if isinstance(t, ast.Name)
                }
                if {"BASE_DIR", "mcp"} & names:
                    continue
            if _is_docstring(node):
                continue
            if _is_main_guard(node):
                continue
            if isinstance(node, ast.Expr) and not _is_random_seed(node):
                continue
            stripped.append(node)

        module = ast.Module(body=stripped, type_ignores=[])
        return ast.unparse(module)

    @staticmethod
    def _namespace_fragment(code: str, namespace: str, keep_names: set[str]) -> str:
        """Rename module-level symbols to avoid cross-fragment collisions."""
        if not namespace:
            return code
        try:
            tree = ast.parse(code)
        except Exception:
            return code

        mapping: dict[str, str] = {}

        def should_rename(name: str) -> bool:
            if name in keep_names:
                return False
            if name.startswith("__") and name.endswith("__"):
                return False
            return True

        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if should_rename(node.name):
                    mapping.setdefault(node.name, f"{namespace}_{node.name}")
            elif isinstance(node, (ast.Assign, ast.AnnAssign)):
                targets = []
                if isinstance(node, ast.Assign):
                    targets = node.targets
                else:
                    targets = [node.target]
                for target in targets:
                    if isinstance(target, ast.Name) and should_rename(target.id):
                        mapping.setdefault(target.id, f"{namespace}_{target.id}")

        if not mapping:
            return code

        class Renamer(ast.NodeTransformer):
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
                if node.name in mapping:
                    node.name = mapping[node.name]
                self.generic_visit(node)
                return node

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
                if node.name in mapping:
                    node.name = mapping[node.name]
                self.generic_visit(node)
                return node

            def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
                if node.name in mapping:
                    node.name = mapping[node.name]
                self.generic_visit(node)
                return node

            def visit_Name(self, node: ast.Name) -> ast.AST:
                if node.id in mapping:
                    node.id = mapping[node.id]
                return node

        renamer = Renamer()
        tree = renamer.visit(tree)
        if tree is None:
            return code
        ast.fix_missing_locations(tree)
        return ast.unparse(tree)

    @staticmethod
    def _detect_global_name_collisions(code: str) -> list[str]:
        """Return duplicated top-level symbols (functions/classes/constants)."""
        try:
            tree = ast.parse(code)
        except Exception:
            return []
        counts: dict[str, int] = {}
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
                counts[name] = counts.get(name, 0) + 1
                continue
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        counts[name] = counts.get(name, 0) + 1
                continue
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                name = node.target.id
                counts[name] = counts.get(name, 0) + 1
        return sorted(name for name, count in counts.items() if count > 1)

    def _build_file_tool_prompt(
        self,
        *,
        topic: str,
        file_kind: str,
        rel_path: str,
        entry_json: str,
        field_inventory_json: str,
        list_tool_name: str,
        main_tool_name: str,
        required_fields: set[str] | None = None,
        all_errors: list[str] | None = None,
    ) -> str:
        error_context = ""
        if all_errors:
            error_context = (
                "Previous attempts encountered the following issues:\n"
                + "\n".join(f"- {error}" for error in all_errors)
                + "\n\n"
            )
        required_hint = ""
        if required_fields:
            required_hint = f"Required output fields (use if applicable): {sorted(required_fields)}\n"
        rules_parts = [
            "- Output ONLY one Python code block.\n",
            f"- Define tools with @mcp.tool() for `{list_tool_name}` and `{main_tool_name}`.\n",
            "- You MAY define helper functions/classes/constants and imports.\n",
            "- Avoid adding extra @mcp.tool-decorated functions; they will be ignored.\n",
            "- Do NOT define mcp, FastMCP, or BASE_DIR; those are provided by the framework.\n",
            f"- Use BASE_DIR and the exact path parts for this file: {rel_path}.\n",
            "- Each tool function must read the local file and must not be empty (no pass/raise).\n",
            f"- `{list_tool_name}` returns a list of allowed values for a filter parameter used by `{main_tool_name}`.\n",
            f"- `{main_tool_name}` accepts only those values (enum), and returns list[dict] or a dict summary.\n",
            "- Add type hints for all parameters and returns; list tools should return list[str].\n",
            "- Keep return types stable; if a field is missing, use a type-appropriate default (e.g., \"\" for strings, 0 for numbers, [] for lists) instead of None.\n",
        ]
        if file_kind == "txt":
            rules_parts.append(
                "- When parsing text/markdown, strip formatting markers like **, *, and trailing colons from extracted fields.\n"
            )
            rules_parts.append(
                "- Build a cleaned copy of the text (remove markdown markers like **, *, backticks) and use it for section/header detection.\n"
            )
            rules_parts.append(
                "- Headings may look like '**Title:**' or '* Title:'; normalize these to 'Title' before matching.\n"
            )
            rules_parts.append(
                "- Do not stop parsing after the first paragraph; scan the full file for section headers and lists.\n"
            )
            rules_parts.append(
                "- If you see numbered or bulleted lists under a section, map those list items into a list field.\n"
            )
        if file_kind == "csv":
            rules_parts.append(
                "- CSV files may not be comma-delimited; detect dialect with csv.Sniffer or use the file profile, "
                "then pass the dialect/delimiter to csv.reader/csv.DictReader.\n"
            )
        rules_parts.extend(
            [
                "- Do NOT append raw csv.DictReader rows; map into a new dict with explicit keys.\n",
                "- Avoid TypedDict/custom class return annotations; use list[dict[str, Any]] or dict[str, Any].\n",
                "- Do NOT name parameters query/q/text/search/keyword/term.\n",
                "- Avoid free-text parameters; use explicit, data-driven filters.\n",
                "- Deterministic: set random.seed(0) if randomness is used.\n",
                "- Tool descriptions must be plain string literals.\n",
                "- Return JSON-serializable shapes with stable keys.\n",
            ]
        )
        rules = "".join(rules_parts)
        prompt = (
            f"{error_context}"
            "Generate tools for exactly ONE local data file.\n"
            "Hard rules:\n"
            f"{rules}\n"
            f"Topic: {topic}\n"
            f"File type: {file_kind}\n"
            f"File profile (JSON): {entry_json}\n"
            f"Field inventory (JSON): {field_inventory_json}\n"
            f"{required_hint}"
            "Output ONLY one Python code block defining the two tools. No prose."
        )
        return prompt

    def _synthesize_file_tools(
        self,
        *,
        topic: str,
        entry: dict[str, Any],
        list_tool_name: str,
        main_tool_name: str,
        ctx: TaskContext,
        required_fields: set[str] | None = None,
        all_errors: list[str] | None = None,
    ) -> tuple[str, list[ToolSpec], list[str]]:
        file_profile: dict[str, Any] = {"csv": [], "json": [], "sqlite": [], "txt": []}
        file_profile[entry["kind"]] = [entry["entry"]]
        field_inventory = self._build_field_inventory(file_profile)
        field_inventory_json = json.dumps(field_inventory, ensure_ascii=True)
        if len(field_inventory_json) > self._MAX_FIELD_INVENTORY_SIZE:
            field_inventory_json = field_inventory_json[:self._MAX_FIELD_INVENTORY_SIZE] + "..."
        entry_json = json.dumps(entry["entry"], ensure_ascii=False)
        if len(entry_json) > 2000:
            entry_json = entry_json[:2000] + "..."

        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        attempt_errors = list(all_errors or [])
        for attempt in range(1, self._MAX_REGEN_ATTEMPTS + 1):
            prompt = self._build_file_tool_prompt(
                topic=topic,
                file_kind=entry["kind"],
                rel_path=entry["path"],
                entry_json=entry_json,
                field_inventory_json=field_inventory_json,
                list_tool_name=list_tool_name,
                main_tool_name=main_tool_name,
                required_fields=required_fields,
                all_errors=attempt_errors or None,
            )
            raw = self.llm.simple_complete(prompt, temperature=0.55, max_tokens=max_tokens)
            ctx.add_step(
                {
                    "type": "tool_synthesis_file",
                    "path": entry["path"],
                    "attempt": attempt,
                    "content": raw,
                }
            )
            code = self._sanitize_llm_code(raw)
            code = self._sanitize_tool_decorators(code)
            errors: list[str] = []
            if not code:
                errors.append("empty_tool_code")
            imports = self._collect_module_imports(code)
            funcs, tree = self._extract_decorated_tool_functions(code)
            if tree is None:
                errors.append("parse_failed")
            names = {func["name"] for func in funcs}
            missing = {list_tool_name, main_tool_name} - names
            if missing:
                errors.append("tool_names_missing")
            if "submit_result" in names or "def submit_result" in code:
                errors.append("submit_result_disallowed")
            typed_dict_names = self._typed_dict_class_names(code)
            for func in funcs:
                node = func["node"]
                if not self._function_body_has_logic(node):
                    errors.append(f"empty_body:{func['name']}")
                if self._annotation_uses_names(node.returns, typed_dict_names):
                    errors.append(f"typed_dict_return_disallowed:{func['name']}")
            if not self._path_literals_ok(code, entry["path"]):
                errors.append("path_missing_or_mismatch")
            if self._tool_code_returns_raw_csv_rows(code):
                errors.append("raw_csv_rows_returned")
            if not self._tool_code_reads_data(code):
                errors.append("no_data_io_detected")
            if self._tool_code_embeds_dataset(code):
                errors.append("embedded_dataset_literal")
            if self._tool_code_uses_path_replace(code):
                errors.append("path_replace_on_path")
            if re.search(r"\bDATA_DIR\s*/\s*BASE_DIR\b", code) or re.search(
                r"\bBASE_DIR\s*/\s*DATA_DIR\b", code
            ):
                errors.append("invalid_path_join_base_dir")
            if re.search(r"\bBASE_DIR\s*/\s*BASE_DIR\b", code):
                errors.append("duplicated_base_dir_in_path")

            if errors:
                prompt_errors: list[str] = []
                for err in errors:
                    if err == "raw_csv_rows_returned":
                        prompt_errors.append(
                            "raw_csv_rows_returned (do NOT append DictReader rows; create a new dict with explicit keys)"
                        )
                    elif err.startswith("typed_dict_return_disallowed:"):
                        prompt_errors.append(f"{err} (use list[dict[str, Any]] or dict[str, Any])")
                    else:
                        prompt_errors.append(err)
                logger.info(
                    "Tool synthesis retry for file %s (attempt %s/%s): %s",
                    entry["path"],
                    attempt,
                    self._MAX_REGEN_ATTEMPTS,
                    errors,
                )
                ctx.add_step(
                    {
                        "type": "tool_synthesis_file_validation_failed",
                        "path": entry["path"],
                        "attempt": attempt,
                        "errors": errors,
                    }
                )
                attempt_errors = prompt_errors
                continue

            keep_names = {list_tool_name, main_tool_name}
            demoted = self._demote_extra_tool_decorators(code, keep_names)
            fragment = self._strip_module_imports_and_globals(demoted).strip()
            namespace = self._namespace_for_path(entry["path"])
            fragment = self._namespace_fragment(fragment, namespace, keep_names).strip()
            if not fragment:
                errors = ["empty_tool_code"]
                logger.info(
                    "Tool synthesis retry for file %s (attempt %s/%s): %s",
                    entry["path"],
                    attempt,
                    self._MAX_REGEN_ATTEMPTS,
                    errors,
                )
                ctx.add_step(
                    {
                        "type": "tool_synthesis_file_validation_failed",
                        "path": entry["path"],
                        "attempt": attempt,
                        "errors": errors,
                    }
                )
                attempt_errors = errors
                continue
            specs = self._extract_mcp_tools_from_python(fragment)
            spec_names = {spec.name for spec in specs}
            missing_specs = {list_tool_name, main_tool_name} - spec_names
            if missing_specs:
                errors = ["tool_spec_name_mismatch"]
            else:
                output_keys = self._extract_tool_output_keys(fragment)
                keys_for_main = output_keys.get(main_tool_name)
                if not keys_for_main and output_keys:
                    union_keys: set[str] = set()
                    for keys in output_keys.values():
                        union_keys |= {k for k in keys if isinstance(k, str)}
                    if union_keys:
                        keys_for_main = sorted(union_keys)
                if keys_for_main:
                    enriched: list[ToolSpec] = []
                    for spec in specs:
                        if spec.name == main_tool_name:
                            meta = dict(spec.meta or {})
                            meta["output_keys"] = keys_for_main
                            enriched.append(spec.copy(update={"meta": meta}))
                        else:
                            enriched.append(spec)
                    specs = enriched
                return fragment, specs, imports
            if errors:
                logger.info(
                    "Tool synthesis retry for file %s (attempt %s/%s): %s",
                    entry["path"],
                    attempt,
                    self._MAX_REGEN_ATTEMPTS,
                    errors,
                )
                ctx.add_step(
                    {
                        "type": "tool_synthesis_file_validation_failed",
                        "path": entry["path"],
                        "attempt": attempt,
                        "errors": errors,
                    }
                )
                attempt_errors = errors
                continue

        raise RuntimeError(
            f"Tool synthesis failed for {entry['path']} after {self._MAX_REGEN_ATTEMPTS} attempts."
        )

    def _assemble_tools_module(
        self,
        *,
        fragments: list[str],
        import_lines: list[str],
        data_profile: dict[str, Any],
    ) -> str:
        body = "\n\n".join(fragment.strip() for fragment in fragments if fragment.strip()).strip()
        body = self._sanitize_tool_decorators(body)
        known_paths: set[str] = set()
        for key in ("csv", "json", "sqlite", "txt"):
            for entry in data_profile.get(key, []):
                path = entry.get("path")
                if isinstance(path, str):
                    known_paths.add(path)
        known_basenames = {Path(path).name for path in known_paths}
        body = self._fix_file_paths(
            body,
            known_basenames=known_basenames,
            add_imports=False,
            add_base_dir=False,
        )

        typing_names = self._collect_annotation_names(body)
        header_lines: list[str] = []
        seen: set[str] = set()

        def add_import(line: str) -> None:
            if line not in seen:
                header_lines.append(line)
                seen.add(line)

        add_import("from __future__ import annotations")
        for line in import_lines:
            if line.startswith("from __future__ import"):
                continue
            if "mcp.server.fastmcp" in line or line.startswith("import mcp") or line.startswith("from mcp"):
                continue
            add_import(line)
        if typing_names:
            add_import("from typing import " + ", ".join(sorted(typing_names)))
        if "typing." in body and not any(line.startswith("import typing") for line in header_lines):
            add_import("import typing")
        if re.search(r"\bEnum\b", body) and not any("from enum import Enum" in line for line in header_lines):
            add_import("from enum import Enum")
        add_import("import json")
        add_import("from pathlib import Path")
        add_import("from mcp.server.fastmcp import FastMCP")
        if "csv." in body and not any("import csv" in line for line in header_lines):
            add_import("import csv")
        if "sqlite3." in body and not any("import sqlite3" in line for line in header_lines):
            add_import("import sqlite3")
        if "random." in body and not any("import random" in line for line in header_lines):
            add_import("import random")
        if "re." in body and not any("import re" in line for line in header_lines):
            add_import("import re")
        if "os." in body and not any(line.startswith(("import os", "from os")) for line in header_lines):
            add_import("import os")

        header = "\n".join(header_lines).strip()
        module_code = "\n".join(
            [
                header,
                "",
                "BASE_DIR = Path(__file__).parent",
                "mcp = FastMCP(\"Tools\")",
                "",
                body,
            ]
        ).strip()
        module_code = self._ensure_submit_result_raw(module_code)
        module_code = self._normalize_typeddict_imports(module_code)
        return module_code.strip() + "\n"

    def _validate_paths_in_module(self, code: str, entries: list[dict[str, Any]]) -> list[str]:
        errors: list[str] = []
        for entry in entries:
            if not self._path_literals_ok(code, entry["path"]):
                errors.append(f"path_missing:{entry['path']}")
        return errors

    def _synthesize_task_tools(
        self,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        data_profile: dict[str, Any],
        required_fields: set[str] | None = None,
        all_errors: list[str] | None = None,
        entry_states: list[dict[str, Any]] | None = None,
        regen_paths: set[str] | None = None,
    ) -> tuple[list[ToolSpec], str, dict[str, Any]]:
        """Generate task-specific tools using detected data sources.
        
        Returns:
            (tool_specs, tools_code, tool_selftest) tuple
        """
        entries = self._iter_data_entries(data_profile)
        if not entries:
            raise RuntimeError("Tool synthesis failed: no usable data files found.")

        if entry_states is None:
            entry_states = self._prepare_entry_states(entries)
        else:
            entries = [
                {"kind": state["kind"], "path": state["path"], "entry": state["entry"]}
                for state in entry_states
            ]

        regen_set = set(regen_paths or [])
        fragments: list[str] = []
        import_lines: list[str] = []
        tool_specs: list[ToolSpec] = []

        for state in entry_states:
            fragment = state.get("fragment")
            specs = state.get("specs")
            imports = state.get("imports")
            if fragment and specs is not None and imports is not None and state["path"] not in regen_set:
                fragments.append(fragment)
                import_lines.extend(imports)
                tool_specs.extend(specs)
                continue

            entry = {"kind": state["kind"], "path": state["path"], "entry": state["entry"]}
            per_entry_errors = (
                all_errors if all_errors and (not regen_set or state["path"] in regen_set) else None
            )
            fragment, specs, imports = self._synthesize_file_tools(
                topic=topic,
                entry=entry,
                list_tool_name=state["list_tool_name"],
                main_tool_name=state["main_tool_name"],
                ctx=ctx,
                required_fields=required_fields,
                all_errors=per_entry_errors,
            )
            state["fragment"] = fragment
            state["specs"] = specs
            state["imports"] = imports
            fragments.append(fragment)
            import_lines.extend(imports)
            tool_specs.extend(specs)

        tools_code = self._assemble_tools_module(
            fragments=fragments,
            import_lines=import_lines,
            data_profile=data_profile,
        )

        path_errors = self._validate_paths_in_module(tools_code, entries)
        if path_errors:
            raise RuntimeError(f"tools.py path validation failed: {path_errors}")

        # Filter and prepare specs
        seen: set[str] = set()
        filtered: list[ToolSpec] = []
        for spec in tool_specs:
            if spec.name in self._FILTERED_TOOL_NAMES:
                continue
            if spec.name in seen:
                continue
            seen.add(spec.name)
            filtered.append(
                spec.copy(
                    update={
                        "description": spec.description or f"Query curated records about {topic}",
                        "meta": (spec.meta or {}) | {"topic": topic},
                    }
                )
            )

        output_keys = self._extract_tool_output_keys(tools_code)
        if output_keys:
            enriched: list[ToolSpec] = []
            for spec in filtered:
                if spec.name == self._SUBMIT_RESULT_TOOL:
                    enriched.append(spec)
                    continue
                meta = dict(spec.meta or {})
                keys = output_keys.get(spec.name)
                if keys:
                    meta["output_keys"] = keys
                enriched.append(spec.copy(update={"meta": meta}))
            filtered = enriched

        param_errors = self._validate_tool_specs_parameters(filtered)
        if param_errors:
            raise RuntimeError(f"tools.py parameter validation failed: {param_errors}")

        non_submit = [spec for spec in filtered if spec.name != self._SUBMIT_RESULT_TOOL]
        if len(non_submit) < 2:
            raise RuntimeError("tools.py validation failed: too_few_data_tools")
        if not any(spec.name.startswith("list_") for spec in non_submit):
            raise RuntimeError("tools.py validation failed: missing_list_discovery_tool")

        output_errors = self._validate_tool_output_schema(filtered)
        if output_errors:
            raise RuntimeError(f"tools.py output schema validation failed: {output_errors}")

        filtered = self._ensure_submit_result_tool(filtered)

        # Ensure mcp is installed before import test
        mcp_installed, mcp_error = self._ensure_mcp_installed(sandbox)
        if not mcp_installed:
            raise RuntimeError(f"mcp_installation_failed: {mcp_error}")

        # Write tools.py and test import
        tools_path = sandbox.sandbox_dir / "tools.py"
        implemented_code = self._generate_tool_implementations(
            tools_code=tools_code,
            sandbox_dir=sandbox.sandbox_dir,
            tool_specs=filtered,
            topic=topic,
            data_profile=data_profile,
            sandbox=sandbox,
        )
        collision_names = self._detect_global_name_collisions(implemented_code)
        if collision_names:
            raise RuntimeError(f"tools.py global name collisions: {collision_names}")
        tools_path.write_text(implemented_code, encoding="utf-8")

        import_result = sandbox.execute_bash('python -c "import tools; print(\'TOOLS_IMPORT_OK\')"')
        stderr = (import_result.get("stderr") or "").strip()
        stdout = (import_result.get("stdout") or "").strip()
        import_ok = import_result.get("returncode") == 0 or "TOOLS_IMPORT_OK" in stdout
        if not import_ok:
            raise RuntimeError(self._summarize_import_error(stdout, stderr))

        registration_ok, registration_errors = self._register_task_tools(
            filtered, sandbox, ctx, tools_code=tools_code
        )
        if not registration_ok:
            raise RuntimeError(f"tool_registration_failed: {registration_errors}")

        tool_selftest = self._self_test_tools(filtered, sandbox, topic, ctx, data_profile)
        regen_needed, regen_reasons = self._needs_tool_regeneration(tool_selftest)
        if regen_needed:
            raise RuntimeError(f"tool_selftest_failed: {regen_reasons}")

        ctx.add_step(
            {
                "type": "tool_synthesis",
                "tool_count": len(filtered),
                "tools": [spec.model_dump() for spec in filtered],
            }
        )
        return filtered, tools_code, tool_selftest

    @staticmethod
    def _is_retryable_tool_synthesis_error(message: str) -> bool:
        non_retryable_prefixes = ("mcp_installation_failed",)
        non_retryable_substrings = ("no usable data files found",)
        if any(message.startswith(prefix) for prefix in non_retryable_prefixes):
            return False
        if any(text in message for text in non_retryable_substrings):
            return False
        return True

    def _synthesize_task_tools_with_retry(
        self,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        data_profile: dict[str, Any],
        required_fields: set[str] | None = None,
        all_errors: list[str] | None = None,
    ) -> tuple[list[ToolSpec], str, dict[str, Any]]:
        entries = self._iter_data_entries(data_profile)
        if not entries:
            raise RuntimeError("Tool synthesis failed: no usable data files found.")
        entry_states = self._prepare_entry_states(entries)
        attempt_errors = list(all_errors or [])
        regen_paths: set[str] | None = None
        for attempt in range(1, self._MAX_REGEN_ATTEMPTS + 1):
            try:
                return self._synthesize_task_tools(
                    topic,
                    records,
                    ctx,
                    sandbox,
                    data_profile,
                    required_fields=required_fields,
                    all_errors=attempt_errors,
                    entry_states=entry_states,
                    regen_paths=regen_paths,
                )
            except RuntimeError as exc:
                message = str(exc)
                ctx.add_step(
                    {
                        "type": "tool_synthesis_retry",
                        "attempt": attempt,
                        "error": message,
                    }
                )
                logger.warning(
                    "Tool synthesis retry for tools.py (attempt %s/%s): %s",
                    attempt,
                    self._MAX_REGEN_ATTEMPTS,
                    message,
                )
                if not self._is_retryable_tool_synthesis_error(message) or attempt >= self._MAX_REGEN_ATTEMPTS:
                    raise
                regen_paths = self._target_regen_paths(message, entry_states)
                if not regen_paths:
                    regen_paths = {state["path"] for state in entry_states}
                attempt_errors = [message]
        raise RuntimeError("tool_synthesis_failed_after_retries")

    def _ensure_tools_meet_format_requirements(
        self,
        topic: str,
        records: list[dict[str, Any]],
        ctx: TaskContext,
        sandbox: SandboxExecutor,
        data_profile: dict[str, Any],
        tool_specs: list[ToolSpec],
        tools_code: str,
        tool_selftest: dict[str, Any],
        expected_fields: set[str],
    ) -> tuple[list[ToolSpec], str, dict[str, Any]]:
        """Ensure tools meet format requirements, regenerating if needed.
        
        Returns:
            (tool_specs, tools_code, tool_selftest) tuple, potentially regenerated
        """
        for regen_attempt in range(self._MAX_REGEN_ATTEMPTS):
            regen_needed, regen_reasons = self._needs_tool_regeneration(
                tool_selftest, required_fields=expected_fields
            )
            
            if not regen_needed:
                break
            
            logger.info(
                "Tool regeneration for format requirements (attempt %s/%s): %s, fields: %s",
                regen_attempt + 1,
                self._MAX_REGEN_ATTEMPTS,
                regen_reasons,
                sorted(expected_fields),
            )
            # Convert regen_reasons to error format
            all_errors = [f"Format requirement issue: {reason}" for reason in regen_reasons] if regen_reasons else None
            if expected_fields:
                all_errors = (all_errors or []) + [f"Missing required fields: {sorted(expected_fields)}"]
            tool_specs, tools_code, tool_selftest = self._synthesize_task_tools_with_retry(
                topic,
                records,
                ctx,
                sandbox,
                data_profile,
                required_fields=expected_fields,
                all_errors=all_errors,
            )
            ctx.add_step(
                {
                    "type": "tool_regeneration_for_format",
                    "reasons": regen_reasons,
                    "attempt": regen_attempt + 1,
                    "expected_fields": sorted(expected_fields),
                }
            )
        
        return tool_specs, tools_code, tool_selftest

    def _generate_tool_implementations(
        self,
        *,
        tools_code: str,
        sandbox_dir: Path,
        tool_specs: list[ToolSpec],
        topic: str,
        data_profile: dict[str, Any],
        sandbox: SandboxExecutor,
    ) -> str:
        """Return agent-authored tools code with minimal formatting cleanup.

        User requirement: do NOT inject/override any tool implementations here.
        We only validate and fail-fast so the agent can regenerate.
        """
        code = (tools_code or "").strip()
        if not code:
            raise RuntimeError("tools.py generation failed: empty tools code")

        # Validate: tool code must be data-driven (read local files) and avoid embedding datasets.
        code_ok, reasons = self._validate_tool_code(code)
        if not code_ok:
            raise RuntimeError(f"tools.py invalid (not data-driven): {reasons}")

        # Validate: submit_result must be implemented by the agent and persist the payload
        # only when the toolset expects it (e.g., full tool synthesis vs. augmentation).
        requires_submit_result = any(spec.name == self._SUBMIT_RESULT_TOOL for spec in tool_specs)
        if requires_submit_result:
            if "def submit_result" not in code:
                raise RuntimeError("tools.py missing required submit_result implementation (agent-authored).")
            if self._SUBMITTED_RESULT_FILE not in code:
                raise RuntimeError(
                    f"tools.py submit_result must persist to {self._SUBMITTED_RESULT_FILE} (agent-authored)."
                )

        # Validate: required tool functions should exist in the module source.
        try:
            tree = ast.parse(code)
            defined = {
                node.name
                for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
        except Exception as exc:
            raise RuntimeError(f"tools.py parse failed: {exc}")
        required = {spec.name for spec in tool_specs}
        missing = sorted(name for name in required if name not in defined)
        if missing:
            raise RuntimeError(f"tools.py missing required tool defs: {missing}")

        return code + ("\n" if not code.endswith("\n") else "")
    
    @staticmethod
    def _fix_file_paths(
        code: str,
        known_basenames: set[str] | None = None,
        *,
        add_imports: bool = True,
        add_base_dir: bool = True,
    ) -> str:
        """Convert relative file paths to absolute paths based on __file__.
        
        This ensures tools can find data files regardless of the current working directory.
        Uses regex-based replacement for reliability.
        """
        import re
        
        lines = code.splitlines()
        
        # Step 1: Add Path import if missing
        if add_imports:
            has_pathlib = any(
                "from pathlib import Path" in line
                or ("import pathlib" in line and "Path" in line)
                for line in lines
            )
            
            if not has_pathlib:
                # Find insertion point (after other imports)
                insert_idx = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")):
                        insert_idx = i + 1
                    elif stripped and not stripped.startswith("#") and insert_idx > 0:
                        break
                lines.insert(insert_idx, "from pathlib import Path")
        
        # Step 2: Add BASE_DIR definition if missing
        if add_base_dir:
            has_base_dir = any("BASE_DIR" in line and "=" in line for line in lines)
            
            if not has_base_dir:
                # Find insertion point (after imports, before first function/class)
                insert_idx = 0
                for i, line in enumerate(lines):
                    stripped = line.strip()
                    if stripped.startswith(("import ", "from ")):
                        insert_idx = i + 1
                    elif stripped and not stripped.startswith("#") and insert_idx > 0:
                        # Check if this is a function/class definition
                        if not (stripped.startswith(("def ", "class ", "@"))):
                            insert_idx = i
                        break
                lines.insert(insert_idx, "BASE_DIR = Path(__file__).parent")
        
        code = "\n".join(lines)
        
        # Step 3: Replace relative file paths with BASE_DIR / path
        # Match patterns like: "data/file.csv", 'data/file.csv', "records.json"
        # But avoid replacing if already using BASE_DIR or absolute paths
        file_ext_pattern = r'\.(csv|tsv|json|jsonl|ndjson|sqlite|sqlite3|db|txt|log|parquet)'
        
        def replace_file_path(match):
            full_match = match.group(0)
            quote_char = match.group(1)
            path = match.group(2)
            
            # Skip if already using BASE_DIR
            if "BASE_DIR" in code[max(0, match.start()-50):match.end()+50]:
                return full_match
            if "DATA_DIR" in code[max(0, match.start()-50):match.end()+50]:
                return full_match
            
            # Skip absolute paths
            if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
                return full_match
            
            # Skip if it's part of a larger expression that already uses BASE_DIR
            context_start = max(0, match.start() - 100)
            context = code[context_start:match.end()+50]
            if "BASE_DIR" in context:
                return full_match
            
            # Replace the path
            if "/" in path or "\\" in path:
                # Multi-part path: BASE_DIR / "part1" / "part2" / "file.ext"
                parts = [p for p in path.replace("\\", "/").split("/") if p]
                path_expr = " / ".join([f'{quote_char}{part}{quote_char}' for part in parts])
                return f'BASE_DIR / {path_expr}'
            else:
                # Single filename
                if known_basenames and path in known_basenames:
                    return f'BASE_DIR / {quote_char}data{quote_char} / {quote_char}{path}{quote_char}'
                return f'BASE_DIR / {quote_char}{path}{quote_char}'
        
        # Pattern: matches quoted strings that look like file paths
        # Matches: "data/file.csv", 'data/file.csv', "records.json"
        # Excludes: already absolute paths, paths in BASE_DIR expressions
        pattern = rf'(["\'])((?:data/|\./)?[^"\']+{file_ext_pattern})(["\'])'
        
        # Replace in code
        code = re.sub(pattern, replace_file_path, code)
        
        # Also handle os.path.exists("path") patterns
        def replace_os_path(match):
            prefix = match.group(1)  # "os.path.exists(" or similar
            quote = match.group(2)
            path = match.group(3)
            suffix = match.group(4)  # closing quote and paren
            
            # Skip if already absolute or using BASE_DIR
            if path.startswith("/") or (len(path) > 1 and path[1] == ":"):
                return match.group(0)
            
            # Replace
            if "/" in path or "\\" in path:
                parts = [p for p in path.replace("\\", "/").split("/") if p]
                path_expr = " / ".join([f'{quote}{p}{quote}' for p in parts])
                return f'{prefix}BASE_DIR / {path_expr}{suffix}'
            else:
                return f'{prefix}BASE_DIR / {quote}{path}{quote}{suffix}'
        
        os_path_pattern = rf'(os\.path\.(?:exists|join|isfile|isdir)\()(["\'])((?:data/|\./)?[^"\']+{file_ext_pattern})(["\']\))'
        code = re.sub(os_path_pattern, replace_os_path, code)
        
        return code

    @staticmethod
    def _convert_paths_to_strings(obj: Any) -> Any:
        """Recursively convert PosixPath objects to strings for JSON serialization."""
        if isinstance(obj, Path):
            return str(obj)
        elif isinstance(obj, dict):
            return {key: ToolSynthesisMixin._convert_paths_to_strings(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [ToolSynthesisMixin._convert_paths_to_strings(item) for item in obj]
        elif isinstance(obj, tuple):
            return tuple(ToolSynthesisMixin._convert_paths_to_strings(item) for item in obj)
        else:
            return obj

    def _self_test_tools(
        self,
        tool_specs: list[ToolSpec],
        sandbox: SandboxExecutor,
        topic: str,
        ctx: TaskContext,
        data_profile: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Lightweight tool smoke-test: ensure tool functions exist and are non-empty."""
        tools_path = sandbox.sandbox_dir / "tools.py"
        profile: dict[str, Any] = {}
        if not tools_path.exists():
            ctx.add_step({"type": "tool_self_test", "content": "tools.py missing"})
            return profile
        try:
            source = tools_path.read_text(encoding="utf-8")
        except Exception as exc:
            ctx.add_step({"type": "tool_self_test", "content": f"read_failed: {exc}"})
            return profile
        if not source.strip():
            ctx.add_step({"type": "tool_self_test", "content": "tools.py empty"})
            return profile
        try:
            tree = ast.parse(source)
        except Exception as exc:
            ctx.add_step({"type": "tool_self_test", "content": f"parse_failed: {exc}"})
            return profile

        func_nodes = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        def _is_docstring(stmt: ast.stmt) -> bool:
            return (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )

        def _is_empty_stmt(stmt: ast.stmt) -> bool:
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Raise):
                return True
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is Ellipsis:
                return True
            return False

        for spec in tool_specs:
            if spec.name == self._SUBMIT_RESULT_TOOL:
                continue
            node = func_nodes.get(spec.name)
            if node is None:
                profile[spec.name] = {"ok": False, "error": "missing_function"}
                continue
            body = list(node.body)
            if body and _is_docstring(body[0]):
                body = body[1:]
            meaningful = any(not _is_empty_stmt(stmt) for stmt in body)
            if not body or not meaningful:
                profile[spec.name] = {"ok": False, "error": "empty_body"}
            else:
                profile[spec.name] = {"ok": True}

        tool_names = [spec.name for spec in tool_specs if spec.name != self._SUBMIT_RESULT_TOOL]
        samples = self._collect_tool_samples(sandbox, tool_names)
        for name, info in samples.items():
            if not isinstance(info, dict):
                continue
            existing = profile.get(name)
            if isinstance(existing, dict) and existing.get("ok") is False:
                continue
            merged = dict(existing or {})
            merged.setdefault("ok", True)
            for key, value in info.items():
                if value is not None:
                    merged[key] = value
            profile[name] = merged

        profile_serializable = self._convert_paths_to_strings(profile)
        ctx.add_step(
            {
                "type": "tool_self_test",
                "content": json.dumps(profile_serializable, ensure_ascii=False)[:2000],
            }
        )
        return profile

    def _collect_tool_samples(
        self,
        sandbox: SandboxExecutor,
        tool_names: list[str],
    ) -> dict[str, Any]:
        if not tool_names:
            return {}
        names_json = json.dumps(sorted(tool_names), ensure_ascii=True)
        script = f"""python - <<'PY'
import importlib.util
import inspect
import json
from pathlib import Path

TOOLS_PATH = Path("tools.py")
TOOL_NAMES = {names_json}

def _truncate(value, limit=120):
    if isinstance(value, str) and len(value) > limit:
        return value[:limit] + "..."
    return value

def _simple_value(value):
    if isinstance(value, (str, int, float, bool)):
        return _truncate(value)
    return None

def _unwrap(value):
    if isinstance(value, dict) and "result" in value:
        if len(value) == 1 or (len(value) == 2 and "error" in value):
            return value.get("result")
    return value

def _enum_from_annotation(annotation):
    try:
        from enum import Enum
        from typing import get_args, get_origin
    except Exception:
        return None
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    origin = get_origin(annotation)
    if origin is None:
        return None
    for arg in get_args(annotation):
        enum_type = _enum_from_annotation(arg)
        if enum_type is not None:
            return enum_type
    return None

def _coerce_enum_value(value, enum_type):
    if enum_type is None:
        return value
    if value is None or isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        try:
            return enum_type(value)
        except Exception:
            member = enum_type.__members__.get(value)
            if member is not None:
                return member
    return value

def _collect_fields(sample):
    if isinstance(sample, dict):
        return list(sample.keys())
    return []

def _collect_sample_values(sample):
    values = {{}}
    if not isinstance(sample, dict):
        return values
    for key, raw in sample.items():
        if isinstance(raw, (str, int, float, bool)):
            values[key] = [_simple_value(raw)]
            continue
        if isinstance(raw, list):
            items = []
            for item in raw:
                simple = _simple_value(item)
                if simple is None:
                    continue
                if simple not in items:
                    items.append(simple)
                if len(items) >= 3:
                    break
            if items:
                values[key] = items
    return values

def _collect_empty_list_fields(sample):
    if not isinstance(sample, dict):
        return []
    empty = []
    for key, raw in sample.items():
        if isinstance(raw, list) and not raw:
            empty.append(key)
    return empty

def _safe_call(fn, *args, **kwargs):
    try:
        return True, fn(*args, **kwargs)
    except Exception as exc:
        return False, "call_failed: " + str(exc)

spec = importlib.util.spec_from_file_location("generated_tools", TOOLS_PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

available = {{name: getattr(module, name) for name in TOOL_NAMES if hasattr(module, name)}}
list_tools = {{name: fn for name, fn in available.items() if name.startswith("list_")}}
get_tools = {{name: fn for name, fn in available.items() if name.startswith("get_")}}

list_outputs = {{}}
profile = {{}}

for name, fn in list_tools.items():
    ok, result = _safe_call(fn)
    entry = {{"ok": bool(ok)}}
    if ok:
        result = _unwrap(result)
        if isinstance(result, list):
            options = []
            for item in result:
                simple = _simple_value(item)
                if simple is None:
                    continue
                options.append(simple)
                if len(options) >= 6:
                    break
            entry["options"] = options
            list_outputs[name] = options
        else:
            entry["options"] = []
    else:
        entry["error"] = result
    profile[name] = entry

def _base_from_list(name):
    if not (name.startswith("list_") and name.endswith("_options")):
        return ""
    return name[len("list_"):-len("_options")]

list_by_base = {{}}
for name, options in list_outputs.items():
    base = _base_from_list(name)
    if base:
        list_by_base[base] = options

for name, fn in get_tools.items():
    entry = {{"ok": True}}
    sample = None
    ok = True
    sig = inspect.signature(fn)
    try:
        from typing import get_type_hints
        type_hints = get_type_hints(fn)
    except Exception:
        type_hints = getattr(fn, "__annotations__", {{}}) or {{}}
    params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
    kwargs = {{}}
    if params:
        base = ""
        if name.startswith("get_") and name.endswith("_records"):
            base = name[len("get_"):-len("_records")]
        options = list_by_base.get(base, [])
        if options:
            first_param = params[0]
            ann = type_hints.get(first_param.name)
            enum_type = _enum_from_annotation(ann)
            kwargs[first_param.name] = _coerce_enum_value(options[0], enum_type)
        elif params[0].default is inspect._empty:
            ann = type_hints.get(params[0].name)
            enum_type = _enum_from_annotation(ann)
            if enum_type is not None:
                try:
                    kwargs[params[0].name] = list(enum_type)[0]
                except Exception:
                    ok = False
                    entry["error"] = "missing_required_params"
            else:
                ok = False
                entry["error"] = "missing_required_params"
    if ok and params:
        # Fill remaining required params with defaults if present.
        for param in params[1:]:
            if param.name in kwargs:
                continue
            if param.default is not inspect._empty:
                kwargs[param.name] = param.default
    if ok:
        called, result = _safe_call(fn, **kwargs)
        if not called:
            entry["ok"] = False
            entry["error"] = result
        else:
            result = _unwrap(result)
            if isinstance(result, list) and result:
                sample = result[0]
            elif isinstance(result, dict):
                sample = result
    if sample is not None:
        fields = _collect_fields(sample)
        entry["fields"] = fields[:10]
        entry["sample_values"] = _collect_sample_values(sample)
        entry["empty_list_fields"] = _collect_empty_list_fields(sample)
    else:
        entry.setdefault("fields", [])
    profile[name] = entry

print(json.dumps(profile, ensure_ascii=False))
PY"""
        result = sandbox.execute_bash(script, timeout_s=20)
        stdout = (result.get("stdout") or "").strip()
        if result.get("returncode") != 0:
            logger.debug("tool sample collection failed: %s", result.get("stderr"))
            return {}
        if not stdout:
            return {}
        try:
            parsed = json.loads(stdout)
        except Exception:
            logger.debug("tool sample collection parse failed")
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _expected_fields_from_format(submit_result_format: Any) -> set[str]:
        fmt = submit_result_format
        if isinstance(fmt, str):
            try:
                fmt = json.loads(fmt)
            except Exception:
                return set()
        fields: set[str] = set()
        if isinstance(fmt, dict):
            props = fmt.get("properties") if isinstance(fmt.get("properties"), dict) else None
            if props:
                fields.update([k for k in props.keys() if isinstance(k, str)])
            required = fmt.get("required")
            if isinstance(required, list):
                fields.update([k for k in required if isinstance(k, str)])
        if isinstance(fmt, list) and fmt and isinstance(fmt[0], dict):
            fields.update([k for k in fmt[0].keys() if isinstance(k, str)])
        return fields

    def _needs_tool_regeneration(
        self,
        tool_selftest: dict[str, Any],
        required_fields: set[str] | None = None,
    ) -> tuple[bool, list[str]]:
        reasons: list[str] = []
        if not tool_selftest:
            return True, ["selftest_missing"]
        union_fields = set()
        for name, info in tool_selftest.items():
            if name == self._SUBMIT_RESULT_TOOL:
                continue
            if not isinstance(info, dict):
                continue
            if info.get("ok") is False or info.get("error"):
                reasons.append(f"{name}:error")
                continue
            fields = set(info.get("fields") or [])
            union_fields |= {f for f in fields if isinstance(f, str)}
        if required_fields and union_fields and not (required_fields & union_fields):
            reasons.append("missing_required_fields")
        return bool(reasons), reasons

    def _register_task_tools(
        self,
        tool_specs: list[ToolSpec],
        sandbox: SandboxExecutor,
        ctx: TaskContext,
        *,
        tools_code: str | None = None,
    ) -> tuple[bool, list[str]]:
        """Register tools from tools.py.
        
        Returns:
            (success, error_reasons) tuple. If success is False, error_reasons contains error messages.
        """
        tool_specs = self._ensure_submit_result_tool(list(tool_specs))
        tools_path = sandbox.sandbox_dir / "tools.py"
        module = None
        import_error = None
        if tools_path.exists():
            try:
                # Tools are expected to import FastMCP directly; rely on the real mcp package.
                spec_module = importlib.util.spec_from_file_location("generated_tools", tools_path)
                if spec_module and spec_module.loader:
                    module = importlib.util.module_from_spec(spec_module)
                    spec_module.loader.exec_module(module)
            except Exception as exc:
                import_error = str(exc)
                logger.debug("Failed to import generated tools.py", exc_info=True)

        registration_errors: list[str] = []
        for spec in tool_specs:
            registered = False
            if module and hasattr(module, spec.name):
                handler = getattr(module, spec.name)
                if callable(handler):
                    sandbox.register_tool(
                        CallableTool(
                            name=spec.name,
                            description=spec.description,
                            handler=handler,
                        )
                    )
                    registered = True
            if not registered:
                # User requirement: ALL tool implementations must be agent-authored.
                error_msg = f"Missing agent-authored tool implementation in tools.py: {spec.name}"
                if import_error:
                    error_msg += f"\nImport error: {import_error}"
                elif not module:
                    error_msg += "\nFailed to import tools.py module (check logs for details)"
                elif not hasattr(module, spec.name):
                    available_names = [name for name in dir(module) if not name.startswith("_") and callable(getattr(module, name, None))]
                    error_msg += f"\nAvailable functions in tools.py: {available_names}"
                registration_errors.append(error_msg)
        
        if registration_errors:
            return False, registration_errors

        sandbox.set_tool_call_allowlist({spec.name for spec in tool_specs})
        ctx.add_step(
            {
                "type": "tool_registration",
                "registered_tools": [spec.name for spec in tool_specs],
                "tools_code_preview": (tools_code or "")[:300],
            }
        )
        self.writer.record_steps(ctx.task_id, self.agent_type, ctx.history)
        return True, []

    def _augment_toolset(
        self,
        topic: str,
        records: list[dict[str, Any]],
        existing_specs: list[ToolSpec],
        data_profile: dict[str, Any] | None,
        ctx: TaskContext,
        sandbox: SandboxExecutor,
    ) -> tuple[list[ToolSpec], str, bool]:
        """Generate incremental tools to augment the current toolset."""
        existing_names = {spec.name for spec in existing_specs}
        data_profile_json = json.dumps(data_profile or {}, ensure_ascii=False)
        field_inventory_json = json.dumps(self._build_field_inventory(data_profile or {}), ensure_ascii=True)
        if len(field_inventory_json) > self._MAX_FIELD_INVENTORY_SIZE:
            field_inventory_json = field_inventory_json[:self._MAX_FIELD_INVENTORY_SIZE] + "..."
        base_rules = (
            "- CRITICAL: File paths must be absolute. Use BASE_DIR = Path(__file__).parent and BASE_DIR / \"data\" / \"file.csv\".\n"
            "- NO network, NO external APIs. Only read the listed local files.\n"
            "- Avoid generic free-text parameters; use enums derived from real fields.\n"
            "- Use @mcp.tool(description=...) with a plain string literal.\n"
            "- Implement real logic; no stubs, no pass/raise placeholders.\n"
            "- Do NOT append raw csv.DictReader rows; map into a new dict with explicit keys.\n"
            "- If you read CSV files, detect the delimiter (csv.Sniffer) or use the file profile; do not assume commas.\n"
            "- When parsing markdown/text, strip formatting markers like **, *, and trailing colons from extracted fields.\n"
            "- Avoid TypedDict/custom class return annotations; use list[dict[str, Any]] or dict[str, Any].\n"
            "- Keep return types stable; if a field is missing, use a type-appropriate default (e.g., \"\" for strings, 0 for numbers, [] for lists) instead of None.\n"
            "- Do NOT define submit_result here.\n"
        )
        max_tokens = getattr(ctx.request, "max_tokens", 10000)
        attempt_errors: list[str] = []
        last_tools_code = ""
        for attempt in range(1, self._MAX_REGEN_ATTEMPTS + 1):
            error_context = ""
            if attempt_errors:
                error_context = (
                    "Previous attempts encountered the following issues:\n"
                    + "\n".join(f"- {error}" for error in attempt_errors)
                    + "\n\n"
                )
            prompt = (
                f"{error_context}"
                "You are augmenting an existing toolset to make a task solvable and verifiable.\n"
                "Return Python code defining 1-2 NEW tools decorated with @mcp.tool(), avoiding duplicates.\n"
                "Include `from mcp.server.fastmcp import FastMCP` and `mcp = FastMCP(\"Tools\")` in the output.\n"
                "Do NOT call mcp.run() or start any server.\n"
                "Hard rules:\n"
                f"{base_rules}"
                "Constraints:\n"
                f"- Tool names must be unique, snake_case, and NOT {', '.join(repr(name) for name in self._FILTERED_TOOL_NAMES)}.\n"
                "- Implement real logic that reads ONLY local files (CSV/JSON/TXT/SQLite) or the provided records; no stubs, no pass/raise placeholders.\n"
                "- Deterministic: set random.seed(0) if any randomness is used.\n"
                "Only output a single Python code block.\n"
                f"Field Inventory (use these field names for parameters): {field_inventory_json}\n"
                f"Topic: {topic}\n"
                f"Existing tools: {sorted(existing_names)}\n"
                f"Records (JSON sample): {json.dumps(records[:5], ensure_ascii=False)}\n"
                f"Data profile (JSON): {data_profile_json[:800]}\n"
            )
            raw = self.llm.simple_complete(prompt, temperature=0.55, max_tokens=max_tokens)
            ctx.add_step({"type": "tool_augmentation", "attempt": attempt, "content": raw})

            tools_code = self._sanitize_llm_code(raw)
            last_tools_code = tools_code or ""
            if not tools_code:
                attempt_errors = ["empty_tool_code"]
                ctx.add_step(
                    {"type": "tool_augmentation_validation_failed", "attempt": attempt, "errors": attempt_errors}
                )
                continue

            try:
                tree = ast.parse(tools_code)
            except Exception as exc:
                attempt_errors = [f"parse_failed:{exc}"]
                ctx.add_step(
                    {"type": "tool_augmentation_validation_failed", "attempt": attempt, "errors": attempt_errors}
                )
                continue

            typed_dict_names = self._typed_dict_class_names(tools_code)
            typed_dict_returns = []
            for node in tree.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if self._annotation_uses_names(node.returns, typed_dict_names):
                    typed_dict_returns.append(node.name)
            if typed_dict_returns:
                attempt_errors = [f"typed_dict_return_disallowed:{name}" for name in typed_dict_returns]
                ctx.add_step(
                    {"type": "tool_augmentation_validation_failed", "attempt": attempt, "errors": attempt_errors}
                )
                continue

            specs = self._extract_mcp_tools_from_python(tools_code)
            if not specs:
                attempt_errors = ["no_tools_found"]
                ctx.add_step(
                    {"type": "tool_augmentation_validation_failed", "attempt": attempt, "errors": attempt_errors}
                )
                continue

            candidate_names = set(existing_names)
            new_specs: list[ToolSpec] = []
            for spec in specs:
                if spec.name in self._FILTERED_TOOL_NAMES:
                    continue
                if spec.name in candidate_names:
                    continue
                candidate_names.add(spec.name)
                new_specs.append(
                    spec.copy(
                        update={
                            "description": spec.description
                            or f"Augmented tool for {topic}",
                            "meta": (spec.meta or {}) | {"topic": topic, "augmented": True},
                        }
                    )
                )

            if not new_specs:
                attempt_errors = ["no_new_tools"]
                ctx.add_step(
                    {"type": "tool_augmentation_validation_failed", "attempt": attempt, "errors": attempt_errors}
                )
                continue

            code_ok, code_reasons = self._validate_tool_code(tools_code)
            if not code_ok:
                prompt_errors: list[str] = []
                for reason in code_reasons:
                    if reason == "raw_csv_rows_returned":
                        prompt_errors.append(
                            "raw_csv_rows_returned (do NOT append DictReader rows; create a new dict with explicit keys)"
                        )
                    else:
                        prompt_errors.append(reason)
                attempt_errors = prompt_errors
                ctx.add_step(
                    {
                        "type": "tool_augmentation_invalid_tool_code",
                        "attempt": attempt,
                        "tool_code_reasons": code_reasons,
                    }
                )
                continue

            output_keys = self._extract_tool_output_keys(tools_code)
            if output_keys:
                enriched: list[ToolSpec] = []
                for spec in new_specs:
                    if spec.name == self._SUBMIT_RESULT_TOOL:
                        enriched.append(spec)
                        continue
                    meta = dict(spec.meta or {})
                    keys = output_keys.get(spec.name)
                    if keys:
                        meta["output_keys"] = keys
                    enriched.append(spec.copy(update={"meta": meta}))
                new_specs = enriched

            tools_path = sandbox.sandbox_dir / "tools.py"
            implemented_code = self._generate_tool_implementations(
                tools_code=tools_code,
                sandbox_dir=sandbox.sandbox_dir,
                tool_specs=new_specs,
                topic=topic,
                data_profile=data_profile or {},
                sandbox=sandbox,
            )
            mode = "a" if tools_path.exists() else "w"
            with tools_path.open(mode, encoding="utf-8") as handle:
                if tools_path.exists():
                    handle.write("\n\n")
                handle.write(implemented_code)
            # Ensure mcp is installed before import test
            mcp_installed, _ = self._ensure_mcp_installed(sandbox)
            if not mcp_installed:
                logger.warning("mcp installation failed during tool augmentation, continuing anyway")
            sandbox.execute_bash("python -m py_compile tools.py")
            sandbox.execute_bash('python -c "import tools; print(\'TOOLS_IMPORT_OK\')"')

            self._register_task_tools(new_specs, sandbox, ctx, tools_code=tools_code)
            combined_specs = self._ensure_submit_result_tool(existing_specs + new_specs)
            return combined_specs, tools_code, True

        ctx.add_step(
            {"type": "tool_augmentation_failed", "errors": attempt_errors or ["unknown_failure"]}
        )
        return existing_specs, last_tools_code, False

    @staticmethod
    def _extract_mcp_tools(raw: str) -> list[ToolSpec]:
        """
        Extract MCP tool decorators and generate ToolSpec objects from the provided Python code string.

        Args:
        - raw: The Python code string containing function definitions with @mcp.tool decorators.

        Returns:
        - A list of ToolSpec objects.
        """
        text = (raw or "").strip()
        if not text:
            return []

        # Extract code blocks wrapped in markdown
        code_blocks = BaseAgent._extract_code_blocks(text)
        mcp_tools: list[ToolSpec] = []
        if code_blocks:
            for code in code_blocks:
                mcp_tools.extend(ToolSynthesisMixin._extract_mcp_tools_from_python(code))
        return mcp_tools

    @staticmethod
    def _sanitize_tool_block(function_string: str) -> str:
        lines = function_string.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        if not lines:
            return function_string
        def_idx = None
        for idx, line in enumerate(lines):
            stripped = line.lstrip()
            if stripped.startswith("def ") or stripped.startswith("async def "):
                def_idx = idx
                break
        if def_idx is None:
            return function_string
        decorators = [line for line in lines[:def_idx] if line.lstrip().startswith("@")]
        trimmed = decorators + lines[def_idx:]
        def_line = trimmed[len(decorators)]
        def_indent = len(def_line) - len(def_line.lstrip(" "))
        if def_indent <= 0:
            return "\n".join(trimmed)
        fixed: list[str] = []
        for idx, line in enumerate(trimmed):
            if idx < len(decorators):
                fixed.append(line.lstrip())
                continue
            if not line.strip():
                fixed.append("")
                continue
            fixed.append(line[def_indent:] if len(line) >= def_indent else line.lstrip())
        return "\n".join(fixed)

    @staticmethod
    def _build_tool_spec(function_string: str, name_hint: str) -> ToolSpec | None:
        try:
            return ToolSpec.from_function_string(function_string)
        except (ValueError, SyntaxError, IndentationError) as exc:
            repaired = ToolSynthesisMixin._sanitize_tool_block(function_string)
            if repaired != function_string:
                try:
                    return ToolSpec.from_function_string(repaired)
                except (ValueError, SyntaxError, IndentationError) as repaired_exc:
                    logger.error(
                        "Error creating ToolSpec for function %s after repair: %s",
                        name_hint,
                        repaired_exc,
                    )
                    return None
            logger.error("Error creating ToolSpec for function %s: %s", name_hint, exc)
            return None

    @staticmethod
    def _extract_mcp_tools_from_python(raw: str) -> list[ToolSpec]:
        text = (raw or "").strip()
        if not text:
            return []
        normalized = re.sub(r"@mcp\.tool\s*(?=\n)", "@mcp.tool()", text)
        mcp_tools: list[ToolSpec] = []
        try:
            module = ast.parse(normalized)
            lines = normalized.splitlines()
            for node in module.body:
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if not any(
                    isinstance(dec, ast.Call)
                    and isinstance(dec.func, ast.Attribute)
                    and dec.func.attr == "tool"
                    for dec in node.decorator_list
                ):
                    continue
                if node.lineno is None or node.end_lineno is None:
                    continue
                decorator_lines = [
                    dec.lineno for dec in node.decorator_list if getattr(dec, "lineno", None)
                ]
                start_line = min(decorator_lines) if decorator_lines else node.lineno
                function_string = "\n".join(lines[start_line - 1 : node.end_lineno]).strip()
                tool = ToolSynthesisMixin._build_tool_spec(function_string, node.name)
                if tool is not None:
                    mcp_tools.append(tool)
            if mcp_tools:
                return mcp_tools
        except Exception:
            logger.debug("AST tool extraction failed; falling back to regex.", exc_info=True)

        # Regex fallback for malformed code blocks.
        mcp_pattern = r"(@mcp\.tool(?:\((.*?)\))?\s*def\s+([a-zA-Z_][a-zA-Z0-9_]*\s*\([^\)]*\))\s*(->\s*[^:]+)?\s*:([\s\S]+?))(?=\n\s*@mcp|\Z)"
        matches = re.findall(mcp_pattern, normalized, re.DOTALL)
        for match in matches:
            function_signature = match[2]
            function_string = match[0]
            tool = ToolSynthesisMixin._build_tool_spec(function_string, function_signature)
            if tool is not None:
                mcp_tools.append(tool)
        return mcp_tools
