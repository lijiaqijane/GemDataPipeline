from __future__ import annotations

from agent_gem.core.task_schema import ToolSpec


def test_from_function_string_reads_decorator_metadata_and_signature() -> None:
    code = """
@mcp.tool(
    name="find_products",
    description="Search the product catalog with optional category filtering.",
    meta={"version": "1.2", "author": "product-team"},
)
def search_products_implementation(query: str, category: str | None = None) -> list[dict]:
    \"""Internal function description (ignored if description is provided above).\"""
    raise RuntimeError("should not run")
"""
    spec = ToolSpec.from_function_string(code)

    assert spec.name == "find_products"
    assert spec.description == "Search the product catalog with optional category filtering."
    assert spec.meta == {
        "version": "1.2",
        "author": "product-team",
    }
    assert set(spec.parameters.get("properties", {}).keys()) == {"query", "category"}


def test_from_function_string_falls_back_to_docstring_description() -> None:
    code = """
@mcp.tool(name="doc_only")
def impl(x: int) -> str:
    \"""Docstring description.\"""
    return str(x)
"""
    spec = ToolSpec.from_function_string(code)
    assert spec.name == "doc_only"
    assert spec.description == "Docstring description."


def test_from_function_string_accepts_positional_name_and_ignores_top_level_code() -> None:
    code = """
raise RuntimeError("top-level should not run")

@mcp.tool("positional_name")
def impl(x: int) -> int:
    return x
"""
    spec = ToolSpec.from_function_string(code)
    assert spec.name == "positional_name"


def test_from_function_string_stashes_unknown_decorator_kwargs() -> None:
    code = """
@mcp.tool(name="extras", foo="bar", meta={"x": 1})
def impl(x: int) -> int:
    return x
"""
    spec = ToolSpec.from_function_string(code)
    assert spec.meta == {"x": 1, "_decorator_extras": {"foo": "bar"}}


def test_from_function_string_validates_decorator_types() -> None:
    code = """
@mcp.tool(name=123)
def impl(x: int) -> int:
    return x
"""
    try:
        ToolSpec.from_function_string(code)
    except TypeError as exc:
        assert "name" in str(exc)
    else:
        raise AssertionError("Expected TypeError")
