"""Utility functions for synthesis."""

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


def parse_json_response(raw: str, max_retries: int = 3) -> Any:
    """Parse JSON from LLM response with robust error handling.
    
    Handles common LLM hallucination issues:
    - Trailing commas
    - Unescaped quotes
    - Markdown code blocks
    - Multiple JSON objects
    """
    text = raw.strip()
    
    def try_load(candidate: str) -> Any:
        """Try to load JSON with common fixes."""
        # Attempt 1: Direct parsing
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass
        
        # Attempt 2: Fix trailing comma
        try:
            fixed = re.sub(r',\s*}', '}', candidate)
            fixed = re.sub(r',\s*]', ']', fixed)
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        # Attempt 3: Fix unescaped quotes (simple cases)
        try:
            fixed = candidate.replace("'", '"')
            return json.loads(fixed)
        except json.JSONDecodeError:
            pass
        
        raise json.JSONDecodeError(f"Unable to parse JSON: {candidate[:100]}", candidate, 0)
    
    # 尝试1: 直接解析整个文本
    try:
        return try_load(text)
    except json.JSONDecodeError:
        pass

    # 尝试2: 从Markdown代码块中提取
    fence = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.S | re.MULTILINE)
    if fence:
        candidate = fence.group(1)
        try:
            return try_load(candidate)
        except json.JSONDecodeError:
            pass

    # 尝试3: 查找第一个完整的JSON对象
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return try_load(candidate)
        except json.JSONDecodeError:
            pass

    # 尝试4: 查找第一个完整的JSON数组
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1 and end > start:
        candidate = text[start : end + 1]
        try:
            return try_load(candidate)
        except json.JSONDecodeError:
            pass

    raise json.JSONDecodeError(f"Unable to extract valid JSON from LLM response. First 200 chars: {text[:200]}", raw, 0)

