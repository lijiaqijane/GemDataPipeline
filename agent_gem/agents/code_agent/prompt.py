FEATURE_REQUEST_PROMPT = """You are a senior engineer drafting a concise feature request for a code entity.

Context you must use:
- Repository: {repo_name}
- Entity: {entity_name}
- File: {file_path}
- Location: line {start_line}
- Signature: {signature_content}
- Docstring or summary: {docstring}

Implementation snippet:
{body_excerpt}

Write a feature request issue with:
Title: one-line capability statement
Context: bullet points about current behavior and limitation
Proposal: bullet list of the desired change
Acceptance Criteria: bullets that are testable

Keep the tone professional, stay under 300 words, avoid code fences, and focus on developer-facing guidance.
"""
