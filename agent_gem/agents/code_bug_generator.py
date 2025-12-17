"""Generate synthetic bugs and corresponding issue/PR information for code tasks."""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from agent_gem.llm import LLMClient
from agent_gem.agents.code_repo_analyzer import RepoMetadata

logger = logging.getLogger(__name__)


def extract_json_from_response(response: str) -> str:
    """Extract JSON from LLM response, handling markdown code blocks."""
    response = response.strip()
    
    # Remove markdown code blocks
    if response.startswith("```json"):
        response = response[7:]
    elif response.startswith("```"):
        response = response[3:]
    if response.endswith("```"):
        response = response[:-3]
    response = response.strip()
    
    # Try to find JSON object boundaries
    if not response.startswith("{"):
        # Look for the first { character
        match = re.search(r'\{', response)
        if match:
            response = response[match.start():]
    
    if not response.endswith("}"):
        # Look for the last } character
        match = re.search(r'\}[^}]*$', response)
        if match:
            response = response[:match.end()]
    
    return response


@dataclass
class BugInfo:
    """Information about a synthetic bug."""

    bug_id: str
    bug_title: str
    bug_description: str
    bug_type: str  # "logic", "performance", "security", "memory", "type"
    severity: str  # "low", "medium", "high", "critical"
    affected_file: str
    affected_function: str
    bug_location: str  # line range
    buggy_code: str


@dataclass
class IssuePRInfo:
    """Issue and PR information for a bug fix task."""

    issue_title: str
    issue_body: str
    issue_labels: List[str]
    pr_title: str
    pr_description: str
    pr_changes_summary: str
    fixed_code: str
    test_additions: str


class BugGenerator:
    """Generate synthetic bugs based on repository analysis."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_bug(
        self,
        metadata: RepoMetadata,
        source_code: str,
        file_path: str,
        max_tokens: int = 1500,
        temperature: float = 0.7,
    ) -> Optional[BugInfo]:
        """
        Generate a synthetic bug in the given source code.
        
        Args:
            metadata: Repository metadata
            source_code: Source code content
            file_path: Path to the file in the repository
            
        Returns:
            BugInfo with bug details
        """
        bug_type = random.choice(
            ["logic", "performance", "security", "memory", "off-by-one"]
        )

        prompt = f"""You are a code bug generator. Given a code snippet from a repository, generate a realistic synthetic bug that:
1. Is subtle but realistic (not obvious)
2. Would cause tests to fail
3. Is fixable with a small patch
4. Is appropriate for the {metadata.language} programming language

Repository: {metadata.repo_name}
Language: {metadata.language}
File: {file_path}
Bug type to introduce: {bug_type}

Source code:
```{metadata.language}
{source_code[:2000]}
```

Generate a JSON response with:
{{
    "bug_id": "BUG-XXXX",
    "bug_title": "Brief title of the bug",
    "bug_description": "Detailed description of what the bug is and why it's a problem",
    "bug_type": "{bug_type}",
    "severity": "low|medium|high|critical",
    "affected_function": "function_name",
    "bug_location": "line X-Y",
    "buggy_code": "The buggy code snippet (5-10 lines)",
    "root_cause": "What causes the bug",
    "impact": "How this bug affects the system"
}}

Return only valid JSON."""

        try:
            response = self.llm.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract JSON from response
            response = extract_json_from_response(response)
            data = json.loads(response)

            bug = BugInfo(
                bug_id=data.get("bug_id", "BUG-0000"),
                bug_title=data.get("bug_title", "Unknown bug"),
                bug_description=data.get("bug_description", ""),
                bug_type=data.get("bug_type", "logic"),
                severity=data.get("severity", "medium"),
                affected_file=file_path,
                affected_function=data.get("affected_function", "unknown"),
                bug_location=data.get("bug_location", "unknown"),
                buggy_code=data.get("buggy_code", ""),
            )

            logger.info(
                f"Generated bug: {bug.bug_id} ({bug.bug_type}) in {file_path}"
            )
            return bug

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to generate bug: {e}")
            # Log the raw response for debugging
            logger.debug(f"Raw LLM response: {response[:500] if 'response' in locals() else 'N/A'}")
            return None

    def generate_multiple_bugs(
        self,
        metadata: RepoMetadata,
        source_codes: Dict[str, str],
        num_bugs: int = 3,
    ) -> List[BugInfo]:
        """
        Generate multiple bugs from different files.
        
        Args:
            metadata: Repository metadata
            source_codes: Dict mapping file paths to source code
            num_bugs: Number of bugs to generate
            
        Returns:
            List of BugInfo objects
        """
        bugs: List[BugInfo] = []
        files = list(source_codes.items())

        for _ in range(num_bugs):
            if not files:
                break

            file_path, code = random.choice(files)
            bug = self.generate_bug(metadata, code, file_path)

            if bug:
                bugs.append(bug)

        logger.info(f"Generated {len(bugs)} bugs")
        return bugs


class IssuePRGenerator:
    """Generate GitHub issue and PR descriptions for bug fixes."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_issue_and_pr(
        self, bug: BugInfo, metadata: RepoMetadata, fixed_code: Optional[str] = None,
        max_tokens: int = 2000, temperature: float = 0.6
    ) -> Optional[IssuePRInfo]:
        """
        Generate GitHub issue and PR descriptions for a bug.
        
        Args:
            bug: BugInfo object
            metadata: Repository metadata
            fixed_code: Optional fixed code (if not provided, will be generated)
            
        Returns:
            IssuePRInfo with issue and PR details
        """
        prompt = f"""You are a GitHub issues and pull requests generator. Create realistic GitHub issue and PR descriptions for a bug fix.

Repository: {metadata.repo_name}
Language: {metadata.language}
Bug Title: {bug.bug_title}
Bug Description: {bug.bug_description}
Bug Type: {bug.bug_type}
Severity: {bug.severity}
Affected File: {bug.affected_file}
Buggy Code:
```{metadata.language}
{bug.buggy_code}
```

Generate a JSON response with:
{{
    "issue_title": "Title for the GitHub issue",
    "issue_body": "Detailed issue description in markdown, including reproduction steps",
    "issue_labels": ["label1", "label2"],
    "pr_title": "Title for the pull request",
    "pr_description": "Detailed PR description in markdown",
    "pr_changes_summary": "Summary of what was changed and why",
    "fixed_code": "The fixed code (corrected version of buggy_code)",
    "test_additions": "Test code that would catch this bug (pytest/unittest format)"
}}

Return only valid JSON."""

        try:
            response = self.llm.chat_completion(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            
            # Extract JSON from response
            response = extract_json_from_response(response)
            data = json.loads(response)

            issue_pr = IssuePRInfo(
                issue_title=data.get("issue_title", f"Fix: {bug.bug_title}"),
                issue_body=data.get("issue_body", bug.bug_description),
                issue_labels=data.get("issue_labels", ["bug"]),
                pr_title=data.get("pr_title", f"Fix: {bug.bug_title}"),
                pr_description=data.get("pr_description", ""),
                pr_changes_summary=data.get("pr_changes_summary", ""),
                fixed_code=data.get("fixed_code", ""),
                test_additions=data.get("test_additions", ""),
            )

            logger.info(f"Generated issue and PR for bug: {bug.bug_id}")
            return issue_pr

        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Failed to generate issue/PR: {e}")
            # Log the raw response for debugging
            if 'response' in locals():
                logger.error(f"Raw LLM response length: {len(response)}")
                logger.error(f"Raw LLM response (first 1000 chars): {response[:1000]}")
                logger.error(f"Raw LLM response (last 500 chars): {response[-500:]}")
                
                # Try to salvage what we can - create a basic Issue/PR with what we have
                logger.info("Attempting to create basic Issue/PR from bug info...")
                try:
                    return IssuePRInfo(
                        issue_title=f"Fix: {bug.bug_title}",
                        issue_body=f"## Bug Description\n\n{bug.bug_description}\n\n## Affected Code\n\n```python\n{bug.buggy_code}\n```",
                        issue_labels=["bug", bug.bug_type],
                        pr_title=f"Fix: {bug.bug_title}",
                        pr_description=f"This PR fixes a {bug.bug_type} bug in {bug.affected_file}.",
                        pr_changes_summary="Applied fix to resolve the issue",
                        fixed_code="# TODO: Generate fixed code",
                        test_additions="# TODO: Generate tests",
                    )
                except Exception as fallback_error:
                    logger.error(f"Fallback creation failed: {fallback_error}")
            return None

    def batch_generate(
        self, bugs: List[BugInfo], metadata: RepoMetadata
    ) -> Dict[str, IssuePRInfo]:
        """
        Generate issue/PR for multiple bugs.
        
        Args:
            bugs: List of BugInfo objects
            metadata: Repository metadata
            
        Returns:
            Dict mapping bug_id to IssuePRInfo
        """
        result: Dict[str, IssuePRInfo] = {}

        for bug in bugs:
            issue_pr = self.generate_issue_and_pr(bug, metadata)
            if issue_pr:
                result[bug.bug_id] = issue_pr

        logger.info(f"Generated {len(result)} issue/PR pairs")
        return result
