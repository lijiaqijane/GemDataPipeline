from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Dict, List, Optional

from agent_gem.core.task_schema import EvaluationCriteria, TaskDefinition, TaskPackage, ToolSpec
from agent_gem.core.validation import validate_task_package

from .base import BaseAgent

if TYPE_CHECKING:  # pragma: no cover
    from agent_gem.generator import GenerationRequest


logger = logging.getLogger(__name__)


class GeneralAgent(BaseAgent):
    agent_type = "general_agent"
    description = (
        "Automatic environment-synthesis agent that creates diverse, verifiable tasks with growing toolsets"
    )

    def generate(self, request: GenerationRequest) -> Optional[TaskPackage]:
        """Generate tasks following the agentic workflow"""
        self._configure_for_request(request)
        logger.info(
            "[agent:%s] Starting task synthesis workflow (topic=%s, difficulty=%s)",
            self.agent_type,
            request.topic or "auto-generate",
            request.difficulty,
        )

        # Step 1: Gather data and build database
        self._gather_data_and_build_database(request.topic)

        # Step 2: Synthesize task-specific tools
        self._synthesize_tools(request.topic)

        # Step 3: Generate initial simple task and iterate
        packages = self._iterative_task_synthesis(request)

        return self._validate_packages(packages, request)

    def _gather_data_and_build_database(self, topic: Optional[str]) -> None:
        """Gather relevant data using bash and search tools"""
        if not topic:
            topic = "general knowledge tasks"

        logger.info("[agent:%s] Gathering data for topic: %s", self.agent_type, topic)

        # Use search tool to find relevant information
        search_results = self._execute_tool(
            "search",
            {"query": f"information about {topic} for creating challenging tasks"},
        )

        # Use bash to process and organize data
        # For example, download datasets or process information
        bash_results = self._execute_tool(
            "bash",
            {"command": "mkdir -p /tmp/data && echo 'Data organization complete'"},
        )

        # Simulate database population (in real implementation, this would be more complex)
        self.task_state.database_content = {
            "topic": topic,
            "search_results": search_results,
            "metadata": {
                "data_points": 100,
                "last_updated": "2024-01-01",
                "sources": ["web_search", "curated_datasets"],
            },
        }

        logger.info(
            "[agent:%s] Database built with %d data points",
            self.agent_type,
            self.task_state.database_content.get("metadata", {}).get("data_points", 0),
        )

    def _synthesize_tools(self, topic: Optional[str]) -> None:
        """Synthesize task-specific tools based on the database"""
        logger.info("[agent:%s] Synthesizing task-specific tools", self.agent_type)

        prompt = f"""
        Based on the topic "{topic}" and the available database, design 3-5 specialized tools 
        that would be useful for solving tasks in this domain. Each tool should be:
        1. Specific to the domain
        2. Implementable as a Python function
        3. Useful for solving challenging but verifiable tasks
        
        Database context: {json.dumps(self.task_state.database_content, indent=2)[:500]}...
        
        Return a JSON array of tool specifications with:
        - tool_name: Unique identifier
        - tool_description: What it does
        - tool_functionality: Python function signature and behavior
        - implementation: Actual Python code (optional, can be generated later)
        """

        response = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1500,
        )

        try:
            tools_data = self._extract_json(response)
            if isinstance(tools_data, list):
                for tool_data in tools_data:
                    tool_spec = ToolSpec(
                        tool_name=tool_data.get("tool_name", ""),
                        tool_description=tool_data.get("tool_description", ""),
                        tool_functionality=tool_data.get("tool_functionality", ""),
                    )
                    self.task_state.current_tools.append(tool_spec)

                    # Store implementation if provided
                    if "implementation" in tool_data:
                        self.task_state.tool_implementations[tool_spec.tool_name] = tool_data[
                            "implementation"
                        ]

                logger.info(
                    "[agent:%s] Synthesized %d new tools",
                    self.agent_type,
                    len(tools_data),
                )
        except Exception as e:
            logger.warning("[agent:%s] Failed to synthesize tools: %s", self.agent_type, e)
            # Use default tools as fallback

    def _iterative_task_synthesis(self, request: GenerationRequest) -> List[TaskPackage]:
        """Generate tasks iteratively with increasing difficulty"""
        packages = []

        # Start with simple task
        current_difficulty = "easy"
        max_attempts = 5  # Prevent infinite loops

        for attempt in range(max_attempts):
            logger.info(
                "[agent:%s] Generating task at difficulty: %s (attempt %d)",
                self.agent_type,
                current_difficulty,
                attempt + 1,
            )

            # Generate task at current difficulty
            task_pkg = self._generate_single_task(request, current_difficulty)
            if not task_pkg:
                break

            # Test if task can be solved with current tools
            if not self._can_solve_with_current_tools(task_pkg):
                # Augment toolset if needed
                self._augment_toolset(task_pkg.task)
                continue

            # Test solution and verification
            if self._test_solution_verification(task_pkg):
                packages.append(task_pkg)

                # Check if we've reached target difficulty
                if self._difficulty_level(current_difficulty) >= self._difficulty_level(request.difficulty):
                    break

                # Increase difficulty for next iteration
                current_difficulty = self._increase_difficulty(current_difficulty)
                self.task_state.current_difficulty = current_difficulty
            else:
                # Try to fix solution/verification
                fixed_pkg = self._fix_solution_verification(task_pkg)
                if fixed_pkg:
                    packages.append(fixed_pkg)
                    current_difficulty = self._increase_difficulty(current_difficulty)
                else:
                    # If can't fix, try with different task
                    continue

        return packages if packages else [self._fallback_package(request)]

    def _generate_single_task(self, request: GenerationRequest, difficulty: str) -> Optional[TaskPackage]:
        """Generate a single task at specified difficulty"""
        prompt = self._build_task_generation_prompt(request, difficulty)

        raw = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.65,
            max_tokens=2000,
        )

        packages = self._parse_response(raw, request)
        return packages[0] if packages else None

    def _build_task_generation_prompt(self, request: GenerationRequest, difficulty: str) -> str:
        """Build prompt for task generation with current context"""
        tools_description = "\n".join(
            [
                f"- {tool.tool_name}: {tool.tool_description} ({tool.tool_functionality})"
                for tool in self.task_state.current_tools
            ]
        )

        return f"""You are a task synthesis agent creating challenging but verifiable tasks.

CURRENT CONTEXT:
- Topic: {request.topic or "General domain"}
- Database contains: {json.dumps(self.task_state.database_content, indent=2)[:300]}...
- Available tools:
{tools_description}

TASK REQUIREMENTS:
1. Create a {difficulty} difficulty task that is HARD to solve but EASY to verify
2. Task must be solvable ONLY using the available tools (no direct database access)
3. Solution function must only call tool functions or perform logical computations
4. Verification function must automatically validate the solution
5. Task should leverage the database content appropriately

FORMAT REQUIREMENTS:
Return a JSON object with:
{{
    "task_title": "Descriptive title",
    "task_content": "Detailed task description",
    "submit_result_format": "{request.submit_result_format}",
    "tool_set": [list of tool specs to use],
    "evaluation_criteria": {{
        "correctness": "how correctness is determined",
        "diversity": "how diverse the solution space is",
        "complexity": "task complexity level",
        "solution_verifiability": "how easily solution can be verified"
    }},
    "difficulty_level": "{difficulty}",
    "solution": "Python function that solves the task using only tool calls",
    "verification": "Python function that validates the solution"
}}

SOLUTION FUNCTION CONSTRAINTS:
- Must be named "solve"
- Takes one parameter: tools (a dict of tool functions)
- Can only call functions from the tools parameter
- Cannot access database directly
- Must return result in the specified format

VERIFICATION FUNCTION CONSTRAINTS:
- Must be named "verify"
- Takes two parameters: tools and answer
- Must return a boolean (True if answer is correct)
- Should be deterministic and automated

Generate the task now:"""

    def _can_solve_with_current_tools(self, task_pkg: TaskPackage) -> bool:
        """Check if task can be solved with current toolset"""
        # Extract tool names from solution
        solution = task_pkg.solution
        required_tools = set()

        # Simple pattern matching for tool calls
        # In production, you'd want a more robust AST-based analysis
        tool_pattern = r'tools\[["\']([^"\']+)["\']\]|tools\.(\w+)'
        matches = re.findall(tool_pattern, solution)
        for match in matches:
            tool_name = match[0] or match[1]
            if tool_name:
                required_tools.add(tool_name)

        # Check if all required tools are available
        available_tools = {tool.tool_name for tool in self.task_state.current_tools}
        missing_tools = required_tools - available_tools

        if missing_tools:
            logger.warning(
                "[agent:%s] Missing tools for solution: %s",
                self.agent_type,
                missing_tools,
            )
            return False

        return True

    def _augment_toolset(self, task_def: TaskDefinition) -> None:
        """Augment toolset when current tools are insufficient"""
        logger.info(
            "[agent:%s] Augmenting toolset for task: %s",
            self.agent_type,
            task_def.task_title,
        )

        prompt = f"""The current task requires additional tools:

Task: {task_def.task_content}
Current tools: {[t.tool_name for t in self.task_state.current_tools]}

Design 1-2 new tools that would help solve this type of task.
Each tool should:
1. Fill a specific gap in current capabilities
2. Be generalizable to similar tasks
3. Have clear, verifiable functionality

Return as JSON array of tool specifications."""

        response = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1000,
        )

        try:
            new_tools = self._extract_json(response)
            if isinstance(new_tools, list):
                for tool_data in new_tools:
                    tool_spec = ToolSpec(
                        tool_name=tool_data.get("tool_name", f"tool_{len(self.task_state.current_tools)}"),
                        tool_description=tool_data.get("tool_description", ""),
                        tool_functionality=tool_data.get("tool_functionality", ""),
                    )
                    self.task_state.current_tools.append(tool_spec)
                logger.info("[agent:%s] Added %d new tools", self.agent_type, len(new_tools))
        except Exception as e:
            logger.warning("[agent:%s] Failed to augment tools: %s", self.agent_type, e)

    def _test_solution_verification(self, task_pkg: TaskPackage) -> bool:
        """Test if solution passes verification"""
        try:
            # In a real implementation, this would execute the code in a sandbox
            # For now, we'll do basic static analysis

            # Check that solution only uses tools
            solution = task_pkg.solution
            verification = task_pkg.verification

            # Basic checks
            has_solve_function = "def solve(" in solution
            has_verify_function = "def verify(" in verification
            returns_boolean = "return True" in verification or "return False" in verification

            return has_solve_function and has_verify_function and returns_boolean
        except Exception as e:
            logger.warning("[agent:%s] Solution verification test failed: %s", self.agent_type, e)
            return False

    def _fix_solution_verification(self, task_pkg: TaskPackage) -> Optional[TaskPackage]:
        """Attempt to fix solution/verification functions"""
        logger.info("[agent:%s] Attempting to fix solution/verification", self.agent_type)

        prompt = f"""Fix the following task solution and verification functions:

Task: {task_pkg.task.task_content}
Current Solution:
```python
{task_pkg.solution}
```
Current Verification:
```python
{task_pkg.verification}
```

Issues to fix:

1. Solution must only use tool calls (no direct data access)

2. Verification must automatically validate the solution

3. Both functions must work correctly

Provide fixed versions. Return as JSON:
{{
"solution": "fixed solution code",
"verification": "fixed verification code"
}}"""

        response = self.llm.chat_completion(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=1500,
        )

        try:
            fix_data = self._extract_json(response)
            if isinstance(fix_data, dict):
                # Create new package with fixes
                fixed_pkg = TaskPackage(
                    task=task_pkg.task,
                    solution=fix_data.get("solution", task_pkg.solution),
                    verification=fix_data.get("verification", task_pkg.verification),
                    agent_type=self.agent_type,
                    metadata=task_pkg.metadata,
                )

                # Test if fixed version works
                if self._test_solution_verification(fixed_pkg):
                    return fixed_pkg
        except Exception as e:
            logger.warning("[agent:%s] Failed to fix solution: %s", self.agent_type, e)

        return None

    def _execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Any:
        """Execute a tool in the sandbox"""
        if self.sandbox:
            result = self.sandbox.execute(tool_name, parameters)
            if not (
                isinstance(result, dict)
                and isinstance(result.get("error"), str)
                and result["error"].startswith("Tool not available:")
            ):
                return result

        # For synthesized tools, we would execute the implementation
        if tool_name in self.task_state.tool_implementations:
            # In production, this would execute the code in a sandbox
            logger.debug("[agent:%s] Would execute tool: %s", self.agent_type, tool_name)
            return {"result": f"Executed {tool_name}"}

        logger.warning("[agent:%s] Tool not available: %s", self.agent_type, tool_name)
        return None

    def _difficulty_level(self, difficulty: str) -> int:
        """Convert difficulty string to numeric level"""
        levels = {
            "very easy": 1,
            "easy": 2,
            "medium": 3,
            "hard": 4,
            "very hard": 5,
            "expert": 6,
        }
        return levels.get(difficulty.lower(), 2)

    def _increase_difficulty(self, current: str) -> str:
        """Increase difficulty level"""
        progression = ["very easy", "easy", "medium", "hard", "very hard", "expert"]
        try:
            idx = progression.index(current)
            return progression[min(idx + 1, len(progression) - 1)]
        except ValueError:
            return "medium"

    def _validate_packages(
        self, packages: List[TaskPackage], request: GenerationRequest
    ) -> List[TaskPackage]:
        """Validate generated packages"""
        validated = []
        for idx, pkg in enumerate(packages, start=1):
            try:
                validated.append(validate_task_package(pkg))
                logger.info(
                    "[agent:%s] Accepted task %d: %s [%s]",
                    self.agent_type,
                    idx,
                    pkg.task.task_title,
                    pkg.task.difficulty_level,
                )
            except Exception as exc:
                logger.warning(
                    "[agent:%s] Dropping task %d due to validation error: %s",
                    self.agent_type,
                    idx,
                    exc,
                )
                continue
        return validated if validated else [self._fallback_package(request)]

    def _default_tools(self) -> List[ToolSpec]:
        """Override to include both base and synthesized tools"""
        base_tools = super()._default_tools()
        return base_tools  # Synthesized tools are added dynamically
