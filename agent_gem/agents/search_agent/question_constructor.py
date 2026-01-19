from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent_gem.llm import LLMClient
from agent_gem.tools import SearchTool

from .entity_sampler import Entity
from .prompt_mixin import PromptMixin

logger = logging.getLogger(__name__)


@dataclass
class QuestionAnswerPair:
    """Represents a question-answer pair for search agent tasks.

    Attributes:
        question: The generated multi-hop question
        answer: The ground-truth answer (entity name)
        entity: The entity this QA pair is about
        search_context: Search results used to construct the QA pair
    """

    question: str
    answer: str
    entity: Entity
    search_context: List[str]
    all_search_context: List[str]


class QuestionConstructorMixin(PromptMixin):
    """Question construction utilities for search agent.

    This mixin provides functionality to construct multi-hop questions
    that require cross-referencing multiple constraints to find the answer.
    """

    def _construct_question(
        self,
        llm: LLMClient,
        tools: List[Dict[str, Any]],
        tool_call_map: Dict[str, str],
        entity: Entity,
        num_tasks: int = 3,
    ) -> List[QuestionAnswerPair]:
        """Construct multi-hop questions for the given entity.

        Args:
            llm: LLM client for generating questions
            search_tool: Search tool to gather entity context
            entity: Entity to construct questions for
            num_tasks: Number of questions to generate
            search_depth: Number of search result pages to check
            search_breadth: Number of results per page

        Returns:
            List of QuestionAnswerPair objects
        """
        tasks: List[QuestionAnswerPair] = []

        for task_idx in range(num_tasks):
            logger.info(f"Constructing question {task_idx + 1}/{num_tasks} for entity: {entity.name}")

            # Get relevant context for the entity
            max_retries = 3
            context = None
            search_context = None
            for context_attempt in range(max_retries):
                try:
                    context, search_context = self._get_entity_context(llm, tools, tool_call_map, entity)
                    if context and isinstance(context, dict) and context.get("obscure_info"):
                        break  # Success, exit retry loop
                    logger.warning(
                        f"Context retrieval attempt {context_attempt + 1}/{max_retries} for {entity.name}: No obscure info found"
                    )
                except Exception as e:
                    logger.warning(
                        f"Context retrieval attempt {context_attempt + 1}/{max_retries} for {entity.name}: {e}"
                    )
                if context_attempt == max_retries - 1:
                    logger.error(f"Failed to get context for {entity.name} after {max_retries} attempts")

            obscure_info = context.get("obscure_info", "") if isinstance(context, dict) else ""

            if not obscure_info:
                logger.warning(
                    f"Error constructing question {task_idx + 1}/{num_tasks} for {entity.name}: No obscure information found"
                )
                continue

            for attempt in range(max_retries):
                try:
                    prompt = self.QUESTION_CONSTRUCTOR_PROMPT.format(
                        entity_name=entity.name,
                        context=obscure_info,
                    )
                    messages = [
                        {"role": "user", "content": prompt},
                    ]
                    response = llm.chat_completion(
                        messages=messages,
                        temperature=1.0,
                        top_p=0.95,
                        max_tokens=2048,
                    )

                    # Parse response
                    parsed_response = self._parse_output(response)

                    if parsed_response and "question" in parsed_response and "answer" in parsed_response:
                        tasks.append(
                            QuestionAnswerPair(
                                question=parsed_response["question"],
                                answer=parsed_response["answer"],
                                entity=entity,
                                search_context=context,
                                all_search_context=search_context,
                            )
                        )
                        logger.info(
                            f"Successfully constructing question {task_idx + 1}/{num_tasks} for {entity.name}"
                        )
                        break  # Success, exit retry loop
                    else:
                        logger.warning(
                            f"Error constructing question {task_idx + 1}/{num_tasks} for {entity.name} (attempt {attempt + 1}/{max_retries}): Invalid response format"
                        )

                except Exception as e:
                    logger.warning(
                        f"Error constructing question {task_idx + 1}/{num_tasks} for {entity.name} (attempt {attempt + 1}/{max_retries}): {e}"
                    )

                if attempt == max_retries - 1:
                    logger.error(
                        f"Failed to construct question {task_idx + 1}/{num_tasks} for {entity.name} after {max_retries} attempts"
                    )

        return tasks

    def _get_entity_context(
        self,
        llm: LLMClient,
        tools: List[Dict[str, Any]],
        tool_call_map: Dict[str, str],
        entity: Entity,
    ) -> str:
        """Get relevant context for the given entity by searching and filtering.

        Args:
            llm: LLM client for relevance checking
            search_tool: Search tool to query
            entity_name: Name of the entity to get context for
            search_depth: Number of search result pages to check
            search_breadth: Number of results per page

        Returns:
            Formatted context string relevant to the entity
        """
        try:
            messages = [
                {
                    "role": "user",
                    "content": self.RETRIEVE_CONTEXT_PROMPT.format(
                        entity_name=entity.name, entity_domain=entity.domain
                    ),
                }
            ]
            response, search_context = llm.chat_with_agent(
                messages=messages,
                tools=tools,
                tool_call_map=tool_call_map,
                temperature=1.0,
                top_p=0.95,
                max_tokens=2048,
                max_sub_turns=100,
                is_summary=True,
            )
            res = self._parse_output(response)
            return res, search_context
        except Exception as e:
            logger.error(f"Error getting context for {entity.name}: {e}")
            return "", ""

    def _parse_output(self, llm_response: str) -> Optional[Dict[str, Any]]:
        """Parse the JSON string generated by the LLM and return a dictionary.

        Supports strings with Markdown code block tags (```json ... ```).

        Args:
            llm_response: Raw response string from LLM

        Returns:
            Parsed dictionary if successful, None otherwise
        """
        try:
            json_str = llm_response.strip()
            if not json_str:
                return None

            # Handle markdown code blocks
            if json_str.startswith("```"):
                match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_str, re.DOTALL)
                if match:
                    json_str = match.group(1)

            data = json.loads(json_str)
            return data

        except json.JSONDecodeError as e:
            logger.warning(f"Error parsing JSON from LLM response: {e}")
            logger.info(f"Response content: {llm_response[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM output: {e}")
            return None
