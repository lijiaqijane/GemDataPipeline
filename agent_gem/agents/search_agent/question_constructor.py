from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from agent_gem.llm import LLMClient, _format_tool_result
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


class QuestionConstructorMixin(PromptMixin):
    """Question construction utilities for search agent.

    This mixin provides functionality to construct multi-hop questions
    that require cross-referencing multiple constraints to find the answer.
    """

    def _construct_question(
        self,
        llm: LLMClient,
        search_tool: SearchTool,
        entity: Entity,
        num_tasks: int = 3,
        search_depth: int = 1,
        search_breadth: int = 1,
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
            logger.debug(f"Constructing question {task_idx + 1}/{num_tasks} for entity: {entity.name}")

            # Get relevant context for the entity
            context = self._get_entity_context(llm, search_tool, entity.name, search_depth, search_breadth)

            entity_info = f"Entity: {entity.name}\nDomain: {entity.domain}"
            prompt = self.QUESTION_CONSTRUCTOR_PROMPT.format(
                entity_name=entity.name,
                entity_info=entity_info,
                context=context,
            )

            max_retries = 3
            for retry in range(max_retries):
                try:
                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "You are an expert Intelligence Analyst and Trivia Designer. "
                                "Your goal is to create complex, multi-hop reasoning questions "
                                "based on the provided Entity Context."
                            ),
                        },
                        {"role": "user", "content": prompt},
                    ]
                    response = llm.chat_completion(
                        messages=messages,
                        temperature=0.8,
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
                                search_context=[context] if isinstance(context, str) else context,
                            )
                        )
                        logger.debug(f"Successfully generated question for {entity.name}")
                        break
                    else:
                        logger.warning(
                            f"Invalid response format for entity {entity.name}, retry {retry + 1}/{max_retries}"
                        )

                except Exception as e:
                    logger.warning(
                        f"Error generating QA pair for {entity.name} (retry {retry + 1}/{max_retries}): {e}"
                    )

            if len(tasks) <= task_idx:
                logger.warning(f"Failed to generate question {task_idx + 1} for entity: {entity.name}")

        return tasks

    def _examine_context_relevance(
        self,
        llm: LLMClient,
        context: str,
        entity: str,
    ) -> bool:
        """Examine if the context is relevant to the entity.

        Args:
            llm: LLM client for relevance checking
            context: Context text to examine
            entity: Entity name to check relevance against

        Returns:
            True if context is relevant to the entity, False otherwise
        """
        examine_context_prompt = self.EXAMINE_CONTEXT_RELEVANCE_PROMPT.format(
            context=context,
            entity=entity,
        )

        messages = [
            {
                "role": "system",
                "content": ("You are an expert in examining the relevance of context to an entity."),
            },
            {
                "role": "user",
                "content": examine_context_prompt,
            },
        ]

        max_tokens = getattr(self, "max_tokens", 1000)
        try:
            response = llm.chat_completion(messages=messages, temperature=0, max_tokens=max_tokens)
            data = self._parse_output(response)
            return data.get("is_relevant", False) if data else False
        except Exception as e:
            logger.warning(f"Error examining context relevance for {entity}: {e}")
            return False

    def _get_entity_context(
        self,
        llm: LLMClient,
        search_tool: SearchTool,
        entity_name: str,
        search_depth: int = 1,
        search_breadth: int = 1,
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
        context = ""
        for page in range(search_depth, 0, -1):
            try:
                search_result = search_tool.execute(entity_name, max_results=search_breadth, page=page)
                context = _format_tool_result(search_result)

                if self._examine_context_relevance(llm, context, entity_name):
                    logger.debug(f"Found relevant context for {entity_name} at page {page}")
                    return context
            except Exception as e:
                logger.warning(f"Error getting context for {entity_name} at page {page}: {e}")

        # Return the last context found if no relevant context was identified
        if context:
            logger.debug(f"Using context from last page for {entity_name}")
        else:
            logger.warning(f"No context found for entity: {entity_name}")

        return context

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
                logger.warning("Empty LLM response")
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
            logger.debug(f"Response content: {llm_response[:200]}...")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM output: {e}")
            return None
