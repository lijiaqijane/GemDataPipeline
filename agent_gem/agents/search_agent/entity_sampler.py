from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from agent_gem.llm import LLMClient

from .prompt_mixin import PromptMixin

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Represents a sampled entity for question generation.

    Attributes:
        name: Entity name
        domain: Domain category the entity belongs to
        description: Optional description of the entity
    """

    name: str
    domain: str
    description: Optional[str] = None


class EntitySamplerMixin(PromptMixin):
    """Samples informative long-tail entities across diverse domains from large-scale web corpora.

    This mixin provides functionality to sample entities from specified domains
    and optionally expand them iteratively by extracting related entities.
    """

    def sample_entities(
        self,
        llm: LLMClient,
        tools: List[Dict[str, Any]],
        tool_call_map: Dict[str, Callable],
        num_entities_each_domain: int,
        domains: List[str],
        num_iterations: int = 2,
        entities_per_entity: int = 3,
    ) -> List[Entity]:
        """Sample informative entities from diverse domains with iterative expansion.

        Args:
            llm: LLM client for entity generation
            tools: List of tool specifications
            tool_call_map: Mapping of tool names to execution functions
            num_entities_each_domain: Number of entities to sample per domain
            domains: List of domain names to sample from
            num_iterations: Number of expansion iterations (currently unused)
            entities_per_entity: Number of new entities to extract per entity (currently unused)

        Returns:
            List of sampled entities, shuffled randomly
        """
        entities: List[Entity] = []

        # Step 1: Initial sampling from each domain
        for domain in domains:
            logger.debug(f"Sampling {num_entities_each_domain} entities from domain: {domain}")
            domain_entities = self._sample_domain_entities(
                llm, tools, tool_call_map, domain, num_entities_each_domain
            )
            if not domain_entities:
                continue
            entities.extend(domain_entities)
            logger.info(f"Sampled {len(domain_entities)} entities from domain: {domain}")

        # Step 2-4: Iterative entity expansion (currently disabled)
        # This can be enabled when needed for more comprehensive entity sampling
        # entities = self._iterative_entity_expansion(
        #     entities,
        #     num_iterations=num_iterations,
        #     entities_per_entity=entities_per_entity
        # )

        # Step 5: Shuffle and return
        random.shuffle(entities)
        logger.info(f"Total entities sampled: {len(entities)}")

        return entities

    def _sample_domain_entities(
        self,
        llm: LLMClient,
        tools: List[Dict[str, Any]],
        tool_call_map: Dict[str, Callable],
        domain: str,
        num_entities_each_domain: int,
    ) -> List[Entity]:
        """Sample entities from a specific domain.

        Args:
            llm: LLM client for entity generation
            tools: List of tool specifications
            tool_call_map: Mapping of tool names to execution functions
            domain: Domain name to sample entities from
            num_entities_each_domain: Number of entities to sample

        Returns:
            List of Entity objects from the specified domain
        """
        try:
            messages = [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": self.ENTITY_SAMPLER_PROMPT.format(
                        num_entities_each_domain=num_entities_each_domain, domain=domain
                    ),
                },
            ]

            response, search_context = llm.chat_with_agent(
                messages=messages,
                tools=tools,
                tool_call_map=tool_call_map,
                temperature=0.8,
                max_tokens=2048,
            )

            # Parse response with proper JSON parsing
            entities = self._parse_entity_response(response, domain)
            return entities[:num_entities_each_domain]
        except Exception as e:
            logger.error(f"Error sampling entities from domain {domain}: {e}")
            # Fallback: generate simple placeholder entities
            logger.warning(f"Using fallback entities for domain: {domain}")
            return None

    def _parse_entity_response(self, response: str, domain: str) -> List[Entity]:
        """Parse LLM response into Entity objects.

        Attempts to parse JSON array from LLM response. Falls back to simplified
        text parsing if JSON parsing fails.

        Args:
            response: Raw response string from LLM
            domain: Domain name for the entities

        Returns:
            List of Entity objects parsed from the response
        """
        entities: List[Entity] = []

        # Try to extract JSON array from response
        # Look for JSON array pattern (may be wrapped in markdown code blocks or text)
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                data = json.loads(json_str)

                # Parse each entity from JSON
                for item in data:
                    if isinstance(item, dict):
                        name = item.get("name", "").strip()
                        description = item.get("description", "").strip()

                        if name:
                            entities.append(
                                Entity(
                                    name=name,  # Limit length to prevent issues
                                    domain=domain,
                                    description=description or f"Entity from {domain}",
                                )
                            )

                if entities:
                    logger.debug(f"Successfully parsed {len(entities)} entities from JSON")
                    return entities
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                logger.warning(f"Failed to parse JSON from LLM response: {e}")

        # Fallback: simplified text parsing (legacy behavior)
        # This should rarely be used if LLM returns proper JSON
        logger.warning(f"Using fallback text parsing for domain: {domain}")
        lines = response.split("\n")
        for line in lines:
            if "name" in line.lower() or len(line.strip()) > 5:
                # Extract entity name (simplified)
                if ":" in line:
                    name = line.strip().replace('"', "").replace("'", "").split(":")[0]
                else:
                    name = line.strip()

                if name and len(name) > 2:
                    entities.append(
                        Entity(
                            name=name[:100],  # Limit length
                            domain=domain,
                            description=f"Entity from {domain}",
                        )
                    )

        if not entities:
            logger.warning(f"No entities parsed, creating fallback entity for {domain}")
            entities = [Entity(name=f"{domain}_entity", domain=domain)]

        return entities

    def _fetch_wiki_content(self, entity: Entity) -> str:
        """Fetch Wikipedia content for an entity using search.

        Note: This method requires access to self.search_tool, which should be
        available in the SearchAgent class that uses this mixin.

        Args:
            entity: The entity to fetch wiki content for

        Returns:
            Wiki content text as a string, or error message if fetch fails
        """
        if not hasattr(self, "search_tool"):
            logger.error("search_tool not available for fetching wiki content")
            return f"Search tool not available for {entity.name}"

        try:
            # Construct wiki search query
            query = f"{entity.name} wikipedia"
            logger.info(f"Fetching wiki content for: {entity.name}")

            # Call search tool
            # Note: Search results are limited; can be configured via max_results parameter
            search_result = self.search_tool.execute(query, max_results=5, page=1)

            # Extract wiki content from search results
            wiki_content = ""

            # Search results are already formatted as text by format_serper_results
            # Priority: knowledgeGraph (most authoritative) → answerBox (most relevant) →
            # organic results (most comprehensive) → string (most direct)
            if isinstance(search_result, str):
                wiki_content = search_result
            elif isinstance(search_result, dict):
                # Fallback if raw dict is returned
                if "knowledgeGraph" in search_result:
                    wiki_content = search_result["knowledgeGraph"].get("description", "")
                elif "answerBox" in search_result:
                    wiki_content = search_result["answerBox"].get("snippet", "")
                else:
                    # Extract from organic results
                    organic = search_result.get("organic", [])
                    snippets = [r.get("snippet", "") for r in organic[:3]]
                    wiki_content = "\n".join(snippets)

            if not wiki_content:
                logger.warning(f"No wiki content found for: {entity.name}")
                wiki_content = f"No detailed information available for {entity.name}"

            return wiki_content

        except Exception as e:
            logger.warning(f"Error fetching wiki content for {entity.name}: {e}")
            return f"Error retrieving information for {entity.name}"

    def _extract_entities_from_wiki(
        self,
        entity: Entity,
        wiki_content: str,
        num_entities: int = 3,
    ) -> List[Entity]:
        """Extract new entities from wiki content using LLM.

        Note: This could be optimized to use common entity extraction libraries
        instead of LLM for better performance and cost efficiency.

        Args:
            entity: The source entity
            wiki_content: Wiki content text to extract entities from
            num_entities: Number of new entities to extract

        Returns:
            List of newly extracted Entity objects
        """
        if not hasattr(self, "llm"):
            logger.error("LLM client not available for entity extraction")
            return []

        try:
            prompt = f"""Based on the following Wikipedia content about {entity.name}, 
extract {num_entities} new related entities that are mentioned.

Wikipedia content:
{wiki_content[:2000]}

Return a JSON array with format:
[
  {{"name": "Entity1", "description": "brief description"}},
  {{"name": "Entity2", "description": "brief description"}}
]

Focus on extracting specific, notable entities (people, places, concepts, organizations, etc.) that are directly mentioned or related to {entity.name}."""

            logger.info(f"Extracting {num_entities} entities from wiki content of: {entity.name}")

            response = self.llm.simple_complete(prompt, temperature=0.7, max_tokens=1024)

            # Parse the response to extract entities
            new_entities = self._parse_entity_response(response, entity.domain)

            logger.info(f"Extracted {len(new_entities)} entities from {entity.name}")
            return new_entities[:num_entities]

        except Exception as e:
            logger.warning(f"Error extracting entities from wiki content of {entity.name}: {e}")
            return []

    def _iterative_entity_expansion(
        self,
        initial_entities: List[Entity],
        num_iterations: int = 2,
        entities_per_entity: int = 3,
    ) -> List[Entity]:
        """Iteratively expand entities by extracting related entities from wiki content.

        This method expands the initial entity set by:
        1. Fetching Wikipedia content for each entity
        2. Extracting related entities mentioned in the content
        3. Repeating the process for the newly extracted entities

        Args:
            initial_entities: Initial list of entities to expand
            num_iterations: Number of expansion iterations
            entities_per_entity: Number of new entities to extract per entity

        Returns:
            Expanded list of entities
        """
        current_entities = initial_entities.copy()

        for iteration in range(num_iterations):
            logger.info(f"Entity expansion iteration {iteration + 1}/{num_iterations}")
            new_entities: List[Entity] = []

            for entity in current_entities:
                try:
                    # Step 1: Fetch wiki content for the entity
                    wiki_content = self._fetch_wiki_content(entity)

                    # Update entity description with wiki content
                    entity.description = wiki_content

                    # Step 2: Extract new entities from wiki content
                    extracted = self._extract_entities_from_wiki(
                        entity, wiki_content, num_entities=entities_per_entity
                    )
                    new_entities.extend(extracted)

                except Exception as e:
                    logger.warning(f"Error processing entity {entity.name}: {e}")

            # Update current entities list (old entities are removed, new ones added)
            current_entities = new_entities

            if not current_entities:
                logger.warning(f"No entities generated in iteration {iteration + 1}")
                break

            logger.info(f"Generated {len(current_entities)} entities in iteration {iteration + 1}")

        return current_entities
