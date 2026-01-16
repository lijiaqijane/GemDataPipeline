from __future__ import annotations

import html
import json
import logging
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        pageview: pageview count in Wikipedia in 180 days (for prioritizing long-tail entities)
        fetched: Whether content has been fetched for this entity
        parent: Optional parent entity from which this entity was extracted
    """

    name: str
    domain: str
    description: Optional[str] = None
    pageview: Optional[int] = None
    fetched: bool = False
    parent: Optional[Entity] = None


class EntitySamplerMixin(PromptMixin):
    """Samples informative long-tail entities across diverse domains from large-scale web corpora.

    This mixin provides functionality to sample entities from specified domains
    and optionally expand them iteratively by extracting related entities.
    """

    def _sample_entities(
        self,
        llm: LLMClient,
        num_entities_each_domain: int,
        domains: List[str],
        num_iterations: int = 2,
        entities_per_entity: int = 2,
        max_workers: int = 4,
    ) -> List[Entity]:
        """Sample informative entities from diverse domains with iterative expansion.

        Args:
            llm: LLM client for entity generation
            tools: List of tool specifications
            tool_call_map: Mapping of tool names to execution functions
            num_entities_each_domain: Number of entities to sample per domain
            domains: List of domain names to sample from
            num_iterations: Number of expansion iterations
            entities_per_entity: Number of new entities to extract per entity
            max_workers: Maximum number of worker threads for parallel processing

        Returns:
            List of sampled entities, shuffled randomly
        """
        # Initialize tracking sets for deduplication
        self._seen_names = set()
        self._fetched_names = set()
        # Initialize locks for thread-safe operations
        self._names_lock = threading.Lock()

        entities: List[Entity] = []

        # Step 1.1: Parallel sampling from each domain
        def process_domain(domain: str) -> List[Entity]:
            """Process a single domain: sample entities and perform iterative expansion."""
            try:
                logger.info(f"Sampling {num_entities_each_domain} entities from domain: {domain}")
                domain_entities = self._sample_domain_entities(llm, domain, num_entities_each_domain)

                # Step 1.2-1.4: Iterative entity expansion
                domain_entities = self._iterative_entity_expansion(
                    domain_entities,
                    num_iterations=num_iterations,
                    entities_per_entity=entities_per_entity,
                )

                # Step 1.5: Deduplicate and prioritize long-tail entities
                domain_entities = self._dedupe_and_sort_long_tail(domain_entities)[
                    :num_entities_each_domain
                ]
                logger.info(f"Sampled {len(domain_entities)} entities from domain: {domain}")
                return domain_entities
            except Exception as e:
                logger.error(f"Error processing domain {domain}: {e}")
                return []

        logger.info(f"Starting parallel entity sampling with {max_workers} workers")
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_domain = {executor.submit(process_domain, domain): domain for domain in domains}
            for future in as_completed(future_to_domain):
                domain_entities = future.result()
                entities.extend(domain_entities)

        return entities

    def _sample_domain_entities(
        self,
        llm: LLMClient,
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
        domian_entities: List[Entity] = []
        entity_names: List[str] = []
        while len(domian_entities) < num_entities_each_domain:
            try:
                messages = [
                    {
                        "role": "user",
                        "content": self.ENTITY_SAMPLER_PROMPT.format(
                            num_entities_each_domain=1, domain=domain
                        ),
                    },
                ]

                response = llm.chat_completion(
                    messages=messages,
                    temperature=1.0,
                    top_p=0.95,
                    max_tokens=128,
                )

                # Parse response with proper JSON parsing
                entities = self._parse_entity_response(response, domain)
                if entities[0].name in entity_names:
                    continue
                domian_entities.append(entities[0])
                entity_names.append(entities[0].name)
            except Exception as e:
                logger.error(f"Error sampling entity from domain {domain}: {e}")

        return domian_entities

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

        # Attempt to extract a JSON array from the response (may be inside markdown code blocks or plain text)
        json_match = re.search(r"\[.*\]", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(0)
            try:
                data = json.loads(json_str)
            except Exception as e:
                logger.error(f"Failed to parse JSON: {e}")
                return entities

            # Build Entity objects from the JSON items
            if not isinstance(data, list):
                logger.error(f"Expected JSON array but got {type(data)}: {data}")
                return entities

            for item in data:
                if not isinstance(item, dict):
                    continue
                name = item.get("name", "").strip()
                description = item.get("description", "").strip()
                if name:
                    entities.append(
                        Entity(name=name, domain=domain, description=description or f"Entity from {domain}")
                    )

            if entities:
                logger.info(f"Parsed {len(entities)} entities from JSON successfully")
                return entities

    def _fetch_wiki_content(self, entity: Entity) -> str:
        """Fetch Wikipedia content for an entity using MediaWiki API or search fallback.

        Note: This method requires access to self.search_tool and self.visit_tool,
        which should be available in the SearchAgent class that uses this mixin.

        Args:
            entity: The entity to fetch wiki content for

        Returns:
            Wiki content text as a string
        """
        if entity.fetched:
            logger.debug(f"Skip fetch (already fetched): {entity.name}")
            return entity.description or ""

        # Try MediaWikiTool first
        try:
            logger.info(f"Fetching wiki content via MediaWiki API for: {entity.name}")
            wiki_results = self.wiki_tool.fetch_entity_data(entity, limit=1)

            if wiki_results:
                page_data = wiki_results[0]
                entity.description = page_data.get("description", "")
                entity.pageview = page_data.get("pageview")
                entity.fetched = True

                # Update fetched names for deduplication (thread-safe)
                if hasattr(self, "_fetched_names") and entity.name:
                    with self._names_lock:
                        self._fetched_names.add(self._normalize_name(entity.name))

                return entity.description
        except Exception as e:
            logger.warning(f"MediaWikiTool failed for {entity.name}: {e}")

        # Fallback to search tool and visit tool
        if not hasattr(self, "search_tool") or not hasattr(self, "visit_tool"):
            logger.error("search_tool or visit_tool not available for fetching wiki content")
            entity.fetched = True
            return entity.description or ""

        try:
            logger.info(f"No wiki content found for: {entity.name}. Falling back to search and visit.")
            # Call search tool to get the first link
            search_results = self.search_tool.execute(entity.name, max_results=1)

            if search_results and isinstance(search_results, list) and len(search_results) > 0:
                first_link = search_results[0].get("url")
                if first_link:
                    logger.info(f"Found link via search: {first_link}. Fetching content via visit tool.")
                    # Call visit tool
                    web_content = self.visit_tool.execute(
                        first_link, goal=f"Fetch information about {entity.name}"
                    )

                    if web_content and "[visit] Failed to read page." not in web_content:
                        cleaned = self._normalize_external_content(web_content)
                        entity.description = cleaned
                        entity.pageview = None
                        entity.fetched = True

                        if hasattr(self, "_fetched_names") and entity.name:
                            with self._names_lock:
                                self._fetched_names.add(self._normalize_name(entity.name))

                        logger.info(
                            f"Successfully fetched and cleaned content via visit_tool for: {entity.name}"
                        )
                        return entity.description

            logger.warning(f"No content found via search/visit fallback for: {entity.name}")
            entity.fetched = True
            return entity.description or ""

        except Exception as e:
            logger.warning(f"Error fetching wiki content for {entity.name}: {e}")
            entity.fetched = True
            return entity.description or ""

    def _normalize_external_content(self, text: str) -> str:
        """Clean HTML/markup-like text returned by external visit/search tools.

        Comprehensive cleaning pipeline for web content:
        - Remove script/style blocks and common UI containers
        - Strip navigation menus, footers, headers, sidebars
        - Remove image markers, social media buttons, ads
        - Clean bracketed references and citations
        - Remove URLs, metadata, and redundant whitespace

        Returns:
            Cleaned text content or original text if cleaning fails
        """
        if not text:
            return ""

        original_text = text

        try:
            # Phase 1: Remove script/style blocks (most aggressive first)
            text = re.sub(r"<script[\s\S]*?<\/script>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<style[\s\S]*?<\/style>", "", text, flags=re.IGNORECASE)
            text = re.sub(r"<noscript[\s\S]*?<\/noscript>", "", text, flags=re.IGNORECASE)

            # Phase 2: Remove common noisy HTML containers
            noisy_tags = r"(nav|footer|header|aside|menu|sidebar|widget|banner|advertisement|ad|promo)"
            text = re.sub(rf"<{noisy_tags}[^>]*>[\s\S]*?<\/\1>", "", text, flags=re.IGNORECASE)

            # Phase 3: Strip all remaining HTML tags
            text = re.sub(r"<[^>]+>", "", text)

            # Phase 4: Unescape HTML entities
            text = html.unescape(text)

            # Phase 5: Remove common UI/navigation patterns
            ui_patterns = [
                r"Skip to (main )?content",
                r"Main menu",
                r"Toggle [\w\s]+subsection",
                r"move to sidebar hide",
                r"Jump to (content|navigation)",
                r"Sign (In|Out|Up)",
                r"(Log|Register) (in|out|here)",
                r"Subscribe( now)?",
                r"Share (this|on):?",
                r"Follow (us|me) on",
                r"Continue reading",
                r"Read more",
                r"Posted on:",
                r"Published( Time)?:",
                r"Author:",
                r"Category:",
                r"Tags?:",
                r"Comments?:",
                r"View (all|more)",
                r"See also",
                r"Related (articles|links|posts)",
                r"Explore more",
                r"Previous (article|page)",
                r"Next (article|page)",
                r"Back to (top|home)",
                r"Edit (this|links)",
                r"Powered by",
                r"Copyright ©",
                r"All rights reserved",
                r"Privacy (policy|statement)",
                r"Terms (of|and) (Use|Service|Conditions)",
                r"Cookie (policy|statement|settings)",
                r"Manage consent",
                r"(Accept|Reject) (all )?cookies?",
            ]
            for pattern in ui_patterns:
                text = re.sub(pattern, "", text, flags=re.IGNORECASE)

            # Phase 6: Remove image markers
            text = re.sub(r"Image \d+:", "", text, flags=re.IGNORECASE)
            text = re.sub(r"!\[Image \d+:.*?\]", "", text)
            text = re.sub(r"!\(.*?\)", "", text)

            # Phase 7: Remove markdown links
            text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)

            # Phase 8: Remove bare URLs
            text = re.sub(r"https?://\S+", "", text, flags=re.IGNORECASE)
            text = re.sub(r"www\.\S+", "", text, flags=re.IGNORECASE)

            # Phase 9: Remove bracketed citations and metadata
            text = re.sub(r"\[citation needed\]", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\[edit\]", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\[clarification needed\]", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\[\d+\]", "", text)
            text = re.sub(r"\[.*?\]", "", text)

            # Phase 10: Remove social media patterns
            social_patterns = [
                r"(Facebook|Twitter|X|Instagram|LinkedIn|YouTube|Pinterest|WhatsApp|Reddit)",
                r"(Like|Follow|Subscribe|Share|Tweet|Pin|Upvote)",
            ]
            for pattern in social_patterns:
                text = re.sub(pattern + r"\s*\d*", "", text, flags=re.IGNORECASE)

            # Phase 11: Clean up CSS/JS remnants
            text = re.sub(r"\.mw-parser-output.*", "", text)
            text = re.sub(r"\{.*?\}", "", text)
            text = re.sub(r"`.*?`", "", text)

            # Phase 12: Remove repetitive punctuation
            text = re.sub(r"[_*]{2,}", "", text)
            text = re.sub(r"\.{2,}", ".", text)
            text = re.sub(r"-{3,}", "", text)
            text = re.sub(r"={3,}", "", text)

            # Phase 13: Normalize line breaks
            text = re.sub(r"\r\n|\r", "\n", text)
            text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
            text = re.sub(r"[ \t]+", " ", text)
            text = re.sub(r" ?\n ?", "\n", text)

            text = text.strip()

            if len(text) < 50:
                logger.warning(
                    f"Content too short after normalization ({len(text)} chars), returning original"
                )
                return original_text

            return text

        except Exception as e:
            logger.error(f"Error in _normalize_external_content: {e}")
            return original_text

    def _extract_entities_from_wiki(
        self,
        entity: Entity,
        num_entities: int = 3,
    ) -> List[Entity]:
        """Extract new entities from wiki content using LLM.

        Args:
            entity: The source entity
            num_entities: Number of new entities to extract

        Returns:
            List of newly extracted Entity objects
        """
        if not hasattr(self, "llm"):
            logger.error("LLM client not available for entity extraction")
            return []

        try:
            parent_pv_hint = f" (approx. {entity.pageview} pageviews)" if entity.pageview else ""
            prompt = f"""
Based on the following Wikipedia content about {entity.name}{parent_pv_hint},
extract {num_entities} related but MORE OBSCURE entities that are mentioned or referenced.

Wikipedia content:
{entity.description[:5000] if entity.description else ""}

Return a JSON array with format:
[
    {{"name": "Entity1", "description": "brief description"}},
    {{"name": "Entity2", "description": "brief description"}}
]

Guidance / Requirements (to prioritize concrete, low-traffic long-tail entities):
1) Focus on entities from the {entity.domain} domain that are LESS well-known than {entity.name}.
2) Target "long-tail" entities: niche topics, specialized concepts, lesser-known figures, or sub-topics.
3) For each topic, identify several factual answers (e.g., person names, dates, titles, institutions, specific works)
     that meet TWO criteria: (a) the selected facts must be objective statements that can be independently verified
     through reliable sources without interpretation or inference; and (b) they must be concrete and specific enough
     to exclude overly generic or widely known common-sense facts.
4) Avoid mainstream or highly popularized entities — prefer items with low public visibility/pageviews.
5) Ensure each returned entity has enough descriptive information (one or two sentences) so it can be validated by
     follow-up searches (not mere single-word labels or ambiguous terms).
6) Prefer entities that are academically, culturally, or historically significant but not widely known to the general public.
7) When possible, include a short parenthetical hint (1-6 words) such as a minor date range, locality, affiliated institution,
     or a concise descriptor to make the entity verifiable while keeping it long-tail.
"""

            logger.info(f"Extracting {num_entities} entities from wiki content of: {entity.name}")

            response = self.llm.simple_complete(prompt, temperature=0.7, max_tokens=1024)

            # Parse the response to extract entities
            new_entities = self._parse_entity_response(response, entity.domain)

            logger.info(f"Extracted {len(new_entities)} entities from {entity.name}")
            return new_entities[:num_entities]

        except Exception as e:
            logger.warning(f"Error extracting entities from wiki content of {entity.name}: {e}")
            return []

    def _normalize_name(self, name: str) -> str:
        """Normalize entity name for lightweight deduplication."""
        if not name:
            return ""
        # Lowercase, strip, remove surrounding punctuation and collapse whitespace
        n = name.strip().lower()
        n = re.sub(r"[\W_]+", " ", n)
        n = re.sub(r"\s+", " ", n).strip()
        return n

    @staticmethod
    def _dedupe_and_sort_long_tail(entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities by name and sort by pageview (ascending).

        Args:
            entities: List of entities to process

        Returns:
            Deduplicated and sorted list of entities
        """
        unique: Dict[str, Entity] = {}
        for e in entities:
            key = (e.name or "").strip().lower()
            if not key:
                continue
            prev = unique.get(key)
            # Keep the one with lower pageview (more long-tail)
            if prev is None or (e.pageview or 0) < (prev.pageview or 0):
                unique[key] = e

        sorted_entities = list(unique.values())
        # Sort by pageview ascending (None/0 first), then by name
        sorted_entities.sort(key=lambda x: (x.pageview or 0, (x.name or "").lower()))
        logger.info(
            f"Successfully sampled {len(sorted_entities)} unique entities (prioritized by pageview)"
        )
        return sorted_entities

    def _iterative_entity_expansion(
        self,
        initial_entities: List[Entity],
        num_iterations: int = 2,
        entities_per_entity: int = 2,
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
                    self._fetch_wiki_content(entity)

                    # Step 2: Extract new entities from wiki content (request 2x entities_per_entity)
                    extracted = self._extract_entities_from_wiki(
                        entity, num_entities=2 * entities_per_entity
                    )

                    # Lightweight deduplication (thread-safe)
                    filtered = []
                    for child in extracted:
                        key = self._normalize_name(child.name or "")
                        if not key:
                            continue
                        with self._names_lock:
                            if (hasattr(self, "_seen_names") and key in self._seen_names) or (
                                hasattr(self, "_fetched_names") and key in self._fetched_names
                            ):
                                continue
                            if hasattr(self, "_seen_names"):
                                self._seen_names.add(key)
                        filtered.append(child)

                    # Fetch content for children immediately and filter
                    valid_children = []
                    for child in filtered:
                        child.parent = entity
                        self._fetch_wiki_content(child)
                        # Filter by description length (threshold: 100 words)
                        if child.description and len(child.description.split()) > 100:
                            valid_children.append(child)

                    # Sort by pageview ascending to prioritize low pageview entities
                    valid_children.sort(key=lambda x: x.pageview or 0)
                    selected_children = valid_children[:entities_per_entity]

                    # Add selected children to next generation
                    new_entities.extend(selected_children)

                    # Parent retention logic
                    parent_pv = entity.pageview or 0
                    if selected_children:
                        child_pvs = [c.pageview or 0 for c in selected_children]
                        child_avg_pv = sum(child_pvs) / len(child_pvs) if child_pvs else 0

                        should_keep_parent = (
                            parent_pv < 500  # Already long-tail, keep it
                            or len(selected_children)
                            < entities_per_entity * 0.5  # Insufficient quality children
                            or child_avg_pv >= parent_pv * 0.8  # Children not sufficiently lower
                            or (
                                parent_pv > 0 and (parent_pv - child_avg_pv) / parent_pv < 0.2
                            )  # Relative decrease < 20%
                        )

                        if should_keep_parent:
                            new_entities.append(entity)
                            logger.debug(
                                f"Keeping parent {entity.name} (pv={parent_pv}, child_avg={child_avg_pv:.0f})"
                            )
                        else:
                            logger.info(
                                f"Replacing parent {entity.name} (pv={parent_pv}) with {len(selected_children)} children (avg_pv={child_avg_pv:.0f})"
                            )
                    else:
                        # If no valid children found, keep parent
                        new_entities.append(entity)

                except Exception as e:
                    logger.warning(f"Error processing entity {entity.name}: {e}")

            # Update current entities list
            current_entities = new_entities

            if not current_entities:
                logger.warning(f"No entities generated in iteration {iteration + 1}")
                break

            logger.info(f"Generated {len(current_entities)} entities in iteration {iteration + 1}")

        # Ensure all final entities are fetched
        for entity in current_entities:
            if not getattr(entity, "fetched", False):
                try:
                    self._fetch_wiki_content(entity)
                except Exception as e:
                    logger.warning(f"Error fetching final entity {entity.name}: {e}")

        return current_entities
