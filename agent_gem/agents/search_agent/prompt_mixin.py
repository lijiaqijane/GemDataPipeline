from __future__ import annotations


class PromptMixin:
    """Prompt templates and utilities for search agent.

    This mixin provides standardized prompt templates used throughout
    the search agent pipeline for entity sampling, question construction,
    answer verification, and webpage summarization.
    """

    SUMMARY_PROMPT = """Please process the following webpage content and user goal to extract relevant information:

## **Webpage Content** 
{webpage_content}

## **User Goal**
{goal}

## **Task Guidelines**
1. **Content Scanning for Rationale**: Locate the **specific sections/data** directly related to the user's goal within the webpage content
2. **Key Extraction for Evidence**: Identify and extract the **most relevant information** from the content, you never miss any important information, output the **full original context** of the content as far as possible, it can be more than three paragraphs.
3. **Summary Output for Summary**: Organize into a concise paragraph with logical flow, prioritizing clarity and judge the contribution of the information to the goal.

**Final Output Format using JSON format has "rational", "evidence", "summary" feilds**
"""

    QUESTION_CONSTRUCTOR_PROMPT = """# Goal
Generate a question where the answer is {entity_name}, but the constraints used in the question are broad enough that no single constraint reveals the answer immediately. The answer must only be findable by intersecting all constraints.** You should use the fewest constraints as long as these constraints can determine the answer.**

The information you can only use is the entity_info and context, you can not use any other information.
# Information Context
{entity_info}

{context}

# Question Design Principles
1. **Attribute Substitution**: Instead of naming a related entity, describe its properties. (e.g., Instead of "Company X," use "The company founded in [Year] by a former [Role] at [Company Y]").
2. **The "No-Shortcut" Rule**: A user should NOT be able to find the answer by searching for just one of the constraints.
3. **Broad to Specific**: Use ranges (e.g., "between 2010 and 2015"), geographic regions, or general categories to keep the initial search broad.
4. **Inter-Entity Relationships**: Link the Target Entity to other entities via non-obvious relationships (e.g., "The CEO of the vendor that supplied the hardware for [Project]").
5. **Multi-constraint design**: Each question combines temporal, spatial, categorical, or descriptive conditions to ensure answer uniqueness; not all conditions are required, but at least two dimensions are typically combined.

# Output Format
You must respond ONLY with a JSON object containing the following keys:
- "question": The generated multi-hop question.
- "answer": The specific entity/title.
- "constraints": A list of the specific constraints used.
- "reasoning_chain": A brief explanation of why this requires cross-referencing.

Example structure:
{{
"question": "...",
"answer": "...",
"constraints": ["..."],
"reasoning_chain": "Explain why this requires cross-referencing."
}}
"""

    EXAMINE_CONTEXT_RELEVANCE_PROMPT = """You should examine the context and determine if it is relevant to the entity.

## **Context**
{context}

## **Entity**
{entity}

## Output Format
You must respond ONLY with a JSON object containing the following keys:
- "is_relevant": True if the context is relevant to the entity, False otherwise.

Example structure:
{{
"is_relevant": True or False
}}
"""

    VERIFICATION_PROMPT = """Perform a comprehensive verification of the answer.

Question: {question}
Answer: {answer}

Evidence from multiple sources:
{context}

After analyzing all evidence, determine:
1. Is the answer factually correct? (YES/NO)

Format:
Correct: [YES/NO]
"""

    ENTITY_SAMPLER_PROMPT = """Generate a list of {num_entities_each_domain} informative long-tail entities from the {domain} domain.
These should be specific, less commonly known entities that would benefit from search-based exploration.
For each entity, provide:
1. Entity name
2. Brief description

Format as JSON array with keys: name, description

For example:
[
{{
    "name": "Pando (tree colony)",
    "description": "A quaking aspen colony in Utah believed to be one of the oldest and heaviest living organisms.",
}},
{{
    "name": "Turritopsis dohrnii",
    "description": "A species of jellyfish known as the 'immortal jellyfish' because it can revert to its juvenile form.",
}}
]
"""
