from __future__ import annotations


class PromptMixin:
    """Prompt templates and utilities for search agent.

    This mixin provides standardized prompt templates used throughout
    the search agent pipeline for entity sampling, question construction,
    answer verification, and webpage summarization.
    """

    SYSTEM_PROMPT = """You are DeepSeek, an AI assistant created by DeepSeek Company. Your purpose is to provide helpful, accurate, and engaging assistance to users.

**Core Guidelines:**
1. **Helpfulness**: Prioritize being genuinely useful. Provide thorough, thoughtful responses that address the user's actual needs.
2. **Accuracy**: Be precise and factually correct. Acknowledge uncertainties and avoid speculation when lacking information.
3. **Clarity**: Express ideas in clear, well-structured language. Adapt your communication style to match the user's needs.
4. **Engagement**: Be warm, friendly, and approachable while maintaining professionalism.

**Response Style:**
- Use natural, conversational English
- Break down complex topics into digestible parts
- Include relevant examples when helpful
- Maintain appropriate tone for the context (casual, professional, technical, etc.)

**Knowledge & Capabilities:**
- You have knowledge up to July 2024
- You support file uploads (images, txt, pdf, ppt, word, excel) and can process text content from these files
- You have web search capabilities when enabled by the user
- You have a 128K context window for handling long conversations

**Ethical Framework:**
- Prioritize user safety and well-being
- Respect privacy and confidentiality
- Avoid harmful, unethical, or illegal content
- Maintain neutrality on sensitive topics while providing factual information
- Decline requests that violate ethical guidelines

**When Users Need Specialized Help:**
- Technical questions: Provide detailed, accurate explanations
- Creative tasks: Offer imaginative but coherent suggestions
- Analysis: Present balanced, evidence-based perspectives
- Learning support: Explain concepts at appropriate difficulty levels

Remember: Your goal is to be the most helpful AI assistant possible within ethical boundaries. Build rapport, understand context, and provide value in every interaction.
"""

    SUMMARY_PROMPT = """Analyze the provided webpage content and user goal below. Extract all relevant information—be thorough and do not miss any important details directly supporting the user's goal.

## Webpage Content
{webpage_content}

## User Goal
{goal}

## Task Guidelines
1. **Rationale Identification**: Find and highlight sections or data from the content that directly relate to the user goal.
2. **Evidence Extraction**: Extract the most pertinent and comprehensive information—include full context and do not omit significant details, even if this results in outputting several paragraphs.
3. **Concise Summary**: Write a clear and logically organized summary, emphasizing how each piece of information supports the user's goal.

**Output Format**  
Respond only with a valid JSON object containing the following fields:
- "rationale": The reasoning behind why certain content from the webpage is relevant.
- "evidence": The direct excerpts or synthesized information from the webpage content supporting the goal.
- "summary": A concise, logically-structured summary contextualizing the evidence in relation to the goal.

**Example:**
```json
{{
  "rationale": "Sections mentioning 'Company X quarterly earnings' are relevant because the user's goal is to analyze financial trends in 2023.",
  "evidence": "In Q1 2023, Company X reported a net income of $2.5 billion... (full relevant excerpts continued)",
  "summary": "Company X experienced sustained growth in 2023, as quarterly reports highlight a cumulative net income increase driven by strong sales in Q2 and Q4."
}}
```
"""

    QUESTION_CONSTRUCTOR_PROMPT = """# Goal
Generate a complex question where the answer is {entity_name}. The answer must be derived strictly from the provided text.
The question must rely on intersecting constraints found ONLY within the source text. **External knowledge is strictly PROHIBITED.**

# Information Context
{entity_info}

{context}

# Strict Context Adherence Rules (CRITICAL)
1. **Source of Truth**: You must assume you have NO knowledge of the world outside of the provided information context.
2. **Fact Verification**: Every constraint, attribute, date, or relationship you mention in the question must be **explicitly present** in the text.
3. **No Hallucination**: If the text does not mention a specific year, role, or relationship, DO NOT invent it or retrieve it from your pre-training data. If the context is insufficient to form a complex question, use the available information to make the best possible question without adding external facts.

# Question Design Principles
1. **Context-Based Attribute Substitution**: Instead of naming a related entity, describe its properties **using only descriptions found in the text**. (e.g., If the text says "Apple released the iPhone", do not refer to Apple as "The company founded by Steve Jobs" unless "Steve Jobs" is mentioned in the text. Instead, use "The company that released the iPhone").
2. **The "No-Shortcut" Rule**: A user should NOT be able to find the answer by searching for just one of the constraints.
3. **Broad to Specific**: Use ranges or categories mentioned in the text to keep the search broad initially.
4. **Inter-Entity Relationships**: Link the Target Entity to other entities via relationships **explicitly defined in the text**.
5. **Multi-constraint design**: Combine temporal, spatial, or descriptive conditions found in the text to ensure answer uniqueness.

# Output Format
You must respond ONLY with a JSON object containing the following keys:
- "question": The generated multi-hop question based strictly on the context.
- "answer": The specific entity/title.
- "constraints": A list of the specific constraints used.
- "reasoning_chain": A step-by-step explanation citing exactly which sentences in the text support each constraint.

Example structure:
{{
"question": "...",
"answer": "...",
"constraints": ["..."],
"reasoning_chain": "Constraint A comes from sentence X; Constraint B comes from sentence Y. Intersection leads to Answer."
}}
"""

    EXAMINE_CONTEXT_RELEVANCE_PROMPT = """Identify {num_entities_each_domain} distinct, informative, and verifiable "long-tail" entities within the {domain} domain.

Selection Criteria:
1. **Obscurity**: The entity should not be general knowledge (avoid top-tier famous examples).
2. **Searchability**: The entity must exist and have verifiable details available online (e.g., scientific papers, historical records, specific news events).
3. **Complexity**: The entity should involve enough depth to warrant a search query (e.g., a specific algorithm, a rare historical event, a specialized biological species).

Output Requirement:
Return strictly a valid JSON array containing objects with "name" and "description". Do not include markdown formatting or explanations.

Example format:
[
  {{
    "name": "Pando (tree colony)",
    "description": "A clonal colony of an individual male quaking aspen determined to be a single living organism by identical genetic markers and assumed to have one massive underground root system."
  }}
]
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

    ENTITY_SAMPLER_PROMPT = """Identify {num_entities_each_domain} distinct, informative, and verifiable "long-tail" entities within the {domain} domain.

Selection Criteria:
1. **Obscurity**: The entity should not be general knowledge (avoid top-tier famous examples).
2. **Searchability**: The entity must exist and have verifiable details available online (e.g., scientific papers, historical records, specific news events).
3. **Complexity**: The entity should involve enough depth to warrant a search query (e.g., a specific algorithm, a rare historical event, a specialized biological species).

Output Requirement:
Return strictly a valid JSON array containing objects with "name" and "description". Do not include markdown formatting or explanations.

Example format:
[
  {{
    "name": "Pando (tree colony)",
    "description": "A clonal colony of an individual male quaking aspen determined to be a single living organism by identical genetic markers and assumed to have one massive underground root system."
  }}
]
"""

    RETRIEVE_CONTEXT_PROMPT = """# Role
You are an Efficient Data Miner. Your goal is to identify **{entity_name}** and extract a "Composite Fingerprint" consisting of **4-5 distinct, obscure constraints** in a single workflow.

# Efficiency Protocol (Single-Shot Strategy)
To avoid multiple tool loops, you must find a **"High-Density Source"**.
1.  **Search**: Do NOT search for generic summaries. Construct a single, complex query to find a document containing lists, tables, appendices, or technical logs.
    * *Query Strategy*: `"{entity_name}" AND ("specifications" OR "appendix" OR "census data" OR "logistics" OR "chronology") -site:wikipedia.org`
2.  **Visit**: Enter the most promising link.
3.  **Extract**: Find **4 to 5** specific data points that act as "hard constraints" (unique identifiers).

# Extraction Criteria (The 4-5 Constraints)
You must find facts that belong to different categories (e.g., 1 Date, 1 Number, 1 ID Code, 1 Physical Attribute).
* *Bad:* "It is very old." (Too vague)
* *Good:* "Built in 1894; Soil pH 6.5; Census ID 40221; Located 4 miles north of the river." (Specific)

# Final Output Format
Return the result strictly in the following JSON format.
**Crucial**: Consolidate all 4-5 facts into the single `obscure_info` string, separated by semicolons or numbered.

```json
{{
  "entity_name": "{entity_name}",
  "obscure_info": "1. [Type]: Specific Fact A; 2. [Type]: Specific Fact B; 3. [Type]: Specific Fact C; 4. [Type]: Specific Fact D; 5. [Type]: Specific Fact E",
  "source_url": "The single URL where these facts were found",
  "is_common_knowledge": false
}}
```
"""

    DOMAIN_SAMPLER_PROMPT = """Task: Generate a strictly numbered list of exactly {num_domains} distinct, specific domains.

Constraints:
1. QUANTITY: The output list must contain EXACTLY {num_domains} items.
2. SPECIFICITY: Use highly specific sub-fields (e.g., "Quantum Cryptography" instead of "Science").
3. DIVERSITY: Ensure maximum categorical variance. The list MUST span across unrelated pillars such as:
   - Natural Sciences & Engineering
   - Arts & Humanities
   - Applied Technology & Digital Economy
   - Social Sciences & Governance
   - Healthcare & Bio-ethics
   - Traditional Crafts & Niche Industries
   Avoid clustering (e.g., do not provide multiple items within "Information Technology").
4. FORMAT: Return ONLY a valid JSON array of strings.

Example Output (if num_domains=3):

[
"Sub-field A",
"Sub-field B",
"Sub-field C"
]

Your Output (Quantity: {num_domains}):
"""
