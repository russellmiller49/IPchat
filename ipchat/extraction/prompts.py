"""
Centralized prompt management for extraction.
Keep prompts simple and focused on RAG needs.
"""

RESEARCH_EXTRACTION_PROMPT = """
Extract key information from this interventional pulmonology research article.
Focus on information that would help answer clinical questions.

Required JSON structure:
{
    "population": "patient population description or null",
    "intervention": "primary intervention/procedure or null",
    "comparator": "control/comparison group or null", 
    "outcomes": {
        "primary": "primary outcome with results",
        "secondary": ["list of secondary outcomes"]
    },
    "key_findings": ["up to 5 bullet points of main findings"],
    "summary": "2-3 sentence summary"
}

Only include information explicitly stated in the text.
"""

TEXTBOOK_EXTRACTION_PROMPT = """
Extract clinical guidance from this interventional pulmonology textbook chapter.
Focus on actionable clinical information.

Required JSON structure:
{
    "procedures": [
        {"name": "procedure name", "description": "brief description"}
    ],
    "indications": ["list of clinical indications"],
    "contraindications": ["list of contraindications"],
    "algorithms": [
        {"name": "algorithm name", "steps": ["step 1", "step 2"]}
    ],
    "key_points": ["important clinical pearls"],
    "summary": "2-3 sentence chapter summary"
}

Only include explicitly stated information.
"""

QUESTION_GENERATION_PROMPT = """
Generate 3 clinical questions that this content could answer.
Questions should be specific to interventional pulmonology practice.

Format:
1. [Question about primary finding/procedure]
2. [Question about patient selection/indications]
3. [Question about outcomes/complications]
"""