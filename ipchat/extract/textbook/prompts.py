SYSTEM = (
  "You extract structured clinical education content from TEXTBOOK CHAPTERS only. "
  "If the text appears to be a research article (Abstract/Methods/Results/Discussion), refuse and return an empty object. "
  "Return strictly valid JSON according to the provided schema."
)

USER_TEMPLATE = """Extract textbook content for chapter "{title}".

Rules:
- Include page numbers for every item (page or page_range).
- Do not fabricate guideline grades/evidence levels if not explicitly present.
- Preserve table rows exactly; if an Adobe table sheet path is provided, include it as source_xlsx.
- Procedures must have step-by-step instructions and critical points if stated.
- If an element is absent, use an empty list.

TABLE_HINTS:
{table_hints}

TEXT (TRUNCATED TO MODEL LIMITS):
{truncated_text}
"""