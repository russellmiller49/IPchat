SYSTEM = (
  "You extract structured clinical education content from TEXTBOOK CHAPTERS only. "
  "If the text appears to be a research article (Abstract/Methods/Results/Discussion), "
  "return an empty object. Return strictly valid JSON according to the provided schema; "
  "do not invent guideline grades or dosages."
)

USER_TEMPLATE = """Extract textbook content for chapter "{title}".

Rules:
- Include page numbers for every item (page or page_range).
- Preserve table rows exactly; include source_xlsx when provided by Adobe.
- Procedures must include step-by-step instructions and critical points if stated.
- If an element is absent, output an empty list.
- No research-article fields; if the input looks like a research article, return an empty object.

TABLE_HINTS:
{table_hints}

TEXT (TRUNCATED):
{truncated_text}
"""