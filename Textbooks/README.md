# Chapter JSON Builder (RAG-ready)

## Quick start
```bash
# (optional) create a venv and install tools if you want validation
pip install pydantic pdfplumber PyPDF2 jsonschema pyyaml

# Emit JSON Schema (optional)
python chapter_models.py

# Build a chapter JSON
python build_chapter.py "Malignant Central Airway Obstruction.pdf"   --title "Malignant Central Airway Obstruction"   --authors "John E. Howe" "Coral X. Giovacchini" "Kamran Mahmood"   --source-url "https://doi.org/10.1007/978-3-031-49583-0_32-1"

# Batch mode
python batch_build.py book.yaml
```

- Output file: `<chapter>.chapter.json`
- Paragraph-level `content.text_units` with `{page, paragraph_index}` provenance
- Optional validation: `python validate_jsons.py textbook_chapter.schema.json`
