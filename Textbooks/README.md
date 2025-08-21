# Textbook Chapter Extraction

## Production Extractor (Recommended)

Extract comprehensive clinical content from textbook chapters with anti-hallucination guardrails:

```bash
# Extract single chapter
python ../tools/production_multipass_textbook_extractor.py \
  --single "Chapter pdfs/Airway Anatomy.pdf" \
  --adobe-json "Chapter json/Airway Anatomy.json" \
  --output-dir ../data/textbook_extractions

# Extract all 38 chapters
python ../tools/production_multipass_textbook_extractor.py --batch
```

### Features
- Anti-hallucination guardrails (only extracts what's present)
- Full provenance tracking (source_page + source_excerpt)
- Multi-pass extraction for different content types
- Quality assurance and error detection
- Deterministic output (temperature=0.0)

### Output Format
Structured JSON with:
- Diagnostic approaches
- Clinical guidelines  
- Treatment algorithms
- Drug information
- Tables and figures
- Educational content
- References

See [EXTRACTION_README.md](EXTRACTION_README.md) for detailed documentation.

## Legacy Tools (Archived)

For basic chapter metadata extraction:
```bash
python build_chapter.py "chapter.pdf" --title "Title" --authors "Author1" "Author2"
```

These older tools are maintained in `tools/archive/` for backward compatibility.
