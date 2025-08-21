# Legacy Code Archive

This directory contains the original extraction pipeline components archived for reference.
These files are not actively used but contain valuable logic that may be referenced.

## Files:
- `production_multipass_textbook_extractor.py`: Original multi-pass textbook extraction with complex validation
- `gold_standard_pipeline.py`: Complex validation pipeline with quality scoring
- `extractor_gpt5_oe_final.py`: OpenEvidence schema extractor with GPT-5
- `textbook_gold_standard_enhancer.py`: Enhancement pipeline for textbook chapters
- `extract_missing_data.py`: Script to fix missing data in extractions
- Other supporting scripts

## Migration Notes:
See `/docs/MIGRATION_NOTES.md` for how functionality was simplified and migrated.

## Why Archived:
These scripts were archived because:
1. **Over-complexity**: Multiple passes and validation steps that added minimal value
2. **Token inefficiency**: Used 5-10x more tokens than necessary
3. **Schema bloat**: Attempted to extract 50+ fields when only 10-15 were used
4. **Maintenance burden**: Complex interdependencies made updates difficult

## Key Learnings:
- Simple, focused extraction is more effective than comprehensive extraction
- Semantic chunking outperforms fixed-size chunking for retrieval
- Quality scoring added complexity without improving outcomes
- PICO framework is sufficient for most medical literature extraction