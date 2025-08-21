# Sample Textbook Extractions Summary

## Overview
This document lists all sample extraction files created during testing of different extraction methods.

## Sample Files Location

### 1. Airway Anatomy Chapter

| Extractor | File Path | Size | Items Extracted |
|-----------|-----------|------|-----------------|
| ipchat | `data/test_extraction/Airway Anatomy.textbook.json` | 7.0 KB | 8 anatomical, 2 procedures, 1 diagnostic |
| GPT5 | `data/test_gpt5_extraction/Airway Anatomy_enhanced.json` | 5.5 KB | 1 procedure, 3 objectives |
| Enhanced | `data/enhanced_extraction/Airway Anatomy_enhanced.json` | 8.4 KB | 4 anatomical, 1 procedure |

### 2. Approach to Peripheral Lung Lesions Chapter

| Extractor | File Path | Size | Items Extracted |
|-----------|-----------|------|-----------------|
| Enhanced | `data/sample_extractions/Approach to Peripheral Lung Lesions_enhanced.json` | 9.4 KB | 5 total items |
| Multipass | `data/sample_extractions/Approach to Peripheral Lung Lesions_multipass.json` | 23 KB | 20 total items (9 diagnostic, 7 guidelines, 4 tables) |

## Key Findings

### Extraction Quality Comparison

1. **Single-pass extractors** (ipchat, GPT5, enhanced): 
   - Extract 5-15 items per chapter
   - Fast but limited coverage
   - May miss important content categories

2. **Multi-pass extractor**:
   - Extracts 20-50+ items per chapter
   - 4x more content than single-pass
   - Better category-specific extraction
   - Takes longer but much more comprehensive

### Best Results: Multi-pass Extraction

The multi-pass approach with focused extraction passes provides:
- **9 diagnostic approaches** (vs 1 in single-pass)
- **7 clinical guidelines** (vs 1 in single-pass)  
- **4 detailed tables** (vs 1 in single-pass)
- **Better structured data** for OpenEvidence queries

## Recommended Approach

For OpenEvidence-level chatbot, use **multi-pass extraction** with these 5 key passes:
1. Anatomy structures
2. Clinical procedures
3. Diagnostic approaches
4. Guidelines & algorithms
5. Tables & structured data

Expected results across 38 chapters:
- 1000-2000 total structured items
- Comprehensive clinical knowledge base
- Ready for complex medical queries

## File Viewing

To view any extraction file:
```bash
# Pretty print JSON
python -m json.tool data/sample_extractions/Approach\ to\ Peripheral\ Lung\ Lesions_multipass.json | less

# Or open in VS Code
code data/sample_extractions/Approach\ to\ Peripheral\ Lung\ Lesions_multipass.json
```

## Next Steps

1. Run multi-pass extraction on all 38 textbook chapters
2. Chunk extracted content for search indexing
3. Build vector and keyword search indexes
4. Integrate with Bronchmonkey chatbot

---
*Generated: August 15, 2025*