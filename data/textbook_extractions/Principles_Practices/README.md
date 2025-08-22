# Principles and Practice of Interventional Pulmonology - Textbook Extractions

## Overview
This directory contains structured JSON extractions from the textbook "Principles and Practice of Interventional Pulmonology" (Springer, 2025). All content has been processed and optimized for NLP/RAG applications.

## Contents
- **41 Chapter Files**: Complete chapter extractions with procedures, techniques, and clinical guidance
- **15 Supplementary Table Files**: Structured data tables referenced by chapters
- **1 Master Index**: `MASTER_INDEX.json` - Complete catalog of all content
- **1 Analysis Report**: `EXTRACTION_ANALYSIS_REPORT.md` - Quality assessment

## File Structure

### Chapter Files
Each chapter JSON contains:
```json
{
  "chapter_metadata": { /* title, authors, key points */ },
  "anatomical_content": { /* relevant structures */ },
  "clinical_procedures": [ /* step-by-step procedures */ ],
  "diagnostic_approaches": [ /* diagnostic methods */ ],
  "technology_and_technique": [ /* equipment details */ ],
  "tables": [ /* embedded tables */ ],
  "figures": [ /* figure descriptions */ ],
  "references": [ /* citations */ ]
}
```

### Table Files
Supplementary tables (ending in `_table.json`) contain:
- Device comparisons
- Risk factors
- Therapeutic options
- Technical parameters

## Usage for NLP/RAG

### Quick Access
Use `MASTER_INDEX.json` to:
- Find chapters by category
- Search by procedure name
- Locate supplementary tables
- Navigate by condition

### Example Query Patterns
```python
# Load master index
import json
with open('MASTER_INDEX.json', 'r') as f:
    index = json.load(f)

# Find all diagnostic procedures
diagnostic_chapters = [ch for ch in index['chapters'] 
                       if ch['category'] == 'Diagnostic']

# Find chapter on specific procedure
bal_chapter = [ch for ch in index['chapters'] 
               if 'BAL' in ch['procedures']][0]

# Load chapter content
with open(bal_chapter['filename'], 'r') as f:
    chapter_data = json.load(f)
```

### Integration with RAG Systems

1. **Direct Indexing**: Each JSON file can be directly indexed
2. **Chunking Strategy**: 
   - Use `clinical_procedures` for procedure-based chunks
   - Use `key_points` for summary chunks
   - Use `tables` for structured data retrieval
3. **Metadata Filtering**: Category, procedures, and topics enable precise filtering

## Data Quality
✅ **Production Ready**
- All files validated for proper JSON structure
- Unicode characters properly encoded
- Source page references maintained
- Consistent schema across all files

## Updates
- **Last Updated**: August 22, 2025
- **CSV Files Removed**: Redundant CSV files deleted to avoid confusion
- **Format**: JSON-only for optimal NLP performance

## Statistics
- Total Procedures Covered: 50+
- Total Pages Extracted: 500+
- Average Key Points per Chapter: 5
- Extraction Confidence: High

---

*For questions about the extraction process or schema, see `EXTRACTION_ANALYSIS_REPORT.md`*