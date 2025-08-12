# Medical Evidence Extraction Workflow

## Overview
This document describes the complete extraction pipeline for converting medical research papers from Adobe Extract JSON format (with optional PDFs) into structured OpenEvidence JSON format.

## Directory Structure

```
data/
├── input_articles/        # INPUT: Adobe Extract JSON files (*.json)
├── raw_pdfs/              # INPUT: Optional PDF files for additional context
├── oe_final_outputs/      # OUTPUT: Final extracted OpenEvidence JSON (*.oe_final.json)
└── oe_batch_outputs/      # LOGS: Batch processing summaries

tools/
├── medical_extractor.py        # Main unified extraction interface
├── extractor_gpt5_oe_final.py  # Core extraction logic using GPT-5
├── extractor_gpt5_batch.py     # Batch processing wrapper
├── extract_missing_data.py     # Fills gaps when Adobe JSON is incomplete
└── archive/                     # Old/test versions (ignore these)
```

## Extraction Pipeline

### Step 1: Input Preparation
- Place Adobe Extract JSON files in `data/input_articles/`
- Optionally place corresponding PDF files in `data/raw_pdfs/` (same filename, .pdf extension)

### Step 2: Extraction Process
The pipeline performs the following:

1. **Read Adobe JSON** - Parses the structured data from Adobe Extract
2. **Extract PDF Text** (if available) - Gets raw text with page numbers for context
3. **Build Prompt** - Creates a detailed extraction prompt with medical schema
4. **Call GPT-5** - Uses OpenAI's GPT-5 model to extract structured evidence
5. **Post-Process** - Calculates derived measures, validates data
6. **Save Output** - Writes to `data/oe_final_outputs/` as `*.oe_final.json`

### Step 3: Output Format
The extracted files follow the OpenEvidence schema with sections for:
- Metadata (title, authors, journal, etc.)
- Study design and population
- Primary and secondary outcomes with statistical measures
- Adverse events
- Tables and figures
- Key findings and conclusions

## Usage Examples

### Single File Extraction
```bash
# Extract with just Adobe JSON
python tools/medical_extractor.py --single "paper.json"

# Extract with Adobe JSON + PDF for better context
python tools/medical_extractor.py --single "paper.json" --pdf "paper.pdf"
```

### Batch Processing
```bash
# Process all JSON files in input_articles/
python tools/medical_extractor.py --batch

# Process files matching a pattern
python tools/medical_extractor.py --batch --pattern "A*.json"

# Process with more parallel workers (default is 3)
python tools/medical_extractor.py --batch --workers 5

# Resume from a previous batch (skips already processed files)
python tools/medical_extractor.py --batch --resume "data/oe_batch_outputs/batch_summary_20240808_100113.json"
```

### Verification and Listing
```bash
# Verify extraction quality
python tools/medical_extractor.py --verify "paper.oe_final.json"

# List all completed extractions
python tools/medical_extractor.py --list
```

## Special Scripts

### extract_missing_data.py
Used when Adobe Extract JSON is truncated or missing data:
```bash
python tools/extract_missing_data.py --input "truncated.json" --pdf "full.pdf"
```
This script specifically handles cases where:
- Adobe JSON is incomplete (truncated sections)
- Tables are referenced but empty
- PDFs have data not captured in Adobe Extract

### extractor_gpt5_batch.py
Direct batch processing script (used by medical_extractor.py internally):
```bash
python tools/extractor_gpt5_batch.py --batch
```

## Configuration

Environment variables (in `.env` file):
```bash
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-5              # or gpt-4o for fallback
OPENAI_TIMEOUT=300               # seconds
MAX_PARALLEL_EXTRACTIONS=3       # parallel API calls
BATCH_SIZE=5                     # files per batch
RATE_LIMIT_DELAY=1.0            # seconds between API calls
```

## Output Files

### Successful Extraction
`data/oe_final_outputs/paper_name.oe_final.json`
- Contains structured medical evidence
- Includes provenance (page numbers, table references)
- Has calculated statistical measures

### Batch Summary
`data/oe_batch_outputs/batch_summary_TIMESTAMP.json`
- Lists all processed files
- Shows success/failure status
- Contains error messages for debugging

## Quality Metrics

The extraction quality can be verified with:
```bash
python tools/medical_extractor.py --verify "output.oe_final.json"
```

Quality score (0-100) based on:
- Metadata completeness (25%)
- Primary outcomes presence (35%)
- Population data (20%)
- Tables extracted (20%)

## Common Issues and Solutions

### Issue: "File not found"
**Solution**: Ensure files are in correct directories or provide full paths

### Issue: "API timeout"
**Solution**: Increase OPENAI_TIMEOUT in .env file

### Issue: "Rate limit exceeded"
**Solution**: Reduce MAX_PARALLEL_EXTRACTIONS or increase RATE_LIMIT_DELAY

### Issue: "Incomplete extraction"
**Solution**: Use extract_missing_data.py with PDF for additional context

## Current Status
- **292 papers** successfully extracted in `data/oe_final_outputs/`
- All extractions use the **oe_final** schema format
- Ready for downstream processing (chunking, indexing, search)

## Next Steps
After extraction, the data flows to:
1. **Chunking** - Breaking documents into searchable chunks
2. **Indexing** - Building FAISS/BM25 indices for search
3. **Database** - Loading into PostgreSQL for structured queries
4. **API/UI** - Serving through FastAPI and Streamlit interfaces