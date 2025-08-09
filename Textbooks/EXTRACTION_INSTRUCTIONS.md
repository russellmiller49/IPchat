# Textbook Chapter Extraction Instructions

## Overview
This system extracts structured medical content from textbook chapters using GPT-5, including:
- Clinical procedures with step-by-step instructions
- Diagnostic/treatment algorithms
- Clinical guidelines and recommendations
- Tables with clinical data
- Drug information and dosages
- Clinical cases
- Key terminology and definitions

## Prerequisites

### 1. Install Required Libraries
```bash
pip install openai python-dotenv PyMuPDF jsonschema
```

### 2. Set up OpenAI API Key
Ensure your `.env` file contains:
```
OPENAI_API_KEY=your_actual_api_key_here
```

## Usage Instructions

### Option 1: Extract Single Chapter from PDF

```bash
# Basic extraction from PDF
python tools/textbook_extractor_gpt5.py \
  --single "Textbooks/Chapter pdfs/Airway Anatomy.pdf" \
  --title "Airway Anatomy"

# Specify custom output directory
python tools/textbook_extractor_gpt5.py \
  --single "Textbooks/Chapter pdfs/Pneumothorax.pdf" \
  --title "Pneumothorax" \
  --output-dir "Textbooks/my_extractions"
```

### Option 2: Re-extract from Existing JSON (Faster)

If you already have basic JSON extractions and want to enhance them:

```bash
python tools/textbook_extractor_gpt5.py \
  --single "Textbooks/Chapter json/Bronchoscopic Transparenchymal Nodule Access.json" \
  --use-json \
  --title "Bronchoscopic Transparenchymal Nodule Access"
```

### Option 3: Batch Process All Chapters

```bash
# Extract from all PDFs
python tools/textbook_extractor_gpt5.py --batch

# Re-extract from existing JSONs (faster, uses pre-extracted text)
python tools/textbook_extractor_gpt5.py --batch --use-json
```

## Output Structure

Enhanced extractions are saved to `Textbooks/enhanced_extractions/` by default with the filename pattern: `{original_name}_enhanced.json`

Each extraction contains:

### 1. Chapter Metadata
- Title, authors, chapter number
- Learning objectives with page references
- Key points summary

### 2. Clinical Content
- **Procedures**: Step-by-step instructions, indications, contraindications, complications
- **Algorithms**: Decision trees with criteria and pathways
- **Clinical Guidelines**: Recommendations with evidence grades (A, B, C) and levels (I, II, III)
- **Drug Information**: Dosages, indications, contraindications, side effects

### 3. Structured Data
- **Tables**: Headers, rows, clinical relevance, reference ranges
- **Figures**: Captions, types, clinical significance
- **Boxes**: Clinical pearls, warnings, quick references

### 4. Clinical Cases
- Patient presentation
- History and examination
- Investigations and diagnosis
- Management and outcomes
- Learning points

### 5. Definitions & References
- Medical terminology
- Citations with DOI/PMID when available

## Example Commands

### Quick Test (Single Chapter)
```bash
# Test with a small chapter first
python tools/textbook_extractor_gpt5.py \
  --single "Textbooks/Chapter pdfs/Cricothyroidotomy.pdf" \
  --title "Cricothyroidotomy"
```

### Process Specific Chapters
```bash
# Process procedural chapters
for chapter in "Thoracentesis Technique" "Transbronchial Cryobiopsy in Diffuse" "Large Bore Chest Tubes"; do
  python tools/textbook_extractor_gpt5.py \
    --single "Textbooks/Chapter pdfs/${chapter}.pdf" \
    --title "${chapter}"
done
```

### Full Batch Processing
```bash
# Process all 38 chapters (will take ~30-45 minutes)
python tools/textbook_extractor_gpt5.py --batch --use-json
```

## Monitoring Progress

During batch processing, you'll see:
- Current chapter being processed (e.g., `[5/38] Processing: Pneumothorax.json`)
- Extraction summary for each chapter showing counts of extracted elements
- Final batch summary with success/failure counts

## Output Validation

After extraction, check the output:

```bash
# View extraction summary
python -c "
import json
from pathlib import Path

enhanced_dir = Path('Textbooks/enhanced_extractions')
for f in enhanced_dir.glob('*_enhanced.json'):
    with open(f) as file:
        data = json.load(file)
        if 'clinical_content' in data:
            content = data['clinical_content']
            print(f'{f.stem}:')
            print(f'  Procedures: {len(content.get(\"procedures\", []))}')
            print(f'  Guidelines: {len(content.get(\"clinical_guidelines\", []))}')
            print(f'  Tables: {len(data.get(\"structured_data\", {}).get(\"tables\", []))}')
"
```

## Cost Estimation

- **Per chapter**: ~$0.05-0.15 (depending on chapter length)
- **Full batch (38 chapters)**: ~$2-6
- Using `--use-json` is faster and may be slightly cheaper as text is pre-extracted

## Troubleshooting

### API Key Issues
```bash
# Check if API key is loaded
python -c "import os; print('API Key set:', bool(os.getenv('OPENAI_API_KEY')))"
```

### Memory Issues with Large PDFs
If you encounter memory issues with large PDFs, use the `--use-json` option with pre-extracted text.

### Rate Limiting
If you hit rate limits during batch processing, the script will automatically retry. You can also process chapters individually.

### Missing Text Extraction
If no text is extracted from a PDF:
1. Check if PyMuPDF is installed: `pip install PyMuPDF`
2. Try the alternative JSON extraction: `--use-json`
3. Check if the PDF is text-based (not scanned images)

## Tips for Best Results

1. **Start Small**: Test with 1-2 chapters before running full batch
2. **Use JSON Mode**: If basic JSONs exist, use `--use-json` for faster processing
3. **Review Output**: Check the first few extractions to ensure quality
4. **Monitor Costs**: Keep track of API usage in your OpenAI dashboard

## Next Steps

After extraction, you can:
1. Use the Evidence Inspector to browse extracted content
2. Import into your research database
3. Generate summaries or study guides
4. Create clinical reference materials

## Support

For issues or questions:
- Check the extraction logs in each output file's `extraction_metadata`
- Review failed extractions in the batch summary JSON
- Verify your OpenAI API key has GPT-5 access