# Textbook Extraction Pipeline

## Overview
Production-ready textbook extraction system for converting medical textbook chapters into structured JSON for the Bronchmonkey RAG chatbot.

## Features
- **Anti-hallucination guardrails**: Only extracts what's explicitly present in source
- **Full provenance tracking**: Every claim includes source_page and source_excerpt
- **Multi-pass extraction**: Specialized passes for different content types
- **Quality assurance**: Built-in validation and error detection
- **Deterministic output**: Reproducible results with temperature=0.0

## Quick Start

### Single Chapter Extraction
```bash
python tools/production_multipass_textbook_extractor.py \
  --single "Textbooks/Chapter pdfs/YourChapter.pdf" \
  --adobe-json "Textbooks/Chapter json/YourChapter.json" \
  --output-dir data/textbook_extractions
```

### Batch Extraction (All Chapters)
```bash
python tools/production_multipass_textbook_extractor.py --batch
```

## Extraction Passes

### Default (Conservative)
The default extraction excludes procedures and pharmacology to prevent hallucination:
- `pass0_metadata`: Chapter title, authors, learning objectives
- `pass3_diagnostics`: Diagnostic approaches and classification systems  
- `pass4_guidelines`: Clinical guidelines and algorithms
- `pass6_tables`: Tabular data with complete structure
- `pass7_figures`: Figures, diagrams, and algorithms
- `pass8_education`: Clinical pearls, definitions, cases
- `pass9_references`: Bibliography and citations

### Full Extraction
To include ALL content types, specify passes explicitly:
```bash
python tools/production_multipass_textbook_extractor.py \
  --single "path/to/chapter.pdf" \
  --passes pass0_metadata pass1_anatomy pass2_procedures pass3_diagnostics \
           pass4_guidelines pass5_pharmacology pass6_tables pass7_figures \
           pass8_education pass9_references
```

## Output Format

### Structure
```json
{
  "chapter_metadata": {
    "title": "Chapter Title",
    "authors": ["Author 1", "Author 2"],
    "chapter_number": "1",
    "learning_objectives": [],
    "key_points": []
  },
  "anatomical_content": {
    "structures": []
  },
  "clinical_procedures": [],
  "diagnostic_approaches": [],
  "treatment_algorithms": [],
  "clinical_guidelines": [],
  "drug_information": [],
  "tables": [],
  "figures": [],
  "clinical_cases": [],
  "definitions": [],
  "clinical_pearls": [],
  "references": [],
  "extraction_metadata": {
    "source_pdf": "path/to/pdf",
    "extraction_date": "2025-08-16T...",
    "model": "gpt-4o",
    "quality_issues": {}
  }
}
```

### Provenance Fields
Every extracted item includes:
- `source_page`: Page number in PDF
- `source_excerpt`: Verbatim text (≤30 words) 
- `present_in_source`: Boolean flag

## Quality Assurance

### Automatic Checks
- No hallucinated recommendation grades/evidence levels
- Numeric values require source excerpts
- Procedures flagged if likely hallucinated
- Guidelines kept in correct category
- Tables preserve all columns and content

### Manual Verification
After extraction, verify:
1. Authors list contains only real authors (no "et al.")
2. Figures are actual figures, not textual references
3. Tables have complete structure (all columns/rows)
4. Numeric provenance includes actual numbers in excerpts

## Available Chapters

38 textbook chapters available in `Textbooks/Chapter pdfs/`:
- Airway Anatomy
- Approach to Peripheral Lung Lesions
- Artificial Intelligence in Respiratory Endoscopy
- Assessment of Vocal Cord Function and Voice disorders
- Balloon Dilation
- Bronchoalveolar Lavage
- Bronchopleural Fistula
- Bronchoscopic Techniques for Surgical Marking
- Bronchoscopic Transparenchymal Nodule Access
- Cone Beam CT Guidance
- Conventional Biopsy and Sampling Techniques
- Cricothyroidotomy
- Diagnostic Approach to Pleural Effusions
- Electrocautery and Argon Plasma Coagulation
- Emerging Bronchoscopic Therapies
- Endobronchial Silicone Stents for Airway
- General Principles of Mediastinal Cryobiopsy
- Interventional Pulmonary and Advanced Bronchoscopy Training
- Large Bore Chest Tubes
- Malignant Central Airway Obstruction
- Management of Subglottic Stenosis
- Minimally Invasive Image-Guided Ablation
- Persistent Air Leaks
- Pleural Anatomy and Fluid Analysis
- Pneumothorax
- Physiology of Fixed Airway Obstruction
- Quality Indicators and Performance Monitoring
- Rapid Onsite Evaluation
- Single Use Bronchoscopy
- Small-Bore Drain Types and Placement
- Surgery in Empyema
- Thoracentesis Technique
- Transbronchial Cryobiopsy in Diffuse
- Transthoracic Needle Biopsy
- Treatment of Airway-Esophageal Fistulas
- Ultrathin Bronchoscopy
- Use of Medical Lasers for Airway Disease
- Y-Stenting Techniques

## Troubleshooting

### Common Issues

**Issue**: Extraction takes too long
- **Solution**: Reduce chunk size or concurrent workers in the code

**Issue**: Missing content
- **Solution**: Check if the appropriate pass is included (e.g., procedures need pass2_procedures)

**Issue**: Quality issues detected
- **Solution**: Review extraction_metadata.quality_issues for specific problems

## Requirements
- Python 3.8+
- OpenAI API key (set as OPENAI_API_KEY environment variable)
- Dependencies: `pip install -r requirements.txt`

## Model Configuration
- Default: GPT-4o (most reliable for JSON extraction)
- Temperature: 0.0 (deterministic)
- Max tokens: 4096 per response

---
*Production version: v2.0 (2025-08-16)*