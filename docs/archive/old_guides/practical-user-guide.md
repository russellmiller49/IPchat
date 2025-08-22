# IPchat Practical Clinical Extraction System - User Guide

## Overview

IPchat Clinical Extraction is a focused NLP system for interventional pulmonology that extracts actionable clinical information from medical documents without the complexity of meta-analysis.

### What It Extracts
- **Diagnostic Yields**: Specific percentages for procedures (e.g., "EBUS-TBNA yield: 92.5%")
- **Complication Rates**: Detailed rates with management strategies
- **Methodology**: Equipment, techniques, patient positioning, sedation
- **Clinical Pearls**: Tips, tricks, and learning points
- **Practical Information**: Indications, contraindications, patient selection
- **Key Findings**: Summary points from each document

### System Architecture
```
Documents (PDFs)
    ↓
Clinical Extractor (extracts granular information)
    ↓
Practical Chunker (creates specialized chunks)
    ↓
Clinical Index Manager
    ├── Vector Index (FAISS for semantic search)
    ├── Keyword Index (BM25 for exact matches)
    └── SQLite Database (for numerical queries)
    ↓
Query Interface (hybrid retrieval)
```

## Getting Started

### Step 1: Initial Setup

```bash
# Clone your repository
git clone https://github.com/yourusername/IPchat.git
cd IPchat

# Install dependencies
pip install -r requirements.txt

# Additional dependencies for clinical extraction
pip install pdfplumber  # For PDF text extraction
pip install sentence-transformers  # For embeddings
pip install faiss-cpu  # For vector search
pip install rank-bm25  # For keyword search
```

### Step 2: Organize Your Documents

```bash
# Create directory structure
mkdir -p data/raw/research
mkdir -p data/raw/textbooks

# Copy your documents
cp Studies/*.pdf data/raw/research/
cp "Textbooks/Chapter pdfs"/*.pdf data/raw/textbooks/
```

### Step 3: Test with Single Document

```python
from ipchat.pipeline.practical_pipeline import PracticalClinicalPipeline

# Initialize pipeline
pipeline = PracticalClinicalPipeline()

# Process a single document
result = pipeline.process_document(
    "data/raw/research/sample_study.pdf",
    document_type="research"
)

print(f"Extracted:")
print(f"  - {result['diagnostic_yields']} diagnostic yields")
print(f"  - {result['complications']} complications")
print(f"  - {result['clinical_pearls']} clinical pearls")
```

## Processing Your Existing Data

### Option 1: Process All Documents at Once

```bash
# Process all documents
python tools/scripts/migrate_to_practical.py --full-run

# This will process:
# - All research papers from Studies/
# - All textbook chapters from Textbooks/Chapter pdfs/
```

### Option 2: Test with Small Batch First

```bash
# Process only 5 documents from each category for testing
python tools/scripts/migrate_to_practical.py --max-docs 5

# Check results
ls data/extracted/  # Should see JSON files with extractions
ls data/chunks/     # Should see chunked documents
```

### Option 3: Process Specific Directory

```python
from ipchat.pipeline.practical_pipeline import PracticalClinicalPipeline

pipeline = PracticalClinicalPipeline()

# Process research papers
research_results = pipeline.batch_process(
    "Studies",
    document_type="research",
    max_documents=None  # Process all
)

# Process textbook chapters
textbook_results = pipeline.batch_process(
    "Textbooks/Chapter pdfs",
    document_type="textbook",
    max_documents=None  # Process all
)

print(f"Processed {len(research_results)} research papers")
print(f"Processed {len(textbook_results)} textbook chapters")
```

## Adding New Documents

### Step 1: Place New Document in Appropriate Directory

```bash
# For research papers
cp new_study.pdf data/raw/research/

# For textbook chapters
cp new_chapter.pdf data/raw/textbooks/
```

### Step 2: Process the New Document

```python
from ipchat.pipeline.practical_pipeline import PracticalClinicalPipeline

pipeline = PracticalClinicalPipeline()

# Process single new document
result = pipeline.process_document(
    "data/raw/research/new_study.pdf",
    document_type="research"
)

# The system automatically:
# 1. Extracts clinical information
# 2. Saves to data/extracted/
# 3. Creates chunks in data/chunks/
# 4. Updates the search indices
# 5. Adds to SQLite database

print(f"Document processed: {result['title']}")
print(f"Confidence: {result['extraction_confidence']}")
```

### Step 3: Verify Extraction Quality

```python
# Check what was extracted
import json

doc_id = result['document_id']
with open(f"data/extracted/{doc_id}.json") as f:
    extraction = json.load(f)

# View diagnostic yields
for procedure, yields in extraction['diagnostic_yields'].items():
    print(f"\n{procedure.upper()} Yields:")
    for metric, data in yields.items():
        print(f"  - {metric}: {data['value']}%")

# View complications
for comp, data in extraction['complication_rates'].items():
    print(f"\n{comp}: {data['rate']}%")
    print(f"  Management: {data.get('management', 'Not specified')}")
```

## Querying the System

### General Queries (Semantic Search)

```python
from ipchat.pipeline.practical_pipeline import PracticalClinicalPipeline

pipeline = PracticalClinicalPipeline()

# General clinical question
results = pipeline.query(
    "What are the indications for EBUS-TBNA?",
    query_type="general"
)

# Results include relevant chunks ranked by relevance
for chunk in results['chunks'][:5]:
    print(f"- {chunk['chunk_type']}: {chunk['score']:.2f}")
    print(f"  From: {chunk['source_document']}")
```

### Specific Clinical Data Queries

```python
# Query diagnostic yields
yields = pipeline.query(
    "EBUS",  # Procedure name
    query_type="yields"
)

for yield_data in yields['yields']:
    print(f"- {yield_data['procedure']} {yield_data['metric']}: {yield_data['value']}%")

# Query complications
complications = pipeline.query(
    "pneumothorax",  # Complication type
    query_type="complications"
)

for comp in complications['complications']:
    print(f"- {comp['complication_type']}: {comp['rate']}%")
    print(f"  Management: {comp['management']}")

# Query clinical pearls
pearls = pipeline.query(
    "bronchoscopy",  # Keyword
    query_type="pearls"
)

for pearl in pearls['pearls'][:5]:
    print(f"- {pearl['pearl']}")
```

### Direct Database Queries

```python
import sqlite3
import pandas as pd

# Connect to database
conn = sqlite3.connect("data/indices/clinical_data.db")

# Get all EBUS diagnostic yields
ebus_yields = pd.read_sql_query("""
    SELECT procedure, metric, value, document_id
    FROM diagnostic_yields
    WHERE procedure LIKE '%ebus%'
    ORDER BY value DESC
""", conn)

print(ebus_yields)

# Get average complication rates
avg_complications = pd.read_sql_query("""
    SELECT complication_type, AVG(rate) as avg_rate, COUNT(*) as n_studies
    FROM complications
    GROUP BY complication_type
    ORDER BY avg_rate DESC
""", conn)

print(avg_complications)

conn.close()
```

## Understanding the Extracted Data

### Extraction Structure

Each document produces an extraction with:

```python
{
    "document_id": "abc123",
    "title": "EBUS-TBNA in Lung Cancer Staging",
    "document_type": "research",
    
    "diagnostic_yields": {
        "ebus": {
            "diagnostic yield": {"value": 92.5, "context": "..."},
            "sensitivity": {"value": 95.2, "context": "..."},
            "specificity": {"value": 100.0, "context": "..."}
        }
    },
    
    "complication_rates": {
        "pneumothorax": {
            "rate": 0.4,
            "management": "Conservative management with observation"
        },
        "bleeding": {
            "rate": 1.6,
            "management": "Minor, self-limited"
        }
    },
    
    "methodology": {
        "ebus": {
            "technique": ["21-gauge needle used", "3-5 passes per station"],
            "equipment": ["Olympus BF-UC180F bronchoscope"],
            "positioning": "Supine position",
            "sedation": "Conscious sedation with midazolam and fentanyl"
        }
    },
    
    "clinical_pearls": [
        "Maintain bronchoscope in neutral position for stability",
        "Use color Doppler before needle insertion",
        "Avoid excessive suction to prevent blood contamination"
    ],
    
    "indications": [
        "Mediastinal lymph node staging",
        "Diagnosis of mediastinal masses"
    ],
    
    "contraindications": [
        "Severe coagulopathy",
        "Inability to tolerate bronchoscopy"
    ],
    
    "key_findings": [
        "EBUS diagnostic yield: 92.5%",
        "Pneumothorax rate: 0.4%",
        "Superior to mediastinoscopy for staging"
    ]
}
```

### Chunk Types

The system creates specialized chunks:

1. **Diagnostic Chunks**: Contains yield percentages and metrics
2. **Complication Chunks**: Lists complications with rates and management
3. **Methodology Chunks**: Procedural techniques and equipment
4. **Pearl Chunks**: Clinical tips and recommendations
5. **Summary Chunks**: Key findings from the document
6. **General Chunks**: Standard text chunks with overlap

## Common Use Cases

### Use Case 1: Preparing for a Procedure

```python
# Get all information about EBUS-TBNA
procedure = "EBUS"

# Get diagnostic yields
yields = pipeline.query(procedure, query_type="yields")
print(f"Average yield: {sum(y['value'] for y in yields['yields'])/len(yields['yields']):.1f}%")

# Get complications
complications = pipeline.query(procedure, query_type="complications")
for comp in complications['complications']:
    print(f"- {comp['complication_type']}: {comp['rate']}%")

# Get clinical pearls
pearls = pipeline.query(procedure, query_type="pearls")
print(f"\nClinical Tips:")
for pearl in pearls['pearls'][:5]:
    print(f"- {pearl['pearl']}")
```

### Use Case 2: Literature Review

```python
# Find all information about diagnostic yields
import sqlite3
conn = sqlite3.connect("data/indices/clinical_data.db")

# Get yields by procedure
query = """
    SELECT procedure, metric, 
           AVG(value) as mean_value,
           MIN(value) as min_value,
           MAX(value) as max_value,
           COUNT(*) as n_studies
    FROM diagnostic_yields
    GROUP BY procedure, metric
    ORDER BY procedure, mean_value DESC
"""

results = pd.read_sql_query(query, conn)
print(results)
```

### Use Case 3: Teaching Preparation

```python
# Gather teaching points about bronchoscopy
teaching_material = {}

# Get procedures and techniques
methods = pipeline.query("bronchoscopy technique", query_type="general")
teaching_material['techniques'] = [chunk for chunk in methods['chunks'] 
                                  if chunk['chunk_type'] == 'methodology']

# Get complications to discuss
complications = pipeline.query("bronchoscopy", query_type="complications")
teaching_material['complications'] = complications['complications']

# Get clinical pearls
pearls = pipeline.query("bronchoscopy", query_type="pearls")
teaching_material['pearls'] = pearls['pearls'][:10]

# Create teaching outline
print("Bronchoscopy Teaching Points:")
print("\n1. Techniques:")
for tech in teaching_material['techniques'][:3]:
    print(f"   - Review from: {tech['source_document']}")

print("\n2. Complications to Discuss:")
for comp in teaching_material['complications']:
    print(f"   - {comp['complication_type']}: {comp['rate']}%")

print("\n3. Clinical Pearls:")
for pearl in teaching_material['pearls'][:5]:
    print(f"   - {pearl['pearl'][:100]}...")
```

## Maintenance

### Weekly Tasks

```python
# Check extraction quality
import json
from pathlib import Path

extracted_dir = Path("data/extracted")
low_confidence = []

for json_file in extracted_dir.glob("*.json"):
    with open(json_file) as f:
        data = json.load(f)
    
    if data['extraction_confidence'] < 0.7:
        low_confidence.append({
            'file': json_file.name,
            'confidence': data['extraction_confidence'],
            'title': data['title']
        })

if low_confidence:
    print(f"Documents needing review: {len(low_confidence)}")
    for doc in low_confidence:
        print(f"  - {doc['title']}: {doc['confidence']:.2f}")
```

### Monthly Tasks

```python
# Export clinical data for backup
import pandas as pd
import sqlite3

conn = sqlite3.connect("data/indices/clinical_data.db")

# Export all tables
tables = ['diagnostic_yields', 'complications', 'clinical_pearls']

for table in tables:
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    df.to_csv(f"data/backup_{table}.csv", index=False)
    print(f"Exported {len(df)} records from {table}")

conn.close()
```

## Troubleshooting

### Issue: PDF text extraction fails

```python
# Try alternative extraction
import pdfplumber
import PyPDF2

def extract_pdf_robust(pdf_path):
    text = ""
    
    # Try pdfplumber first
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""
        if text.strip():
            return text
    except:
        pass
    
    # Fallback to PyPDF2
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text()
    except:
        pass
    
    return text
```

### Issue: Low extraction quality

```python
# Re-process with different parameters
from ipchat.extraction.clinical_extractor import ClinicalDataExtractor

extractor = ClinicalDataExtractor()

# Get more context for extraction
with open("problematic_document.txt") as f:
    content = f.read()

# Extract with validation
extraction = extractor.extract(content, document_type="research")

# Check what was missed
if not extraction.diagnostic_yields:
    print("No yields found - check for alternative terminology")
    # Search for alternative patterns
    import re
    alt_patterns = ['success rate', 'detection rate', 'accuracy']
    for pattern in alt_patterns:
        if pattern in content.lower():
            print(f"Found '{pattern}' - consider updating extraction patterns")
```

### Issue: Search returns irrelevant results

```python
# Tune search parameters
results = pipeline.query(
    query_text="your query",
    query_type="general"
)

# Filter by chunk type
relevant = [r for r in results['chunks'] 
           if r['chunk_type'] in ['diagnostic', 'methodology']]

# Or use keyword search only
pipeline.index_manager.search(
    query="exact phrase",
    search_type="keyword"  # Skip semantic search
)
```

## Summary

This practical clinical extraction system provides:

1. **Focused Extraction**: Gets the clinical information you actually need
2. **Fast Queries**: Instant lookup of yields, complications, and pearls
3. **Flexible Search**: Hybrid semantic and keyword search
4. **Simple Maintenance**: Easy to understand and modify
5. **Scalable**: Handles your ~1000 documents efficiently

The system extracts granular clinical information that goes beyond simple overviews, giving you diagnostic yields, complication rates, methodologies, and clinical pearls - exactly what you need for clinical reference and decision support.
