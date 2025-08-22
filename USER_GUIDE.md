# Bronchmonkey User Guide
**Version 2.0** | Updated: August 22, 2025

## Table of Contents
1. [Overview](#overview)
2. [System Architecture](#system-architecture)
3. [Getting Started](#getting-started)
4. [Using the Chat Interface](#using-the-chat-interface)
5. [Data Extraction Pipeline](#data-extraction-pipeline)
6. [Clinical Data Migration](#clinical-data-migration)
7. [Advanced Features](#advanced-features)
8. [API Reference](#api-reference)
9. [Troubleshooting](#troubleshooting)

## Overview

Bronchmonkey is an AI-powered research assistant designed specifically for **interventional pulmonology and critical care research**. It provides instant access to medical evidence through a sophisticated hybrid search system and natural language interface.

### Key Capabilities
- 🔍 **Hybrid Search**: Combines vector, keyword, and SQL search for optimal retrieval
- 💬 **Natural Language Interface**: Ask questions in plain English
- 📚 **Comprehensive Knowledge Base**: 292+ medical studies and textbook chapters
- 🏥 **Clinical Focus**: Specialized for interventional pulmonology procedures
- 📊 **Structured Extraction**: Automated extraction of diagnostic yields, complications, and clinical pearls
- 📖 **Professional Citations**: Automatic author-year citations and MLA bibliography

## System Architecture

```
Bronchmonkey/
├── Frontend Layer
│   └── Streamlit Chat Interface (chatbot_app.py)
├── Processing Layer
│   ├── ipchat/extraction/     # Document extraction
│   ├── ipchat/processing/     # Text processing
│   └── ipchat/migration/      # Data migration
├── Retrieval Layer
│   ├── FAISS Vector Search    # Semantic search
│   ├── BM25 Keyword Search    # Term matching
│   └── PostgreSQL             # Structured queries
└── Data Layer
    ├── data/raw_pdfs/         # Source documents
    ├── data/migrated_extracted/ # Processed extractions
    └── data/indices/          # Search indices
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- PostgreSQL (optional, for full SQL search)
- 4GB+ RAM recommended
- OpenAI API key

### Installation

1. **Clone and enter the repository:**
```bash
cd IP_chat2
```

2. **Set up environment:**
```bash
# Create virtual environment
python -m venv .venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

3. **Configure environment:**
```bash
# Copy example environment file
cp .env.example .env

# Edit .env and add your OpenAI API key
OPENAI_API_KEY=your_actual_api_key_here
```

4. **Initialize the system:**
```bash
# Run setup script
./setup.sh  # Mac/Linux
# OR
python setup.sh  # Windows
```

### Quick Start

Start Bronchmonkey with one command:
```bash
./start.sh  # Mac/Linux
# OR
python chatbot_app.py  # Windows
```

The application will open at `http://localhost:8501`

## Using the Chat Interface

### Basic Queries

Ask natural language questions about medical procedures:

**Example queries:**
- "What is the diagnostic yield of EBUS-TBNA for lung cancer?"
- "Compare complication rates between rigid and flexible bronchoscopy"
- "What are the contraindications for bronchial thermoplasty?"
- "Show me pneumothorax rates for BLVR procedures"

### Understanding Responses

Responses include:
- **Direct answers** with specific data points
- **In-text citations** in (Author Year) format
- **Confidence scores** for search results
- **Source bibliography** in MLA format

### Search Strategies

The system uses three complementary search methods:

1. **Vector Search (50% weight)**: Finds semantically similar content
2. **Keyword Search (30% weight)**: Matches exact medical terms
3. **SQL Search (20% weight)**: Queries structured data fields

## Data Extraction Pipeline

### Processing New Documents

#### 1. Single Document Extraction
```python
from ipchat.extraction.clinical_extractor import ClinicalDataExtractor

extractor = ClinicalDataExtractor()
with open('document.txt', 'r') as f:
    content = f.read()
    
result = extractor.extract(content, document_type="research")
print(f"Found {len(result.diagnostic_yields)} diagnostic yield metrics")
print(f"Found {len(result.complication_rates)} complications")
```

#### 2. Batch Processing
```python
from ipchat.extraction.unified_extractor import UnifiedExtractor
from pathlib import Path

extractor = UnifiedExtractor()
input_dir = Path("data/raw_pdfs")

for pdf_file in input_dir.glob("*.pdf"):
    # Extract and process
    result = extractor.process_document(pdf_file)
    # Save to migrated_extracted
    output_path = Path("data/migrated_extracted") / f"{pdf_file.stem}.json"
    result.save(output_path)
```

### Extraction Output Structure

Each extraction produces:
```json
{
  "document_id": "unique_identifier",
  "title": "Study Title",
  "document_type": "research|textbook",
  "clinical_extraction": {
    "diagnostic_yields": {
      "procedure_name": {
        "sensitivity": 95.2,
        "specificity": 100.0
      }
    },
    "complication_rates": {
      "pneumothorax": {
        "rate": 2.4,
        "management": "chest tube drainage"
      }
    },
    "clinical_pearls": [
      "Maintain bronchoscope in neutral position"
    ],
    "key_findings": [
      "EBUS-TBNA diagnostic yield: 92.5%"
    ]
  }
}
```

## Clinical Data Migration

### Evaluating Existing Extractions

```python
from ipchat.migration.evaluator import ExtractionEvaluator

evaluator = ExtractionEvaluator()
result = evaluator.evaluate_extraction(extraction_path)
print(f"Quality Score: {result['score']}/100")
print(f"Recommendation: {result['recommendation']}")
```

### Migration Workflow

1. **Evaluate** existing extractions
2. **Backup** original data
3. **Migrate** based on quality score:
   - Score 80+: Keep and enhance
   - Score 50-79: Augment with clinical data
   - Score 25-49: Restructure completely
   - Score <25: Re-extract from source

### Running Migration

```python
from ipchat.migration.migrator import ExtractionMigrator

migrator = ExtractionMigrator()
migrator.migrate_all(
    source_dir=Path("data/oe_final_outputs"),
    output_dir=Path("data/migrated_extracted"),
    evaluation_report=Path("data/evaluation_report.json")
)
```

## Advanced Features

### Custom Extraction Schemas

Define custom extraction templates:
```python
from ipchat.schemas.textbook import TextbookSchema

schema = TextbookSchema()
schema.add_field("procedure_steps", type="list")
schema.add_field("equipment_required", type="list")
```

### API Integration

Start the FastAPI server:
```bash
python -m ipchat.api.server
```

Query the API:
```python
import requests

response = requests.post(
    "http://localhost:8000/search",
    json={"query": "EBUS diagnostic yield"}
)
results = response.json()
```

### Database Management

Initialize PostgreSQL database:
```sql
-- Run SQL scripts
psql -U postgres -d bronchmonkey -f sql/schema.sql
psql -U postgres -d bronchmonkey -f sql/populate.sql
```

### Index Rebuilding

Rebuild search indices after adding new documents:
```python
from ipchat.core.indexing.index_builder import IndexBuilder

builder = IndexBuilder()
builder.rebuild_all_indices(
    documents_path="data/migrated_extracted",
    output_path="data/indices"
)
```

## API Reference

### Core Classes

#### ClinicalDataExtractor
```python
extractor = ClinicalDataExtractor()
result = extractor.extract(content, document_type="research")
```

#### UnifiedExtractor
```python
extractor = UnifiedExtractor(model="gpt-4o-mini")
result = extractor.extract(content, document_type, metadata)
```

#### ExtractionMigrator
```python
migrator = ExtractionMigrator()
migrator.migrate_all(source_dir, output_dir, evaluation_report)
```

### Search Functions

```python
from ipchat.core.retrieval.hybrid_search import HybridSearch

searcher = HybridSearch()
results = searcher.search(
    query="bronchoscopy complications",
    top_k=10,
    weights={"vector": 0.5, "bm25": 0.3, "sql": 0.2}
)
```

## Troubleshooting

### Common Issues

#### 1. Import Errors
```bash
# Ensure you're in the virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows
```

#### 2. OpenAI API Errors
```bash
# Check your API key
echo $OPENAI_API_KEY
# Ensure it's set in .env file
```

#### 3. Database Connection Issues
```bash
# Check PostgreSQL is running
pg_ctl status
# Start if needed
pg_ctl start
```

#### 4. Memory Issues
```python
# For large batches, process in chunks
from ipchat.processing.chunker import ChunkProcessor

processor = ChunkProcessor(chunk_size=10)
processor.process_directory("data/raw_pdfs")
```

### Performance Optimization

1. **Reduce search scope**: Use specific document types
2. **Adjust weights**: Tune search weights for your use case
3. **Cache results**: Enable caching for repeated queries
4. **Batch processing**: Process multiple documents together

### Getting Help

- **Documentation**: Check `/docs` folder
- **Issues**: Review `CLEANUP_REPORT.md` for structure
- **Logs**: Check `data/logs/` for error details

## Best Practices

### For Researchers
1. Use specific medical terminology in queries
2. Request numerical data explicitly ("show percentages")
3. Compare procedures using "versus" or "compared to"
4. Ask for citations when needed

### For Developers
1. Always backup before migration
2. Validate extractions with the evaluator
3. Use appropriate document types (research vs textbook)
4. Monitor extraction confidence scores
5. Keep indices updated after adding documents

## Appendix

### Supported Procedures
- EBUS-TBNA (Endobronchial Ultrasound)
- Rigid & Flexible Bronchoscopy
- Navigational Bronchoscopy
- Cryobiopsy
- Bronchial Thermoplasty
- Pleural Procedures
- BLVR (Bronchoscopic Lung Volume Reduction)
- Airway Stenting

### File Structure Reference
```
data/
├── raw_pdfs/              # Original PDF documents
├── oe_final_outputs/      # Legacy extractions
├── migrated_extracted/    # Enhanced extractions
├── indices/               # Search indices
├── backup/                # Backup copies
└── evaluation_report.json # Migration analysis
```

### Environment Variables
```bash
OPENAI_API_KEY=            # Required: OpenAI API key
DATABASE_URL=              # Optional: PostgreSQL connection
LOG_LEVEL=INFO            # Optional: Logging level
MAX_TOKENS=2000           # Optional: Response length
TEMPERATURE=0.0           # Optional: AI creativity (0-1)
```

---

*For additional support, consult the technical documentation in `/docs` or review the codebase in `/ipchat`.*