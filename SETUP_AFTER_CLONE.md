# Setup Instructions After Cloning from GitHub

## Required Data Files

After cloning this repository, you'll need to ensure the following data files are present:

### Essential Files (MUST HAVE):
The `data/oe_final_outputs/` directory contains 292 extracted medical research JSON files that are **required** for the application to function. These should be included in the repository.

### Files That Will Be Generated:
The following files/directories are excluded from git but will be automatically created when you run the setup:

1. **Search Indexes** (created by setup.sh):
   - `data/index/faiss.index` - Vector search index
   - `data/index/bm25.pkl` - Keyword search index
   - `data/index/*.jsonl` - Metadata files
   - `data/chunks/` - Document chunks

2. **Environment Configuration**:
   - `.env` - Created from `.env.example`

## Setup Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd IP_chat2
```

### 2. Check Required Data Files
```bash
# Verify the essential data files exist
ls -la data/oe_final_outputs/ | head -5
# Should show JSON files like "A Multicenter RCT of Zephyr Endobronchial Valv.oe_final.json"
```

### 3. Run Setup Script
```bash
# This will:
# - Install Python dependencies
# - Create .env from template
# - Setup PostgreSQL database
# - Build search indexes from the data files
# - Create document chunks

./setup.sh
```

### 4. Configure API Keys
Edit the `.env` file with your OpenAI API key:
```bash
nano .env
# Add: OPENAI_API_KEY=sk-your-actual-key-here
```

### 5. Start the Application
```bash
./start.sh
# Access at http://localhost:8501
```

## What Gets Built Locally

When you run `setup.sh`, the following will be created from the source data:

1. **PostgreSQL Database**: 
   - 292 studies loaded from `data/oe_final_outputs/`
   - Structured medical evidence tables

2. **FAISS Vector Index**:
   - ~874 document chunks
   - Semantic search capabilities

3. **BM25 Keyword Index**:
   - ~8508 vocabulary terms
   - Exact term matching

4. **Document Chunks**:
   - Granular text segments for retrieval
   - Generated from the JSON source files

## Troubleshooting

### Missing Data Files
If `data/oe_final_outputs/` is empty or missing:
- Contact the repository maintainer for the data files
- These are extracted medical research papers required for the app

### Index Building Fails
```bash
# Rebuild indexes manually:
python3 chunking/chunker.py --trials-dir data/oe_final_outputs
python3 indexing/build_faiss.py
python3 indexing/build_bm25.py
```

### Database Issues
```bash
# Reset database:
psql -c "DROP DATABASE IF EXISTS ip_rag"
psql -c "CREATE DATABASE ip_rag"
psql -d ip_rag -f sql/schema.sql
python3 ingestion/load_json_to_pg.py --trials-dir data/oe_final_outputs
```

## File Size Considerations

The `data/oe_final_outputs/` directory contains ~292 JSON files (~100MB total). If these files are too large for regular git, consider:

1. **Git LFS** (Recommended):
```bash
git lfs track "data/oe_final_outputs/*.json"
git add .gitattributes
git commit -m "Track large JSON files with LFS"
```

2. **External Storage**:
- Host files on cloud storage (S3, Google Drive)
- Provide download script in setup.sh

3. **Compressed Archive**:
- Include a `data.tar.gz` file
- Extract during setup

## Support

If you encounter issues with missing data or setup, please open an issue on GitHub.