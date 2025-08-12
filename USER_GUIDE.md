# 🐵 Bronchmonkey User Guide
## Your AI-Powered Medical Research Assistant

### 📚 Table of Contents
1. [What is Bronchmonkey?](#what-is-bronchmonkey)
2. [Choose Your Edition](#choose-your-edition)
3. [Initial Setup](#initial-setup)
4. [Daily Use Guide](#daily-use-guide)
5. [Advanced Features](#advanced-features)
6. [All Features Explained](#all-features-explained)
7. [Troubleshooting](#troubleshooting)

---

## What is Bronchmonkey?

Bronchmonkey is your personal AI assistant for medical research that can:
- 🔍 **Search** through 292+ medical papers instantly using hybrid search (vector + keyword + SQL)
- 💬 **Answer** complex medical questions with proper citations
- 🔬 **Depth Mode** for comprehensive analysis with multiple perspectives
- 🐛 **Debug Mode** to see the AI's reasoning process
- 📄 **Extract** data from new research papers automatically
- 📊 **Find** specific statistics and outcomes from studies
- 🎓 **Cite** sources properly in academic format (MLA)
- 🚀 **Three Editions** to match your needs (Lite, Full, Space)

---

## Choose Your Edition

Bronchmonkey comes in three editions to match your needs:

### 🚀 **Lite Edition** (Recommended for most users)
- **Best for**: Personal use, demos, quick research
- **Features**: Fast startup, no database needed, local indexes
- **Storage**: Uses local files only
- **Install**: `pip install -e ".[lite]"`

### 🏢 **Full Edition** (Enterprise/Team)
- **Best for**: Research teams, production deployments
- **Features**: PostgreSQL database, API server, authentication, all features
- **Storage**: PostgreSQL + distributed indexes
- **Install**: `pip install -e ".[full]"`

### ☁️ **Space Edition** (Cloud)
- **Best for**: Hugging Face Spaces, public demos
- **Features**: Pre-built indexes, cloud-optimized
- **Storage**: Pre-indexed data
- **Install**: `pip install -e ".[space]"`

---

## Initial Setup

### Quick Install (Lite Edition - Most Users)

```bash
# 1. Clone the repository
git clone https://github.com/russellmiller49/IPchat.git
cd IPchat

# 2. Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 3. Install Lite edition
pip install -e ".[lite]"

# 4. Set up your OpenAI API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 5. Build the knowledge base
./rebuild_knowledge_base.sh

# 6. Start Bronchmonkey!
ipchat run --edition lite
```

Then open your browser to: **http://localhost:8501**

### Full Edition Setup (Advanced)

```bash
# Install with PostgreSQL support
pip install -e ".[full]"

# Set up PostgreSQL (if not already installed)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:14

# Configure database
export DATABASE_URL=postgresql://postgres:postgres@localhost:5432/medical_rag

# Initialize database
psql $DATABASE_URL < sql/schema.sql

# Load data
python ingestion/load_json_to_pg.py --trials-dir data/oe_final_outputs

# Start Full edition
ipchat run --edition full
```

---

## Daily Use Guide

### 🚀 Starting Bronchmonkey

**Simple Start (After Setup)**:
```bash
# Activate virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Start your chosen edition
ipchat run --edition lite   # Most users
ipchat run --edition full   # Enterprise
```

### 💬 Using the Chat Interface

#### Basic Queries
Just type your medical question naturally:
- "What are the pneumothorax rates for endobronchial valves?"
- "Compare rigid vs flexible bronchoscopy outcomes"
- "Show FEV1 improvements at 12 months"

#### Using Special Modes

**🔬 Depth Mode** (Comprehensive Analysis):
1. Toggle "Depth Mode" in the sidebar
2. Ask your question
3. The AI will:
   - Generate multiple search queries
   - Search from different angles
   - Provide nuanced synthesis
   - Include contrasting viewpoints
   - Verify numeric claims

**🐛 Debug Mode** (See AI Reasoning):
1. Toggle "Debug Mode" in the sidebar
2. Ask your question
3. You'll see:
   - Search strategy used
   - Documents retrieved
   - Reasoning process
   - Citation extraction
   - Synthesis steps

#### Model Selection
Choose your AI model based on needs:
- **gpt-5-mini**: Fast, cheapest (routine queries)
- **gpt-4o**: Balanced (standard research)
- **gpt-5**: Best quality (complex analysis)

In Depth Mode, GPT-5 is automatically selected for maximum quality.

---

## Advanced Features

### 📊 Feature 1: Depth Mode Analysis

**What it does**: Provides comprehensive, nuanced analysis of medical evidence

**How to activate**:
1. Toggle "🔬 Depth Mode" in sidebar
2. Ask your question
3. Wait for multi-stage analysis

**What happens behind the scenes**:
- **Query Expansion**: Generates 4-5 query variations
- **Multi-Search**: Searches with all variations
- **Reranking**: Orders results by relevance
- **Contrastive Analysis**: Finds supporting AND conflicting evidence
- **Numeric Verification**: Double-checks all statistics
- **Synthesis**: Creates balanced, comprehensive answer

**Example**:
```
Question: "What are the long-term outcomes of bronchial thermoplasty?"

Depth Mode will:
1. Search for: "bronchial thermoplasty outcomes", "BT long-term results", 
   "asthma thermoplasty 5-year", "airway smooth muscle ablation"
2. Find studies with different follow-up periods
3. Identify both positive and negative outcomes
4. Verify specific percentages and p-values
5. Synthesize a nuanced answer with all perspectives
```

### 🐛 Feature 2: Debug Mode Transparency

**What it does**: Shows the AI's complete reasoning process

**Information displayed**:
- Search queries used
- Number of documents retrieved
- Relevance scores
- Citation extraction process
- Synthesis reasoning
- Model decisions

**Use cases**:
- Understanding how answers are generated
- Verifying search coverage
- Debugging unexpected results
- Learning the system's capabilities

### 📥 Feature 3: Extract Data from New Papers

**Complete workflow**:

1. **Prepare files**:
   ```bash
   # Put Adobe JSON in:
   data/input_articles/NewPaper.json
   
   # Put PDF in (optional):
   data/raw_pdfs/NewPaper.pdf
   ```

2. **Extract data**:
   ```bash
   # Single paper with PDF
   python tools/medical_extractor.py --single "NewPaper.json" --pdf "NewPaper.pdf"
   
   # Single paper without PDF
   python tools/medical_extractor.py --single "NewPaper.json"
   
   # Multiple papers
   python tools/medical_extractor.py --batch
   ```

3. **Rebuild knowledge base**:
   ```bash
   ./rebuild_knowledge_base.sh
   ```

4. **Restart Bronchmonkey**:
   ```bash
   # Stop with Ctrl+C, then:
   ipchat run --edition lite
   ```

### 🔍 Feature 4: Hybrid Search System

**Three search methods combined**:

1. **Vector Search (50% weight)**
   - Semantic similarity using FAISS
   - Finds conceptually related content
   - Good for: synonyms, related concepts

2. **BM25 Keyword Search (30% weight)**
   - Traditional term matching
   - Exact phrase finding
   - Good for: acronyms, specific terms

3. **SQL Database (20% weight)**
   - Structured queries on outcomes
   - Numerical comparisons
   - Good for: "p < 0.05", "FEV1 > 15%"

**Automatic score fusion** creates optimal ranking.

### 📚 Feature 5: Smart Citations

**Automatic formatting**:
- In-text: (Author Year) format
- Bibliography: Full MLA format
- Fallbacks for missing metadata

**Example output**:
```
According to recent studies, endobronchial valves show 
significant improvements (Criner 2018, Kemp 2017).

References:
- Criner, Gerald J., et al. "A Multicenter RCT of Zephyr 
  Endobronchial Valves." AJRCCM, 2018.
- Kemp, Samuel V., et al. "A Multicenter Randomized Controlled 
  Trial." AJRCCM, 2017.
```

### 🎯 Feature 6: Edition-Specific Commands

**Lite Edition**:
```bash
ipchat run --edition lite
ipchat index --rebuild        # Rebuild search indexes
ipchat stats                  # Show database statistics
```

**Full Edition**:
```bash
ipchat run --edition full
ipchat api --port 8000        # Start API server
ipchat db --migrate           # Update database schema
ipchat db --load              # Load new data
```

**Space Edition**:
```bash
ipchat run --edition space
ipchat export --format hf     # Export for Hugging Face
```

---

## All Features Explained

### System Capabilities

| Feature | Description | Editions |
|---------|-------------|----------|
| **Hybrid Search** | Vector + BM25 + SQL combined | All |
| **Depth Mode** | Comprehensive multi-angle analysis | All |
| **Debug Mode** | Transparent reasoning display | All |
| **GPT-5 Support** | Latest AI model integration | All |
| **Smart Citations** | Automatic MLA formatting | All |
| **Batch Extraction** | Process multiple papers | All |
| **PostgreSQL** | Structured data queries | Full only |
| **API Server** | REST API for integrations | Full only |
| **Authentication** | User access control | Full, Space |
| **Cloud Deployment** | Hugging Face ready | Space only |

### Configuration Options

**Environment Variables** (`.env` file):
```bash
# Core settings
OPENAI_API_KEY=sk-your-key-here
GEN_MODEL=gpt-5-mini          # Default model
DEPTH_FEATURES=1               # Enable depth mode
DEBUG_MODE=0                   # Debug off by default

# Search weights
VECTOR_WEIGHT=0.5              # Semantic search
BM25_WEIGHT=0.3               # Keyword search
SQL_WEIGHT=0.2                # Database queries

# Performance
MAX_PARALLEL_EXTRACTIONS=3     # Batch processing
CHUNK_SIZE=450                # Text chunk size
RATE_LIMIT_DELAY=1.0          # API throttling

# Full edition only
DATABASE_URL=postgresql://...  # PostgreSQL connection
API_PORT=8000                 # API server port

# Authentication (optional)
BASIC_AUTH_USERS=alice:pw1,bob:pw2
```

### Quick Command Reference

| Task | Command |
|------|---------|
| **Start Bronchmonkey** | `ipchat run --edition lite` |
| **Extract single paper** | `python tools/medical_extractor.py --single "paper.json"` |
| **Extract all papers** | `python tools/medical_extractor.py --batch` |
| **Rebuild indexes** | `./rebuild_knowledge_base.sh` |
| **Check status** | `python tools/check_extraction_status.py` |
| **List extractions** | `python tools/medical_extractor.py --list` |
| **Verify quality** | `python tools/medical_extractor.py --verify "paper.oe_final.json"` |

---

## Troubleshooting

### Installation Issues

**Problem**: "Module not found"
```bash
# Solution: Install the package properly
pip install -e ".[lite]"  # Note the -e flag for development mode
```

**Problem**: "No API key found"
```bash
# Solution: Create .env file
echo "OPENAI_API_KEY=sk-your-actual-key" > .env
```

### Runtime Issues

**Problem**: "Port already in use"
```bash
# Solution 1: Kill existing process
pkill -f streamlit
pkill -f uvicorn

# Solution 2: Use different port
ipchat run --edition lite --port 8502
```

**Problem**: "Search returns nothing"
```bash
# Solution: Rebuild indexes
./rebuild_knowledge_base.sh
```

**Problem**: "Extraction failed"
- Check API limits (wait if rate limited)
- Verify file formats (Adobe JSON required)
- Try without PDF if corrupted
- Check logs in `tools/archive/debug_logs/`

### Performance Issues

**Problem**: "Slow responses"
- Switch to gpt-5-mini for faster responses
- Disable Depth Mode for simple queries
- Reduce CHUNK_SIZE for faster indexing

**Problem**: "High API costs"
- Use gpt-5-mini as default model
- Process papers in batches
- Set RATE_LIMIT_DELAY=2.0

---

## Tips for Best Results

### Search Strategies
1. **Be specific**: "Zephyr valve pneumothorax rates" > "valve complications"
2. **Use medical terms**: Include proper medical terminology
3. **Specify timeframes**: "12-month outcomes" for temporal data
4. **Ask for comparisons**: "Compare X vs Y" for side-by-side analysis

### Using Depth Mode Effectively
- Enable for: Complex questions, controversial topics, comprehensive reviews
- Disable for: Simple lookups, known facts, quick checks

### Managing Citations
- Citations appear automatically in responses
- Full bibliography at the end of each answer
- Export citations for your papers

### Optimizing for Your Workflow
- **Researchers**: Use Full edition with PostgreSQL for complex queries
- **Clinicians**: Use Lite edition for quick evidence lookups
- **Students**: Enable Debug Mode to understand evidence synthesis
- **Teams**: Deploy Full edition with authentication

---

## Summary

Bronchmonkey provides state-of-the-art medical evidence retrieval with:
- **Three editions** for different use cases
- **Hybrid search** combining multiple strategies
- **Depth Mode** for comprehensive analysis
- **Debug Mode** for transparency
- **Smart citations** for academic use
- **GPT-5 support** for best quality

Start with Lite edition, upgrade to Full when needed, deploy to Space for sharing!

---

*Version 2.0 | Last updated: August 2025*
*For technical details, see [EXTRACTION_WORKFLOW.md](tools/EXTRACTION_WORKFLOW.md)*