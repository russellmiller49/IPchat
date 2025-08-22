# Bronchmonkey Quick Start Guide
**Get up and running in 5 minutes!**

## 🚀 Installation (One-Time Setup)

### 1. Prerequisites
- Python 3.8+
- Git
- OpenAI API key

### 2. Clone & Setup
```bash
# Clone repository (or use existing)
cd IP_chat2

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
```bash
# Create .env file from template
cp .env.example .env

# Edit .env and add your OpenAI API key:
# OPENAI_API_KEY=sk-your-actual-key-here
```

## 🎯 Start Bronchmonkey

```bash
# Simple start command
python chatbot_app.py
```

Open browser to: **http://localhost:8501**

## 💬 Example Queries

Try these queries in the chat interface:

### Clinical Questions
- "What is the diagnostic yield of EBUS-TBNA?"
- "Show pneumothorax rates for lung volume reduction"
- "Compare rigid vs flexible bronchoscopy complications"

### Procedure Information
- "What are contraindications for bronchial thermoplasty?"
- "How do you manage post-bronchoscopy bleeding?"
- "Best practices for navigational bronchoscopy"

### Evidence Requests
- "Show me studies on cryobiopsy for ILD diagnosis"
- "What's the sensitivity of EBUS for staging lung cancer?"
- "Adverse events in endobronchial valve trials"

## 📊 Understanding Responses

Each response includes:
- **Direct Answer**: Key findings and data
- **Citations**: (Author Year) format inline
- **Confidence Score**: Search relevance 0-1
- **Bibliography**: Full MLA citations at bottom

## 🔧 Common Tasks

### Process New Documents
```python
from ipchat.extraction.clinical_extractor import ClinicalDataExtractor

# Extract from text
extractor = ClinicalDataExtractor()
result = extractor.extract(document_text, "research")
```

### Search the Knowledge Base
```python
from ipchat.core.retrieval.hybrid_search import HybridSearch

# Search for information
searcher = HybridSearch()
results = searcher.search("EBUS complications", top_k=5)
```

### Migrate/Update Extractions
```python
from ipchat.migration.migrator import ExtractionMigrator

# Migrate existing data
migrator = ExtractionMigrator()
migrator.migrate_all(source_dir, output_dir, evaluation_report)
```

## 📁 Key Directories

```
IP_chat2/
├── ipchat/                 # Core application code
├── data/
│   ├── migrated_extracted/ # Knowledge base (292 studies)
│   ├── raw_pdfs/          # Source PDFs
│   └── indices/           # Search indices
├── chatbot_app.py         # Main application
└── .env                   # Configuration
```

## ⚡ Quick Commands

| Task | Command |
|------|---------|
| Start Bronchmonkey | `python chatbot_app.py` |
| Test extraction | `python -m ipchat.extraction.clinical_extractor` |
| Check status | `curl http://localhost:8501` |
| View logs | `tail -f data/logs/bronchmonkey.log` |
| Stop application | `Ctrl+C` in terminal |

## 🆘 Troubleshooting

### App won't start
```bash
# Check virtual environment is activated
which python  # Should show .venv path

# Reinstall dependencies
pip install -r requirements.txt
```

### OpenAI errors
```bash
# Verify API key is set
cat .env | grep OPENAI_API_KEY

# Test API key
python -c "import openai; print('API key configured')"
```

### No search results
```bash
# Check data directory has files
ls data/migrated_extracted/ | wc -l  # Should show 292

# Rebuild indices if needed
python -m ipchat.core.indexing.index_builder
```

## 📚 Learn More

- **Full User Guide**: See `USER_GUIDE.md`
- **Technical Docs**: Check `/docs` folder
- **API Reference**: `ipchat/api/README.md`
- **Migration Guide**: `data/MIGRATION_RESULTS.md`

## 💡 Tips

1. **Be specific**: Use medical terminology for better results
2. **Ask for numbers**: "What percentage..." gets statistical data
3. **Compare procedures**: Use "versus" or "compared to"
4. **Request citations**: Ask "with citations" for academic use

---

**Ready to explore?** Start asking questions about interventional pulmonology!

*Need help? Check the full USER_GUIDE.md or review docs/ folder*