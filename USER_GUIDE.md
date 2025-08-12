# Bronchmonkey User Guide
## Your AI-Powered Medical Research Assistant

### 📚 Table of Contents
1. [What is Bronchmonkey?](#what-is-bronchmonkey)
2. [Initial Setup (One-Time)](#initial-setup-one-time)
3. [Daily Use Guide](#daily-use-guide)
4. [All Features Explained](#all-features-explained)
5. [Troubleshooting](#troubleshooting)

---

## What is Bronchmonkey?

Bronchmonkey is your personal AI assistant for medical research that can:
- 🔍 **Search** through 292+ medical papers instantly
- 💬 **Answer** complex medical questions with citations
- 📄 **Extract** data from new research papers automatically
- 📊 **Find** specific statistics and outcomes from studies
- 🎓 **Cite** sources properly in academic format

Think of it as having a research assistant who has read every paper in your library and can instantly find and explain anything you need.

---

## Initial Setup (One-Time)

### Prerequisites Checklist
Before starting, you need:
- ✅ A computer with Windows, Mac, or Linux
- ✅ Python installed (version 3.8 or newer)
- ✅ An OpenAI API key (get one at https://platform.openai.com)
- ✅ About 2GB of free disk space

### Step 1: Download the Project
```bash
# Open your terminal/command prompt and run:
git clone https://github.com/yourusername/IPchat.git
cd IPchat
```

### Step 2: Install Requirements
```bash
# Install all necessary packages (this takes 5-10 minutes)
pip install -r requirements.txt
```

### Step 3: Set Up Your API Key
1. Create a file named `.env` in the main folder
2. Add this line (replace with your actual key):
```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

### Step 4: Build the Knowledge Base
```bash
# This processes all the medical papers (takes about 5 minutes)
./rebuild_knowledge_base.sh
```

---

## Daily Use Guide

### 🚀 Starting Bronchmonkey

**Option 1: Simple Start (Recommended)**
```bash
# Just run this command:
./start.sh
```
Then open your browser to: http://localhost:8501

**Option 2: Manual Start**
```bash
# Terminal 1: Start the search engine
uvicorn backend.api.main:app --reload

# Terminal 2: Start the chat interface
streamlit run chatbot_app.py
```

### 💬 Using the Chat Interface

Once Bronchmonkey is running:

1. **Ask Medical Questions**
   - Type: "What are the outcomes for bronchial thermoplasty?"
   - Get: Detailed answer with citations from multiple studies

2. **Find Specific Data**
   - Type: "Show me FEV1 improvements at 12 months"
   - Get: Exact statistics from relevant studies

3. **Compare Treatments**
   - Type: "Compare endobronchial valves vs coils for emphysema"
   - Get: Side-by-side comparison with evidence

4. **Request Summaries**
   - Type: "Summarize pneumothorax rates in BLVR studies"
   - Get: Comprehensive overview with percentages

---

## All Features Explained

### 📥 Feature 1: Extract Data from New Papers

**What it does**: Converts PDFs and research papers into searchable data

**How to use it**:

1. **Prepare your files**:
   - Put PDF files in: `data/raw_pdfs/`
   - Put Adobe JSON files in: `data/input_articles/`

2. **Extract a single paper**:
```bash
python tools/medical_extractor.py --single "YourPaper.json" --pdf "YourPaper.pdf"
```

3. **Extract many papers at once**:
```bash
python tools/medical_extractor.py --batch
```

4. **Check extraction status**:
```bash
python tools/check_extraction_status.py
```

**What you'll see**:
```
Processing: YourPaper.json
================================
✓ Extraction complete: data/oe_final_outputs/YourPaper.oe_final.json
```

### 🔍 Feature 2: Search the Knowledge Base

**What it does**: Find information across all papers instantly

**Search modes available**:

1. **Natural Language** (Just ask normally)
   - "What causes pneumothorax in COPD patients?"
   - "Best practices for bronchoscopy"

2. **Statistical Queries**
   - "Studies with p-value < 0.05"
   - "Trials with more than 100 patients"

3. **Specific Interventions**
   - "Zephyr valve outcomes"
   - "Rigid bronchoscopy complications"

### 📊 Feature 3: View Extraction Quality

**What it does**: Check if papers were properly extracted

**How to use it**:
```bash
python tools/medical_extractor.py --verify "PaperName.oe_final.json"
```

**What you'll see**:
```
EXTRACTION VERIFICATION
========================
File: PaperName.oe_final.json
Quality Score: 85/100
Has Metadata: ✓
Has Outcomes: ✓
Has Population: ✓
Outcome Count: 3
Table Count: 5
```

### 📚 Feature 4: Add Papers to Knowledge Base

**What it does**: Makes new papers searchable

**Step-by-step process**:

1. **Extract the paper** (if not already done):
```bash
python tools/medical_extractor.py --single "NewPaper.json"
```

2. **Rebuild the knowledge base**:
```bash
./rebuild_knowledge_base.sh
```

3. **Restart Bronchmonkey**:
```bash
# Stop with Ctrl+C, then:
./start.sh
```

The new paper is now searchable!

### 📈 Feature 5: Check System Status

**What it does**: Shows what's in your database

**How to use it**:
```bash
python tools/check_extraction_status.py
```

**What you'll see**:
```
EXTRACTION PIPELINE STATUS
==========================
Adobe JSON files:  312
PDF files:         312
Extracted files:   292
Completion:        93.6%

KNOWLEDGE BASE STATUS
====================
Chunks:     874
Index Size: 3.5MB
Ready:      ✓
```

### 🔧 Feature 6: Advanced Settings

**Change AI Model** (in `.env` file):
```
OPENAI_MODEL=gpt-4o-mini     # Fastest, cheapest (routine tasks)
OPENAI_MODEL=gpt-4o          # Balanced performance
OPENAI_MODEL=gpt-5            # Best quality (complex extractions)
```

**Adjust Processing Speed**:
```
MAX_PARALLEL_EXTRACTIONS=3    # How many papers at once
RATE_LIMIT_DELAY=1.0         # Seconds between API calls
```

**Change Chunk Size** (for search precision):
```
CHUNK_SIZE=450               # Tokens per chunk
CHUNK_OVERLAP=80            # Overlap between chunks
```

---

## Troubleshooting

### Problem: "Command not found"
**Solution**: Make sure you're in the IPchat folder:
```bash
cd /path/to/IPchat
```

### Problem: "No API key found"
**Solution**: Create `.env` file with your key:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### Problem: "Module not found"
**Solution**: Install missing packages:
```bash
pip install -r requirements.txt
```

### Problem: "Port already in use"
**Solution**: Stop other programs or use different port:
```bash
# Kill existing process
pkill -f streamlit
pkill -f uvicorn

# Or use different ports
streamlit run chatbot_app.py --server.port 8502
```

### Problem: "Extraction failed"
**Common causes**:
1. **API limit reached** → Wait a few minutes
2. **PDF corrupted** → Try without PDF (JSON only)
3. **File too large** → Split into smaller sections

### Problem: "Search returns nothing"
**Solution**: Rebuild the knowledge base:
```bash
./rebuild_knowledge_base.sh
```

---

## Quick Reference Card

### 🎯 Most Common Commands

| What You Want | Command to Use |
|--------------|---------------|
| Start Bronchmonkey | `./start.sh` |
| Add a new paper | `python tools/medical_extractor.py --single "paper.json"` |
| Process all papers | `python tools/medical_extractor.py --batch` |
| Rebuild search index | `./rebuild_knowledge_base.sh` |
| Check what's loaded | `python tools/check_extraction_status.py` |
| Stop Bronchmonkey | Press `Ctrl+C` in terminal |

### 📁 Important Folders

| Folder | What Goes There |
|--------|----------------|
| `data/input_articles/` | Adobe JSON files to process |
| `data/raw_pdfs/` | PDF files (optional) |
| `data/oe_final_outputs/` | Extracted data (don't modify) |
| `data/chunks/` | Search index files |

### 💡 Pro Tips

1. **Better Search Results**:
   - Be specific: "endobronchial valve pneumothorax rates" 
   - Not vague: "valve problems"

2. **Faster Processing**:
   - Process papers in batches, not one by one
   - Use JSON files without PDFs when possible

3. **Save Money on API**:
   - Use `gpt-4o-mini` for routine tasks (cheapest)
   - Use `gpt-4o` for standard extractions
   - Use `gpt-5` only for complex/critical extractions

4. **Backup Your Work**:
   - Copy `data/oe_final_outputs/` folder regularly
   - Save `.env` file securely

---

## Getting Help

### Resources
- **Technical Issues**: Check the Troubleshooting section above
- **Understanding Results**: Look for citations at the bottom of responses
- **Adding Papers**: Follow Feature 1 instructions step-by-step

### File Locations
- **This guide**: `USER_GUIDE.md`
- **Technical docs**: `EXTRACTION_WORKFLOW.md`
- **Pipeline details**: `KNOWLEDGE_BASE_PIPELINE.md`

### Remember
- 🔄 Always rebuild knowledge base after adding papers
- 💾 Keep backups of your extracted data
- 🔑 Never share your API key publicly
- 📊 Check extraction quality for important papers

---

## Summary

Bronchmonkey turns complex medical research into simple Q&A:

1. **Setup Once**: Install → Configure → Build
2. **Use Daily**: Start → Ask → Get Citations
3. **Add Papers**: Extract → Rebuild → Search

You don't need to understand the code - just follow the commands exactly as shown, and Bronchmonkey will handle the technical details for you.

---

*Last updated: August 2025*
*Version: 1.0*