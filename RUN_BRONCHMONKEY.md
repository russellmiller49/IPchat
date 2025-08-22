# 🐵 Bronchmonkey - Ready to Run!

## ✅ System Status

Your Bronchmonkey system is **READY FOR TESTING** with:

### Knowledge Base Prepared
- **292 Research Articles** - Clinical trials, systematic reviews, and studies
- **41 Textbook Chapters** - Procedures, techniques, and guidelines  
- **712 Search Chunks** - Optimized for retrieval
- **Quick Lookup Index** - Fast access to common queries

### Indices Created
- `data/indices/migrated_articles_index.json` - Research article catalog
- `data/indices/combined_knowledge_base.json` - Unified knowledge base
- `data/indices/search_chunks.json` - Searchable content chunks
- `data/indices/quick_lookup.json` - Common queries pre-indexed

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)
```bash
python start_bronchmonkey.py
```
This will:
1. Check all requirements
2. Verify knowledge base is ready
3. Let you choose Lite or Full version
4. Start the application

### Option 2: Direct Launch - Lite Version
```bash
streamlit run bronchmonkey_lite.py
```
Simple, fast version with basic search

### Option 3: Direct Launch - Full Version
```bash
streamlit run chatbot_app.py
```
Advanced features with depth mode

## 📱 Using Bronchmonkey

### Access the Interface
Once running, open your browser to:
**http://localhost:8501**

### Example Queries to Try

#### Diagnostic Yields
- "What is the diagnostic yield of EBUS-TBNA?"
- "Show sensitivity and specificity for navigational bronchoscopy"
- "Compare diagnostic yields between cryobiopsy and forceps biopsy"

#### Complications
- "What are pneumothorax rates for transbronchial biopsy?"
- "Show bleeding complications for bronchoscopy"
- "How common is pneumothorax with BLVR?"

#### Procedures
- "How do you perform balloon bronchoplasty?"
- "What are the steps for EBUS-TBNA?"
- "Explain the technique for transbronchial cryobiopsy"

#### Clinical Questions
- "What are contraindications for bronchial thermoplasty?"
- "When should you use rigid vs flexible bronchoscopy?"
- "Management of persistent air leaks"

## 🎯 Features Available

### Lite Version (bronchmonkey_lite.py)
- ✅ Simple keyword search
- ✅ Quick lookup for common queries
- ✅ GPT-4 response generation
- ✅ Source citations
- ✅ Fast response times

### Full Version (chatbot_app.py)
- ✅ Everything in Lite
- ✅ Advanced hybrid search
- ✅ Depth mode for comprehensive analysis
- ✅ Debug mode to see reasoning
- ✅ Model selection (GPT-4/GPT-5)
- ✅ PostgreSQL integration (if configured)

## 🔧 Troubleshooting

### If the app won't start:
```bash
# Check Python version (needs 3.8+)
python --version

# Check virtual environment
source .venv/bin/activate  # Mac/Linux
.venv\Scripts\activate     # Windows

# Reinstall requirements
pip install -r requirements.txt
```

### If searches return no results:
```bash
# Rebuild knowledge base
python prepare_knowledge_base.py
```

### If OpenAI errors occur:
```bash
# Check API key
cat .env | grep OPENAI_API_KEY
```

## 📊 Knowledge Base Stats

| Category | Count |
|----------|-------|
| Research Articles | 292 |
| Textbook Chapters | 41 |
| Total Documents | 333 |
| Search Chunks | 712 |
| Unique Procedures | 30+ |
| Unique Conditions | 14+ |

## 🎉 You're Ready!

1. Run: `python start_bronchmonkey.py`
2. Choose: Option 1 (Lite) for testing
3. Browse to: http://localhost:8501
4. Start asking questions!

---

**Tips:**
- Start with simple queries to test the system
- Use medical terminology for better results
- Ask for specific data (yields, rates, percentages)
- The Lite version is perfect for initial testing

**Need help?** 
- Check `USER_GUIDE.md` for detailed instructions
- Review `QUICK_START.md` for setup help
- See `docs/` folder for technical documentation

---

*System prepared and ready for interventional pulmonology research assistance!*