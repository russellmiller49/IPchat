# 🚀 Bronchmonkey Quick Reference Card

## Essential Commands (Copy & Paste These!)

### 📦 Setup (First Time Only)
```bash
# 1. Install everything
pip install -r requirements.txt

# 2. Add your OpenAI key
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# 3. Build the database
./rebuild_knowledge_base.sh
```

### 🎯 Daily Operations

#### Start Bronchmonkey
```bash
./start.sh
# Then open browser to: http://localhost:8501
```

#### Add ONE New Paper
```bash
# With PDF
python tools/medical_extractor.py --single "PaperName.json" --pdf "PaperName.pdf"

# Without PDF (JSON only)
python tools/medical_extractor.py --single "PaperName.json"

# Then rebuild
./rebuild_knowledge_base.sh
```

#### Add MANY Papers
```bash
# Process all papers in input folder
python tools/medical_extractor.py --batch

# Then rebuild
./rebuild_knowledge_base.sh
```

#### Check What You Have
```bash
# See extraction status
python tools/check_extraction_status.py

# List all extracted papers
python tools/medical_extractor.py --list
```

### 🔍 Search Examples

**Good Questions to Ask:**
- "What are the pneumothorax rates for endobronchial valves?"
- "Compare outcomes between rigid and flexible bronchoscopy"
- "Show me studies with FEV1 improvement > 15%"
- "What are complications of bronchial thermoplasty?"
- "Find all studies with more than 100 patients"

**Getting Better Results:**
- ✅ Be specific: "Zephyr valve 12-month outcomes"
- ❌ Too vague: "valve results"
- ✅ Use medical terms: "bronchoscopic lung volume reduction"
- ❌ Too simple: "lung treatment"

### 📁 Where Files Go

| File Type | Put It Here | Example |
|-----------|------------|---------|
| Adobe JSON | `data/input_articles/` | Study2024.json |
| PDF | `data/raw_pdfs/` | Study2024.pdf |
| Extracted Data | `data/oe_final_outputs/` | Study2024.oe_final.json |
| Your API Key | `.env` file in main folder | OPENAI_API_KEY=sk-... |

### 🆘 Quick Fixes

| Problem | Solution |
|---------|----------|
| "Command not found" | `cd /path/to/IPchat` first |
| "No API key" | Create `.env` file with key |
| "Module not found" | `pip install -r requirements.txt` |
| "Port in use" | Kill with `pkill streamlit` |
| "Extraction failed" | Wait 1 min, try again |
| "No search results" | Run `./rebuild_knowledge_base.sh` |
| Bronchmonkey won't start | Check `.env` file has API key |

### ⚡ Speed Tips

1. **Process papers in batches** (not one by one)
2. **Use JSON without PDF** when possible (faster)
3. **Set parallel workers to 5** for faster batch processing
4. **Use gpt-4o-mini model** for routine tasks (cheapest/fastest)

### 💰 Save Money

- Use `gpt-4o-mini` for regular searches (cheapest option)
- Use `gpt-4o` for standard extractions (balanced cost)
- Only use `gpt-5` for complex/critical extractions
- Process in batches to reduce API calls
- Set `RATE_LIMIT_DELAY=2.0` to avoid rate limits

### 📊 Status Indicators

When you see these, here's what they mean:
- ✓ = Success/Complete
- ✗ = Failed/Error  
- ⚠ = Warning/Partial
- 🔄 = Processing
- ⏳ = Waiting

### 🎓 Example Workflow

**Monday: Add new papers**
```bash
# 1. Copy files to folders
# 2. Extract them
python tools/medical_extractor.py --batch
# 3. Rebuild index
./rebuild_knowledge_base.sh
# 4. Start and use
./start.sh
```

**Daily: Just search**
```bash
# Start and ask questions
./start.sh
```

**Friday: Check everything**
```bash
# See what's loaded
python tools/check_extraction_status.py
# Verify quality
python tools/medical_extractor.py --verify "ImportantPaper.oe_final.json"
```

### 📝 Notes Section
_Use this space for your API key, common searches, or notes:_

```
API Key: sk-_______________________
Common searches:
- 
- 
- 
Notes:
- 
- 
```

### 🔗 Help Resources

- **Full Guide**: See `USER_GUIDE.md`
- **Technical Details**: See `EXTRACTION_WORKFLOW.md`
- **Visual Diagrams**: See `WORKFLOW_DIAGRAM.md`
- **GitHub Issues**: Report problems online

### Remember: You Don't Need to Understand the Code!
Just copy and paste the commands exactly as shown. Bronchmonkey handles all the complex parts automatically.

---
*Print this page and keep it handy!*
*Version 1.0 | August 2024*