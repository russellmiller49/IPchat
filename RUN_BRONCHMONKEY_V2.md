# 🐵 Bronchmonkey V2 - GPT-5 Enhanced Edition

## 🚀 What's New in V2

### GPT-5 Integration
- **Quick Mode**: Uses `gpt-5-mini` for fast, concise responses
- **In-Depth Mode**: Uses `gpt-5-2025-08-07` for comprehensive analysis
- **Toggle Switch**: Easy switching between modes in the sidebar

### MLA Citations
- **Proper Academic Format**: All sources cited in MLA style
- **Author Names**: Correctly formatted (Last, First for primary author)
- **Multiple Authors**: Handles et al. for 3+ authors
- **Textbook Chapters**: Special formatting for book chapters

## 📱 Running Bronchmonkey V2

### Quick Start
```bash
# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Run the new version
streamlit run bronchmonkey_lite_v2.py
```

### Access the Interface
Open browser to: **http://localhost:8501**

## 🎯 Using the Two Modes

### ⚡ Quick Answer Mode (Default)
- **Model**: GPT-5-mini
- **Response**: 2-3 paragraphs
- **Speed**: Fast (~2-3 seconds)
- **Best for**: 
  - Quick clinical questions
  - Diagnostic yields
  - Simple comparisons
  - Rapid reference checks

**Example queries for Quick Mode:**
- "What is the diagnostic yield of EBUS?"
- "Pneumothorax rate for transbronchial biopsy?"
- "Main contraindications for bronchial thermoplasty?"

### 🔬 In-Depth Analysis Mode
- **Model**: GPT-5-2025-08-07 (Full GPT-5)
- **Response**: Comprehensive multi-paragraph analysis
- **Speed**: Slower (~5-10 seconds)
- **Best for**:
  - Complex clinical decisions
  - Comparing multiple studies
  - Understanding controversies
  - Detailed technique explanations
  - Literature reviews

**Example queries for In-Depth Mode:**
- "Compare all available diagnostic techniques for peripheral lung nodules including yields, complications, and cost-effectiveness"
- "Review the evidence for endobronchial valve placement in emphysema including patient selection criteria"
- "Analyze the role of cryobiopsy vs surgical lung biopsy for ILD diagnosis"

## 📚 MLA Citation Examples

The system now generates proper MLA citations:

### Journal Article:
```
Criner, Gerard J., et al. "A Multicenter Randomized Controlled Trial of 
Zephyr Endobronchial Valve Treatment in Heterogeneous Emphysema (LIBERATE)." 
*American Journal of Respiratory and Critical Care Medicine*, 2018.
```

### Textbook Chapter:
```
"Balloon Dilation Techniques." *Principles and Practice of Interventional 
Pulmonology*, Springer, 2025.
```

## 🎛️ Interface Features

### Sidebar Controls
- **Mode Toggle**: Switch between Quick/In-Depth
- **Model Display**: Shows current GPT-5 model
- **Knowledge Base Stats**: 292 articles + 41 chapters
- **Sample Queries**: Quick reference examples
- **Clear Chat**: Reset conversation

### Main Display
- **Metrics Bar**: Shows document counts and current mode
- **Chat History**: Maintains conversation context
- **Citations Section**: MLA-formatted references after each response

## 🔍 Search & Retrieval

### Quick Lookup
Instant answers for common queries:
- Diagnostic yields by procedure
- Complication rates
- Procedure steps
- Standard techniques

### Full Search
- **712 searchable chunks**
- **Keyword matching** with title boost
- **Context-aware** responses
- **Source tracking** for citations

## 💡 Tips for Best Results

### For Quick Mode:
1. Ask specific, focused questions
2. Request single data points or comparisons
3. Use medical terminology
4. Keep queries concise

### For In-Depth Mode:
1. Ask complex, multi-part questions
2. Request comprehensive reviews
3. Ask for clinical implications
4. Seek controversy discussions
5. Request evidence synthesis

## 📊 Performance Comparison

| Feature | Quick Mode (GPT-5-mini) | In-Depth Mode (GPT-5-full) |
|---------|-------------------------|----------------------------|
| Response Time | 2-3 seconds | 5-10 seconds |
| Response Length | 200-400 words | 800-1500 words |
| Detail Level | Key points only | Comprehensive analysis |
| Citations | 1-3 sources | 3-5 sources |
| Best Use | Quick reference | Research & decisions |
| Temperature | 0.2 (factual) | 0.3 (analytical) |

## 🧪 Testing the System

### Quick Test Queries:
1. **Quick Mode**: "EBUS diagnostic yield for lung cancer"
   - Should return: ~90% yield with 1-2 citations

2. **In-Depth Mode**: "Compare all bronchoscopic techniques for peripheral nodule diagnosis"
   - Should return: Detailed comparison with multiple studies

### Verify Citations:
- Check that author names are properly formatted
- Verify journal names are italicized
- Confirm years are included

## 🛠️ Troubleshooting

### If GPT-5 models aren't working:
```bash
# Test your API key supports GPT-5
python -c "import openai; client = openai.OpenAI(); print(client.models.list())"
```

### If citations appear incorrect:
- The system pulls from `data/indices/migrated_articles_index.json`
- Rebuild if needed: `python prepare_knowledge_base.py`

### If search returns no results:
- Check that indices exist in `data/indices/`
- Verify 712 chunks are loaded

## 📈 What's Next

### Planned Enhancements:
- Real-time model selection (GPT-5, GPT-4, etc.)
- Export citations to bibliography
- Save chat history
- Custom prompt templates
- Multi-document comparison mode

## 🎉 Ready to Use!

Your enhanced Bronchmonkey V2 with GPT-5 and MLA citations is ready:

```bash
streamlit run bronchmonkey_lite_v2.py
```

**Enjoy the power of GPT-5 for interventional pulmonology research!** 🐵

---

*For the original version without GPT-5, use `bronchmonkey_lite.py`*  
*For questions or issues, see `USER_GUIDE.md`*