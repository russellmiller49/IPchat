# 🚀 Bronchmonkey GPT-5 Edition - Direct Implementation

## ✨ What's New

This is the **direct GPT-5 implementation** without fallbacks, based on the HuggingFace version:

- **Direct GPT-5 Usage**: No fallback to GPT-4
- **Enhanced Search**: Query expansion and result reranking  
- **Better Retrieval**: Multi-query search with diversity enforcement
- **Cached Responses**: LRU caching for faster repeated queries
- **Improved Context**: Better evidence synthesis from multiple sources

## 🎯 Key Differences from V2

| Feature | V2 (with fallbacks) | GPT-5 Edition (direct) |
|---------|-------------------|----------------------|
| Model Selection | Tests availability, falls back to GPT-4 | Direct GPT-5 usage |
| Search | Simple keyword matching | Query expansion + reranking |
| Retrieval | Basic scoring | Multi-query with diversity |
| Caching | No caching | LRU cache for responses |
| Error Handling | Fallback to GPT-4 | Direct error reporting |

## 📦 Running the GPT-5 Edition

### Quick Start
```bash
# Activate virtual environment
source .venv/bin/activate  # Mac/Linux
# or
.venv\Scripts\activate     # Windows

# Run the GPT-5 edition
streamlit run bronchmonkey_gpt5.py
```

### Access Interface
Open browser to: **http://localhost:8501**

## 🔧 Model Configuration

The system uses these GPT-5 models directly:
- **Quick Mode**: `gpt-5-mini`
- **In-Depth Mode**: `gpt-5`

No fallback logic - if GPT-5 isn't available, you'll see an error.

## 🔍 Enhanced Search Features

### Query Expansion
Automatically expands queries with:
- Medical term extraction
- Acronym expansions (EBUS → endobronchial ultrasound)
- Multiple query variations

### Result Reranking
Scores results based on:
- Exact query matches (highest weight)
- Individual term frequencies
- Title relevance boost
- Recency preference (2020+ studies)
- Statistical data presence

### Diversity Enforcement
In depth mode, ensures:
- Maximum 2 chunks from same source
- Broader evidence coverage
- Multiple study perspectives

## 💡 Usage Tips

### For Best Results:
1. **Be specific** - Include procedure names and outcomes
2. **Use medical terms** - The system recognizes medical vocabulary
3. **Ask for comparisons** - "Compare X vs Y" works well
4. **Request statistics** - "What is the yield/rate/percentage"

### Example Queries:
- "Compare diagnostic yields of EBUS-TBNA vs mediastinoscopy for lung cancer staging"
- "What are the pneumothorax rates for different lung volume reduction techniques?"
- "Analyze outcomes of cryobiopsy vs surgical biopsy for ILD diagnosis"
- "Review evidence for bronchial thermoplasty in severe asthma"

## 🚨 Troubleshooting

### If you see "Model not found" errors:
Your API key may not have GPT-5 access yet. GPT-5 is being rolled out gradually.

### To test your GPT-5 access:
```bash
.venv/bin/python3 test_simple_gpt5.py
```

### If responses are slow:
- First queries take longer (building cache)
- Subsequent similar queries use cached results
- In-depth mode is naturally slower than quick mode

## 📊 Performance Expectations

| Mode | Response Time | Response Length | Best For |
|------|--------------|-----------------|----------|
| Quick | 1-2 seconds* | 2-3 paragraphs | Rapid lookups |
| In-Depth | 3-5 seconds* | 5-8 paragraphs | Research & analysis |

*After initial cache warming

## 🔄 Switching Between Versions

You have three versions available:

1. **bronchmonkey_lite.py** - Original GPT-4 version
2. **bronchmonkey_lite_v2.py** - GPT-5 with fallback to GPT-4
3. **bronchmonkey_gpt5.py** - Direct GPT-5 (this version)

Choose based on your API access and needs.

## 📈 What Makes This Version Better?

1. **More Robust Answers**: Enhanced search finds more relevant evidence
2. **Better Context**: Multi-query approach captures different perspectives  
3. **Faster Repeated Queries**: Response caching speeds up similar questions
4. **Higher Quality**: Direct GPT-5 usage without fallback compromises
5. **Smarter Retrieval**: Query expansion catches related concepts

## 🎉 Ready to Use!

```bash
streamlit run bronchmonkey_gpt5.py
```

Experience the full power of GPT-5 for interventional pulmonology research!

---

*Note: This version requires GPT-5 API access. If you don't have access yet, use bronchmonkey_lite_v2.py which includes fallback options.*