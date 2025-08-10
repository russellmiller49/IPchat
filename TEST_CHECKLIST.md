# Pre-Deployment Test Checklist

## 🔍 Basic Functionality
- [ ] App starts without errors
- [ ] Search returns results
- [ ] Answer generation works
- [ ] Citations display correctly
- [ ] Sources expand properly

## 🔐 Authentication (if using BASIC_AUTH_USERS)
- [ ] Login prompt appears when env var set
- [ ] Correct credentials allow access
- [ ] Wrong credentials show error
- [ ] No auth required when env var not set

## 🔬 Depth Mode
- [ ] Toggle appears in sidebar
- [ ] Depth mode shows "expanding queries" message
- [ ] More sources returned (10 vs 5)
- [ ] Answer format changes to structured
- [ ] Settings display in sidebar when active

## 🤖 Model Switching
- [ ] Fast mode uses gpt-5-mini (or gpt-4o-mini for testing)
- [ ] Max mode uses gpt-5 (or gpt-4o for testing)
- [ ] Depth mode forces max model
- [ ] Model name shows in sidebar

## 📊 Data & Indexes
- [ ] FAISS index loads (check sidebar for chunk count)
- [ ] BM25 index works (keyword searches return results)
- [ ] API status shows green in sidebar

## 🚀 Performance
- [ ] Answer caching works (repeat questions are instant)
- [ ] Clear cache button works
- [ ] No memory leaks on multiple queries

## 🐛 Error Handling
- [ ] Handles API errors gracefully
- [ ] Shows meaningful error messages
- [ ] Still shows sources even if generation fails

## 📱 UI/UX
- [ ] Responsive layout
- [ ] Clear branding (Bronchmonkey)
- [ ] Intuitive controls
- [ ] Status indicators work

## 🔧 Environment Variables
```bash
# Test these configurations:
OPENAI_API_KEY=sk-...           # Required
BASIC_AUTH_USERS=test:pass      # Optional auth
GEN_MODEL=gpt-4o-2024-08-06    # Override model
EMBED_MODEL=intfloat/e5-small-v2
ENABLE_RERANKER=0               # Disable reranker if issues
```

## 🎯 Test Queries
Try these to verify quality:

### Simple Query
"What is the diagnostic yield of robotic bronchoscopy?"

### Depth Mode Query (toggle ON)
"Compare pneumothorax rates between BLVR valves and coils"

### Safety Query
"What are adverse events in bronchial thermoplasty?"

### Complex Query
"FEV1 improvement at 12 months for endobronchial valves in heterogeneous emphysema"

## 🚨 Common Issues & Fixes

### "No module named utils.openai_client"
```bash
# Make sure you're in the right directory
cd /path/to/IP_chat2
# Check file exists
ls utils/openai_client.py
```

### "API offline" in sidebar
```bash
# Check backend is running
curl http://localhost:8000/docs
# Restart if needed
uvicorn backend.api.main:app --reload --port 8000
```

### GPT-5 errors
```bash
# Use GPT-4 for testing since GPT-5 doesn't exist yet
export GEN_MODEL=gpt-4o-2024-08-06
```

### Authentication not working
```bash
# Check format is correct (colon separator)
export BASIC_AUTH_USERS="alice:password1,bob:password2"
```

## 📝 Final Pre-deployment Steps

1. **Test with real API key**
   ```bash
   export OPENAI_API_KEY=sk-your-actual-key
   ```

2. **Verify indexes are built**
   ```bash
   ls -la data/index/
   # Should see: faiss.index, bm25.pkl, meta.jsonl
   ```

3. **Check memory usage**
   ```bash
   docker stats
   # Should be under 2GB for HF Spaces free tier
   ```

4. **Test with slow connection**
   - Use browser dev tools to throttle network
   - Ensure timeouts are reasonable

5. **Clear test data**
   ```bash
   # Clear any test sessions
   rm -rf .streamlit/
   ```

## 🚀 Deploy to HF Spaces

Once all tests pass:

```bash
# Create HF Space (if not exists)
# Go to https://huggingface.co/spaces
# Create new Space → Docker → Private

# Add HF remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/bronchmonkey

# Push lite-perf branch as main
git push hf lite-perf:main

# Set secrets in HF Space settings:
# - OPENAI_API_KEY
# - BASIC_AUTH_USERS (optional)
```

## 📊 Monitor After Deployment

- Check Space logs for errors
- Test login if auth enabled
- Verify search and generation work
- Monitor usage/costs in OpenAI dashboard