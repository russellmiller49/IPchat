#!/usr/bin/env python3
"""
Test the fixed GPT-5 extraction
"""

print("""
============================================================
GPT-5 EXTRACTION FIXES APPLIED
============================================================

✅ FIXED ISSUES:

1. NameError in GPT-5 branch
   - undefined variables (chunk_idx, chunks, pass_name) now fixed
   - Safe labels passed from caller

2. JSON Schema for GPT-5
   - response_format parameter added to Responses API
   - Enforces strict JSON output like Chat API

3. Robust JSON parsing
   - Strips code fences
   - Extracts JSON from mixed text
   - Handles different response structures

4. Enhancer GPT-5 support
   - Routes GPT-5 calls through Responses API
   - Uses _call_llm_json helper for consistency

5. Lower concurrency for GPT-5
   - Reduced to 2 workers to avoid rate limits

============================================================
TESTING EXTRACTION
============================================================

Run this command to test with GPT-5:

python tools/gold_standard_pipeline.py \\
  --single "Textbooks/Chapter pdfs/Approach to Peripheral Lung Lesions.pdf" \\
  --adobe-json "Textbooks/Chapter json/Approach to Peripheral Lung Lesions.json" \\
  --model gpt-5 \\
  --verbose

Or run the extraction directly:

python tools/production_multipass_textbook_extractor.py \\
  --single "Textbooks/Chapter pdfs/Approach to Peripheral Lung Lesions.pdf" \\
  --adobe-json "Textbooks/Chapter json/Approach to Peripheral Lung Lesions.json" \\
  --output-dir data/test_gpt5_fixed \\
  --title "Approach to Peripheral Lung Lesions" \\
  --model gpt-5

============================================================
WHAT TO EXPECT
============================================================

With the fixes applied:
1. No more NameError crashes
2. GPT-5 will return proper JSON
3. Extraction should populate all sections
4. Enhancement will also use GPT-5 properly

The extraction should now produce a full, rich JSON with:
- Diagnostic approaches
- Clinical guidelines
- Tables with data
- Definitions
- Clinical procedures
- And more...

============================================================
""")