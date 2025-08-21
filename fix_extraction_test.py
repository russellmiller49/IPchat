#!/usr/bin/env python3
"""
Test extraction with GPT-4o to verify it works
"""

print("""
============================================================
DIAGNOSIS: Empty Extraction Issue
============================================================

The gold standard extraction returned empty results because:

1. The raw extraction (before enhancement) was already empty
2. Only "enhancer" added definitions were present
3. All content arrays were empty (no diagnostic approaches, tables, etc.)

This indicates the GPT-5 extraction is failing silently.

Let's test with GPT-4o to verify the extraction works:
============================================================
""")

import subprocess
import sys

# Command to run extraction with GPT-4o
cmd = [
    sys.executable,
    "tools/production_multipass_textbook_extractor.py",
    "--single", "Textbooks/Chapter pdfs/Approach to Peripheral Lung Lesions.pdf",
    "--adobe-json", "Textbooks/Chapter json/Approach to Peripheral Lung Lesions.json",
    "--output-dir", "data/test_gpt4o_extraction",
    "--title", "Approach to Peripheral Lung Lesions",
    "--model", "gpt-4o"  # Use GPT-4o instead of GPT-5
]

print("Running extraction with GPT-4o...")
print("Command:", " ".join(cmd))
print("-" * 50)

result = subprocess.run(cmd, capture_output=True, text=True)

if result.returncode == 0:
    print("✅ Extraction completed successfully!")
    print("\nOutput:")
    print(result.stdout)
    
    # Check the output file
    import json
    from pathlib import Path
    
    output_file = Path("data/test_gpt4o_extraction/Approach to Peripheral Lung Lesions_production.json")
    if output_file.exists():
        with open(output_file) as f:
            data = json.load(f)
        
        print("\n" + "="*50)
        print("EXTRACTION RESULTS:")
        print("="*50)
        
        # Check content
        sections = [
            'diagnostic_approaches',
            'clinical_guidelines', 
            'tables',
            'clinical_procedures',
            'definitions'
        ]
        
        for section in sections:
            count = len(data.get(section, []))
            print(f"{section}: {count} items")
        
        # Check if it's empty
        total_content = sum(len(data.get(s, [])) for s in sections)
        
        if total_content > 20:
            print("\n✅ EXTRACTION SUCCESSFUL - Content extracted properly!")
            print("\nRECOMMENDATION: Use GPT-4o for now instead of GPT-5")
            print("The GPT-5 API may need different parameters or isn't responding correctly.")
        else:
            print("\n⚠️ Extraction produced minimal content")
            print("Check API keys and model availability")
    else:
        print("❌ Output file not found")
else:
    print("❌ Extraction failed!")
    print("Error:", result.stderr)

print("\n" + "="*50)
print("SOLUTION:")
print("="*50)
print("""
To fix the gold standard pipeline:

1. Use GPT-4o instead of GPT-5 for now:
   python tools/gold_standard_pipeline.py \\
     --single "chapter.pdf" \\
     --model gpt-4o

2. Or fix GPT-5 API calls:
   - Check if GPT-5 is available in your account
   - Verify the API parameters are correct
   - Add more verbose error handling

3. The enhancement system works fine - it's the extraction that's failing.
""")