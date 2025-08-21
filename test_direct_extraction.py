#!/usr/bin/env python3
"""
Test direct extraction to diagnose the issue
"""

import sys
import json
from pathlib import Path

# Add tools to path
sys.path.insert(0, 'tools')

from production_multipass_textbook_extractor import extract_multipass_production

print("Testing direct extraction...")
print("-" * 50)

# Test with the same files
pdf_path = Path("Textbooks/Chapter pdfs/Approach to Peripheral Lung Lesions.pdf")
adobe_json = Path("Textbooks/Chapter json/Approach to Peripheral Lung Lesions.json")

if not pdf_path.exists():
    print(f"❌ PDF not found: {pdf_path}")
    sys.exit(1)

if not adobe_json.exists():
    print(f"⚠️ Adobe JSON not found: {adobe_json}")
    adobe_json = None

print(f"PDF: {pdf_path}")
print(f"Adobe JSON: {adobe_json}")
print()

# Try extraction with verbose output
try:
    print("Starting extraction...")
    result = extract_multipass_production(
        pdf_path=pdf_path,
        adobe_json_path=adobe_json,
        chapter_title="Approach to Peripheral Lung Lesions",
        model="gpt-4o",  # Try with GPT-4o first
        passes_to_run=["pass0_metadata"]  # Just test metadata pass
    )
    
    print("\nExtraction completed!")
    print(f"Result type: {type(result)}")
    print(f"Result keys: {list(result.keys())}")
    
    # Check metadata
    metadata = result.get('chapter_metadata', {})
    print(f"\nMetadata extracted:")
    print(f"  Title: {metadata.get('title', {})}")
    print(f"  Authors: {len(metadata.get('authors', []))} authors")
    print(f"  Key points: {len(metadata.get('key_points', []))} points")
    
    # Check other sections
    for section in ['diagnostic_approaches', 'clinical_guidelines', 'tables']:
        count = len(result.get(section, []))
        print(f"  {section}: {count} items")
    
    # Save for inspection
    with open("test_extraction_debug.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n✅ Saved to test_extraction_debug.json for inspection")
    
except Exception as e:
    print(f"\n❌ Extraction failed with error:")
    print(f"  Type: {type(e).__name__}")
    print(f"  Message: {str(e)}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()