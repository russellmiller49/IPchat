#!/usr/bin/env python3
"""Test batch extraction with just 2 chapters"""

import subprocess
import sys
from pathlib import Path

# Test with just 2 chapters first
test_chapters = [
    "Airway Anatomy.pdf",
    "Approach to Peripheral Lung Lesions.pdf"
]

# Create test directory with just these chapters
test_dir = Path("Textbooks/test_batch")
test_dir.mkdir(exist_ok=True)
(test_dir / "Chapter pdfs").mkdir(exist_ok=True)
(test_dir / "Chapter json").mkdir(exist_ok=True)

# Copy test chapters
for chapter in test_chapters:
    pdf_src = Path(f"Textbooks/Chapter pdfs/{chapter}")
    json_src = Path(f"Textbooks/Chapter json/{chapter[:-4]}.json")
    
    if pdf_src.exists():
        import shutil
        shutil.copy(pdf_src, test_dir / "Chapter pdfs" / chapter)
        if json_src.exists():
            shutil.copy(json_src, test_dir / "Chapter json" / f"{chapter[:-4]}.json")

print("Running test batch extraction on 2 chapters...")
result = subprocess.run([
    sys.executable, 
    "tools/production_multipass_textbook_extractor.py",
    "--batch",
    "--output-dir", "data/test_batch_extractions"
], capture_output=False, text=True)

sys.exit(result.returncode)