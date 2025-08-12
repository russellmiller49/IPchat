#!/usr/bin/env python3
"""
Check the status of the extraction pipeline without dependencies
"""

import json
from pathlib import Path

def check_extraction_status():
    """Check extraction pipeline status"""
    
    print("="*60)
    print("EXTRACTION PIPELINE STATUS")
    print("="*60)
    
    # Directory paths
    input_dir = Path("data/input_articles")
    pdf_dir = Path("data/raw_pdfs")
    output_dir = Path("data/oe_final_outputs")
    batch_dir = Path("data/oe_batch_outputs")
    
    # 1. Check directories
    print("\n1. Directory Structure:")
    for name, path in [
        ("Input", input_dir),
        ("PDF", pdf_dir),
        ("Output", output_dir),
        ("Batch", batch_dir)
    ]:
        exists = path.exists()
        print(f"   {name:8} {path}: {'✓ EXISTS' if exists else '✗ MISSING'}")
    
    # 2. Count files
    print("\n2. File Counts:")
    
    if input_dir.exists():
        input_files = list(input_dir.glob("*.json"))
        print(f"   Adobe JSON files:  {len(input_files)}")
    else:
        input_files = []
        print("   Adobe JSON files:  N/A (directory missing)")
    
    if pdf_dir.exists():
        pdf_files = list(pdf_dir.glob("*.pdf"))
        print(f"   PDF files:         {len(pdf_files)}")
    else:
        pdf_files = []
        print("   PDF files:         N/A (directory missing)")
    
    if output_dir.exists():
        output_files = list(output_dir.glob("*.oe_final.json"))
        print(f"   Extracted files:   {len(output_files)}")
    else:
        output_files = []
        print("   Extracted files:   N/A (directory missing)")
    
    if batch_dir.exists():
        batch_files = list(batch_dir.glob("batch_summary_*.json"))
        print(f"   Batch summaries:   {len(batch_files)}")
    else:
        batch_files = []
        print("   Batch summaries:   N/A (directory missing)")
    
    # 3. Processing status
    if input_files and output_files:
        print("\n3. Processing Status:")
        
        # Get list of processed files
        processed = set()
        for f in output_files:
            base_name = f.name.replace('.oe_final.json', '.json')
            processed.add(base_name)
        
        # Get list of unprocessed files
        unprocessed = []
        for f in input_files:
            if f.name not in processed:
                unprocessed.append(f.name)
        
        completion_pct = len(output_files) / len(input_files) * 100
        
        print(f"   Total input:       {len(input_files)} files")
        print(f"   Completed:         {len(output_files)} files")
        print(f"   Remaining:         {len(unprocessed)} files")
        print(f"   Completion:        {completion_pct:.1f}%")
        
        if unprocessed and len(unprocessed) <= 5:
            print("\n   Unprocessed files:")
            for f in unprocessed[:5]:
                print(f"   - {f}")
    
    # 4. Latest batch summary
    if batch_files:
        print("\n4. Latest Batch Summary:")
        latest_batch = sorted(batch_files)[-1]
        
        try:
            with open(latest_batch, 'r') as f:
                summary = json.load(f)
            
            print(f"   File: {latest_batch.name}")
            print(f"   Time: {summary.get('timestamp', 'N/A')}")
            print(f"   Total: {summary.get('total_files', 0)}")
            print(f"   Success: {summary.get('successful', 0)}")
            print(f"   Failed: {summary.get('failed', 0)}")
            
            # Show failed files if any
            if summary.get('failed', 0) > 0:
                print("\n   Failed files:")
                for detail in summary.get('details', []):
                    if not detail.get('success'):
                        print(f"   - {detail.get('file', 'N/A')}: {detail.get('error', 'N/A')[:50]}")
        except Exception as e:
            print(f"   Error reading batch summary: {e}")
    
    # 5. Sample extraction check
    if output_files:
        print("\n5. Sample Extraction Check:")
        sample_file = output_files[0]
        
        try:
            with open(sample_file, 'r') as f:
                data = json.load(f)
            
            print(f"   File: {sample_file.name[:50]}...")
            print(f"   Has metadata: {'✓' if data.get('metadata', {}).get('title') else '✗'}")
            print(f"   Has outcomes: {'✓' if data.get('outcomes', {}).get('primary') else '✗'}")
            print(f"   Has population: {'✓' if data.get('population', {}).get('total') else '✗'}")
            print(f"   Table count: {len(data.get('tables', []))}")
            
        except Exception as e:
            print(f"   Error reading sample: {e}")
    
    print("\n" + "="*60)
    print("EXTRACTION WORKFLOW FILES")
    print("="*60)
    
    # Check for extraction scripts
    tools_dir = Path("tools")
    if tools_dir.exists():
        key_files = [
            "medical_extractor.py",
            "extractor_gpt5_oe_final.py",
            "extractor_gpt5_batch.py",
            "extract_missing_data.py",
            "EXTRACTION_WORKFLOW.md"
        ]
        
        for fname in key_files:
            fpath = tools_dir / fname
            exists = fpath.exists()
            print(f"   {fname:35} {'✓' if exists else '✗'}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    check_extraction_status()