#!/usr/bin/env python3
"""
Test script for the medical extraction pipeline
Verifies that the extraction workflow is working correctly
"""

import sys
import json
from pathlib import Path
from medical_extractor import MedicalExtractor

def test_extraction_pipeline():
    """Test the extraction pipeline with verification"""
    
    print("="*60)
    print("EXTRACTION PIPELINE TEST")
    print("="*60)
    
    # Initialize extractor
    extractor = MedicalExtractor()
    
    # 1. Check directories
    print("\n1. Checking directories...")
    dirs_ok = True
    for name, path in [
        ("Input", extractor.input_dir),
        ("PDF", extractor.pdf_dir),
        ("Output", extractor.output_dir),
        ("Batch", extractor.batch_dir)
    ]:
        exists = path.exists()
        print(f"   {name:8} {path}: {'✓' if exists else '✗'}")
        if not exists:
            dirs_ok = False
            
    if not dirs_ok:
        print("✗ Some directories are missing!")
        return False
    
    # 2. Check for input files
    print("\n2. Checking input files...")
    input_files = list(extractor.input_dir.glob("*.json"))
    print(f"   Found {len(input_files)} Adobe JSON files")
    
    if not input_files:
        print("✗ No input files found!")
        return False
    
    # 3. Check for existing extractions
    print("\n3. Checking existing extractions...")
    extractions = extractor.list_extractions()
    print(f"   Found {len(extractions)} completed extractions")
    
    # 4. Verify a sample extraction
    if extractions:
        print("\n4. Verifying sample extraction...")
        sample = extractor.output_dir / extractions[0]
        report = extractor.verify_extraction(sample)
        
        print(f"   File: {report['file'][:50]}...")
        print(f"   Quality Score: {report['quality_score']}/100")
        print(f"   Has Metadata: {'✓' if report['has_metadata'] else '✗'}")
        print(f"   Has Outcomes: {'✓' if report['has_outcomes'] else '✗'}")
        
        if report['quality_score'] >= 60:
            print("   ✓ Extraction quality is good")
        else:
            print("   ⚠ Extraction quality is low")
    
    # 5. Check for PDFs
    print("\n5. Checking PDF files...")
    pdf_files = list(extractor.pdf_dir.glob("*.pdf"))
    print(f"   Found {len(pdf_files)} PDF files")
    
    # 6. Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Input Articles:    {len(input_files)}")
    print(f"PDF Files:         {len(pdf_files)}")
    print(f"Completed:         {len(extractions)}")
    print(f"Remaining:         {len(input_files) - len(extractions)}")
    print(f"Completion:        {len(extractions)/len(input_files)*100:.1f}%")
    
    print("\n✓ Extraction pipeline is configured correctly!")
    
    # Show example commands
    print("\n" + "="*60)
    print("EXAMPLE COMMANDS")
    print("="*60)
    print("\n# Extract a single file:")
    if input_files:
        example = input_files[0].name
        print(f'python tools/medical_extractor.py --single "{example}"')
    
    print("\n# Process remaining files:")
    print("python tools/medical_extractor.py --batch")
    
    print("\n# List all extractions:")
    print("python tools/medical_extractor.py --list")
    
    print("\n# Verify extraction quality:")
    if extractions:
        print(f'python tools/medical_extractor.py --verify "{extractions[0]}"')
    
    return True


if __name__ == "__main__":
    success = test_extraction_pipeline()
    sys.exit(0 if success else 1)