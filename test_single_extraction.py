#!/usr/bin/env python3
"""
Test extraction on a single textbook chapter
"""
import json
import sys
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from ipchat.extract.textbook.pipeline import extract_textbook

def test_single_chapter():
    """Test extraction on Airway Anatomy chapter"""
    
    # Test chapter: Airway Anatomy
    pdf_path = Path("Textbooks/Chapter pdfs/Airway Anatomy.pdf")
    adobe_json_path = Path("Textbooks/Chapter json/Airway Anatomy.json")
    title = "Airway Anatomy"
    
    print("=" * 60)
    print("Testing textbook extraction on single chapter")
    print("=" * 60)
    print(f"Chapter: {title}")
    print(f"PDF: {pdf_path}")
    print(f"Adobe JSON: {adobe_json_path}")
    print()
    
    # Check files exist
    if not pdf_path.exists():
        print(f"❌ PDF not found: {pdf_path}")
        return False
    
    if not adobe_json_path.exists():
        print(f"❌ Adobe JSON not found: {adobe_json_path}")
        return False
    
    print("✅ Input files found")
    print()
    
    # Create output directory
    output_dir = Path("data/test_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        print("🔄 Running extraction...")
        result = extract_textbook(pdf_path, adobe_json_path, title)
        
        # Save output
        output_file = output_dir / f"{pdf_path.stem}.textbook.json"
        output_data = result.model_dump(mode='json')
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Extraction successful! Output saved to: {output_file}")
        print()
        
        # Display summary of extracted content
        print("Extraction Summary:")
        print("-" * 40)
        
        # Metadata
        if hasattr(result, 'metadata') and result.metadata:
            print(f"Title: {result.metadata.title if hasattr(result.metadata, 'title') else 'N/A'}")
            print(f"Chapter: {result.metadata.chapter_number if hasattr(result.metadata, 'chapter_number') else 'N/A'}")
            if hasattr(result.metadata, 'authors') and result.metadata.authors:
                print(f"Authors: {', '.join(result.metadata.authors)}")
        
        # Content sections
        print()
        print("Content Sections:")
        
        # Clinical procedures
        if hasattr(result, 'clinical_procedures') and result.clinical_procedures:
            print(f"  - Clinical Procedures: {len(result.clinical_procedures)} procedures")
            for i, proc in enumerate(result.clinical_procedures[:3], 1):
                name = proc.name if hasattr(proc, 'name') else 'Unnamed'
                print(f"      {i}. {name}")
            if len(result.clinical_procedures) > 3:
                print(f"      ... and {len(result.clinical_procedures) - 3} more")
        
        # Treatment algorithms
        if hasattr(result, 'treatment_algorithms') and result.treatment_algorithms:
            print(f"  - Treatment Algorithms: {len(result.treatment_algorithms)} algorithms")
        
        # Clinical guidelines
        if hasattr(result, 'clinical_guidelines') and result.clinical_guidelines:
            print(f"  - Clinical Guidelines: {len(result.clinical_guidelines)} guidelines")
        
        # Drug information
        if hasattr(result, 'drug_information') and result.drug_information:
            print(f"  - Drug Information: {len(result.drug_information)} drugs")
        
        # Tables
        if hasattr(result, 'tables') and result.tables:
            print(f"  - Tables: {len(result.tables)} tables")
            for i, table in enumerate(result.tables[:2], 1):
                caption = table.caption if hasattr(table, 'caption') else 'No caption'
                print(f"      {i}. {caption[:60]}...")
        
        # Figures
        if hasattr(result, 'figures') and result.figures:
            print(f"  - Figures: {len(result.figures)} figures")
        
        # Key points
        if hasattr(result, 'key_points') and result.key_points:
            print(f"  - Key Points: {len(result.key_points)} points")
        
        # Clinical cases
        if hasattr(result, 'clinical_cases') and result.clinical_cases:
            print(f"  - Clinical Cases: {len(result.clinical_cases)} cases")
        
        print()
        print("✅ Extraction test completed successfully!")
        print()
        print(f"📄 Full output saved to: {output_file}")
        print("   Review the file to check extraction quality.")
        
        return True
        
    except Exception as e:
        print(f"❌ Extraction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_single_chapter()
    sys.exit(0 if success else 1)