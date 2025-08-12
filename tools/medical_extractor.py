#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Medical Evidence Extractor
Main extraction pipeline for converting Adobe JSON/PDFs to OpenEvidence format

Workflow:
1. Adobe JSON + PDF → OpenEvidence extraction (oe_final format)
2. Missing data recovery from PDFs when Adobe JSON is incomplete
3. Batch processing support with parallelization
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List
import concurrent.futures

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import extraction modules
from extractor_gpt5_oe_final import (
    extract_one_oe_final,
    read_json,
    extract_text_with_pages,
    read_pdf_with_pages,
    build_oe_final_prompt,
    call_gpt5_oe_final,
    post_process_extraction,
    calculate_derived_measures,
    parse_p_value
)

# Configuration
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
API_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "300"))
MAX_WORKERS = int(os.getenv("MAX_PARALLEL_EXTRACTIONS", "3"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))

# Directory setup - FIXED to use correct paths
INPUT_DIR = Path("data/input_articles")
PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/oe_final_outputs")  # Correct output directory
BATCH_DIR = Path("data/oe_batch_outputs")

# Ensure directories exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR.mkdir(parents=True, exist_ok=True)


class MedicalExtractor:
    """Unified medical evidence extraction pipeline"""
    
    def __init__(self):
        """Initialize the extractor"""
        self.input_dir = INPUT_DIR
        self.pdf_dir = PDF_DIR
        self.output_dir = OUTPUT_DIR
        self.batch_dir = BATCH_DIR
        
    def extract_single(self, json_path: Path, pdf_path: Optional[Path] = None) -> Tuple[Optional[Path], Optional[str]]:
        """
        Extract evidence from a single document
        
        Args:
            json_path: Path to Adobe JSON file
            pdf_path: Optional path to PDF file for additional context
            
        Returns:
            Tuple of (output_path, error_message)
        """
        print(f"\n{'='*60}")
        print(f"Processing: {json_path.name}")
        print(f"{'='*60}")
        
        try:
            # Use the OE-final extraction pipeline
            output_path, error = extract_one_oe_final(json_path, pdf_path)
            
            if error:
                print(f"✗ Extraction failed: {error}")
                return None, error
            else:
                print(f"✓ Extraction complete: {output_path}")
                return Path(output_path), None
                
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"✗ {error_msg}")
            return None, error_msg
    
    def extract_batch(self, 
                     pattern: str = "*.json",
                     max_files: Optional[int] = None,
                     resume_from: Optional[str] = None) -> Dict[str, Any]:
        """
        Process multiple documents in batch
        
        Args:
            pattern: Glob pattern for input files
            max_files: Maximum number of files to process
            resume_from: Path to previous batch summary to resume from
            
        Returns:
            Batch processing summary
        """
        print(f"\n{'='*60}")
        print(f"BATCH EXTRACTION")
        print(f"{'='*60}")
        
        # Get list of files to process
        json_files = sorted(self.input_dir.glob(pattern))
        
        if max_files:
            json_files = json_files[:max_files]
            
        # Check for already processed files
        already_processed = set()
        
        if resume_from:
            # Load previous batch summary
            with open(resume_from, 'r') as f:
                prev_summary = json.load(f)
                already_processed = {
                    d['file'] for d in prev_summary['details'] 
                    if d['success']
                }
        
        # Check existing outputs in oe_final_outputs directory
        for output_file in self.output_dir.glob("*.oe_final.json"):
            base_name = output_file.name.replace('.oe_final.json', '.json')
            already_processed.add(base_name)
        
        # Filter files to process
        files_to_process = []
        for json_path in json_files:
            if json_path.name not in already_processed:
                # Try to find matching PDF
                pdf_name = json_path.stem + ".pdf"
                pdf_path = self.pdf_dir / pdf_name
                if not pdf_path.exists():
                    pdf_path = None
                    
                files_to_process.append((json_path, pdf_path))
        
        if not files_to_process:
            print("No new files to process!")
            print(f"Already have {len(already_processed)} extractions in {self.output_dir}")
            return {"status": "complete", "files_processed": 0}
        
        print(f"Files to process: {len(files_to_process)}")
        print(f"Already processed: {len(already_processed)}")
        print(f"Parallel workers: {MAX_WORKERS}")
        
        # Process in parallel batches
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit tasks
            future_to_file = {}
            for i, (json_path, pdf_path) in enumerate(files_to_process):
                # Add delay between submissions
                if i > 0:
                    time.sleep(RATE_LIMIT_DELAY)
                    
                future = executor.submit(self.extract_single, json_path, pdf_path)
                future_to_file[future] = (json_path, pdf_path)
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_file):
                json_path, pdf_path = future_to_file[future]
                try:
                    output_path, error = future.result(timeout=600)
                    results.append({
                        "file": json_path.name,
                        "success": error is None,
                        "error": error,
                        "output": str(output_path) if output_path else None
                    })
                except Exception as e:
                    results.append({
                        "file": json_path.name,
                        "success": False,
                        "error": str(e),
                        "output": None
                    })
        
        # Save batch summary
        summary = self._save_batch_summary(results)
        
        return summary
    
    def _save_batch_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Save batch processing summary"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_path = self.batch_dir / f"batch_summary_{timestamp}.json"
        
        summary = {
            "timestamp": timestamp,
            "total_files": len(results),
            "successful": sum(1 for r in results if r['success']),
            "failed": sum(1 for r in results if not r['success']),
            "details": results
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"\n{'='*60}")
        print(f"BATCH COMPLETE")
        print(f"Total: {summary['total_files']}")
        print(f"Success: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Summary: {summary_path}")
        print(f"{'='*60}")
        
        return summary
    
    def verify_extraction(self, output_path: Path) -> Dict[str, Any]:
        """
        Verify extraction quality and completeness
        
        Args:
            output_path: Path to extracted JSON file
            
        Returns:
            Verification report
        """
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        report = {
            "file": output_path.name,
            "has_metadata": bool(data.get('metadata', {}).get('title')),
            "has_outcomes": bool(data.get('outcomes', {}).get('primary')),
            "has_population": bool(data.get('population', {}).get('total')),
            "outcome_count": len(data.get('outcomes', {}).get('primary', [])),
            "table_count": len(data.get('tables', [])),
            "quality_score": 0
        }
        
        # Calculate quality score
        if report['has_metadata']:
            report['quality_score'] += 25
        if report['has_outcomes']:
            report['quality_score'] += 35
        if report['has_population']:
            report['quality_score'] += 20
        if report['table_count'] > 0:
            report['quality_score'] += 20
            
        return report
    
    def list_extractions(self) -> List[str]:
        """List all completed extractions"""
        extractions = sorted(self.output_dir.glob("*.oe_final.json"))
        return [f.name for f in extractions]


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="Unified Medical Evidence Extractor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Directory Structure:
  Input:  data/input_articles/  (Adobe JSON files)
  PDFs:   data/raw_pdfs/        (Optional PDF files)
  Output: data/oe_final_outputs/ (Extracted OpenEvidence JSON)
  Logs:   data/oe_batch_outputs/ (Batch processing summaries)

Examples:
  # Extract single document
  python medical_extractor.py --single "paper.json" --pdf "paper.pdf"
  
  # Batch process all JSON files
  python medical_extractor.py --batch
  
  # Process specific pattern with parallelization
  python medical_extractor.py --batch --pattern "A*.json" --workers 4
  
  # Resume from previous batch
  python medical_extractor.py --batch --resume "batch_summary_20240808.json"
  
  # Verify extraction quality
  python medical_extractor.py --verify "paper.oe_final.json"
  
  # List all completed extractions
  python medical_extractor.py --list
        """
    )
    
    # Action arguments
    parser.add_argument("--single", type=str, help="Extract single document (JSON path)")
    parser.add_argument("--batch", action="store_true", help="Batch process documents")
    parser.add_argument("--verify", type=str, help="Verify extraction quality")
    parser.add_argument("--list", action="store_true", help="List all completed extractions")
    
    # Optional arguments
    parser.add_argument("--pdf", type=str, help="PDF file for single extraction")
    parser.add_argument("--pattern", type=str, default="*.json", help="File pattern for batch")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--max-files", type=int, help="Maximum files to process")
    parser.add_argument("--resume", type=str, help="Resume from batch summary")
    
    args = parser.parse_args()
    
    # Initialize extractor
    extractor = MedicalExtractor()
    
    # Handle commands
    if args.single:
        # Single file extraction
        json_path = Path(args.single)
        if not json_path.exists():
            # Try in input directory
            json_path = INPUT_DIR / args.single
            
        if not json_path.exists():
            print(f"Error: File not found: {args.single}")
            sys.exit(1)
            
        pdf_path = None
        if args.pdf:
            pdf_path = Path(args.pdf)
            if not pdf_path.exists():
                pdf_path = PDF_DIR / args.pdf
                
        output_path, error = extractor.extract_single(json_path, pdf_path)
        
        if error:
            sys.exit(1)
            
    elif args.batch:
        # Batch processing
        summary = extractor.extract_batch(
            pattern=args.pattern,
            max_files=args.max_files,
            resume_from=args.resume
        )
        
        if summary['failed'] > 0:
            print(f"\nWarning: {summary['failed']} files failed to process")
            
    elif args.verify:
        # Verify extraction
        output_path = Path(args.verify)
        if not output_path.exists():
            output_path = OUTPUT_DIR / args.verify
            
        if not output_path.exists():
            print(f"Error: File not found: {args.verify}")
            sys.exit(1)
            
        report = extractor.verify_extraction(output_path)
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION VERIFICATION")
        print(f"{'='*60}")
        print(f"File: {report['file']}")
        print(f"Quality Score: {report['quality_score']}/100")
        print(f"Has Metadata: {'✓' if report['has_metadata'] else '✗'}")
        print(f"Has Outcomes: {'✓' if report['has_outcomes'] else '✗'}")
        print(f"Has Population: {'✓' if report['has_population'] else '✗'}")
        print(f"Outcome Count: {report['outcome_count']}")
        print(f"Table Count: {report['table_count']}")
        print(f"{'='*60}")
        
    elif args.list:
        # List all extractions
        extractions = extractor.list_extractions()
        print(f"\n{'='*60}")
        print(f"COMPLETED EXTRACTIONS IN {OUTPUT_DIR}")
        print(f"{'='*60}")
        print(f"Total: {len(extractions)} files")
        print(f"{'='*60}")
        for i, name in enumerate(extractions[:10], 1):
            print(f"{i:3}. {name}")
        if len(extractions) > 10:
            print(f"... and {len(extractions) - 10} more")
        print(f"{'='*60}")
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()