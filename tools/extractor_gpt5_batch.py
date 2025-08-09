#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-5 Medical Evidence Extractor - Batch Processing Version
Processes multiple documents in parallel for efficiency
"""

import argparse
import json
import os
import sys
import time
import concurrent.futures
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# Import the OE-final extractor functions
sys.path.append(str(Path(__file__).parent))
from extractor_gpt5_oe_final import (
    extract_one_oe_final,
    read_json,
    extract_text_with_pages,
    read_pdf_with_pages,
    build_oe_final_prompt,
    call_gpt5_oe_final,
    post_process_extraction,
    OUTPUT_DIR,
    INPUT_DIR,
    PDF_DIR
)

from dotenv import load_dotenv
load_dotenv()

# Batch processing configuration
MAX_WORKERS = int(os.getenv("MAX_PARALLEL_EXTRACTIONS", "3"))  # Parallel API calls
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "5"))  # Files per batch
RATE_LIMIT_DELAY = float(os.getenv("RATE_LIMIT_DELAY", "1.0"))  # Seconds between calls

# Create batch output directory
BATCH_OUTPUT_DIR = Path("data/oe_batch_outputs")
BATCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def process_single_document(args: Tuple[Path, Optional[Path], int]) -> Tuple[str, bool, Optional[str]]:
    """Process a single document (for parallel execution)."""
    json_path, pdf_path, index = args
    
    try:
        # Add delay to respect rate limits
        if index > 0:
            time.sleep(RATE_LIMIT_DELAY * index)
        
        print(f"[{index+1}] Processing: {json_path.name}")
        
        # Use the OE-final extraction
        out_path, error = extract_one_oe_final(json_path, pdf_path)
        
        if error:
            print(f"[{index+1}] ✗ Failed: {error}")
            return json_path.name, False, error
        else:
            print(f"[{index+1}] ✓ Complete: {json_path.name}")
            return json_path.name, True, None
            
    except Exception as e:
        print(f"[{index+1}] ✗ Error: {e}")
        return json_path.name, False, str(e)


def process_batch_parallel(file_list: List[Tuple[Path, Optional[Path]]]) -> List[Tuple[str, bool, Optional[str]]]:
    """Process a batch of files in parallel."""
    results = []
    
    # Prepare arguments with index for rate limiting
    args_list = [(json_path, pdf_path, i) for i, (json_path, pdf_path) in enumerate(file_list)]
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_single_document, args): args[0] 
            for args in args_list
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                result = future.result(timeout=600)  # 10 min timeout per file
                results.append(result)
            except concurrent.futures.TimeoutError:
                json_path = future_to_file[future]
                results.append((json_path.name, False, "Timeout"))
            except Exception as e:
                json_path = future_to_file[future]
                results.append((json_path.name, False, str(e)))
    
    return results


def save_batch_summary(results: List[Tuple[str, bool, Optional[str]]]):
    """Save batch processing summary."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = BATCH_OUTPUT_DIR / f"batch_summary_{timestamp}.json"
    
    summary = {
        "timestamp": timestamp,
        "total_files": len(results),
        "successful": sum(1 for _, success, _ in results if success),
        "failed": sum(1 for _, success, _ in results if not success),
        "details": [
            {
                "file": name,
                "success": success,
                "error": error
            }
            for name, success, error in results
        ]
    }
    
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nBatch summary saved to: {summary_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="GPT-5 Batch Medical Evidence Extractor")
    parser.add_argument("--dir", type=str, default=str(INPUT_DIR), help="Input directory")
    parser.add_argument("--pattern", type=str, default="*.json", help="File pattern to process")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Parallel workers")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Files per batch")
    parser.add_argument("--resume", type=str, help="Resume from batch summary JSON")
    
    args = parser.parse_args()
    
    input_dir = Path(args.dir)
    if not input_dir.exists():
        print(f"[ERROR] Directory not found: {input_dir}")
        sys.exit(1)
    
    # Get list of files to process
    json_files = sorted(input_dir.glob(args.pattern))
    
    # Auto-detect already processed files
    already_processed = set()
    
    # Check output directory for existing extractions
    for output_file in OUTPUT_DIR.glob("*.oe_final.json"):
        base_name = output_file.stem.replace(".oe_final", "")
        already_processed.add(f"{base_name}.json")
    
    # Filter out already processed files if resuming
    if args.resume:
        resume_path = Path(args.resume)
        if resume_path.exists():
            resume_data = json.loads(resume_path.read_text())
            processed = {d["file"] for d in resume_data["details"] if d["success"]}
            already_processed.update(processed)
    
    # Filter out already processed files
    original_count = len(json_files)
    json_files = [f for f in json_files if f.name not in already_processed]
    
    if already_processed:
        print(f"Auto-skip: Found {len(already_processed)} already processed files")
        print(f"Will process {len(json_files)} new files out of {original_count} total")
    
    if not json_files:
        print("No files to process")
        return
    
    print(f"\n{'='*60}")
    print(f"BATCH EXTRACTION")
    print(f"{'='*60}")
    print(f"Input directory: {input_dir}")
    print(f"Files to process: {len(json_files)}")
    print(f"Parallel workers: {args.workers}")
    print(f"Batch size: {args.batch_size}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"{'='*60}\n")
    
    # Prepare file pairs (JSON + PDF)
    file_pairs = []
    for json_path in json_files:
        pdf_path = PDF_DIR / f"{json_path.stem}.pdf"
        if not pdf_path.exists():
            pdf_path = None
        file_pairs.append((json_path, pdf_path))
    
    # Process in batches
    all_results = []
    total_batches = (len(file_pairs) + args.batch_size - 1) // args.batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * args.batch_size
        end_idx = min(start_idx + args.batch_size, len(file_pairs))
        batch = file_pairs[start_idx:end_idx]
        
        print(f"\n--- Batch {batch_num + 1}/{total_batches} ({len(batch)} files) ---")
        
        # Process batch in parallel
        if args.workers > 1:
            batch_results = process_batch_parallel(batch)
        else:
            # Sequential processing
            batch_results = []
            for i, (json_path, pdf_path) in enumerate(batch):
                result = process_single_document((json_path, pdf_path, i))
                batch_results.append(result)
        
        all_results.extend(batch_results)
        
        # Save intermediate results
        if batch_num % 5 == 0:  # Save every 5 batches
            save_batch_summary(all_results)
    
    # Final summary
    print(f"\n{'='*60}")
    print(f"BATCH EXTRACTION COMPLETE")
    print(f"{'='*60}")
    
    summary = save_batch_summary(all_results)
    
    print(f"Total processed: {summary['total_files']}")
    print(f"Successful: {summary['successful']}")
    print(f"Failed: {summary['failed']}")
    
    if summary['failed'] > 0:
        print("\nFailed files:")
        for detail in summary['details']:
            if not detail['success']:
                print(f"  - {detail['file']}: {detail['error']}")
    
    print(f"\nOutput directory: {OUTPUT_DIR}")
    print(f"Batch summaries: {BATCH_OUTPUT_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()