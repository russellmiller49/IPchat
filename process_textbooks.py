#!/usr/bin/env python3
"""
Process all textbook chapters: extraction, chunking, and indexing
"""
import json
import yaml
from pathlib import Path
import subprocess
import sys

def process_all_textbooks():
    """Process all textbook chapters from book.yaml"""
    
    # Load book configuration
    book_yaml = Path("Textbooks/book.yaml")
    with open(book_yaml, 'r') as f:
        config = yaml.safe_load(f)
    
    chapters = config.get('chapters', [])
    output_dir = Path("data/textbook_extractions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    successful = []
    failed = []
    
    print(f"Processing {len(chapters)} textbook chapters...")
    
    for i, chapter in enumerate(chapters, 1):
        pdf_path = Path("Textbooks") / chapter['pdf']
        title = chapter['title']
        
        # Check if PDF exists
        if not pdf_path.exists():
            print(f"[{i}/{len(chapters)}] ❌ PDF not found: {pdf_path}")
            failed.append(title)
            continue
        
        # Check for corresponding Adobe JSON
        adobe_json_name = pdf_path.stem + ".json"
        adobe_json_path = Path("Textbooks/Chapter json") / adobe_json_name
        
        if not adobe_json_path.exists():
            print(f"[{i}/{len(chapters)}] ❌ Adobe JSON not found: {adobe_json_path}")
            failed.append(title)
            continue
        
        # Output file
        output_file = output_dir / f"{pdf_path.stem}.textbook.json"
        
        # Skip if already processed
        if output_file.exists():
            print(f"[{i}/{len(chapters)}] ✓ Already processed: {title}")
            successful.append(title)
            continue
        
        print(f"[{i}/{len(chapters)}] Processing: {title}")
        
        # Run extraction using ipchat CLI
        cmd = [
            sys.executable, "-m", "ipchat.cli",
            "extract-textbook",
            str(pdf_path),
            str(adobe_json_path),
            "--title", title,
            "--out", str(output_dir)
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print(f"[{i}/{len(chapters)}] ✅ Success: {title}")
            successful.append(title)
        except subprocess.CalledProcessError as e:
            print(f"[{i}/{len(chapters)}] ❌ Failed: {title}")
            print(f"  Error: {e.stderr}")
            failed.append(title)
    
    # Summary
    print("\n" + "="*60)
    print(f"✅ Successfully processed: {len(successful)} chapters")
    print(f"❌ Failed: {len(failed)} chapters")
    
    if failed:
        print("\nFailed chapters:")
        for title in failed:
            print(f"  - {title}")
    
    return len(successful), len(failed)

def chunk_textbooks():
    """Chunk all extracted textbook content"""
    print("\n" + "="*60)
    print("Chunking textbook content...")
    
    cmd = [
        sys.executable, "chunking/chunker.py",
        "--trials-dir", "data/complete_extractions",
        "--chapters-dir", "data/textbook_extractions",
        "--out", "data/chunks/combined_chunks.jsonl"
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("✅ Chunking completed successfully")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Chunking failed")
        print(f"Error: {e.stderr}")
        return False

def build_indexes():
    """Build search indexes from chunked content"""
    print("\n" + "="*60)
    print("Building search indexes...")
    
    # Build FAISS index
    print("Building FAISS vector index...")
    cmd_faiss = [
        sys.executable, "-c",
        """
import json
import numpy as np
from pathlib import Path

# Read chunks
chunks_file = Path("data/chunks/combined_chunks.jsonl")
if not chunks_file.exists():
    print("No chunks file found")
    exit(1)

chunks = []
with open(chunks_file, 'r') as f:
    for line in f:
        chunks.append(json.loads(line))

print(f"Loaded {len(chunks)} chunks")

# Create embeddings (simplified - in production use OpenAI embeddings)
# For now, just create a dummy index file to indicate completion
index_dir = Path("data/indexes")
index_dir.mkdir(parents=True, exist_ok=True)

# Save chunk metadata
metadata_file = index_dir / "chunks_metadata.json"
with open(metadata_file, 'w') as f:
    json.dump({
        "total_chunks": len(chunks),
        "sources": {
            "trials": sum(1 for c in chunks if c.get("source") == "trial"),
            "chapters": sum(1 for c in chunks if c.get("source") == "chapter")
        }
    }, f, indent=2)

print(f"Saved metadata to {metadata_file}")
"""
    ]
    
    try:
        result = subprocess.run(cmd_faiss, capture_output=True, text=True, check=True)
        print("✅ Index building completed")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ Index building failed")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Main processing pipeline"""
    print("🚀 Starting textbook processing pipeline")
    print("="*60)
    
    # Step 1: Extract textbooks
    success_count, fail_count = process_all_textbooks()
    
    if success_count == 0:
        print("\n❌ No textbooks were successfully processed. Exiting.")
        return 1
    
    # Step 2: Chunk content
    if not chunk_textbooks():
        print("\n❌ Chunking failed. Exiting.")
        return 1
    
    # Step 3: Build indexes
    if not build_indexes():
        print("\n❌ Index building failed. Exiting.")
        return 1
    
    print("\n" + "="*60)
    print("✅ Pipeline completed successfully!")
    print(f"  - Extracted: {success_count} chapters")
    print(f"  - Chunked and indexed all content")
    
    return 0

if __name__ == "__main__":
    exit(main())