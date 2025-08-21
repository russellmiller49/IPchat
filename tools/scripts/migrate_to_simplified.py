#!/usr/bin/env python3
"""
Main migration script to convert existing data to simplified format.
Run this to migrate your existing extractions to the new format.
"""

import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from ipchat.extraction.unified_extractor import UnifiedExtractor
from ipchat.processing.chunker import HierarchicalChunker
from ipchat.evaluation.benchmarks import IPBenchmark
import argparse

def migrate_existing_extractions(input_dir: Path, output_dir: Path):
    """Migrate existing complex extractions to simplified format"""
    
    print("🔄 Starting migration of existing extractions...")
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each JSON file in input directory
    json_files = list(input_dir.glob("*.json"))
    
    migrated_count = 0
    
    for json_file in json_files:
        try:
            print(f"Processing: {json_file.name}")
            
            with open(json_file, 'r') as f:
                old_data = json.load(f)
            
            # Convert to simplified format
            simplified = {
                'document_id': old_data.get('id', json_file.stem),
                'title': old_data.get('title', 'Unknown'),
                'document_type': 'research' if 'study_type' in old_data else 'textbook',
                'summary': old_data.get('summary', old_data.get('abstract', '')),
            }
            
            # Extract relevant fields based on document type
            if simplified['document_type'] == 'research':
                simplified.update({
                    'population': old_data.get('population', {}).get('description') if isinstance(old_data.get('population'), dict) else old_data.get('population'),
                    'intervention': old_data.get('intervention', {}).get('name') if isinstance(old_data.get('intervention'), dict) else old_data.get('intervention'),
                    'outcomes': old_data.get('outcomes', {}),
                    'key_findings': old_data.get('key_findings', [])[:5] if old_data.get('key_findings') else []
                })
            else:
                # Textbook format
                simplified.update({
                    'procedures': old_data.get('procedures', []),
                    'indications': old_data.get('indications', []),
                    'contraindications': old_data.get('contraindications', [])
                })
            
            # Save simplified version
            output_file = output_dir / f"{simplified['document_id']}_simplified.json"
            with open(output_file, 'w') as f:
                json.dump(simplified, f, indent=2)
            
            migrated_count += 1
            
        except Exception as e:
            print(f"  ⚠️ Failed to migrate {json_file.name}: {e}")
    
    print(f"✅ Migrated {migrated_count}/{len(json_files)} files to simplified format")

def process_new_documents(input_dir: Path, output_dir: Path, doc_type: str = 'research'):
    """Process new documents with simplified pipeline"""
    
    print(f"📄 Processing new {doc_type} documents...")
    
    extractor = UnifiedExtractor()
    chunker = HierarchicalChunker()
    
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    
    # Create output directories
    extracted_dir = output_dir / 'extracted'
    chunks_dir = output_dir / 'chunks'
    extracted_dir.mkdir(parents=True, exist_ok=True)
    chunks_dir.mkdir(parents=True, exist_ok=True)
    
    # Process JSON files with Adobe Extract content
    json_files = list(input_dir.glob("*.json"))[:5]  # Start with 5 files
    
    processed_count = 0
    
    for json_file in json_files:
        try:
            print(f"Processing: {json_file.name}")
            
            # Load JSON content
            with open(json_file, 'r') as f:
                doc_data = json.load(f)
            
            # Extract text content (handle Adobe Extract format)
            if 'elements' in doc_data:
                # Adobe Extract format
                content = ' '.join([
                    elem.get('Text', '') 
                    for elem in doc_data.get('elements', []) 
                    if elem.get('Text')
                ])
            else:
                # Simple text format
                content = doc_data.get('content', doc_data.get('text', str(doc_data)))
            
            if not content:
                print(f"  ⚠️ No content found in {json_file.name}")
                continue
            
            # Extract with unified extractor
            extracted = extractor.extract(
                content=content[:50000],  # Limit content length
                document_type=doc_type,
                document_metadata={'id': json_file.stem, 'title': doc_data.get('title', json_file.name)}
            )
            
            # Save extraction
            extracted_file = extracted_dir / f"{json_file.stem}.json"
            with open(extracted_file, 'w') as f:
                json.dump(extracted.__dict__, f, indent=2, default=str)
            
            # Create chunks
            chunk_result = chunker.chunk_with_hierarchy(
                document={'id': json_file.stem, 'content': content, 'title': json_file.name},
                extracted_data=extracted.__dict__
            )
            
            # Save chunks
            chunks_file = chunks_dir / f"{json_file.stem}_chunks.json"
            chunks_data = {
                'chunks': [
                    {
                        'chunk_id': chunk.chunk_id,
                        'document_id': chunk.document_id,
                        'content': chunk.content,
                        'metadata': chunk.metadata,
                        'token_count': chunk.token_count,
                        'chunk_index': chunk.chunk_index,
                        'total_chunks': chunk.total_chunks
                    } 
                    for chunk in chunk_result['chunks']
                ],
                'hierarchy': chunk_result['hierarchy']
            }
            with open(chunks_file, 'w') as f:
                json.dump(chunks_data, f, indent=2, default=str)
            
            processed_count += 1
            
        except Exception as e:
            print(f"  ⚠️ Failed to process {json_file.name}: {e}")
    
    print(f"✅ Processed {processed_count}/{len(json_files)} new documents")

def create_benchmark_dataset(output_dir: Path):
    """Create initial benchmark dataset"""
    
    print("📊 Creating benchmark dataset...")
    
    benchmark = IPBenchmark()
    benchmark_file = Path(output_dir) / 'benchmarks' / 'ip_benchmark_v1.json'
    benchmark_file.parent.mkdir(parents=True, exist_ok=True)
    benchmark.save_benchmark(benchmark_file)
    
    print(f"✅ Created benchmark with {len(benchmark.questions)} questions")

def main():
    parser = argparse.ArgumentParser(description='Migrate to simplified IPchat pipeline')
    parser.add_argument('--migrate-existing', action='store_true', 
                       help='Migrate existing extractions')
    parser.add_argument('--process-new', action='store_true',
                       help='Process new documents')
    parser.add_argument('--create-benchmark', action='store_true',
                       help='Create benchmark dataset')
    parser.add_argument('--input-dir', type=str, default='data/input_articles',
                       help='Input directory')
    parser.add_argument('--output-dir', type=str, default='data/simplified',
                       help='Output directory')
    parser.add_argument('--doc-type', type=str, default='research',
                       choices=['research', 'textbook'],
                       help='Document type to process')
    
    args = parser.parse_args()
    
    if args.migrate_existing:
        # Try multiple possible input directories
        possible_dirs = [
            Path('data/gold_standard_extractions'),
            Path('data/extracted'),
            Path('data/oe_final_outputs'),
            Path(args.input_dir)
        ]
        
        for input_dir in possible_dirs:
            if input_dir.exists() and any(input_dir.glob('*.json')):
                print(f"Found extractions in: {input_dir}")
                migrate_existing_extractions(input_dir, Path(args.output_dir))
                break
        else:
            print("❌ No existing extractions found to migrate")
    
    if args.process_new:
        process_new_documents(
            Path(args.input_dir),
            Path(args.output_dir),
            args.doc_type
        )
    
    if args.create_benchmark:
        create_benchmark_dataset(Path(args.output_dir))
    
    if not any([args.migrate_existing, args.process_new, args.create_benchmark]):
        print("❌ No action specified. Use --help for options")
        print("\nQuick start:")
        print("  python tools/scripts/migrate_to_simplified.py --migrate-existing")
        print("  python tools/scripts/migrate_to_simplified.py --process-new")
        print("  python tools/scripts/migrate_to_simplified.py --create-benchmark")

if __name__ == "__main__":
    main()