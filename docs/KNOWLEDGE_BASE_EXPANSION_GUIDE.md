# Knowledge Base Expansion Guide

## Quick Answer: How Does Adding Knowledge Work?

**Yes and No** - Simply creating chunk files isn't enough. You need to **index** them into your retrieval system. Here's the complete flow:

```
New Document → Extract → Chunk → Save Files → INDEX → Update Retrieval System → Available for Queries
                                      ↑                           ↑
                                   You stop here          This step makes it searchable!
```

---

## The Complete Knowledge Addition Pipeline

### Step 1: Extract and Chunk (What You Have Now)
```python
# This creates files but doesn't add to searchable knowledge
extractor = UnifiedExtractor()
chunker = HierarchicalChunker()

# Process document
extracted = extractor.extract(content, 'research', metadata)
chunks = chunker.chunk_document(document, extracted)

# Save to files
with open('data/chunks/new_doc_chunks.json', 'w') as f:
    json.dump(chunks, f)  # ← File created, but NOT YET searchable!
```

### Step 2: Index for Retrieval (The Missing Piece)
```python
from ipchat.core.indexing.index_builder import IndexBuilder
from ipchat.processing.embedder import ChunkEmbedder

# Initialize indexing components
embedder = ChunkEmbedder()
index_builder = IndexBuilder()

# Load your new chunks
with open('data/chunks/new_doc_chunks.json', 'r') as f:
    chunk_data = json.load(f)

# Generate embeddings
embedded_chunks = embedder.embed_chunks(chunk_data['chunks'])

# Add to FAISS index
index_builder.add_to_faiss_index(embedded_chunks)

# Add to BM25 index
index_builder.add_to_bm25_index(chunk_data['chunks'])

# Save updated indices
index_builder.save_indices('data/indices/')

print("✓ Knowledge base updated and searchable!")
```

---

## Automated Knowledge Base Update System

Here's a complete system that automatically updates your knowledge base:

### Create: `add_to_knowledge_base.py`
```python
#!/usr/bin/env python3
"""
Automated knowledge base updater.
Run this after adding new documents to make them searchable.
"""

import json
from pathlib import Path
import faiss
import pickle
from rank_bm25 import BM25Okapi
import numpy as np

from ipchat.extraction.unified_extractor import UnifiedExtractor
from ipchat.processing.chunker import HierarchicalChunker
from ipchat.processing.embedder import ChunkEmbedder

class KnowledgeBaseManager:
    """Manages the complete knowledge base pipeline"""
    
    def __init__(self, base_dir: Path = Path('data')):
        self.base_dir = base_dir
        self.chunks_dir = base_dir / 'chunks'
        self.indices_dir = base_dir / 'indices'
        self.extracted_dir = base_dir / 'extracted'
        
        # Create directories
        for dir in [self.chunks_dir, self.indices_dir, self.extracted_dir]:
            dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.extractor = UnifiedExtractor()
        self.chunker = HierarchicalChunker()
        self.embedder = ChunkEmbedder()
        
        # Load or create indices
        self.load_or_create_indices()
    
    def load_or_create_indices(self):
        """Load existing indices or create new ones"""
        
        # FAISS index
        faiss_path = self.indices_dir / 'faiss.index'
        if faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            print(f"✓ Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        else:
            # Create new index (dimension 1536 for text-embedding-3-small)
            self.faiss_index = faiss.IndexFlatL2(1536)
            print("✓ Created new FAISS index")
        
        # BM25 index
        bm25_path = self.indices_dir / 'bm25.pkl'
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                self.bm25_data = pickle.load(f)
                self.bm25_index = BM25Okapi(self.bm25_data['corpus'])
            print(f"✓ Loaded BM25 index with {len(self.bm25_data['corpus'])} documents")
        else:
            self.bm25_data = {'corpus': [], 'metadata': []}
            self.bm25_index = None
            print("✓ Created new BM25 index")
        
        # Metadata store
        metadata_path = self.indices_dir / 'metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {'chunks': [], 'documents': {}}
    
    def add_document(self, 
                     file_path: Path, 
                     doc_type: str = 'research',
                     force_reprocess: bool = False) -> dict:
        """
        Add a single document to the knowledge base.
        
        Args:
            file_path: Path to document JSON file
            doc_type: 'research' or 'textbook'
            force_reprocess: Reprocess even if already exists
            
        Returns:
            Statistics about the addition
        """
        
        doc_id = file_path.stem
        stats = {'doc_id': doc_id, 'status': 'processed'}
        
        # Check if already processed
        if not force_reprocess and doc_id in self.metadata['documents']:
            print(f"⚠️  {doc_id} already in knowledge base. Use force_reprocess=True to update.")
            stats['status'] = 'skipped'
            return stats
        
        print(f"\n📄 Processing: {file_path.name}")
        
        try:
            # 1. Load document
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract content (handle Adobe Extract format)
            if 'elements' in data:
                content = ' '.join([
                    elem.get('Text', '') 
                    for elem in data.get('elements', [])
                    if elem.get('Text')
                ])
            else:
                content = data.get('content', str(data))
            
            # 2. Extract structured data
            print("  → Extracting structured data...")
            extracted = self.extractor.extract(
                content=content[:50000],  # Limit for tokens
                document_type=doc_type,
                document_metadata={
                    'id': doc_id,
                    'title': data.get('title', file_path.name)
                }
            )
            
            # Save extraction
            extracted_file = self.extracted_dir / f"{doc_id}.json"
            with open(extracted_file, 'w') as f:
                json.dump(extracted.__dict__, f, indent=2, default=str)
            
            # 3. Create chunks
            print("  → Creating semantic chunks...")
            chunk_result = self.chunker.chunk_with_hierarchy(
                document={
                    'id': doc_id,
                    'content': content,
                    'title': extracted.title
                },
                extracted_data=extracted.__dict__
            )
            
            chunks = chunk_result['chunks']
            stats['num_chunks'] = len(chunks)
            
            # Save chunks
            chunks_file = self.chunks_dir / f"{doc_id}_chunks.json"
            with open(chunks_file, 'w') as f:
                json.dump({
                    'chunks': [c.__dict__ for c in chunks],
                    'hierarchy': chunk_result['hierarchy']
                }, f, indent=2, default=str)
            
            # 4. Generate embeddings
            print(f"  → Generating embeddings for {len(chunks)} chunks...")
            embedded_chunks = self.embedder.embed_chunks(chunks)
            
            # 5. Update FAISS index
            print("  → Updating FAISS index...")
            embeddings = np.array([ec.embedding for ec in embedded_chunks])
            start_idx = self.faiss_index.ntotal
            self.faiss_index.add(embeddings)
            
            # 6. Update BM25 index
            print("  → Updating BM25 index...")
            chunk_texts = [chunk.content for chunk in chunks]
            tokenized_chunks = [text.lower().split() for text in chunk_texts]
            
            # Rebuild BM25 with all documents
            self.bm25_data['corpus'].extend(tokenized_chunks)
            for i, chunk in enumerate(chunks):
                self.bm25_data['metadata'].append({
                    'chunk_id': chunk.chunk_id,
                    'doc_id': doc_id,
                    'chunk_index': i,
                    'faiss_index': start_idx + i
                })
            
            self.bm25_index = BM25Okapi(self.bm25_data['corpus'])
            
            # 7. Update metadata
            self.metadata['documents'][doc_id] = {
                'title': extracted.title,
                'type': doc_type,
                'num_chunks': len(chunks),
                'faiss_indices': list(range(start_idx, start_idx + len(chunks))),
                'extraction_path': str(extracted_file),
                'chunks_path': str(chunks_file)
            }
            
            self.metadata['chunks'].extend([{
                'chunk_id': chunk.chunk_id,
                'doc_id': doc_id,
                'faiss_index': start_idx + i,
                'bm25_index': len(self.bm25_data['corpus']) - len(chunks) + i
            } for i, chunk in enumerate(chunks)])
            
            # 8. Save all indices
            self.save_indices()
            
            print(f"  ✓ Added {len(chunks)} chunks to knowledge base")
            stats['status'] = 'success'
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            stats['status'] = 'failed'
            stats['error'] = str(e)
        
        return stats
    
    def add_directory(self, 
                      directory: Path, 
                      doc_type: str = 'research',
                      pattern: str = '*.json') -> dict:
        """Add all documents in a directory"""
        
        directory = Path(directory)
        files = list(directory.glob(pattern))
        
        print(f"\n🗂️  Processing {len(files)} files from {directory}")
        
        results = {
            'total': len(files),
            'success': 0,
            'skipped': 0,
            'failed': 0,
            'documents': []
        }
        
        for file_path in files:
            stats = self.add_document(file_path, doc_type)
            results['documents'].append(stats)
            
            if stats['status'] == 'success':
                results['success'] += 1
            elif stats['status'] == 'skipped':
                results['skipped'] += 1
            else:
                results['failed'] += 1
        
        print(f"\n📊 Summary:")
        print(f"  Total: {results['total']}")
        print(f"  ✓ Success: {results['success']}")
        print(f"  ⚠ Skipped: {results['skipped']}")
        print(f"  ✗ Failed: {results['failed']}")
        
        return results
    
    def save_indices(self):
        """Save all indices to disk"""
        
        # Save FAISS
        faiss_path = self.indices_dir / 'faiss.index'
        faiss.write_index(self.faiss_index, str(faiss_path))
        
        # Save BM25
        bm25_path = self.indices_dir / 'bm25.pkl'
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25_data, f)
        
        # Save metadata
        metadata_path = self.indices_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        
        print(f"  💾 Indices saved to {self.indices_dir}")
    
    def get_stats(self) -> dict:
        """Get knowledge base statistics"""
        
        return {
            'total_documents': len(self.metadata['documents']),
            'total_chunks': len(self.metadata['chunks']),
            'faiss_vectors': self.faiss_index.ntotal,
            'bm25_documents': len(self.bm25_data['corpus']),
            'research_articles': sum(1 for d in self.metadata['documents'].values() if d['type'] == 'research'),
            'textbook_chapters': sum(1 for d in self.metadata['documents'].values() if d['type'] == 'textbook'),
            'index_size_mb': sum(
                (self.indices_dir / f).stat().st_size / 1048576 
                for f in ['faiss.index', 'bm25.pkl', 'metadata.json']
                if (self.indices_dir / f).exists()
            )
        }
    
    def search(self, query: str, k: int = 10) -> list:
        """Search the knowledge base (for testing)"""
        
        # Generate query embedding
        query_embedding = self.embedder.embed_chunks([type('Chunk', (), {'content': query, 'chunk_id': 'query'})])[0]
        
        # Search FAISS
        distances, indices = self.faiss_index.search(
            np.array([query_embedding.embedding]), k
        )
        
        # Get results
        results = []
        for idx in indices[0]:
            if idx < len(self.metadata['chunks']):
                chunk_meta = self.metadata['chunks'][idx]
                doc_meta = self.metadata['documents'][chunk_meta['doc_id']]
                results.append({
                    'chunk_id': chunk_meta['chunk_id'],
                    'document': doc_meta['title'],
                    'type': doc_meta['type']
                })
        
        return results


def main():
    """CLI interface for knowledge base management"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage IPchat Knowledge Base')
    parser.add_argument('--add-document', type=str, help='Add a single document')
    parser.add_argument('--add-directory', type=str, help='Add all documents in directory')
    parser.add_argument('--doc-type', type=str, default='research', 
                       choices=['research', 'textbook'],
                       help='Document type')
    parser.add_argument('--stats', action='store_true', help='Show knowledge base statistics')
    parser.add_argument('--search', type=str, help='Test search with a query')
    parser.add_argument('--force', action='store_true', help='Force reprocess existing documents')
    
    args = parser.parse_args()
    
    # Initialize manager
    kb = KnowledgeBaseManager()
    
    if args.stats:
        stats = kb.get_stats()
        print("\n📊 Knowledge Base Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    elif args.add_document:
        kb.add_document(
            Path(args.add_document),
            doc_type=args.doc_type,
            force_reprocess=args.force
        )
    
    elif args.add_directory:
        kb.add_directory(
            Path(args.add_directory),
            doc_type=args.doc_type
        )
    
    elif args.search:
        results = kb.search(args.search)
        print(f"\n🔍 Search results for: '{args.search}'")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['document']} ({result['type']})")
    
    else:
        parser.print_help()
        print("\n📌 Examples:")
        print("  python add_to_knowledge_base.py --stats")
        print("  python add_to_knowledge_base.py --add-document data/input_articles/new_paper.json")
        print("  python add_to_knowledge_base.py --add-directory data/input_articles --doc-type research")
        print("  python add_to_knowledge_base.py --search 'EBUS-TBNA diagnostic yield'")


if __name__ == "__main__":
    main()
```

---

## Usage Examples

### Adding a Single New Document
```bash
# Extract, chunk, and index a new research paper
python add_to_knowledge_base.py \
    --add-document data/input_articles/new_paper.json \
    --doc-type research

# Output:
# 📄 Processing: new_paper.json
#   → Extracting structured data...
#   → Creating semantic chunks...
#   → Generating embeddings for 12 chunks...
#   → Updating FAISS index...
#   → Updating BM25 index...
#   ✓ Added 12 chunks to knowledge base
#   💾 Indices saved to data/indices
```

### Adding Multiple Documents
```bash
# Add all new articles in a directory
python add_to_knowledge_base.py \
    --add-directory data/new_articles \
    --doc-type research

# Add new textbook chapters
python add_to_knowledge_base.py \
    --add-directory "Textbooks/New Chapters" \
    --doc-type textbook
```

### Check Knowledge Base Status
```bash
python add_to_knowledge_base.py --stats

# Output:
# 📊 Knowledge Base Statistics:
#   total_documents: 325
#   total_chunks: 4,567
#   faiss_vectors: 4,567
#   bm25_documents: 4,567
#   research_articles: 285
#   textbook_chapters: 40
#   index_size_mb: 125.3
```

### Test Search
```bash
python add_to_knowledge_base.py --search "pneumothorax after BLVR"

# Output:
# 🔍 Search results for: 'pneumothorax after BLVR'
#   1. Endobronchial Valve Therapy Outcomes (research)
#   2. Managing BLVR Complications (textbook)
#   3. Pneumothorax Risk Factors Study (research)
```

---

## How It Actually Works

```
┌─────────────────┐
│  New Document   │
└────────┬────────┘
         ↓
┌─────────────────┐
│    Extract      │ → Creates: data/extracted/doc.json
└────────┬────────┘
         ↓
┌─────────────────┐
│     Chunk       │ → Creates: data/chunks/doc_chunks.json
└────────┬────────┘
         ↓
┌─────────────────┐
│     Embed       │ → Generates vector representations
└────────┬────────┘
         ↓
┌─────────────────┐
│  Update Index   │ → Updates: data/indices/faiss.index
│                 │           data/indices/bm25.pkl
│                 │           data/indices/metadata.json
└────────┬────────┘
         ↓
┌─────────────────┐
│   Searchable!   │ → Now available for retrieval
└─────────────────┘
```

---

## Integration with Your Chatbot

```python
# In your chatbot application
from knowledge_base_manager import KnowledgeBaseManager

class IPChatbot:
    def __init__(self):
        self.kb = KnowledgeBaseManager()
    
    def add_new_knowledge(self, file_path: Path, doc_type: str):
        """Add new document to knowledge base"""
        return self.kb.add_document(file_path, doc_type)
    
    def answer_question(self, question: str):
        """Answer using knowledge base"""
        # Search for relevant chunks
        relevant_chunks = self.kb.search(question, k=5)
        
        # Generate answer using retrieved chunks
        answer = self.generate_answer(question, relevant_chunks)
        
        return answer
```

---

## Key Takeaways

1. **Files alone aren't enough** - You need to index them
2. **The pipeline**: Extract → Chunk → Embed → Index → Search
3. **Use the manager script** - Handles all steps automatically
4. **Incremental updates** - Add new documents without reprocessing everything
5. **Mixed content types** - Research and textbooks in same knowledge base

The system is designed to grow incrementally - each new document adds to the collective knowledge without needing to reprocess everything!