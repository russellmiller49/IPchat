#!/usr/bin/env python3
"""
Build BM25 index for keyword-based search.
Uses rank_bm25 for efficient sparse retrieval.
"""

import json
import pickle
from pathlib import Path
from typing import List, Dict, Set
import re
from rank_bm25 import BM25Okapi

# Common English stopwords (avoiding NLTK dependency issues)
STOPWORDS = set([
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
    'was', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
    'shall', 'can', 'need', 'dare', 'ought', 'used', 'have', 'had', 'having',
    'do', 'does', 'did', 'doing', 'done', 'be', 'am', 'is', 'are', 'was',
    'were', 'being', 'been', 'get', 'gets', 'got', 'getting', 'gotten',
    'become', 'becomes', 'became', 'becoming', 'seem', 'seems', 'seemed',
    'seeming', 'remain', 'remains', 'remained', 'remaining', 'keep', 'keeps',
    'kept', 'keeping', 'stay', 'stays', 'stayed', 'staying'
])

# Important medical/research terms to keep even if they might be stopwords
MEDICAL_KEEP = {
    'no', 'not', 'with', 'without', 'after', 'before', 'during', 'between',
    'versus', 'vs', 'compared', 'than', 'more', 'less', 'greater', 'fewer',
    'increase', 'decrease', 'improve', 'worsen', 'significant', 'significantly'
}

def preprocess_text(text: str) -> List[str]:
    """Tokenize and clean text for BM25."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep medical terms intact
    text = re.sub(r'[^\w\s\-\%\.]', ' ', text)
    
    # Simple word tokenization
    tokens = text.split()
    
    # Remove stopwords but keep medical terms
    filtered_tokens = []
    for token in tokens:
        # Keep if: medical term, contains numbers, or not a stopword
        if (token in MEDICAL_KEEP or 
            any(c.isdigit() for c in token) or 
            token not in STOPWORDS):
            if len(token) > 1:  # Skip single characters
                filtered_tokens.append(token)
    
    return filtered_tokens

def build_bm25_index(chunks_file: Path, output_dir: Path):
    """Build BM25 index from chunks."""
    
    print("Loading chunks...")
    chunks = []
    texts = []
    
    with open(chunks_file, 'r') as f:
        for line in f:
            chunk = json.loads(line)
            chunks.append(chunk)
            texts.append(chunk.get('text', ''))
    
    print(f"Loaded {len(chunks)} chunks")
    
    print("Preprocessing texts...")
    tokenized_texts = [preprocess_text(text) for text in texts]
    
    print("Building BM25 index...")
    bm25 = BM25Okapi(tokenized_texts)
    
    # Save the index
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save BM25 model
    with open(output_dir / 'bm25.pkl', 'wb') as f:
        pickle.dump(bm25, f)
    
    # Save chunk metadata for BM25 (same as FAISS meta)
    with open(output_dir / 'bm25_meta.jsonl', 'w') as f:
        for chunk in chunks:
            meta = {
                'chunk_id': chunk['chunk_id'],
                'document_id': chunk['document_id']
            }
            f.write(json.dumps(meta) + '\n')
    
    # Save vocabulary for debugging
    vocab = set()
    for tokens in tokenized_texts:
        vocab.update(tokens)
    
    with open(output_dir / 'vocab.txt', 'w') as f:
        for word in sorted(vocab):
            f.write(word + '\n')
    
    print(f"BM25 index saved to {output_dir}")
    print(f"Vocabulary size: {len(vocab)}")
    
    return bm25

if __name__ == "__main__":
    chunks_file = Path("data/chunks/chunks.jsonl")
    output_dir = Path("data/index")
    
    build_bm25_index(chunks_file, output_dir)