#!/usr/bin/env python3
"""Test the search functionality"""

from pathlib import Path
import json
from indexing.search import FaissSearcher

def test_vector_search():
    # Initialize searcher
    searcher = FaissSearcher("data/index")
    
    # Test queries
    queries = [
        "What is the effectiveness of Zephyr EBV for emphysema?",
        "Pneumothorax rates after endobronchial valve placement",
        "FEV1 improvement at 12 months",
        "Bronchoscopic lung volume reduction outcomes"
    ]
    
    print("=" * 80)
    print("TESTING VECTOR SEARCH")
    print("=" * 80)
    
    for query in queries:
        print(f"\nQuery: {query}")
        print("-" * 40)
        
        # Search
        results = searcher.search(query, k=3)
        
        # Load chunk content
        chunks_by_id = {}
        with open("data/chunks/chunks.jsonl", "r") as f:
            for line in f:
                chunk = json.loads(line)
                chunks_by_id[chunk["chunk_id"]] = chunk
        
        # Display results
        for score, meta in results:
            chunk_id = meta["chunk_id"]
            doc_id = meta["document_id"]
            chunk = chunks_by_id.get(chunk_id, {})
            text_preview = chunk.get("text", "")[:200] + "..."
            
            print(f"Score: {score:.3f}")
            print(f"Document: {doc_id}")
            print(f"Chunk: {chunk_id}")
            print(f"Preview: {text_preview}")
            print()

if __name__ == "__main__":
    test_vector_search()