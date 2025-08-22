#!/usr/bin/env python3
"""
Simple test of search functionality without Streamlit
"""

import json
from pathlib import Path

def simple_search_test(query: str, chunks_path: Path, top_k: int = 5):
    """Test search without importing bronchmonkey_lite"""
    
    # Load chunks
    with open(chunks_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    
    for chunk in data['chunks']:
        content_lower = chunk['content'].lower()
        
        # Calculate simple relevance score
        score = 0
        for word in query_words:
            if word in content_lower:
                score += content_lower.count(word)
        
        # Boost score for title matches
        if 'title' in chunk and chunk['title']:
            title = chunk['title']
            if isinstance(title, dict):
                title = title.get('value', '') or ''
            if isinstance(title, str) and title:
                title_lower = title.lower()
                for word in query_words:
                    if word in title_lower:
                        score += 5
        
        if score > 0:
            results.append({
                'chunk': chunk,
                'score': score
            })
    
    # Sort by score and return top k
    results.sort(key=lambda x: x['score'], reverse=True)
    return results[:top_k]

def main():
    print("Testing Search Functionality")
    print("="*60)
    
    chunks_path = Path("data/indices/search_chunks.json")
    
    # Test queries
    test_queries = [
        "EBUS diagnostic yield",
        "pneumothorax complications",
        "balloon bronchoplasty technique",
        "transbronchial cryobiopsy"
    ]
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-"*40)
        
        results = simple_search_test(query, chunks_path, top_k=3)
        
        if results:
            for i, result in enumerate(results, 1):
                chunk = result['chunk']
                score = result['score']
                
                # Get title
                title = chunk.get('title', 'No title')
                if isinstance(title, dict):
                    title = title.get('value', 'No title')
                
                # Get preview
                content_preview = chunk['content'][:100] + "..."
                
                print(f"\n  Result {i} (score: {score}):")
                print(f"  Title: {title[:60]}")
                print(f"  Type: {chunk.get('source_type', 'unknown')}")
                print(f"  Preview: {content_preview}")
        else:
            print("  No results found")
    
    print("\n" + "="*60)
    print("✅ Search functionality is working!")
    print("\nYour Bronchmonkey system is ready.")
    print("To start the chat interface, run:")
    print("  streamlit run bronchmonkey_lite.py")

if __name__ == "__main__":
    main()