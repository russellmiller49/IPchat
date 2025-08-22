#!/usr/bin/env python3
"""
Test script to verify Bronchmonkey is working
"""

import json
from pathlib import Path
import sys

def test_knowledge_base():
    """Test that knowledge base is properly loaded"""
    print("Testing Bronchmonkey Knowledge Base...")
    print("="*50)
    
    # Test 1: Check indices exist
    indices_dir = Path("data/indices")
    required_files = [
        "combined_knowledge_base.json",
        "search_chunks.json",
        "quick_lookup.json",
        "migrated_articles_index.json"
    ]
    
    print("1. Checking indices...")
    for file in required_files:
        file_path = indices_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file} - MISSING")
            return False
    
    # Test 2: Load and verify chunks
    print("\n2. Loading search chunks...")
    chunks_path = indices_dir / "search_chunks.json"
    with open(chunks_path, 'r', encoding='utf-8') as f:
        chunks_data = json.load(f)
    
    print(f"   ✅ Loaded {chunks_data['total_chunks']} chunks")
    
    # Test 3: Check chunk structure
    print("\n3. Verifying chunk structure...")
    sample_chunk = chunks_data['chunks'][0]
    required_fields = ['chunk_id', 'source_file', 'source_type', 'title', 'content']
    
    for field in required_fields:
        if field in sample_chunk:
            # Check that title is a string
            if field == 'title':
                if isinstance(sample_chunk[field], str):
                    print(f"   ✅ {field}: string")
                else:
                    print(f"   ❌ {field}: not a string - {type(sample_chunk[field])}")
                    return False
            else:
                print(f"   ✅ {field}: present")
        else:
            print(f"   ❌ {field}: MISSING")
            return False
    
    # Test 4: Test search functionality
    print("\n4. Testing search...")
    from bronchmonkey_lite import simple_search, load_knowledge_base
    
    kb = load_knowledge_base()
    if not kb:
        print("   ❌ Failed to load knowledge base")
        return False
    
    # Test query
    test_query = "EBUS diagnostic yield"
    results = simple_search(test_query, kb, top_k=3)
    
    if results:
        print(f"   ✅ Found {len(results)} results for '{test_query}'")
        for i, result in enumerate(results[:2], 1):
            title = result['chunk'].get('title', 'No title')
            if isinstance(title, dict):
                title = title.get('value', 'No title')
            score = result['score']
            print(f"      {i}. {title[:50]}... (score: {score})")
    else:
        print(f"   ⚠️  No results found for '{test_query}'")
    
    # Test 5: Check quick lookup
    print("\n5. Testing quick lookup...")
    lookup_path = indices_dir / "quick_lookup.json"
    with open(lookup_path, 'r', encoding='utf-8') as f:
        lookup = json.load(f)
    
    if lookup.get('diagnostic_yields'):
        print(f"   ✅ Quick lookup has {len(lookup['diagnostic_yields'])} procedures with yields")
    if lookup.get('procedure_steps'):
        print(f"   ✅ Quick lookup has {len(lookup['procedure_steps'])} procedures with steps")
    
    print("\n" + "="*50)
    print("✅ All tests passed! Bronchmonkey is ready to use.")
    print("\nTo start the application, run:")
    print("  streamlit run bronchmonkey_lite.py")
    print("\nOr use the startup script:")
    print("  python start_bronchmonkey.py")
    
    return True

if __name__ == "__main__":
    success = test_knowledge_base()
    if not success:
        print("\n❌ Tests failed. Please fix the issues above.")
        sys.exit(1)