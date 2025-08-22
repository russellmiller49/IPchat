#!/usr/bin/env python3
"""
Test MLA citation formatting
"""

import json
from pathlib import Path

def test_citations():
    """Test the citation extraction and formatting"""
    print("Testing MLA Citation Formatting")
    print("="*60)
    
    # Sample citation data for testing
    test_cases = [
        {
            'authors': ['Gerard J. Criner', 'Richard Sue', 'Shawn Wright'],
            'title': 'A Multicenter Randomized Controlled Trial of Zephyr Endobronchial Valve Treatment',
            'year': 2018,
            'journal': 'American Journal of Respiratory and Critical Care Medicine',
            'type': 'article'
        },
        {
            'authors': ['John Smith'],
            'title': 'Single Author Study on EBUS',
            'year': 2023,
            'journal': 'Chest',
            'type': 'article'
        },
        {
            'authors': [],
            'title': 'Balloon Dilation Techniques',
            'year': 2025,
            'journal': 'Principles and Practice of Interventional Pulmonology',
            'type': 'chapter'
        }
    ]
    
    # Test MLA formatting (simplified version)
    for i, case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"  Type: {case['type']}")
        
        if case['type'] == 'chapter':
            citation = f'"{case["title"]}." *{case["journal"]}*, Springer, {case["year"]}.'
        else:
            # Format authors
            authors = case['authors']
            if not authors:
                author_str = "Unknown Author"
            elif len(authors) == 1:
                name_parts = authors[0].split()
                if len(name_parts) >= 2:
                    author_str = f"{name_parts[-1]}, {' '.join(name_parts[:-1])}"
                else:
                    author_str = authors[0]
            elif len(authors) == 2:
                first_parts = authors[0].split()
                if len(first_parts) >= 2:
                    first_author = f"{first_parts[-1]}, {' '.join(first_parts[:-1])}"
                else:
                    first_author = authors[0]
                author_str = f"{first_author}, and {authors[1]}"
            else:
                first_parts = authors[0].split()
                if len(first_parts) >= 2:
                    first_author = f"{first_parts[-1]}, {' '.join(first_parts[:-1])}"
                else:
                    first_author = authors[0]
                author_str = f"{first_author}, et al."
            
            citation = f'{author_str}. "{case["title"]}." *{case["journal"]}*, {case["year"]}.'
        
        print(f"  MLA Citation:")
        print(f"  {citation}")
    
    # Test with actual knowledge base
    print("\n" + "="*60)
    print("Testing with Actual Files:")
    
    articles_index = Path("data/indices/migrated_articles_index.json")
    if articles_index.exists():
        with open(articles_index, 'r') as f:
            data = json.load(f)
        
        # Show first 3 articles with citations
        for article in data['articles'][:3]:
            authors = article.get('authors', [])
            title = article.get('title', 'Unknown Title')
            year = article.get('year', 'n.d.')
            
            # Format author string
            if authors:
                if len(authors) >= 3:
                    first_parts = authors[0].split()
                    if len(first_parts) >= 2:
                        author_str = f"{first_parts[-1]}, {' '.join(first_parts[:-1])}, et al."
                    else:
                        author_str = f"{authors[0]}, et al."
                else:
                    author_str = authors[0]
            else:
                author_str = "Unknown Author"
            
            # Truncate long titles
            if len(title) > 60:
                title = title[:60] + "..."
            
            print(f"\n{author_str}. \"{title}.\" *Journal*, {year}.")
    
    print("\n" + "="*60)
    print("✅ Citation formatting test complete!")

if __name__ == "__main__":
    test_citations()