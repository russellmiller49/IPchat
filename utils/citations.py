#!/usr/bin/env python3
"""
Citation formatting utilities for medical evidence.
Extracts author/year info and formats MLA citations.
"""

import json
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
import psycopg2
from psycopg2.extras import RealDictCursor
import os


def extract_author_year(document_id: str) -> Tuple[str, str]:
    """
    Extract first author's last name and year from document.
    Returns (author_last_name, year) or (document_id, "")
    """
    # Clean up document ID (remove .oe_final suffix if present)
    clean_id = document_id.replace('.oe_final', '')
    
    # Try to load from JSON file
    json_path = Path(f"data/oe_final_outputs/{document_id}.json")
    if not json_path.exists():
        json_path = Path(f"data/oe_final_outputs/{clean_id}.oe_final.json")
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract metadata
            meta = data.get('document', {}).get('metadata', {}) or data.get('metadata', {})
            
            # Get first author's last name
            authors = meta.get('authors', [])
            if authors and authors[0]:
                # Extract last name from first author
                first_author = authors[0]
                # Handle different name formats
                if ',' in first_author:
                    # Format: "Last, First"
                    author_last = first_author.split(',')[0].strip()
                else:
                    # Format: "First Last" or "First M. Last"
                    parts = first_author.strip().split()
                    author_last = parts[-1] if parts else ""
            else:
                author_last = ""
            
            # Get year
            year = str(meta.get('year', ''))
            
            if author_last and year:
                return (author_last, year)
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
    
    # Fallback: try to extract from filename
    # Some files have format like "Author-Year-Title"
    if '-' in clean_id:
        parts = clean_id.split('-')
        if len(parts) >= 2:
            potential_author = parts[0]
            potential_year = parts[1]
            if potential_year.isdigit() and len(potential_year) == 4:
                return (potential_author, potential_year)
    
    # Last resort: return document ID
    return (clean_id[:20], "")


def get_study_metadata(document_id: str) -> Dict:
    """
    Get full study metadata from database or JSON file.
    """
    metadata = {
        'title': '',
        'authors': [],
        'year': '',
        'journal': '',
        'doi': '',
        'citation_key': document_id
    }
    
    # Try database first
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        try:
            db_parts = db_url.replace("postgresql+psycopg2://", "postgresql://")
            conn = psycopg2.connect(db_parts, cursor_factory=RealDictCursor)
            
            with conn.cursor() as cur:
                # Clean document ID for database query
                clean_id = document_id.replace('.oe_final', '')
                
                cur.execute("""
                    SELECT title, year, journal, doi
                    FROM studies
                    WHERE study_id = %s OR study_id LIKE %s
                    LIMIT 1
                """, (document_id, f"%{clean_id}%"))
                
                row = cur.fetchone()
                if row:
                    metadata.update({
                        'title': row.get('title', ''),
                        'year': str(row.get('year', '')) if row.get('year') else '',
                        'journal': row.get('journal', ''),
                        'doi': row.get('doi', '')
                    })
            
            conn.close()
        except Exception as e:
            print(f"Database error: {e}")
    
    # Try JSON file for author information
    json_path = Path(f"data/oe_final_outputs/{document_id}.json")
    if not json_path.exists():
        clean_id = document_id.replace('.oe_final', '')
        json_path = Path(f"data/oe_final_outputs/{clean_id}.oe_final.json")
    
    if json_path.exists():
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            meta = data.get('document', {}).get('metadata', {}) or data.get('metadata', {})
            
            if not metadata['title']:
                metadata['title'] = meta.get('title', '')
            if not metadata['year']:
                metadata['year'] = str(meta.get('year', '')) if meta.get('year') else ''
            if not metadata['journal']:
                metadata['journal'] = meta.get('journal', '')
            if not metadata['doi']:
                metadata['doi'] = meta.get('doi', '')
            
            metadata['authors'] = meta.get('authors', [])
            
        except Exception as e:
            print(f"Error loading {json_path}: {e}")
    
    # Create citation key (Author Year)
    author_last, year = extract_author_year(document_id)
    if author_last and year:
        metadata['citation_key'] = f"{author_last} {year}"
    
    return metadata


def format_mla_citation(metadata: Dict) -> str:
    """
    Format a citation in MLA style.
    
    MLA format for journal articles:
    Author(s). "Title of Article." Title of Journal, vol. #, no. #, Year, pp. #-#. DOI.
    """
    citation_parts = []
    
    # Authors
    authors = metadata.get('authors', [])
    if authors:
        if len(authors) == 1:
            citation_parts.append(authors[0])
        elif len(authors) == 2:
            citation_parts.append(f"{authors[0]}, and {authors[1]}")
        elif len(authors) >= 3:
            citation_parts.append(f"{authors[0]}, et al")
    
    # Title
    title = metadata.get('title', '')
    if title:
        # Remove trailing period if present
        title = title.rstrip('.')
        citation_parts.append(f'"{title}."')
    
    # Journal
    journal = metadata.get('journal', '')
    if journal:
        citation_parts.append(f"*{journal}*")
    
    # Year
    year = metadata.get('year', '')
    if year:
        citation_parts.append(str(year))
    
    # DOI
    doi = metadata.get('doi', '')
    if doi:
        if not doi.startswith('http'):
            doi = f"https://doi.org/{doi}"
        citation_parts.append(doi)
    
    # Join parts
    if citation_parts:
        # First part (authors) ends with period
        result = citation_parts[0] + "."
        
        # Remaining parts joined with comma and space
        if len(citation_parts) > 1:
            result += " " + ", ".join(citation_parts[1:])
        
        # Ensure it ends with period
        if not result.endswith('.'):
            result += "."
        
        return result
    
    return metadata.get('citation_key', 'Unknown source')


def format_inline_citation(document_id: str) -> str:
    """
    Format an inline citation as (Author Year).
    """
    author_last, year = extract_author_year(document_id)
    if author_last and year:
        return f"({author_last} {year})"
    return f"({document_id[:20]})"


if __name__ == "__main__":
    # Test with some examples
    test_ids = [
        "A Multicenter RCT of Zephyr Endobronchial Valv.oe_final",
        "Valipour-2014-Expert statement_ pneumothorax a.oe_final",
        "BLVR with endobronchial valves for patients wi.oe_final"
    ]
    
    for doc_id in test_ids:
        print(f"\nDocument: {doc_id}")
        
        # Get inline citation
        inline = format_inline_citation(doc_id)
        print(f"Inline: {inline}")
        
        # Get full metadata and MLA citation
        metadata = get_study_metadata(doc_id)
        print(f"Citation key: {metadata['citation_key']}")
        
        mla = format_mla_citation(metadata)
        print(f"MLA: {mla}")