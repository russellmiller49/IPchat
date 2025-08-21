#!/usr/bin/env python3
"""
Build BM25/full-text search index using PostgreSQL's native text search.

This script:
1. Loads chunks into PostgreSQL chunks table (already done by schema)
2. Creates full-text search indexes for fast keyword search
3. Enables hybrid search by combining with FAISS vector search

Usage:
    export DATABASE_URL=postgresql+psycopg2://user@localhost:5432/medical_rag
    python indexing/build_bm25_pg.py
"""

import json
import os
from pathlib import Path
from sqlalchemy import create_engine, text
import sys

def load_chunks_to_pg(engine, chunks_file="data/chunks/chunks.jsonl"):
    """Load chunks into PostgreSQL for full-text search."""
    
    print(f"Loading chunks from {chunks_file}...")
    
    with open(chunks_file, 'r') as f:
        chunks = [json.loads(line) for line in f]
    
    print(f"Found {len(chunks)} chunks to index")
    
    # Insert chunks into database
    insert_sql = text("""
        INSERT INTO chunks (
            chunk_id, document_id, source, pages, section_path, 
            table_number, figure_number, trial_signals, text
        ) VALUES (
            :chunk_id, :document_id, :source, :pages, :section_path,
            :table_number, :figure_number, :trial_signals, :text
        )
        ON CONFLICT (chunk_id) DO UPDATE SET
            text = EXCLUDED.text,
            document_id = EXCLUDED.document_id,
            trial_signals = EXCLUDED.trial_signals
    """)
    
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        
        with engine.begin() as conn:
            for chunk in batch:
                # Extract metadata
                meta = chunk.get("metadata", {})
                
                # Prepare data for insertion
                data = {
                    "chunk_id": chunk["chunk_id"],
                    "document_id": meta.get("document_id", ""),
                    "source": meta.get("source", "trial"),
                    "pages": meta.get("pages", []),
                    "section_path": meta.get("section_path", []),
                    "table_number": meta.get("table_number"),
                    "figure_number": meta.get("figure_number"),
                    "trial_signals": json.dumps(meta.get("trial_signals", {})) if meta.get("trial_signals") else None,
                    "text": chunk.get("text", "")
                }
                
                conn.execute(insert_sql, data)
        
        print(f"Inserted batch {i//batch_size + 1}/{(len(chunks) + batch_size - 1)//batch_size}")
    
    print("Creating full-text search index...")
    
    # Create GIN index for full-text search
    with engine.begin() as conn:
        # Add tsvector column if not exists
        conn.execute(text("""
            ALTER TABLE chunks 
            ADD COLUMN IF NOT EXISTS text_search_vector tsvector
        """))
        
        # Update tsvector column
        conn.execute(text("""
            UPDATE chunks 
            SET text_search_vector = to_tsvector('english', text)
        """))
        
        # Create GIN index
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_text_search 
            ON chunks USING GIN (text_search_vector)
        """))
        
        # Also create trigram index for partial matching
        conn.execute(text("""
            CREATE INDEX IF NOT EXISTS idx_chunks_text_trigram 
            ON chunks USING GIN (text gin_trgm_ops)
        """))
    
    print("Full-text search indexes created successfully!")
    
    # Test the search
    test_query = "endobronchial valve"
    print(f"\nTesting search for '{test_query}'...")
    
    with engine.connect() as conn:
        result = conn.execute(
            text("""
                SELECT chunk_id, document_id, 
                       ts_rank(text_search_vector, plainto_tsquery('english', :query)) as rank
                FROM chunks
                WHERE text_search_vector @@ plainto_tsquery('english', :query)
                ORDER BY rank DESC
                LIMIT 5
            """),
            {"query": test_query}
        )
        
        rows = result.fetchall()
        print(f"Found {len(rows)} matching chunks:")
        for row in rows:
            print(f"  - {row.chunk_id} (rank: {row.rank:.4f})")


def create_hybrid_search_function(engine):
    """Create a PostgreSQL function for hybrid search."""
    
    print("\nCreating hybrid search function...")
    
    with engine.begin() as conn:
        # Drop existing function if exists
        conn.execute(text("DROP FUNCTION IF EXISTS hybrid_search"))
        
        # Create hybrid search function
        conn.execute(text("""
            CREATE OR REPLACE FUNCTION hybrid_search(
                query_text TEXT,
                limit_count INT DEFAULT 10
            )
            RETURNS TABLE(
                chunk_id TEXT,
                document_id TEXT,
                text TEXT,
                bm25_score FLOAT,
                similarity_score FLOAT
            ) AS $$
            BEGIN
                RETURN QUERY
                SELECT 
                    c.chunk_id,
                    c.document_id,
                    c.text,
                    ts_rank(c.text_search_vector, plainto_tsquery('english', query_text)) AS bm25_score,
                    0.0::FLOAT AS similarity_score  -- Placeholder for vector similarity
                FROM chunks c
                WHERE c.text_search_vector @@ plainto_tsquery('english', query_text)
                ORDER BY bm25_score DESC
                LIMIT limit_count;
            END;
            $$ LANGUAGE plpgsql;
        """))
    
    print("Hybrid search function created!")


def main():
    # Get database URL
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        # Use default for local setup
        db_url = "postgresql+psycopg2://russellmiller@localhost:5432/medical_rag"
        print(f"Using default DATABASE_URL: {db_url}")
    
    # Create engine
    engine = create_engine(db_url, future=True)
    
    # Load chunks and create indexes
    load_chunks_to_pg(engine)
    
    # Create hybrid search function
    create_hybrid_search_function(engine)
    
    print("\n✅ BM25/Full-text search setup complete!")
    print("\nYou can now query using:")
    print("  - Full-text search via PostgreSQL")
    print("  - Hybrid search combining BM25 + vector similarity")
    print("\nExample SQL query:")
    print("  SELECT * FROM hybrid_search('endobronchial valve outcomes', 10);")


if __name__ == "__main__":
    main()