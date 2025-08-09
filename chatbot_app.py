#!/usr/bin/env python3
"""
Bronchmonkey - Interventional Pulmonology Research Assistant
Powered by hybrid search (FAISS + BM25 + PostgreSQL)

Run with:
    streamlit run chatbot_app.py
"""

# Standard libs
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional

# Third-party
import streamlit as st
import requests
from openai import OpenAI
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor

# Add citation utilities
import sys
sys.path.append(str(Path(__file__).parent))
from utils.citations import extract_author_year, get_study_metadata, format_mla_citation, format_inline_citation

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = "http://localhost:8000"
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://russellmiller@localhost:5432/medical_rag")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Utility helpers for citation formatting
# ---------------------------------------------------------------------------

COMPLETE_EXTRACTIONS_DIR = Path("data/complete_extractions")

# ---------------------------------------------------------------------------
# Structured-data helpers (PostgreSQL safety table)
# ---------------------------------------------------------------------------


def fetch_safety_rows(keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return safety rows whose PT matches the keyword (ILIKE)."""
    try:
        db_parts = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(db_parts, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT study_id, pt, arm_id, patients, events, percentage
                FROM safety
                WHERE pt ILIKE %s
                ORDER BY percentage DESC NULLS LAST
                LIMIT %s
                """,
                (f"%{keyword}%", limit),
            )
            rows = cur.fetchall()
        conn.close()
        return rows
    except Exception:
        return []


def load_metadata(document_id: str) -> Optional[Dict[str, Any]]:
    """Return metadata dict for a given `document_id` (file stem)."""
    # Handle various possible filename suffixes (e.g. _complete.json, _completev2.json)
    candidates = list(COMPLETE_EXTRACTIONS_DIR.glob(f"{document_id}*.json"))
    if not candidates:
        return None
    try:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        return data.get("metadata") or {}
    except Exception:
        return None


def citation_key(document_id: str) -> str:
    """Generate in-text citation key like 'Valipour 2014'."""
    # Use the improved citation utility
    author_last, year = extract_author_year(document_id)
    if author_last and year:
        return f"{author_last} {year}"
    
    # Fallback to old method if needed
    meta = load_metadata(document_id)
    if not meta:
        return document_id  # fallback
    authors = meta.get("authors") or []
    year = meta.get("year") or "n.d."
    if authors:
        first_author_last = authors[0].split(",")[0] if "," in authors[0] else authors[0].split()[-1]
        return f"{first_author_last} {year}"
    return f"Unknown {year}"


def citation_mla(document_id: str) -> str:
    """Return a simple MLA-style citation string for the document."""
    # Use the improved MLA formatter
    metadata = get_study_metadata(document_id)
    return format_mla_citation(metadata)

# Page config
st.set_page_config(
    page_title="Bronchmonkey",
    page_icon="🐵",
    layout="wide"
)

# Title with monkey image
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("""<div style="text-align: center;">
    <span style="font-size: 80px;">🐵</span>
    </div>""", unsafe_allow_html=True)
with col2:
    st.title("Bronchmonkey")
    st.caption("Your Interventional Pulmonology Research Assistant")
    st.caption("Powered by hybrid search and advanced language models")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add welcome message
    st.session_state.messages.append({
        "role": "assistant",
        "content": "🐵 Welcome! I'm Bronchmonkey, your interventional pulmonology research assistant. I can help you find evidence from clinical trials, systematic reviews, and medical literature. Ask me about central airway obstruction, BLVR outcomes, rigid bronchoscopy techniques and outcomes or any other interventional pulmonology topic!"
    })
if "search_results" not in st.session_state:
    st.session_state.search_results = []

def search_evidence(query: str, k: int = 10) -> List[Dict]:
    """Search for relevant evidence using the API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/search",
            json={"query": query, "k": k}
        )
        if response.status_code == 200:
            return response.json()["hits"]
        else:
            st.error(f"Search API error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Search error: {e}")
        return []

def query_database(query: str) -> List[Dict]:
    """Query PostgreSQL for structured data."""
    # This function is deprecated - we use the API's hybrid search instead
    # Returning empty list to avoid errors
    return []

def get_chunk_text(chunk_id: str) -> str:
    """Retrieve full text for a chunk."""
    try:
        # Load from chunks file
        chunks_file = Path("data/chunks/chunks.jsonl")
        if chunks_file.exists():
            with open(chunks_file, 'r') as f:
                for line in f:
                    chunk = json.loads(line)
                    if chunk["chunk_id"] == chunk_id:
                        return chunk["text"]
        return ""
    except Exception as e:
        return f"Error loading chunk: {e}"

def generate_answer(query: str, context: List[Dict]) -> str:
    """Generate answer using GPT-5 with retrieved context."""
    
    # Prepare context from search results
    context_texts: List[str] = []
    citation_keys: Dict[str, str] = {}

    for i, hit in enumerate(context[:5], 1):
        doc_id = hit.get("document_id", "Unknown")
        chunk_text = get_chunk_text(hit.get("chunk_id", ""))
        if not chunk_text:
            continue

        key = citation_key(doc_id)
        citation_keys[f"Source_{i}"] = key  # Map enumeration to key for prompt clarity
        context_texts.append(f"[{key}]:\n{chunk_text[:1000]}")
    
    context_str = "\n\n".join(context_texts)

    # ------------------------------------------------------------------
    # Augment context with structured safety rows if the query seems to
    # request incidence/percentage data for a specific adverse event.
    # ------------------------------------------------------------------
    structured_notes = []
    keywords = ["pneumothorax", "hemoptysis", "respiratory infection"]
    for kw in keywords:
        if kw in query.lower():
            rows = fetch_safety_rows(kw, limit=15)
            for r in rows:
                note = (
                    f"[{citation_key(r['study_id'])}] Safety row: {r['pt']} – {r['events']} / {r['patients']}"
                    f" patients ({r['percentage']}%) in arm {r['arm_id']}"
                )
                structured_notes.append(note)
            break

    if structured_notes:
        context_str += "\n\nStructured Safety Data:\n" + "\n".join(structured_notes)
    
    # Create prompt
    system_prompt = """You are a medical evidence expert assistant specializing in interventional pulmonology.
Your role is to provide accurate, evidence-based answers using the provided research context.
Always cite specific studies and include numerical values when available.
Use in-text citations in the form (Author Year) where Author is the first author's last name and Year is the publication year, matching the keys provided in square brackets in the context (e.g., [Valipour 2014])."""
    
    user_prompt = f"""Based on the following medical research evidence, answer this question: {query}

Research Context:
{context_str}

Please provide a comprehensive answer with specific citations to the sources provided."""
    
    try:
        # Use GPT-5 for answer generation
        response = client.chat.completions.create(
            model="gpt-4o-2024-08-06",  # Using GPT-4o as GPT-5 may not be available yet
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating answer: {e}"

# Set default search parameters
search_k = 10
search_type = "Hybrid"

# Chat interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

# Chat input
if prompt := st.chat_input("Ask about interventional pulmonology research..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching evidence..."):
            # Search for relevant evidence
            if search_type == "Vector Only":
                search_results = search_evidence(prompt, search_k)
                db_results = []
            elif search_type == "Text Only":
                search_results = []
                db_results = query_database(prompt)
            else:  # Hybrid
                search_results = search_evidence(prompt, search_k // 2)
                db_results = query_database(prompt)
            
            # Combine results
            all_results = search_results + [
                {"chunk_id": r["chunk_id"], "document_id": r["document_id"], "score": float(r["bm25_score"])}
                for r in db_results
            ]
            
            # Store results
            st.session_state.search_results = all_results
            
        with st.spinner("Generating answer..."):
            # Generate answer
            answer = generate_answer(prompt, all_results)
            
            # Display answer
            st.markdown(answer)
            
            # Extract unique document ids for citation listing
            doc_ids = list({r.get("document_id", "Unknown") for r in all_results[:5]})
            sources = [citation_mla(doc_id) for doc_id in doc_ids]
            
            # Add assistant message
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })
            
            # Show sources
            with st.expander("View Sources"):
                for source in sources:
                    st.write(f"• {source}")

# Sidebar with monkey branding and status
with st.sidebar:
    st.markdown("""<div style="text-align: center; padding: 20px;">
    <h2>🩺 Bronchmonkey 🐵</h2>
    <p style="font-size: 14px; color: #666;">Interventional Pulmonology<br>Research Assistant</p>
    </div>""", unsafe_allow_html=True)
    
    st.divider()
    st.caption("System Status")
    
    # Check API status
    try:
        api_status = requests.get(f"{API_BASE_URL}/docs")
        if api_status.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
    except:
        st.error("❌ API Offline")
    
    # Check database and index status
    try:
        # Count studies in database
        db_parts = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(db_parts)
        with conn.cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM studies")
            study_count = cur.fetchone()[0]
        conn.close()
        
        # Count chunks from the FAISS index metadata
        chunk_count = 0
        meta_file = Path("data/index/meta.jsonl")
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                chunk_count = sum(1 for _ in f)
        
        st.info(f"📊 {study_count} studies, {chunk_count} chunks indexed")
    except Exception as e:
        st.warning(f"⚠️ Database connection issue")

    st.divider()
    st.caption("Built with hybrid search technology")
    st.caption("Interventional Pulmonology Research Tool")
