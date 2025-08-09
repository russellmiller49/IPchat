#!/usr/bin/env python3
"""
Bronchmonkey - Interventional Pulmonology Research Assistant
Powered by hybrid search (FAISS + BM25 + PostgreSQL)

Run with: streamlit run chatbot_app.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import lru_cache

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
from utils.citations import (
    extract_author_year,
    get_study_metadata,
    format_mla_citation,
    format_inline_citation,
)

# Load environment variables
load_dotenv()

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "")
GEN_MODEL = os.getenv("GEN_MODEL", "gpt-5-2025-08-07")  # Using GPT-5 latest

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Structured-data helpers (PostgreSQL safety table)
# ---------------------------------------------------------------------------
COMPLETE_EXTRACTIONS_DIR = Path("data/complete_extractions")


def fetch_safety_rows(keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
    """Return safety rows whose PT matches the keyword (ILIKE)."""
    if not DATABASE_URL:
        return []
    try:
        db_parts = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(db_parts, cursor_factory=RealDictCursor)
        with conn.cursor() as cur:
            cur.execute(
                \
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
    author_last, year = extract_author_year(document_id)
    if author_last and year:
        return f"{author_last} {year}"
    # Fallback
    meta = load_metadata(document_id)
    if not meta:
        return document_id
    authors = meta.get("authors") or []
    year = meta.get("year") or "n.d."
    if authors:
        first_author_last = authors[0].split(",")[0] if "," in authors[0] else authors[0].split()[-1]
        return f"{first_author_last} {year}"
    return f"Unknown {year}"


def citation_mla(document_id: str) -> str:
    """Return a simple MLA-style citation string for the document."""
    metadata = get_study_metadata(document_id)
    return format_mla_citation(metadata)


# Page config
st.set_page_config(page_title="Bronchmonkey", page_icon="", layout="wide")

# Title (image removed for compactness)
st.title("Bronchmonkey")
st.caption("Your Interventional Pulmonology Research Assistant")
st.caption("Powered by hybrid search and efficient language models")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []

# Welcome (shown once)
if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": (
            "Welcome! I'm Bronchmonkey, your interventional pulmonology research assistant.\n"
            "I can help you find evidence from clinical trials, systematic reviews, and medical literature."
        )
    })


def search_evidence(query: str, k: int = 10) -> List[Dict]:
    """Search for relevant evidence using the API."""
    try:
        response = requests.post(f"{API_BASE_URL}/search", json={"query": query, "k": k}, timeout=15)
        if response.status_code == 200:
            return response.json().get("hits", [])
        st.error(f"Search API error: {response.status_code}")
        return []
    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def get_chunk_text(chunk_id: str) -> str:
    """Retrieve full text for a chunk."""
    try:
        chunks_file = Path("data/chunks/chunks.jsonl")
        if chunks_file.exists():
            with open(chunks_file, "r") as f:
                for line in f:
                    chunk = json.loads(line)
                    if chunk.get("chunk_id") == chunk_id:
                        return chunk.get("text", "")
        return ""
    except Exception as e:
        return f"Error loading chunk: {e}"


@lru_cache(maxsize=256)
def _cached_answer(system_prompt: str, user_prompt: str, model: str) -> str:
    # Use appropriate parameter based on model
    completion_params = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "timeout": 30,
    }
    
    # GPT-5 and newer models have different parameters
    if "gpt-5" in model or "o1" in model:
        completion_params["max_completion_tokens"] = 900
        # GPT-5 only supports default temperature (1)
        # Note: GPT-5 may not exist yet, falling back to gpt-4o if needed
    else:
        completion_params["max_tokens"] = 900
        completion_params["temperature"] = 0.2
    
    try:
        resp = client.chat.completions.create(**completion_params)
        return resp.choices[0].message.content
    except Exception as e:
        # If GPT-5 fails, try with GPT-4o
        if "gpt-5" in model:
            print(f"GPT-5 failed ({e}), falling back to gpt-4o-2024-08-06")
            completion_params["model"] = "gpt-4o-2024-08-06"
            completion_params["max_tokens"] = 900
            completion_params.pop("max_completion_tokens", None)
            completion_params["temperature"] = 0.2
            resp = client.chat.completions.create(**completion_params)
            return resp.choices[0].message.content
        raise


def generate_answer(query: str, context: List[Dict]) -> str:
    """Generate answer using OpenAI with retrieved context (cached)."""
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

    # Augment with structured rows if DATABASE_URL provided
    structured_notes = []
    for kw in ["pneumothorax", "hemoptysis", "respiratory infection"]:
        if kw in query.lower():
            for r in fetch_safety_rows(kw, limit=15):
                note = (
                    f"[{citation_key(r['study_id'])}] Safety row: {r['pt']} – "
                    f"{r['events']} / {r['patients']} patients ({r['percentage']}%) "
                    f"in arm {r['arm_id']}"
                )
                structured_notes.append(note)
            break
    if structured_notes:
        context_str += "\n\nStructured Safety Data:\n" + "\n".join(structured_notes)

    system_prompt = (
        "You are a medical evidence expert assistant specializing in interventional pulmonology.\n"
        "Provide accurate, evidence-based answers using ONLY the provided research context.\n"
        "Cite specific studies inline like (Author Year). Use numbers when available."
    )
    user_prompt = (
        f"Question: {query}\n\nResearch Context:\n{context_str}\n\n"
        "Write a concise, well-cited answer."
    )

    try:
        return _cached_answer(system_prompt, user_prompt, GEN_MODEL)
    except Exception as e:
        return f"Error generating answer: {e}"


# --- Chat UI ---
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("View Sources"):
                for source in message["sources"]:
                    st.write(f"• {source}")

if prompt := st.chat_input("Ask about interventional pulmonology research..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching evidence..."):
            search_results = search_evidence(prompt, 10)

        with st.spinner("Generating answer..."):
            answer = generate_answer(prompt, search_results)

        # Only show answer if it's not an error
        if answer and not answer.startswith("Error"):
            st.markdown(answer)
        else:
            st.error(answer if answer else "No answer generated. Please try again.")
            # Still show sources even if answer generation failed
            st.info("However, I found these relevant sources for your query:")

        # Collect sources from top results
        doc_ids = list({r.get("document_id", "Unknown") for r in search_results[:5]})
        sources = [citation_mla(doc_id) for doc_id in doc_ids]

        # Save assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer if answer else "Failed to generate answer, but sources are available below.",
            "sources": sources
        })

        with st.expander("View Sources"):
            for source in sources:
                st.write(f"• {source}")

# Sidebar status
with st.sidebar:
    st.caption("System Status")
    # API
    try:
        api_status = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        if api_status.status_code == 200:
            st.success("✅ API Online")
        else:
            st.error("❌ API Offline")
    except Exception:
        st.error("❌ API Offline")

    # Index counters
    try:
        # Count chunks via metadata
        meta_file = Path("data/index/meta.jsonl")
        chunk_count = sum(1 for _ in meta_file.open("r")) if meta_file.exists() else 0
        st.info(f"{chunk_count} chunks indexed")
    except Exception:
        st.warning("⚠️ Index info unavailable")

    # Cache controls
    if st.button("Clear Answer Cache"):
        _cached_answer.cache_clear()
        st.toast("Answer cache cleared")
