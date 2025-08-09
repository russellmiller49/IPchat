#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from functools import lru_cache

import streamlit as st
import requests
from dotenv import load_dotenv

# Local utilities
from utils.citations import extract_author_year, get_study_metadata, format_mla_citation
from utils.openai_client import chat_complete
from utils.streamlit_auth import check_password
from utils.depth_mode import (
    DepthConfig, expand_queries, multi_query_search,
    rerank_hits, enforce_breadth, get_contrastive_prompt,
    get_concise_prompt, critique_and_improve, verify_numeric_claims
)

load_dotenv()

# Basic auth gate (uses BASIC_AUTH_USERS env like "alice:pw1,bob:pw2")
check_password()

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
DATABASE_URL = os.getenv("DATABASE_URL", "")

# Page config
st.set_page_config(page_title="Bronchmonkey", page_icon="🐵", layout="wide")

# Sidebar controls
st.sidebar.title("⚙️ Settings")

# Depth mode toggle
depth_mode = st.sidebar.toggle(
    "🔬 **Depth Mode**",
    value=False,
    help="Enable comprehensive analysis with multiple queries, reranking, and contrastive synthesis"
)

# Model selection changes with depth mode
if depth_mode:
    st.sidebar.info("📊 Depth Mode Active: Enhanced search & synthesis")
    GEN_MODEL = "gpt-5"  # Force max model for depth
    quality_label = "Max Quality (gpt-5)"
else:
    # Standard model toggle
    env_model = os.getenv("GEN_MODEL", "").strip()
    options = ["Fast (gpt-5-mini)", "Max (gpt-5)"]
    if env_model and env_model not in ["gpt-5", "gpt-5-mini"]:
        options.append(f"Env ({env_model})")
    
    quality = st.sidebar.selectbox("Answer quality", options, index=0)
    if quality.startswith("Fast"):
        GEN_MODEL = "gpt-5-mini"
        quality_label = "Fast (gpt-5-mini)"
    elif quality.startswith("Max"):
        GEN_MODEL = "gpt-5"
        quality_label = "Max (gpt-5)"
    else:
        GEN_MODEL = env_model or "gpt-5-mini"
        quality_label = f"Custom ({GEN_MODEL})"

st.sidebar.caption(f"**Model:** {quality_label}")

# Initialize depth configuration
depth_config = DepthConfig(depth_enabled=depth_mode)

# Optional Postgres support for safety rows
try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except Exception:
    psycopg2 = None
    RealDictCursor = None

COMPLETE_EXTRACTIONS_DIR = Path("data/complete_extractions")


def fetch_safety_rows(keyword: str, limit: int = 20) -> List[Dict[str, Any]]:
    if not (DATABASE_URL and psycopg2 and RealDictCursor):
        return []
    try:
        db_parts = DATABASE_URL.replace("postgresql+psycopg2://", "postgresql://")
        conn = psycopg2.connect(db_parts, cursor_factory=RealDictCursor)  # type: ignore
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
    candidates = list(COMPLETE_EXTRACTIONS_DIR.glob(f"{document_id}*.json"))
    if not candidates:
        return None
    try:
        data = json.loads(candidates[0].read_text(encoding="utf-8"))
        return data.get("metadata") or {}
    except Exception:
        return None


def citation_key(document_id: str) -> str:
    author_last, year = extract_author_year(document_id)
    if author_last and year:
        return f"{author_last} {year}"
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
    metadata = get_study_metadata(document_id)
    return format_mla_citation(metadata)


st.title("Bronchmonkey")
st.caption("Interventional Pulmonology Research Assistant — lite-perf (GPT‑5 ready)")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "search_results" not in st.session_state:
    st.session_state.search_results = []

if not st.session_state.messages:
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Welcome! Ask me about trials, outcomes, and safety in interventional pulmonology. I will cite sources inline."
    })


def search_evidence(query: str, k: int = 10) -> List[Dict]:
    try:
        r = requests.post(f"{API_BASE_URL}/search", json={"query": query, "k": k}, timeout=15)
        if r.ok:
            return r.json().get("hits", [])
        st.error(f"Search API error: {r.status_code}")
        return []
    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def get_chunk_text(chunk_id: str) -> str:
    try:
        p = Path("data/chunks/chunks.jsonl")
        if not p.exists():
            return ""
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                chunk = json.loads(line)
                if chunk.get("chunk_id") == chunk_id:
                    return chunk.get("text", "")
        return ""
    except Exception as e:
        return f"Error loading chunk: {e}"


@lru_cache(maxsize=256)
def _cached_answer(system_prompt: str, user_prompt: str, model: str) -> str:
    return chat_complete(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )


def generate_answer(query: str, context: List[Dict], use_depth: bool = False) -> str:
    """Generate answer with optional depth mode enhancements"""
    
    # Apply breadth enforcement if in depth mode
    if use_depth:
        context = enforce_breadth(context, min_docs=depth_config.min_docs)
    
    context_texts: List[str] = []
    for i, hit in enumerate(context[:10 if use_depth else 5], 1):
        doc_id = hit.get("document_id", "Unknown")
        chunk_text = get_chunk_text(hit.get("chunk_id", ""))
        if not chunk_text:
            continue
        key = citation_key(doc_id)
        context_texts.append(f"[{key}]:\n{chunk_text[:1000]}")

    context_str = "\n\n".join(context_texts)

    # Structured safety data (if DB available)
    structured_notes = []
    try_keywords = ["pneumothorax", "hemoptysis", "respiratory infection"]
    for kw in try_keywords:
        if kw in query.lower():
            for r in fetch_safety_rows(kw, limit=15):
                structured_notes.append(
                    f"[{citation_key(r['study_id'])}] Safety: {r['pt']} – "
                    f"{r['events']} / {r['patients']} ({r['percentage']}%) in arm {r['arm_id']}"
                )
            break
    if structured_notes:
        context_str += "\n\nStructured Safety Data:\n" + "\n".join(structured_notes)

    # Use appropriate prompt based on mode
    system_prompt = get_contrastive_prompt() if use_depth else get_concise_prompt()
    
    user_prompt = f"Question: {query}\n\nResearch Context:\n{context_str}"
    if use_depth:
        user_prompt += "\n\nProvide a comprehensive analysis following the structured format."
    else:
        user_prompt += "\n\nWrite a concise, well-cited answer."

    try:
        # Generate initial answer
        answer = _cached_answer(system_prompt, user_prompt, depth_config.model if use_depth else GEN_MODEL)
        
        # Apply critique pass if in depth mode
        if use_depth and depth_config.use_critique:
            with st.spinner("Refining answer for nuance..."):
                answer = critique_and_improve(query, context_str, answer, chat_complete, depth_config.model)
        
        # Verify numeric claims
        if use_depth:
            warnings = verify_numeric_claims(answer, context_str)
            if warnings:
                st.warning(f"⚠️ Some claims may need verification: {', '.join(warnings[:3])}")
        
        return answer
    except Exception as e:
        return f"Error generating answer: {e}"


# --- Chat UI ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("View Sources"):
                for s in msg["sources"]:
                    st.write(f"• {s}")

if prompt := st.chat_input("Ask about BLVR outcomes, airway stents, thermoplasty, pleural procedures..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        # Enhanced search in depth mode
        if depth_mode:
            with st.spinner("🔍 Expanding query and searching multiple angles..."):
                # Expand queries
                queries = expand_queries(prompt, chat_complete)
                if len(queries) > 1:
                    st.caption(f"Searching with {len(queries)} query variations")
                
                # Multi-query search
                results = multi_query_search(queries, search_evidence, k_each=depth_config.k_each)
                
                # Rerank if enabled
                if depth_config.use_reranker:
                    with st.spinner("🎯 Reranking for relevance..."):
                        results = rerank_hits(prompt, results, top_n=depth_config.top_k_final)
        else:
            # Standard search
            with st.spinner("Searching evidence..."):
                results = search_evidence(prompt, depth_config.k_each)

        # Generate answer with appropriate mode
        model_label = depth_config.model if depth_mode else GEN_MODEL
        with st.spinner(f"Generating {'comprehensive analysis' if depth_mode else 'answer'} with {model_label}..."):
            answer = generate_answer(prompt, results, use_depth=depth_mode)

        st.markdown(answer)

        # Collect sources - more in depth mode
        num_sources = 10 if depth_mode else 5
        doc_ids = list({r.get("document_id", "Unknown") for r in results[:num_sources]})
        sources = [citation_mla(d) for d in doc_ids]
        
        # Show study diversity in depth mode
        if depth_mode and len(doc_ids) > 1:
            st.caption(f"📚 Synthesized from {len(doc_ids)} distinct studies")
        
        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

        with st.expander(f"View Sources ({len(sources)} studies)"):
            for s in sources:
                st.write(f"• {s}")

# Sidebar status
with st.sidebar:
    st.subheader("System Status")
    try:
        resp = requests.get(f"{API_BASE_URL}/docs", timeout=5)
        st.success("✅ API Online" if resp.ok else "❌ API Offline")
    except Exception:
        st.error("❌ API Offline")

    meta_file = Path("data/index/meta.jsonl")
    try:
        count = sum(1 for _ in meta_file.open()) if meta_file.exists() else 0
        st.caption(f"{count} chunks indexed")
    except Exception:
        st.caption("Index info unavailable")
    
    # Depth mode settings display
    if depth_mode:
        st.subheader("🔬 Depth Settings")
        st.caption(f"• Queries expanded: Yes")
        st.caption(f"• Min documents: {depth_config.min_docs}")
        st.caption(f"• Reranking: {'Yes' if depth_config.use_reranker else 'No'}")
        st.caption(f"• Critique pass: {'Yes' if depth_config.use_critique else 'No'}")
        st.caption(f"• Max tokens: {depth_config.max_tokens}")

    if st.button("Clear Answer Cache"):
        _cached_answer.cache_clear()
        st.toast("Answer cache cleared")
