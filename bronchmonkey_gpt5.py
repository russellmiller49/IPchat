#!/usr/bin/env python3
"""
Bronchmonkey GPT-5 Edition - Direct GPT-5 Implementation
Based on HuggingFace version with improved search and retrieval
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import streamlit as st
from dotenv import load_dotenv
import openai
from datetime import datetime
from functools import lru_cache

# Load environment
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("⚠️ Please set OPENAI_API_KEY in your .env file")
    st.stop()

# Page config
st.set_page_config(
    page_title="Bronchmonkey - GPT-5 Edition",
    page_icon="🐵",
    layout="wide"
)

# Knowledge base paths
INDICES_DIR = Path("data/indices")
MIGRATED_DIR = Path("data/migrated_extracted")
TEXTBOOK_DIR = Path("data/textbook_extractions/Principles_Practices")

# Direct GPT-5 models (matching HuggingFace implementation)
GPT5_MINI = "gpt-5-mini"    # Quick mode
GPT5_FULL = "gpt-5"          # In-depth mode

@st.cache_data
def load_knowledge_base():
    """Load the prepared knowledge base indices with enhanced structure"""
    kb = {}
    
    # Load combined knowledge base
    combined_path = INDICES_DIR / "combined_knowledge_base.json"
    if combined_path.exists():
        with open(combined_path, 'r', encoding='utf-8') as f:
            kb['combined'] = json.load(f)
    
    # Load search chunks
    chunks_path = INDICES_DIR / "search_chunks.json"
    if chunks_path.exists():
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            # Extract the actual chunks array from the structure
            if isinstance(chunks_data, dict) and 'chunks' in chunks_data:
                kb['chunks'] = chunks_data['chunks']
            else:
                kb['chunks'] = chunks_data
    
    # Load quick lookup
    lookup_path = INDICES_DIR / "quick_lookup.json"
    if lookup_path.exists():
        with open(lookup_path, 'r', encoding='utf-8') as f:
            kb['lookup'] = json.load(f)
    
    # Load article index for citations
    articles_path = INDICES_DIR / "migrated_articles_index.json"
    if articles_path.exists():
        with open(articles_path, 'r', encoding='utf-8') as f:
            kb['articles'] = json.load(f)

    # Load textbook master index for citations
    textbook_index_path = TEXTBOOK_DIR / "MASTER_INDEX.json"
    if textbook_index_path.exists():
        try:
            with open(textbook_index_path, 'r', encoding='utf-8') as f:
                kb['textbook_index'] = json.load(f)
        except Exception:
            kb['textbook_index'] = {"chapters": []}
    
    # Create enhanced index structure (similar to HuggingFace meta.jsonl)
    if 'chunks' in kb:
        kb['chunk_index'] = {}
        for i, chunk in enumerate(kb['chunks']):
            # Handle both dict and string chunks
            if isinstance(chunk, dict):
                chunk_id = chunk.get('id', f'chunk_{i}')
                kb['chunk_index'][chunk_id] = chunk
            else:
                # If chunk is a string, create a simple structure
                chunk_id = f'chunk_{i}'
                kb['chunk_index'][chunk_id] = {'content': chunk, 'id': chunk_id}
    
    return kb

def expand_query(query: str) -> List[str]:
    """Expand query with variations for better retrieval (from HuggingFace version)"""
    queries = [query]
    
    # Add lowercase version
    queries.append(query.lower())
    
    # Extract medical terms
    medical_terms = re.findall(
        r'\b(EBUS|TBNA|bronchoscopy|biopsy|pneumothorax|FEV1|BLVR|valve|'
        r'thermoplasty|cryobiopsy|navigation|electromagnetic|robotic|'
        r'endobronchial|transbronchial|mediastinal|peripheral|nodule|'
        r'adenopathy|staging|diagnosis|yield|complication|\\w+oscopy)\b', 
        query, re.IGNORECASE
    )
    if medical_terms:
        queries.extend([term.lower() for term in medical_terms])
    
    # Acronym expansions
    acronym_map = {
        'EBUS': 'endobronchial ultrasound',
        'TBNA': 'transbronchial needle aspiration',
        'TBLB': 'transbronchial lung biopsy',
        'TBLC': 'transbronchial lung cryobiopsy',
        'BLVR': 'bronchoscopic lung volume reduction',
        'FEV1': 'forced expiratory volume',
        'ILD': 'interstitial lung disease',
        'NSCLC': 'non-small cell lung cancer',
        'SCLC': 'small cell lung cancer',
        'ENB': 'electromagnetic navigation bronchoscopy',
        'RAB': 'robotic-assisted bronchoscopy',
        'COPD': 'chronic obstructive pulmonary disease'
    }
    
    for acronym, expansion in acronym_map.items():
        if acronym.lower() in query.lower():
            # Add version with expansion
            expanded = query.lower().replace(acronym.lower(), expansion)
            queries.append(expanded)
    
    return list(set(queries))  # Remove duplicates

def rerank_results(results: List[Dict], query: str) -> List[Dict]:
    """Rerank search results based on relevance (inspired by HuggingFace)"""
    scored_results = []
    query_lower = query.lower()
    query_terms = set(query_lower.split())
    
    for result in results:
        score = 0
        content = str(result.get('content', '')).lower()
        # Get title from metadata if not directly available
        title = result.get('title', '')
        if not title and result.get('metadata'):
            title = result['metadata'].get('title', '')
        if isinstance(title, dict):
            title = title.get('value', '')
        title = str(title).lower()
        
        # Exact query match
        if query_lower in content:
            score += 10
        if query_lower in title:
            score += 15

        # Term matches
        for term in query_terms:
            if len(term) > 2:  # Skip very short terms
                score += content.count(term) * 0.5
                score += title.count(term) * 2

        # Domain-specific boosts for this corpus
        if 'rigid bronchoscopy' in content or 'rigid bronchoscopy' in title:
            score += 12
        elif ('rigid' in content and 'bronch' in content) or ('rigid' in title and 'bronch' in title):
            score += 6

        if any(k in content for k in ['complication', 'adverse event', 'bleeding', 'hypox', 'pneumothorax', 'perforation']):
            score += 4
        
        # Boost recent studies
        year = result.get('year')
        if year:
            try:
                year_int = int(year)
                if year_int >= 2020:
                    score += 3
                elif year_int >= 2015:
                    score += 1
            except:
                pass
        
        # Boost if has statistical data
        if any(marker in content for marker in ['p=', 'CI', '%', 'n=', 'mean', 'median']):
            score += 2
        
        scored_results.append((score, result))
    
    # Sort by score descending
    scored_results.sort(key=lambda x: x[0], reverse=True)
    return [result for score, result in scored_results]

def enhanced_search(query: str, knowledge_base: Dict, top_k: int = 8, depth_mode: bool = False, prefer_textbook: bool = False) -> List[Dict]:
    """Enhanced search with query expansion and reranking"""
    all_results = []
    seen_content = set()
    
    # Expand query for better coverage
    expanded_queries = expand_query(query)
    
    # Search across all query variations
    for q in expanded_queries[:3]:  # Limit to avoid too many results
        results = simple_search(q, knowledge_base, top_k=top_k*2)  # Get more for reranking
        
        for result in results:
            # Deduplicate by content
            content_key = result.get('content', '')[:200]
            if content_key and content_key not in seen_content:
                seen_content.add(content_key)
                all_results.append(result)
    
    # Optional filtering based on query intent (e.g., rigid bronchoscopy)
    ql = query.lower()
    rigid_mode = ('rigid bronchoscopy' in ql) or ('rigid' in ql and 'bronch' in ql)

    if rigid_mode:
        filtered = []
        for r in all_results:
            c = str(r.get('content', '')).lower()
            t = str(r.get('title', '')).lower()
            # Must indicate rigid bronchoscopy by phrase or co-occurrence
            has_rigid_signal = ('rigid bronchoscopy' in c) or ('rigid bronchoscopy' in t) or (('rigid' in c and 'bronch' in c) or ('rigid' in t and 'bronch' in t))
            # Exclude tracheostomy/gastrostomy-only content
            irrelevant = (('tracheostomy' in c or 'gastrostomy' in c) and 'bronch' not in c)
            if has_rigid_signal and not irrelevant:
                filtered.append(r)
        all_results = filtered or all_results  # fall back if over-filtered

    # Rerank all results
    ranked_results = rerank_results(all_results, query)

    # In depth mode, ensure diversity of sources
    if depth_mode and len(ranked_results) > top_k:
        diverse_results = []
        seen_sources = set()

        for result in ranked_results:
            # Prefer source_file for grouping, fallback to generic source
            source = result.get('source_file') or result.get('source') or 'unknown'
            # Allow up to 2 chunks from same source (compare using same fallback logic)
            source_count = sum(1 for r in diverse_results if (r.get('source_file') or r.get('source') or 'unknown') == source)
            if source_count < 2:
                diverse_results.append(result)
                if len(diverse_results) >= top_k:
                    break
        
        return diverse_results[:top_k]

    # Ensure at least one relevant textbook chapter if available; optionally prefer it
    selected = ranked_results[:top_k]
    has_textbook = any((r.get('source') == 'textbook' or r.get('source_type') == 'textbook') for r in selected)

    # Gather candidates
    tb_candidates = [r for r in ranked_results if (r.get('source') == 'textbook' or r.get('source_type') == 'textbook')]
    if not tb_candidates:
        fresh = simple_search(query, knowledge_base, top_k=top_k * 2)
        tb_candidates = [r for r in fresh if (r.get('source') == 'textbook' or r.get('source_type') == 'textbook')]

    if tb_candidates:
        best_tb = tb_candidates[0]
        # Ensure presence
        if not has_textbook:
            if len(selected) < top_k:
                selected.append(best_tb)
            else:
                # replace the lowest non-textbook
                non_tb_indices = [i for i, r in enumerate(selected) if not (r.get('source') == 'textbook' or r.get('source_type') == 'textbook')]
                if non_tb_indices:
                    idx = non_tb_indices[-1]
                    selected[idx] = best_tb
        # Prefer textbook: promote to front
        if prefer_textbook:
            # Move any textbook entries to the front
            selected = sorted(selected, key=lambda r: 0 if (r.get('source') == 'textbook' or r.get('source_type') == 'textbook') else 1)

    return selected

def simple_search(query: str, knowledge_base: Dict, top_k: int = 5) -> List[Dict]:
    """Basic search implementation for fallback"""
    results = []
    
    # Quick lookup for specific data
    if knowledge_base.get('lookup'):
        for category, category_data in knowledge_base['lookup'].items():
            if any(term in query.lower() for term in category.lower().split('_')):
                # Handle the nested structure of lookup data
                if isinstance(category_data, dict):
                    # For nested dicts like diagnostic_yields with 'ebus' key
                    for subcategory, items in category_data.items():
                        if isinstance(items, list):
                            for item in items[:2]:
                                results.append({
                                    'content': f"{category.replace('_', ' ').title()} - {subcategory}: {json.dumps(item, indent=2)}",
                                    'source': 'quick_lookup',
                                    'title': f"{category.replace('_', ' ').title()} ({subcategory})",
                                    'section': 'quick_lookup',
                                    'source_file': None,
                                    'relevance': 0.9
                                })
                elif isinstance(category_data, list):
                    # For direct lists
                    for item in category_data[:2]:
                        results.append({
                            'content': f"{category.replace('_', ' ').title()}: {json.dumps(item, indent=2)}",
                            'source': 'quick_lookup',
                            'title': f"{category.replace('_', ' ').title()}",
                            'section': 'quick_lookup',
                            'source_file': None,
                            'relevance': 0.9
                        })
    
    # Search chunks
    if knowledge_base.get('chunks'):
        query_lower = query.lower()
        scored_chunks = []

        for chunk in knowledge_base['chunks']:
            # Handle both string and dict chunks
            if isinstance(chunk, str):
                content = chunk.lower()
                metadata = {}
            else:
                content = chunk.get('content', '').lower() if isinstance(chunk.get('content'), str) else str(chunk.get('content', '')).lower()
                metadata = chunk.get('metadata', {})

            # Calculate relevance score
            score = 0
            if query_lower in content:
                score += 10

            # Check individual terms
            query_terms = query_lower.split()
            for term in query_terms:
                if len(term) > 2:
                    score += content.count(term)

            # Boost if title matches
            title = chunk.get('title') or (metadata.get('title', '') if metadata else '')
            if isinstance(title, dict):
                title = title.get('value', '')
            title = str(title).lower()

            if any(term in title for term in query_terms):
                score += 5

            if score > 0:
                if isinstance(chunk, str):
                    scored_chunks.append((score, {'content': chunk, 'metadata': {}, 'title': '', 'section': '', 'source_file': None}))
                else:
                    # Ensure downstream has consistent fields for context/citations
                    enriched = dict(chunk)
                    enriched.setdefault('metadata', {})
                    # pass through top-level fields title/section/source_file
                    scored_chunks.append((score, enriched))
        
        # Sort by score and add top results
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        for score, chunk in scored_chunks[:top_k]:
            if isinstance(chunk, dict):
                results.append({
                    'content': chunk.get('content', ''),
                    'title': chunk.get('title', ''),
                    'section': chunk.get('section', ''),
                    'source_file': chunk.get('source_file') or chunk.get('metadata', {}).get('source_file'),
                    'source': chunk.get('source_type') or chunk.get('metadata', {}).get('source', 'unknown'),
                    'metadata': chunk.get('metadata', {}),
                    'relevance': min(score / 10, 1.0)
                })
            else:
                results.append({
                    'content': str(chunk),
                    'title': '',
                    'section': '',
                    'source_file': None,
                    'source': 'unknown',
                    'metadata': {},
                    'relevance': min(score / 10, 1.0)
                })
    
    return results[:top_k]

def extract_citation_info(filename: str, kb: Dict) -> Dict[str, Any]:
    """Extract citation information from a source file"""
    citation_info: Dict[str, Any] = {
        'authors': [],
        'title': '',
        'year': None,
        'journal': '',
        'doi': '',
        'type': 'article'
    }

    # 1) Textbook chapter detection via master index
    tbi = kb.get('textbook_index', {})
    if isinstance(filename, str):
        chapters = tbi.get('chapters', []) if isinstance(tbi, dict) else []
        for ch in chapters:
            if ch.get('filename') == filename:
                citation_info['type'] = 'chapter'
                citation_info['title'] = ch.get('title') or filename.replace('.json', '').replace('_', ' ')
                citation_info['year'] = tbi.get('year', 2025)
                citation_info['journal'] = tbi.get('textbook', 'Principles and Practice of Interventional Pulmonology')
                citation_info['publisher'] = tbi.get('publisher', 'Springer')
                return citation_info

    # 2) Research article via articles index
    if kb.get('articles') and 'articles' in kb['articles'] and isinstance(filename, str):
        clean_filename = filename.replace('.oe_final.json', '').replace('.json', '')
        for article in kb['articles']['articles']:
            if clean_filename in (article.get('filename') or '') or clean_filename in (article.get('title') or ''):
                citation_info['authors'] = article.get('authors', [])
                citation_info['title'] = article.get('title', '')
                citation_info['year'] = article.get('year')
                # journal/doi may be missing in index: try to load from file below
                break

    # 3) Fallback: read rich metadata from migrated file
    try:
        file_path = MIGRATED_DIR / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                meta = data.get('document', {}).get('metadata', {})
                src = data.get('source', {})
                # Only overwrite if empty
                if not citation_info.get('title'):
                    citation_info['title'] = meta.get('title', '')
                if not citation_info.get('year'):
                    citation_info['year'] = meta.get('year')
                if not citation_info.get('authors'):
                    citation_info['authors'] = meta.get('authors', [])
                citation_info['journal'] = meta.get('journal', '') or citation_info.get('journal', '')
                citation_info['doi'] = meta.get('doi') or src.get('doi') or ''
    except Exception:
        pass

    return citation_info

def format_mla_citation(citation_info: Dict[str, Any]) -> str:
    """Format citation in MLA style with DOI/URL when available."""

    def format_authors(authors: List[str]) -> str:
        if not authors:
            return "Unknown Author"
        if len(authors) == 1:
            parts = authors[0].split()
            return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else authors[0]
        if len(authors) == 2:
            parts = authors[0].split()
            first_author = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else authors[0]
            return f"{first_author}, and {authors[1]}"
        parts = authors[0].split()
        first_author = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else authors[0]
        return f"{first_author}, et al."

    ctype = citation_info.get('type', 'article')
    title = citation_info.get('title') or 'Unknown Title'
    year = citation_info.get('year') or 'n.d.'
    doi = citation_info.get('doi')
    authors = citation_info.get('authors', [])
    author_str = format_authors(authors)

    if ctype == 'chapter':
        book = citation_info.get('journal') or 'Principles and Practice of Interventional Pulmonology'
        publisher = citation_info.get('publisher') or 'Springer'
        return f'{author_str}. "{title}." {book}, {publisher}, {year}.'

    # Article
    journal = citation_info.get('journal') or 'Unknown Journal'
    if doi:
        doi_url = doi if doi.startswith('http') else f'https://doi.org/{doi}'
        return f'{author_str}. "{title}." {journal}, {year}, {doi_url}.'
    return f'{author_str}. "{title}." {journal}, {year}.'

@lru_cache(maxsize=128)
def generate_gpt5_response(query: str, context: str, depth_mode: bool = False) -> str:
    """Generate response using GPT-5 models with caching"""
    
    # Select model based on mode
    model = GPT5_FULL if depth_mode else GPT5_MINI
    
    # Construct prompt based on mode
    if depth_mode:
        system_prompt = """You are Bronchmonkey, an expert interventional pulmonology research assistant with access to a comprehensive medical database.
        
Provide a detailed, comprehensive analysis that:
- Synthesizes evidence from multiple studies
- Includes specific data, statistics, and outcomes
- Discusses clinical implications and controversies
- Compares different techniques or approaches when relevant
- Addresses limitations and future directions

Be thorough and academic in your response."""
        
        max_tokens = 1500
        temperature = 0.3
    else:
        system_prompt = """You are Bronchmonkey, an expert interventional pulmonology research assistant.

Provide a concise, focused response that:
- Directly answers the question
- Includes key statistics and outcomes
- Uses clear, clinical language

Keep your response brief (2-3 paragraphs)."""
        
        max_tokens = 600
        temperature = 0.2
    
    user_prompt = f"""Based on the following medical evidence, answer this question: {query}

Evidence:
{context}

Provide a {'comprehensive analysis' if depth_mode else 'concise answer'} based on the evidence provided."""
    
    try:
        client = openai.OpenAI()

        # Build parameters and only include temperature for models that support it
        is_gpt5_or_o1 = ("gpt-5" in model.lower()) or ("o1" in model.lower())

        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
        }

        if is_gpt5_or_o1:
            # Use completion-specific token field; omit temperature (unsupported)
            params["max_completion_tokens"] = max_tokens
        else:
            params["max_tokens"] = max_tokens
            params["temperature"] = temperature

        response = client.chat.completions.create(**params)
        answer = response.choices[0].message.content if response.choices else ""
        if not answer or not answer.strip():
            # Soft fallback: try GPT-4o once if GPT-5 returns empty
            try:
                fallback = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    max_tokens=max_tokens,
                    temperature=0.2
                )
                answer = fallback.choices[0].message.content if fallback.choices else ""
            except Exception:
                pass
        return answer
    except Exception as e:
        # Fallback path: if tokens field mismatch, flip and retry once
        try:
            client = openai.OpenAI()
            params = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
            }
            is_gpt5_or_o1 = ("gpt-5" in model.lower()) or ("o1" in model.lower())
            if is_gpt5_or_o1:
                params["max_tokens"] = max_tokens
            else:
                params["max_completion_tokens"] = max_tokens
                params["temperature"] = temperature
            response = client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e2:
            return f"Error with GPT-5: {str(e2)}"

def main():
    """Main Streamlit application"""
    
    # Header with model display
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🐵 Bronchmonkey")
        st.markdown("*GPT-5 Enhanced Interventional Pulmonology Research Assistant*")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Mode selection
        depth_mode = st.toggle(
            "🔬 In-Depth Analysis Mode",
            value=False,
            help="Toggle for comprehensive analysis vs quick answers"
        )
        
        # Prefer textbook evidence toggle
        prefer_textbook = st.toggle(
            "📘 Prefer textbook evidence",
            value=True,
            help="When available, include and prioritize at least one relevant textbook chapter."
        )

        # Display current model
        current_model = GPT5_FULL if depth_mode else GPT5_MINI
        st.info(f"**Model:** {current_model.upper()}")
        
        # Display mode description
        if depth_mode:
            st.markdown("""
            **In-Depth Mode Active** 
            - Comprehensive multi-study analysis
            - Detailed statistics and outcomes
            - Clinical implications discussion
            - Controversy and limitation analysis
            """)
        else:
            st.markdown("""
            **Quick Answer Mode Active**
            - Concise, focused responses
            - Key statistics and yields
            - Direct clinical answers
            - 2-3 paragraph summaries
            """)
        
        st.divider()
        
        # Knowledge base info
        st.header("📚 Knowledge Base")
        kb = load_knowledge_base()
        
        if kb.get('combined'):
            # Get counts from the combined knowledge base
            sources = kb['combined'].get('sources', {})
            total_articles = sources.get('research_articles', {}).get('count', 0)
            total_chapters = sources.get('textbook_chapters', {}).get('count', 0)
            st.metric("Research Articles", total_articles)
            st.metric("Textbook Chapters", total_chapters)
        
        if kb.get('chunks'):
            st.metric("Searchable Chunks", len(kb['chunks']))
        
        st.divider()
        
        # Sample queries
        st.header("💡 Sample Queries")
        sample_queries = [
            "What is the diagnostic yield of EBUS-TBNA for lung cancer?",
            "Compare navigation bronchoscopy techniques",
            "Pneumothorax rates in lung volume reduction",
            "Management of malignant central airway obstruction",
            "Cryobiopsy vs forceps biopsy for ILD"
        ]
        
        for query in sample_queries:
            if st.button(f"→ {query}", key=f"sample_{query[:20]}"):
                st.session_state.sample_query = query
        
        # Clear chat button
        if st.button("🗑️ Clear Chat", type="secondary"):
            st.session_state.messages = []
            st.rerun()
    
    # Main chat interface
    st.markdown("---")
    
    # Display metrics bar
    metrics_cols = st.columns(4)
    with metrics_cols[0]:
        st.metric("Mode", "IN-DEPTH" if depth_mode else "QUICK")
    with metrics_cols[1]:
        st.metric("Model", current_model.upper())
    with metrics_cols[2]:
        # Get article count properly
        if kb.get('combined'):
            article_count = kb['combined'].get('sources', {}).get('research_articles', {}).get('count', 0)
        else:
            article_count = len(kb.get('articles', {}).get('articles', []))
        st.metric("Articles", article_count)
    with metrics_cols[3]:
        st.metric("Chunks", len(kb.get('chunks', [])))
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display citations if present
            if "citations" in message:
                with st.expander("📚 Sources"):
                    for citation in message["citations"]:
                        st.markdown(f"- {citation}")
    
    # Handle sample query
    if "sample_query" in st.session_state:
        query = st.session_state.sample_query
        del st.session_state.sample_query
    else:
        # Chat input
        query = st.chat_input("Ask about interventional pulmonology research...")
    
    if query:
        # Display user message
        with st.chat_message("user"):
            st.markdown(query)
        
        # Add to history
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"{'Analyzing comprehensively' if depth_mode else 'Finding answer'}..."):
                # Search knowledge base with enhanced search
                search_results = enhanced_search(
                    query, 
                    kb, 
                    top_k=8 if depth_mode else 5,
                    depth_mode=depth_mode,
                    prefer_textbook=prefer_textbook
                )
                
                if search_results:
                    # Prepare context
                    context_parts = []
                    sources_seen = set()
                    citations = []
                    
                    for i, result in enumerate(search_results, 1):
                        content = result.get('content', '')
                        title = result.get('title') or 'Unknown Source'
                        section = result.get('section') or 'Unknown Section'
                        source_file = result.get('source_file')

                        # Build readable, attributed evidence block
                        context_parts.append(
                            f"---\nSource: {title}\nSection: {section}\nContent: {content}\n"
                        )

                        # Extract citation info when we have a source file
                        if source_file and source_file not in sources_seen:
                            sources_seen.add(source_file)
                            citation_info = extract_citation_info(source_file, kb)
                            mla_citation = format_mla_citation(citation_info)
                            citations.append(mla_citation)
                    
                    context = "\n".join(context_parts)
                    
                    # Generate GPT-5 response
                    response = generate_gpt5_response(query, context, depth_mode)
                    
                    # Display response
                    st.markdown(response)
                    
                    # Display citations
                    if citations:
                        st.markdown("---")
                        st.markdown("Sources (MLA Format):")
                        for citation in citations[:5]:  # Limit to 5 citations
                            st.markdown(f"- {citation}")

                    # Evidence viewer
                    with st.expander("Show evidence"):
                        for i, result in enumerate(search_results, 1):
                            title = result.get('title') or result.get('metadata', {}).get('title') or 'Unknown Source'
                            section = result.get('section') or 'Unknown Section'
                            content = result.get('content') or ''
                            st.markdown(f"{i}. Source: {title}\n\nSection: {section}\n\n{content}")
                    
                    # Add to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "citations": citations
                    })
                else:
                    st.warning("No relevant information found in the knowledge base.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't find relevant information for your query."
                    })

if __name__ == "__main__":
    main()
