#!/usr/bin/env python3
"""
Bronchmonkey Lite V2 - Streamlit Chat Interface with GPT-5
Enhanced version with GPT-5 models and MLA citations
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

# Load environment
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("⚠️ Please set OPENAI_API_KEY in your .env file")
    st.stop()

# Page config
st.set_page_config(
    page_title="Bronchmonkey - Interventional Pulmonology Assistant",
    page_icon="🐵",
    layout="wide"
)

# Knowledge base paths
INDICES_DIR = Path("data/indices")
MIGRATED_DIR = Path("data/migrated_extracted")
TEXTBOOK_DIR = Path("data/textbook_extractions/Principles_Practices")

# Model Configuration
# Try GPT-5 models first, with GPT-4 fallbacks
# Note: OpenAI may have released GPT-5 as "o1" models
DEFAULT_MODELS = {
    "quick": ["o1-mini", "gpt-5-mini", "gpt-4o-mini", "gpt-4-turbo-preview"],
    "depth": ["o1-preview", "gpt-5-2025-08-07", "gpt-5", "gpt-4o", "gpt-4-turbo-preview"]
}

# These will be set after testing which models are available
GPT5_MINI = None
GPT5_FULL = None

@st.cache_data
def detect_available_models():
    """Detect which models are actually available"""
    global GPT5_MINI, GPT5_FULL
    
    client = openai.OpenAI()
    
    # Test quick models
    for model in DEFAULT_MODELS["quick"]:
        try:
            # Test with appropriate parameter
            params = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
            }

            # Use correct parameter based on model
            if "gpt-5" in model or "o1" in model:
                params["max_completion_tokens"] = 5
            else:
                params["max_tokens"] = 5
                # Only non-GPT-5/O1 models support explicit temperature reliably
                params["temperature"] = 0
            
            response = client.chat.completions.create(**params)
            GPT5_MINI = model
            break
        except:
            continue
    
    # Test depth models
    for model in DEFAULT_MODELS["depth"]:
        try:
            params = {
                "model": model,
                "messages": [{"role": "user", "content": "test"}],
            }

            if "gpt-5" in model or "o1" in model:
                params["max_completion_tokens"] = 5
            else:
                params["max_tokens"] = 5
                params["temperature"] = 0
            
            response = client.chat.completions.create(**params)
            GPT5_FULL = model
            break
        except:
            continue
    
    # Fallback to GPT-4 if no GPT-5 available
    if not GPT5_MINI:
        GPT5_MINI = "gpt-4o-mini"
    if not GPT5_FULL:
        GPT5_FULL = "gpt-4o"
    
    return GPT5_MINI, GPT5_FULL

@st.cache_data
def load_knowledge_base():
    """Load the prepared knowledge base indices"""
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
            kb['chunks'] = json.load(f)
    
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

    # Load textbook master index to enrich citations
    textbook_index_path = TEXTBOOK_DIR / "MASTER_INDEX.json"
    if textbook_index_path.exists():
        try:
            with open(textbook_index_path, 'r', encoding='utf-8') as f:
                kb['textbook_index'] = json.load(f)
        except Exception:
            kb['textbook_index'] = {"chapters": []}
    
    return kb

def extract_citation_info(filename: str, kb: Dict) -> Dict[str, Any]:
    """Extract citation information from a source file"""
    citation_info = {
        'authors': [],
        'title': '',
        'year': None,
        'journal': '',
        'publisher': '',
        'doi': '',
        'type': 'article'
    }

    # Check if it's a textbook chapter
    tbi = kb.get('textbook_index', {})
    if isinstance(filename, str) and tbi:
        chapters = tbi.get('chapters', []) if isinstance(tbi, dict) else []
        for ch in chapters:
            if ch.get('filename') == filename:
                citation_info['type'] = 'chapter'
                citation_info['title'] = ch.get('title') or filename.replace('.json', '').replace('_', ' ')
                citation_info['year'] = tbi.get('year', 2025)
                citation_info['journal'] = tbi.get('textbook', 'Principles and Practice of Interventional Pulmonology')
                citation_info['publisher'] = tbi.get('publisher', 'Springer')
                return citation_info

    # For research articles, look up in articles index
    if kb.get('articles') and 'articles' in kb['articles']:
        clean_filename = filename.replace('.oe_final.json', '').replace('.json', '')
        for article in kb['articles']['articles']:
            if clean_filename in article.get('filename', '') or clean_filename in article.get('title', ''):
                citation_info['authors'] = article.get('authors', [])
                citation_info['title'] = article.get('title', '')
                citation_info['year'] = article.get('year')
                break

    # Fallback: try to load the actual file
    try:
        file_path = MIGRATED_DIR / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if 'document' in data and 'metadata' in data['document']:
                    meta = data['document']['metadata']
                    if not citation_info.get('authors'):
                        citation_info['authors'] = meta.get('authors', [])
                    if not citation_info.get('title'):
                        citation_info['title'] = meta.get('title', '')
                    if not citation_info.get('year'):
                        citation_info['year'] = meta.get('year')
                    citation_info['journal'] = meta.get('journal', '') or citation_info.get('journal', '')
                    citation_info['doi'] = meta.get('doi') or data.get('source', {}).get('doi') or ''
    except:
        pass

    return citation_info

def format_mla_citation(citation_info: Dict[str, Any]) -> str:
    """Format citation in MLA style (plain text, include DOI when available)"""

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
    authors = citation_info.get('authors', [])
    author_str = format_authors(authors)

    if ctype == 'chapter':
        book = citation_info.get('journal') or 'Principles and Practice of Interventional Pulmonology'
        publisher = citation_info.get('publisher') or 'Springer'
        return f'{author_str}. "{title}." {book}, {publisher}, {year}.'

    journal = citation_info.get('journal') or 'Unknown Journal'
    doi = citation_info.get('doi')
    if doi:
        doi_url = doi if doi.startswith('http') else f'https://doi.org/{doi}'
        return f'{author_str}. "{title}." {journal}, {year}, {doi_url}.'
    return f'{author_str}. "{title}." {journal}, {year}.'

def simple_search(query: str, knowledge_base: Dict, top_k: int = 5) -> List[Dict]:
    """Simple keyword-based search through chunks"""
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    results = []
    
    if 'chunks' in knowledge_base and 'chunks' in knowledge_base['chunks']:
        for chunk in knowledge_base['chunks']['chunks']:
            content_lower = chunk['content'].lower()
            
            # Calculate simple relevance score
            score = 0
            for word in query_words:
                if word in content_lower:
                    score += content_lower.count(word)
            
            # Boost score for title matches
            if 'title' in chunk and chunk['title']:
                # Handle title that might be a dict or string
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

def check_quick_lookup(query: str, knowledge_base: Dict) -> Optional[str]:
    """Check if query matches quick lookup patterns"""
    query_lower = query.lower()
    
    if 'lookup' not in knowledge_base:
        return None
    
    lookup = knowledge_base['lookup']
    
    # Check for diagnostic yield queries
    if 'diagnostic yield' in query_lower or 'sensitivity' in query_lower:
        if lookup.get('diagnostic_yields'):
            for proc, data in lookup['diagnostic_yields'].items():
                if proc.lower() in query_lower:
                    return f"Diagnostic yields for {proc}: {json.dumps(data, indent=2)}"
    
    # Check for complication queries
    if 'complication' in query_lower or 'pneumothorax' in query_lower or 'bleeding' in query_lower:
        if lookup.get('complication_rates'):
            for comp, data in lookup['complication_rates'].items():
                if comp.lower() in query_lower:
                    rates = [f"{d['source']}: {d['rate']}%" for d in data[:3]]
                    return f"{comp.capitalize()} rates:\n" + "\n".join(rates)
    
    # Check for procedure steps
    if 'how to' in query_lower or 'steps' in query_lower or 'technique' in query_lower:
        if lookup.get('procedure_steps'):
            for proc, data in lookup['procedure_steps'].items():
                if proc.lower() in query_lower:
                    steps = "\n".join(f"{i+1}. {step}" for i, step in enumerate(data['steps'][:5]))
                    return f"**{proc}** (from {data['source']}):\n{steps}"
    
    return None

def generate_response(query: str, context: List[Dict], model: str, depth_mode: bool, kb: Dict) -> Tuple[str, List[str]]:
    """Generate response using GPT-5 with proper citations"""
    
    # Prepare context from search results
    context_text = ""
    sources = []
    source_files = []
    
    for result in context:
        chunk = result['chunk']
        context_text += f"\n\n---\nSource: {chunk.get('title', 'Unknown')}\n"
        context_text += f"Section: {chunk.get('section', 'Unknown')}\n"
        context_text += f"Content: {chunk['content']}\n"
        
        # Track source files for citations
        if chunk.get('source_file'):
            source_files.append(chunk['source_file'])
    
    # Create the prompt based on depth mode
    if depth_mode:
        prompt = f"""You are Bronchmonkey, an expert AI assistant specializing in interventional pulmonology.
    
Provide a comprehensive, in-depth analysis for this question: {query}

Context from medical literature:
{context_text}

Instructions for IN-DEPTH mode:
1. Provide a thorough, detailed answer with multiple perspectives
2. Include ALL relevant numbers, statistics, and data points
3. Compare and contrast different studies or approaches
4. Discuss clinical implications and practical applications
5. Address potential limitations or controversies
6. Use appropriate medical terminology with explanations
7. Structure the response with clear sections if needed
8. Provide specific recommendations based on the evidence

Generate a comprehensive response that would satisfy a specialist physician:"""
    else:
        prompt = f"""You are Bronchmonkey, an AI assistant specializing in interventional pulmonology.
    
Answer this question concisely: {query}

Context from medical literature:
{context_text}

Instructions for QUICK mode:
1. Provide a clear, direct answer in 2-3 paragraphs
2. Include the most important numbers and key findings
3. Focus on practical, clinically relevant information
4. Use medical terminology appropriately
5. Be concise but complete

Answer:"""
    
    try:
        client = openai.OpenAI()
        
        # Build parameters based on model type
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": "You are a medical AI assistant specializing in interventional pulmonology. Provide evidence-based answers using the context provided."},
                {"role": "user", "content": prompt}
            ],
        }

        # GPT-5/O1 models use max_completion_tokens instead of max_tokens
        if "gpt-5" in model.lower() or "o1" in model.lower():
            params["max_completion_tokens"] = 2000 if depth_mode else 800
        else:
            params["max_tokens"] = 2000 if depth_mode else 800
            # Only include temperature for models that support it
            params["temperature"] = 0.3 if depth_mode else 0.2
        
        response = client.chat.completions.create(**params)
        
        answer = response.choices[0].message.content
        
        # Generate MLA citations for unique sources
        citations = []
        seen_files = set()
        for source_file in source_files:
            if source_file and source_file not in seen_files:
                seen_files.add(source_file)
                citation_info = extract_citation_info(source_file, kb)
                if citation_info.get('title'):
                    mla_citation = format_mla_citation(citation_info)
                    citations.append(mla_citation)
        
        # Limit to top 5 citations
        citations = citations[:5]
        
        return answer, citations
        
    except Exception as e:
        return f"Error generating response: {str(e)}", []

def main():
    # Detect available models
    global GPT5_MINI, GPT5_FULL
    if GPT5_MINI is None or GPT5_FULL is None:
        with st.spinner("Detecting available models..."):
            GPT5_MINI, GPT5_FULL = detect_available_models()
    
    # Header with branding
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🐵 Bronchmonkey")
        model_info = "GPT-5" if "gpt-5" in GPT5_FULL.lower() or "o1" in GPT5_FULL.lower() else "GPT-4"
        st.caption(f"Interventional Pulmonology Research Assistant - Powered by {model_info}")
    
    # Load knowledge base
    with st.spinner("Loading knowledge base..."):
        kb = load_knowledge_base()
    
    if not kb:
        st.error("❌ Knowledge base not found. Please run `python prepare_knowledge_base.py` first.")
        st.stop()
    
    # Sidebar with response mode toggle
    with st.sidebar:
        st.title("⚙️ Settings")
        
        # Response mode toggle
        depth_mode = st.toggle(
            "🔬 **In-Depth Analysis**",
            value=False,
            help="Toggle between quick answers (GPT-5-mini) and comprehensive analysis (GPT-5-full)"
        )
        
        # Prefer textbook evidence toggle
        prefer_textbook = st.toggle(
            "📘 Prefer textbook evidence",
            value=True,
            help="When available, include and prioritize at least one relevant textbook chapter."
        )

        # Show current model
        if depth_mode:
            current_model = GPT5_FULL
            model_display = GPT5_FULL.replace("gpt-", "GPT-").replace("o1-", "O1-").upper()
            st.info(f"📊 **Mode:** In-Depth Analysis\n**Model:** {model_display}\n**Response:** Comprehensive")
        else:
            current_model = GPT5_MINI
            model_display = GPT5_MINI.replace("gpt-", "GPT-").replace("o1-", "O1-").upper()
            st.success(f"⚡ **Mode:** Quick Answer\n**Model:** {model_display}\n**Response:** Concise")
        
        st.markdown("---")
        
        # Knowledge base stats
        if 'combined' in kb:
            stats = kb['combined']
            st.markdown("### 📚 Knowledge Base")
            st.metric("Research Articles", stats['sources']['research_articles']['count'])
            st.metric("Textbook Chapters", stats['sources']['textbook_chapters']['count'])
            st.metric("Total Documents", stats['total_documents'])
        
        st.markdown("---")
        st.markdown("### 🎯 Sample Queries")
        st.markdown("- Diagnostic yields")
        st.markdown("- Complication rates")
        st.markdown("- Procedure techniques")
        st.markdown("- Clinical guidelines")
        
        # Clear chat button
        st.markdown("---")
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Display main stats
    if 'combined' in kb:
        stats = kb['combined']
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Articles", stats['sources']['research_articles']['count'])
        with col2:
            st.metric("📚 Chapters", stats['sources']['textbook_chapters']['count'])
        with col3:
            st.metric("🔍 Searchable Chunks", "712")
        with col4:
            mode_emoji = "🔬" if depth_mode else "⚡"
            st.metric(f"{mode_emoji} Mode", "In-Depth" if depth_mode else "Quick")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Welcome to Bronchmonkey! I'm powered by GPT-5 and can help you with interventional pulmonology questions.\n\n**Choose your mode:**\n- ⚡ **Quick Answer** (default): Fast, concise responses using GPT-5-mini\n- 🔬 **In-Depth Analysis**: Comprehensive analysis using GPT-5-full\n\n**Example queries:**\n- What is the diagnostic yield of EBUS-TBNA?\n- Compare pneumothorax rates between different procedures\n- Explain the technique for balloon bronchoplasty\n- Review complications of transbronchial cryobiopsy"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant" and "citations" in message:
                st.markdown(message["content"])
                if message["citations"]:
                    st.markdown("---")
                    st.markdown("**References (MLA Format):**")
                    for i, citation in enumerate(message["citations"], 1):
                        st.markdown(f"{i}. {citation}")
            else:
                st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about interventional pulmonology..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner(f"{'Analyzing comprehensively...' if depth_mode else 'Generating quick answer...'}"):
                
                # First check quick lookup for simple queries
                if not depth_mode:
                    quick_answer = check_quick_lookup(prompt, kb)
                    if quick_answer:
                        st.markdown(quick_answer)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": quick_answer,
                            "citations": []
                        })
                        st.stop()
                # Perform search
                search_results = simple_search(prompt, kb, top_k=8 if depth_mode else 5)

                # Ensure at least one textbook source when relevant/available
                def ensure_textbook(results: List[Dict], kb: Dict, query: str, prefer: bool) -> List[Dict]:
                    has_tb = any(r['chunk'].get('source_type') == 'textbook' for r in results)
                    if has_tb and not prefer:
                        return results
                    # Find best textbook candidates using the same search and filter
                    tb_candidates = [r for r in simple_search(query, kb, top_k=12) if r['chunk'].get('source_type') == 'textbook']
                    if not tb_candidates:
                        return results
                    best_tb = tb_candidates[0]
                    # If prefer, put textbook first; otherwise, ensure presence
                    out = list(results)
                    if prefer:
                        # Place at the front if not already present
                        if all(best_tb['chunk'].get('chunk_id') != r['chunk'].get('chunk_id') for r in out):
                            out = [best_tb] + out[:-1] if len(out) >= 1 else [best_tb]
                        else:
                            # Move existing textbook to front
                            out = sorted(out, key=lambda r: 0 if r['chunk'].get('source_type') == 'textbook' else 1)
                    else:
                        if not has_tb:
                            if len(out) >= 1:
                                out[-1] = best_tb
                            else:
                                out.append(best_tb)
                    return out

                search_results = ensure_textbook(search_results, kb, prompt, prefer_textbook)
                
                if search_results:
                    # Generate response from context
                        response, citations = generate_response(
                            prompt, 
                            search_results,
                            model=current_model,
                            depth_mode=depth_mode,
                            kb=kb
                        )
                        
                        # Display response
                        st.markdown(response)
                        
                        # Display citations if available
                        if citations:
                            st.markdown("---")
                        st.markdown("References (MLA Format):")
                        for i, citation in enumerate(citations, 1):
                            st.markdown(f"{i}. {citation}")

                        # Evidence viewer
                        with st.expander("Show evidence"):
                            for i, result in enumerate(search_results, 1):
                                ch = result['chunk']
                                title = ch.get('title') or 'Unknown Source'
                                section = ch.get('section') or 'Unknown Section'
                                content = ch.get('content') or ''
                                st.markdown(f"{i}. Source: {title}\n\nSection: {section}\n\n{content}")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "citations": citations
                    })
                else:
                    # No results found
                    fallback = "I couldn't find specific information about that in my knowledge base. Please try rephrasing your question or ask about:\n- Diagnostic procedures (EBUS, bronchoscopy, cryobiopsy)\n- Therapeutic interventions (stenting, valves, ablation)\n- Complications and management\n- Specific techniques and equipment"
                    st.markdown(fallback)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": fallback,
                        "citations": []
                    })

if __name__ == "__main__":
    main()
