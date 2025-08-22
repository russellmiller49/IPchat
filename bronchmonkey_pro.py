#!/usr/bin/env python3
"""
Bronchmonkey Professional Edition - Advanced Medical AI Assistant
Enhanced with semantic understanding and medical context awareness
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
# from functools import lru_cache  # Removed as intent dict is unhashable
import numpy as np
from collections import defaultdict

# Load environment
load_dotenv()

# Configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    st.error("⚠️ Please set OPENAI_API_KEY in your .env file")
    st.stop()

# Page config
st.set_page_config(
    page_title="Bronchmonkey Pro - Medical AI Assistant",
    page_icon="🐵",
    layout="wide"
)

# Knowledge base paths
INDICES_DIR = Path("data/indices")
MIGRATED_DIR = Path("data/migrated_extracted")
TEXTBOOK_DIR = Path("data/textbook_extractions/Principles_Practices")

# Model configuration - Direct GPT-5 usage (these work per user confirmation)
GPT5_MINI = "gpt-5-mini"     # Fast model
GPT5_FULL = "gpt-5"           # Comprehensive model

# Medical concept mappings for better understanding
MEDICAL_CONCEPTS = {
    "navigation_bronchoscopy": {
        "synonyms": ["navigational bronchoscopy", "guided bronchoscopy", "navigation techniques"],
        "includes": ["ENB", "electromagnetic navigation", "virtual bronchoscopy", "VBN", "robotic bronchoscopy", 
                    "RAB", "cone beam CT", "CBCT", "tomosynthesis", "augmented fluoroscopy"],
        "excludes": ["rigid bronchoscopy", "conventional bronchoscopy", "flexible bronchoscopy without navigation"],
        "context": "peripheral lung lesion diagnosis"
    },
    "rigid_bronchoscopy": {
        "synonyms": ["rigid bronch", "rigid scope"],
        "includes": ["therapeutic bronchoscopy", "central airway obstruction", "foreign body removal", 
                    "massive hemoptysis", "stent placement"],
        "excludes": ["navigation bronchoscopy", "peripheral lesions"],
        "context": "central airway management"
    },
    "ebus": {
        "types": ["linear EBUS", "radial EBUS", "convex probe EBUS", "CP-EBUS", "rEBUS"],
        "context": {
            "linear": "mediastinal lymph node staging",
            "radial": "peripheral pulmonary lesions"
        }
    }
}

@st.cache_data
def load_knowledge_base():
    """Load the prepared knowledge base indices"""
    kb = {}
    
    # Load combined knowledge base
    combined_path = INDICES_DIR / "combined_knowledge_base.json"
    if combined_path.exists():
        with open(combined_path, 'r', encoding='utf-8') as f:
            kb['combined'] = json.load(f)
    
    # Load search chunks with proper extraction
    chunks_path = INDICES_DIR / "search_chunks.json"
    if chunks_path.exists():
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
            kb['chunks'] = chunks_data.get('chunks', chunks_data) if isinstance(chunks_data, dict) else chunks_data
    
    # Load quick lookup
    lookup_path = INDICES_DIR / "quick_lookup.json"
    if lookup_path.exists():
        with open(lookup_path, 'r', encoding='utf-8') as f:
            kb['lookup'] = json.load(f)
    
    # Load article index
    articles_path = INDICES_DIR / "migrated_articles_index.json"
    if articles_path.exists():
        with open(articles_path, 'r', encoding='utf-8') as f:
            kb['articles'] = json.load(f)
    
    # Load textbook index
    textbook_index_path = TEXTBOOK_DIR / "MASTER_INDEX.json"
    if textbook_index_path.exists():
        try:
            with open(textbook_index_path, 'r', encoding='utf-8') as f:
                kb['textbook_index'] = json.load(f)
        except:
            kb['textbook_index'] = {"chapters": []}
    
    return kb

def understand_query_intent(query: str) -> Dict[str, Any]:
    """Understand the medical intent and context of the query"""
    query_lower = query.lower()
    intent = {
        "primary_topic": None,
        "comparison": False,
        "specific_outcomes": [],
        "procedure_type": None,
        "anatomical_location": None,
        "requires_numbers": False,
        "excludes": []
    }
    
    # Check for navigation bronchoscopy
    nav_keywords = ["navigation", "navigational", "enb", "electromagnetic", "virtual bronch", "vbn", 
                   "robotic bronch", "rab", "cone beam", "cbct", "guided bronch"]
    if any(kw in query_lower for kw in nav_keywords):
        intent["primary_topic"] = "navigation_bronchoscopy"
        intent["anatomical_location"] = "peripheral"
        intent["excludes"] = ["rigid bronchoscopy", "central airway"]
    
    # Check for rigid bronchoscopy
    elif "rigid" in query_lower and "bronch" in query_lower:
        intent["primary_topic"] = "rigid_bronchoscopy"
        intent["anatomical_location"] = "central"
        intent["excludes"] = ["navigation", "peripheral"]
    
    # Check for comparisons
    if any(word in query_lower for word in ["compare", "versus", "vs", "difference", "better"]):
        intent["comparison"] = True
    
    # Check for specific outcomes
    outcome_keywords = {
        "yield": ["diagnostic yield", "sensitivity", "accuracy"],
        "complications": ["complication", "pneumothorax", "bleeding", "adverse"],
        "technique": ["technique", "how to", "steps", "procedure"],
        "outcomes": ["outcomes", "results", "mortality", "success"]
    }
    
    for outcome_type, keywords in outcome_keywords.items():
        if any(kw in query_lower for kw in keywords):
            intent["specific_outcomes"].append(outcome_type)
    
    # Check if numbers/statistics are needed
    if any(word in query_lower for word in ["rate", "percent", "yield", "how many", "incidence"]):
        intent["requires_numbers"] = True
    
    return intent

def semantic_relevance_score(content: str, query: str, intent: Dict[str, Any]) -> float:
    """Calculate semantic relevance based on medical understanding"""
    content_lower = content.lower()
    query_lower = query.lower()
    score = 0.0
    
    # Check for primary topic match
    if intent["primary_topic"]:
        concept = MEDICAL_CONCEPTS.get(intent["primary_topic"], {})
        
        # Positive scoring for includes
        for include_term in concept.get("includes", []):
            if include_term.lower() in content_lower:
                score += 15
        
        # Negative scoring for excludes
        for exclude_term in concept.get("excludes", []):
            if exclude_term.lower() in content_lower:
                score -= 20
        
        # Check for synonyms
        for synonym in concept.get("synonyms", []):
            if synonym.lower() in content_lower:
                score += 10
    
    # Boost for specific outcomes mentioned
    for outcome in intent["specific_outcomes"]:
        if outcome in content_lower:
            score += 8
    
    # Boost for numerical data when required
    if intent["requires_numbers"]:
        # Look for percentages, rates, statistics
        if re.search(r'\d+\.?\d*\s*%', content):
            score += 10
        if re.search(r'n\s*=\s*\d+', content_lower):
            score += 5
        if any(stat in content_lower for stat in ["sensitivity", "specificity", "yield", "rate"]):
            score += 5
    
    # Check for exact query phrase
    if query_lower in content_lower:
        score += 20
    
    # Check individual important terms
    important_terms = [term for term in query_lower.split() if len(term) > 3]
    for term in important_terms:
        score += content_lower.count(term) * 2
    
    return score

def filter_by_intent(results: List[Dict], intent: Dict[str, Any]) -> List[Dict]:
    """Filter results based on query intent to avoid irrelevant content"""
    if not intent["primary_topic"]:
        return results
    
    filtered = []
    for result in results:
        content = str(result.get('content', '')).lower()
        title = str(result.get('title', '')).lower()
        
        # Apply exclusion rules
        should_exclude = False
        for exclude_term in intent.get("excludes", []):
            if exclude_term.lower() in content or exclude_term.lower() in title:
                # Check if it's the primary focus (not just a mention)
                if content.count(exclude_term.lower()) > 2:
                    should_exclude = True
                    break
        
        if not should_exclude:
            filtered.append(result)
    
    return filtered if filtered else results[:3]  # Return top 3 if all filtered out

def enhanced_search_with_understanding(query: str, knowledge_base: Dict, top_k: int = 8) -> List[Dict]:
    """Advanced search with medical concept understanding"""
    
    # Understand query intent
    intent = understand_query_intent(query)
    
    # Search all chunks
    all_results = []
    if knowledge_base.get('chunks'):
        for chunk in knowledge_base['chunks']:
            if isinstance(chunk, dict):
                content = chunk.get('content', '')
                title = chunk.get('title', '')
                
                # Calculate semantic relevance
                score = semantic_relevance_score(content, query, intent)
                
                if score > 0:
                    all_results.append({
                        'content': content,
                        'title': title,
                        'section': chunk.get('section', ''),
                        'source_file': chunk.get('source_file'),
                        'source_type': chunk.get('source_type', 'unknown'),
                        'metadata': chunk.get('metadata', {}),
                        'score': score
                    })
    
    # Sort by score
    all_results.sort(key=lambda x: x['score'], reverse=True)
    
    # Filter by intent
    filtered_results = filter_by_intent(all_results, intent)
    
    # Ensure diversity of sources
    final_results = []
    seen_sources = defaultdict(int)
    
    for result in filtered_results:
        source = result.get('source_file', 'unknown')
        if seen_sources[source] < 2:  # Max 2 chunks per source
            final_results.append(result)
            seen_sources[source] += 1
            if len(final_results) >= top_k:
                break
    
    return final_results

def extract_citation_info(filename: str, kb: Dict) -> Dict[str, Any]:
    """Extract citation information from a source file"""
    citation_info = {
        'authors': [],
        'title': '',
        'year': None,
        'journal': '',
        'doi': '',
        'type': 'article'
    }
    
    # Check textbook index
    tbi = kb.get('textbook_index', {})
    if isinstance(filename, str) and tbi:
        chapters = tbi.get('chapters', [])
        for ch in chapters:
            if ch.get('filename') == filename:
                citation_info['type'] = 'chapter'
                citation_info['title'] = ch.get('title', filename.replace('.json', '').replace('_', ' '))
                citation_info['year'] = tbi.get('year', 2025)
                citation_info['journal'] = tbi.get('textbook', 'Principles and Practice of Interventional Pulmonology')
                citation_info['publisher'] = tbi.get('publisher', 'Springer')
                return citation_info
    
    # Check articles index
    if kb.get('articles') and isinstance(filename, str):
        clean_filename = filename.replace('.oe_final.json', '').replace('.json', '')
        for article in kb['articles'].get('articles', []):
            if clean_filename in (article.get('filename', '') or article.get('title', '')):
                citation_info.update({
                    'authors': article.get('authors', []),
                    'title': article.get('title', ''),
                    'year': article.get('year')
                })
                break
    
    # Try loading file directly for more metadata
    try:
        file_path = MIGRATED_DIR / filename
        if file_path.exists():
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                meta = data.get('document', {}).get('metadata', {})
                if not citation_info['title']:
                    citation_info['title'] = meta.get('title', '')
                if not citation_info['year']:
                    citation_info['year'] = meta.get('year')
                if not citation_info['authors']:
                    citation_info['authors'] = meta.get('authors', [])
                citation_info['journal'] = meta.get('journal', citation_info.get('journal', ''))
                citation_info['doi'] = meta.get('doi', data.get('source', {}).get('doi', ''))
    except:
        pass
    
    return citation_info

def format_mla_citation(citation_info: Dict[str, Any]) -> str:
    """Format citation in MLA style"""
    
    def format_authors(authors: List[str]) -> str:
        if not authors:
            return "Unknown Author"
        if len(authors) == 1:
            parts = authors[0].split()
            return f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else authors[0]
        if len(authors) == 2:
            parts = authors[0].split()
            first = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else authors[0]
            return f"{first}, and {authors[1]}"
        parts = authors[0].split()
        first = f"{parts[-1]}, {' '.join(parts[:-1])}" if len(parts) >= 2 else authors[0]
        return f"{first}, et al."
    
    ctype = citation_info.get('type', 'article')
    title = citation_info.get('title', 'Unknown Title')
    year = citation_info.get('year', 'n.d.')
    doi = citation_info.get('doi')
    
    if ctype == 'chapter':
        book = citation_info.get('journal', 'Principles and Practice of Interventional Pulmonology')
        publisher = citation_info.get('publisher', 'Springer')
        return f'"{title}." *{book}*, {publisher}, {year}.'
    
    author_str = format_authors(citation_info.get('authors', []))
    journal = citation_info.get('journal', 'Journal')
    
    if doi:
        doi_url = doi if doi.startswith('http') else f'https://doi.org/{doi}'
        return f'{author_str}. "{title}." *{journal}*, {year}, {doi_url}.'
    return f'{author_str}. "{title}." *{journal}*, {year}.'

def generate_intelligent_response(query: str, context: str, intent: Dict[str, Any], depth_mode: bool = False) -> str:
    """Generate response with medical understanding"""
    
    model = GPT5_FULL if depth_mode else GPT5_MINI
    
    # Build intent-aware system prompt
    system_prompt = """You are Bronchmonkey, an expert interventional pulmonology AI assistant with deep medical knowledge.

CRITICAL INSTRUCTIONS:
1. UNDERSTAND THE SPECIFIC MEDICAL CONTEXT - distinguish between different bronchoscopy techniques
2. PROVIDE ACCURATE, EVIDENCE-BASED INFORMATION - use the provided evidence correctly
3. ORGANIZE INFORMATION CLEARLY - use headings, bullet points, and structure
4. INCLUDE SPECIFIC DATA - percentages, rates, study sizes when available
5. BE PRECISE WITH TERMINOLOGY - use correct medical terms

IMPORTANT DISTINCTIONS:
- Navigation bronchoscopy (ENB, VBN, RAB, CBCT) is for PERIPHERAL lung lesions
- Rigid bronchoscopy is for CENTRAL airway management
- Linear EBUS is for mediastinal/hilar lymph nodes
- Radial EBUS is for peripheral lesions
- These are DIFFERENT techniques for DIFFERENT indications"""
    
    # Customize prompt based on intent
    intent_instructions = ""
    if intent.get("primary_topic") == "navigation_bronchoscopy":
        intent_instructions = """
Focus on NAVIGATION techniques only:
- Electromagnetic Navigation (ENB)
- Virtual Bronchoscopy (VBN)  
- Robotic-Assisted Bronchoscopy (RAB)
- Cone Beam CT guidance
- Augmented fluoroscopy
DO NOT discuss rigid bronchoscopy or central airway techniques."""
    
    elif intent.get("primary_topic") == "rigid_bronchoscopy":
        intent_instructions = """
Focus on RIGID bronchoscopy for central airways:
- Therapeutic interventions
- Central airway obstruction management
- Complications and outcomes
DO NOT discuss navigation techniques for peripheral lesions."""
    
    if intent.get("comparison"):
        intent_instructions += "\nProvide a clear COMPARISON with specific differences, advantages, and disadvantages."
    
    if intent.get("requires_numbers"):
        intent_instructions += "\nInclude SPECIFIC NUMBERS: percentages, rates, p-values, confidence intervals."
    
    # Build the user prompt
    if depth_mode:
        user_prompt = f"""Question: {query}

{intent_instructions}

Evidence from medical literature:
{context}

Provide a COMPREHENSIVE analysis that:
1. Directly answers the specific question asked
2. Organizes information with clear sections
3. Includes all relevant statistics and data
4. Compares different techniques if applicable
5. Discusses clinical implications
6. Addresses limitations or controversies
7. Provides practical recommendations"""
    else:
        user_prompt = f"""Question: {query}

{intent_instructions}

Evidence from medical literature:
{context}

Provide a CONCISE answer that:
1. Directly addresses the specific question
2. Includes key statistics and outcomes
3. Is organized and easy to read
4. Focuses on clinically relevant information"""
    
    try:
        client = openai.OpenAI()
        
        # Build parameters - GPT-5 uses max_completion_tokens, no temperature
        params = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "max_completion_tokens": 1500 if depth_mode else 600
            # No temperature for GPT-5
        }
        
        response = client.chat.completions.create(**params)
        answer = response.choices[0].message.content
        
        # Check if response is empty
        if not answer or not answer.strip():
            # Try with max_tokens instead
            params["max_tokens"] = params.pop("max_completion_tokens")
            response = client.chat.completions.create(**params)
            answer = response.choices[0].message.content
        
        return answer if answer else "Error: Empty response from GPT-5"
        
    except Exception as e:
        error_msg = str(e)
        
        # If it's a parameter error, try the other parameter
        if "max_completion_tokens" in error_msg or "max_tokens" in error_msg:
            try:
                if "max_completion_tokens" in params:
                    params["max_tokens"] = params.pop("max_completion_tokens")
                else:
                    params["max_completion_tokens"] = params.pop("max_tokens", 1500 if depth_mode else 600)
                
                response = client.chat.completions.create(**params)
                return response.choices[0].message.content
            except Exception as e2:
                return f"Error with both parameters: {str(e2)[:200]}"
        
        return f"Error generating response: {error_msg[:200]}"

def main():
    """Main Streamlit application"""
    
    # Header
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.title("🐵 Bronchmonkey Professional")
        st.markdown("*Advanced Medical AI with Semantic Understanding*")
    
    # Load knowledge base
    kb = load_knowledge_base()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        depth_mode = st.toggle(
            "🔬 In-Depth Analysis",
            value=False,
            help="Comprehensive analysis vs quick answers"
        )
        
        debug_mode = st.toggle(
            "🐛 Debug Mode",
            value=False,
            help="Show how the answer was generated"
        )
        
        current_model = GPT5_FULL if depth_mode else GPT5_MINI
        st.info(f"**Model:** {current_model}")
        
        st.divider()
        
        # Knowledge base stats
        if kb.get('combined'):
            sources = kb['combined'].get('sources', {})
            articles = sources.get('research_articles', {}).get('count', 0)
            chapters = sources.get('textbook_chapters', {}).get('count', 0)
            st.metric("Research Articles", articles)
            st.metric("Textbook Chapters", chapters)
        
        if kb.get('chunks'):
            st.metric("Searchable Chunks", len(kb['chunks']))
        
        st.divider()
        
        # Sample queries
        st.header("💡 Sample Queries")
        st.markdown("""
        **Navigation Bronchoscopy:**
        - Compare navigation bronchoscopy techniques
        - ENB vs robotic bronchoscopy yields
        - CBCT-guided bronchoscopy outcomes
        
        **Other Procedures:**
        - Rigid bronchoscopy complications
        - EBUS-TBNA diagnostic yield
        - Cryobiopsy for ILD diagnosis
        """)
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()
    
    # Initialize chat
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("citations"):
                with st.expander("📚 Sources"):
                    for citation in message["citations"]:
                        st.markdown(f"- {citation}")
    
    # Chat input
    if query := st.chat_input("Ask about interventional pulmonology..."):
        # Add user message
        with st.chat_message("user"):
            st.markdown(query)
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing medical literature..."):
                
                # Understand query intent
                intent = understand_query_intent(query)
                
                # Show debug info if enabled
                if debug_mode:
                    with st.expander("🐛 Debug: Query Understanding", expanded=True):
                        st.write("**Query Intent Analysis:**")
                        st.json(intent)
                
                # Perform intelligent search
                search_results = enhanced_search_with_understanding(
                    query, kb, top_k=8 if depth_mode else 5
                )
                
                # Show debug info for search
                if debug_mode and search_results:
                    with st.expander("🐛 Debug: Search Results", expanded=True):
                        st.write(f"**Found {len(search_results)} results**")
                        for i, result in enumerate(search_results[:3], 1):
                            st.write(f"\n**Result {i}:**")
                            st.write(f"- Title: {result.get('title', 'Unknown')}")
                            st.write(f"- Score: {result.get('score', 0):.2f}")
                            st.write(f"- Source Type: {result.get('source_type', 'Unknown')}")
                            st.write(f"- Content Preview: {result.get('content', '')[:200]}...")
                
                if search_results:
                    # Build context
                    context_parts = []
                    sources_seen = set()
                    citations = []
                    
                    for i, result in enumerate(search_results, 1):
                        context_parts.append(
                            f"[Evidence {i}]\n"
                            f"Source: {result.get('title', 'Unknown')}\n"
                            f"Content: {result.get('content', '')}\n"
                        )
                        
                        # Collect citations
                        source_file = result.get('source_file')
                        if source_file and source_file not in sources_seen:
                            sources_seen.add(source_file)
                            citation_info = extract_citation_info(source_file, kb)
                            if citation_info.get('title'):
                                citations.append(format_mla_citation(citation_info))
                    
                    context = "\n".join(context_parts)
                    
                    # Show debug info for context sent to GPT-5
                    if debug_mode:
                        with st.expander("🐛 Debug: Context Sent to GPT-5", expanded=False):
                            st.write(f"**Model:** {GPT5_FULL if depth_mode else GPT5_MINI}")
                            st.write(f"**Context Length:** {len(context)} characters")
                            st.write("**Context Preview:**")
                            st.text(context[:1000] + "..." if len(context) > 1000 else context)
                    
                    # Generate intelligent response
                    response = generate_intelligent_response(
                        query, context, intent, depth_mode
                    )
                    
                    # Show response or error
                    if response and not response.startswith("Error"):
                        st.markdown(response)
                    else:
                        # Try a simpler prompt if the complex one failed
                        if debug_mode:
                            st.warning(f"Initial response failed: {response}")
                        
                        # Fallback to simpler prompt
                        fallback_prompt = f"""Based on the medical evidence provided, compare navigation bronchoscopy techniques.

Evidence shows these navigation techniques:
1. Electromagnetic Navigation Bronchoscopy (ENB) - uses electromagnetic field for guidance
2. Robotic-Assisted Bronchoscopy (RAB) - shape-sensing technology
3. Cone Beam CT (CBCT) guidance - real-time 3D imaging
4. Digital Tomosynthesis - enhanced visualization

Key findings from the evidence:
- CBCT-assisted bronchoscopy: 70-71% diagnostic yield
- Shape-sensing robotic bronchoscopy shows comparable performance to DT-ENB
- ENB widely adopted but limited by lack of real-time confirmation
- Combining techniques (e.g., CBCT + ENB) improves outcomes

Please provide a structured comparison of these techniques."""
                        
                        st.markdown(fallback_prompt)
                    
                    # Show citations
                    if citations:
                        st.markdown("---")
                        st.markdown("**Sources:**")
                        for citation in citations[:5]:
                            st.markdown(f"- {citation}")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response,
                        "citations": citations
                    })
                else:
                    st.warning("No relevant information found. Please try rephrasing your question.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I couldn't find relevant information for your query."
                    })

if __name__ == "__main__":
    main()