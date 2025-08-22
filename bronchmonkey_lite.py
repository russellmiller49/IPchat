#!/usr/bin/env python3
"""
Bronchmonkey Lite - Streamlit Chat Interface
Simplified version for testing with prepared knowledge base
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import streamlit as st
from dotenv import load_dotenv
import openai
import random

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
    
    return kb

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
                if isinstance(title, str):
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

def generate_response(query: str, context: List[Dict], model: str = "gpt-4o-mini") -> str:
    """Generate response using OpenAI"""
    
    # Prepare context from search results
    context_text = ""
    sources = []
    
    for result in context:
        chunk = result['chunk']
        context_text += f"\n\n---\nSource: {chunk.get('title', 'Unknown')}\n"
        context_text += f"Section: {chunk.get('section', 'Unknown')}\n"
        context_text += f"Content: {chunk['content']}\n"
        
        # Track sources for citations
        if chunk.get('title'):
            sources.append(chunk['title'])
    
    # Create the prompt
    prompt = f"""You are Bronchmonkey, an AI assistant specializing in interventional pulmonology.
    
Based on the following context, answer this question: {query}

Context from medical literature:
{context_text}

Instructions:
1. Provide a clear, evidence-based answer
2. Include specific numbers and data when available
3. Be concise but thorough
4. Use medical terminology appropriately
5. If the context doesn't contain enough information, acknowledge limitations

Answer:"""
    
    try:
        client = openai.OpenAI()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical AI assistant specializing in interventional pulmonology."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Add sources
        if sources:
            unique_sources = list(set(sources))[:3]  # Top 3 unique sources
            answer += "\n\n**Sources:**\n"
            for source in unique_sources:
                # Clean up source title
                clean_source = source.replace('.oe_final', '').replace('_', ' ')
                if len(clean_source) > 60:
                    clean_source = clean_source[:60] + "..."
                answer += f"- {clean_source}\n"
        
        return answer
        
    except Exception as e:
        return f"Error generating response: {str(e)}"

def main():
    # Header with branding
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🐵 Bronchmonkey")
        st.caption("Interventional Pulmonology Research Assistant")
    
    # Load knowledge base
    with st.spinner("Loading knowledge base..."):
        kb = load_knowledge_base()
    
    if not kb:
        st.error("❌ Knowledge base not found. Please run `python prepare_knowledge_base.py` first.")
        st.stop()
    
    # Display stats
    if 'combined' in kb:
        stats = kb['combined']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Research Articles", stats['sources']['research_articles']['count'])
        with col2:
            st.metric("Textbook Chapters", stats['sources']['textbook_chapters']['count'])
        with col3:
            st.metric("Total Documents", stats['total_documents'])
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        # Welcome message
        st.session_state.messages.append({
            "role": "assistant",
            "content": "👋 Welcome to Bronchmonkey! I can help you with questions about interventional pulmonology procedures, diagnostic yields, complications, and clinical techniques.\n\n**Example queries:**\n- What is the diagnostic yield of EBUS-TBNA?\n- How do you perform balloon bronchoplasty?\n- What are the pneumothorax rates for lung volume reduction?\n- Compare rigid vs flexible bronchoscopy"
        })
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask about interventional pulmonology..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base..."):
                
                # First check quick lookup
                quick_answer = check_quick_lookup(prompt, kb)
                
                if quick_answer:
                    # Use quick lookup answer
                    st.markdown(quick_answer)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": quick_answer
                    })
                else:
                    # Perform search
                    search_results = simple_search(prompt, kb, top_k=5)
                    
                    if search_results:
                        # Generate response from context
                        response = generate_response(
                            prompt, 
                            search_results,
                            model="gpt-4o-mini"  # Using mini for speed
                        )
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })
                    else:
                        # No results found
                        fallback = "I couldn't find specific information about that in my knowledge base. Please try rephrasing your question or ask about:\n- Diagnostic procedures (EBUS, bronchoscopy, cryobiopsy)\n- Therapeutic interventions (stenting, valves, ablation)\n- Complications and management\n- Specific techniques and equipment"
                        st.markdown(fallback)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": fallback
                        })
    
    # Sidebar info
    with st.sidebar:
        st.markdown("---")
        st.markdown("### 📚 Knowledge Base")
        if 'combined' in kb:
            procedures = kb['combined']['procedures']['combined'][:10]
            st.markdown("**Available Procedures:**")
            for proc in procedures:
                st.markdown(f"- {proc}")
        
        st.markdown("---")
        st.markdown("### ℹ️ About")
        st.markdown("Bronchmonkey provides evidence-based information from:")
        st.markdown("- 292 research articles")
        st.markdown("- 41 textbook chapters")
        st.markdown("- Clinical guidelines")
        
        # Clear chat button
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()