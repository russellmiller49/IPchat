"""
Unified Streamlit UI for all IPChat editions.
"""

import streamlit as st
from typing import Optional
import os

from ipchat.core.config import IPChatConfig, Edition
from ipchat.core.pipelines import RAGPipeline


def run_streamlit_ui(pipeline: RAGPipeline, config: IPChatConfig):
    """
    Run the main Streamlit UI with edition-specific customizations.
    
    Args:
        pipeline: The RAG pipeline instance
        config: Configuration object
    """
    
    # Header with branding
    col1, col2 = st.columns([1, 4])
    with col1:
        st.markdown("# 🐵")
    with col2:
        st.markdown("# Bronchmonkey")
        st.caption("Interventional Pulmonology Research Assistant")
    
    st.divider()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("### Settings")
        
        # Depth mode toggle (if enabled for edition)
        if config.depth_features:
            depth_mode = st.toggle(
                "🔬 **Depth Mode**",
                value=True,
                help="Enable comprehensive analysis with nuanced synthesis"
            )
        else:
            depth_mode = False
        
        # Number of results slider
        num_results = st.slider(
            "Search Results",
            min_value=5,
            max_value=20,
            value=config.retrieval.num_results,
            help="Number of evidence chunks to retrieve"
        )
        
        # Model selection (for full edition)
        if config.edition == Edition.FULL:
            st.divider()
            st.markdown("### Advanced")
            
            model_options = ["gpt-4o-mini", "gpt-4o", "gpt-5-2025-08-07"]
            selected_model = st.selectbox(
                "AI Model",
                options=model_options,
                index=model_options.index(config.llm.model) if config.llm.model in model_options else 0
            )
            pipeline.llm.model = selected_model
        
        # Debug mode toggle
        st.divider()
        st.markdown("### Developer")
        
        debug_mode = st.toggle(
            "🐛 **Debug Mode**",
            value=False,
            help="Show underlying reasoning, search strategy, and synthesis process"
        )
        
        if debug_mode:
            st.info("Debug mode enabled - reasoning steps will be shown")
        
        # About section
        st.divider()
        st.markdown("### About")
        st.markdown(f"**Edition:** {config.edition.value.title()}")
        st.caption("Ask questions about interventional pulmonology evidence")
        
        # Example queries
        with st.expander("Example Queries"):
            st.markdown("""
            - What percent of BLVR patients had pneumothorax?
            - Compare robotic bronchoscopy diagnostic yields
            - FEV1 improvements with endobronchial valves
            - Rigid bronchoscopy complications
            - Cryobiopsy vs forceps for ILD diagnosis
            """)
    
    # Main chat interface
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Show debug info if it was shown when generated
            if message["role"] == "assistant" and message.get("debug_shown") and debug_mode:
                st.caption("ℹ️ Debug information was shown for this response")
            
            # Show bibliography if present
            if message["role"] == "assistant" and "bibliography" in message:
                with st.expander("📚 Sources"):
                    st.markdown(message["bibliography"])
    
    # Chat input
    if prompt := st.chat_input("Ask about medical evidence..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            debug_data = {}  # Collect debug info
            
            # Different processing based on depth mode
            if depth_mode:
                with st.spinner("🔍 Expanding query and searching multiple angles..."):
                    # Query the pipeline with depth
                    result = pipeline.query(
                        question=prompt,
                        num_results=num_results,
                        use_depth=True
                    )
                    
                    # Collect debug info if available
                    if hasattr(pipeline, 'last_debug_info'):
                        debug_data = pipeline.last_debug_info
            else:
                with st.spinner("Searching evidence and generating response..."):
                    # Standard query
                    result = pipeline.query(
                        question=prompt,
                        num_results=num_results,
                        use_depth=False
                    )
            
            # Display answer
            st.markdown(result["answer"])
            
            # Show debug info if debug mode is on
            if debug_mode:
                with st.expander("🐛 Debug Information", expanded=True):
                    # Query expansion (if depth mode)
                    if depth_mode and "expanded_queries" in result.get("metadata", {}):
                        st.subheader("Query Expansion")
                        for i, q in enumerate(result["metadata"]["expanded_queries"], 1):
                            st.text(f"{i}. {q}")
                    
                    # Search statistics
                    st.subheader("Search Statistics")
                    st.json({
                        "Total chunks retrieved": len(result.get("search_results", [])),
                        "Unique documents": len(set(r.get("document_id", r.get("chunk_id", ""))[:30] 
                                                   for r in result.get("search_results", []))),
                        "Model used": result.get("metadata", {}).get("model", pipeline.llm.model),
                        "Depth mode": result.get("metadata", {}).get("depth_mode", depth_mode),
                        "Edition": config.edition.value
                    })
                    
                    # Top search results with scores
                    if result.get("search_results"):
                        st.subheader("Top Search Results")
                        for i, res in enumerate(result["search_results"][:3], 1):
                            score = res.get("score", res.get("combined_score", 0))
                            source = res.get("source", res.get("chunk_id", "Unknown"))
                            st.text(f"{i}. Score: {score:.3f} | Source: {source[:50]}...")
                            with st.container():
                                st.caption(res.get("text_preview", res.get("text", ""))[:200] + "...")
                    
                    # Reasoning process (if available)
                    if "reasoning" in result.get("metadata", {}):
                        st.subheader("Reasoning Process")
                        st.text(result["metadata"]["reasoning"])
            
            # Display sources
            if result.get("bibliography"):
                with st.expander("📚 Sources"):
                    st.markdown(result["bibliography"])
            
            # Save to chat history
            message_data = {
                "role": "assistant",
                "content": result["answer"],
                "bibliography": result.get("bibliography", ""),
                "debug_shown": debug_mode
            }
            st.session_state.messages.append(message_data)
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns(3)
    with col1:
        st.caption("🐵 Bronchmonkey v0.2.0")
    with col2:
        st.caption(f"Edition: {config.edition.value}")
    with col3:
        st.caption(f"Model: {pipeline.llm.model}")


def check_basic_auth() -> bool:
    """
    Check basic authentication for Space edition.
    
    Returns:
        True if authenticated, False otherwise
    """
    # Get auth config from environment
    auth_users = os.getenv("BASIC_AUTH_USERS", "")
    
    if not auth_users:
        return True  # No auth configured
    
    # Parse auth users (format: user1:pass1,user2:pass2)
    valid_users = {}
    for pair in auth_users.split(","):
        if ":" in pair:
            user, password = pair.split(":", 1)
            valid_users[user.strip()] = password.strip()
    
    # Show login form
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.markdown("## 🔒 Login Required")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
            
            if submitted:
                if username in valid_users and valid_users[username] == password:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid credentials")
        
        return False
    
    return True