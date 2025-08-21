"""
Streamlit app for FULL edition - includes all features, PostgreSQL support.
"""

import streamlit as st
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ipchat.core.config import load_config, Edition
from ipchat.core.pipelines import create_rag_pipeline
from ipchat.core.ui.streamlit_ui import run_streamlit_ui


def main():
    """Run the full edition Streamlit app."""
    
    # Load configuration for full edition
    config = load_config(edition="full")
    
    # Page config
    st.set_page_config(
        page_title="Bronchmonkey - Full Edition",
        page_icon="🐵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize pipeline
    if "pipeline" not in st.session_state:
        with st.spinner("Initializing Bronchmonkey (Full Edition)..."):
            st.session_state.pipeline = create_rag_pipeline(config)
    
    # Add edition badge to sidebar
    with st.sidebar:
        st.markdown("### 🏢 **Full Edition**")
        st.caption("PostgreSQL + API Server + All Features")
        st.divider()
    
    # Run the main UI
    run_streamlit_ui(st.session_state.pipeline, config)


if __name__ == "__main__":
    main()