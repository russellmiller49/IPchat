"""
Streamlit app for SPACE edition - Hugging Face Space deployment.
Includes basic auth and pre-built indexes.
"""

import os
import streamlit as st
from pathlib import Path
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ipchat.core.config import load_config, Edition
from ipchat.core.pipelines import create_rag_pipeline
from ipchat.core.ui.streamlit_ui import run_streamlit_ui
from ipchat.core.utils.auth import check_basic_auth


def main():
    """Run the Hugging Face Space edition app."""
    
    # Load configuration for space edition
    config = load_config(edition="space")
    
    # Page config
    st.set_page_config(
        page_title="Bronchmonkey",
        page_icon="🐵",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Check authentication if enabled
    if config.auth.enabled:
        if not check_basic_auth():
            st.stop()
    
    # Initialize pipeline
    if "pipeline" not in st.session_state:
        with st.spinner("Initializing Bronchmonkey..."):
            st.session_state.pipeline = create_rag_pipeline(config)
    
    # Add edition badge to sidebar
    with st.sidebar:
        st.markdown("### 🤗 **Hugging Face Space**")
        st.caption("Cloud-hosted research assistant")
        st.divider()
    
    # Run the main UI
    run_streamlit_ui(st.session_state.pipeline, config)


if __name__ == "__main__":
    main()