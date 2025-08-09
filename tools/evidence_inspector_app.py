#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lightweight Evidence Inspector
- Browse *.structured.json outputs
- Preview key fields and provenance
"""
import json
from pathlib import Path
import streamlit as st

OUTPUT_DIR = Path("data/outputs")
PDF_DIR = Path("data/raw_pdfs")

st.set_page_config(page_title="Evidence Inspector", layout="wide")
st.title("Evidence Inspector")

files = sorted(OUTPUT_DIR.glob("*.structured.json"))
if not files:
    st.info("No structured outputs found in data/outputs.")
    st.stop()

sel = st.selectbox("Select an output file", files, index=0, format_func=lambda p: p.name)

data = json.loads(Path(sel).read_text(encoding="utf-8"))

col1, col2 = st.columns(2)
with col1:
    st.subheader("Metadata")
    st.json(data.get("metadata", {}))

    st.subheader("Study Details")
    st.json(data.get("study_details", {}))

with col2:
    st.subheader("Outcomes")
    st.json(data.get("outcomes", {}))

    st.subheader("Adverse Events")
    st.json(data.get("adverse_events", []))

st.subheader("Key Findings")
st.json(data.get("key_findings", []))

st.subheader("Tables")
st.json(data.get("tables", []))

# Try to guess matching PDF
stem = Path(sel).stem.replace(".structured", "")
pdf_guess = PDF_DIR / f"{stem}.pdf"
if pdf_guess.exists():
    st.caption(f"Matched PDF: {pdf_guess}")
else:
    st.caption("No matching PDF found in data/raw_pdfs")
