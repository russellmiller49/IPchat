# IPchat

Tools for extracting, indexing, and chatting with interventional pulmonary PDFs using
[google/langextract](https://github.com/google/langextract), Gemini, and Chroma.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Install Tesseract OCR separately if your PDFs are scanned images.

2. Export your Gemini API key:
   ```bash
   export GOOGLE_API_KEY="your-key"
   ```

3. Place PDFs to index in `data/pdfs/`.

## Usage

1. **Extract text from PDFs**
   ```bash
   python src/extract.py
   ```
   This writes `.txt` files next to each PDF.

2. **Index extracted text**
   ```bash
   python src/index.py
   ```
   Embeddings are generated with Gemini and stored in a local Chroma collection.

3. **Chat with the collection**
   ```bash
   python src/chat.py
   ```
   You can then ask questions about the indexed documents.

## Project layout
```
IPchat/
├── data/
│   └── pdfs/
├── docs/
├── src/
│   ├── chat.py
│   ├── extract.py
│   └── index.py
├── requirements.txt
└── README.md
```
