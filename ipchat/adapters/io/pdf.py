from pathlib import Path
from typing import List, Dict

def extract_pages(pdf_path: Path) -> List[Dict]:
    """Return [{'page': N, 'text': '...'}] using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF
    doc = fitz.open(str(pdf_path))
    pages = []
    for i in range(len(doc)):
        pages.append({"page": i+1, "text": doc[i].get_text("text")})
    doc.close()
    return pages