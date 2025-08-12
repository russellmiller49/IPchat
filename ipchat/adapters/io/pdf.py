from pathlib import Path
from typing import List, Dict

def extract_pages(pdf_path: Path) -> List[Dict]:
    import fitz
    doc = fitz.open(str(pdf_path))
    out = []
    for i in range(len(doc)):
        out.append({"page": i+1, "text": doc[i].get_text("text")})
    doc.close()
    return out