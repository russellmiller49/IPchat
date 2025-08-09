
import re, json, hashlib
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Union

# Try to import optional libs
try:
    import pdfplumber
except Exception:
    pdfplumber = None

try:
    import PyPDF2
except Exception:
    PyPDF2 = None

HEADING_RE = re.compile(r'^\s*(\d+(?:\.\d+)*)\s+(.+)$')
FIG_LABEL_RE = re.compile(r'^\s*Fig\.?\s*([0-9]+[A-Za-z\-]*)', re.IGNORECASE)

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def clean_text(s: str) -> str:
    s = s.replace("\u00ad", "")                       # soft hyphen
    s = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', s)     # dehyphenate linebreaks
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n+', '\n', s)
    return s.strip()

def get_pages_text(pdf_path: Path) -> Dict[int, str]:
    pages = {}
    if pdfplumber is not None:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                txt = page.extract_text() or ""
                pages[i] = clean_text(txt)
            return pages
    # fallback
    if PyPDF2 is not None:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, start=1):
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                pages[i] = clean_text(txt)
    return pages

def split_paragraphs(page_text: str) -> List[str]:
    # Split on blank lines; then collapse short wraps
    lines = [ln.strip() for ln in page_text.split("\n")]
    paras, buf = [], []
    for ln in lines:
        if not ln:
            if buf:
                paras.append(" ".join(buf).strip())
                buf = []
            continue
        buf.append(ln)
    if buf:
        paras.append(" ".join(buf).strip())
    return [p for p in paras if len(p) > 8]

def parse_section_tree_from_contents(first_page_text: str) -> List[Dict[str, Any]]:
    tree: Dict[str, Dict[str, Any]] = {}
    order = []
    for ln in clean_text(first_page_text).split("\n"):
        m = HEADING_RE.match(ln)
        if not m:
            continue
        num, title = m.group(1), m.group(2)
        level = num.count(".") + 1
        node = {
            "id": f"sec{num.replace('.','_')}",
            "title": title.strip(),
            "level": level,
            "page_start": "?",
            "page_end": "?",
            "children": []
        }
        tree[num] = node
        order.append(num)
        if "." in num:
            parent_num = ".".join(num.split(".")[:-1])
            if parent_num in tree:
                tree[parent_num]["children"].append(node)
    top = [tree[k] for k in order if "." not in k]
    seen, uniq = set(), []
    for n in top:
        if n["id"] not in seen:
            uniq.append(n); seen.add(n["id"])
    return uniq

def extract_figures(page_num: int, page_text: str) -> List[Dict[str, Any]]:
    figs = []
    for ln in page_text.split("\n"):
        m = FIG_LABEL_RE.match(ln)
        if m:
            idx = m.group(1)
            figs.append({
                "id": f"fig_{page_num}_{idx}",
                "number": idx,
                "page": str(page_num),
                "provenance": {"page": str(page_num), "label": f"Fig {idx}"}
            })
    return figs

def build_text_units_and_figures(pdf_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    text_units: List[Dict[str, Any]] = []
    figures: List[Dict[str, Any]] = []
    section_tree: List[Dict[str, Any]] = []

    pages = get_pages_text(pdf_path)
    first_page_text = pages.get(1, "")
    if "Contents" in first_page_text:
        section_tree = parse_section_tree_from_contents(first_page_text)

    for pidx in sorted(pages.keys()):
        txt = pages[pidx]
        if not txt:
            continue

        # Figures on this page
        figures.extend(extract_figures(pidx, txt))

        # Paragraphs
        paras = split_paragraphs(txt)
        for i, para in enumerate(paras, start=1):
            unit = {
                "id": f"p{pidx}_{i}",
                "unit_type": "paragraph" if not para.lower().startswith(("abstract","references")) else "note",
                "section_id": "auto",
                "order": i,
                "text": para,
                "tokens": int(len(para.split()) * 1.3),
                "page_range": [str(pidx)],
                "provenance": {"page": str(pidx), "paragraph_index": i}
            }
            text_units.append(unit)

    return text_units, figures, section_tree

def assemble_chapter(
    pdf_path: Path,
    chapter_title: str,
    authors: List[str],
    book_title: str = "Principles and Practice of Interventional Pulmonology",
    publisher: str = "Springer Nature Switzerland AG",
    isbn13: Optional[str] = None,
    chapter_number: Optional[Union[int, str]] = None,
    source_url: Optional[str] = None
) -> Dict[str, Any]:

    sha = sha256_file(pdf_path)
    text_units, figures, section_tree = build_text_units_and_figures(pdf_path)

    chapter: Dict[str, Any] = {
        "schema_version": "1.0.0",
        "source": {
            "document_id": pdf_path.stem,
            "book_title": book_title,
            "publisher": publisher,
            "isbn13": isbn13,
            "chapter_number": chapter_number,
            "source_url": source_url,
            "file_sha256": sha,
            "license": "All rights reserved",
            "rights_holder": publisher,
            "access": "institutional"
        },
        "document": {
            "chapter_title": chapter_title,
            "authors": authors
        },
        "structure": {
            "toc_path": [],
            "section_tree": section_tree
        },
        "content": {
            "text_units": text_units,
            "figures": figures,
            "tables": [],
            "boxes": [],
            "equations": [],
            "cases": [],
            "references": []
        },
        "retrieval": {
            "keywords": [],
            "summary_tldr": f"Auto-built chapter JSON for {chapter_title}.",
            "nuggets": [],
            "chunks": []
        },
        "versioning": {
            "extraction_tool": "build_chapter.py",
            "model": "none",
            "timestamp": __import__("datetime").datetime.utcnow().isoformat() + "Z",
            "revision": "r1"
        }
    }
    return chapter

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--title", required=True)
    ap.add_argument("--authors", nargs="+", required=True)
    ap.add_argument("--book-title", default="Principles and Practice of Interventional Pulmonology")
    ap.add_argument("--publisher", default="Springer Nature Switzerland AG")
    ap.add_argument("--isbn13", default=None)
    ap.add_argument("--chapter-number", default=None)
    ap.add_argument("--source-url", default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    doc = assemble_chapter(
        pdf_path=args.pdf,
        chapter_title=args.title,
        authors=args.authors,
        book_title=args.book_title,
        publisher=args.publisher,
        isbn13=args.isbn13,
        chapter_number=args.chapter_number,
        source_url=args.source_url
    )
    out = args.out or args.pdf.with_suffix(".chapter.json")
    out.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Wrote", out)
