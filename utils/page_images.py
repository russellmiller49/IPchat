from pathlib import Path
from functools import lru_cache
import fitz  # PyMuPDF

CACHE_DIR = Path("data/page_images")
CACHE_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=512)
def render_page(pdf_path: str, page_num: int, width: int = 900) -> str:
    """Render a single PDF page to a PNG and return its path.

    Args:
        pdf_path: Path to PDF file.
        page_num: Zero-based page index.
        width: Approximate pixel width for output image.

    Returns:
        str path to the rendered PNG.
    """
    out = CACHE_DIR / f"{Path(pdf_path).stem}_p{page_num+1}.png"
    if out.exists():
        return str(out)
    doc = fitz.open(pdf_path)
    page = doc.load_page(page_num)
    zoom = width / 72.0  # 72 dpi base
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
    pix.save(out)
    return str(out)
