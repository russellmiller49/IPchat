from pathlib import Path

from langextract import ExtractorPipeline
from langextract.detectors import SimpleLanguageDetector
from langextract.extractors import PdfExtractor


pipeline = ExtractorPipeline(
    extractor=PdfExtractor(),
    detector=SimpleLanguageDetector(),
)


def extract_from_pdf(pdf_path: Path) -> str:
    """Extract text from a PDF file."""
    result = pipeline.extract(pdf_path)
    return result.text


if __name__ == "__main__":
    pdf_dir = Path("data/pdfs")
    for pdf_file in pdf_dir.glob("*.pdf"):
        text = extract_from_pdf(pdf_file)
        output_path = pdf_file.with_suffix(".txt")
        output_path.write_text(text, encoding="utf-8")
