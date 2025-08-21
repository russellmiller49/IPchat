import json
from pathlib import Path

import pytest

from tools.gold_standard_pipeline import GoldStandardPipeline


@pytest.fixture()
def stub_raw_extraction(tmp_path: Path) -> dict:
    # Reuse the invariant fixture content but ensure it looks like a raw extraction
    fixture = json.loads((Path(__file__).parent / "fixtures" / "gold_invariant_input.json").read_text(encoding="utf-8"))
    # Mimic production extractor metadata footprint
    fixture.setdefault("extraction_metadata", {})
    fixture["extraction_metadata"].update({
        "model": "gpt-4o",
        "text_pages": 11,
        "chunks_processed": 2,
    })
    return fixture


def test_gold_standard_pipeline_end_to_end_stub(tmp_path: Path, monkeypatch, stub_raw_extraction: dict):
    outdir = tmp_path / "gold"
    outdir.mkdir(parents=True, exist_ok=True)

    # Stub the extraction step to write our fixture into the expected raw_extractions path
    def fake_process_single_chapter(*, pdf_path, adobe_json_path=None, output_dir: Path, chapter_title=None, model=None):
        output_dir.mkdir(parents=True, exist_ok=True)
        p = output_dir / f"{Path(pdf_path).stem}_production.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(stub_raw_extraction, f, indent=2)
        return p

    # Patch the imported symbol in the pipeline module
    import tools.gold_standard_pipeline as gsp
    monkeypatch.setattr(gsp, "process_single_chapter", fake_process_single_chapter)

    # Avoid any PDF parsing and disable missing section extraction by returning empty text
    pipeline = GoldStandardPipeline(model="gpt-4o", output_dir=outdir, verbose=False)
    monkeypatch.setattr(pipeline, "_extract_text_from_pdf", lambda _: "")

    # Create a dummy PDF path
    pdf_path = tmp_path / "Approach.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n%stub\n")

    # Run the pipeline
    success, gold_path = pipeline.process_chapter(pdf_path=pdf_path, adobe_json_path=None, title="Approach")
    assert success is True
    assert gold_path.exists()

    enhanced = json.loads(gold_path.read_text(encoding="utf-8"))

    # Invariants: authors present
    authors = enhanced.get("chapter_metadata", {}).get("authors", [])
    assert isinstance(authors, list) and len(authors) >= 3

    # Invariants: page_range objects
    def walk(o):
        if isinstance(o, dict):
            yield o
            for v in o.values():
                yield from walk(v)
        elif isinstance(o, list):
            for v in o:
                yield from walk(v)

    for d in walk(enhanced):
        if "page_range" in d:
            pr = d["page_range"]
            assert isinstance(pr, dict) and {"start", "end"}.issubset(pr.keys())

    # Invariants: metrics normalized
    algs = enhanced.get("treatment_algorithms", [])
    if algs:
        perf = algs[0].get("performance", {})
        assert "accuracy" in perf and 0 <= perf["accuracy"]["value"] <= 1

    # Invariants: no stray top-level reference
    assert "reference" not in enhanced

    # Invariants: risk model references null when blank
    for rm in enhanced.get("risk_models", []):
        if rm.get("model_name", "").lower().startswith("mayo"):
            assert rm.get("reference") is None

    # Invariants: paths normalized to forward slashes
    meta = enhanced.get("extraction_metadata", {})
    if meta.get("source_pdf"):
        assert "/" in meta.get("source_pdf")
    if meta.get("adobe_json"):
        assert "/" in meta.get("adobe_json")

