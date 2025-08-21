from ipchat.schemas.textbook import TextbookChapter

def test_min_schema_round_trip():
    data = {
      "chapter_metadata": {"title":"Sample"},
      "clinical_content": {"procedures":[], "algorithms":[], "clinical_guidelines":[], "drug_information":[]},
      "structured_data": {"tables":[], "figures":[], "boxes":[]},
      "clinical_cases": [],
      "definitions": [],
      "references": [],
      "summary": {"chapter_summary":"", "clinical_pearls":[], "practice_recommendations":[], "future_directions":None},
      "provenance": {"pdf_path":"p.pdf","adobe_json_path":"a.json","file_hash":"x","model":"m","extracted_at":"2025-01-01T00:00:00Z"}
    }
    chap = TextbookChapter.model_validate(data)
    assert chap.chapter_metadata.title == "Sample"