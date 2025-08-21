import json
from pathlib import Path

import pytest

from tools.textbook_gold_standard_enhancer import (
    TextbookGoldStandardEnhancer,
    EnhancementConfig,
)


FIXTURE = Path(__file__).parent / "fixtures" / "gold_invariant_input.json"


def load_fixture():
    with open(FIXTURE, "r", encoding="utf-8") as f:
        return json.load(f)


def iter_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from iter_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from iter_dicts(v)


@pytest.mark.parametrize("model", ["gpt-4o"])  # avoid network-only behavior in tests
def test_enhancer_invariants_basic(model):
    data = load_fixture()
    config = EnhancementConfig(
        model=model,
        extract_missing_sections=False,  # avoid any network calls
        verbose=False,
    )
    enhancer = TextbookGoldStandardEnhancer(config)

    enhanced = enhancer.enhance(data, source_text=None, adobe_json=None)

    # 1) Authors restored/present
    authors = enhanced.get("chapter_metadata", {}).get("authors", [])
    assert isinstance(authors, list) and len(authors) >= 3

    # 2) Conclusion cleaned
    concl = enhanced.get("conclusion", {})
    if isinstance(concl, dict):
        points = concl.get("points", [])
        assert all("References" not in (p or "") for p in points)
        assert all("…" not in (p or "") for p in points)
        assert all(len(p) >= 20 for p in points)

    # 3) Metrics moved to performance (algorithm accuracy)
    alg = enhanced.get("treatment_algorithms", [{}])[0]
    perf = alg.get("performance", {})
    assert "accuracy" in perf and pytest.approx(perf["accuracy"]["value"], rel=1e-3) == 0.76
    assert "accuracy" not in {k for k in alg.keys() if k != "performance"}

    # 4) Diagnostic approaches filled from text (sensitivity/spec specificity)
    das = enhanced.get("diagnostic_approaches", [])
    # The volumetric assessment remains (Mayo moved to risk_models)
    vol = next((d for d in das if d.get("name") == "Volumetric assessment"), None)
    assert vol and isinstance(vol.get("performance"), dict)
    assert pytest.approx(vol["performance"]["sensitivity"]["value"], rel=1e-3) == 0.91
    assert pytest.approx(vol["performance"]["specificity"]["value"], rel=1e-3) == 0.90

    # 5) Page ranges unified to objects
    for d in iter_dicts(enhanced):
        if "page_range" in d:
            pr = d["page_range"]
            assert isinstance(pr, dict) and {"start", "end"}.issubset(pr.keys())

    # 6) Risk models created with null reference when blank
    rms = enhanced.get("risk_models", [])
    rm = next((r for r in rms if r.get("model_name") == "Mayo risk model"), None)
    assert rm is not None
    assert rm.get("reference") is None

    # 7) Definitions standardized
    defs = enhanced.get("definitions", [])
    assert isinstance(defs, list) and len(defs) >= 2
    quoted = next((d for d in defs if "content" in d), None)
    glossary = next((d for d in defs if "term" in d and "definition" in d), None)
    assert quoted and isinstance(quoted.get("page_range"), dict)
    assert quoted.get("reference")
    assert glossary and glossary.get("present_in_source") is False and glossary.get("added_by") == "enhancer"

    # 8) No stray top-level reference
    assert "reference" not in enhanced

    # 9) Paths normalized to forward slashes
    meta = enhanced.get("extraction_metadata", {})
    assert "/" in meta.get("source_pdf", "") and "/" in meta.get("adobe_json", "")

    # 10) present_in_source enforcement (algorithm item had empty source_excerpt)
    alg = enhanced.get("treatment_algorithms", [{}])[0]
    assert alg.get("present_in_source") is False

    # 11) Guideline source_organization autopopulated
    gl = enhanced.get("clinical_guidelines", [{}])[0]
    assert gl.get("source_organization") == "American College of Chest Physicians"

