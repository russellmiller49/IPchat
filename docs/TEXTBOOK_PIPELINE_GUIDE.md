# Textbook Chapter Extraction Pipeline: Runbook & Troubleshooting

This guide explains how to run the complete textbook chapter extraction pipeline end‑to‑end, what it produces, and how to troubleshoot common issues. It covers both Linux/macOS and Windows usage.

## Overview

The pipeline has three phases:

1) Multi‑pass extraction (production)
   - Parses PDF text and Adobe Extract JSON tables
   - Runs several domain‑specific LLM passes (metadata, diagnostics, guidelines, tables, figures, education, references)
   - Output: `data/gold_standard_extractions/raw_extractions/<Chapter>_production.json`

2) Gold‑standard enhancement
   - Adds risk_models, technology_technique, treatment_algorithms
   - Normalizes metrics into a `performance` block (proportions)
   - Cleans OCR artifacts; standardizes `page_range`; adds clinical interpretations to tables
   - Output: `data/gold_standard_extractions/<Chapter>_gold_standard.json`

3) Quality validation & summary
   - Computes a quality score and flags missing sections
   - Output: `data/gold_standard_extractions/<Chapter>_gold_standard_quality.json`, plus `extraction_summary.json`

Key scripts:

- `tools/gold_standard_pipeline.py` (main entrypoint)
- `tools/production_multipass_textbook_extractor.py` (phase 1)
- `tools/textbook_gold_standard_enhancer.py` (phase 2)

Optional UI:
- `tools/evidence_inspector_app.py` (Streamlit UI to browse outputs)

## Prerequisites

- Python 3.10+
- An OpenAI API key in your environment
  - Create `.env` in repo root and add `OPENAI_API_KEY=...`
- Install dependencies:

```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Quick API sanity check:

```bash
python tools/quick_api_test.py  # or: python tools/archive/quick_api_test.py
```

## Data Layout

Place files under `Textbooks/`:

- `Textbooks/Chapter pdfs/<Chapter>.pdf`
- `Textbooks/Chapter json/<Chapter>.json` (Adobe Extract JSON; optional but recommended)

Outputs go to `data/gold_standard_extractions/`.

## Running the Pipeline

Single chapter:

```bash
python tools/gold_standard_pipeline.py \
  --single "Textbooks/Chapter pdfs/Approach to Peripheral Lung Lesions.pdf" \
  --adobe-json "Textbooks/Chapter json/Approach to Peripheral Lung Lesions.json" \
  --model gpt-5 \
  --verbose
```

Batch all chapters:

```bash
python tools/gold_standard_pipeline.py --batch --model gpt-5
```

Model choices:
- `gpt-5`: Highest quality, uses the Responses API (no `response_format` parameter). Good default for production.
- `gpt-4o`: Reliable fallback using Chat Completions with JSON Schema outputs.

Performance tips:
- To iterate faster, run fewer passes: `--passes pass0_metadata pass3_diagnostics pass6_tables`
- If you hit rate limits, re‑run; the extractor uses a small concurrency for GPT‑5.

## Inspecting Outputs

- Raw extractions: `data/gold_standard_extractions/raw_extractions/`
- Gold standard: `data/gold_standard_extractions/`
- Quality report: `<Chapter>_gold_standard_quality.json`

Streamlit UI:

```bash
streamlit run tools/evidence_inspector_app.py
```

## Testing & Invariants

Fast tests (no network):

```bash
make test-gold-invariants  # or: python tools/run_invariant_tests.py
```

All tests:

```bash
make test  # or: pytest -q
```

The invariant tests assert gold‑standard requirements:
- Authors preserved; clean conclusions
- All metrics moved into `performance` blocks with proportions
- `page_range` is an object `{start, end}` everywhere
- Risk model references null when blank; definitions shaped consistently
- No stray top‑level `reference`; paths normalized to `/`

## Windows Tips

- Activate venv: `.venv\Scripts\activate`
- Use quotes around paths containing spaces
- Formatting without `make`:

```powershell
./tools/format.ps1           # format changed files
./tools/format.ps1 -All      # format key directories
```

## Troubleshooting

1) TypeError: `Responses.create() got an unexpected keyword argument 'response_format'`
   - Fixed in this repo by removing `response_format` from Responses API calls and enforcing JSON via prompts.
   - Ensure your local repo is up‑to‑date.

2) Enhancement crash: `TypeError: expected string or bytes-like object, got 'NoneType'`
   - Caused by null `interpretation` fields; fixed with null‑safe handling in enhancer.

3) Authors missing in `chapter_metadata`
   - Enhancer restores authors from original extraction when dropped. If still missing, verify the metadata pass found authors in the PDF.

4) Poor coverage / sparse output
   - Try `--model gpt-4o` as a fallback to compare results.
   - Provide the Adobe JSON via `--adobe-json` for tables.
   - Reduce passes to isolate issues, then add them back.
   - Verify PDF text extraction (corrupt or image‑only PDFs may need OCR).

5) Metrics appear as strings instead of `performance`
   - Enhancer now sweeps and normalizes stray metrics into `performance` objects with proportion units.

6) Mixed `page_range` types
   - Enhancer standardizes to object form; ensure you’re using the updated enhancer.

7) Path separators are backslashes on Windows
   - Enhancer normalizes to forward slashes in `extraction_metadata`. This avoids portability issues.

8) Rate limits / retries
   - The extractor uses a semaphore‑guarded concurrency for GPT‑5. If calls fail, it retries with exponential backoff. Re‑run the job; consider temporary `--model gpt-4o` if needed.

9) Large tables cause huge prompts
   - Consider running only `pass6_tables` initially, or skipping tables temporarily to debug other passes.

## Advanced Options

- Force Chat Completions:

```bash
python tools/gold_standard_pipeline.py --single ... --model gpt-4o
```

- Run a subset of passes via extractor directly:

```bash
python tools/production_multipass_textbook_extractor.py \
  --single "Textbooks/Chapter pdfs/Approach.pdf" \
  --adobe-json "Textbooks/Chapter json/Approach.json" \
  --passes pass0_metadata pass3_diagnostics pass6_tables
```

## Ready‑to‑Ship Checklist

- [ ] Restore chapter authors and re‑extract a clean conclusion
- [ ] Remove stray top‑level `reference`
- [ ] Unify `page_range` type across the file
- [ ] Normalize all quantitative metrics into `performance`
- [ ] Fill `risk_models.reference` and ensure `definitions` are schema‑consistent
- [ ] Validate with `make test-gold-invariants`

