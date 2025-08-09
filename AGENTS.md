# Repository Guidelines

## Project Structure & Module Organization
- `extract_missing_data.py`: CLI to fill gaps in Adobe Extract JSONs using matching PDFs and OpenAI.
- `tools/`: utility scripts (e.g., `evidence_inspector_app.py`, `quick_api_test.py`, extractor variants).
- `schemas/`: JSON Schemas for structured outputs.
- `data/`: working files
  - `input_articles/` (Adobe JSON), `raw_pdfs/` (source PDFs)
  - `complete_extractions/` (enriched JSON), `outputs/` (structured outputs)
- `.env` / `.env.example`: configuration (e.g., `OPENAI_API_KEY`).

## Build, Test, and Development Commands
```bash
# Setup (Python 3.10+ recommended)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Verify API access
python tools/quick_api_test.py

# Run extractor
python extract_missing_data.py --limit 5 --start 0
python extract_missing_data.py --single example.json

# Inspect outputs (Streamlit UI)
streamlit run tools/evidence_inspector_app.py
```

## Coding Style & Naming Conventions
- Pythonic defaults: 4‑space indent, `snake_case` for functions/files, `CamelCase` for classes.
- Prefer type hints and docstrings for public functions.
- Use `pathlib.Path` for file paths; avoid hard‑coded absolute paths.
- Keep modules focused; place utilities under `tools/`. Name new scripts descriptively (e.g., `pdf_text_sampler.py`).

## Testing Guidelines
- Framework: pytest is recommended (not yet included). Name tests `tests/test_*.py`.
- Cover: argument parsing, directory handling, JSON merge logic (`merge_extractions`), and PDF parsing fallbacks.
- Fixtures: place small sample JSON/PDF stubs under `tests/fixtures/` (do not commit large real PDFs).
- Run: `pytest -q` (add `pytest` to your environment if you introduce tests).

## Commit & Pull Request Guidelines
- Use clear, imperative commits; Conventional Commits style is preferred (e.g., `feat: add PDF truncation detection`).
- PRs should include: concise description, linked issue (if any), sample output snippet from `data/complete_extractions/`, and notes on schema impact.
- Do not include secrets or large binaries. Keep `.env` local and document new env vars in `.env.example`.

## Security & Configuration Tips
- Required env: `OPENAI_API_KEY`. Optionally use `.env` in repo root.
- Be mindful of token usage; batch operations already pause for rate limits.
- Avoid committing files from `data/raw_pdfs/` and bulk outputs; prefer reproducible steps.
