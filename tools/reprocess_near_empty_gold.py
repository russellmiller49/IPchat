#!/usr/bin/env python3
"""
Reprocess near-empty gold-standard textbook extractions.

Detects gold-standard JSONs that look effectively empty (tiny size and
no diagnostic approaches/tables/references) and re-runs just those
chapters through the gold-standard pipeline using a fallback model
(default: gpt-4o) to improve reliability.

Usage:
  python tools/reprocess_near_empty_gold.py              # dry-run summary
  python tools/reprocess_near_empty_gold.py --run        # reprocess flagged
  python tools/reprocess_near_empty_gold.py --model gpt-4o --run

Notes:
  - Requires OPENAI_API_KEY set in your environment or .env
  - Uses Textbooks/Chapter pdfs + Chapter json to locate sources
  - Writes outputs to data/gold_standard_extractions (same location)
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
import sys


GOLD_DIR = Path("data/gold_standard_extractions")
PDF_DIR = Path("Textbooks/Chapter pdfs")
ADOBE_DIR = Path("Textbooks/Chapter json")


def is_near_empty(gold_path: Path) -> bool:
    """Heuristic to detect near-empty gold files."""
    try:
        data = json.loads(gold_path.read_text(encoding="utf-8"))
    except Exception:
        return False

    size_kb = gold_path.stat().st_size / 1024
    diag = len(data.get("diagnostic_approaches", []) or [])
    tabs = len(data.get("tables", []) or [])
    refs = len(data.get("references", []) or [])

    # Treat files with no content beyond default definitions as near-empty
    return (size_kb < 10) and (diag == 0) and (tabs == 0) and (refs == 0)


def find_near_empty() -> List[Dict]:
    """Scan gold_standard_extractions for near-empty files and collect metadata."""
    results: List[Dict] = []
    for gold_path in sorted(GOLD_DIR.glob("*_gold_standard.json")):
        if not is_near_empty(gold_path):
            continue
        stem = gold_path.stem.replace("_gold_standard", "")
        pdf_path = PDF_DIR / f"{stem}.pdf"
        adobe_path = ADOBE_DIR / f"{stem}.json"
        quality_path = gold_path.with_name(f"{gold_path.stem}_quality.json")

        # Read quick quality info if present
        score = None
        level = None
        if quality_path.exists():
            try:
                q = json.loads(quality_path.read_text(encoding="utf-8"))
                score = q.get("score")
                level = q.get("quality_level")
            except Exception:
                pass

        results.append({
            "name": stem,
            "gold": gold_path,
            "pdf": pdf_path,
            "adobe": adobe_path if adobe_path.exists() else None,
            "size_kb": round(gold_path.stat().st_size / 1024, 1),
            "quality_score": score,
            "quality_level": level,
        })
    return results


def main():
    ap = argparse.ArgumentParser(description="Reprocess near-empty gold-standard textbook chapters")
    ap.add_argument("--run", action="store_true", help="Execute reprocessing (otherwise do a dry-run summary)")
    ap.add_argument("--model", default="gpt-4o", choices=["gpt-4o", "gpt-5"], help="Model to use for re-extraction")
    ap.add_argument("--verbose", action="store_true", help="Verbose pipeline output")
    args = ap.parse_args()

    GOLD_DIR.mkdir(parents=True, exist_ok=True)

    flagged = find_near_empty()
    if not flagged:
        print("No near-empty gold-standard files detected.")
        return 0

    print("Near-empty gold-standard chapters detected:\n")
    for item in flagged:
        print(f"- {item['name']}  size={item['size_kb']}KB  quality={item['quality_level'] or 'N/A'} ({item['quality_score'] if item['quality_score'] is not None else 'N/A'})")

    if not args.run:
        print("\nAdd --run to reprocess these chapters.")
        return 0

    # Reprocess with fallback model
    print("\nReprocessing with model:", args.model)
    # Lazy import so dry-run doesn't require dependencies
    try:
        try:
            from tools.gold_standard_pipeline import GoldStandardPipeline  # type: ignore
        except Exception:
            # Add tools/ to path if running from repo root
            tools_dir = Path(__file__).parent
            sys.path.insert(0, str(tools_dir))
            from gold_standard_pipeline import GoldStandardPipeline  # type: ignore
    except Exception as e:
        print("Failed to import GoldStandardPipeline. Ensure dependencies are installed:")
        print("  pip install -r requirements.txt")
        print(f"Import error: {type(e).__name__}: {e}")
        return 1

    pipeline = GoldStandardPipeline(model=args.model, output_dir=GOLD_DIR, verbose=args.verbose)

    for i, item in enumerate(flagged, 1):
        pdf = item["pdf"]
        adobe = item["adobe"]
        title = pdf.stem
        if not pdf.exists():
            print(f"[{i}/{len(flagged)}] ❌ PDF not found: {pdf}")
            continue
        print(f"[{i}/{len(flagged)}] Processing {title} ...")
        ok, out = pipeline.process_chapter(pdf, adobe, title)
        if ok:
            print(f"   ✅ Updated: {out.name}")
        else:
            print(f"   ❌ Failed: {title}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
