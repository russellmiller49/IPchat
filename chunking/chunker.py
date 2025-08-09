#!/usr/bin/env python3
"""
Chunk trials and chapters into ~800-token passages with overlap.

Outputs JSONL to data/chunks/chunks.jsonl with fields:
  chunk_id, document_id, text, source, pages, section_path,
  table_number, figure_number, trial_signals

Usage:
  python chunking/chunker.py \
      --trials-dir data/complete_extractions \
      --chapters-dir Textbooks \
      --out data/chunks/chunks.jsonl
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
import re


def iter_json_files(root: Path) -> Iterable[Path]:
    if not root or not root.exists():
        return []
    for p in sorted(root.glob("**/*.json")):
        yield p


def tokenize(text: str) -> List[str]:
    return re.findall(r"\w+|[^\w\s]", text)


def sliding_windows(tokens: List[str], size: int, overlap: int) -> Iterable[Tuple[int, List[str]]]:
    if size <= 0:
        size = 800
    if overlap < 0:
        overlap = 120
    step = max(1, size - overlap)
    i = 0
    while i < len(tokens):
        yield i, tokens[i : i + size]
        i += step


def chunk_text(text: str, base_id: str, pages: Optional[List[int]] = None, section_path: Optional[List[str]] = None):
    toks = tokenize(text)
    for idx, window in sliding_windows(toks, size=850, overlap=120):
        yield {
            "chunk_id": f"{base_id}#{idx}",
            "document_id": base_id.split("#")[0],
            "text": " ".join(window),
            "source": "trial",
            "pages": pages or [],
            "section_path": section_path or [],
            "table_number": None,
            "figure_number": None,
            "trial_signals": {}
        }


def trial_text_blocks(data: Dict) -> Iterable[Tuple[str, str, List[int]]]:
    # Handle nested document structure from oe_final outputs
    if "document" in data:
        doc = data["document"]
        sections = doc.get("sections") or {}
    else:
        sections = data.get("sections") or {}
    
    # Prefer structured sections
    for name in ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
        texts = sections.get(name) or []
        if isinstance(texts, str):
            texts = [texts]
        for t in texts:
            if t and isinstance(t, str) and len(t.strip()) > 50:
                yield name, t, []

    # Key findings as fallback
    key_findings = data.get("key_findings") or []
    if "document" in data:
        key_findings = data.get("document", {}).get("key_findings") or key_findings
    for t in key_findings:
        if isinstance(t, str) and len(t) > 50:
            yield "key_findings", t, []


def build_trial_signals(data: Dict) -> Dict:
    signals = {}
    # Try to capture some common hints
    if "pico" in data:
        pop = data.get("pico", {}).get("population") or {}
    else:
        pop = data.get("population") or {}
    if isinstance(pop, dict):
        signals["total"] = pop.get("total")
    return signals


def process_trials(trials_dir: Path) -> Iterable[Dict]:
    for p in iter_json_files(trials_dir):
        base_id = p.stem
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        signals = build_trial_signals(data)
        for section, text, pages in trial_text_blocks(data):
            for ch in chunk_text(text, base_id=base_id, pages=pages, section_path=[section]):
                ch["source"] = "trial"
                ch["trial_signals"] = signals
                yield ch

        # ------------------------------------------------------------------
        # Additionally index adverse event rows so that incidence questions
        # such as "what percent had pneumothorax" are retrievable by the
        # vector store. We convert each adverse-event dict into a short
        # English sentence and store as its own chunk (without further
        # token splitting) to keep the numeric signal intact.
        # ------------------------------------------------------------------
        aes = data.get("adverse_events") or []
        if isinstance(aes, list):
            for idx, evt in enumerate(aes):
                try:
                    event = evt.get("event") or "adverse event"
                    int_n = evt.get("intervention_n")
                    int_pct = evt.get("intervention_percent")
                    ctrl_n = evt.get("control_n")
                    ctrl_pct = evt.get("control_percent")
                    serious = evt.get("serious")

                    parts = [f"Adverse Event: {event}."]
                    if int_n is not None:
                        parts.append(f"Intervention: {int_n} patients ({int_pct}%).")
                    if ctrl_n is not None:
                        parts.append(f"Control: {ctrl_n} patients ({ctrl_pct}%).")
                    if serious is not None:
                        parts.append("Serious." if serious else "Non-serious.")

                    text = " ".join(parts)

                    yield {
                        "chunk_id": f"{base_id}#ae{idx}",
                        "document_id": base_id,
                        "text": text,
                        "source": "trial",
                        "pages": [],
                        "section_path": ["adverse_events"],
                        "table_number": None,
                        "figure_number": None,
                        "trial_signals": signals,
                    }
                except Exception:
                    continue


def chapter_text_blocks(obj: Dict) -> Iterable[Tuple[str, str, List[int]]]:
    # Expect paragraphs in text_units or paragraphs
    units = obj.get("text_units") or obj.get("paragraphs") or []
    for u in units:
        txt = u.get("text") if isinstance(u, dict) else None
        if txt and len(txt.strip()) > 50:
            pages = u.get("pages") or ([u.get("page")] if u.get("page") else [])
            section_path = u.get("section_path") or []
            yield ",".join(section_path) if section_path else "paragraph", txt, pages


def process_chapters(chapters_dir: Path) -> Iterable[Dict]:
    for p in iter_json_files(chapters_dir):
        base_id = p.stem
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        for section, text, pages in chapter_text_blocks(data):
            for ch in chunk_text(text, base_id=base_id, pages=pages, section_path=[section]):
                ch["source"] = "chapter"
                yield ch


def main():
    import argparse
    out_default = Path("data/chunks/chunks.jsonl")
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials-dir", type=Path, default=Path("data/complete_extractions"))
    parser.add_argument("--chapters-dir", type=Path, default=Path("Textbooks"))
    parser.add_argument("--out", type=Path, default=out_default)
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with args.out.open("w", encoding="utf-8") as f:
        for ch in process_trials(args.trials_dir):
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
            n += 1
        for ch in process_chapters(args.chapters_dir):
            f.write(json.dumps(ch, ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} chunks → {args.out}")


if __name__ == "__main__":
    main()
