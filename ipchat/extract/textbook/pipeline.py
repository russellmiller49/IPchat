import json, hashlib, time
from pathlib import Path
from typing import Optional, Dict, Any, List
from pydantic import ValidationError
from ipchat.schemas.textbook import TextbookChapter
from ipchat.adapters.io.pdf import extract_pages
from ipchat.adapters.io.adobe_extract import load_adobe_extract, parse_tables, parse_figures, flatten_text_units
from ipchat.extract.textbook import prompts

RESEARCH_PATTERNS = ("ABSTRACT", "METHODS", "RESULTS", "DISCUSSION", "CONCLUSION", "INTRODUCTION")

def looks_like_research_article(text: str) -> bool:
    # crude, conservative gate
    hits = sum(1 for k in RESEARCH_PATTERNS if f"\n{k}\n" in text.upper())
    return hits >= 3

def merge_text(pdf_pages, adobe_text: str) -> str:
    if adobe_text.strip():
        return adobe_text
    return "".join([f"\n[PAGE {p['page']}]\n{p['text']}" for p in pdf_pages])

def call_llm(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    # Provide either OpenAI or Anthropic implementation; choose via env
    import os
    prov = os.getenv("IPCHAT_LLM", "openai")
    if prov == "anthropic":
        # implement a tools/JSON schema call
        raise NotImplementedError("Add anthropic client here")
    else:
        from openai import OpenAI
        client = OpenAI()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[{"role":"system","content":prompts.SYSTEM},
                      {"role":"user","content":prompt}],
            response_format={"type":"json_schema","json_schema":{"name":"textbook_chapter","schema":schema,"strict":True}},
            temperature=0.1
        )
        return json.loads(resp.choices[0].message.content)

def extract_textbook(pdf_path: Path, adobe_json_path: Path, title: Optional[str]=None) -> TextbookChapter:
    pdf_pages = extract_pages(pdf_path)
    adobe = load_adobe_extract(adobe_json_path)
    ad_text = flatten_text_units(adobe)
    merged = merge_text(pdf_pages, ad_text)

    if looks_like_research_article(merged):
        raise ValueError("Non-textbook source detected — use article extractor.")

    # Hints for tables/figures
    table_hints = parse_tables(adobe)
    figure_hints = parse_figures(adobe)  # optionally used downstream

    trunc = merged[:100000]
    prompt = prompts.USER_TEMPLATE.format(
        title=title or pdf_path.stem,
        table_hints=json.dumps(table_hints, ensure_ascii=False),
        truncated_text=trunc
    )

    schema = TextbookChapter.model_json_schema()
    data = call_llm(prompt, schema)

    # Attach provenance
    file_hash = hashlib.sha256(merged.encode("utf-8")).hexdigest()
    data["provenance"] = {
        "pdf_path": str(pdf_path),
        "adobe_json_path": str(adobe_json_path),
        "file_hash": file_hash,
        "model": "json-schema-LLM",
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # Validate
    try:
        return TextbookChapter.model_validate(data)
    except ValidationError as e:
        Path("last_textbook_extraction_debug.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        raise