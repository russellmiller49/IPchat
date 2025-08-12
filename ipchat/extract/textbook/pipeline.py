import json, hashlib, time, os
from pathlib import Path
from typing import Optional, Dict, Any

from pydantic import ValidationError
from ipchat.schemas.textbook import TextbookChapter
from ipchat.adapters.io.pdf import extract_pages
from ipchat.adapters.io.adobe_extract import (
    load_adobe_extract, parse_tables, parse_figures, flatten_text_units
)
from ipchat.extract.textbook import prompts

RESEARCH_PATTERNS = ("ABSTRACT", "METHODS", "RESULTS", "DISCUSSION", "INTRODUCTION")

def looks_like_research_article(text: str) -> bool:
    up = text.upper()
    hits = sum(1 for k in RESEARCH_PATTERNS if f"\n{k}\n" in up)
    return hits >= 3

def _merge_text(pdf_pages, adobe_text: str) -> str:
    if adobe_text.strip():
        return adobe_text
    return "".join([f"\n[PAGE {p['page']}]\n{p['text']}" for p in pdf_pages])

def _call_openai(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
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

def _call_anthropic(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    # If you prefer Anthropic, implement a tools/function call here
    raise NotImplementedError("Set IPCHAT_LLM=openai or implement anthropic client.")

def _call_llm(prompt: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    prov = os.getenv("IPCHAT_LLM", "openai")
    return _call_openai(prompt, schema) if prov == "openai" else _call_anthropic(prompt, schema)

def extract_textbook(pdf_path: Path, adobe_json_path: Path, title: Optional[str]=None) -> TextbookChapter:
    pdf_pages = extract_pages(pdf_path)
    adobe = load_adobe_extract(adobe_json_path)
    ad_text = flatten_text_units(adobe)
    merged = _merge_text(pdf_pages, ad_text)

    if looks_like_research_article(merged):
        raise ValueError("Non-textbook source detected — use article extractor.")

    table_hints = parse_tables(adobe)
    truncated = merged[:100000]
    prompt = prompts.USER_TEMPLATE.format(
        title=title or pdf_path.stem,
        table_hints=json.dumps(table_hints, ensure_ascii=False),
        truncated_text=truncated
    )

    schema = TextbookChapter.model_json_schema()
    data = _call_llm(prompt, schema)

    # Attach provenance
    file_hash = hashlib.sha256(merged.encode("utf-8")).hexdigest()
    data["provenance"] = {
        "pdf_path": str(pdf_path),
        "adobe_json_path": str(adobe_json_path),
        "file_hash": hashlib.sha256(merged.encode("utf-8")).hexdigest(),
        "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        "extracted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }

    # Validate
    try:
        return TextbookChapter.model_validate(data)
    except ValidationError as e:
        Path("last_textbook_extraction_debug.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
        raise