#!/usr/bin/env python3
"""
Production Multi-Pass Textbook Extractor with Map-Reduce Architecture
Implements all correctness fixes and performance improvements
"""

import json
import os
import sys
import time
import hashlib
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
try:
    from datetime import UTC
except ImportError:
    from datetime import timezone as _tz
    UTC = _tz.utc
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Semaphore
from collections import defaultdict
import re

from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF
from tqdm import tqdm
import openpyxl

# Load environment variables
load_dotenv()

# Initialize OpenAI client
# Supports both GPT-4o (Chat API) and GPT-5 (Responses API)
# 
# IMPORTANT API Differences:
# - GPT-5 Responses API:
#   * Uses simple 'input' parameter (not messages)
#   * Returns JSON by default when prompted (no format specification needed)
#   * Does NOT accept 'temperature', 'top_p', or format parameters
#   * Based on working article extractor pattern
# - GPT-4o/GPT-5-chat (Chat API):
#   * Uses 'messages' and 'response_format'
#   * Uses 'max_tokens' and accepts temperature/top_p
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Semaphore for API rate limiting
api_semaphore = Semaphore(5)  # Max 5 concurrent API calls

# Global anti-hallucination guardrail
STRICT_GUARDRAIL = """
You must extract only what is explicitly present in the provided TEXT.
• Do NOT infer, generalize, or import facts from outside knowledge.
• If a requested field is not explicitly stated, set it to null or [] and include `present_in_source: false`.
• For every item, include `source_page` (int) and `source_excerpt` (≤30 words) copied verbatim.
• Any numeric value (percentages, thresholds, sizes, intervals) MUST include a `source_excerpt`.
• If nothing is present for a category, return an empty array for that category.
"""

# Regex for detecting numeric values that need provenance
NUMERIC_RE = re.compile(r"""
    (?:                             # Start non-capturing group
        \b[<>≤≥]?\s*\d+(?:\.\d+)?  # number with optional comparator
        \s*
        (?:%|mm³|mm3|cm³|cm3|mm²|mm2|cm²|cm2|mm|cm|m|
           mL|ml|L|HU|SUV(?:max)?|
           days?|months?|years?|y|hrs?|hours?|mins?|minutes?)\b
    |                               # OR
        \bSUV(?:max)?\s*\d+(?:\.\d+)?  # SUVmax followed by number
    |                               # OR
        \b\d+(?:\.\d+)?%            # percentage (number followed by %)
    )
""", re.I | re.X)

# Fields that are numeric by meaning even when unit/sign is omitted
NUMERIC_FIELDS = {
    'sensitivity','specificity','ppv','npv','accuracy','auc',
    'odds_ratio','risk_ratio','vdt','vdts'
}

# OCR artifacts and clinical term corrections
OCR_CORRECTIONS = {
    # Ligature fixes
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    
    # Clinical term normalization
    "PDG-PET": "FDG-PET",
    "PDG PET": "FDG-PET",
    "CTscan": "CT scan",
    "CT-scan": "CT scan",
    "Extra thoracic": "Extrathoracic",
    "extra-thoracic": "extrathoracic",
    "Calciﬁcation": "Calcification",
    "Trans-thoracic": "Transthoracic",
    "trans-thoracic": "transthoracic",
    "Pre-test": "Pretest",
    "pre-test": "pretest",
    "Post-test": "Post-test",
    "SUVmax": "SUVmax",
    "SUV max": "SUVmax",
}

# Common clinical abbreviations
CLINICAL_DEFINITIONS = {
    "pCA": "pretest probability of cancer",
    "IPN": "indeterminate pulmonary nodule",
    "SPN": "solitary pulmonary nodule",
    "VDT": "volume doubling time",
    "FDG": "fluorodeoxyglucose",
    "PET": "positron emission tomography",
    "EBUS": "endobronchial ultrasound",
    "TBNA": "transbronchial needle aspiration",
    "PPV": "positive predictive value",
    "NPV": "negative predictive value",
    "ACCP": "American College of Chest Physicians",
    "BTS": "British Thoracic Society",
}

# Define all extraction passes including metadata and figures
EXTRACTION_PASSES = {
    "pass0_metadata": {
        "name": "Chapter Metadata",
        "system_prompt": STRICT_GUARDRAIL + "\nExtract chapter metadata including title, authors, chapter number, and learning objectives.\nFocus ONLY on metadata and overview information.\nDo NOT include running headers or 'et al.' references as authors.",
        "user_prompt": """Extract metadata from this textbook chapter.

Extract:
- Chapter title
- All authors (full names ONLY from the author byline, NOT from running headers or 'et al.' references)
- Chapter number
- Learning objectives (if stated)
- Key points or chapter summary
- Book title and edition if mentioned
- Include source_page and source_excerpt for each item

IMPORTANT: Only include actual author names from the chapter's author section, not running headers like "V. Y. Aroumougame et al."

TEXT: {text}

Return JSON with keys: title, authors (array), chapter_number, learning_objectives (array), key_points (array)\nReturn only content that satisfies STRICT_GUARDRAIL."""
    },
    
    "pass1_anatomy": {
        "name": "Anatomical Structures",
        "system_prompt": STRICT_GUARDRAIL + "\nYou are an expert anatomist extracting detailed anatomical information.\nFocus ONLY on anatomical structures, descriptions, relationships, and clinical significance.",
        "user_prompt": """Extract ALL anatomical structures from this text.

For EACH structure include:
- Name
- Description (location, boundaries, appearance)
- Relationships to other structures
- Clinical significance
- Variations
- Blood supply/innervation if mentioned
- source_page (int)
- source_excerpt (≤30 words verbatim)
- present_in_source (boolean)

TEXT: {text}

Return JSON with key "anatomical_structures" containing an array.\nReturn only content that satisfies STRICT_GUARDRAIL."""
    },
    
    "pass2_procedures": {
        "name": "Clinical Procedures",
        "system_prompt": STRICT_GUARDRAIL + "\nExtract detailed procedural information including techniques and interventions.\nFocus on step-by-step instructions and critical details.\nIf no procedures are explicitly described, return empty array.",
        "user_prompt": """Extract ALL clinical procedures and techniques.

For EACH procedure include:
- Name
- Indications (all)
- Contraindications
- Equipment required
- Patient preparation
- Step-by-step instructions (if present)
- Complications (with rates if given)
- Post-procedure care
- Success rates
- source_page (int)
- source_excerpt (≤30 words verbatim)
- present_in_source (boolean)

TEXT: {text}

Return JSON with key "procedures" containing an array.\nReturn only content that satisfies STRICT_GUARDRAIL."""
    },
    
    "pass3_diagnostics": {
        "name": "Diagnostic Approaches",
        "system_prompt": STRICT_GUARDRAIL + "\nExtract diagnostic methods, criteria, and classification systems.\nInclude physical exam techniques, imaging interpretation, and scoring systems.\nDo NOT include clinical guidelines here - those belong in the guidelines pass.",
        "user_prompt": """Extract ALL diagnostic approaches and classification systems.

For EACH include:
- name
- purpose
- criteria_or_scoring (nullable)
- how_to_perform (nullable)
- interpretation (nullable)
- sensitivity (nullable, include % if given with the actual numeric value in source_excerpt)
- specificity (nullable, include % if given with the actual numeric value in source_excerpt)
- limitations (nullable)
- when_to_use (nullable)
- source_page (int)
- source_excerpt (≤30 words verbatim that MUST contain the actual numeric values if any)
- present_in_source (boolean)

NOTE: Do NOT include guidelines (ACCP, BTS, Fleischner) here - those are clinical guidelines, not diagnostic approaches.

TEXT: {text}

Return JSON with key "diagnostic_approaches" containing an array.\nReturn only content that satisfies STRICT_GUARDRAIL."""
    },
    
    "pass4_guidelines": {
        "name": "Guidelines & Algorithms",
        "system_prompt": STRICT_GUARDRAIL + "\nExtract clinical guidelines, recommendations, and algorithms.\nDo NOT fabricate recommendation grades or evidence levels. Only include them if explicitly present in the text/captions/tables.\nInclude guidelines from organizations like ACCP, BTS, Fleischner Society.",
        "user_prompt": """Extract ALL guidelines/algorithms.

For EACH include:
- title
- condition
- source_organization (nullable)
- recommendation_grade (nullable; include only if explicitly labeled)
- evidence_level (nullable; include only if explicitly labeled)
- specific_recommendations (array of strings with units/thresholds intact)
- decision_points_and_pathways (array)
- patient_population_criteria (nullable)
- contraindications_and_exceptions (nullable)
- monitoring_requirements (nullable)
- outcome_measures (nullable)
- alternative_approaches (nullable)
- source_page (int)
- source_excerpt (≤30 words)
- present_in_source (boolean)

TEXT: {text}

Return JSON with key "guidelines_algorithms" (array).\nReturn only content that satisfies STRICT_GUARDRAIL."""
    },
    
    "pass5_pharmacology": {
        "name": "Drug Information",
        "system_prompt": STRICT_GUARDRAIL + "\nExtract comprehensive drug and medication information.\nInclude dosing, contraindications, and monitoring.\nIf no drugs are explicitly mentioned, return empty array.",
        "user_prompt": """Extract ALL drug/medication information.

For EACH drug include:
- drug_name (generic and brand if present)
- drug_class (nullable)
- mechanism (nullable)
- indications (array)
- dosing (object with adult/pediatric/renal/hepatic if present)
- contraindications (array)
- side_effects (array)
- interactions (array)
- monitoring (nullable)
- source_page (int)
- source_excerpt (≤30 words verbatim)
- present_in_source (boolean)

TEXT: {text}

Return JSON with key "drugs" containing an array.\nReturn only content that satisfies STRICT_GUARDRAIL."""
    },
    
    "pass6_tables": {
        "name": "Tables & Data",
        "system_prompt": STRICT_GUARDRAIL + "\nExtract all tabular data including reference values and comparisons.\nPreserve exact values and units.\nEnsure ALL columns and ALL cell content are captured completely.",
        "user_prompt": """Extract ALL tables and structured data.

For EACH table:
- title
- headers (array - include ALL column headers including 'Characteristic' if present)
- rows (array of arrays - preserve COMPLETE cell content including all text)
- units (nullable)
- footnotes (array if present)
- clinical_interpretation (nullable)
- source_page (int)
- source_excerpt (first row or caption, ≤30 words)
- content_provenance ('xlsx' or 'pdf_text')

IMPORTANT: For tables with characteristics, include the characteristic name as the first column.
Example: ['Margin', 'Smooth, well defined', 'Irregular, Lobulated, Spiculated, Sunburst or corona radiata appearance']

XLSX content if available:
{xlsx_content}

TEXT: {text}

Return JSON with key "tables" containing an array.\nReturn only content that satisfies STRICT_GUARDRAIL."""
    },
    
    "pass7_figures": {
        "name": "Figures & Images",
        "system_prompt": STRICT_GUARDRAIL + "\nExtract information about all figures, diagrams, and images.\nInclude captions and clinical significance.\nDo NOT include mere textual references to figures - only actual figure captions and descriptions.",
        "user_prompt": """Extract ALL figures and images mentioned.

For EACH figure:
- figure_id (e.g., "Figure 1" or "Fig. 2")
- title (nullable)
- caption (verbatim if present)
- type (photo/diagram/graph/algorithm if indicated)
- key_findings (array)
- clinical_significance (nullable)
- source_page (int)
- source_excerpt (actual figure caption, NOT a textual reference, ≤30 words)
- present_in_source (boolean)

IMPORTANT: Only include actual figures with captions, not textual references like "see Fig. 1" within paragraphs.

TEXT: {text}

Return JSON with key "figures" containing an array.\nReturn only content that satisfies STRICT_GUARDRAIL."""
    },
    
    "pass8_education": {
        "name": "Educational Content",
        "system_prompt": STRICT_GUARDRAIL + "\nExtract educational elements including pearls, definitions, and cases.",
        "user_prompt": """Extract educational content:

1. Clinical pearls and tips
2. Important definitions
3. Case examples
4. Common pitfalls
5. Practice recommendations
6. Board exam facts

For each include:
- content (verbatim)
- source_page (int)
- source_excerpt (≤30 words)
- present_in_source (boolean)

TEXT: {text}

Return JSON with keys: clinical_pearls, definitions, case_examples, pitfalls, practice_recommendations, board_exam_facts (all arrays).\nReturn only content that satisfies STRICT_GUARDRAIL."""
    },
    
    "pass9_references": {
        "name": "References",
        "system_prompt": STRICT_GUARDRAIL + "\nExtract all references and citations.",
        "user_prompt": """Extract ALL references:

For EACH:
- citation (complete, verbatim)
- authors (array)
- year (nullable)
- journal (nullable)
- doi_pmid (nullable)
- study_type (nullable)
- key_findings (nullable)
- source_page (int)
- present_in_source (boolean)

TEXT: {text}

Return JSON with key "references" containing an array.\nReturn only content that satisfies STRICT_GUARDRAIL."""
    }
}


def build_schema(pass_key: str) -> Tuple[Optional[str], Optional[dict]]:
    """
    Build JSON Schema for Structured Outputs based on pass type
    Returns (schema_name, schema_dict) or (None, None) if no schema for this pass
    """
    if pass_key == "pass0_metadata":
        return (
            "ChapterMetadata",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "title": {"type": "string"},
                    "authors": {"type": "array", "items": {"type": "string"}},
                    "chapter_number": {"type": ["integer", "string", "null"]},
                    "learning_objectives": {"type": "array", "items": {"type": "string"}},
                    "key_points": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["title", "authors"]
            }
        )

    if pass_key == "pass3_diagnostics":
        return (
            "DiagnosticsSchema",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "diagnostic_approaches": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "name": {"type": "string"},
                                "purpose": {"type": ["string", "null"]},
                                "criteria_or_scoring": {"type": ["array", "string", "null"]},
                                "how_to_perform": {"type": ["string", "null"]},
                                "interpretation": {"type": ["string", "null"]},
                                "sensitivity": {"type": ["number", "string", "null"]},
                                "specificity": {"type": ["number", "string", "null"]},
                                "ppv": {"type": ["number", "string", "null"]},
                                "npv": {"type": ["number", "string", "null"]},
                                "accuracy": {"type": ["number", "string", "null"]},
                                "limitations": {"type": ["string", "array", "null"]},
                                "when_to_use": {"type": ["string", "null"]},
                                "source_page": {"type": "integer"},
                                "source_excerpt": {"type": "string", "maxLength": 240},
                                "present_in_source": {"type": "boolean"}
                            },
                            "required": ["name", "source_page", "source_excerpt", "present_in_source"]
                        }
                    }
                },
                "required": ["diagnostic_approaches"]
            }
        )

    if pass_key == "pass6_tables":
        return (
            "TablesSchema",
            {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tables": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "properties": {
                                "title": {"type": ["string", "null"]},
                                "headers": {"type": "array", "items": {"type": "string"}},
                                "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                                "units": {"type": ["string", "null"]},
                                "footnotes": {"type": ["array", "null"], "items": {"type": "string"}},
                                "clinical_interpretation": {"type": ["string", "null"]},
                                "source_page": {"type": "integer"},
                                "source_excerpt": {"type": "string", "maxLength": 240},
                                "content_provenance": {"type": "string", "enum": ["xlsx", "pdf_text"]}
                            },
                            "required": ["headers", "rows", "source_page", "source_excerpt", "content_provenance"]
                        }
                    }
                },
                "required": ["tables"]
            }
        )

    return (None, None)


def extract_text_from_pdf(pdf_path: Path) -> List[Tuple[int, str]]:
    """Extract text from PDF with page markers and figure captions"""
    try:
        doc = fitz.open(str(pdf_path))
        pages = []
        for i in range(len(doc)):
            page = doc[i]
            text = page.get_text()
            
            # Heuristic: pull likely captions/labels
            blocks = page.get_text("blocks")
            fig_lines = []
            for (_x0, _y0, _x1, _y1, btxt, _bid, _b) in blocks:
                if re.search(r"\b(Fig\.?|Figure)\s*\d+", btxt, re.I):
                    fig_lines.append(btxt.strip())
            
            all_text = f"[PAGE {i+1}]\n{(text or '')}\n" + ("\n".join(fig_lines) if fig_lines else "")
            pages.append((i+1, all_text))
        doc.close()
        return pages
    except Exception as e:
        print(f"Error extracting PDF: {e}")
        return []


def parse_xlsx_table(xlsx_path: Path) -> Dict[str, Any]:
    """Parse XLSX file to extract table content"""
    try:
        wb = openpyxl.load_workbook(xlsx_path, read_only=True)
        sheet = wb.active
        
        headers = []
        rows = []
        
        for i, row in enumerate(sheet.iter_rows(values_only=True)):
            if i == 0:
                headers = [str(cell) if cell else "" for cell in row]
            else:
                rows.append([str(cell) if cell else "" for cell in row])
        
        wb.close()
        
        return {
            "headers": headers,
            "rows": rows
        }
    except Exception as e:
        print(f"Error parsing XLSX {xlsx_path}: {e}")
        return {"headers": [], "rows": []}


def extract_tables_from_adobe_json(json_path: Path) -> Dict[str, Any]:
    """Extract table content from Adobe JSON including XLSX data"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tables = {}
        for table in data.get('tables', []):
            table_id = str(table.get('ObjectID', ''))
            xlsx_path = (table.get('filePaths') or [None])[0]
            
            table_data = {
                'table_id': table_id,
                'page': table.get('Page'),
                'bounds': table.get('Bounds'),
                'xlsx_path': xlsx_path,
                'content': {"headers": [], "rows": []},
                'content_provenance': 'pdf_text'
            }
            
            # Try to resolve XLSX path more flexibly
            if xlsx_path and not Path(xlsx_path).exists():
                alt = Path(json_path).parent / Path(xlsx_path).name
                if alt.exists():
                    xlsx_path = str(alt)
            
            # Parse XLSX if available
            if xlsx_path and Path(xlsx_path).exists():
                table_data['content'] = parse_xlsx_table(Path(xlsx_path))
                table_data['content_provenance'] = 'xlsx'
            
            tables[table_id] = table_data
        
        return tables
    except Exception as e:
        print(f"Error extracting tables from Adobe JSON: {e}")
        return {}


def chunk_pages(pages: List[Tuple[int, str]], chunk_size: int = 6, 
                max_tokens: int = 100000) -> List[Dict[str, Any]]:
    """Chunk pages respecting token limits"""
    chunks = []
    current_chunk = {
        "pages": [],
        "text": "",
        "page_start": 0,
        "page_end": 0,
        "token_estimate": 0
    }
    
    for page_num, page_text in pages:
        # Estimate tokens (rough: 1 token per 4 chars)
        page_tokens = len(page_text) // 4
        
        # Check if adding this page would exceed limits
        if (len(current_chunk["pages"]) >= chunk_size or 
            current_chunk["token_estimate"] + page_tokens > max_tokens):
            
            if current_chunk["pages"]:
                chunks.append(current_chunk)
            
            current_chunk = {
                "pages": [],
                "text": "",
                "page_start": page_num,
                "page_end": page_num,
                "token_estimate": 0
            }
        
        # Add page to current chunk
        if not current_chunk["pages"]:
            current_chunk["page_start"] = page_num
        
        current_chunk["pages"].append(page_num)
        current_chunk["text"] += page_text + "\n"
        current_chunk["page_end"] = page_num
        current_chunk["token_estimate"] += page_tokens
    
    # Add final chunk
    if current_chunk["pages"]:
        chunks.append(current_chunk)
    
    return chunks


def perform_extraction_pass_with_retry(
    text: str,
    pass_config: Dict,
    xlsx_content: Optional[str] = None,
    model: str = "gpt-4o",
    max_retries: int = 3,
    schema_name: Optional[str] = None,
    schema: Optional[dict] = None,
    pass_name: str = "unknown",
    chunk_label: str = ""
) -> Dict[str, Any]:
    """
    Perform extraction pass with exponential backoff retry
    Supports both GPT-4 (Chat API) and GPT-5 (Responses API)
    
    IMPORTANT: GPT-5 via Responses API does NOT accept temperature or top_p parameters
    """
    
    # Inject anti-hallucination guardrail
    system_content = STRICT_GUARDRAIL + "\n" + pass_config["system_prompt"]
    
    # Prepare prompt with guardrail reminder
    user_prompt = pass_config["user_prompt"].format(
        text=text,
        xlsx_content=json.dumps(xlsx_content) if xlsx_content else ""
    ) + "\n\nReturn only content that satisfies STRICT_GUARDRAIL."
    
    for attempt in range(max_retries):
        try:
            with api_semaphore:
                # Decide which API to use based on model name
                use_responses = (
                    model.lower().startswith("gpt-5") and 
                    not model.lower().startswith("gpt-5-chat")
                )
                
                if use_responses:
                    # --- GPT-5 via Responses API ---
                    # Build a single combined prompt and enforce JSON via instructions.
                    json_enforcer = "Return one valid JSON object only. Do not include any prose outside the JSON."
                    if schema:
                        # Embed a compact schema hint to steer structure for SDKs that don't support response_format here
                        try:
                            schema_snippet = json.dumps(schema)
                            schema_hint = (
                                f"\nStrictly match this JSON Schema (no extra keys): {schema_snippet}"
                            )
                        except Exception:
                            schema_hint = ""
                    else:
                        schema_hint = ""

                    combined_prompt = f"{system_content}\n\n{user_prompt}\n\n{json_enforcer}{schema_hint}"

                    # Use safe labels instead of undefined variables
                    if chunk_label:
                        print(f"\n  🚀 Calling GPT-5 Responses API ({chunk_label}, pass {pass_name})")
                    else:
                        print(f"\n  🚀 Calling GPT-5 Responses API (pass {pass_name})")
                    print(f"     Model: {model}")
                    print(f"     Input length: {len(combined_prompt)} chars")

                    import time
                    start_time = time.time()

                    # GPT-5 Responses API (do not pass response_format — not supported in some SDK versions)
                    resp = client.responses.create(
                        model=model,
                        input=combined_prompt
                    )
                    
                    elapsed = time.time() - start_time
                    print(f"  ✅ GPT-5 response received in {elapsed:.1f} seconds")
                    
                    # Consolidate text from Responses API (handle different response structures)
                    content = getattr(resp, "output_text", None)
                    if not content and hasattr(resp, "output"):
                        try:
                            # Some SDKs expose a list of parts
                            content = "".join(
                                p.get("text", "") if isinstance(p, dict) else str(p)
                                for p in (resp.output or [])
                            )
                        except Exception:
                            content = None
                    if not content:
                        content = str(resp)
                    
                else:
                    # --- Chat Completions path (gpt-4o / gpt-5-chat variants) ---
                    # Build response format for Chat API
                    if schema:
                        # Chat API also supports JSON Schema
                        response_format = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": schema_name or "ExtractionSchema",
                                "schema": schema
                            }
                        }
                    else:
                        response_format = {"type": "json_object"}
                    
                    resp = client.chat.completions.create(
                        model=model,
                        messages=[
                            {"role": "system", "content": system_content},
                            {"role": "user", "content": user_prompt}
                        ],
                        response_format=response_format,
                        max_tokens=4096,    # Chat API uses max_tokens
                        temperature=0.0,    # OK for Chat API
                        top_p=1.0          # OK for Chat API
                    )
                    content = resp.choices[0].message.content
                
                # Remove code fences and clamp to first JSON object if needed
                if content and content.strip().startswith("```"):
                    stripped = content.strip().strip("`")
                    content = stripped[4:].lstrip() if stripped.lower().startswith("json") else stripped
                
                # Extract JSON even if there's extra text
                if content and not content.strip().startswith("{"):
                    import re
                    m = re.search(r'\{.*\}', content, re.S)
                    if m:
                        content = m.group(0)
                
                return json.loads(content or "{}")
        
        except Exception as e:
            wait_time = (2 ** attempt)  # Exponential backoff: 1, 2, 4 seconds
            print(f"\n  ⚠️ API Error on attempt {attempt + 1}/{max_retries}:")
            print(f"     Error type: {type(e).__name__}")
            print(f"     Error message: {str(e)[:200]}")
            print(f"     Retrying in {wait_time} seconds...")
            time.sleep(wait_time)
    
    print(f"  Failed after {max_retries} retries")
    return {}


def extract_chunk_concurrent(
    chunk: Dict[str, Any],
    passes: List[str],
    table_data: Dict[str, Any],
    model: str = "gpt-4o"  # Using GPT-4o (most reliable advanced model)
) -> Dict[str, Any]:
    """Extract from a single chunk using concurrent passes"""
    
    results = {}
    
    # Lower concurrency for GPT-5 to avoid rate limits
    max_workers = 2 if model.lower().startswith("gpt-5") else 3
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        
        for pass_key in passes:
            if pass_key not in EXTRACTION_PASSES:
                continue
            
            pass_config = EXTRACTION_PASSES[pass_key]
            
            # Add XLSX content for tables pass
            xlsx_content = None
            if "table" in pass_key.lower() and table_data:
                # Get tables for pages in this chunk with provenance
                chunk_tables = {}
                for table_id, data in table_data.items():
                    if data['page'] in chunk['pages']:
                        chunk_tables[table_id] = {
                            'content': data['content'],
                            'content_provenance': data.get('content_provenance', 'pdf_text'),
                            'source_page': data['page']
                        }
                xlsx_content = chunk_tables
            
            # Get schema for structured outputs if available
            schema_name, schema = build_schema(pass_key)
            
            # Create chunk label for debugging
            chunk_label = f"pages {chunk['page_start']}-{chunk['page_end']}"
            
            future = executor.submit(
                perform_extraction_pass_with_retry,
                chunk['text'],
                pass_config,
                xlsx_content=xlsx_content,
                model=model,
                schema_name=schema_name,
                schema=schema,
                pass_name=pass_key,
                chunk_label=chunk_label
            )
            futures[future] = pass_key
        
        # Collect results
        for future in as_completed(futures):
            pass_key = futures[future]
            try:
                result = future.result()
                # Add page range to items if not present
                for key, items in result.items():
                    if isinstance(items, list):
                        for item in items:
                            if isinstance(item, dict) and 'page' not in item:
                                item['page_range'] = f"{chunk['page_start']}-{chunk['page_end']}"
                
                results[pass_key] = result
            except Exception as e:
                print(f"  Error in pass {pass_key}: {e}")
                results[pass_key] = {}
    
    return results


def normalize_and_deduplicate(items: List[Dict], key_field: str = "name") -> List[Dict]:
    """Normalize whitespace and deduplicate items by key and page"""
    seen = set()
    unique_items = []
    
    for item in items:
        # Skip non-dict items
        if not isinstance(item, dict):
            continue
            
        # Normalize whitespace in all string fields
        for field, value in item.items():
            if isinstance(value, str):
                item[field] = " ".join(value.split())
        
        # Create dedup key (case/space tolerant)
        # Normalize multiple spaces to single space
        key = " ".join(str(item.get(key_field, "")).split()).lower()
        page = item.get("page", item.get("page_range", ""))
        dedup_key = f"{key}|{page}"
        
        if dedup_key not in seen:
            seen.add(dedup_key)
            unique_items.append(item)
    
    return unique_items


def merge_chunk_results(all_chunk_results: List[Dict]) -> Dict[str, Any]:
    """Merge results from all chunks with deduplication"""
    
    merged = {}  # Use regular dict instead of defaultdict
    
    # Collect all items by category
    for chunk_results in all_chunk_results:
        for pass_key, pass_results in chunk_results.items():
            for key, items in pass_results.items():
                if isinstance(items, list):
                    if key not in merged:
                        merged[key] = []
                    merged[key].extend(items)
                elif isinstance(items, dict):
                    # For dict items (like metadata), just store the first non-empty one
                    if key not in merged or not merged[key]:
                        merged[key] = items
                else:
                    # For simple values (strings, numbers), store first non-empty
                    if key not in merged or not merged[key]:
                        merged[key] = items
    
    # Deduplicate each category
    final = {}
    for key, items in merged.items():
        if isinstance(items, list) and items:
            # Determine key field for deduplication
            sample = items[0]
            if 'name' in sample:
                key_field = 'name'
            elif 'title' in sample:
                key_field = 'title'
            elif 'guideline' in sample:
                key_field = 'guideline'
            elif 'drug_name' in sample:
                key_field = 'drug_name'
            elif 'term' in sample:
                key_field = 'term'
            else:
                key_field = 'name'  # default
            
            final[key] = normalize_and_deduplicate(items, key_field)
        else:
            final[key] = items
    
    return final


def clean_ocr_artifacts(text: str) -> str:
    """Clean OCR artifacts and normalize clinical terms"""
    if not text:
        return ""  # Return empty string instead of None/falsy value
    
    # Convert to string if not already
    text = str(text)
    
    for artifact, correction in OCR_CORRECTIONS.items():
        text = text.replace(artifact, correction)
    
    # Additional cleanup
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    return text


def normalize_numeric_value(value: Any, field_name: str = "") -> Dict[str, Any]:
    """Normalize numeric values to consistent format"""
    if value is None:
        return None
    
    # If already normalized
    if isinstance(value, dict) and "value" in value:
        return value
    
    # Handle string percentages
    if isinstance(value, str):
        value = value.strip()
        
        # Check for percentage
        percent_match = re.match(r'^(\d+(?:\.\d+)?)\s*%?$', value)
        if percent_match:
            num_val = float(percent_match.group(1))
            if field_name in ["sensitivity", "specificity", "ppv", "npv", "accuracy"]:
                # Convert to proportion (0-1) for these fields
                return {"value": num_val / 100.0, "unit": "proportion"}
            else:
                return {"value": num_val, "unit": "%"}
        
        # Check for other numeric patterns
        num_match = re.match(r'^[<>≤≥]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|mL|L|mg|g)?$', value)
        if num_match:
            return {
                "value": float(num_match.group(1)),
                "unit": num_match.group(2) or "unitless"
            }
    
    # Handle plain numbers
    if isinstance(value, (int, float)):
        if field_name in ["sensitivity", "specificity", "ppv", "npv", "accuracy"]:
            # Assume these are percentages if > 1
            if value > 1:
                return {"value": value / 100.0, "unit": "proportion"}
            else:
                return {"value": value, "unit": "proportion"}
        else:
            return {"value": value, "unit": "unitless"}
    
    return value


def parse_criteria_string(criteria: Any) -> List[str]:
    """Convert comma-separated criteria strings to arrays"""
    if criteria is None:
        return []
    
    if isinstance(criteria, list):
        return [clean_ocr_artifacts(str(item)) for item in criteria]
    
    if isinstance(criteria, str):
        # Split by common delimiters
        items = re.split(r'[,;•·]\s*', criteria)
        # Clean each item
        items = [clean_ocr_artifacts(item.strip()) for item in items if item.strip()]
        return items
    
    return [str(criteria)]


def normalize_page_range_dict(page_range: Any) -> Dict[str, int]:
    """Normalize page ranges to consistent format"""
    if isinstance(page_range, dict) and "start" in page_range:
        return page_range
    
    if isinstance(page_range, str):
        match = re.match(r'(\d+)(?:\s*[-–]\s*(\d+))?', page_range)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            return {"start": start, "end": end}
    
    if isinstance(page_range, (int, float)):
        page = int(page_range)
        return {"start": page, "end": page}
    
    return None


def calculate_actual_page_range(data: Dict) -> Dict[str, int]:
    """Calculate the actual page range from all extracted content"""
    all_pages = []
    
    def extract_pages(obj: Any, pages: List[int]):
        """Recursively extract page numbers"""
        if isinstance(obj, dict):
            if "source_page" in obj and obj["source_page"]:
                pages.append(int(obj["source_page"]))
            if "page" in obj and obj["page"]:
                pages.append(int(obj["page"]))
            if "page_range" in obj and obj["page_range"]:
                pr = normalize_page_range_dict(obj["page_range"])
                if pr:
                    pages.extend(range(pr["start"], pr["end"] + 1))
            
            for value in obj.values():
                extract_pages(value, pages)
        elif isinstance(obj, list):
            for item in obj:
                extract_pages(item, pages)
    
    extract_pages(data, all_pages)
    
    if all_pages:
        return {"start": min(all_pages), "end": max(all_pages)}
    else:
        return {"start": 1, "end": 1}


def post_process_cleanup(extracted_data: Dict) -> None:
    """Post-process cleanup to fix common extraction issues"""
    
    # Clean up authors - remove running headers and 'et al.' entries
    if 'chapter_metadata' in extracted_data:
        meta = extracted_data['chapter_metadata']
        
        # Clean title
        if 'title' in meta:
            if isinstance(meta['title'], dict):
                meta['title']['value'] = clean_ocr_artifacts(meta['title'].get('value', ''))
            else:
                meta['title'] = clean_ocr_artifacts(str(meta['title']))
        
        # Fix page range to span all content
        actual_range = calculate_actual_page_range(extracted_data)
        meta['page_range'] = actual_range
        
        if 'authors' in meta and isinstance(meta['authors'], list):
            cleaned_authors = []
            for author in meta['authors']:
                # Skip entries containing 'et al.' or that are too short
                if isinstance(author, dict):
                    name = clean_ocr_artifacts(author.get('name', ''))
                    author['name'] = name
                elif isinstance(author, str):
                    name = clean_ocr_artifacts(author)
                    author = {'name': name}
                else:
                    continue
                    
                if 'et al.' not in name and len(name) > 5 and '.' not in name[:3]:
                    cleaned_authors.append(author)
            meta['authors'] = cleaned_authors
    
    # Remove duplicate figures (keep only actual figure captions, not references)
    if 'figures' in extracted_data:
        seen_figures = {}
        cleaned_figures = []
        for fig in extracted_data['figures']:
            fig_id = fig.get('figure_id', '')
            caption = fig.get('caption', '')
            page = fig.get('source_page', 0)
            
            # Skip if this looks like a reference rather than actual figure
            if caption and not caption.startswith('In patients with'):
                if fig_id not in seen_figures or page > seen_figures[fig_id]:
                    seen_figures[fig_id] = page
                    cleaned_figures.append(fig)
        extracted_data['figures'] = cleaned_figures
    
    # Enhance diagnostic approaches
    if 'diagnostic_approaches' in extracted_data:
        enhanced_approaches = []
        for approach in extracted_data['diagnostic_approaches']:
            # Clean text fields
            for field in ['name', 'purpose', 'how_to_perform', 'interpretation', 'when_to_use']:
                if field in approach and approach[field]:
                    approach[field] = clean_ocr_artifacts(approach[field])
            
            # Parse criteria as array
            if 'criteria_or_scoring' in approach:
                approach['criteria_or_scoring'] = parse_criteria_string(approach['criteria_or_scoring'])
            
            # Normalize performance metrics
            performance = {}
            for metric in ['sensitivity', 'specificity', 'ppv', 'npv', 'accuracy']:
                if metric in approach and approach[metric] is not None:
                    normalized = normalize_numeric_value(approach[metric], metric)
                    if normalized:
                        performance[metric] = normalized
                        del approach[metric]
            
            if performance:
                approach['performance'] = performance
            
            # Parse limitations as array
            if 'limitations' in approach and isinstance(approach['limitations'], str):
                approach['limitations'] = parse_criteria_string(approach['limitations'])
            
            enhanced_approaches.append(approach)
        
        extracted_data['diagnostic_approaches'] = enhanced_approaches
    
    # Move guidelines from diagnostic_approaches to clinical_guidelines
    if 'diagnostic_approaches' in extracted_data:
        guidelines_keywords = ['Guidelines', 'ACCP', 'BTS', 'Fleischner', 'Society']
        actual_diagnostics = []
        moved_guidelines = []
        
        for item in extracted_data['diagnostic_approaches']:
            name = item.get('name', '')
            if any(keyword in name for keyword in guidelines_keywords):
                moved_guidelines.append(item)
            else:
                actual_diagnostics.append(item)
        
        extracted_data['diagnostic_approaches'] = actual_diagnostics
        
        # Add moved items to clinical_guidelines if not already there
        if moved_guidelines:
            existing = extracted_data.get('clinical_guidelines', [])
            existing_names = {g.get('title', '') for g in existing}
            for guide in moved_guidelines:
                # Convert diagnostic format to guideline format
                guide_name = guide.get('name', '')
                if guide_name not in existing_names:
                    existing.append({
                        'title': guide_name,
                        'condition': guide.get('purpose', ''),
                        'source_organization': guide_name.split(' Guidelines')[0] if 'Guidelines' in guide_name else None,
                        'source_page': guide.get('source_page'),
                        'source_excerpt': guide.get('source_excerpt'),
                        'present_in_source': guide.get('present_in_source', True)
                    })
            extracted_data['clinical_guidelines'] = existing
    
    # Enhance clinical guidelines
    if 'clinical_guidelines' in extracted_data:
        for guideline in extracted_data['clinical_guidelines']:
            # Clean text fields
            for field in ['title', 'condition', 'source_organization']:
                if field in guideline and guideline[field]:
                    guideline[field] = clean_ocr_artifacts(guideline[field])
            
            # Parse recommendations as array
            if 'specific_recommendations' in guideline:
                if isinstance(guideline['specific_recommendations'], str):
                    guideline['specific_recommendations'] = parse_criteria_string(
                        guideline['specific_recommendations']
                    )
                elif isinstance(guideline['specific_recommendations'], list):
                    guideline['specific_recommendations'] = [
                        clean_ocr_artifacts(rec) for rec in guideline['specific_recommendations']
                    ]
        
        # Deduplicate guidelines
        seen = {}
        deduped = []
        for guideline in extracted_data['clinical_guidelines']:
            title = guideline.get('title', '')
            org = guideline.get('source_organization', '')
            key = f"{title}|{org}".lower()
            
            if key not in seen or len(str(guideline)) > len(str(seen[key])):
                seen[key] = guideline
                if key in seen:
                    # Replace existing with more complete version
                    deduped = [g for g in deduped if f"{g.get('title', '')}|{g.get('source_organization', '')}".lower() != key]
                deduped.append(guideline)
        
        extracted_data['clinical_guidelines'] = deduped
    
    # Enhance tables with object representation
    if 'tables' in extracted_data:
        for table in extracted_data['tables']:
            # Clean title
            if 'title' in table:
                table['title'] = clean_ocr_artifacts(table['title'])
            
            # Add object representation if we have headers and rows
            if 'headers' in table and 'rows' in table:
                headers = table['headers']
                rows = table['rows']
                
                # Create object representation
                table['data_objects'] = []
                for row in rows:
                    if len(row) == len(headers):
                        obj = {}
                        for i, header in enumerate(headers):
                            # Safely clean header, handling None/empty values
                            cleaned = clean_ocr_artifacts(str(header) if header else "")
                            clean_header = cleaned.lower().replace(' ', '_') if cleaned else f"column_{i}"
                            value = row[i] if i < len(row) else None
                            
                            # Try to normalize numeric values in tables
                            if any(term in clean_header for term in ['sensitivity', 'specificity', 'ppv', 'npv']):
                                normalized = normalize_numeric_value(value, clean_header)
                                obj[clean_header] = normalized if normalized else value
                            else:
                                obj[clean_header] = clean_ocr_artifacts(str(value)) if value else None
                        
                        table['data_objects'].append(obj)
    
    # Clean clinical pearls and definitions
    if 'clinical_pearls' in extracted_data:
        for pearl in extracted_data['clinical_pearls']:
            if 'content' in pearl:
                pearl['content'] = clean_ocr_artifacts(pearl['content'])
    
    if 'definitions' in extracted_data:
        for definition in extracted_data['definitions']:
            if 'term' in definition:
                definition['term'] = clean_ocr_artifacts(definition['term'])
            if 'definition' in definition:
                definition['definition'] = clean_ocr_artifacts(definition['definition'])
    
    # Add common clinical definitions if not present
    if 'definitions' not in extracted_data or not extracted_data['definitions']:
        extracted_data['definitions'] = []
    
    existing_terms = {d.get('term', '').upper() for d in extracted_data['definitions']}
    for abbrev, full_name in CLINICAL_DEFINITIONS.items():
        if abbrev not in existing_terms:
            extracted_data['definitions'].append({
                'term': abbrev,
                'definition': full_name,
                'present_in_source': False,
                'added_by': 'enhancer'
            })


def enforce_provenance(extracted_data: Dict) -> None:
    """Enforce provenance for items with numeric values"""
    def needs_prov(item: Dict) -> bool:
        text_blob = json.dumps(item)
        return bool(NUMERIC_RE.search(text_blob))
    
    for cat in [
        "diagnostic_approaches", "clinical_guidelines", "treatment_algorithms",
        "tables", "figures", "clinical_pearls", "definitions"
    ]:
        items = extracted_data.get(cat, [])
        for item in items:
            needs_because_field = any(k in item for k in NUMERIC_FIELDS)
            if needs_prov(item) or needs_because_field:
                if not item.get("source_page") or not item.get("source_excerpt"):
                    item.setdefault("_errors", []).append("missing_provenance_for_numeric")
            
            # cap excerpt length at 30 words (as required in your guardrail)
            if item.get("source_excerpt") and len(item.get("source_excerpt").split()) > 30:
                item.setdefault("_errors", []).append("source_excerpt_too_long")
            
            # fill source_page from page_range if available
            if not item.get("source_page") and item.get("page_range"):
                try:
                    item["source_page"] = int(str(item["page_range"]).split("-")[0])
                except Exception:
                    pass


def quality_audit(extracted_data: Dict) -> Dict[str, List[str]]:
    """Enhanced audit for quality issues including provenance and hallucination checks"""
    issues = defaultdict(list)
    
    # Check guidelines for unsourced grades/levels
    for g in extracted_data.get('clinical_guidelines', []):
        if (g.get('recommendation_grade') or g.get('evidence_level')) and not g.get('present_in_source'):
            issues['guidelines'].append(f"Unsourced grade/level: {g.get('title','(unknown)')}")
        if re.search(NUMERIC_RE, json.dumps(g)) and not g.get('source_excerpt'):
            issues['guidelines'].append(f"Missing source excerpt for numerics: {g.get('title','(unknown)')}")
    
    # Check diagnostics for numeric provenance
    for a in extracted_data.get('diagnostic_approaches', []):
        if re.search(NUMERIC_RE, json.dumps(a)) and not a.get('source_excerpt'):
            issues['diagnostics'].append(f"Missing source excerpt for numerics: {a.get('name','(unknown)')}")
    
    # Procedures should rarely exist for this chapter type
    for p in extracted_data.get('clinical_procedures', []):
        if p.get('steps'):
            issues['procedures'].append(f"Likely hallucinated procedure with steps: {p.get('name','(unknown)')}")
    
    # Check tables
    for t in extracted_data.get('tables', []):
        if 'headers' in t and (not t.get('footnotes') and 'footnotes' in t):
            issues['tables'].append(f"Check footnotes presence: {t.get('title','(unknown)')}")
        if re.search(NUMERIC_RE, json.dumps(t)) and not t.get('source_page'):
            issues['tables'].append(f"Table missing page: {t.get('title','(unknown)')}")
    
    # Check drugs for missing dosing
    for drug in extracted_data.get('drug_information', []):
        if not drug.get('dosing'):
            issues['drugs'].append(f"Missing dosing: {drug.get('drug_name', 'Unknown')}")
    
    # Check all items for missing page numbers
    for category in ['anatomical_content', 'clinical_procedures', 'diagnostic_approaches', 
                     'clinical_guidelines', 'treatment_algorithms', 'drug_information', 'tables', 'figures']:
        if category == 'anatomical_content':
            items = extracted_data.get('anatomical_content', {}).get('structures', [])
        else:
            items = extracted_data.get(category, [])
        for item in items:
            if not item.get('source_page') and not item.get('page_range'):
                name = item.get('name') or item.get('title') or item.get('drug_name', 'Unknown')
                issues['missing_pages'].append(f"{category}: {name}")
    
    return dict(issues)


def extract_multipass_production(
    pdf_path: Path,
    adobe_json_path: Optional[Path] = None,
    chapter_title: Optional[str] = None,
    model: str = None,
    passes_to_run: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Production multi-pass extraction with map-reduce architecture"""
    
    if not model:
        # Default to environment variable or GPT-5 for best performance
        # Falls back to GPT-4o if GPT-5 not available
        model = os.getenv("OPENAI_MODEL", "gpt-5")
    
    # Extract pages with markers
    print(f"📖 Extracting pages from PDF...")
    pages = extract_text_from_pdf(pdf_path)
    if not pages:
        raise ValueError(f"No text extracted from {pdf_path}")
    
    # Extract and parse table data
    table_data = {}
    if adobe_json_path and adobe_json_path.exists():
        print(f"📊 Extracting table data from Adobe JSON...")
        table_data = extract_tables_from_adobe_json(adobe_json_path)
        print(f"   Found {len(table_data)} tables")
    
    # Chunk pages
    print(f"📄 Chunking {len(pages)} pages...")
    chunks = chunk_pages(pages)
    print(f"   Created {len(chunks)} chunks")
    
    # Use provided passes or conservative defaults (skip procedures and pharmacology by default)
    if passes_to_run:
        passes = passes_to_run
    else:
        # Conservative default: skip procedures and pharmacology unless explicitly requested
        # Include references by default for completeness
        passes = [
            "pass0_metadata", "pass3_diagnostics", "pass4_guidelines",
            "pass6_tables", "pass7_figures", "pass8_education", "pass9_references"
        ]
    
    print(f"🔄 Running {len(passes)} extraction passes on {len(chunks)} chunks...")
    
    # Process chunks
    all_chunk_results = []
    
    for i, chunk in enumerate(tqdm(chunks, desc="Processing chunks")):
        chunk_results = extract_chunk_concurrent(chunk, passes, table_data, model)
        all_chunk_results.append(chunk_results)
    
    # Merge results
    print(f"📋 Merging results from {len(chunks)} chunks...")
    merged_data = merge_chunk_results(all_chunk_results)
    
    # Structure final output with normalized metadata shapes
    md_title = merged_data.get('title')
    title_obj = md_title if isinstance(md_title, dict) else {
        "value": (md_title if md_title else (chapter_title or pdf_path.stem)), 
        "present_in_source": bool(md_title)
    }
    
    md_ch_num = merged_data.get('chapter_number')
    chnum_obj = md_ch_num if isinstance(md_ch_num, dict) else {
        "value": (md_ch_num if md_ch_num else None),
        "present_in_source": bool(md_ch_num)
    }
    
    authors = merged_data.get('authors', [])
    authors = [{"name": a} if isinstance(a, str) else a for a in authors]
    
    final_output = {
        "chapter_metadata": {
            "title": title_obj,
            "authors": authors,
            "chapter_number": chnum_obj,
            "learning_objectives": merged_data.get('learning_objectives', []),
            "key_points": merged_data.get('key_points', [])
        },
        "anatomical_content": {
            "structures": merged_data.get('anatomical_structures', [])
        },
        "clinical_procedures": merged_data.get('procedures', []),
        "diagnostic_approaches": merged_data.get('diagnostic_approaches', []),
        "treatment_algorithms": [],
        "clinical_guidelines": [],
        "drug_information": merged_data.get('drugs', []),
        "tables": merged_data.get('tables', []),
        "figures": merged_data.get('figures', []),
        "clinical_cases": merged_data.get('case_examples', []),
        "definitions": merged_data.get('definitions', []),
        "clinical_pearls": merged_data.get('clinical_pearls', []),
        "references": merged_data.get('references', []),
        "summary": {
            "clinical_applications": merged_data.get('clinical_applications', []),
            "practice_recommendations": merged_data.get('practice_recommendations', []),
            "future_directions": merged_data.get('future_directions', []),
            "controversies": merged_data.get('controversies', [])
        }
    }
    
    # Split guidelines vs algorithms
    for item in merged_data.get('guidelines_algorithms', []):
        if 'algorithm' in str(item.get('title', '')).lower():
            final_output['treatment_algorithms'].append(item)
        else:
            final_output['clinical_guidelines'].append(item)
    
    # Add extraction metadata
    final_output['extraction_metadata'] = {
        'source_pdf': str(pdf_path),
        'adobe_json': str(adobe_json_path) if adobe_json_path else None,
        'extraction_date': datetime.now(UTC).isoformat(),
        'text_pages': len(pages),
        'chunks_processed': len(chunks),
        'model': model,
        'extractor_version': 'production_multipass_v3.2_gpt5_simple',
        'enhancements_applied': [
            'ocr_cleanup',
            'type_normalization',
            'criteria_tokenization',
            'guideline_deduplication',
            'page_range_normalization',
            'performance_metric_structuring',
            'definition_enrichment'
        ],
        'passes_completed': len(passes)
    }
    
    # Post-process cleanup
    post_process_cleanup(final_output)
    
    # Enforce provenance for numeric values
    enforce_provenance(final_output)
    
    # Quality audit
    issues = quality_audit(final_output)
    if issues:
        final_output['extraction_metadata']['quality_issues'] = issues
        print(f"⚠️  Quality issues found: {len(issues)} categories")
    
    # Schema validation for required metadata
    REQUIRED_META = ["title", "authors"]
    md = final_output.get("chapter_metadata", {})
    if not md.get("title") or not isinstance(md.get("authors"), list):
        final_output.setdefault("extraction_metadata", {}).setdefault("quality_issues", {}).setdefault("metadata", []).append("Missing title/authors")
    
    return final_output


def process_single_chapter(
    pdf_path: Path,
    adobe_json_path: Optional[Path] = None,
    output_dir: Path = Path('data/production_extractions'),
    chapter_title: Optional[str] = None,
    model: Optional[str] = None,
    passes: Optional[List[str]] = None
):
    """Process a single chapter with production extractor"""
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract content
    extracted_data = extract_multipass_production(
        pdf_path,
        adobe_json_path,
        chapter_title,
        model,
        passes  # Now properly forwarded!
    )
    
    # Save to file
    output_file = output_dir / f"{pdf_path.stem}_production.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved to: {output_file}")
    
    # Print summary
    print(f"\n📊 Extraction Summary:")
    print("="*50)
    
    total = 0
    for category, label in [
        ('anatomical_content', 'Anatomical structures'),
        ('clinical_procedures', 'Clinical procedures'),
        ('diagnostic_approaches', 'Diagnostic approaches'),
        ('treatment_algorithms', 'Treatment algorithms'),
        ('clinical_guidelines', 'Clinical guidelines'),
        ('drug_information', 'Drug information'),
        ('tables', 'Tables'),
        ('figures', 'Figures'),
        ('clinical_cases', 'Clinical cases'),
        ('definitions', 'Definitions'),
        ('clinical_pearls', 'Clinical pearls'),
        ('references', 'References')
    ]:
        if category == 'anatomical_content':
            count = len(extracted_data.get(category, {}).get('structures', []))
        else:
            count = len(extracted_data.get(category, []))
        
        if count > 0:
            print(f"  {label:.<30} {count:>3}")
            total += count
    
    print(f"  {'TOTAL':<30} {total:>3}")
    
    # Show quality issues if any
    issues = extracted_data.get('extraction_metadata', {}).get('quality_issues', {})
    if issues:
        print(f"\n⚠️  Quality Issues:")
        for category, problems in issues.items():
            print(f"  {category}: {len(problems)} issues")
    
    return output_file


def main():
    print("Entering main function...")
    parser = argparse.ArgumentParser(
        description='Production Multi-Pass Textbook Extractor',
        epilog="""Example usage:
python production_multipass_extractor.py --single "Textbooks/Chapter pdfs/Approach to Peripheral Lung Lesions.pdf" \\
  --adobe-json "Textbooks/Chapter json/Approach to Peripheral Lung Lesions.json" \\
  --passes pass0_metadata pass3_diagnostics pass4_guidelines pass6_tables pass7_figures pass8_education pass9_references
        """
    )
    
    parser.add_argument('--single', type=Path, help='Process single PDF')
    parser.add_argument('--adobe-json', type=Path, help='Adobe Extract JSON')
    parser.add_argument('--title', type=str, help='Chapter title')
    parser.add_argument('--batch', action='store_true', help='Process all chapters')
    parser.add_argument('--model', type=str, help='OpenAI model')
    parser.add_argument(
        '--passes',
        type=str,
        nargs='+',
        choices=list(EXTRACTION_PASSES.keys()),
        help='Specific passes to run'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/production_extractions'),
        help='Output directory'
    )
    
    args = parser.parse_args()
    print(f"Arguments parsed: single={args.single}, batch={args.batch}, model={args.model}")
    
    if not args.single and not args.batch:
        parser.error('Specify --single or --batch')
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)
    
    if args.single:
        process_single_chapter(
            args.single,
            args.adobe_json,
            args.output_dir,
            args.title,
            args.model,
            args.passes  # Now properly passed through!
        )
    elif args.batch:
        # Batch processing for all textbook chapters
        textbook_dir = Path("Textbooks")
        pdf_dir = textbook_dir / "Chapter pdfs"
        json_dir = textbook_dir / "Chapter json"
        
        if not pdf_dir.exists():
            print(f"Error: {pdf_dir} does not exist")
            sys.exit(1)
        
        # Get all PDF files
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            print(f"No PDF files found in {pdf_dir}")
            sys.exit(1)
        
        print(f"\n🚀 Starting batch extraction for {len(pdf_files)} chapters")
        print("="*60)
        
        successful = []
        failed = []
        
        for i, pdf_path in enumerate(pdf_files, 1):
            chapter_name = pdf_path.stem
            adobe_json = json_dir / f"{chapter_name}.json"
            
            print(f"\n📚 Chapter {i}/{len(pdf_files)}: {chapter_name}")
            print("-"*40)
            
            # Check if Adobe JSON exists
            if not adobe_json.exists():
                print(f"   ⚠️  Warning: Adobe JSON not found, extracting from PDF only")
                adobe_json = None
            
            try:
                output_file = process_single_chapter(
                    pdf_path,
                    adobe_json,
                    args.output_dir,
                    chapter_title=chapter_name,
                    model=args.model,
                    passes=args.passes
                )
                successful.append(chapter_name)
                print(f"   ✅ Success: {output_file}")
            except Exception as e:
                failed.append((chapter_name, str(e)))
                print(f"   ❌ Failed: {str(e)[:100]}")
        
        # Print summary
        print("\n" + "="*60)
        print("📊 BATCH EXTRACTION SUMMARY")
        print("="*60)
        print(f"✅ Successful: {len(successful)}/{len(pdf_files)} chapters")
        if successful:
            for name in successful[:5]:  # Show first 5
                print(f"   • {name}")
            if len(successful) > 5:
                print(f"   • ... and {len(successful)-5} more")
        
        if failed:
            print(f"\n❌ Failed: {len(failed)} chapters")
            for name, error in failed[:3]:  # Show first 3 errors
                print(f"   • {name}: {error[:50]}...")
        
        print(f"\n📁 Output directory: {args.output_dir}")
        print("✨ Batch extraction complete!")


if __name__ == "__main__":
    print("Starting production_multipass_textbook_extractor...")
    main()
    print("Extractor completed.")
