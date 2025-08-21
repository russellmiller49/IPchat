#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust extractor for medical papers -> structured JSON (schema: medical_rag_chatbot_v1).
- Uses chat.completions with model-supplied constraints (max_completion_tokens, default temperature)
- Three-attempt flow: json_schema -> json_object -> shortened prompt
- Logs raw model output to last_model_output.json on each attempt
- More-forgiving JSON parser (handles code fences, leading/trailing noise, trailing commas)
- Smaller prompt slices to reduce empty responses
- CLI: --single/--pdf or --batch, optional --open-ui to launch inspector
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv is optional

# Check TEST_MODE early to avoid import errors  
TEST_MODE = os.getenv("TEST_MODE", "false").lower() in ("true", "1", "yes")

# Check command line args early for --test flag
if len(sys.argv) > 1 and "--test" in sys.argv:
    TEST_MODE = True

# OpenAI import with explicit error handling
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI  # type: ignore
    OPENAI_AVAILABLE = True
except ImportError as e:
    if not TEST_MODE:
        print("[FATAL] openai package missing or incompatible. pip install openai>=1.0", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        print("Alternatively, use TEST_MODE=true environment variable or --test flag to run without API calls", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[INFO] OpenAI package not available, but running in test mode", file=sys.stderr)
        OpenAI = None  # Define as None for test mode
except Exception as e:
    print(f"[FATAL] Unexpected error importing openai: {e}", file=sys.stderr)
    sys.exit(1)

# ----------------------------
# Config
# ----------------------------
# Available GPT-5 models (as of Aug 2025):
# - gpt-5: Best for logic and multi-step tasks
# - gpt-5-mini: Lightweight for cost-sensitive applications  
# - gpt-5-nano: Optimized for ultra-low latency
# - gpt-5-chat: Built for advanced multimodal conversations
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "8192"))
API_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "300"))  # 5 minutes default
SCHEMA_PATH = Path("schemas/medical_rag_chatbot_v1.schema.json")

INPUT_DIR = Path("data/input_articles")
PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_LOG = Path("last_model_output.json")


# ----------------------------
# Utilities
# ----------------------------
def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(read_text(path))


def read_adobe_json_text(adobe_json: Dict[str, Any]) -> str:
    """Flatten the Adobe Extract JSON-ish file into a long text blob.
    We intentionally cap the slice to avoid context overflows.
    """
    parts: List[str] = []
    
    # Extract text from Adobe Extract API elements
    if "elements" in adobe_json:
        for element in adobe_json.get("elements", []):
            # Get text from Text elements
            if element.get("Text"):
                parts.append(element["Text"])
            # Get text from attributes
            if element.get("attributes", {}).get("TextContent"):
                parts.append(element["attributes"]["TextContent"])
    
    # Also check for direct text fields
    for key in ["title", "authors", "abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
        val = adobe_json.get(key)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            parts.extend([str(x) for x in val if isinstance(x, str)])
    
    # Check for content/text/body fields
    for k, v in adobe_json.items():
        if k in {"content", "text", "body"} and isinstance(v, str):
            parts.append(v)

    blob = "\n".join(parts)
    # Reduce to 30K chars for GPT-5 testing
    return blob[:30000]  # trimmed slice for GPT-5


def read_pdf_text(pdf_path: Path, page_limit: int = 50) -> str:
    """Read text from PDF file using PyMuPDF (fitz)."""
    try:
        import fitz  # type: ignore # PyMuPDF
    except ImportError:
        print("[WARNING] PyMuPDF not available. Install with: pip install PyMuPDF")
        return ""
    except Exception as e:
        print(f"[WARNING] Failed to import PyMuPDF: {e}")
        return ""
    
    try:
        doc = fitz.open(str(pdf_path))
        pages = min(len(doc), page_limit)
        chunks = []
        for i in range(pages):
            chunks.append(doc.load_page(i).get_text("text"))
        doc.close()
        return "".join(chunks)[:90000]  # trimmed slice
    except Exception as e:
        print(f"[WARNING] Failed to read PDF {pdf_path}: {e}")
        return ""


def build_user_prompt(doc_id: str, adobe_text: str, pdf_text: str) -> str:
    from datetime import datetime, timezone
    current_date = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    return f"""Extract structured medical evidence for interventional pulmonology & critical care.

Rules:
- Return ONLY a single JSON object that validates against the schema "medical_rag_chatbot_v1".
- REQUIRED fields that MUST be included:
  - source.document_id: "{doc_id}"
  - source.ingest_date: "{current_date}"
  - document.metadata.title: Extract from document
  - document.metadata.year: Extract publication year
  - document.design: Include study design details if available
  - retrieval.keywords: Extract relevant keywords
  - retrieval.summary_tldr: Create a brief summary
  - retrieval.embedding_ref: Can use placeholder values if needed
- For every numeric datum, include a provenance span with page(s) and table_id/figure_id when applicable.
- Use canonical units when possible and include raw counts for recomputation.
- If a field is unknown, use empty string "" for strings, empty array [] for arrays, or empty object {{}} for objects.
- Do not include commentary or code fences. JSON only.

CONTEXT:
[DOCUMENT_ID]: {doc_id}

[ADOBE_TEXT_SNIPPET]:
{adobe_text}

[PDF_TEXT_SNIPPET]:
{pdf_text}
"""


def _sanitize_candidate_json(candidate: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", candidate)


def _strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|```\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    return cleaned


def _parse_json_from_text(text: str) -> Dict[str, Any]:
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        candidate = _sanitize_candidate_json(candidate)
        return json.loads(candidate)
    raise ValueError("Model did not return valid JSON content")


def clean_last_model_output():
    """Clean up any existing empty or invalid last_model_output.json file."""
    if RAW_LOG.exists():
        try:
            content = RAW_LOG.read_text(encoding="utf-8").strip()
            if not content:
                print("[DEBUG] Found empty last_model_output.json, removing it")
                RAW_LOG.unlink()
            else:
                # Try to parse as JSON to check validity
                json.loads(content)
                print(f"[DEBUG] Found valid last_model_output.json ({len(content)} chars)")
        except json.JSONDecodeError:
            print("[DEBUG] Found invalid JSON in last_model_output.json, removing it")
            RAW_LOG.unlink()
        except Exception as e:
            print(f"[DEBUG] Error checking last_model_output.json: {e}")


def create_test_response(doc_id: str) -> Dict[str, Any]:
    """Create a mock response for test mode."""
    from datetime import datetime, timezone
    current_date = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    return {
        "source": {
            "document_id": doc_id,
            "ingest_date": current_date
        },
        "document": {
            "metadata": {
                "title": "Test Document - Mock Response",
                "year": "2024",
                "authors": ["Test Author"]
            },
            "sections": {
                "abstract": "This is a test response generated in test mode.",
                "methods": "Mock methods section",
                "results": "Mock results section",
                "conclusion": "Mock conclusion section"
            }
        },
        "design": {
            "study_type": "test",
            "participants": {"total": 100}
        },
        "retrieval": {
            "keywords": ["test", "mock", "response"],
            "summary_tldr": "Test response for debugging purposes",
            "embedding_ref": "test_embedding_ref"
        },
        "data": {
            "primary_endpoints": [],
            "secondary_endpoints": [],
            "adverse_events": []
        }
    }


def call_gpt5(system_text: str, user_text: str, schema_obj: Dict[str, Any]) -> Dict[str, Any]:
    # Check for test mode first
    if TEST_MODE:
        print("[INFO] Running in TEST MODE - no API calls will be made")
        doc_id = user_text.split('document_id: "')[1].split('"')[0] if 'document_id: "' in user_text else "test_doc"
        print("[DEBUG] Creating mock response for test mode")
        time.sleep(1)  # Simulate some processing time
        mock_response = create_test_response(doc_id)
        # Still write to RAW_LOG for consistency
        RAW_LOG.write_text(json.dumps(mock_response, indent=2), encoding="utf-8")
        return mock_response
    
    # Check if OpenAI is available
    if not OPENAI_AVAILABLE:
        raise ValueError(
            "OpenAI package not available. Install with: pip install openai>=1.0\n"
            "Alternatively, use TEST_MODE=true to run without API calls."
        )
    
    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError(
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable or create a .env file with:\n"
            "OPENAI_API_KEY=your_actual_api_key_here\n\n"
            "You can copy .env.example to .env and fill in your API key.\n\n"
            "Alternatively, set TEST_MODE=true to run without API calls."
        )
    
    print(f"[INFO] Using model: {MODEL}")
    print(f"[INFO] Max output tokens: {MAX_OUTPUT_TOKENS}")
    print(f"[INFO] API timeout: {API_TIMEOUT} seconds")
    print(f"[DEBUG] System prompt length: {len(system_text)} chars")
    print(f"[DEBUG] User prompt length: {len(user_text)} chars")
    print(f"[DEBUG] Total prompt size: {len(system_text) + len(user_text)} chars")
    
    # Clean up any invalid previous output
    clean_last_model_output()
    
    print("[DEBUG] Initializing OpenAI client...")
    client = OpenAI(timeout=API_TIMEOUT)
    print("[DEBUG] OpenAI client initialized")
    
    # First attempt: JSON Schema
    try:
        print(f"[DEBUG] Starting API call (attempt 1: json_schema) at {datetime.now().strftime('%H:%M:%S')}...")
        print(f"[DEBUG] Sending request to OpenAI API...")
        start_time = time.time()
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "medical_rag_chatbot_v1", "schema": schema_obj, "strict": False},
            },
            max_completion_tokens=MAX_OUTPUT_TOKENS,
        )
        elapsed = time.time() - start_time
        print(f"[DEBUG] API call completed in {elapsed:.1f}s")
        
        content = (resp.choices[0].message.content or "").strip()
        print(f"[DEBUG] Received {len(content)} characters from API")
        RAW_LOG.write_text(content, encoding="utf-8")
        if not content:
            raise ValueError("Empty content from schema call")
        return _parse_json_from_text(content)
        
    except Exception as e:
        print(f"[DEBUG] First attempt failed: {e}")
        
        # Second attempt: JSON Object
        try:
            print("[DEBUG] Starting API call (attempt 2: json_object)...")
            start_time = time.time()
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=MAX_OUTPUT_TOKENS,
            )
            elapsed = time.time() - start_time
            print(f"[DEBUG] API call completed in {elapsed:.1f}s")
            
            content = (resp.choices[0].message.content or "").strip()
            print(f"[DEBUG] Received {len(content)} characters from API")
            RAW_LOG.write_text(content, encoding="utf-8")
            
            if not content:
                print("[DEBUG] Empty response, trying shortened prompt...")
                # Third attempt: Shortened prompt
                short_user = user_text.split("[PDF_TEXT_SNIPPET]:", 1)[0] + "\n\nReturn ONLY a single valid JSON object. No prose."
                print("[DEBUG] Starting API call (attempt 3: shortened prompt)...")
                start_time = time.time()
                resp2 = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": short_user},
                    ],
                    max_completion_tokens=MAX_OUTPUT_TOKENS,
                )
                elapsed = time.time() - start_time
                print(f"[DEBUG] API call completed in {elapsed:.1f}s")
                
                content = (resp2.choices[0].message.content or "").strip()
                print(f"[DEBUG] Received {len(content)} characters from API")
                RAW_LOG.write_text(content, encoding="utf-8")
                if not content:
                    raise ValueError("Empty content after all attempts")
            
            return _parse_json_from_text(content)
            
        except Exception as e2:
            print(f"[DEBUG] All attempts failed. Last error: {e2}")
            raise e2


def validate_schema(data: Dict[str, Any], schema_obj: Dict[str, Any]) -> Optional[str]:
    try:
        import jsonschema  # type: ignore
    except ImportError:
        print("[WARNING] jsonschema not available. Install with: pip install jsonschema")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to import jsonschema: {e}")
        return None
    try:
        jsonschema.validate(data, schema_obj)
        return None
    except Exception as e:
        return str(e)


def test_api_connection() -> bool:
    """Test if we can connect to OpenAI API with a simple call."""
    if TEST_MODE:
        print("[DEBUG] Test mode enabled - skipping API connection test")
        return True
        
    if not OPENAI_AVAILABLE:
        print("[DEBUG] OpenAI package not available")
        return False
        
    if not os.getenv("OPENAI_API_KEY"):
        print("[DEBUG] No API key found")
        return False
        
    try:
        print("[DEBUG] Testing API connection...")
        client = OpenAI(timeout=10)  # Short timeout for connection test
        
        start_time = time.time()
        # Simple test call
        resp = client.chat.completions.create(
            model="gpt-3.5-turbo",  # Use cheaper model for test
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=1
        )
        elapsed = time.time() - start_time
        print(f"[DEBUG] API connection test successful ({elapsed:.1f}s)")
        return True
        
    except Exception as e:
        print(f"[DEBUG] API connection test failed: {e}")
        return False


def extract_one(json_path: Path, pdf_path: Path, outdir: Path) -> Tuple[Optional[Path], Optional[str]]:
    try:
        adobe_json = read_json(json_path)
    except Exception as e:
        return None, f"Failed to read JSON: {json_path} -> {e}"
    schema_obj = read_json(SCHEMA_PATH) if SCHEMA_PATH.exists() else None
    if not schema_obj:
        return None, f"Schema not found at {SCHEMA_PATH}"

    pdf_text = ""
    if pdf_path and pdf_path.exists():
        try:
            pdf_text = read_pdf_text(pdf_path)
        except Exception:
            pdf_text = ""

    doc_id = adobe_json.get("document_id") or json_path.stem
    adobe_text = read_adobe_json_text(adobe_json)
    
    print(f"[INFO] Document ID: {doc_id}")
    print(f"[INFO] Adobe text extracted: {len(adobe_text)} characters")
    print(f"[INFO] PDF text extracted: {len(pdf_text)} characters")
    
    # Log first 500 chars of adobe text to verify extraction
    if adobe_text:
        print(f"[INFO] Adobe text preview: {adobe_text[:500]}...")
    else:
        print("[WARNING] No text extracted from Adobe JSON")
    
    # Add current timestamp for ingest_date if not present
    from datetime import datetime, timezone
    if "source" not in adobe_json or "ingest_date" not in adobe_json.get("source", {}):
        if "source" not in adobe_json:
            adobe_json["source"] = {}
        adobe_json["source"]["ingest_date"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    user_text = build_user_prompt(doc_id, adobe_text, pdf_text)

    system_text = "You are a careful information extraction system. Output must be a single valid JSON object that matches the provided schema. Never include explanations."

    print("[DEBUG] Starting extraction process...")
    try:
        data = call_gpt5(system_text, user_text, schema_obj)
        print("[DEBUG] Extraction completed successfully")
    except Exception as e:
        raw = RAW_LOG.read_text(encoding="utf-8") if RAW_LOG.exists() else ""
        print(f"[DEBUG] Extraction failed: {e}")
        return None, f"Model error: {e}\n--- Raw model output (first 2000 chars) ---\n{raw[:2000]}"

    err = validate_schema(data, schema_obj)
    if err:
        user_text_retry = (
            "Return ONLY a single JSON object that VALIDATES against the following JSON Schema. "
            "Fix these validation errors: " + err + "\n\nJSON Schema:\n" + json.dumps(schema_obj)
            + "\n\nContext follows. Do not include any commentary.\n\n" + user_text
        )
        print("[DEBUG] Schema validation failed, retrying with corrected prompt...")
        try:
            data = call_gpt5(system_text, user_text_retry, schema_obj)
            print("[DEBUG] Validation retry completed successfully")
        except Exception as e:
            raw = RAW_LOG.read_text(encoding="utf-8") if RAW_LOG.exists() else ""
            print(f"[DEBUG] Validation retry failed: {e}")
            return None, f"Model error after validation retry: {e}\n--- Raw model output (first 2000 chars) ---\n{raw[:2000]}"

    out_path = outdir / f"{json_path.stem}.structured.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", type=str, help="Path to Adobe Extract JSON file")
    parser.add_argument("--pdf", type=str, help="Path to the corresponding PDF")
    parser.add_argument("--batch", action="store_true", help="Process all JSON files under data/input_articles")
    parser.add_argument("--open-ui", action="store_true", help="Launch Evidence Inspector UI after extraction")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no API calls)")
    parser.add_argument("--test-connection", action="store_true", help="Test API connection only")
    args = parser.parse_args()
    
    # Set test mode if requested
    if args.test:
        os.environ["TEST_MODE"] = "true"
        global TEST_MODE
        TEST_MODE = True
        print("[INFO] Test mode enabled via command line")
    
    # Test connection if requested
    if args.test_connection:
        success = test_api_connection()
        if success:
            print("[OK] API connection test passed")
        else:
            print("[FAIL] API connection test failed")
            sys.exit(1)
        return

    if args.single:
        jpath = Path(args.single)
        ppath = Path(args.pdf) if args.pdf else None
        out, err = extract_one(jpath, ppath, OUTPUT_DIR)
        if err:
            print(f"[FAIL] {err}")
            sys.exit(1)
        print(f"[OK] Wrote {out}")
    elif args.batch:
        failures = 0
        for jpath in sorted(INPUT_DIR.glob("*.json")):
            ppdf = (PDF_DIR / (jpath.stem + ".pdf"))
            out, err = extract_one(jpath, ppdf if ppdf.exists() else None, OUTPUT_DIR)
            if err:
                print(f"[FAIL] {jpath.name}: {err}")
                failures += 1
            else:
                print(f"[OK] {jpath.name} -> {out.name}")
        if failures:
            print(f"[DONE] Completed with {failures} failures")
        else:
            print("[DONE] All files processed successfully")
    else:
        parser.print_help()
        return

    if args.open_ui:
        try:
            import subprocess, sys as _sys
            subprocess.run([_sys.executable, "-m", "streamlit", "run", "tools/evidence_inspector_app.py"])
        except Exception as e:
            print(f"[WARN] Could not launch Streamlit UI: {e}")


if __name__ == "__main__":
    main()
