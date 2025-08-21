#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-5 Medical Evidence Extractor - Fixed Version
Uses the new Responses API with proper structured output support
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
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    if not TEST_MODE:
        print("[FATAL] openai package missing or incompatible. pip install openai>=1.0", file=sys.stderr)
        print(f"Error details: {e}", file=sys.stderr)
        print("Alternatively, use TEST_MODE=true environment variable or --test flag to run without API calls", file=sys.stderr)
        sys.exit(1)
    else:
        print(f"[INFO] OpenAI package not available, but running in test mode", file=sys.stderr)
        OpenAI = None

# ----------------------------
# Config
# ----------------------------
# GPT-5 models available (as of Aug 2025):
# - gpt-5: Best for logic and multi-step tasks  
# - gpt-5-mini: Lightweight for cost-sensitive applications
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
    """Flatten the Adobe Extract JSON-ish file into a long text blob."""
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
    return blob[:30000]


def read_pdf_text(pdf_path: Path, page_limit: int = 50) -> str:
    """Read text from PDF file using PyMuPDF (fitz)."""
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[WARNING] PyMuPDF not available. Install with: pip install PyMuPDF")
        return ""
    
    try:
        doc = fitz.open(str(pdf_path))
        pages = min(len(doc), page_limit)
        chunks = []
        for i in range(pages):
            chunks.append(doc.load_page(i).get_text("text"))
        doc.close()
        return "".join(chunks)[:90000]
    except Exception as e:
        print(f"[WARNING] Failed to read PDF {pdf_path}: {e}")
        return ""


def build_extraction_prompt(doc_id: str, adobe_text: str, pdf_text: str, schema_obj: Dict[str, Any]) -> str:
    """Build the extraction prompt for GPT-5."""
    current_date = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    prompt = f"""You are a medical evidence extraction specialist. Extract structured information from the following medical document.

TASK: Extract all relevant medical evidence and return it as a JSON object matching this schema:

{json.dumps(schema_obj, indent=2)}

REQUIREMENTS:
- Return ONLY valid JSON that matches the schema above
- Include document_id: "{doc_id}"
- Include ingest_date: "{current_date}"
- Extract all available fields from the document
- Use empty strings, arrays, or objects for missing fields
- Include page numbers and references for all data points

DOCUMENT CONTENT:

[ADOBE EXTRACT TEXT]:
{adobe_text[:15000]}

[PDF TEXT]:
{pdf_text[:15000]}

Output the extracted data as JSON:"""
    
    return prompt


def create_test_response(doc_id: str) -> Dict[str, Any]:
    """Create a mock response for test mode."""
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


def call_gpt5_responses_api(prompt: str, schema_obj: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call GPT-5 using the new Responses API with structured output.
    """
    # Check for test mode
    if TEST_MODE:
        print("[INFO] Running in TEST MODE - no API calls will be made")
        doc_id = "test_doc"
        if 'document_id: "' in prompt:
            doc_id = prompt.split('document_id: "')[1].split('"')[0]
        print("[DEBUG] Creating mock response for test mode")
        time.sleep(1)
        mock_response = create_test_response(doc_id)
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
            "OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.\n"
            "Alternatively, set TEST_MODE=true to run without API calls."
        )
    
    print(f"[INFO] Using GPT-5 Responses API with model: {MODEL}")
    print(f"[DEBUG] Prompt length: {len(prompt)} chars")
    
    # Initialize client with timeout
    client = OpenAI(timeout=API_TIMEOUT)
    
    try:
        # First attempt: with structured output
        print(f"[DEBUG] Calling GPT-5 Responses API with structured output...")
        start_time = time.time()
        
        # Create the structured output schema for Responses API
        structured_schema = {
            "name": "MedicalExtraction",
            "schema": schema_obj,
            "strict": True
        }
        
        # Use the Responses API (not Chat Completions)
        resp = client.responses.create(
            model=MODEL,
            input=prompt,
            response_format={
                "type": "json_schema",
                "json_schema": structured_schema
            }
        )
        
        elapsed = time.time() - start_time
        print(f"[DEBUG] API call completed in {elapsed:.1f}s")
        
        # Extract the response content
        if hasattr(resp, 'output_text'):
            content = resp.output_text
        elif hasattr(resp, 'output') and resp.output:
            # Handle structured output format
            if isinstance(resp.output, list) and len(resp.output) > 0:
                output_item = resp.output[0]
                if hasattr(output_item, 'content') and isinstance(output_item.content, list):
                    content = output_item.content[0].text if output_item.content else ""
                else:
                    content = str(output_item)
            else:
                content = str(resp.output)
        else:
            content = ""
        
        print(f"[DEBUG] Received {len(content)} characters from API")
        
        if not content:
            raise ValueError("Empty response from API")
        
        # Save raw output
        RAW_LOG.write_text(content, encoding="utf-8")
        
        # Parse JSON from response
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from the response
            if "{" in content and "}" in content:
                start = content.find("{")
                end = content.rfind("}") + 1
                json_str = content[start:end]
                return json.loads(json_str)
            raise
            
    except Exception as e:
        print(f"[ERROR] GPT-5 API call failed: {e}")
        
        # Fallback: Try without structured output
        try:
            print("[DEBUG] Retrying without structured output...")
            simple_prompt = prompt + "\n\nReturn ONLY valid JSON, no explanation."
            
            resp = client.responses.create(
                model=MODEL,
                input=simple_prompt
            )
            
            if hasattr(resp, 'output_text'):
                content = resp.output_text
            else:
                content = ""
            
            print(f"[DEBUG] Fallback received {len(content)} characters")
            RAW_LOG.write_text(content, encoding="utf-8")
            
            if not content:
                raise ValueError("Empty response from fallback attempt")
            
            return json.loads(content)
            
        except Exception as e2:
            print(f"[ERROR] All attempts failed: {e2}")
            raise


def test_gpt5_connection() -> bool:
    """Test if we can connect to GPT-5 API with a simple call."""
    if TEST_MODE:
        print("[DEBUG] Test mode enabled - skipping API connection test")
        return True
    
    if not OPENAI_AVAILABLE:
        print("[ERROR] OpenAI package not available")
        return False
    
    if not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] No API key found")
        return False
    
    try:
        print("[INFO] Testing GPT-5 connection...")
        client = OpenAI(timeout=30)
        
        # Simple test with Responses API
        resp = client.responses.create(
            model=MODEL,
            input="Say hello to Dr. Miller in one sentence."
        )
        
        if hasattr(resp, 'output_text') and resp.output_text:
            print(f"[SUCCESS] GPT-5 responded: {resp.output_text[:100]}")
            return True
        else:
            print("[ERROR] Unexpected response format from GPT-5")
            return False
            
    except Exception as e:
        print(f"[ERROR] GPT-5 connection test failed: {e}")
        return False


def extract_one(json_path: Path, pdf_path: Path, outdir: Path) -> Tuple[Optional[Path], Optional[str]]:
    """Extract structured data from one document."""
    try:
        adobe_json = read_json(json_path)
    except Exception as e:
        return None, f"Failed to read JSON: {json_path} -> {e}"
    
    schema_obj = read_json(SCHEMA_PATH) if SCHEMA_PATH.exists() else None
    if not schema_obj:
        return None, f"Schema not found at {SCHEMA_PATH}"
    
    pdf_text = ""
    if pdf_path and pdf_path.exists():
        pdf_text = read_pdf_text(pdf_path)
    
    doc_id = adobe_json.get("document_id") or json_path.stem
    adobe_text = read_adobe_json_text(adobe_json)
    
    print(f"[INFO] Processing document: {doc_id}")
    print(f"[INFO] Adobe text: {len(adobe_text)} chars")
    print(f"[INFO] PDF text: {len(pdf_text)} chars")
    
    # Build the extraction prompt
    prompt = build_extraction_prompt(doc_id, adobe_text, pdf_text, schema_obj)
    
    # Call GPT-5
    try:
        data = call_gpt5_responses_api(prompt, schema_obj)
        print("[SUCCESS] Extraction completed")
    except Exception as e:
        return None, f"Extraction failed: {e}"
    
    # Save the result
    out_path = outdir / f"{json_path.stem}.structured.json"
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path, None


def main():
    parser = argparse.ArgumentParser(description="GPT-5 Medical Evidence Extractor")
    parser.add_argument("--single", type=str, help="Path to Adobe Extract JSON file")
    parser.add_argument("--pdf", type=str, help="Path to the corresponding PDF")
    parser.add_argument("--batch", action="store_true", help="Process all JSON files under data/input_articles")
    parser.add_argument("--open-ui", action="store_true", help="Launch Evidence Inspector UI after extraction")
    parser.add_argument("--test", action="store_true", help="Run in test mode (no API calls)")
    parser.add_argument("--test-connection", action="store_true", help="Test GPT-5 API connection only")
    args = parser.parse_args()
    
    # Set test mode if requested
    if args.test:
        os.environ["TEST_MODE"] = "true"
        global TEST_MODE
        TEST_MODE = True
        print("[INFO] Test mode enabled")
    
    # Test connection if requested
    if args.test_connection:
        success = test_gpt5_connection()
        sys.exit(0 if success else 1)
    
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
            import subprocess
            import sys as _sys
            subprocess.run([_sys.executable, "-m", "streamlit", "run", "tools/evidence_inspector_app.py"])
        except Exception as e:
            print(f"[WARN] Could not launch Streamlit UI: {e}")


if __name__ == "__main__":
    main()