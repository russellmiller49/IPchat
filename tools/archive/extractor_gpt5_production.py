#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-5 Medical Evidence Extractor - Production Version
Uses GPT-5 Responses API for medical document extraction
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

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Check TEST_MODE
TEST_MODE = os.getenv("TEST_MODE", "false").lower() in ("true", "1", "yes")
if "--test" in sys.argv:
    TEST_MODE = True

# Import OpenAI
OPENAI_AVAILABLE = False
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    if not TEST_MODE:
        print("[ERROR] OpenAI package required. Install with: pip install openai>=1.0", file=sys.stderr)
        sys.exit(1)
    OpenAI = None

# Configuration
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
API_TIMEOUT = int(os.getenv("OPENAI_TIMEOUT", "300"))
SCHEMA_PATH = Path("schemas/medical_rag_chatbot_v1.schema.json")

INPUT_DIR = Path("data/input_articles")
PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_LOG = Path("last_model_output.json")


def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def read_adobe_json_text(adobe_json: Dict[str, Any]) -> str:
    """Extract text from Adobe JSON format."""
    parts = []
    
    # Extract from elements
    if "elements" in adobe_json:
        for element in adobe_json.get("elements", []):
            if element.get("Text"):
                parts.append(element["Text"])
            if element.get("attributes", {}).get("TextContent"):
                parts.append(element["attributes"]["TextContent"])
    
    # Extract from direct fields
    for key in ["title", "authors", "abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
        val = adobe_json.get(key)
        if isinstance(val, str):
            parts.append(val)
        elif isinstance(val, list):
            parts.extend([str(x) for x in val if isinstance(x, str)])
    
    # Check content fields
    for k, v in adobe_json.items():
        if k in {"content", "text", "body"} and isinstance(v, str):
            parts.append(v)
    
    return "\n".join(parts)[:30000]  # Limit for context window


def read_pdf_text(pdf_path: Path, page_limit: int = 50) -> str:
    """Extract text from PDF using PyMuPDF."""
    try:
        import fitz
    except ImportError:
        print("[WARNING] PyMuPDF not available")
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
        print(f"[WARNING] Failed to read PDF: {e}")
        return ""


def clean_json_response(text: str) -> str:
    """Clean and extract JSON from GPT-5 response."""
    # Remove code fences if present
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    
    # Find JSON boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start >= 0 and end > start:
        return text[start:end]
    
    return text.strip()


def build_extraction_prompt(doc_id: str, adobe_text: str, pdf_text: str, schema_obj: Dict[str, Any]) -> str:
    """Build the extraction prompt for GPT-5."""
    current_date = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Create a simplified schema description for the prompt
    schema_desc = """
Required JSON structure:
{
  "source": {
    "document_id": "string",
    "ingest_date": "ISO date string"
  },
  "document": {
    "metadata": {
      "title": "string",
      "year": "string",
      "authors": ["array of author names"],
      "journal": "string"
    },
    "sections": {
      "abstract": "string",
      "methods": "string", 
      "results": "string",
      "discussion": "string",
      "conclusion": "string"
    }
  },
  "design": {
    "study_type": "string (RCT, observational, etc.)",
    "participants": {
      "total": number,
      "intervention": number,
      "control": number
    },
    "duration": "string",
    "primary_endpoint": "string"
  },
  "data": {
    "primary_endpoints": [
      {
        "name": "string",
        "value": "string",
        "p_value": "string",
        "confidence_interval": "string"
      }
    ],
    "secondary_endpoints": [],
    "adverse_events": []
  },
  "retrieval": {
    "keywords": ["array of keywords"],
    "summary_tldr": "brief 1-2 sentence summary",
    "embedding_ref": "placeholder"
  }
}"""
    
    prompt = f"""You are a medical evidence extraction specialist for interventional pulmonology research.

TASK: Extract structured information from the medical document below and return it as JSON.

{schema_desc}

IMPORTANT REQUIREMENTS:
- Return ONLY valid JSON - no explanations or commentary
- Include document_id: "{doc_id}"
- Include ingest_date: "{current_date}"
- Extract all available information from the document
- Use empty strings "" for missing text fields
- Use empty arrays [] for missing array fields
- Use null for missing numeric fields
- Focus on interventional pulmonology data: bronchoscopy, lung procedures, airway interventions

DOCUMENT TEXT (Adobe Extract):
{adobe_text[:15000]}

DOCUMENT TEXT (PDF):
{pdf_text[:15000]}

JSON OUTPUT:"""
    
    return prompt


def call_gpt5(prompt: str) -> Dict[str, Any]:
    """Call GPT-5 API and parse response."""
    if TEST_MODE:
        print("[INFO] Running in TEST MODE")
        return {
            "source": {"document_id": "test", "ingest_date": datetime.now(timezone.utc).isoformat()},
            "document": {"metadata": {"title": "Test Document"}},
            "retrieval": {"keywords": ["test"], "summary_tldr": "Test response"}
        }
    
    if not OPENAI_AVAILABLE:
        raise ValueError("OpenAI package not available")
    
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY not set")
    
    print(f"[INFO] Calling GPT-5 ({MODEL})...")
    client = OpenAI(timeout=API_TIMEOUT)
    
    try:
        # Call GPT-5 Responses API
        start_time = time.time()
        response = client.responses.create(
            model=MODEL,
            input=prompt
        )
        elapsed = time.time() - start_time
        print(f"[INFO] Response received in {elapsed:.1f}s")
        
        # Get response text
        if hasattr(response, 'output_text'):
            content = response.output_text
        else:
            raise ValueError("Unexpected response format")
        
        # Log raw response
        RAW_LOG.write_text(content, encoding="utf-8")
        
        # Clean and parse JSON
        json_str = clean_json_response(content)
        return json.loads(json_str)
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] Failed to parse JSON: {e}")
        print(f"[DEBUG] Raw response: {content[:500]}")
        
        # Try to fix common JSON issues
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)  # Remove trailing commas
        try:
            return json.loads(json_str)
        except:
            raise ValueError(f"Could not parse GPT-5 response as JSON: {e}")
    
    except Exception as e:
        print(f"[ERROR] GPT-5 API call failed: {e}")
        raise


def validate_and_fix_response(data: Dict[str, Any], doc_id: str) -> Dict[str, Any]:
    """Validate and fix the extracted data."""
    # Ensure required fields exist
    if "source" not in data:
        data["source"] = {}
    
    if "document_id" not in data["source"]:
        data["source"]["document_id"] = doc_id
    
    if "ingest_date" not in data["source"]:
        data["source"]["ingest_date"] = datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')
    
    # Ensure document structure
    if "document" not in data:
        data["document"] = {"metadata": {}}
    
    if "metadata" not in data["document"]:
        data["document"]["metadata"] = {}
    
    # Ensure retrieval structure
    if "retrieval" not in data:
        data["retrieval"] = {
            "keywords": [],
            "summary_tldr": "",
            "embedding_ref": "placeholder"
        }
    
    return data


def extract_one(json_path: Path, pdf_path: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
    """Extract structured data from one document."""
    print(f"\n[INFO] Processing: {json_path.name}")
    
    try:
        adobe_json = read_json(json_path)
    except Exception as e:
        return None, f"Failed to read JSON: {e}"
    
    # Load schema
    schema_obj = read_json(SCHEMA_PATH) if SCHEMA_PATH.exists() else {}
    
    # Extract text
    adobe_text = read_adobe_json_text(adobe_json)
    pdf_text = read_pdf_text(pdf_path) if pdf_path and pdf_path.exists() else ""
    
    doc_id = adobe_json.get("document_id", json_path.stem)
    
    print(f"  Document ID: {doc_id}")
    print(f"  Adobe text: {len(adobe_text)} chars")
    print(f"  PDF text: {len(pdf_text)} chars")
    
    if not adobe_text and not pdf_text:
        return None, "No text content found"
    
    # Build prompt and call GPT-5
    prompt = build_extraction_prompt(doc_id, adobe_text, pdf_text, schema_obj)
    
    try:
        data = call_gpt5(prompt)
        data = validate_and_fix_response(data, doc_id)
        
        # Save output
        out_path = OUTPUT_DIR / f"{json_path.stem}.structured.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        print(f"  ✓ Saved to: {out_path.name}")
        return out_path, None
        
    except Exception as e:
        return None, f"Extraction failed: {e}"


def test_connection() -> bool:
    """Test GPT-5 API connection."""
    if TEST_MODE:
        print("[INFO] Test mode - skipping connection test")
        return True
    
    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        print("[ERROR] OpenAI not configured")
        return False
    
    try:
        print("[INFO] Testing GPT-5 connection...")
        client = OpenAI(timeout=30)
        response = client.responses.create(
            model=MODEL,
            input="Say 'Connection successful' in 3 words."
        )
        
        if hasattr(response, 'output_text'):
            print(f"[SUCCESS] GPT-5 responded: {response.output_text}")
            return True
        
        return False
        
    except Exception as e:
        print(f"[ERROR] Connection test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="GPT-5 Medical Evidence Extractor")
    parser.add_argument("--single", type=str, help="Process single file (name without extension or full path)")
    parser.add_argument("--pdf", type=str, help="Override PDF path (optional)")
    parser.add_argument("--batch", action="store_true", help="Process all files in input directory")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    parser.add_argument("--test-connection", action="store_true", help="Test API connection")
    parser.add_argument("--open-ui", action="store_true", help="Open Evidence Inspector after extraction")
    
    args = parser.parse_args()
    
    if args.test:
        global TEST_MODE
        TEST_MODE = True
        print("[INFO] Test mode enabled")
    
    if args.test_connection:
        success = test_connection()
        sys.exit(0 if success else 1)
    
    # Process files
    if args.single:
        # Handle different input formats
        single_input = args.single
        
        # Remove extension if provided
        if single_input.endswith('.json'):
            base_name = single_input[:-5]
        elif single_input.endswith('.pdf'):
            base_name = single_input[:-4]
        else:
            base_name = single_input
        
        # Handle full path or just filename
        if '/' in base_name or '\\' in base_name:
            # Full path provided
            base_path = Path(base_name)
            json_path = base_path.with_suffix('.json')
            pdf_path = base_path.with_suffix('.pdf')
        else:
            # Just filename - look in standard directories
            json_path = INPUT_DIR / f"{base_name}.json"
            pdf_path = PDF_DIR / f"{base_name}.pdf"
        
        # Check if files exist
        if not json_path.exists():
            print(f"[ERROR] JSON file not found: {json_path}")
            sys.exit(1)
        
        # Use override PDF if provided
        if args.pdf:
            pdf_path = Path(args.pdf)
        
        # PDF is optional
        if not pdf_path.exists():
            print(f"[WARNING] PDF not found: {pdf_path} - proceeding with JSON only")
            pdf_path = None
        
        print(f"[INFO] Processing:")
        print(f"  JSON: {json_path}")
        if pdf_path:
            print(f"  PDF:  {pdf_path}")
        
        out_path, error = extract_one(json_path, pdf_path)
        
        if error:
            print(f"[FAIL] {error}")
            sys.exit(1)
        else:
            print(f"[SUCCESS] Extraction complete")
    
    elif args.batch:
        print(f"[INFO] Processing batch from: {INPUT_DIR}")
        
        results = []
        for json_path in sorted(INPUT_DIR.glob("*.json")):
            pdf_path = PDF_DIR / f"{json_path.stem}.pdf"
            if not pdf_path.exists():
                pdf_path = None
            
            out_path, error = extract_one(json_path, pdf_path)
            results.append((json_path.name, error is None))
            
            if error:
                print(f"  ✗ Failed: {error}")
        
        # Summary
        success_count = sum(1 for _, success in results if success)
        print(f"\n[SUMMARY] Processed {len(results)} files")
        print(f"  Successful: {success_count}")
        print(f"  Failed: {len(results) - success_count}")
    
    else:
        parser.print_help()
        return
    
    # Open UI if requested
    if args.open_ui:
        try:
            import subprocess
            subprocess.run([sys.executable, "-m", "streamlit", "run", "tools/evidence_inspector_app.py"])
        except Exception as e:
            print(f"[WARNING] Could not launch UI: {e}")


if __name__ == "__main__":
    main()