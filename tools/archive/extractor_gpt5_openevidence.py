#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-5 Medical Evidence Extractor - OpenEvidence Level
Extracts structured medical evidence with complete provenance tracking
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
SCHEMA_PATH = Path("schemas/medical_evidence_openevidence.schema.json")

INPUT_DIR = Path("data/input_articles")
PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/openevidence_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_LOG = Path("last_openevidence_output.json")


def read_json(path: Path) -> Dict[str, Any]:
    """Read JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def extract_text_with_pages(adobe_json: Dict[str, Any]) -> Tuple[str, Dict[str, List[int]]]:
    """Extract text from Adobe JSON with page tracking."""
    text_parts = []
    page_refs = {}
    current_page = 1
    
    if "elements" in adobe_json:
        for element in adobe_json.get("elements", []):
            # Track page numbers
            if element.get("Page"):
                current_page = element["Page"]
            
            # Extract text
            text = None
            if element.get("Text"):
                text = element["Text"]
            elif element.get("attributes", {}).get("TextContent"):
                text = element["attributes"]["TextContent"]
            
            if text:
                text_parts.append(text)
                # Store page reference for this text
                if text not in page_refs:
                    page_refs[text] = []
                page_refs[text].append(current_page)
    
    # Also extract from direct fields
    for key in ["title", "authors", "abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
        val = adobe_json.get(key)
        if isinstance(val, str):
            text_parts.append(val)
        elif isinstance(val, list):
            text_parts.extend([str(x) for x in val if isinstance(x, str)])
    
    return "\n".join(text_parts)[:30000], page_refs


def read_pdf_with_pages(pdf_path: Path, page_limit: int = 50) -> Tuple[str, Dict[int, str]]:
    """Extract text from PDF with page mapping."""
    try:
        import fitz
    except ImportError:
        return "", {}
    
    try:
        doc = fitz.open(str(pdf_path))
        pages = min(len(doc), page_limit)
        full_text = []
        page_texts = {}
        
        for i in range(pages):
            page_text = doc.load_page(i).get_text("text")
            full_text.append(page_text)
            page_texts[i + 1] = page_text  # 1-indexed pages
        
        doc.close()
        return "".join(full_text)[:90000], page_texts
        
    except Exception as e:
        print(f"[WARNING] Failed to read PDF: {e}")
        return "", {}


def build_openevidence_prompt(doc_id: str, adobe_text: str, pdf_text: str, page_refs: Dict) -> str:
    """Build extraction prompt for OpenEvidence-level data."""
    current_date = datetime.now(timezone.utc).isoformat()
    
    prompt = f"""You are an expert medical evidence extractor creating OpenEvidence-level structured data.

CRITICAL REQUIREMENTS:
1. Extract EXACT numeric values with provenance (page numbers, table/figure IDs)
2. Normalize all outcomes with proper statistical measures
3. Map adverse events to MedDRA terms when possible
4. Include risk of bias assessment
5. Convert all percentages to raw numbers when possible

DOCUMENT CONTENT:
{adobe_text[:20000]}

{pdf_text[:20000] if pdf_text else ""}

Extract and return as JSON following this EXACT structure:

{{
  "source": {{
    "document_id": "{doc_id}",
    "ingest_date": "{current_date}",
    "trial_registration_id": "Extract NCT/ISRCTN/etc number",
    "pmid": "Extract if available",
    "doi": "Extract if available"
  }},
  
  "document": {{
    "metadata": {{
      "title": "Full paper title",
      "year": Extract as integer (e.g., 2024),
      "authors": ["List", "of", "authors"],
      "journal": "Journal name",
      "publication_date": "YYYY-MM-DD if available"
    }},
    "sections": {{
      "abstract": "Full abstract text",
      "methods": "Methods section text",
      "results": "Results section text",
      "discussion": "Discussion text",
      "conclusion": "Conclusion text"
    }}
  }},
  
  "pico": {{
    "population": {{
      "text": "Description of study population",
      "inclusion_criteria": ["List inclusion criteria"],
      "exclusion_criteria": ["List exclusion criteria"]
    }},
    "intervention": {{
      "text": "Intervention description",
      "details": "Detailed intervention protocol"
    }},
    "comparison": {{
      "text": "Control/comparator description",
      "details": "Control protocol details"
    }},
    "outcomes": [
      {{
        "name": "Primary outcome name",
        "type": "primary",
        "umls_cui": ""
      }}
    ]
  }},
  
  "design": {{
    "study_type": "RCT/observational/etc",
    "allocation": "randomized/non-randomized",
    "blinding": "double-blind/single-blind/open-label",
    "duration": {{
      "enrollment_period": "e.g., Jan 2020 - Dec 2021",
      "follow_up": "e.g., 12 months",
      "primary_endpoint_time": "e.g., 12 months"
    }},
    "sites_count": Extract number of sites,
    "countries": ["List", "countries"],
    "sample_size": {{
      "planned": Extract planned N,
      "enrolled": Extract enrolled N,
      "analyzed": Extract analyzed N
    }}
  }},
  
  "arms": [
    {{
      "arm_id": "intervention",
      "name": "Intervention arm name",
      "description": "Detailed description",
      "n_randomized": Extract exact N,
      "n_analyzed": Extract N analyzed,
      "n_completed": Extract N completed
    }},
    {{
      "arm_id": "control",
      "name": "Control arm name",
      "description": "Control description",
      "n_randomized": Extract exact N,
      "n_analyzed": Extract N analyzed,
      "n_completed": Extract N completed
    }}
  ],
  
  "outcomes_normalized": [
    {{
      "concept_id": "primary_outcome_12m",
      "name": "Exact outcome name from paper",
      "type": "binary or continuous",
      "outcome_type": "primary",
      "timepoint_iso8601": "P12M for 12 months",
      "timepoint_label": "12 months",
      "groups": [
        {{
          "arm_id": "intervention",
          "raw": {{
            "events": Extract number of events (for binary),
            "total": Total N in group,
            "mean": Extract mean (for continuous),
            "sd": Extract SD,
            "median": Extract median if given
          }}
        }},
        {{
          "arm_id": "control",
          "raw": {{
            "events": Extract events,
            "total": Total N
          }}
        }}
      ],
      "comparison": {{
        "ref_arm_id": "control",
        "measure": "risk_difference or mean_diff or hazard_ratio",
        "est": Extract point estimate as number,
        "ci_lower": Extract lower CI,
        "ci_upper": Extract upper CI,
        "ci_level": 0.95,
        "p_value": Extract exact p-value or "<0.001",
        "adjusted": false
      }},
      "provenance": {{
        "pages": [List page numbers],
        "tables": ["Table 2"],
        "quote": "Exact quote showing this data"
      }}
    }}
  ],
  
  "safety_normalized": [
    {{
      "event_name": "Exact adverse event name",
      "meddra": {{
        "soc": "System organ class if identifiable",
        "pt": "Preferred term"
      }},
      "serious": true or false,
      "groups": [
        {{
          "arm_id": "intervention",
          "events": Number of events,
          "patients": Number of patients with event,
          "percentage": Extract percentage
        }},
        {{
          "arm_id": "control",
          "events": Number,
          "patients": Number,
          "percentage": Percentage
        }}
      ],
      "period": "e.g., 0-45 days",
      "management": "How it was managed if described",
      "provenance": {{
        "pages": [Page numbers],
        "tables": ["Safety table name"],
        "quote": "Supporting quote"
      }}
    }}
  ],
  
  "risk_of_bias": {{
    "tool": "RoB 2",
    "overall_judgment": "low/some_concerns/high",
    "domains": [
      {{
        "name": "Randomization process",
        "judgment": "low/some_concerns/high",
        "support_for_judgment": "Evidence from paper"
      }},
      {{
        "name": "Deviations from intended interventions",
        "judgment": "low/some_concerns/high",
        "support_for_judgment": "Evidence"
      }},
      {{
        "name": "Missing outcome data",
        "judgment": "low/some_concerns/high",
        "support_for_judgment": "Evidence"
      }},
      {{
        "name": "Measurement of the outcome",
        "judgment": "low/some_concerns/high",
        "support_for_judgment": "Evidence"
      }},
      {{
        "name": "Selection of reported result",
        "judgment": "low/some_concerns/high",
        "support_for_judgment": "Evidence"
      }}
    ]
  }},
  
  "retrieval": {{
    "keywords": ["interventional", "pulmonology", "keywords"],
    "summary_tldr": "1-2 sentence summary of key finding",
    "clinical_relevance": "Brief statement of clinical importance"
  }}
}}

IMPORTANT:
- Use exact numbers from the paper, not calculated values
- Include page numbers for ALL numeric data
- Use null for missing values, not empty strings for numbers
- Ensure year is an integer, not a string
- P-values: use exact value or "<0.001" format
- For binary outcomes: always provide events/total
- For continuous: provide mean/SD or median/IQR as available

Return ONLY the JSON object:"""
    
    return prompt


def clean_json_response(text: str) -> str:
    """Clean GPT-5 response to extract JSON."""
    # Remove code fences
    if "```" in text:
        text = re.sub(r"```(?:json)?\s*", "", text)
        text = re.sub(r"```\s*$", "", text)
    
    # Fix common issues
    text = text.replace("−", "-")  # Replace en-dash with minus
    text = text.replace("–", "-")  # Replace em-dash with minus
    
    # Find JSON boundaries
    start = text.find("{")
    end = text.rfind("}") + 1
    
    if start >= 0 and end > start:
        json_str = text[start:end]
        # Remove trailing commas
        json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        return json_str
    
    return text.strip()


def call_gpt5_openevidence(prompt: str) -> Dict[str, Any]:
    """Call GPT-5 for OpenEvidence extraction."""
    if TEST_MODE:
        print("[INFO] Running in TEST MODE")
        return create_test_openevidence_response()
    
    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI not configured")
    
    print(f"[INFO] Calling GPT-5 for OpenEvidence extraction...")
    client = OpenAI(timeout=API_TIMEOUT)
    
    try:
        start_time = time.time()
        response = client.responses.create(
            model=MODEL,
            input=prompt
        )
        elapsed = time.time() - start_time
        print(f"[INFO] Response received in {elapsed:.1f}s")
        
        if hasattr(response, 'output_text'):
            content = response.output_text
        else:
            raise ValueError("Unexpected response format")
        
        # Save raw response
        RAW_LOG.write_text(content, encoding="utf-8")
        
        # Parse JSON
        json_str = clean_json_response(content)
        data = json.loads(json_str)
        
        # Post-process to ensure data types
        if "document" in data and "metadata" in data["document"]:
            if "year" in data["document"]["metadata"]:
                year_str = str(data["document"]["metadata"]["year"])
                # Extract just the year number
                year_match = re.search(r'\d{4}', year_str)
                if year_match:
                    data["document"]["metadata"]["year"] = int(year_match.group())
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        # Try to fix and retry
        try:
            json_str = clean_json_response(content)
            return json.loads(json_str)
        except:
            raise ValueError(f"Could not parse GPT-5 response: {e}")
    
    except Exception as e:
        print(f"[ERROR] GPT-5 call failed: {e}")
        raise


def create_test_openevidence_response() -> Dict[str, Any]:
    """Create test response in OpenEvidence format."""
    return {
        "source": {
            "document_id": "test_doc",
            "ingest_date": datetime.now(timezone.utc).isoformat(),
            "trial_registration_id": "NCT00000000"
        },
        "document": {
            "metadata": {
                "title": "Test Medical Study",
                "year": 2024,
                "authors": ["Test Author"],
                "journal": "Test Journal"
            },
            "sections": {
                "abstract": "Test abstract",
                "methods": "Test methods",
                "results": "Test results"
            }
        },
        "pico": {
            "population": {"text": "Test population"},
            "intervention": {"text": "Test intervention"},
            "comparison": {"text": "Test control"},
            "outcomes": [{"name": "Test outcome", "type": "primary"}]
        },
        "design": {
            "study_type": "RCT",
            "allocation": "randomized",
            "blinding": "double-blind"
        },
        "arms": [
            {"arm_id": "intervention", "name": "Test", "n_randomized": 100},
            {"arm_id": "control", "name": "Control", "n_randomized": 100}
        ],
        "outcomes_normalized": [],
        "safety_normalized": [],
        "retrieval": {
            "keywords": ["test"],
            "summary_tldr": "Test summary"
        }
    }


def extract_one_openevidence(json_path: Path, pdf_path: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
    """Extract OpenEvidence-level structured data."""
    print(f"\n[INFO] Processing: {json_path.name}")
    
    try:
        adobe_json = read_json(json_path)
    except Exception as e:
        return None, f"Failed to read JSON: {e}"
    
    # Extract text with page tracking
    adobe_text, adobe_page_refs = extract_text_with_pages(adobe_json)
    pdf_text, pdf_pages = "", {}
    
    if pdf_path and pdf_path.exists():
        pdf_text, pdf_pages = read_pdf_with_pages(pdf_path)
    
    doc_id = adobe_json.get("document_id", json_path.stem)
    
    print(f"  Document ID: {doc_id}")
    print(f"  Adobe text: {len(adobe_text)} chars")
    print(f"  PDF text: {len(pdf_text)} chars")
    
    if not adobe_text and not pdf_text:
        return None, "No text content found"
    
    # Build prompt
    prompt = build_openevidence_prompt(doc_id, adobe_text, pdf_text, adobe_page_refs)
    
    try:
        # Extract with GPT-5
        data = call_gpt5_openevidence(prompt)
        
        # Validate critical fields
        if "outcomes_normalized" not in data:
            data["outcomes_normalized"] = []
        if "safety_normalized" not in data:
            data["safety_normalized"] = []
        
        # Save output
        out_path = OUTPUT_DIR / f"{json_path.stem}.openevidence.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        # Summary statistics
        n_outcomes = len(data.get("outcomes_normalized", []))
        n_safety = len(data.get("safety_normalized", []))
        print(f"  ✓ Extracted: {n_outcomes} outcomes, {n_safety} safety events")
        print(f"  ✓ Saved to: {out_path.name}")
        
        return out_path, None
        
    except Exception as e:
        return None, f"Extraction failed: {e}"


def main():
    parser = argparse.ArgumentParser(description="GPT-5 OpenEvidence-Level Medical Evidence Extractor")
    parser.add_argument("--single", type=str, help="Process single file (name without extension)")
    parser.add_argument("--batch", action="store_true", help="Process all files")
    parser.add_argument("--test", action="store_true", help="Run in test mode")
    
    args = parser.parse_args()
    
    if args.test:
        global TEST_MODE
        TEST_MODE = True
        print("[INFO] Test mode enabled")
    
    if args.single:
        # Handle single file
        single_input = args.single
        
        # Remove extension if provided
        if single_input.endswith('.json'):
            base_name = single_input[:-5]
        elif single_input.endswith('.pdf'):
            base_name = single_input[:-4]
        else:
            base_name = single_input
        
        # Find files
        if '/' in base_name or '\\' in base_name:
            base_path = Path(base_name)
            json_path = base_path.with_suffix('.json')
            pdf_path = base_path.with_suffix('.pdf')
        else:
            json_path = INPUT_DIR / f"{base_name}.json"
            pdf_path = PDF_DIR / f"{base_name}.pdf"
        
        if not json_path.exists():
            print(f"[ERROR] JSON not found: {json_path}")
            sys.exit(1)
        
        if not pdf_path.exists():
            pdf_path = None
        
        out_path, error = extract_one_openevidence(json_path, pdf_path)
        
        if error:
            print(f"[FAIL] {error}")
            sys.exit(1)
        else:
            print(f"\n[SUCCESS] OpenEvidence extraction complete!")
    
    elif args.batch:
        print(f"[INFO] Batch processing from: {INPUT_DIR}")
        print("="*60)
        
        results = []
        for json_path in sorted(INPUT_DIR.glob("*.json")):
            pdf_path = PDF_DIR / f"{json_path.stem}.pdf"
            if not pdf_path.exists():
                pdf_path = None
            
            out_path, error = extract_one_openevidence(json_path, pdf_path)
            results.append((json_path.name, error is None))
            
            if error:
                print(f"  ✗ Failed: {error}")
        
        # Summary
        print("\n" + "="*60)
        success_count = sum(1 for _, success in results if success)
        print(f"EXTRACTION SUMMARY")
        print(f"  Total: {len(results)} files")
        print(f"  Success: {success_count}")
        print(f"  Failed: {len(results) - success_count}")
        print(f"  Output: {OUTPUT_DIR}")
        print("="*60)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()