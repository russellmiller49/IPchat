#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GPT-5 Medical Evidence Extractor - OpenEvidence Final Production Version
Implements all OE-grade requirements for medical evidence synthesis
"""

import argparse
import json
import math
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List, Union

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
SCHEMA_PATH = Path("schemas/medical_evidence_oe_final.schema.json")

INPUT_DIR = Path("data/input_articles")
PDF_DIR = Path("data/raw_pdfs")
OUTPUT_DIR = Path("data/oe_final_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RAW_LOG = Path("last_oe_final_output.json")


def calculate_derived_measures(groups: List[Dict]) -> Dict[str, Any]:
    """Calculate derived effect measures from raw data."""
    derived = {}
    
    # Extract data for binary outcomes
    if len(groups) >= 2:
        # Assume first group is intervention, second is control
        int_events = groups[0].get("raw", {}).get("events")
        int_total = groups[0].get("raw", {}).get("total")
        ctrl_events = groups[1].get("raw", {}).get("events")
        ctrl_total = groups[1].get("raw", {}).get("total")
        
        if all(x is not None for x in [int_events, int_total, ctrl_events, ctrl_total]):
            # Risk in each group
            risk_int = int_events / int_total if int_total > 0 else 0
            risk_ctrl = ctrl_events / ctrl_total if ctrl_total > 0 else 0
            
            # Risk Ratio (RR)
            if risk_ctrl > 0:
                rr = risk_int / risk_ctrl
                derived["risk_ratio"] = {"est": round(rr, 3)}
            
            # Odds Ratio (OR)
            if int_total > int_events and ctrl_total > ctrl_events:
                odds_int = int_events / (int_total - int_events) if (int_total - int_events) > 0 else 0
                odds_ctrl = ctrl_events / (ctrl_total - ctrl_events) if (ctrl_total - ctrl_events) > 0 else 0
                if odds_ctrl > 0:
                    or_val = odds_int / odds_ctrl
                    derived["odds_ratio"] = {"est": round(or_val, 3)}
            
            # Absolute Risk Reduction (ARR) and NNT
            arr = risk_int - risk_ctrl
            derived["arr"] = round(arr, 3)
            
            if abs(arr) > 0:
                nnt = 1 / abs(arr)
                if arr > 0:  # Benefit
                    derived["nnt"] = round(nnt, 1)
                else:  # Harm
                    derived["nnh"] = round(nnt, 1)
    
    return derived


def parse_p_value(p_str: str) -> Tuple[Optional[float], Optional[str]]:
    """Parse p-value string into numeric value and operator."""
    if not p_str:
        return None, None
    
    p_str = str(p_str).strip()
    
    # Check for operators
    if p_str.startswith("<"):
        return float(p_str[1:].strip()), "<"
    elif p_str.startswith(">"):
        return float(p_str[1:].strip()), ">"
    elif p_str.startswith("≤") or p_str.startswith("<="):
        return float(re.sub(r'^[≤<=]+', '', p_str).strip()), "<="
    elif p_str.startswith("≥") or p_str.startswith(">="):
        return float(re.sub(r'^[≥>=]+', '', p_str).strip()), ">="
    else:
        # Try to parse as plain number
        try:
            return float(p_str), "="
        except:
            return None, None


def clean_numeric_value(val: Any) -> Union[int, float, None]:
    """Clean and convert values to proper numeric types."""
    if val is None or val == "":
        return None
    
    # Handle string representations
    if isinstance(val, str):
        val = val.replace("−", "-").replace("–", "-")  # Fix dashes
        val = val.strip()
        
        if not val or val.lower() in ["na", "n/a", "nr", "not reported"]:
            return None
        
        try:
            # Try integer first
            if "." not in val:
                return int(val)
            else:
                return float(val)
        except:
            return None
    
    # Already numeric
    if isinstance(val, (int, float)):
        return val
    
    return None


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
            if element.get("Page"):
                current_page = element["Page"]
            
            text = None
            if element.get("Text"):
                text = element["Text"]
            elif element.get("attributes", {}).get("TextContent"):
                text = element["attributes"]["TextContent"]
            
            if text:
                text_parts.append(text)
                if text not in page_refs:
                    page_refs[text] = []
                page_refs[text].append(current_page)
    
    # Extract from direct fields
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
            page_texts[i + 1] = page_text
        
        doc.close()
        return "".join(full_text)[:90000], page_texts
        
    except Exception as e:
        print(f"[WARNING] Failed to read PDF: {e}")
        return "", {}


def build_oe_final_prompt(doc_id: str, adobe_text: str, pdf_text: str) -> str:
    """Build extraction prompt with all OE-grade requirements."""
    current_date = datetime.now(timezone.utc).isoformat()
    
    prompt = f"""You are an expert medical evidence extractor creating OpenEvidence-grade structured data.

CRITICAL REQUIREMENTS:
1. ALL numbers must be numeric (not strings with units)
2. P-values: Split into p_value (number) and p_operator ("<", "=", etc.)
3. Extract exact event counts for binary outcomes to enable effect calculations
4. Include analysis metadata (model, covariates, population) per outcome
5. Map adverse events to MedDRA structure with seriousness criteria

DOCUMENT CONTENT:
{adobe_text[:20000]}

{pdf_text[:20000] if pdf_text else ""}

Extract following this EXACT structure with NUMERIC values:

{{
  "source": {{
    "document_id": "{doc_id}",
    "ingest_date": "{current_date}",
    "trial_registration_id": "NCT/ISRCTN number",
    "pmid": "PubMed ID if available",
    "doi": "DOI if available"
  }},
  
  "document": {{
    "metadata": {{
      "title": "Full title",
      "year": 2024,  // INTEGER not string!
      "authors": ["Author1", "Author2"],
      "journal": "Journal name",
      "doi": "Duplicate DOI here for UI",
      "pmid": "PMID here too"
    }},
    "sections": {{
      "abstract": "Full abstract",
      "methods": "Methods text",
      "results": "Results text"
    }}
  }},
  
  "pico": {{
    "population": {{
      "text": "Population description",
      "inclusion_criteria": ["List criteria"],
      "exclusion_criteria": ["List criteria"]
    }},
    "intervention": {{
      "text": "Intervention name",
      "details": "Protocol details"
    }},
    "comparison": {{
      "text": "Control/comparator",
      "details": "Control details"
    }},
    "outcomes": [
      {{"name": "Primary outcome", "type": "primary", "umls_cui": ""}}
    ]
  }},
  
  "design": {{
    "study_type": "RCT",
    "allocation": "randomized",
    "blinding": "double-blind",
    "sites_count": 24,  // NUMBER not string
    "countries": ["US", "UK"],
    "sample_size": {{
      "planned": 200,
      "enrolled": 190,
      "analyzed": 185
    }},
    "analysis_populations": [
      {{"name": "ITT", "description": "All randomized", "n": 190}},
      {{"name": "PP", "description": "Protocol compliant", "n": 170}}
    ]
  }},
  
  "arms": [
    {{
      "arm_id": "intervention",
      "name": "Treatment arm name",
      "n_randomized": 128,  // EXACT number
      "n_analyzed": 125,
      "n_completed": 120
    }},
    {{
      "arm_id": "control", 
      "name": "Control arm",
      "n_randomized": 62,
      "n_analyzed": 60,
      "n_completed": 58
    }}
  ],
  
  "outcomes_normalized": [
    {{
      "concept_id": "primary_fev1_responder_12m",
      "name": "FEV1 ≥15% improvement at 12 months",
      "type": "binary",
      "outcome_type": "primary",
      "timepoint_iso8601": "P12M",
      "timepoint_label": "12 months",
      "groups": [
        {{
          "arm_id": "intervention",
          "raw": {{
            "events": 61,  // EXACT count not percentage!
            "total": 128    // Total N in arm
          }}
        }},
        {{
          "arm_id": "control",
          "raw": {{
            "events": 10,
            "total": 62
          }}
        }}
      ],
      "comparison": {{
        "ref_arm_id": "control",
        "measure": "risk_difference",
        "est": 0.309,  // NUMBER not string
        "ci_lower": 0.186,
        "ci_upper": 0.432,
        "ci_level": 0.95,
        "p_value": 0.001,  // Just the number!
        "p_operator": "<",  // Operator separate!
        "adjusted": true
      }},
      "analysis": {{
        "model": "ANCOVA",
        "adjusted": true,
        "covariates": ["baseline FEV1", "center"],
        "population": "ITT",
        "missing_handling": "Last observation carried forward"
      }},
      "provenance": {{
        "pages": [1156],
        "tables": ["Table 2"],
        "table_number": 2,
        "quote": "47.7% (61/128) vs 16.1% (10/62), p<0.001"
      }}
    }},
    {{
      "concept_id": "fev1_absolute_change_12m",
      "name": "Absolute FEV1 change",
      "type": "continuous",
      "outcome_type": "secondary",
      "timepoint_iso8601": "P12M",
      "unit": "L",
      "unit_canonical": "liter",
      "groups": [
        {{
          "arm_id": "intervention",
          "raw": {{
            "mean": 0.106,  // NUMBER only
            "sd": 0.23,
            "total": 128
          }}
        }},
        {{
          "arm_id": "control",
          "raw": {{
            "mean": -0.003,
            "sd": 0.19,
            "total": 62
          }}
        }}
      ],
      "comparison": {{
        "ref_arm_id": "control",
        "measure": "mean_difference",
        "est": 0.109,
        "ci_lower": 0.068,
        "ci_upper": 0.150,
        "p_value": 0.001,
        "p_operator": "<",
        "adjusted": true
      }},
      "analysis": {{
        "model": "ANCOVA",
        "adjusted": true,
        "covariates": ["baseline value"],
        "population": "ITT"
      }},
      "provenance": {{
        "pages": [1156],
        "tables": ["Table 2"]
      }}
    }}
  ],
  
  "safety_normalized": [
    {{
      "event_name": "Pneumothorax",
      "meddra": {{
        "soc": "Respiratory, thoracic and mediastinal disorders",
        "pt": "Pneumothorax"
      }},
      "serious": true,
      "seriousness_criteria": ["hospitalization"],
      "groups": [
        {{
          "arm_id": "intervention",
          "events": 34,  // Total events if different from patients
          "patients": 34,  // Patients with ≥1 event
          "percentage": 26.6,  // As NUMBER
          "total": 128
        }},
        {{
          "arm_id": "control",
          "events": 0,
          "patients": 0,
          "percentage": 0,
          "total": 62
        }}
      ],
      "period": "0-45 days",
      "management": "Chest tube in 30/34; valve removal in 4/34",
      "provenance": {{
        "pages": [1158],
        "tables": ["Table 3"],
        "quote": "Pneumothorax in 34/128 (26.6%)"
      }}
    }}
  ],
  
  "risk_of_bias": {{
    "tool": "RoB 2",
    "overall_judgment": "low",
    "domains": [
      {{
        "name": "Randomization process",
        "judgment": "low",
        "support_for_judgment": "Central randomization with allocation concealment"
      }}
    ]
  }},
  
  "retrieval": {{
    "keywords": ["emphysema", "endobronchial valve", "bronchoscopy"],
    "summary_tldr": "Zephyr valves improved FEV1 by 47.7% vs 16.1% at 12 months",
    "clinical_relevance": "First FDA-approved valve for severe emphysema"
  }}
}}

REMEMBER:
- Events must be INTEGER counts, not percentages
- P-values split: p_value (number) + p_operator (string)
- Years as INTEGER not string
- Include analysis details per outcome
- Exact quotes with page numbers for provenance

Return ONLY the JSON:"""
    
    return prompt


def post_process_extraction(data: Dict[str, Any]) -> Dict[str, Any]:
    """Post-process extraction to ensure OE-grade quality."""
    
    # Fix year to integer
    if "document" in data and "metadata" in data["document"]:
        meta = data["document"]["metadata"]
        if "year" in meta:
            year_val = meta["year"]
            if isinstance(year_val, str):
                year_match = re.search(r'\d{4}', year_val)
                if year_match:
                    meta["year"] = int(year_match.group())
    
    # Process outcomes
    if "outcomes_normalized" in data:
        for outcome in data["outcomes_normalized"]:
            # Calculate derived measures for binary outcomes
            if outcome.get("type") == "binary" and "groups" in outcome:
                derived = calculate_derived_measures(outcome["groups"])
                if derived:
                    outcome["derived"] = derived
            
            # Fix p-values
            if "comparison" in outcome:
                comp = outcome["comparison"]
                if "p_value" in comp:
                    p_val = comp["p_value"]
                    if isinstance(p_val, str):
                        p_num, p_op = parse_p_value(p_val)
                        if p_num is not None:
                            comp["p_value"] = p_num
                            comp["p_operator"] = p_op or "="
                
                # Clean numeric values
                for field in ["est", "ci_lower", "ci_upper"]:
                    if field in comp:
                        comp[field] = clean_numeric_value(comp[field])
            
            # Clean raw data
            if "groups" in outcome:
                for group in outcome["groups"]:
                    if "raw" in group:
                        raw = group["raw"]
                        for field in ["events", "total"]:
                            if field in raw:
                                raw[field] = clean_numeric_value(raw[field])
                        for field in ["mean", "sd", "median"]:
                            if field in raw:
                                raw[field] = clean_numeric_value(raw[field])
    
    # Process safety events
    if "safety_normalized" in data:
        for event in data["safety_normalized"]:
            if "groups" in event:
                for group in event["groups"]:
                    for field in ["events", "patients", "percentage"]:
                        if field in group:
                            group[field] = clean_numeric_value(group[field])
    
    # Ensure numeric values in design
    if "design" in data:
        if "sites_count" in data["design"]:
            data["design"]["sites_count"] = clean_numeric_value(data["design"]["sites_count"])
        
        if "sample_size" in data["design"]:
            for field in ["planned", "enrolled", "analyzed"]:
                if field in data["design"]["sample_size"]:
                    data["design"]["sample_size"][field] = clean_numeric_value(
                        data["design"]["sample_size"][field]
                    )
    
    # Clean arms
    if "arms" in data:
        for arm in data["arms"]:
            for field in ["n_randomized", "n_analyzed", "n_completed"]:
                if field in arm:
                    arm[field] = clean_numeric_value(arm[field])
    
    return data


def call_gpt5_oe_final(prompt: str) -> Dict[str, Any]:
    """Call GPT-5 for OE-final extraction."""
    if TEST_MODE:
        print("[INFO] Running in TEST MODE")
        return create_test_oe_response()
    
    if not OPENAI_AVAILABLE or not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OpenAI not configured")
    
    print(f"[INFO] Calling GPT-5 for OE-final extraction...")
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
        
        # Clean response
        if "```" in content:
            content = re.sub(r"```(?:json)?\s*", "", content)
            content = re.sub(r"```\s*$", "", content)
        
        content = content.replace("−", "-").replace("–", "-")
        
        # Extract JSON
        start = content.find("{")
        end = content.rfind("}") + 1
        if start >= 0 and end > start:
            json_str = content[start:end]
            json_str = re.sub(r',(\s*[}\]])', r'\1', json_str)
        else:
            json_str = content.strip()
        
        data = json.loads(json_str)
        
        # Post-process for OE quality
        data = post_process_extraction(data)
        
        return data
        
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        raise ValueError(f"Could not parse GPT-5 response: {e}")
    
    except Exception as e:
        print(f"[ERROR] GPT-5 call failed: {e}")
        raise


def create_test_oe_response() -> Dict[str, Any]:
    """Create test response in OE-final format."""
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
            }
        },
        "pico": {
            "population": {"text": "Test population"},
            "intervention": {"text": "Test intervention"},
            "comparison": {"text": "Control"}
        },
        "arms": [
            {"arm_id": "intervention", "name": "Test", "n_randomized": 100},
            {"arm_id": "control", "name": "Control", "n_randomized": 100}
        ],
        "outcomes_normalized": [
            {
                "concept_id": "test_outcome",
                "name": "Test outcome",
                "type": "binary",
                "groups": [
                    {"arm_id": "intervention", "raw": {"events": 50, "total": 100}},
                    {"arm_id": "control", "raw": {"events": 25, "total": 100}}
                ],
                "comparison": {
                    "ref_arm_id": "control",
                    "measure": "risk_difference",
                    "est": 0.25,
                    "p_value": 0.01,
                    "p_operator": "<"
                },
                "analysis": {
                    "model": "Chi-square",
                    "population": "ITT"
                },
                "provenance": {
                    "pages": [1],
                    "quote": "Test quote"
                }
            }
        ],
        "safety_normalized": [],
        "retrieval": {
            "keywords": ["test"],
            "summary_tldr": "Test summary"
        }
    }


def extract_one_oe_final(json_path: Path, pdf_path: Optional[Path]) -> Tuple[Optional[Path], Optional[str]]:
    """Extract OE-final structured data."""
    print(f"\n[INFO] Processing: {json_path.name}")
    
    try:
        adobe_json = read_json(json_path)
    except Exception as e:
        return None, f"Failed to read JSON: {e}"
    
    # Extract text
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
    prompt = build_oe_final_prompt(doc_id, adobe_text, pdf_text)
    
    try:
        # Extract with GPT-5
        data = call_gpt5_oe_final(prompt)
        
        # Summary statistics
        n_outcomes = len(data.get("outcomes_normalized", []))
        n_safety = len(data.get("safety_normalized", []))
        n_derived = sum(1 for o in data.get("outcomes_normalized", []) if "derived" in o)
        
        print(f"  ✓ Extracted: {n_outcomes} outcomes ({n_derived} with derived measures)")
        print(f"  ✓ Safety events: {n_safety}")
        
        # Validate critical fields
        if n_outcomes > 0:
            outcome = data["outcomes_normalized"][0]
            if "comparison" in outcome:
                p_val = outcome["comparison"].get("p_value")
                p_op = outcome["comparison"].get("p_operator")
                if isinstance(p_val, (int, float)):
                    print(f"  ✓ P-value properly formatted: {p_val} (operator: {p_op})")
        
        # Save output
        out_path = OUTPUT_DIR / f"{json_path.stem}.oe_final.json"
        out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        
        print(f"  ✓ Saved to: {out_path.name}")
        
        return out_path, None
        
    except Exception as e:
        return None, f"Extraction failed: {e}"


def main():
    parser = argparse.ArgumentParser(description="GPT-5 OpenEvidence-Final Medical Evidence Extractor")
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
        
        # Remove extension
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
        
        out_path, error = extract_one_oe_final(json_path, pdf_path)
        
        if error:
            print(f"[FAIL] {error}")
            sys.exit(1)
        else:
            print(f"\n[SUCCESS] OE-final extraction complete!")
    
    elif args.batch:
        print(f"[INFO] Batch processing from: {INPUT_DIR}")
        print("="*60)
        
        results = []
        for json_path in sorted(INPUT_DIR.glob("*.json")):
            pdf_path = PDF_DIR / f"{json_path.stem}.pdf"
            if not pdf_path.exists():
                pdf_path = None
            
            out_path, error = extract_one_oe_final(json_path, pdf_path)
            results.append((json_path.name, error is None))
            
            if error:
                print(f"  ✗ Failed: {error}")
        
        # Summary
        print("\n" + "="*60)
        success_count = sum(1 for _, success in results if success)
        print(f"OE-FINAL EXTRACTION SUMMARY")
        print(f"  Total: {len(results)} files")
        print(f"  Success: {success_count}")
        print(f"  Failed: {len(results) - success_count}")
        print(f"  Output: {OUTPUT_DIR}")
        print("="*60)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()