#!/usr/bin/env python3
"""
Test JSON processing without calling OpenAI API.
"""

import json
import re
from pathlib import Path
from datetime import datetime, timezone

def _sanitize_candidate_json(candidate: str) -> str:
    """Remove trailing commas from JSON."""
    return re.sub(r",(\s*[}\]])", r"\1", candidate)

def _strip_code_fences(text: str) -> str:
    """Strip markdown code fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|```\s*$", "", cleaned, flags=re.IGNORECASE | re.MULTILINE)
    return cleaned

def _parse_json_from_text(text: str):
    """Parse JSON from text, handling various formats."""
    cleaned = _strip_code_fences(text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    
    # Try to find JSON object boundaries
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start:end + 1]
        candidate = _sanitize_candidate_json(candidate)
        return json.loads(candidate)
    raise ValueError("Could not parse JSON from text")

def test_json_parsing():
    """Test various JSON parsing scenarios."""
    
    test_cases = [
        # Case 1: Clean JSON
        ('{"source": {"document_id": "test", "ingest_date": "2025-08-08T00:00:00Z"}}', "Clean JSON"),
        
        # Case 2: JSON with code fences
        ('```json\n{"source": {"document_id": "test", "ingest_date": "2025-08-08T00:00:00Z"}}\n```', "JSON with code fences"),
        
        # Case 3: JSON with trailing comma
        ('{"source": {"document_id": "test", "ingest_date": "2025-08-08T00:00:00Z",},}', "JSON with trailing commas"),
        
        # Case 4: JSON with extra text
        ('Here is the JSON:\n{"source": {"document_id": "test", "ingest_date": "2025-08-08T00:00:00Z"}}\nEnd of JSON', "JSON with surrounding text"),
        
        # Case 5: Complex nested structure
        ('''
        {
            "source": {
                "document_id": "complex_test",
                "ingest_date": "2025-08-08T00:00:00Z"
            },
            "document": {
                "metadata": {
                    "title": "Test Title",
                    "year": 2024,
                    "authors": ["Author 1", "Author 2"],
                },
                "design": {
                    "type": "RCT",
                    "multicenter": true,
                },
            },
            "retrieval": {
                "keywords": ["keyword1", "keyword2",],
                "summary_tldr": "Test summary",
                "embedding_ref": {}
            },
        }
        ''', "Complex nested JSON with trailing commas"),
    ]
    
    for test_input, description in test_cases:
        print(f"\nTesting: {description}")
        print(f"Input length: {len(test_input)} chars")
        
        try:
            result = _parse_json_from_text(test_input)
            print(f"✓ Success! Parsed keys: {list(result.keys())}")
            
            # Validate structure
            if "source" in result and "document_id" in result["source"]:
                print(f"  - document_id: {result['source']['document_id']}")
            
        except Exception as e:
            print(f"✗ Failed: {e}")

def create_sample_output():
    """Create a sample output file that validates against the schema."""
    
    sample = {
        "source": {
            "document_id": "A Multicenter RCT of Zephyr Endobronchial Valv",
            "ingest_date": datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z'),
            "source_url": "https://example.com/paper.pdf",
            "file_sha256": "abc123def456",
            "license": "CC-BY-4.0"
        },
        "provenance_defaults": {
            "parser": "Adobe Extract API",
            "model": "gpt-4-turbo-preview",
            "extraction_version": "1.0"
        },
        "document": {
            "metadata": {
                "title": "A Multicenter Randomized Controlled Trial of Zephyr Endobronchial Valve Treatment in Heterogeneous Emphysema (LIBERATE)",
                "year": 2018,
                "authors": [
                    "Gerard J. Criner",
                    "Richard Sue",
                    "Shawn Wright",
                    "Mark Dransfield"
                ],
                "journal": "American Journal of Respiratory and Critical Care Medicine",
                "doi": "10.1164/rccm.201803-0590OC",
                "publication_date": "2018-05-24"
            },
            "design": {
                "type": "Randomized Controlled Trial",
                "prospective": True,
                "multicenter": True,
                "blinded": "Single-blind",
                "randomized": True,
                "duration": "12 months",
                "setting": "Hospital",
                "centers_involved": 24,
                "countries": ["United States"]
            },
            "population": {
                "total_enrolled": 190,
                "total_analyzed": 190,
                "inclusion_criteria": [
                    "Heterogeneous emphysema",
                    "FEV1 15-45% predicted",
                    "Hyperinflation (RV ≥ 175% predicted)"
                ],
                "exclusion_criteria": [
                    "Collateral ventilation",
                    "Active pulmonary infection",
                    "Unstable cardiac disease"
                ]
            }
        },
        "retrieval": {
            "keywords": [
                "emphysema",
                "endobronchial valve",
                "bronchoscopic lung volume reduction",
                "COPD",
                "heterogeneous emphysema",
                "Zephyr valve",
                "LIBERATE trial"
            ],
            "summary_tldr": "Multicenter RCT showing Zephyr endobronchial valves significantly improved lung function, exercise capacity, and quality of life in patients with heterogeneous emphysema without collateral ventilation.",
            "faq_nuggets": [
                {
                    "question": "What was the primary outcome?",
                    "answer": "The primary outcome was the between-group difference in percentage of patients with ≥15% improvement in FEV1 at 12 months."
                },
                {
                    "question": "What were the main results?",
                    "answer": "47.7% of valve group vs 16.8% of control group achieved ≥15% FEV1 improvement (p<0.001)"
                }
            ],
            "embedding_ref": {
                "vector_store": "chroma",
                "key": "liberate_trial_2018"
            }
        }
    }
    
    output_path = Path("data/outputs/sample_complete.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sample, f, indent=2, ensure_ascii=False)
    
    print(f"\nCreated sample output at: {output_path}")
    print(f"Output size: {len(json.dumps(sample))} characters")
    
    # Validate against schema
    schema_path = Path("schemas/medical_rag_chatbot_v1.schema.json")
    if schema_path.exists():
        try:
            import jsonschema
            with open(schema_path) as f:
                schema = json.load(f)
            jsonschema.validate(sample, schema)
            print("✓ Sample validates against schema!")
        except ImportError:
            print("⚠ jsonschema not installed, skipping validation")
        except Exception as e:
            print(f"✗ Schema validation error: {e}")

if __name__ == "__main__":
    print("=" * 60)
    print("Testing JSON Parsing Functions")
    print("=" * 60)
    test_json_parsing()
    
    print("\n" + "=" * 60)
    print("Creating Sample Complete Output")
    print("=" * 60)
    create_sample_output()