#!/usr/bin/env python3
"""
Enhanced Textbook Extractor - Combines best of both extraction approaches
For OpenEvidence-level medical chatbot with comprehensive clinical content
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, UTC
import hashlib
import argparse
from dotenv import load_dotenv
from openai import OpenAI
import fitz  # PyMuPDF

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Enhanced schema combining structured format with comprehensive content types
ENHANCED_TEXTBOOK_SCHEMA = {
    "type": "object",
    "properties": {
        "chapter_metadata": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "authors": {"type": "array", "items": {"type": "string"}},
                "chapter_number": {"type": "string"},
                "learning_objectives": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "objective": {"type": "string"},
                            "page": {"type": "string"}
                        }
                    }
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "point": {"type": "string"},
                            "page": {"type": "string"}
                        }
                    }
                }
            }
        },
        "anatomical_content": {
            "type": "object",
            "properties": {
                "structures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "relationships": {"type": "array", "items": {"type": "string"}},
                            "clinical_significance": {"type": "string"},
                            "variations": {"type": "array", "items": {"type": "string"}},
                            "page": {"type": "string"}
                        }
                    }
                }
            }
        },
        "clinical_procedures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "indications": {"type": "array", "items": {"type": "string"}},
                    "contraindications": {"type": "array", "items": {"type": "string"}},
                    "equipment": {"type": "array", "items": {"type": "string"}},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "step_number": {"type": "integer"},
                                "description": {"type": "string"},
                                "critical_points": {"type": "array", "items": {"type": "string"}},
                                "warnings": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "complications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "complication": {"type": "string"},
                                "frequency": {"type": "string"},
                                "management": {"type": "string"}
                            }
                        }
                    },
                    "post_procedure_care": {"type": "array", "items": {"type": "string"}},
                    "success_rates": {"type": "string"},
                    "page_range": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "diagnostic_approaches": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "purpose": {"type": "string"},
                    "criteria": {"type": "array", "items": {"type": "string"}},
                    "interpretation": {"type": "string"},
                    "sensitivity": {"type": "string"},
                    "specificity": {"type": "string"},
                    "limitations": {"type": "array", "items": {"type": "string"}},
                    "page": {"type": "string"}
                }
            }
        },
        "treatment_algorithms": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "condition": {"type": "string"},
                    "decision_points": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {"type": "string"},
                                "options": {"type": "array", "items": {"type": "string"}},
                                "actions": {"type": "array", "items": {"type": "string"}},
                                "next_steps": {"type": "array", "items": {"type": "string"}}
                            }
                        }
                    },
                    "outcomes": {"type": "array", "items": {"type": "string"}},
                    "evidence_level": {"type": "string"},
                    "page": {"type": "string"}
                }
            }
        },
        "clinical_guidelines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "guideline": {"type": "string"},
                    "source": {"type": "string"},
                    "recommendation_grade": {"type": "string"},
                    "evidence_level": {"type": "string"},
                    "patient_population": {"type": "string"},
                    "key_recommendations": {"type": "array", "items": {"type": "string"}},
                    "contraindications": {"type": "array", "items": {"type": "string"}},
                    "monitoring": {"type": "array", "items": {"type": "string"}},
                    "page": {"type": "string"}
                }
            }
        },
        "drug_information": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "drug_name": {"type": "string"},
                    "drug_class": {"type": "string"},
                    "mechanism": {"type": "string"},
                    "indications": {"type": "array", "items": {"type": "string"}},
                    "dosing": {
                        "type": "object",
                        "properties": {
                            "adult": {"type": "string"},
                            "pediatric": {"type": "string"},
                            "renal_adjustment": {"type": "string"},
                            "hepatic_adjustment": {"type": "string"}
                        }
                    },
                    "contraindications": {"type": "array", "items": {"type": "string"}},
                    "side_effects": {
                        "type": "object",
                        "properties": {
                            "common": {"type": "array", "items": {"type": "string"}},
                            "serious": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "interactions": {"type": "array", "items": {"type": "string"}},
                    "monitoring": {"type": "array", "items": {"type": "string"}},
                    "page": {"type": "string"}
                }
            }
        },
        "tables": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "table_id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "headers": {"type": "array", "items": {"type": "string"}},
                    "rows": {"type": "array", "items": {"type": "array", "items": {"type": "string"}}},
                    "footnotes": {"type": "array", "items": {"type": "string"}},
                    "clinical_relevance": {"type": "string"},
                    "interpretation_guide": {"type": "string"},
                    "page": {"type": "string"}
                }
            }
        },
        "figures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "figure_id": {"type": "string"},
                    "title": {"type": "string"},
                    "caption": {"type": "string"},
                    "type": {"type": "string"},
                    "clinical_significance": {"type": "string"},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "page": {"type": "string"}
                }
            }
        },
        "clinical_cases": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "presentation": {"type": "string"},
                    "history": {"type": "string"},
                    "physical_exam": {"type": "string"},
                    "investigations": {
                        "type": "object",
                        "properties": {
                            "labs": {"type": "array", "items": {"type": "string"}},
                            "imaging": {"type": "array", "items": {"type": "string"}},
                            "other": {"type": "array", "items": {"type": "string"}}
                        }
                    },
                    "differential_diagnosis": {"type": "array", "items": {"type": "string"}},
                    "final_diagnosis": {"type": "string"},
                    "management": {"type": "string"},
                    "outcome": {"type": "string"},
                    "learning_points": {"type": "array", "items": {"type": "string"}},
                    "page_range": {"type": "array", "items": {"type": "string"}}
                }
            }
        },
        "definitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "term": {"type": "string"},
                    "definition": {"type": "string"},
                    "context": {"type": "string"},
                    "clinical_relevance": {"type": "string"},
                    "page": {"type": "string"}
                }
            }
        },
        "clinical_pearls": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "pearl": {"type": "string"},
                    "category": {"type": "string"},
                    "evidence_base": {"type": "string"},
                    "page": {"type": "string"}
                }
            }
        },
        "references": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "citation": {"type": "string"},
                    "doi": {"type": "string"},
                    "pmid": {"type": "string"},
                    "type": {"type": "string"},
                    "evidence_level": {"type": "string"},
                    "key_findings": {"type": "array", "items": {"type": "string"}},
                    "page": {"type": "string"}
                }
            }
        },
        "summary": {
            "type": "object",
            "properties": {
                "chapter_summary": {"type": "string"},
                "clinical_applications": {"type": "array", "items": {"type": "string"}},
                "practice_recommendations": {"type": "array", "items": {"type": "string"}},
                "future_directions": {"type": "array", "items": {"type": "string"}},
                "controversies": {"type": "array", "items": {"type": "string"}}
            }
        }
    }
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF with page markers"""
    try:
        doc = fitz.open(str(pdf_path))
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += f"\n[PAGE {page_num + 1}]\n"
            text += page.get_text()
        doc.close()
        return text
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""


def extract_tables_from_adobe_json(json_path: Path) -> List[Dict]:
    """Extract table metadata from Adobe Extract JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        tables = []
        for table in data.get('tables', []):
            tables.append({
                'table_id': str(table.get('ObjectID', '')),
                'page': table.get('Page'),
                'bounds': table.get('Bounds'),
                'file_path': table.get('filePaths', [None])[0]
            })
        return tables
    except Exception as e:
        print(f"Error extracting tables from Adobe JSON: {e}")
        return []


def create_enhanced_prompt(text: str, chapter_title: str, table_hints: List[Dict]) -> str:
    """Create comprehensive extraction prompt"""
    return f"""Extract comprehensive medical education content from this textbook chapter: "{chapter_title}"

EXTRACTION PRIORITIES:
1. ANATOMICAL CONTENT: Extract ALL anatomical structures, their descriptions, relationships, clinical significance, and variations
2. CLINICAL PROCEDURES: Capture complete procedures with step-by-step instructions, equipment, critical points, and complications with management
3. DIAGNOSTIC APPROACHES: Include all diagnostic methods, criteria, interpretation guides, sensitivity/specificity
4. TREATMENT ALGORITHMS: Extract decision trees, pathways, and evidence-based treatment protocols
5. CLINICAL GUIDELINES: Capture recommendations with grades, evidence levels, and specific patient populations
6. DRUG INFORMATION: Complete drug details including mechanisms, dosing (adult/pediatric/adjusted), interactions, monitoring
7. TABLES & FIGURES: Extract all tables and figures with full content, clinical relevance, and interpretation guides
8. CLINICAL CASES: Complete case presentations with investigations, differential diagnosis, management, outcomes
9. DEFINITIONS: Medical terms with comprehensive definitions and clinical context
10. CLINICAL PEARLS: High-yield clinical insights with evidence base
11. REFERENCES: Complete citations with evidence levels and key findings

CRITICAL INSTRUCTIONS:
- Include page numbers for EVERY extracted item
- For procedures: Include ALL steps with critical points and warnings
- For drugs: Include complete dosing information for all populations
- For tables: Extract complete content including all rows and clinical interpretation
- For anatomy: Include relationships between structures and clinical correlations
- For guidelines: Include recommendation grades (A, B, C) and evidence levels (I, II, III) when present
- Extract content comprehensively - capture ALL relevant information
- Preserve numerical values, percentages, and measurements exactly
- Include success rates, complication rates, and outcome data when available

TABLE HINTS FROM ADOBE EXTRACT:
{json.dumps(table_hints, indent=2)}

CHAPTER TEXT (Extended - {len(text)} characters):
{text[:250000]}  # Increased to 250k for comprehensive extraction
"""


def extract_enhanced_content(
    pdf_path: Path,
    adobe_json_path: Optional[Path] = None,
    chapter_title: Optional[str] = None
) -> Dict[str, Any]:
    """Extract comprehensive structured content from textbook chapter"""
    
    # Extract text from PDF
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError(f"No text extracted from {pdf_path}")
    
    # Extract table hints from Adobe JSON if available
    table_hints = []
    if adobe_json_path and adobe_json_path.exists():
        table_hints = extract_tables_from_adobe_json(adobe_json_path)
    
    # Use filename as title if not provided
    if not chapter_title:
        chapter_title = pdf_path.stem
    
    # Create comprehensive prompt
    prompt = create_enhanced_prompt(text, chapter_title, table_hints)
    
    print(f"Extracting enhanced content from: {chapter_title}")
    print(f"Text length: {len(text)} characters")
    print(f"Table hints: {len(table_hints)} tables found")
    
    try:
        # Call OpenAI with enhanced prompt
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {
                    "role": "system", 
                    "content": """You are an expert medical educator and content extractor specializing in interventional pulmonology.
                    Extract comprehensive structured information from textbook chapters for use in an OpenEvidence-level medical chatbot.
                    Focus on clinical accuracy, completeness, and practical applicability.
                    Include ALL relevant medical information with proper context and evidence levels."""
                },
                {
                    "role": "user", 
                    "content": prompt + f"\n\nReturn a JSON object following this schema:\n{json.dumps(ENHANCED_TEXTBOOK_SCHEMA, indent=2)}"
                }
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=16384
        )
        
        # Parse the response
        extracted_data = json.loads(response.choices[0].message.content)
        
        # Add extraction metadata
        extracted_data['extraction_metadata'] = {
            'source_pdf': str(pdf_path),
            'adobe_json': str(adobe_json_path) if adobe_json_path else None,
            'extraction_date': datetime.now(UTC).isoformat(),
            'text_length': len(text),
            'model': os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            'extractor_version': 'enhanced_v1.0',
            'file_hash': hashlib.sha256(text.encode()).hexdigest()
        }
        
        return extracted_data
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        raise


def process_single_chapter(
    pdf_path: Path,
    adobe_json_path: Optional[Path] = None,
    output_dir: Path = Path('data/enhanced_extractions'),
    chapter_title: Optional[str] = None
):
    """Process a single textbook chapter"""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract content
    extracted_data = extract_enhanced_content(
        pdf_path,
        adobe_json_path,
        chapter_title
    )
    
    # Save to file
    output_file = output_dir / f"{pdf_path.stem}_enhanced.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved enhanced extraction to: {output_file}")
    
    # Print summary
    print(f"\nExtraction Summary for '{chapter_title or pdf_path.stem}':")
    print(f"  - Anatomical structures: {len(extracted_data.get('anatomical_content', {}).get('structures', []))}")
    print(f"  - Clinical procedures: {len(extracted_data.get('clinical_procedures', []))}")
    print(f"  - Diagnostic approaches: {len(extracted_data.get('diagnostic_approaches', []))}")
    print(f"  - Treatment algorithms: {len(extracted_data.get('treatment_algorithms', []))}")
    print(f"  - Clinical guidelines: {len(extracted_data.get('clinical_guidelines', []))}")
    print(f"  - Drug information: {len(extracted_data.get('drug_information', []))}")
    print(f"  - Tables: {len(extracted_data.get('tables', []))}")
    print(f"  - Clinical cases: {len(extracted_data.get('clinical_cases', []))}")
    print(f"  - Definitions: {len(extracted_data.get('definitions', []))}")
    print(f"  - Clinical pearls: {len(extracted_data.get('clinical_pearls', []))}")
    
    return output_file


def process_batch(
    textbooks_dir: Path,
    output_dir: Path = Path('data/enhanced_extractions')
):
    """Process all textbook chapters in batch"""
    
    # Find all PDF files
    pdf_files = sorted(textbooks_dir.glob("Chapter pdfs/*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {textbooks_dir / 'Chapter pdfs'}")
        return
    
    print(f"Found {len(pdf_files)} chapters to process")
    
    successful = []
    failed = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        # Look for corresponding Adobe JSON
        adobe_json_name = pdf_path.stem + ".json"
        adobe_json_path = textbooks_dir / "Chapter json" / adobe_json_name
        
        if not adobe_json_path.exists():
            adobe_json_path = None
        
        print(f"\n[{i}/{len(pdf_files)}] Processing: {pdf_path.stem}")
        
        try:
            output_file = process_single_chapter(
                pdf_path,
                adobe_json_path,
                output_dir,
                chapter_title=pdf_path.stem
            )
            successful.append(pdf_path.stem)
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            failed.append(pdf_path.stem)
    
    # Print summary
    print("\n" + "="*60)
    print(f"Batch processing complete:")
    print(f"  ✓ Successful: {len(successful)} chapters")
    print(f"  ✗ Failed: {len(failed)} chapters")
    
    if failed:
        print(f"\nFailed chapters:")
        for chapter in failed:
            print(f"  - {chapter}")


def main():
    parser = argparse.ArgumentParser(
        description='Enhanced Textbook Extractor for OpenEvidence-level Medical Chatbot'
    )
    
    parser.add_argument(
        '--single',
        type=Path,
        help='Process a single PDF file'
    )
    parser.add_argument(
        '--adobe-json',
        type=Path,
        help='Adobe Extract JSON file (optional, for table extraction)'
    )
    parser.add_argument(
        '--title',
        type=str,
        help='Chapter title (optional)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all chapters in Textbooks directory'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/enhanced_extractions'),
        help='Output directory for extractions'
    )
    
    args = parser.parse_args()
    
    if not args.single and not args.batch:
        parser.error('Please specify either --single or --batch')
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        sys.exit(1)
    
    if args.single:
        process_single_chapter(
            args.single,
            args.adobe_json,
            args.output_dir,
            args.title
        )
    elif args.batch:
        textbooks_dir = Path('Textbooks')
        process_batch(textbooks_dir, args.output_dir)


if __name__ == "__main__":
    main()