#!/usr/bin/env python3
"""
Enhanced Textbook Chapter Extractor using GPT-5
Extracts structured content including tables, algorithms, clinical guidelines, and procedures
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib
import argparse
from dotenv import load_dotenv
from openai import OpenAI

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Schema for textbook chapter extraction
TEXTBOOK_SCHEMA = {
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
        "clinical_content": {
            "type": "object",
            "properties": {
                "procedures": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "indications": {"type": "array", "items": {"type": "string"}},
                            "contraindications": {"type": "array", "items": {"type": "string"}},
                            "steps": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "step_number": {"type": "integer"},
                                        "description": {"type": "string"},
                                        "critical_points": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            },
                            "complications": {"type": "array", "items": {"type": "string"}},
                            "page_range": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "algorithms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "purpose": {"type": "string"},
                            "decision_points": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "question": {"type": "string"},
                                        "options": {"type": "array", "items": {"type": "string"}},
                                        "actions": {"type": "array", "items": {"type": "string"}}
                                    }
                                }
                            },
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
                            "category": {"type": "string"},
                            "recommendation_grade": {"type": "string"},
                            "evidence_level": {"type": "string"},
                            "details": {"type": "string"},
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
                            "indications": {"type": "array", "items": {"type": "string"}},
                            "dosage": {"type": "string"},
                            "contraindications": {"type": "array", "items": {"type": "string"}},
                            "side_effects": {"type": "array", "items": {"type": "string"}},
                            "page": {"type": "string"}
                        }
                    }
                }
            }
        },
        "structured_data": {
            "type": "object",
            "properties": {
                "tables": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "table_id": {"type": "string"},
                            "title": {"type": "string"},
                            "description": {"type": "string"},
                            "headers": {"type": "array", "items": {"type": "string"}},
                            "rows": {
                                "type": "array",
                                "items": {"type": "array", "items": {"type": "string"}}
                            },
                            "footnotes": {"type": "array", "items": {"type": "string"}},
                            "clinical_relevance": {"type": "string"},
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
                            "page": {"type": "string"}
                        }
                    }
                },
                "boxes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "box_id": {"type": "string"},
                            "title": {"type": "string"},
                            "type": {"type": "string"},
                            "content": {"type": "string"},
                            "page": {"type": "string"}
                        }
                    }
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
                    "examination_findings": {"type": "string"},
                    "investigations": {"type": "array", "items": {"type": "string"}},
                    "diagnosis": {"type": "string"},
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
                    "page": {"type": "string"}
                }
            }
        },
        "summary": {
            "type": "object",
            "properties": {
                "chapter_summary": {"type": "string"},
                "clinical_pearls": {"type": "array", "items": {"type": "string"}},
                "practice_recommendations": {"type": "array", "items": {"type": "string"}},
                "future_directions": {"type": "string"}
            }
        }
    },
    "required": ["chapter_metadata", "clinical_content", "structured_data", "summary"]
}


def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract text from PDF using PyMuPDF"""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(str(pdf_path))
        text = ""
        for page_num in range(len(doc)):
            page = doc[page_num]
            text += f"\n[PAGE {page_num + 1}]\n"
            text += page.get_text()
        doc.close()
        return text
    except ImportError:
        print("PyMuPDF not installed. Install with: pip install PyMuPDF")
        return ""
    except Exception as e:
        print(f"Error extracting PDF text: {e}")
        return ""


def extract_text_from_json(json_path: Path) -> str:
    """Extract text from existing chapter JSON"""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        text = ""
        if 'content' in data and 'text_units' in data['content']:
            for unit in data['content']['text_units']:
                if 'provenance' in unit:
                    page = unit['provenance'].get('page', '?')
                    text += f"\n[PAGE {page}]\n"
                text += unit.get('text', '') + "\n"
        return text
    except Exception as e:
        print(f"Error reading JSON: {e}")
        return ""


def create_extraction_prompt(text: str, chapter_title: str) -> str:
    """Create the extraction prompt for GPT-5"""
    return f"""Extract structured medical education content from this textbook chapter: "{chapter_title}"

Focus on extracting:
1. Learning objectives and key points
2. Clinical procedures with step-by-step instructions
3. Diagnostic/treatment algorithms and decision trees
4. Clinical guidelines and recommendations with evidence grades
5. Tables with clinical data (lab values, drug dosages, differential diagnoses)
6. Clinical cases with presentations and management
7. Important definitions and terminology
8. Drug information including dosages and contraindications
9. Boxes/callouts with clinical pearls or warnings

For each extracted element, include the page number where it appears.

For procedures, extract:
- Complete step-by-step instructions
- Indications and contraindications
- Required equipment
- Potential complications
- Tips and tricks

For algorithms, capture:
- Decision points and branches
- Criteria for each pathway
- Final outcomes/actions

For tables, preserve:
- All headers and row data
- Clinical relevance
- Reference ranges if present

For clinical guidelines:
- Recommendation grade (A, B, C, etc.)
- Level of evidence (I, II, III, etc.)
- Specific clinical scenarios

TEXT:
{text[:100000]}  # Limit to ~100k chars for context window
"""


def extract_chapter_content(
    input_path: Path,
    chapter_title: str,
    use_existing_json: bool = False
) -> Dict[str, Any]:
    """Extract structured content from textbook chapter using GPT-5"""
    
    # Extract text
    if use_existing_json and input_path.suffix == '.json':
        text = extract_text_from_json(input_path)
    else:
        text = extract_text_from_pdf(input_path)
    
    if not text:
        raise ValueError(f"No text extracted from {input_path}")
    
    # Create prompt
    prompt = create_extraction_prompt(text, chapter_title)
    
    print(f"Extracting content from: {chapter_title}")
    print(f"Text length: {len(text)} characters")
    
    try:
        # Call GPT-5 with structured output
        response = client.chat.completions.create(
            model="gpt-5-2025-08-07",
            messages=[
                {"role": "system", "content": "You are a medical education content extractor. Extract structured information from textbook chapters, focusing on clinical procedures, algorithms, guidelines, and educational content."},
                {"role": "user", "content": prompt}
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "textbook_extraction",
                    "schema": TEXTBOOK_SCHEMA,
                    "strict": True
                }
            },
            temperature=0.1,
            max_tokens=16384
        )
        
        # Parse the response
        extracted_data = json.loads(response.choices[0].message.content)
        
        # Add metadata
        extracted_data['extraction_metadata'] = {
            'source_file': str(input_path),
            'extraction_date': datetime.utcnow().isoformat() + 'Z',
            'model': 'gpt-5-2025-08-07',
            'text_length': len(text),
            'file_hash': hashlib.sha256(text.encode()).hexdigest()
        }
        
        return extracted_data
        
    except Exception as e:
        print(f"Error during GPT-5 extraction: {e}")
        raise


def process_single_chapter(
    input_path: Path,
    output_dir: Path,
    chapter_title: Optional[str] = None,
    use_existing_json: bool = False
) -> Path:
    """Process a single textbook chapter"""
    
    # Determine chapter title
    if not chapter_title:
        chapter_title = input_path.stem.replace('_', ' ').replace('-', ' ')
    
    # Extract content
    extracted_data = extract_chapter_content(input_path, chapter_title, use_existing_json)
    
    # Save output
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_path.stem}_enhanced.json"
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Saved enhanced extraction to: {output_file}")
    
    # Print summary
    print(f"\nExtraction Summary for '{chapter_title}':")
    if 'clinical_content' in extracted_data:
        content = extracted_data['clinical_content']
        print(f"  - Procedures: {len(content.get('procedures', []))}")
        print(f"  - Algorithms: {len(content.get('algorithms', []))}")
        print(f"  - Guidelines: {len(content.get('clinical_guidelines', []))}")
        print(f"  - Drug info: {len(content.get('drug_information', []))}")
    if 'structured_data' in extracted_data:
        data = extracted_data['structured_data']
        print(f"  - Tables: {len(data.get('tables', []))}")
        print(f"  - Figures: {len(data.get('figures', []))}")
        print(f"  - Boxes: {len(data.get('boxes', []))}")
    if 'clinical_cases' in extracted_data:
        print(f"  - Clinical cases: {len(extracted_data['clinical_cases'])}")
    
    return output_file


def process_batch(input_dir: Path, output_dir: Path, use_existing_json: bool = False):
    """Process all chapters in a directory"""
    
    # Determine file pattern based on mode
    if use_existing_json:
        pattern = "*.json"
        source_dir = input_dir / "Chapter json"
    else:
        pattern = "*.pdf"
        source_dir = input_dir / "Chapter pdfs"
    
    files = list(source_dir.glob(pattern))
    
    if not files:
        print(f"No {pattern} files found in {source_dir}")
        return
    
    print(f"Found {len(files)} chapters to process")
    
    results = []
    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        try:
            output_file = process_single_chapter(
                file_path,
                output_dir,
                use_existing_json=use_existing_json
            )
            results.append({
                'input': str(file_path),
                'output': str(output_file),
                'status': 'success'
            })
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append({
                'input': str(file_path),
                'error': str(e),
                'status': 'failed'
            })
    
    # Save batch summary
    summary_file = output_dir / f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'total_processed': len(files),
            'successful': sum(1 for r in results if r['status'] == 'success'),
            'failed': sum(1 for r in results if r['status'] == 'failed'),
            'results': results
        }, f, indent=2)
    
    print(f"\n✓ Batch processing complete. Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(description='Extract structured content from textbook chapters using GPT-5')
    parser.add_argument(
        '--single',
        type=Path,
        help='Process a single chapter (PDF or JSON file)'
    )
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all chapters in the Textbooks directory'
    )
    parser.add_argument(
        '--use-json',
        action='store_true',
        help='Use existing JSON files instead of PDFs (for re-extraction)'
    )
    parser.add_argument(
        '--title',
        type=str,
        help='Chapter title (for single file processing)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('Textbooks/enhanced_extractions'),
        help='Output directory for enhanced extractions'
    )
    
    args = parser.parse_args()
    
    if not args.single and not args.batch:
        parser.error('Please specify either --single or --batch')
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment variables")
        print("Please set it in your .env file or environment")
        sys.exit(1)
    
    if args.single:
        process_single_chapter(
            args.single,
            args.output_dir,
            chapter_title=args.title,
            use_existing_json=args.use_json
        )
    elif args.batch:
        # Use the Textbooks directory
        textbooks_dir = Path(__file__).parent.parent / 'Textbooks'
        process_batch(
            textbooks_dir,
            args.output_dir,
            use_existing_json=args.use_json
        )


if __name__ == "__main__":
    main()