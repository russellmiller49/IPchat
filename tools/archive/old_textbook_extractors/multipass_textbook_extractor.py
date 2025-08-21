#!/usr/bin/env python3
"""
Multi-Pass Textbook Extractor for OpenEvidence-level Medical Chatbot
Extracts comprehensive content through multiple focused passes
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
from tqdm import tqdm

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Define extraction passes with focused prompts
EXTRACTION_PASSES = {
    "pass1_anatomy": {
        "name": "Anatomical Structures",
        "system_prompt": """You are an expert anatomist extracting detailed anatomical information from medical textbooks.
        Focus ONLY on anatomical structures, their descriptions, relationships, clinical significance, and variations.
        Be comprehensive and extract ALL anatomical content.""",
        "user_prompt": """Extract ALL anatomical structures and descriptions from this chapter.

For EACH anatomical structure, include:
- Name of the structure
- Detailed description (location, appearance, boundaries)
- Relationships to other structures
- Clinical significance and relevance
- Common variations and anomalies
- Blood supply and innervation if mentioned
- Page number

Include structures at all levels: organs, regions, tissues, spaces, landmarks, etc.

TEXT: {text}

Return a JSON object with key "anatomical_structures" containing an array of structures."""
    },
    
    "pass2_procedures": {
        "name": "Clinical Procedures",
        "system_prompt": """You are an expert interventional pulmonologist extracting detailed procedural information.
        Focus ONLY on clinical procedures, techniques, and interventions.
        Extract complete step-by-step instructions with all critical details.""",
        "user_prompt": """Extract ALL clinical procedures, techniques, and interventions from this chapter.

For EACH procedure, include:
- Procedure name
- Indications (all conditions/situations when performed)
- Contraindications (absolute and relative)
- Required equipment and materials
- Patient preparation and positioning
- Step-by-step instructions with:
  - Step number
  - Detailed description
  - Critical points and tips
  - Warnings and precautions
- Complications (with incidence rates if provided)
- Post-procedure care and monitoring
- Success rates and outcomes
- Alternative techniques
- Page numbers

Include ALL procedures: diagnostic, therapeutic, emergency, elective.

TEXT: {text}

Return a JSON object with key "procedures" containing an array of procedures."""
    },
    
    "pass3_diagnostics": {
        "name": "Diagnostic Approaches",
        "system_prompt": """You are an expert diagnostician extracting diagnostic methods, criteria, and classification systems.
        Focus ONLY on diagnostic approaches, tests, scoring systems, and classification methods.""",
        "user_prompt": """Extract ALL diagnostic approaches, tests, and classification systems from this chapter.

For EACH diagnostic method, include:
- Name of the diagnostic approach/test/classification
- Purpose and clinical applications
- Criteria or scoring system (complete details)
- How to perform or apply it
- Interpretation guidelines
- Sensitivity and specificity if provided
- Positive and negative predictive values
- Limitations and pitfalls
- When to use vs alternatives
- Page number

Include: physical exam techniques, imaging interpretations, classification systems (like Mallampati), 
scoring systems, diagnostic algorithms, laboratory interpretations.

TEXT: {text}

Return a JSON object with key "diagnostic_approaches" containing an array of diagnostic methods."""
    },
    
    "pass4_guidelines": {
        "name": "Guidelines & Algorithms",
        "system_prompt": """You are an expert in evidence-based medicine extracting clinical guidelines, recommendations, and treatment algorithms.
        Focus ONLY on guidelines, protocols, algorithms, and evidence-based recommendations.""",
        "user_prompt": """Extract ALL clinical guidelines, treatment algorithms, and protocols from this chapter.

For EACH guideline or algorithm, include:
- Title/name
- Condition or situation addressed
- Source organization if mentioned
- Recommendation grade (A, B, C, etc.)
- Level of evidence (I, II, III, etc.)
- Specific recommendations (complete list)
- Decision points and pathways
- Patient population/criteria
- Contraindications and exceptions
- Monitoring requirements
- Outcome measures
- Alternative approaches
- Page number

Include: treatment algorithms, diagnostic algorithms, management protocols, 
society guidelines, consensus recommendations.

TEXT: {text}

Return a JSON object with key "guidelines_algorithms" containing an array."""
    },
    
    "pass5_pharmacology": {
        "name": "Drug Information",
        "system_prompt": """You are a clinical pharmacologist extracting comprehensive drug and medication information.
        Focus ONLY on medications, drugs, dosing, and pharmacological treatments.""",
        "user_prompt": """Extract ALL drug and medication information from this chapter.

For EACH drug mentioned, include:
- Drug name (generic and brand)
- Drug class and mechanism of action
- Indications (all uses mentioned)
- Dosing information:
  - Adult dosing (with routes and frequencies)
  - Pediatric dosing if mentioned
  - Renal adjustment requirements
  - Hepatic adjustment requirements
  - Maximum doses
- Contraindications (absolute and relative)
- Side effects (common and serious)
- Drug interactions
- Monitoring requirements (labs, vitals, etc.)
- Special administration instructions
- Duration of therapy
- Page number

Include: systemic medications, topical agents, anesthetic drugs, contrast agents, etc.

TEXT: {text}

Return a JSON object with key "drugs" containing an array of drug information."""
    },
    
    "pass6_tables_data": {
        "name": "Tables & Structured Data",
        "system_prompt": """You are a data analyst extracting all tabular and structured information from medical textbooks.
        Focus on extracting complete table contents, lists, and structured data.""",
        "user_prompt": """Extract ALL tables, structured lists, and data from this chapter.

For EACH table or structured data element:
- Table/data title
- Complete headers/columns
- ALL rows of data (preserve exact values)
- Units of measurement
- Footnotes and legends
- Clinical interpretation notes
- Reference ranges if applicable
- Page number

Include: comparison tables, reference values, differential diagnoses tables, 
drug dosing tables, classification tables, outcome data, statistical results.

TABLE HINTS: {table_hints}

TEXT: {text}

Return a JSON object with key "tables" containing an array of complete tables."""
    },
    
    "pass7_education": {
        "name": "Educational Content",
        "system_prompt": """You are a medical educator extracting key educational content, clinical pearls, and learning points.
        Focus on educational elements, definitions, clinical tips, and summary content.""",
        "user_prompt": """Extract ALL educational content from this chapter.

Extract:
1. Learning objectives (all stated goals)
2. Key points and takeaways
3. Clinical pearls and tips
4. Important definitions and terminology
5. Common pitfalls and mistakes to avoid
6. Case examples and clinical scenarios
7. Summary points and conclusions
8. Practice recommendations
9. Future directions and controversies
10. Board exam high-yield facts

For each item include the content and page number.

TEXT: {text}

Return a JSON object with keys for each category above."""
    },
    
    "pass8_references": {
        "name": "References & Evidence",
        "system_prompt": """You are a research librarian extracting all references, citations, and evidence basis.
        Focus on capturing complete bibliographic information and evidence quality.""",
        "user_prompt": """Extract ALL references, citations, and evidence mentioned in this chapter.

For EACH reference:
- Complete citation
- Authors and year
- Journal or source
- DOI if provided
- PMID if provided
- Type of study (RCT, meta-analysis, cohort, etc.)
- Key findings mentioned
- Evidence level/quality
- How it's used in the chapter
- Page number where cited

Also extract:
- Evidence quality statements
- Controversial areas mentioned
- Areas needing more research

TEXT: {text}

Return a JSON object with key "references" containing an array of references."""
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


def perform_extraction_pass(
    text: str,
    pass_config: Dict,
    table_hints: Optional[List[Dict]] = None,
    model: str = "gpt-4o-mini"
) -> Dict[str, Any]:
    """Perform a single extraction pass with focused prompt"""
    
    # Prepare the user prompt
    user_prompt = pass_config["user_prompt"].format(
        text=text[:200000],  # Limit text per pass
        table_hints=json.dumps(table_hints, indent=2) if table_hints else "[]"
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": pass_config["system_prompt"]},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_completion_tokens=8192
        )
        
        result = json.loads(response.choices[0].message.content)
        return result
        
    except Exception as e:
        print(f"  Error in pass '{pass_config['name']}': {e}")
        return {}


def merge_extraction_results(results: List[Dict]) -> Dict[str, Any]:
    """Merge results from multiple extraction passes into comprehensive output"""
    
    merged = {
        "chapter_metadata": {
            "title": "",
            "authors": [],
            "chapter_number": "",
            "learning_objectives": [],
            "key_points": []
        },
        "anatomical_content": {
            "structures": []
        },
        "clinical_procedures": [],
        "diagnostic_approaches": [],
        "treatment_algorithms": [],
        "clinical_guidelines": [],
        "drug_information": [],
        "tables": [],
        "figures": [],
        "clinical_cases": [],
        "definitions": [],
        "clinical_pearls": [],
        "references": [],
        "summary": {
            "chapter_summary": "",
            "clinical_applications": [],
            "practice_recommendations": [],
            "future_directions": [],
            "controversies": []
        }
    }
    
    # Merge each pass result
    for pass_result in results:
        # Pass 1: Anatomy
        if "anatomical_structures" in pass_result:
            merged["anatomical_content"]["structures"].extend(
                pass_result.get("anatomical_structures", [])
            )
        
        # Pass 2: Procedures
        if "procedures" in pass_result:
            merged["clinical_procedures"].extend(
                pass_result.get("procedures", [])
            )
        
        # Pass 3: Diagnostics
        if "diagnostic_approaches" in pass_result:
            merged["diagnostic_approaches"].extend(
                pass_result.get("diagnostic_approaches", [])
            )
        
        # Pass 4: Guidelines & Algorithms
        if "guidelines_algorithms" in pass_result:
            for item in pass_result.get("guidelines_algorithms", []):
                if "algorithm" in item.get("title", "").lower() or "decision" in item.get("title", "").lower():
                    merged["treatment_algorithms"].append(item)
                else:
                    merged["clinical_guidelines"].append(item)
        
        # Pass 5: Pharmacology
        if "drugs" in pass_result:
            merged["drug_information"].extend(
                pass_result.get("drugs", [])
            )
        
        # Pass 6: Tables
        if "tables" in pass_result:
            merged["tables"].extend(
                pass_result.get("tables", [])
            )
        
        # Pass 7: Educational content
        if "learning_objectives" in pass_result:
            merged["chapter_metadata"]["learning_objectives"].extend(
                pass_result.get("learning_objectives", [])
            )
        if "key_points" in pass_result:
            merged["chapter_metadata"]["key_points"].extend(
                pass_result.get("key_points", [])
            )
        if "clinical_pearls" in pass_result:
            merged["clinical_pearls"].extend(
                pass_result.get("clinical_pearls", [])
            )
        if "definitions" in pass_result:
            merged["definitions"].extend(
                pass_result.get("definitions", [])
            )
        if "case_examples" in pass_result:
            merged["clinical_cases"].extend(
                pass_result.get("case_examples", [])
            )
        if "practice_recommendations" in pass_result:
            merged["summary"]["practice_recommendations"].extend(
                pass_result.get("practice_recommendations", [])
            )
        if "future_directions" in pass_result:
            merged["summary"]["future_directions"].extend(
                pass_result.get("future_directions", [])
            )
        if "controversies" in pass_result:
            merged["summary"]["controversies"].extend(
                pass_result.get("controversies", [])
            )
        
        # Pass 8: References
        if "references" in pass_result:
            merged["references"].extend(
                pass_result.get("references", [])
            )
    
    return merged


def extract_multipass(
    pdf_path: Path,
    adobe_json_path: Optional[Path] = None,
    chapter_title: Optional[str] = None,
    model: str = None,
    passes_to_run: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Extract comprehensive content using multiple focused passes"""
    
    # Use better model if available
    if not model:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        # Upgrade to better model if API key suggests premium access
        if os.getenv("OPENAI_API_KEY", "").startswith("sk-proj"):
            model = "gpt-4o"  # Use better model for premium accounts
    
    # Extract text from PDF
    print(f"📖 Extracting text from PDF...")
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError(f"No text extracted from {pdf_path}")
    
    # Extract table hints from Adobe JSON if available
    table_hints = []
    if adobe_json_path and adobe_json_path.exists():
        table_hints = extract_tables_from_adobe_json(adobe_json_path)
        print(f"📊 Found {len(table_hints)} table hints from Adobe JSON")
    
    # Use filename as title if not provided
    if not chapter_title:
        chapter_title = pdf_path.stem
    
    print(f"📚 Chapter: {chapter_title}")
    print(f"📝 Text length: {len(text)} characters")
    print(f"🤖 Model: {model}")
    print(f"🔄 Running {len(passes_to_run or EXTRACTION_PASSES)} extraction passes...")
    print()
    
    # Run extraction passes
    results = []
    passes = passes_to_run or list(EXTRACTION_PASSES.keys())
    
    for pass_key in tqdm(passes, desc="Extraction passes"):
        if pass_key not in EXTRACTION_PASSES:
            continue
            
        pass_config = EXTRACTION_PASSES[pass_key]
        print(f"  → Running pass: {pass_config['name']}")
        
        # Add table hints for table extraction pass
        hints = table_hints if "table" in pass_key.lower() else None
        
        result = perform_extraction_pass(text, pass_config, hints, model)
        
        # Count extracted items
        item_count = 0
        for key, value in result.items():
            if isinstance(value, list):
                item_count += len(value)
        
        print(f"    ✓ Extracted {item_count} items")
        results.append(result)
    
    # Merge results
    print(f"\n📋 Merging results from {len(results)} passes...")
    merged_data = merge_extraction_results(results)
    
    # Add metadata
    merged_data["chapter_metadata"]["title"] = chapter_title
    
    # Add extraction metadata
    merged_data["extraction_metadata"] = {
        "source_pdf": str(pdf_path),
        "adobe_json": str(adobe_json_path) if adobe_json_path else None,
        "extraction_date": datetime.now(UTC).isoformat(),
        "text_length": len(text),
        "model": model,
        "extractor_version": "multipass_v1.0",
        "passes_completed": len(results),
        "file_hash": hashlib.sha256(text.encode()).hexdigest()
    }
    
    return merged_data


def process_single_chapter(
    pdf_path: Path,
    adobe_json_path: Optional[Path] = None,
    output_dir: Path = Path('data/multipass_extractions'),
    chapter_title: Optional[str] = None,
    model: Optional[str] = None
):
    """Process a single textbook chapter with multi-pass extraction"""
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract content
    extracted_data = extract_multipass(
        pdf_path,
        adobe_json_path,
        chapter_title,
        model
    )
    
    # Save to file
    output_file = output_dir / f"{pdf_path.stem}_multipass.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(extracted_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Saved multi-pass extraction to: {output_file}")
    
    # Print detailed summary
    print(f"\n📊 Extraction Summary for '{chapter_title or pdf_path.stem}':")
    print("="*60)
    
    anat = len(extracted_data.get('anatomical_content', {}).get('structures', []))
    procs = len(extracted_data.get('clinical_procedures', []))
    diag = len(extracted_data.get('diagnostic_approaches', []))
    alg = len(extracted_data.get('treatment_algorithms', []))
    guide = len(extracted_data.get('clinical_guidelines', []))
    drugs = len(extracted_data.get('drug_information', []))
    tables = len(extracted_data.get('tables', []))
    cases = len(extracted_data.get('clinical_cases', []))
    defs = len(extracted_data.get('definitions', []))
    pearls = len(extracted_data.get('clinical_pearls', []))
    refs = len(extracted_data.get('references', []))
    
    total = anat + procs + diag + alg + guide + drugs + tables + cases + defs + pearls + refs
    
    print(f"  📍 Anatomical structures:  {anat:>3}")
    print(f"  🔧 Clinical procedures:    {procs:>3}")
    print(f"  🔍 Diagnostic approaches:  {diag:>3}")
    print(f"  📋 Treatment algorithms:   {alg:>3}")
    print(f"  📚 Clinical guidelines:    {guide:>3}")
    print(f"  💊 Drug information:       {drugs:>3}")
    print(f"  📊 Tables:                 {tables:>3}")
    print(f"  🏥 Clinical cases:        {cases:>3}")
    print(f"  📖 Definitions:           {defs:>3}")
    print(f"  💎 Clinical pearls:       {pearls:>3}")
    print(f"  📄 References:            {refs:>3}")
    print(f"  {'─'*25}")
    print(f"  📦 TOTAL ITEMS:           {total:>3}")
    
    return output_file


def process_batch(
    textbooks_dir: Path,
    output_dir: Path = Path('data/multipass_extractions'),
    model: Optional[str] = None
):
    """Process all textbook chapters in batch with multi-pass extraction"""
    
    # Find all PDF files
    pdf_files = sorted(textbooks_dir.glob("Chapter pdfs/*.pdf"))
    
    if not pdf_files:
        print(f"No PDF files found in {textbooks_dir / 'Chapter pdfs'}")
        return
    
    print(f"Found {len(pdf_files)} chapters to process")
    print(f"Using model: {model or os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}")
    print()
    
    successful = []
    failed = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        # Look for corresponding Adobe JSON
        adobe_json_name = pdf_path.stem + ".json"
        adobe_json_path = textbooks_dir / "Chapter json" / adobe_json_name
        
        if not adobe_json_path.exists():
            adobe_json_path = None
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(pdf_files)}] Processing: {pdf_path.stem}")
        print(f"{'='*60}")
        
        try:
            output_file = process_single_chapter(
                pdf_path,
                adobe_json_path,
                output_dir,
                chapter_title=pdf_path.stem,
                model=model
            )
            successful.append(pdf_path.stem)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            failed.append(pdf_path.stem)
    
    # Print final summary
    print("\n" + "="*60)
    print(f"🎉 Batch processing complete!")
    print(f"  ✅ Successful: {len(successful)} chapters")
    print(f"  ❌ Failed: {len(failed)} chapters")
    
    if failed:
        print(f"\nFailed chapters:")
        for chapter in failed:
            print(f"  - {chapter}")


def main():
    parser = argparse.ArgumentParser(
        description='Multi-Pass Textbook Extractor for OpenEvidence-level Medical Chatbot'
    )
    
    parser.add_argument(
        '--single',
        type=Path,
        help='Process a single PDF file'
    )
    parser.add_argument(
        '--adobe-json',
        type=Path,
        help='Adobe Extract JSON file (optional, for table hints)'
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
        '--model',
        type=str,
        default=None,
        help='OpenAI model to use (default: gpt-4o-mini or from env)'
    )
    parser.add_argument(
        '--passes',
        type=str,
        nargs='+',
        choices=list(EXTRACTION_PASSES.keys()),
        help='Specific passes to run (default: all)'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/multipass_extractions'),
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
            args.title,
            args.model
        )
    elif args.batch:
        textbooks_dir = Path('Textbooks')
        process_batch(textbooks_dir, args.output_dir, args.model)


if __name__ == "__main__":
    main()