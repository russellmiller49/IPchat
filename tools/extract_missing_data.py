#!/usr/bin/env python3
"""
Extract Missing Data from Adobe JSONs using PDFs
Handles truncated/incomplete Adobe Extract JSONs by filling gaps from PDFs
"""

import sys
import json
import PyPDF2
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import os
from openai import OpenAI
from dotenv import load_dotenv
import time
import signal

# Load environment variables
load_dotenv()

# Global flag for graceful shutdown
shutdown_requested = False

def signal_handler(signum, frame):
    """Handle interrupt signals gracefully"""
    global shutdown_requested
    print("\n\n[INTERRUPT] Shutdown requested. Finishing current file and exiting...")
    shutdown_requested = True

class AdobeDocumentParser:
    """Simple parser for Adobe Extract JSON files"""
    
    def parse_complete_document(self, json_path: Path) -> Dict[str, Any]:
        """Parse Adobe JSON and return structured document"""
        
        with open(json_path, 'r', encoding='utf-8') as f:
            adobe_json = json.load(f)
        
        # Extract metadata
        metadata = {
            'title': adobe_json.get('title', ''),
            'authors': adobe_json.get('authors', []),
            'journal': adobe_json.get('journal', ''),
            'year': adobe_json.get('year', ''),
            'doi': adobe_json.get('doi', '')
        }
        
        # Extract text from elements and organize into sections
        full_text_parts = []
        sections = {
            'abstract': [],
            'introduction': [],
            'methods': [],
            'results': [],
            'discussion': [],
            'conclusion': []
        }
        
        # Extract text from elements if available
        if 'elements' in adobe_json:
            current_section = None
            for element in adobe_json['elements']:
                text = element.get('Text', '')
                if not text:
                    continue
                
                # Clean up the text (remove excessive spaces, fix encoding issues)
                text = self._clean_text(text)
                full_text_parts.append(text)
                
                # Try to identify sections based on keywords
                text_lower = text.lower()
                if any(keyword in text_lower for keyword in ['background:', 'introduction:', 'abstract:']):
                    current_section = 'introduction'
                elif any(keyword in text_lower for keyword in ['methods:', 'method:', 'materials and methods:']):
                    current_section = 'methods'
                elif any(keyword in text_lower for keyword in ['results:', 'result:']):
                    current_section = 'results'
                elif any(keyword in text_lower for keyword in ['discussion:', 'discuss:']):
                    current_section = 'discussion'
                elif any(keyword in text_lower for keyword in ['conclusion:', 'conclusions:', 'summary:']):
                    current_section = 'conclusion'
                elif 'abstract' in text_lower and len(text) > 50:  # Likely abstract content
                    current_section = 'abstract'
                
                # Add text to current section
                if current_section and current_section in sections:
                    sections[current_section].append(text)
        
        # Also check for direct text fields
        for key in ["title", "authors", "abstract", "introduction", "methods", "results", "discussion", "conclusion"]:
            val = adobe_json.get(key)
            if isinstance(val, str):
                sections.get(key, []).append(val)
                full_text_parts.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str):
                        sections.get(key, []).append(item)
                        full_text_parts.append(item)
        
        # Check for content/text/body fields
        for k, v in adobe_json.items():
            if k in {"content", "text", "body"} and isinstance(v, str):
                full_text_parts.append(v)
        
        # Build full text
        full_text = '\n'.join(full_text_parts)
        
        # Extract tables
        tables = adobe_json.get('tables', [])
        
        # Try to extract title and authors from the first few elements if not already present
        if not metadata['title'] and full_text_parts:
            first_text = full_text_parts[0]
            # Look for title pattern (usually the first substantial text)
            if len(first_text) > 20 and not first_text.startswith('Abstract'):
                metadata['title'] = first_text[:200].strip()
        
        if not metadata['authors'] and len(full_text_parts) > 1:
            # Look for authors in the first few elements
            for i, text in enumerate(full_text_parts[:3]):
                if ',' in text and any(title in text.lower() for title in ['md', 'phd', 'dr', 'professor']):
                    # This might be authors
                    authors_text = text.strip()
                    if len(authors_text) < 500:  # Reasonable length for authors
                        metadata['authors'] = [author.strip() for author in authors_text.split(',') if author.strip()]
                        break
        
        return {
            'metadata': metadata,
            'sections': sections,
            'tables': tables,
            'full_text': full_text
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean up text by removing excessive spaces and fixing encoding issues"""
        # Remove excessive spaces
        text = ' '.join(text.split())
        # Fix common encoding issues
        text = text.replace('\u0001', '').replace('\u0003', '').replace('\u0192', '').replace('\u2044', '').replace('\u00a7', '')
        return text
    
    def _extract_section_text(self, adobe_json: Dict, section_name: str) -> List[str]:
        """Extract text for a specific section"""
        section_data = adobe_json.get(section_name, [])
        if isinstance(section_data, str):
            return [section_data]
        elif isinstance(section_data, list):
            return [str(item) for item in section_data if item]
        else:
            return []

class MissingDataExtractor:
    """Extract missing data from Adobe JSONs using PDF content"""
    
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.parser = AdobeDocumentParser()
    
    def analyze_adobe_json(self, json_path: Path) -> Dict[str, Any]:
        """Analyze Adobe JSON and identify what's missing"""
        
        print(f"  Analyzing Adobe JSON...")
        document = self.parser.parse_complete_document(json_path)
        
        analysis = {
            'has_title': bool(document['metadata'].get('title')),
            'has_authors': len(document['metadata'].get('authors', [])) > 0,
            'has_abstract': len(' '.join(document['sections'].get('abstract', []))) > 100,
            'has_methods': len(' '.join(document['sections'].get('methods', []))) > 100,
            'has_results': len(' '.join(document['sections'].get('results', []))) > 100,
            'has_tables': len(document['tables']) > 0 and any(t.get('rows') for t in document['tables']),
            'text_length': len(document['full_text']),
            'truncated_sections': []
        }
        
        # Check for truncation
        abstract = ' '.join(document['sections'].get('abstract', []))
        if abstract and not abstract.rstrip().endswith(('.', '!', '?', ')')):
            analysis['truncated_sections'].append('abstract')
        
        results = ' '.join(document['sections'].get('results', []))
        if results and ('stati' in results[-20:] or not results.rstrip().endswith(('.', '!', '?', ')'))):
            analysis['truncated_sections'].append('results')
        
        return document, analysis
    
    def extract_from_pdf(self, pdf_path: Path, max_pages: int = 30) -> str:
        """Extract text from PDF"""
        
        print(f"  Reading PDF ({max_pages} pages max)...")
        text = ""
        
        try:
            with open(pdf_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                num_pages = min(len(reader.pages), max_pages)
                
                for i in range(num_pages):
                    page_text = reader.pages[i].extract_text()
                    text += page_text + "\n"
                    
        except Exception as e:
            print(f"    Error reading PDF: {str(e)[:100]}")
            
        return text
    
    def fill_missing_data(self, adobe_doc: Dict, pdf_text: str, analysis: Dict) -> Dict:
        """Use GPT-4o to fill missing data"""
        
        print(f"  Extracting missing data with GPT-4o...")
        
        # Check for shutdown request
        if shutdown_requested:
            print(f"    [SHUTDOWN] Skipping API call")
            return adobe_doc
        
        # Prepare what we have
        existing_data = {
            'title': adobe_doc['metadata'].get('title', ''),
            'authors': adobe_doc['metadata'].get('authors', []),
            'abstract': ' '.join(adobe_doc['sections'].get('abstract', []))[:500],
            'has_tables': analysis['has_tables'],
            'truncated': analysis['truncated_sections']
        }
        
        # Create targeted prompt
        prompt = f"""You are extracting medical research data. 
Some data from Adobe Extract JSON is incomplete or missing.

EXISTING DATA FROM ADOBE JSON:
Title: {existing_data['title'] or '[MISSING]'}
Authors: {', '.join(existing_data['authors']) if existing_data['authors'] else '[MISSING]'}
Abstract: {existing_data['abstract'][:300] + '...' if existing_data['abstract'] else '[MISSING]'}
Truncated sections: {', '.join(existing_data['truncated']) if existing_data['truncated'] else 'none'}
Tables in JSON: {'yes but empty' if analysis['has_tables'] else 'no'}

PDF TEXT (for filling gaps):
{pdf_text[:10000]}

Extract comprehensive medical information. Return JSON:
{{
  "metadata": {{
    "title": "actual paper title from PDF if Adobe title is wrong",
    "authors": ["list of authors"],
    "journal": "journal name",
    "year": 2024,
    "doi": "DOI"
  }},
  "study_design": {{
    "type": "RCT/cohort/case-control/systematic review/guideline/technical",
    "randomized": true/false,
    "blinded": "double-blind/single-blind/open-label/not specified",
    "multicenter": true/false,
    "prospective": true/false
  }},
  "population": {{
    "total": 100,
    "intervention_n": 50,
    "control_n": 50,
    "inclusion_criteria": ["criteria 1", "criteria 2"],
    "exclusion_criteria": ["criteria 1", "criteria 2"]
  }},
  "outcomes": {{
    "primary": [
      {{
        "name": "primary outcome",
        "groups": [
          {{"group": "intervention", "n": 50, "value": 12.5, "sd": 2.3, "unit": "mm"}},
          {{"group": "control", "n": 50, "value": 10.1, "sd": 2.1, "unit": "mm"}}
        ],
        "difference": 2.4,
        "p_value": 0.03,
        "ci_95": "0.5 to 4.3",
        "source": "Table 2 or Results section"
      }}
    ],
    "secondary": []
  }},
  "adverse_events": [
    {{
      "event": "nausea",
      "intervention_n": 5,
      "intervention_percent": 10.0,
      "control_n": 2,
      "control_percent": 4.0,
      "serious": false
    }}
  ],
  "tables": [
    {{
      "number": "Table 1",
      "type": "demographics/outcomes/adverse events",
      "caption": "table caption",
      "key_data": ["Age: 65±10 years", "Male: 60%"]
    }}
  ],
  "key_findings": [
    "Main finding 1",
    "Main finding 2"
  ],
  "limitations": ["limitation 1"],
  "conclusions": ["main conclusion"],
  "data_quality": {{
    "adobe_json_complete": false,
    "missing_in_adobe": ["list what was missing"],
    "extracted_from_pdf": ["list what was extracted from PDF"],
    "confidence": "high/medium/low"
  }}
}}

IMPORTANT:
- Use Adobe JSON data when available and correct
- Fill missing data from PDF text
- Use actual numbers, not placeholders
- Note which data came from PDF vs Adobe JSON
- Return valid JSON"""

        try:
            # Check for shutdown request before API call
            if shutdown_requested:
                print(f"    [SHUTDOWN] Skipping API call")
                return adobe_doc
                
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Extract medical data, combining Adobe JSON with PDF content."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4000
            )
            
            # Check for shutdown request after API call
            if shutdown_requested:
                print(f"    [SHUTDOWN] Skipping processing of API response")
                return adobe_doc
            
            extraction = json.loads(response.choices[0].message.content)
            
            # Merge with Adobe data
            merged = self.merge_extractions(adobe_doc, extraction)
            
            return merged
            
        except Exception as e:
            print(f"    GPT-4 error: {str(e)[:200]}")
            return adobe_doc
    
    def merge_extractions(self, adobe_doc: Dict, gpt_extraction: Dict) -> Dict:
        """Merge Adobe and GPT-4 extractions intelligently"""
        
        merged = {
            'metadata': gpt_extraction.get('metadata', adobe_doc['metadata']),
            'study_design': gpt_extraction.get('study_design', {}),
            'population': gpt_extraction.get('population', {}),
            'outcomes': gpt_extraction.get('outcomes', {}),
            'adverse_events': gpt_extraction.get('adverse_events', []),
            'tables': gpt_extraction.get('tables', adobe_doc.get('tables', [])),
            'key_findings': gpt_extraction.get('key_findings', []),
            'limitations': gpt_extraction.get('limitations', []),
            'conclusions': gpt_extraction.get('conclusions', []),
            'extraction_info': {
                'method': 'adobe_json_plus_pdf',
                'adobe_text_length': len(adobe_doc.get('full_text', '')),
                'data_quality': gpt_extraction.get('data_quality', {}),
                'extraction_date': datetime.now().isoformat()
            }
        }
        
        # Preserve Adobe sections if they're good
        if adobe_doc.get('sections'):
            merged['sections'] = adobe_doc['sections']
        
        return merged

def main():
    """Process Adobe JSONs and fill missing data from PDFs"""
    
    # Set up signal handler for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, help='Number to process')
    parser.add_argument('--start', type=int, default=0, help='Starting index')
    parser.add_argument('--single', type=str, help='Process a single JSON file by name (e.g., "filename.json")')
    args = parser.parse_args()
    
    print("=" * 70)
    print("MISSING DATA EXTRACTION FROM ADOBE JSONs + PDFs")
    print("=" * 70)
    print("This fills gaps in truncated/incomplete Adobe Extract JSONs")
    print("-" * 70)
    print("Press Ctrl+C to stop gracefully")
    print("-" * 70)
    
    # Setup
    extractor = MissingDataExtractor()
    json_dir = Path('data/input_articles')
    pdf_dir = Path('data/raw_pdfs')
    output_dir = Path('data/complete_extractions')
    output_dir.mkdir(exist_ok=True)
    
    # Get files
    if args.single:
        # Process single file
        json_file = json_dir / args.single
        if not json_file.exists():
            print(f"Error: File {args.single} not found in {json_dir}")
            return
        json_files = [json_file]
    else:
        # Process all files or subset
        json_files = sorted(list(json_dir.glob('*.json')))
        
        if args.start:
            json_files = json_files[args.start:]
        if args.limit:
            json_files = json_files[:args.limit]
    
    print(f"\nProcessing {len(json_files)} files")
    print()
    
    results = []
    
    for i, json_file in enumerate(json_files, 1):
        # Check for shutdown request
        if shutdown_requested:
            print(f"\n[SHUTDOWN] Stopping at file {i-1}/{len(json_files)}")
            break
            
        print(f"[{i}/{len(json_files)}] {json_file.name}")
        print("-" * 50)
        
        # Find PDF
        pdf_name = json_file.stem + '.pdf'
        pdf_path = pdf_dir / pdf_name
        
        if not pdf_path.exists():
            print(f"  [SKIP] No PDF found")
            continue
        
        try:
            # Analyze Adobe JSON
            adobe_doc, analysis = extractor.analyze_adobe_json(json_file)
            
            # Report analysis
            print(f"  Adobe JSON analysis:")
            print(f"    Title: {'[OK]' if analysis['has_title'] else '[MISSING]'}")
            print(f"    Tables: {'[OK]' if analysis['has_tables'] else '[MISSING/EMPTY]'}")
            print(f"    Text: {analysis['text_length']} chars")
            
            if analysis['truncated_sections']:
                print(f"    Truncated: {', '.join(analysis['truncated_sections'])}")
            
            # Extract from PDF
            pdf_text = extractor.extract_from_pdf(pdf_path)
            print(f"    PDF: {len(pdf_text)} chars extracted")
            
            # Check for shutdown request before API call
            if shutdown_requested:
                print(f"\n[SHUTDOWN] Stopping before API call")
                break
            
            # Fill missing data
            complete_doc = extractor.fill_missing_data(adobe_doc, pdf_text, analysis)
            
            # Save
            output_file = output_dir / f"{json_file.stem}_complete.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(complete_doc, f, indent=2, ensure_ascii=False)
            
            print(f"  [SUCCESS] Saved to {output_file.name}")
            
            # Track results
            results.append({
                'file': json_file.name,
                'adobe_complete': not analysis['truncated_sections'] and analysis['has_tables'],
                'enhanced_from_pdf': True,
                'success': True
            })
            
        except Exception as e:
            print(f"  [ERROR] {str(e)[:100]}")
            results.append({
                'file': json_file.name,
                'error': str(e)[:200]
            })
        
        # Rate limiting (skip for single file)
        if not args.single and i % 3 == 0 and i < len(json_files) and not shutdown_requested:
            print("\n  [Pausing for rate limit...]")
            time.sleep(2)
        
        print()
    
    # Summary
    print("=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    
    successful = [r for r in results if r.get('success')]
    failed = len(results) - len(successful)
    
    print(f"Total processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {failed}")
    
    if successful:
        adobe_complete = sum(1 for r in successful if r.get('adobe_complete'))
        print(f"Adobe JSONs complete: {adobe_complete}")
        print(f"Enhanced from PDF: {len(successful) - adobe_complete}")
    
    print(f"\nOutput directory: {output_dir}")
    
    if shutdown_requested:
        print(f"\n[NOTE] Script was interrupted. {len(results)} files processed.")

if __name__ == "__main__":
    main()