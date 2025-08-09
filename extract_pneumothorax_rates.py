#!/usr/bin/env python3
"""
Extract pneumothorax rates from BLVR studies and create enhanced chunks.
"""
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

def extract_pneumothorax_rate(text: str) -> Optional[Tuple[str, str]]:
    """Extract pneumothorax rate from text using various patterns."""
    patterns = [
        # Pattern: X% pneumothorax or pneumothorax X%
        r'pneumothorax.*?(\d+\.?\d*)\s*%',
        r'(\d+\.?\d*)\s*%.*?pneumothorax',
        # Pattern: pneumothorax (n=X/Y)
        r'pneumothorax.*?\((\d+)/(\d+)',
        r'pneumothorax.*?(\d+)\s*of\s*(\d+)',
        # Pattern: X patients experienced pneumothorax
        r'(\d+)\s*patients?\s*(?:experienced|had|developed).*?pneumothorax',
    ]
    
    text_lower = text.lower()
    for pattern in patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        if matches:
            if isinstance(matches[0], tuple):
                # Calculate percentage from fraction
                try:
                    num, denom = matches[0]
                    percentage = (float(num) / float(denom)) * 100
                    return f"{percentage:.1f}%", f"{num}/{denom}"
                except:
                    pass
            else:
                return f"{matches[0]}%", matches[0]
    return None

def process_blvr_studies(data_dir: Path) -> List[Dict]:
    """Process BLVR studies and extract pneumothorax rates."""
    results = []
    
    # Keywords to identify BLVR studies
    blvr_keywords = ['endobronchial', 'valve', 'zephyr', 'spiration', 'blvr', 'ebv', 'coil']
    
    for json_file in data_dir.glob("*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Get title
            title = None
            if 'document' in data and 'metadata' in data['document']:
                title = data['document']['metadata'].get('title', '')
            elif 'metadata' in data:
                title = data['metadata'].get('title', '')
            
            if not title:
                continue
                
            # Check if it's a BLVR study
            title_lower = title.lower()
            is_blvr = any(kw in title_lower for kw in blvr_keywords)
            
            if not is_blvr:
                continue
            
            # Extract all text sections
            sections = data.get('document', {}).get('sections', {})
            all_text = ""
            
            for section_name, content in sections.items():
                if content:
                    all_text += f"\n{section_name}: {content}\n"
            
            # Look for pneumothorax rates
            ptx_rate = extract_pneumothorax_rate(all_text)
            
            # Get study metadata
            meta = data.get('document', {}).get('metadata', {}) or data.get('metadata', {})
            
            result = {
                'study_id': json_file.stem,
                'title': title,
                'year': meta.get('year'),
                'authors': meta.get('authors', []),
                'pneumothorax_rate': ptx_rate[0] if ptx_rate else None,
                'pneumothorax_details': ptx_rate[1] if ptx_rate else None,
                'has_pneumothorax_data': ptx_rate is not None
            }
            
            # Find context around pneumothorax mentions
            if 'pneumothorax' in all_text.lower():
                # Extract sentences mentioning pneumothorax
                sentences = re.split(r'[.!?]', all_text)
                ptx_sentences = [s.strip() for s in sentences if 'pneumothorax' in s.lower()]
                result['pneumothorax_context'] = ptx_sentences[:3]  # First 3 mentions
            
            results.append(result)
            
            if ptx_rate:
                print(f"Found: {title[:60]}...")
                print(f"  Pneumothorax rate: {ptx_rate[0]} ({ptx_rate[1]})")
                
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
            continue
    
    return results

def create_summary_chunk(results: List[Dict]) -> str:
    """Create a summary chunk with all pneumothorax rates."""
    summary = "PNEUMOTHORAX RATES IN BLVR STUDIES:\n\n"
    
    studies_with_rates = [r for r in results if r['pneumothorax_rate']]
    studies_with_rates.sort(key=lambda x: x.get('year', 0), reverse=True)
    
    for study in studies_with_rates:
        title = study['title'][:80]
        year = study.get('year', 'Unknown')
        rate = study['pneumothorax_rate']
        authors = study['authors'][0] if study['authors'] else 'Unknown'
        
        summary += f"• {authors} {year}: {rate} pneumothorax rate\n"
        summary += f"  Study: {title}\n\n"
    
    # Add overall summary
    if studies_with_rates:
        rates = []
        for s in studies_with_rates:
            try:
                rate_str = s['pneumothorax_rate'].replace('%', '')
                rates.append(float(rate_str))
            except:
                pass
        
        if rates:
            summary += f"\nSummary: Pneumothorax rates range from {min(rates):.1f}% to {max(rates):.1f}%\n"
            summary += f"Average: {sum(rates)/len(rates):.1f}% across {len(rates)} studies\n"
    
    return summary

def main():
    data_dir = Path("data/oe_final_outputs")
    
    print("Extracting pneumothorax rates from BLVR studies...")
    results = process_blvr_studies(data_dir)
    
    print(f"\nProcessed {len(results)} BLVR studies")
    print(f"Found pneumothorax data in {sum(1 for r in results if r['has_pneumothorax_data'])} studies")
    
    # Create summary
    summary = create_summary_chunk(results)
    print("\n" + "="*60)
    print(summary)
    
    # Save enhanced data
    output_file = Path("data/pneumothorax_rates.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved pneumothorax data to {output_file}")
    
    # Create an additional chunk file specifically for pneumothorax queries
    chunk_file = Path("data/chunks/pneumothorax_summary.jsonl")
    chunk = {
        "chunk_id": "pneumothorax_summary#0",
        "document_id": "pneumothorax_summary",
        "text": summary,
        "source": "summary",
        "pages": [],
        "section_path": ["pneumothorax_rates"],
        "table_number": None,
        "figure_number": None,
        "trial_signals": {}
    }
    
    # Append to chunks file
    with open(chunk_file, 'w') as f:
        f.write(json.dumps(chunk) + '\n')
    
    print(f"Created summary chunk at {chunk_file}")

if __name__ == "__main__":
    main()