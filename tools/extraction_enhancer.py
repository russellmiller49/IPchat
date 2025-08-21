#!/usr/bin/env python3
"""
Enhanced Post-Processing Module for Textbook Extraction
Transforms B+ extractions to A-grade quality
"""

import json
import re
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from collections import defaultdict

# OCR artifacts and clinical term corrections
OCR_CORRECTIONS = {
    # Ligature fixes
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    
    # Clinical term normalization
    "PDG-PET": "FDG-PET",
    "PDG PET": "FDG-PET",
    "CTscan": "CT scan",
    "CT-scan": "CT scan",
    "Extra thoracic": "Extrathoracic",
    "extra-thoracic": "extrathoracic",
    "Calciﬁcation": "Calcification",
    "Trans-thoracic": "Transthoracic",
    "trans-thoracic": "transthoracic",
    "Pre-test": "Pretest",
    "pre-test": "pretest",
    "Post-test": "Post-test",
    "SUVmax": "SUVmax",
    "SUV max": "SUVmax",
}

# Common clinical abbreviations to expand in definitions
CLINICAL_DEFINITIONS = {
    "pCA": "pretest probability of cancer",
    "IPN": "indeterminate pulmonary nodule",
    "SPN": "solitary pulmonary nodule",
    "VDT": "volume doubling time",
    "FDG": "fluorodeoxyglucose",
    "PET": "positron emission tomography",
    "EBUS": "endobronchial ultrasound",
    "TBNA": "transbronchial needle aspiration",
    "PPV": "positive predictive value",
    "NPV": "negative predictive value",
    "ACCP": "American College of Chest Physicians",
    "BTS": "British Thoracic Society",
}


def clean_ocr_artifacts(text: str) -> str:
    """Clean OCR artifacts and normalize clinical terms"""
    if not text:
        return text
    
    for artifact, correction in OCR_CORRECTIONS.items():
        text = text.replace(artifact, correction)
    
    # Additional cleanup
    text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
    text = text.strip()
    
    return text


def normalize_numeric_value(value: Any, field_name: str = "") -> Dict[str, Any]:
    """
    Normalize numeric values to consistent format
    Returns: {"value": float/int, "unit": str}
    """
    if value is None:
        return None
    
    # If already normalized
    if isinstance(value, dict) and "value" in value:
        return value
    
    # Handle string percentages
    if isinstance(value, str):
        # Clean the string
        value = value.strip()
        
        # Check for percentage
        percent_match = re.match(r'^(\d+(?:\.\d+)?)\s*%?$', value)
        if percent_match:
            num_val = float(percent_match.group(1))
            if field_name in ["sensitivity", "specificity", "ppv", "npv", "accuracy"]:
                # Convert to proportion (0-1) for these fields
                return {"value": num_val / 100.0, "unit": "proportion"}
            else:
                return {"value": num_val, "unit": "%"}
        
        # Check for other numeric patterns
        num_match = re.match(r'^[<>≤≥]?\s*(\d+(?:\.\d+)?)\s*(mm|cm|mL|L|mg|g)?$', value)
        if num_match:
            return {
                "value": float(num_match.group(1)),
                "unit": num_match.group(2) or "unitless"
            }
    
    # Handle plain numbers
    if isinstance(value, (int, float)):
        if field_name in ["sensitivity", "specificity", "ppv", "npv", "accuracy"]:
            # Assume these are percentages if > 1
            if value > 1:
                return {"value": value / 100.0, "unit": "proportion"}
            else:
                return {"value": value, "unit": "proportion"}
        else:
            return {"value": value, "unit": "unitless"}
    
    # Return as-is if can't normalize
    return value


def parse_criteria_string(criteria: Any) -> List[str]:
    """Convert comma-separated criteria strings to arrays"""
    if criteria is None:
        return []
    
    if isinstance(criteria, list):
        return criteria
    
    if isinstance(criteria, str):
        # Split by common delimiters
        items = re.split(r'[,;•·]\s*', criteria)
        # Clean each item
        items = [clean_ocr_artifacts(item.strip()) for item in items if item.strip()]
        return items
    
    return [str(criteria)]


def normalize_page_range(page_range: Any) -> Dict[str, int]:
    """Normalize page ranges to consistent format"""
    if isinstance(page_range, dict) and "start" in page_range:
        return page_range
    
    if isinstance(page_range, str):
        match = re.match(r'(\d+)(?:\s*[-–]\s*(\d+))?', page_range)
        if match:
            start = int(match.group(1))
            end = int(match.group(2)) if match.group(2) else start
            return {"start": start, "end": end}
    
    if isinstance(page_range, (int, float)):
        page = int(page_range)
        return {"start": page, "end": page}
    
    return None


def deduplicate_guidelines(guidelines: List[Dict]) -> List[Dict]:
    """Deduplicate and disambiguate guidelines"""
    seen = {}
    deduped = []
    
    for guideline in guidelines:
        title = guideline.get("title", "")
        org = guideline.get("source_organization", "")
        
        # Create a key for deduplication
        key = f"{title}|{org}".lower()
        
        if key in seen:
            # Check if this is truly a duplicate or needs disambiguation
            existing = seen[key]
            existing_page = existing.get("source_page", 0)
            new_page = guideline.get("source_page", 0)
            
            # If different pages, disambiguate
            if abs(existing_page - new_page) > 2:
                # Add disambiguation
                if "year" in guideline or "version" in guideline:
                    disambig = guideline.get("year") or guideline.get("version")
                    guideline["title"] = f"{title} ({disambig})"
                else:
                    guideline["title"] = f"{title} (Page {new_page})"
                deduped.append(guideline)
            # Otherwise, merge if new has more content
            elif len(str(guideline)) > len(str(existing)):
                # Replace with more complete version
                seen[key] = guideline
                # Remove old, add new
                deduped = [g for g in deduped if g != existing]
                deduped.append(guideline)
        else:
            seen[key] = guideline
            deduped.append(guideline)
    
    return deduped


def enhance_diagnostic_approach(approach: Dict) -> Dict:
    """Enhance a diagnostic approach with normalized fields"""
    enhanced = approach.copy()
    
    # Clean all text fields
    for field in ["name", "purpose", "how_to_perform", "interpretation", "when_to_use"]:
        if field in enhanced and enhanced[field]:
            enhanced[field] = clean_ocr_artifacts(enhanced[field])
    
    # Parse criteria as array
    if "criteria_or_scoring" in enhanced:
        enhanced["criteria_or_scoring"] = parse_criteria_string(enhanced["criteria_or_scoring"])
    
    # Normalize performance metrics
    performance = {}
    for metric in ["sensitivity", "specificity", "ppv", "npv", "accuracy"]:
        if metric in enhanced and enhanced[metric] is not None:
            normalized = normalize_numeric_value(enhanced[metric], metric)
            if normalized:
                performance[metric] = normalized
                # Remove from top level
                del enhanced[metric]
    
    if performance:
        enhanced["performance"] = performance
    
    # Parse limitations as array
    if "limitations" in enhanced and isinstance(enhanced["limitations"], str):
        enhanced["limitations"] = parse_criteria_string(enhanced["limitations"])
    
    # Add provenance structure
    if "source_page" in enhanced or "source_excerpt" in enhanced:
        enhanced["provenance"] = {
            "source_page": enhanced.get("source_page"),
            "source_excerpt": enhanced.get("source_excerpt"),
            "excerpt_type": "exact" if enhanced.get("source_excerpt") else None,
            "present_in_source": enhanced.get("present_in_source", True)
        }
        # Remove from top level
        for field in ["source_page", "source_excerpt", "present_in_source"]:
            enhanced.pop(field, None)
    
    # Mark missing fields
    for field in ["how_to_perform", "interpretation", "when_to_use"]:
        if field not in enhanced or enhanced[field] is None:
            enhanced[field] = None
            enhanced.setdefault("field_metadata", {})[field] = {"present_in_source": False}
    
    return enhanced


def enhance_table(table: Dict) -> Dict:
    """Enhance table with object representation"""
    enhanced = table.copy()
    
    # Clean title
    if "title" in enhanced:
        enhanced["title"] = clean_ocr_artifacts(enhanced["title"])
    
    # Add object representation if we have headers and rows
    if "headers" in enhanced and "rows" in enhanced:
        headers = enhanced["headers"]
        rows = enhanced["rows"]
        
        # Create object representation
        enhanced["data_objects"] = []
        for row in rows:
            if len(row) == len(headers):
                obj = {}
                for i, header in enumerate(headers):
                    # Clean header name
                    clean_header = clean_ocr_artifacts(header).lower().replace(" ", "_")
                    value = row[i]
                    
                    # Try to normalize numeric values in tables
                    if any(term in clean_header for term in ["sensitivity", "specificity", "ppv", "npv"]):
                        normalized = normalize_numeric_value(value, clean_header)
                        obj[clean_header] = normalized if normalized else value
                    else:
                        obj[clean_header] = clean_ocr_artifacts(str(value)) if value else None
                
                enhanced["data_objects"].append(obj)
    
    return enhanced


def calculate_page_range(data: Dict) -> Dict[str, int]:
    """Calculate the actual page range from all extracted content"""
    all_pages = []
    
    def extract_pages(obj: Any, pages: List[int]):
        """Recursively extract page numbers"""
        if isinstance(obj, dict):
            if "source_page" in obj and obj["source_page"]:
                pages.append(int(obj["source_page"]))
            if "page" in obj and obj["page"]:
                pages.append(int(obj["page"]))
            if "page_range" in obj and obj["page_range"]:
                pr = normalize_page_range(obj["page_range"])
                if pr:
                    pages.extend(range(pr["start"], pr["end"] + 1))
            
            for value in obj.values():
                extract_pages(value, pages)
        elif isinstance(obj, list):
            for item in obj:
                extract_pages(item, pages)
    
    extract_pages(data, all_pages)
    
    if all_pages:
        return {"start": min(all_pages), "end": max(all_pages)}
    else:
        return {"start": 1, "end": 1}


def enhance_extraction(data: Dict) -> Dict:
    """Main enhancement function"""
    enhanced = data.copy()
    
    # 1. Fix pagination consistency
    actual_range = calculate_page_range(enhanced)
    if "chapter_metadata" in enhanced:
        enhanced["chapter_metadata"]["page_range"] = actual_range
    
    # 2. Clean OCR artifacts in metadata
    if "chapter_metadata" in enhanced:
        meta = enhanced["chapter_metadata"]
        if "title" in meta:
            if isinstance(meta["title"], dict):
                meta["title"]["value"] = clean_ocr_artifacts(meta["title"].get("value", ""))
            else:
                meta["title"] = clean_ocr_artifacts(meta["title"])
        
        if "authors" in meta:
            for author in meta["authors"]:
                if isinstance(author, dict) and "name" in author:
                    author["name"] = clean_ocr_artifacts(author["name"])
    
    # 3. Enhance diagnostic approaches
    if "diagnostic_approaches" in enhanced:
        enhanced["diagnostic_approaches"] = [
            enhance_diagnostic_approach(approach) 
            for approach in enhanced["diagnostic_approaches"]
        ]
    
    # 4. Deduplicate and enhance guidelines
    if "clinical_guidelines" in enhanced:
        # First clean all text
        for guideline in enhanced["clinical_guidelines"]:
            for field in ["title", "condition", "source_organization"]:
                if field in guideline and guideline[field]:
                    guideline[field] = clean_ocr_artifacts(guideline[field])
            
            # Parse recommendations as array
            if "specific_recommendations" in guideline:
                if isinstance(guideline["specific_recommendations"], str):
                    guideline["specific_recommendations"] = parse_criteria_string(
                        guideline["specific_recommendations"]
                    )
                elif isinstance(guideline["specific_recommendations"], list):
                    guideline["specific_recommendations"] = [
                        clean_ocr_artifacts(rec) for rec in guideline["specific_recommendations"]
                    ]
        
        # Deduplicate
        enhanced["clinical_guidelines"] = deduplicate_guidelines(enhanced["clinical_guidelines"])
    
    # 5. Enhance tables
    if "tables" in enhanced:
        enhanced["tables"] = [enhance_table(table) for table in enhanced["tables"]]
    
    # 6. Clean clinical pearls and definitions
    if "clinical_pearls" in enhanced:
        for pearl in enhanced["clinical_pearls"]:
            if "content" in pearl:
                pearl["content"] = clean_ocr_artifacts(pearl["content"])
    
    if "definitions" in enhanced:
        for definition in enhanced["definitions"]:
            if "term" in definition:
                definition["term"] = clean_ocr_artifacts(definition["term"])
            if "definition" in definition:
                definition["definition"] = clean_ocr_artifacts(definition["definition"])
    
    # 7. Add common definitions if not present
    if "definitions" not in enhanced or not enhanced["definitions"]:
        enhanced["definitions"] = []
    
    # Add standard clinical definitions
    existing_terms = {d.get("term", "").upper() for d in enhanced["definitions"]}
    for abbrev, full_name in CLINICAL_DEFINITIONS.items():
        if abbrev not in existing_terms:
            enhanced["definitions"].append({
                "term": abbrev,
                "definition": full_name,
                "present_in_source": False,
                "added_by": "enhancer"
            })
    
    # 8. Normalize all page_range fields
    def normalize_all_page_ranges(obj: Any):
        """Recursively normalize page_range fields"""
        if isinstance(obj, dict):
            if "page_range" in obj and not isinstance(obj["page_range"], dict):
                obj["page_range"] = normalize_page_range(obj["page_range"])
            
            for key, value in obj.items():
                normalize_all_page_ranges(value)
        elif isinstance(obj, list):
            for item in obj:
                normalize_all_page_ranges(item)
    
    normalize_all_page_ranges(enhanced)
    
    # 9. Add enhancement metadata
    if "extraction_metadata" not in enhanced:
        enhanced["extraction_metadata"] = {}
    
    enhanced["extraction_metadata"]["enhanced"] = True
    enhanced["extraction_metadata"]["enhancement_version"] = "1.0"
    enhanced["extraction_metadata"]["enhancements_applied"] = [
        "ocr_cleanup",
        "type_normalization", 
        "criteria_tokenization",
        "guideline_deduplication",
        "page_range_normalization",
        "performance_metric_structuring",
        "definition_enrichment"
    ]
    
    return enhanced


def process_file(input_path: Path, output_path: Optional[Path] = None) -> Path:
    """Process a single extraction file"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    enhanced = enhance_extraction(data)
    
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_enhanced.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced extraction saved to: {output_path}")
    
    # Print summary of enhancements
    print("\n📊 Enhancement Summary:")
    print("-" * 40)
    
    if "diagnostic_approaches" in enhanced:
        approaches_with_perf = sum(
            1 for a in enhanced["diagnostic_approaches"] 
            if "performance" in a
        )
        print(f"  Diagnostic approaches with normalized performance: {approaches_with_perf}")
    
    if "clinical_guidelines" in enhanced:
        print(f"  Clinical guidelines after deduplication: {len(enhanced['clinical_guidelines'])}")
    
    if "definitions" in enhanced:
        added_defs = sum(1 for d in enhanced["definitions"] if d.get("added_by") == "enhancer")
        print(f"  Definitions added: {added_defs}")
    
    if "chapter_metadata" in enhanced and "page_range" in enhanced["chapter_metadata"]:
        pr = enhanced["chapter_metadata"]["page_range"]
        print(f"  Page range: {pr['start']}-{pr['end']}")
    
    return output_path


def main():
    """Main entry point for testing"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python extraction_enhancer.py <input_json> [output_json]")
        sys.exit(1)
    
    input_path = Path(sys.argv[1])
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    process_file(input_path, output_path)


if __name__ == "__main__":
    main()