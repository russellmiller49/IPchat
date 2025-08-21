#!/usr/bin/env python3
"""
Demonstration of extraction enhancements
Shows before/after examples of the improvements
"""

import json

# Example of enhanced output structure
enhanced_example = {
    "diagnostic_approaches": [
        {
            "name": "FDG-PET",  # Fixed from "PDG-PET"
            "purpose": "Evaluation of an indeterminate pulmonary nodule (IPN)",
            "criteria_or_scoring": ["SUV ≥ 2.5 suggests malignancy"],  # Array, not string
            "how_to_perform": "Integrated PET/CT for functional and morphologic assessment",
            "interpretation": "Suggests infectious, inflammatory, or malignant process",
            "performance": {  # Normalized numeric fields
                "sensitivity": {"value": 0.89, "unit": "proportion"},  # Was "89%"
                "specificity": {"value": 0.75, "unit": "proportion"}   # Was "75%"
            },
            "limitations": ["Lower utility in endemic infectious lung disease"],  # Array
            "when_to_use": "Risk stratification and localization of malignant lesions",
            "provenance": {
                "source_page": 3,
                "source_excerpt": "FDG-PET sensitivity 89% and specificity 75% for malignancy",
                "excerpt_type": "exact",
                "present_in_source": True
            }
        },
        {
            "name": "Mayo Clinic Model",
            "purpose": "Malignancy risk estimation for pulmonary nodules",
            "criteria_or_scoring": [  # Tokenized array
                "Age",
                "Smoking history", 
                "Extrathoracic malignancy",  # Fixed from "Extra thoracic"
                "Nodule size",
                "Spiculation",
                "Nodule location"
            ],
            "limitations": ["Excludes patients with cancer in previous 5 years"],
            "when_to_use": "Incidentally detected nodules on CXR or CT scan",  # Fixed "CTscan"
            "provenance": {"source_page": 5}
        }
    ],
    
    "clinical_guidelines": [
        {
            "title": "BTS Guidelines for Lung Nodule Investigation (2015)",  # Disambiguated with year
            "condition": "Peripheral lung nodules",
            "source_organization": "British Thoracic Society",
            "specific_recommendations": [  # Array, not comma-separated string
                "Investigate nodules >80 mm³ or >6 mm diameter",
                "Follow-up at 3 months for 5-6mm nodules",
                "Consider PET-CT for nodules 8-10mm"
            ],
            "source_page": 12,
            "source_excerpt": "BTS recommends investigation for nodules >80 mm³",
            "present_in_source": True
        }
    ],
    
    "tables": [
        {
            "title": "Comparison of Diagnostic Modalities",
            "headers": ["Modality", "Sensitivity (%)", "Specificity (%)", "PPV (%)", "NPV (%)"],
            "rows": [
                ["CT scan", "91", "90", "85", "94"],  # Fixed "CTscan"
                ["FDG-PET", "87", "83", "78", "90"],  # Fixed "PDG-PET"
                ["EBUS-TBNA", "89", "100", "100", "91"]
            ],
            "data_objects": [  # Machine-readable object representation
                {
                    "modality": "CT scan",
                    "sensitivity_(%)": {"value": 0.91, "unit": "proportion"},
                    "specificity_(%)": {"value": 0.90, "unit": "proportion"},
                    "ppv_(%)": {"value": 0.85, "unit": "proportion"},
                    "npv_(%)": {"value": 0.94, "unit": "proportion"}
                },
                {
                    "modality": "FDG-PET",
                    "sensitivity_(%)": {"value": 0.87, "unit": "proportion"},
                    "specificity_(%)": {"value": 0.83, "unit": "proportion"},
                    "ppv_(%)": {"value": 0.78, "unit": "proportion"},
                    "npv_(%)": {"value": 0.90, "unit": "proportion"}
                }
            ],
            "source_page": 18,
            "content_provenance": "xlsx"
        }
    ],
    
    "chapter_metadata": {
        "title": {
            "value": "Approach to Peripheral Lung Lesions",
            "present_in_source": True
        },
        "page_range": {"start": 1, "end": 11},  # Normalized structure, correct span
        "authors": [
            {"name": "John Smith"},  # No "et al." entries
            {"name": "Jane Doe"}
        ]
    },
    
    "definitions": [  # Auto-added common abbreviations
        {"term": "pCA", "definition": "pretest probability of cancer", "present_in_source": False, "added_by": "enhancer"},
        {"term": "IPN", "definition": "indeterminate pulmonary nodule", "present_in_source": False, "added_by": "enhancer"},
        {"term": "VDT", "definition": "volume doubling time", "present_in_source": False, "added_by": "enhancer"},
        {"term": "FDG", "definition": "fluorodeoxyglucose", "present_in_source": False, "added_by": "enhancer"},
        {"term": "EBUS", "definition": "endobronchial ultrasound", "present_in_source": False, "added_by": "enhancer"}
    ],
    
    "extraction_metadata": {
        "extractor_version": "production_multipass_v2.1_enhanced",
        "enhancements_applied": [
            "ocr_cleanup",
            "type_normalization",
            "criteria_tokenization",
            "guideline_deduplication",
            "page_range_normalization",
            "performance_metric_structuring",
            "definition_enrichment"
        ]
    }
}

# Print demonstration
print("=" * 60)
print("EXTRACTION ENHANCEMENT DEMONSTRATION")
print("=" * 60)
print("\n✅ Key Improvements Applied:\n")

print("1. **OCR Cleanup**")
print("   - PDG-PET → FDG-PET")
print("   - CTscan → CT scan")
print("   - Calciﬁcation → Calcification (ligature fix)")
print("   - Extra thoracic → Extrathoracic\n")

print("2. **Type Normalization**")
print("   - Sensitivity: '89%' → {'value': 0.89, 'unit': 'proportion'}")
print("   - Page range: '1-11' → {'start': 1, 'end': 11}\n")

print("3. **Criteria Tokenization**")
print("   - 'Age, smoking history, nodule size' → ['Age', 'Smoking history', 'Nodule size']\n")

print("4. **Guideline Deduplication**")
print("   - Multiple BTS entries → Single disambiguated entry with year\n")

print("5. **Table Enhancement**")
print("   - Added data_objects with normalized numeric values")
print("   - Machine-readable format for each row\n")

print("6. **Definition Enrichment**")
print("   - Auto-added common clinical abbreviations (pCA, IPN, VDT, etc.)\n")

print("7. **Provenance Structure**")
print("   - Grouped provenance fields in dedicated object")
print("   - Added excerpt_type flag\n")

print("=" * 60)
print("Ready for A-grade NLP processing! 🎯")
print("=" * 60)

# Save example
with open('data/test_extraction/enhanced_example.json', 'w') as f:
    json.dump(enhanced_example, f, indent=2)
    print(f"\n✅ Example saved to: data/test_extraction/enhanced_example.json")