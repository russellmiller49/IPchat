"""
Evaluate existing extractions
"""

import json
from pathlib import Path
from typing import Dict, Any
import re

class ExtractionEvaluator:
    def evaluate_extraction(self, extraction_path: Path) -> Dict[str, Any]:
        try:
            with open(extraction_path, 'r') as f:
                data = json.load(f)
        except:
            return {
                'path': str(extraction_path),
                'score': 0,
                'issues': ['cannot_read'],
                'recommendation': 're_extract'
            }
        
        score = 0
        issues = []
        
        # Check for existing structure with document/sections
        if 'document' in data:
            doc = data['document']
            
            # Check for outcomes/results data
            if 'sections' in doc and 'results' in doc['sections']:
                results_text = doc['sections']['results']
                if re.search(r'\d+\.?\d*\s*%', results_text):
                    score += 25
                else:
                    issues.append('results_without_data')
            
            # Check for adverse events/complications
            if 'adverse_events' in doc or ('sections' in doc and 'adverse_events' in doc['sections']):
                score += 25
            elif 'sections' in doc and 'results' in doc['sections'] and 'pneumothorax' in doc['sections']['results'].lower():
                score += 20  # Has complications mentioned
            else:
                issues.append('missing_complications')
            
            # Check for key findings/clinical pearls
            if 'key_findings' in doc or 'clinical_pearls' in doc:
                score += 20
            elif 'sections' in doc and 'conclusions' in doc['sections']:
                score += 15
            else:
                issues.append('missing_key_findings')
            
            # Check for methods/methodology
            if 'sections' in doc and 'methods' in doc['sections']:
                score += 15
            else:
                issues.append('missing_methodology')
            
            # Check for proper metadata
            if 'metadata' in doc and doc['metadata'].get('title'):
                score += 10
                
        else:
            # Check old format fields
            if data.get('diagnostic_yield') or data.get('diagnostic_yields'):
                score += 25
            else:
                issues.append('missing_yields')
            
            if data.get('complications') or data.get('adverse_events'):
                score += 25
            else:
                issues.append('missing_complications')
                
            if data.get('clinical_pearls') or data.get('key_takeaways'):
                score += 20
            else:
                issues.append('missing_pearls')
                
            if data.get('methodology') or data.get('procedure_details'):
                score += 15
            else:
                issues.append('missing_methodology')
        
        # Check for numerical data
        text = json.dumps(data)
        if re.search(r'\d+\.?\d*\s*%', text):
            score += 5
        
        # Determine recommendation
        if score >= 80:
            recommendation = 'keep_enhance'
        elif score >= 50:
            recommendation = 'augment'
        elif score >= 25:
            recommendation = 'restructure'
        else:
            recommendation = 're_extract'
        
        return {
            'path': str(extraction_path),
            'score': score,
            'issues': issues,
            'recommendation': recommendation,
            'has_gpt5': 'gpt5' in text.lower() or 'gpt-5' in text.lower()
        }