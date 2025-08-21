"""
Validators for extracted data quality assurance.
"""

from typing import Dict, Any, List, Tuple

class ExtractionValidator:
    """Validate extracted documents for completeness and quality"""
    
    @staticmethod
    def validate_research_extraction(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate research article extraction.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required fields
        if not data.get('summary'):
            issues.append("Missing summary")
        
        # Check for at least some core PICO elements
        pico_elements = ['population', 'intervention', 'outcomes']
        pico_present = sum(1 for elem in pico_elements if data.get(elem))
        
        if pico_present < 2:
            issues.append(f"Only {pico_present}/3 PICO elements present")
        
        # Check key findings
        if data.get('key_findings'):
            if len(data['key_findings']) < 2:
                issues.append("Less than 2 key findings extracted")
        else:
            issues.append("No key findings extracted")
        
        return len(issues) == 0, issues
    
    @staticmethod
    def validate_textbook_extraction(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate textbook chapter extraction.
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check for required fields
        if not data.get('summary'):
            issues.append("Missing summary")
        
        # Check for clinical content
        clinical_fields = ['procedures', 'indications', 'contraindications']
        clinical_present = sum(1 for field in clinical_fields if data.get(field))
        
        if clinical_present < 1:
            issues.append("No clinical guidance extracted")
        
        # Validate procedures if present
        if data.get('procedures'):
            for proc in data['procedures']:
                if not proc.get('name') or not proc.get('description'):
                    issues.append("Incomplete procedure information")
                    break
        
        return len(issues) == 0, issues
    
    @staticmethod
    def calculate_extraction_score(data: Dict[str, Any], doc_type: str) -> float:
        """
        Calculate a quality score for the extraction (0-1).
        
        Args:
            data: Extracted data
            doc_type: 'research' or 'textbook'
            
        Returns:
            Score between 0 and 1
        """
        score = 0.0
        max_score = 0.0
        
        if doc_type == 'research':
            # Score based on PICO completeness
            fields = {
                'population': 0.2,
                'intervention': 0.2,
                'comparator': 0.1,
                'outcomes': 0.2,
                'key_findings': 0.2,
                'summary': 0.1
            }
            
            for field, weight in fields.items():
                max_score += weight
                if data.get(field):
                    if field == 'key_findings':
                        # Scale by number of findings (max 5)
                        score += weight * min(len(data[field]), 5) / 5
                    else:
                        score += weight
        
        elif doc_type == 'textbook':
            # Score based on clinical content
            fields = {
                'procedures': 0.3,
                'indications': 0.2,
                'contraindications': 0.2,
                'algorithms': 0.1,
                'summary': 0.2
            }
            
            for field, weight in fields.items():
                max_score += weight
                if data.get(field):
                    if isinstance(data[field], list):
                        # Scale by content amount (max 10 items)
                        score += weight * min(len(data[field]), 10) / 10
                    else:
                        score += weight
        
        return score / max_score if max_score > 0 else 0.0