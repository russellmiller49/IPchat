"""
Clinical Data Extractor for Interventional Pulmonology
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
import hashlib

@dataclass
class ClinicalExtraction:
    """Structured clinical data extraction"""
    document_id: str
    title: str
    document_type: str
    diagnostic_yields: Dict[str, Any]
    complication_rates: Dict[str, Any]
    methodology: Dict[str, Any]
    clinical_pearls: List[str]
    indications: List[str]
    contraindications: List[str]
    patient_selection: Dict[str, Any]
    equipment: List[str]
    key_findings: List[str]
    practical_tips: List[str]
    extraction_confidence: float
    has_numerical_data: bool
    original_extraction: Dict[str, Any] = None

class ClinicalDataExtractor:
    def __init__(self):
        self.procedure_keywords = [
            'ebus', 'tbna', 'bronchoscopy', 'navigational', 'robotic',
            'cryobiopsy', 'bal', 'thoracentesis', 'pleuroscopy'
        ]
    
    def extract(self, content: str, document_type: str = "research") -> ClinicalExtraction:
        doc_id = hashlib.md5(content[:1000].encode()).hexdigest()[:12]
        
        diagnostic_yields = self._extract_diagnostic_yields(content)
        complications = self._extract_complications(content)
        methodology = self._extract_methodology(content)
        pearls = self._extract_clinical_pearls(content)
        indications = self._extract_indications(content)
        contraindications = self._extract_contraindications(content)
        patient_selection = self._extract_patient_selection(content)
        equipment = self._extract_equipment(content)
        key_findings = self._extract_key_findings(content, diagnostic_yields, complications)
        practical_tips = self._extract_practical_tips(content)
        
        has_numerical = bool(diagnostic_yields or complications)
        confidence = self._calculate_confidence(diagnostic_yields, complications, methodology, pearls)
        
        return ClinicalExtraction(
            document_id=doc_id,
            title=self._extract_title(content),
            document_type=document_type,
            diagnostic_yields=diagnostic_yields,
            complication_rates=complications,
            methodology=methodology,
            clinical_pearls=pearls,
            indications=indications,
            contraindications=contraindications,
            patient_selection=patient_selection,
            equipment=equipment,
            key_findings=key_findings,
            practical_tips=practical_tips,
            extraction_confidence=confidence,
            has_numerical_data=has_numerical
        )
    
    def _extract_diagnostic_yields(self, content: str) -> Dict[str, Any]:
        yields = {}
        patterns = [
            r'diagnostic yield[^\d]*(\d+\.?\d*)\s*%',
            r'sensitivity[^\d]*(\d+\.?\d*)\s*%',
            r'specificity[^\d]*(\d+\.?\d*)\s*%'
        ]
        
        for pattern in patterns:
            matches = re.finditer(pattern, content.lower())
            for match in matches:
                start = max(0, match.start() - 150)
                end = min(len(content), match.end() + 150)
                context = content[start:end]
                procedure = self._identify_procedure(context)
                
                if procedure not in yields:
                    yields[procedure] = {}
                metric = pattern.split('[')[0].strip()
                yields[procedure][metric] = {
                    'value': float(match.group(1)),
                    'context': context.strip()
                }
        return yields
    
    def _extract_complications(self, content: str) -> Dict[str, Any]:
        complications = {}
        terms = ['pneumothorax', 'bleeding', 'hemorrhage', 'infection', 'perforation']
        
        for term in terms:
            pattern = f'{term}[^\\d]*(\\d+\\.?\\d*)\\s*%'
            matches = re.finditer(pattern, content.lower())
            for match in matches:
                rate = float(match.group(1))
                start = max(0, match.start() - 200)
                end = min(len(content), match.end() + 200)
                context = content[start:end]
                
                complications[term] = {
                    'rate': rate,
                    'management': self._extract_management(context),
                    'context': context.strip()
                }
        return complications
    
    def _extract_clinical_pearls(self, content: str) -> List[str]:
        pearls = []
        keywords = ['tip', 'trick', 'pearl', 'important', 'remember', 'recommend']
        sentences = content.split('.')
        
        for sentence in sentences:
            if any(kw in sentence.lower() for kw in keywords):
                pearl = sentence.strip()
                if 20 < len(pearl) < 500:
                    pearls.append(pearl)
        return pearls[:20]
    
    def _extract_indications(self, content: str) -> List[str]:
        indications = []
        if 'indication' in content.lower():
            idx = content.lower().find('indication')
            section = content[idx:idx+1500]
            lines = section.split('\n')
            for line in lines:
                if line.strip() and line.strip()[0] in '•-*123456789':
                    indications.append(line.strip()[:200])
        return indications[:10]
    
    def _extract_contraindications(self, content: str) -> List[str]:
        contraindications = []
        if 'contraindication' in content.lower():
            idx = content.lower().find('contraindication')
            section = content[idx:idx+1500]
            lines = section.split('\n')
            for line in lines:
                if line.strip() and line.strip()[0] in '•-*123456789':
                    contraindications.append(line.strip()[:200])
        return contraindications[:10]
    
    def _extract_methodology(self, content: str) -> Dict[str, Any]:
        methodology = {}
        # Implementation continues...
        return methodology
    
    def _extract_patient_selection(self, content: str) -> Dict[str, Any]:
        return {'inclusion_criteria': [], 'exclusion_criteria': [], 'ideal_candidate': None}
    
    def _extract_equipment(self, content: str) -> List[str]:
        equipment = []
        keywords = ['needle', 'scope', 'catheter', 'guidewire', 'forceps']
        for keyword in keywords:
            if keyword in content.lower():
                equipment.append(keyword)
        return equipment[:15]
    
    def _extract_key_findings(self, content: str, yields: Dict, complications: Dict) -> List[str]:
        findings = []
        for procedure, yield_data in yields.items():
            for metric, data in yield_data.items():
                findings.append(f"{procedure.upper()} {metric}: {data['value']}%")
        for complication, data in complications.items():
            findings.append(f"{complication.capitalize()} rate: {data['rate']}%")
        return findings[:10]
    
    def _extract_practical_tips(self, content: str) -> List[str]:
        tips = []
        patterns = [r'recommend[^.]*\.', r'suggest[^.]*\.']
        for pattern in patterns:
            matches = re.finditer(pattern, content.lower())
            for match in matches:
                tip = content[match.start():match.end()].strip()
                if 20 < len(tip) < 300:
                    tips.append(tip)
        return tips[:10]
    
    def _identify_procedure(self, text: str) -> str:
        text_lower = text.lower()
        for procedure in self.procedure_keywords:
            if procedure in text_lower:
                return procedure
        return 'general'
    
    def _extract_management(self, context: str) -> str:
        keywords = ['managed', 'treated', 'resolved']
        for keyword in keywords:
            if keyword in context.lower():
                return "Management described"
        return "Not specified"
    
    def _extract_title(self, content: str) -> str:
        lines = content.split('\n')
        for line in lines[:10]:
            if 10 < len(line) < 200:
                return line.strip()
        return "Unknown Title"
    
    def _calculate_confidence(self, yields, complications, methodology, pearls) -> float:
        score = 0.5
        if yields: score += 0.2
        if complications: score += 0.15
        if methodology: score += 0.1
        if len(pearls) > 5: score += 0.05
        return min(score, 1.0)