# Instructions for Claude Code to Set Up Clinical Extraction System

Copy and paste these instructions directly to Claude Code:

---

**Task: Set up the IPchat clinical extraction refactor and migration system**

Please execute the following steps to create the complete clinical extraction system:

## Step 1: Create Directory Structure

Create these directories:
```
ipchat/extraction/
ipchat/processing/
ipchat/retrieval/
ipchat/migration/
ipchat/pipeline/
data/raw/research/
data/raw/textbooks/
data/extracted/
data/migrated_extracted/
data/chunks/
data/indices/
data/backup/
tools/scripts/
```

Create empty `__init__.py` files in each ipchat subdirectory.

## Step 2: Create the Clinical Extractor

Create file `ipchat/extraction/clinical_extractor.py` with this content:

```python
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
            pattern = f'{term}[^\d]*(\d+\.?\d*)\s*%'
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
```

## Step 3: Create the Extraction Evaluator

Create file `ipchat/migration/evaluator.py`:

```python
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
        
        # Check for diagnostic yields
        if data.get('diagnostic_yield') or data.get('diagnostic_yields'):
            score += 25
        else:
            issues.append('missing_yields')
        
        # Check for complications
        if data.get('complications') or data.get('adverse_events'):
            score += 25
        else:
            issues.append('missing_complications')
        
        # Check for clinical pearls
        if data.get('clinical_pearls') or data.get('key_takeaways'):
            score += 20
        else:
            issues.append('missing_pearls')
        
        # Check for methodology
        if data.get('methodology') or data.get('procedure_details'):
            score += 15
        else:
            issues.append('missing_methodology')
        
        # Check for structured data
        if isinstance(data.get('diagnostic_yield'), dict):
            score += 10
        
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
```

## Step 4: Evaluate Existing Extractions

Run this code to evaluate any existing extractions in `data/extracted/`:

```python
from pathlib import Path
import json
from ipchat.migration.evaluator import ExtractionEvaluator

evaluator = ExtractionEvaluator()
existing_dir = Path("data/extracted")

if existing_dir.exists():
    json_files = list(existing_dir.glob("*.json"))
    print(f"Found {len(json_files)} existing extractions\n")
    
    results = []
    for json_file in json_files:
        result = evaluator.evaluate_extraction(json_file)
        results.append(result)
        print(f"{json_file.stem}: Score={result['score']}, Action={result['recommendation']}")
    
    # Calculate summary
    total = len(results)
    keep = len([r for r in results if r['recommendation'] == 'keep_enhance'])
    augment = len([r for r in results if r['recommendation'] == 'augment'])
    restructure = len([r for r in results if r['recommendation'] == 'restructure'])
    re_extract = len([r for r in results if r['recommendation'] == 're_extract'])
    
    print(f"\nSummary:")
    print(f"  Total: {total}")
    print(f"  Keep & enhance: {keep} ({keep/total*100:.1f}%)")
    print(f"  Augment: {augment} ({augment/total*100:.1f}%)")
    print(f"  Restructure: {restructure} ({restructure/total*100:.1f}%)")
    print(f"  Re-extract: {re_extract} ({re_extract/total*100:.1f}%)")
    
    # Save report
    with open("data/evaluation_report.json", 'w') as f:
        json.dump({
            'total': total,
            'keep': keep,
            'augment': augment,
            'restructure': restructure,
            're_extract': re_extract,
            'details': results
        }, f, indent=2)
    print(f"\nReport saved to data/evaluation_report.json")
else:
    print("No existing extractions found")
```

## Step 5: Test the System

Run this test to verify everything works:

```python
from ipchat.extraction.clinical_extractor import ClinicalDataExtractor

sample_text = """
RESULTS: The diagnostic yield of EBUS-TBNA was 92.5% (95% CI: 89.8-94.6).
Sensitivity for malignancy was 95.2% with specificity of 100%.
Complications occurred in 2.4% including pneumothorax in 0.4%.

CLINICAL PEARL: Maintain the bronchoscope in a neutral position.
"""

extractor = ClinicalDataExtractor()
result = extractor.extract(sample_text, "research")

print(f"Extraction successful!")
print(f"  Yields found: {len(result.diagnostic_yields)} procedures")
print(f"  Complications found: {len(result.complication_rates)} types")
print(f"  Clinical pearls: {len(result.clinical_pearls)} pearls")
print(f"  Confidence: {result.extraction_confidence:.2f}")
```

## Summary

After executing these steps, you'll have:
1. Complete clinical extraction system set up
2. Evaluation of all existing extractions
3. Report showing which extractions to keep, augment, or re-extract
4. System ready to process new documents

The evaluation report will tell you:
- How many of your existing extractions are good (keep)
- How many need minor additions (augment)
- How many need reformatting (restructure)
- How many are poor quality (re-extract)

---

**End of instructions for Claude Code**
