"""
Migration script to augment and restructure existing extractions
"""

import json
import re
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import asdict
import sys
import hashlib
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ipchat.extraction.clinical_extractor import ClinicalDataExtractor, ClinicalExtraction
from ipchat.migration.evaluator import ExtractionEvaluator


class ExtractionMigrator:
    def __init__(self):
        self.extractor = ClinicalDataExtractor()
        self.evaluator = ExtractionEvaluator()
        self.stats = {
            'total': 0,
            'augmented': 0,
            'restructured': 0,
            'failed': 0,
            'skipped': 0
        }
        self.errors = []
        
    def migrate_all(self, source_dir: Path, output_dir: Path, evaluation_report: Path):
        """Process all extractions based on evaluation report"""
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load evaluation report
        with open(evaluation_report, 'r') as f:
            report = json.load(f)
        
        print(f"Starting migration of {report['total']} extractions...")
        print(f"  - {report['augment']} files to augment")
        print(f"  - {report['restructure']} files to restructure")
        print(f"  - {report['keep']} files to keep as-is")
        print()
        
        # Process each file based on recommendation
        for i, file_info in enumerate(report['details'], 1):
            file_path = Path(file_info['path'])
            recommendation = file_info['recommendation']
            
            if i % 20 == 0:
                print(f"Progress: {i}/{report['total']} files processed...")
            
            self.stats['total'] += 1
            
            try:
                if recommendation == 'augment':
                    self._augment_extraction(file_path, output_dir)
                    self.stats['augmented'] += 1
                elif recommendation == 'restructure':
                    self._restructure_extraction(file_path, output_dir)
                    self.stats['restructured'] += 1
                elif recommendation == 'keep_enhance':
                    # Copy with minimal enhancements
                    self._enhance_extraction(file_path, output_dir)
                    self.stats['augmented'] += 1
                else:
                    self.stats['skipped'] += 1
                    
            except Exception as e:
                self.stats['failed'] += 1
                self.errors.append({
                    'file': file_path.name,
                    'error': str(e),
                    'recommendation': recommendation
                })
                print(f"  ERROR processing {file_path.name}: {str(e)[:100]}")
        
        # Generate summary report
        self._generate_summary_report(output_dir)
        
    def _augment_extraction(self, file_path: Path, output_dir: Path):
        """Augment existing extraction with missing clinical data"""
        
        # Load existing extraction
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        
        # Extract text content for analysis
        text_content = self._extract_text_from_json(existing_data)
        
        # Get clinical extraction
        clinical_data = self.extractor.extract(text_content, "research")
        
        # Merge with existing data
        augmented_data = self._merge_extractions(existing_data, clinical_data)
        
        # Save augmented version
        output_path = output_dir / file_path.name
        with open(output_path, 'w') as f:
            json.dump(augmented_data, f, indent=2)
    
    def _restructure_extraction(self, file_path: Path, output_dir: Path):
        """Completely restructure extraction to new format"""
        
        # Load existing extraction
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        
        # Extract text content
        text_content = self._extract_text_from_json(existing_data)
        
        # Get clinical extraction
        clinical_data = self.extractor.extract(text_content, "research")
        
        # Create new structure preserving important original data
        restructured_data = {
            'source': existing_data.get('source', {}),
            'document': existing_data.get('document', {}),
            'clinical_extraction': asdict(clinical_data),
            'migration_info': {
                'migrated_at': datetime.now().isoformat(),
                'original_structure': 'oe_final',
                'migration_type': 'restructure'
            }
        }
        
        # Add any original numerical data that might be missing
        if 'document' in existing_data and 'sections' in existing_data['document']:
            restructured_data['original_sections'] = existing_data['document']['sections']
        
        # Save restructured version
        output_path = output_dir / file_path.name
        with open(output_path, 'w') as f:
            json.dump(restructured_data, f, indent=2)
    
    def _enhance_extraction(self, file_path: Path, output_dir: Path):
        """Minimally enhance already good extractions"""
        
        # Load existing extraction
        with open(file_path, 'r') as f:
            existing_data = json.load(f)
        
        # Add migration metadata
        existing_data['migration_info'] = {
            'migrated_at': datetime.now().isoformat(),
            'migration_type': 'enhance',
            'original_structure': 'oe_final'
        }
        
        # Save enhanced version
        output_path = output_dir / file_path.name
        with open(output_path, 'w') as f:
            json.dump(existing_data, f, indent=2)
    
    def _extract_text_from_json(self, data: Dict) -> str:
        """Extract all text content from JSON structure"""
        text_parts = []
        
        # Get title
        if 'document' in data and 'metadata' in data['document']:
            metadata = data['document']['metadata']
            if 'title' in metadata:
                text_parts.append(metadata['title'])
        
        # Get all section texts
        if 'document' in data and 'sections' in data['document']:
            sections = data['document']['sections']
            for section_name, section_content in sections.items():
                if isinstance(section_content, str):
                    text_parts.append(f"{section_name.upper()}: {section_content}")
                elif isinstance(section_content, list):
                    text_parts.append(f"{section_name.upper()}: {' '.join(str(item) for item in section_content)}")
        
        # Get any outcome data
        if 'outcomes' in data:
            text_parts.append(f"OUTCOMES: {json.dumps(data['outcomes'])}")
        
        # Get any statistical data
        if 'statistical_analysis' in data:
            text_parts.append(f"STATISTICS: {json.dumps(data['statistical_analysis'])}")
        
        return '\n\n'.join(text_parts)
    
    def _merge_extractions(self, existing: Dict, clinical: ClinicalExtraction) -> Dict:
        """Merge clinical extraction with existing data"""
        
        # Start with existing data
        merged = existing.copy()
        
        # Add clinical extraction as new section
        merged['clinical_extraction'] = asdict(clinical)
        
        # Add migration metadata
        merged['migration_info'] = {
            'migrated_at': datetime.now().isoformat(),
            'migration_type': 'augment',
            'original_structure': 'oe_final',
            'clinical_confidence': clinical.extraction_confidence
        }
        
        # Enhance existing sections if possible
        if 'document' in merged:
            if 'key_findings' not in merged['document'] and clinical.key_findings:
                merged['document']['key_findings'] = clinical.key_findings
            
            if 'clinical_pearls' not in merged['document'] and clinical.clinical_pearls:
                merged['document']['clinical_pearls'] = clinical.clinical_pearls
            
            # Add structured complications if found
            if clinical.complication_rates and 'adverse_events' not in merged['document']:
                merged['document']['adverse_events'] = clinical.complication_rates
        
        return merged
    
    def _generate_summary_report(self, output_dir: Path):
        """Generate migration summary report"""
        
        report = {
            'migration_date': datetime.now().isoformat(),
            'statistics': self.stats,
            'errors': self.errors,
            'success_rate': (self.stats['augmented'] + self.stats['restructured']) / max(self.stats['total'], 1) * 100
        }
        
        # Save report
        report_path = output_dir.parent / 'migration_summary.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        print("\n" + "="*60)
        print("MIGRATION COMPLETE")
        print("="*60)
        print(f"Total files processed: {self.stats['total']}")
        print(f"  - Augmented: {self.stats['augmented']}")
        print(f"  - Restructured: {self.stats['restructured']}")
        print(f"  - Failed: {self.stats['failed']}")
        print(f"  - Skipped: {self.stats['skipped']}")
        print(f"Success rate: {report['success_rate']:.1f}%")
        print(f"\nReport saved to: {report_path}")
        
        if self.errors:
            print(f"\n⚠️  {len(self.errors)} errors occurred. See migration_summary.json for details.")


def main():
    """Execute migration"""
    source_dir = Path("data/oe_final_outputs/Completed extractions")
    output_dir = Path("data/migrated_extracted")
    evaluation_report = Path("data/evaluation_report.json")
    
    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        return
    
    if not evaluation_report.exists():
        print(f"Error: Evaluation report not found: {evaluation_report}")
        return
    
    migrator = ExtractionMigrator()
    migrator.migrate_all(source_dir, output_dir, evaluation_report)


if __name__ == "__main__":
    main()