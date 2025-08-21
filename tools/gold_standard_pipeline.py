#!/usr/bin/env python3
"""
Gold Standard Textbook Extraction Pipeline
==========================================
Complete pipeline for extracting and enhancing textbook chapters to gold-standard quality.

This combines:
1. Multi-pass extraction (production_multipass_textbook_extractor.py)
2. Gold standard enhancement (textbook_gold_standard_enhancer.py)
3. Quality validation and reporting

Usage:
    # Single chapter
    python gold_standard_pipeline.py --single "Chapter.pdf" --adobe-json "Chapter.json"
    
    # Batch process all chapters
    python gold_standard_pipeline.py --batch
    
    # With custom model
    python gold_standard_pipeline.py --batch --model gpt-5
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import subprocess
import sys
from datetime import datetime
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib

# Import our modules
sys.path.append(str(Path(__file__).parent))
from textbook_gold_standard_enhancer import TextbookGoldStandardEnhancer, EnhancementConfig
from production_multipass_textbook_extractor import process_single_chapter


class GoldStandardPipeline:
    """Complete pipeline for gold-standard textbook extraction"""
    
    def __init__(self, 
                 model: str = "gpt-4o",
                 output_dir: Path = Path("data/gold_standard_extractions"),
                 verbose: bool = False,
                 enable_fallback: bool = True,
                 fallback_model: str = "gpt-4o"):
        self.model = model
        self.output_dir = output_dir
        self.verbose = verbose
        self.enable_fallback = enable_fallback
        self.fallback_model = fallback_model
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality thresholds
        self.quality_thresholds = {
            'min_sections': 5,  # Minimum expected sections
            'min_definitions': 3,  # Minimum definitions expected
            'min_approaches': 2,  # Minimum diagnostic approaches
            'min_tables': 1,  # Minimum tables expected
        }
        
        # Track processing stats
        self.stats = {
            'processed': 0,
            'enhanced': 0,
            'failed': 0,
            'quality_issues': []
        }
    
    def process_chapter(self, 
                       pdf_path: Path,
                       adobe_json_path: Optional[Path] = None,
                       title: Optional[str] = None) -> Tuple[bool, Path]:
        """
        Process a single chapter through extraction and enhancement
        
        Returns:
            Tuple of (success, output_path)
        """
        
        print(f"\n{'='*60}")
        print(f"📚 Processing: {pdf_path.name}")
        print(f"{'='*60}")
        
        # Step 1: Initial extraction
        print("\n📝 Step 1: Multi-pass extraction...")
        extraction_path = self._run_extraction(pdf_path, adobe_json_path, title)
        
        if not extraction_path or not extraction_path.exists():
            print("❌ Extraction failed")
            self.stats['failed'] += 1
            return False, None
        
        self.stats['processed'] += 1
        
        # Step 2: Load extraction
        with open(extraction_path, 'r', encoding='utf-8') as f:
            extracted_data = json.load(f)
        
        # Step 3: Load source text for enhancement
        source_text = self._extract_text_from_pdf(pdf_path)
        
        # Step 4: Enhance to gold standard
        print("\n✨ Step 2: Enhancing to gold standard...")
        enhanced_data = self._enhance_extraction(
            extracted_data, 
            source_text,
            adobe_json_path
        )

        # Step 5: Validate quality
        print("\n🔍 Step 3: Validating quality...")
        quality_report = self._validate_quality(enhanced_data)

        # Optional fallback if result is essentially empty/poor
        if self.enable_fallback and self._looks_near_empty(enhanced_data) and quality_report.get('score', 0) < 0.6:
            print("\n🛟 Fallback trigger: near-empty result detected; retrying extraction with fallback model",
                  f"({self.fallback_model})…")
            extraction_path_fb = self._run_extraction(
                pdf_path, adobe_json_path, title, model_override=self.fallback_model
            )
            if extraction_path_fb and extraction_path_fb.exists():
                with open(extraction_path_fb, 'r', encoding='utf-8') as f:
                    extracted_fb = json.load(f)
                enhanced_fb = self._enhance_extraction(extracted_fb, source_text, adobe_json_path)
                quality_fb = self._validate_quality(enhanced_fb)

                # If improved or at least populated, use fallback version
                def populated(d: Dict) -> bool:
                    return any(len(d.get(k, []) or []) > 0 for k in (
                        'diagnostic_approaches', 'tables', 'references'))

                if (quality_fb.get('score', 0) >= quality_report.get('score', 0)) or populated(enhanced_fb):
                    print(f"✅ Fallback improved/filled content (score {quality_fb.get('score',0):.2f}); using fallback result")
                    enhanced_data = enhanced_fb
                    quality_report = quality_fb
                    enhanced_data.setdefault('extraction_metadata', {})
                    enhanced_data['extraction_metadata']['fallback_used'] = True
                    enhanced_data['extraction_metadata']['fallback_model'] = self.fallback_model
                else:
                    print("⚠️ Fallback did not improve result; keeping original")
        
        # Step 6: Save enhanced version
        output_path = self.output_dir / f"{pdf_path.stem}_gold_standard.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Gold standard extraction saved: {output_path.name}")
        
        # Step 7: Generate quality report
        self._save_quality_report(output_path, quality_report)
        
        if quality_report['score'] >= 0.8:
            print(f"⭐ Quality Score: {quality_report['score']:.1%} - EXCELLENT")
            self.stats['enhanced'] += 1
        elif quality_report['score'] >= 0.6:
            print(f"✅ Quality Score: {quality_report['score']:.1%} - GOOD")
            self.stats['enhanced'] += 1
        else:
            print(f"⚠️ Quality Score: {quality_report['score']:.1%} - NEEDS REVIEW")
            self.stats['quality_issues'].append(pdf_path.name)
        
        return True, output_path
    
    def _run_extraction(self, 
                       pdf_path: Path,
                       adobe_json_path: Optional[Path],
                       title: Optional[str],
                       model_override: Optional[str] = None) -> Optional[Path]:
        """Run the multi-pass extraction"""
        
        try:
            # Direct function call instead of subprocess
            output_path = process_single_chapter(
                pdf_path=pdf_path,
                adobe_json_path=adobe_json_path,  # Fixed parameter name
                output_dir=self.output_dir / "raw_extractions",
                chapter_title=title,  # Fixed parameter name
                model=(model_override or self.model)
                # Note: verbose not supported by process_single_chapter
            )
            
            return output_path
            
        except Exception as e:
            print(f"❌ Extraction error: {e}")
            return None

    def _looks_near_empty(self, data: Dict) -> bool:
        """Heuristic: no approaches, no tables, no references, but has default definitions."""
        diag = len(data.get('diagnostic_approaches', []) or [])
        tabs = len(data.get('tables', []) or [])
        refs = len(data.get('references', []) or [])
        defs = len(data.get('definitions', []) or [])
        return (diag == 0 and tabs == 0 and refs == 0 and defs >= 10)
    
    def _extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF for enhancement"""
        
        try:
            import fitz  # PyMuPDF
            doc = fitz.open(pdf_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            print(f"⚠️ Could not extract PDF text: {e}")
            return ""
    
    def _enhance_extraction(self, 
                          extracted_data: Dict,
                          source_text: str,
                          adobe_json_path: Optional[Path]) -> Dict:
        """Enhance extraction to gold standard"""
        
        # Load Adobe JSON if available
        adobe_json = None
        if adobe_json_path and adobe_json_path.exists():
            with open(adobe_json_path, 'r', encoding='utf-8') as f:
                adobe_json = json.load(f)
        
        # Configure enhancement
        config = EnhancementConfig(
            add_clinical_interpretation=True,
            extract_missing_sections=bool(source_text),
            consolidate_duplicates=True,
            add_inline_references=True,
            normalize_performance_metrics=True,
            extract_guideline_adherence=True,
            extract_technology_technique=True,
            model=self.model,
            verbose=self.verbose
        )
        
        # Run enhancement
        enhancer = TextbookGoldStandardEnhancer(config)
        enhanced_data = enhancer.enhance(extracted_data, source_text, adobe_json)
        
        return enhanced_data
    
    def _validate_quality(self, data: Dict) -> Dict:
        """Validate extraction quality"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'sections_present': [],
            'sections_missing': [],
            'metrics': {},
            'issues': [],
            'score': 0.0
        }
        
        # Check required sections
        required_sections = [
            'chapter_metadata',
            'diagnostic_approaches',
            'clinical_guidelines',
            'tables',
            'references'
        ]
        
        optional_sections = [
            'risk_models',
            'definitions',
            'clinical_pearls',
            'clinical_cases',
            'procedures',
            'medications',
            'guideline_adherence',
            'technology_and_technique',
            'conclusion'
        ]
        
        # Check presence and content
        total_score = 0
        max_score = 0
        
        for section in required_sections:
            max_score += 2  # Required sections worth more
            if section in data and data[section]:
                report['sections_present'].append(section)
                total_score += 2
                
                # Check content quality
                if isinstance(data[section], list) and len(data[section]) > 0:
                    report['metrics'][f"{section}_count"] = len(data[section])
                elif isinstance(data[section], dict) and len(data[section]) > 0:
                    report['metrics'][f"{section}_fields"] = len(data[section])
            else:
                report['sections_missing'].append(section)
                report['issues'].append(f"Missing required section: {section}")
        
        for section in optional_sections:
            max_score += 1
            if section in data and data[section]:
                report['sections_present'].append(section)
                total_score += 1
                
                if isinstance(data[section], list):
                    report['metrics'][f"{section}_count"] = len(data[section])
        
        # Check specific quality metrics
        if 'diagnostic_approaches' in data:
            approaches_with_performance = sum(
                1 for a in data['diagnostic_approaches'] 
                if 'performance' in a
            )
            if approaches_with_performance > 0:
                total_score += 1
                report['metrics']['approaches_with_performance'] = approaches_with_performance
            max_score += 1
        
        if 'tables' in data:
            tables_with_interpretation = sum(
                1 for t in data['tables'] 
                if 'clinical_interpretation' in t and t['clinical_interpretation']
            )
            if tables_with_interpretation > 0:
                total_score += 1
                report['metrics']['tables_with_interpretation'] = tables_with_interpretation
            max_score += 1
        
        if 'definitions' in data and len(data.get('definitions', [])) >= 5:
            total_score += 1
            report['metrics']['comprehensive_definitions'] = True
        max_score += 1
        
        # Calculate final score
        report['score'] = total_score / max_score if max_score > 0 else 0
        
        # Add quality flags
        if report['score'] >= 0.8:
            report['quality_level'] = 'GOLD'
        elif report['score'] >= 0.6:
            report['quality_level'] = 'SILVER'
        else:
            report['quality_level'] = 'NEEDS_IMPROVEMENT'
        
        return report
    
    def _save_quality_report(self, output_path: Path, report: Dict):
        """Save quality report alongside extraction"""
        
        report_path = output_path.parent / f"{output_path.stem}_quality.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
    
    def process_batch(self, textbook_dir: Path = Path("Textbooks")):
        """Process all chapters in batch"""
        
        print("\n" + "="*60)
        print("🚀 GOLD STANDARD BATCH EXTRACTION")
        print("="*60)
        
        # Find all PDFs and their corresponding JSONs
        pdf_dir = textbook_dir / "Chapter pdfs"
        json_dir = textbook_dir / "Chapter json"
        
        if not pdf_dir.exists():
            print(f"❌ PDF directory not found: {pdf_dir}")
            return
        
        pdf_files = sorted(pdf_dir.glob("*.pdf"))
        print(f"\n📚 Found {len(pdf_files)} chapters to process")
        
        # Process each chapter
        start_time = time.time()
        
        for i, pdf_path in enumerate(pdf_files, 1):
            print(f"\n[{i}/{len(pdf_files)}] Processing {pdf_path.name}...")
            
            # Find corresponding Adobe JSON
            json_path = json_dir / f"{pdf_path.stem}.json"
            if not json_path.exists():
                print(f"⚠️ Adobe JSON not found: {json_path.name}")
                json_path = None
            
            # Extract title from filename
            title = pdf_path.stem.replace("_", " ")
            
            # Process chapter
            success, output_path = self.process_chapter(pdf_path, json_path, title)
            
            if success:
                print(f"✅ Completed: {output_path.name}")
            else:
                print(f"❌ Failed: {pdf_path.name}")
        
        # Print summary
        elapsed = time.time() - start_time
        print("\n" + "="*60)
        print("📊 BATCH PROCESSING COMPLETE")
        print("="*60)
        print(f"Total time: {elapsed/60:.1f} minutes")
        print(f"Chapters processed: {self.stats['processed']}")
        print(f"Successfully enhanced: {self.stats['enhanced']}")
        print(f"Failed: {self.stats['failed']}")
        
        if self.stats['quality_issues']:
            print(f"\n⚠️ Quality issues in:")
            for chapter in self.stats['quality_issues']:
                print(f"  - {chapter}")
        
        print(f"\n📁 Output directory: {self.output_dir}")
    
    def generate_summary_report(self):
        """Generate a summary report of all processed chapters"""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'statistics': self.stats,
            'chapters': []
        }
        
        # Analyze all gold standard files
        for json_file in self.output_dir.glob("*_gold_standard.json"):
            quality_file = json_file.parent / f"{json_file.stem}_quality.json"
            
            chapter_info = {
                'name': json_file.stem.replace("_gold_standard", ""),
                'file': json_file.name,
                'size_kb': json_file.stat().st_size / 1024
            }
            
            # Load quality report if exists
            if quality_file.exists():
                with open(quality_file, 'r') as f:
                    quality = json.load(f)
                    chapter_info['quality_score'] = quality.get('score', 0)
                    chapter_info['quality_level'] = quality.get('quality_level', 'UNKNOWN')
            
            # Load extraction to get stats
            with open(json_file, 'r') as f:
                data = json.load(f)
                chapter_info['sections'] = len([k for k in data.keys() if data[k]])
                chapter_info['diagnostic_approaches'] = len(data.get('diagnostic_approaches', []))
                chapter_info['risk_models'] = len(data.get('risk_models', []))
                chapter_info['tables'] = len(data.get('tables', []))
                chapter_info['references'] = len(data.get('references', []))
            
            report['chapters'].append(chapter_info)
        
        # Save summary report
        report_path = self.output_dir / "extraction_summary.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Summary report saved: {report_path}")
        
        return report


def main():
    """CLI interface"""
    
    parser = argparse.ArgumentParser(
        description='Gold Standard Textbook Extraction Pipeline'
    )
    
    parser.add_argument(
        '--single',
        type=Path,
        help='Process single PDF chapter'
    )
    
    parser.add_argument(
        '--batch',
        action='store_true',
        help='Process all chapters in batch'
    )
    
    parser.add_argument(
        '--adobe-json',
        type=Path,
        help='Adobe Extract JSON for single chapter'
    )
    
    parser.add_argument(
        '--title',
        help='Chapter title for single extraction'
    )
    
    parser.add_argument(
        '--model',
        default='gpt-4o',
        choices=['gpt-4o', 'gpt-5'],
        help='Model to use for extraction and enhancement'
    )
    
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('data/gold_standard_extractions'),
        help='Output directory for gold standard extractions'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Generate summary report after processing'
    )
    
    args = parser.parse_args()
    
    if not args.single and not args.batch:
        parser.error('Specify --single or --batch')
    
    # Initialize pipeline
    pipeline = GoldStandardPipeline(
        model=args.model,
        output_dir=args.output_dir,
        verbose=args.verbose,
        enable_fallback=True,
        fallback_model='gpt-4o'
    )
    
    # Process chapters
    if args.single:
        success, output = pipeline.process_chapter(
            args.single,
            args.adobe_json,
            args.title
        )
        if success:
            print(f"\n✅ Success! Output: {output}")
        else:
            print("\n❌ Processing failed")
            sys.exit(1)
    
    elif args.batch:
        pipeline.process_batch()
    
    # Generate summary if requested
    if args.summary or args.batch:
        pipeline.generate_summary_report()
    
    print("\n✨ Gold standard extraction pipeline complete!")


if __name__ == "__main__":
    main()
