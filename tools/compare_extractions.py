#!/usr/bin/env python3
"""
Compare original extraction with gold standard enhanced version
Shows the improvements made by the enhancement process
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Any
from collections import defaultdict


def compare_extractions(original: Dict, enhanced: Dict) -> Dict:
    """Compare two extractions and generate report"""
    
    report = {
        'sections_added': [],
        'sections_enhanced': [],
        'metrics': {
            'original': {},
            'enhanced': {},
            'improvement': {}
        },
        'quality_improvements': []
    }
    
    # Check sections
    original_sections = set(k for k in original.keys() if original[k])
    enhanced_sections = set(k for k in enhanced.keys() if enhanced[k])
    
    report['sections_added'] = list(enhanced_sections - original_sections)
    
    # Count improvements
    for section in ['diagnostic_approaches', 'risk_models', 'tables', 'clinical_guidelines']:
        if section in original:
            report['metrics']['original'][section] = len(original.get(section, []))
        if section in enhanced:
            report['metrics']['enhanced'][section] = len(enhanced.get(section, []))
    
    # Check specific enhancements
    
    # 1. Risk model separation
    if 'risk_models' in enhanced and len(enhanced['risk_models']) > 0:
        if 'risk_models' not in original or len(original.get('risk_models', [])) == 0:
            report['quality_improvements'].append(
                f"✅ Separated {len(enhanced['risk_models'])} risk models from diagnostic approaches"
            )
    
    # 2. Clinical interpretations in tables
    if 'tables' in enhanced:
        tables_with_interp = sum(
            1 for t in enhanced['tables'] 
            if 'clinical_interpretation' in t and t['clinical_interpretation']
        )
        if tables_with_interp > 0:
            report['quality_improvements'].append(
                f"✅ Added clinical interpretation to {tables_with_interp} tables"
            )
    
    # 3. Performance metrics normalization
    def count_normalized_metrics(data):
        count = 0
        def traverse(obj):
            nonlocal count
            if isinstance(obj, dict):
                if 'performance' in obj:
                    perf = obj['performance']
                    if isinstance(perf, dict):
                        for metric in ['sensitivity', 'specificity', 'ppv', 'npv']:
                            if metric in perf and isinstance(perf[metric], dict):
                                if 'value' in perf[metric] and 'unit' in perf[metric]:
                                    count += 1
                for v in obj.values():
                    traverse(v)
            elif isinstance(obj, list):
                for item in obj:
                    traverse(item)
        traverse(data)
        return count
    
    enhanced_metrics = count_normalized_metrics(enhanced)
    if enhanced_metrics > 0:
        report['quality_improvements'].append(
            f"✅ Normalized {enhanced_metrics} performance metrics to standard format"
        )
    
    # 4. Missing sections added
    for section in ['guideline_adherence', 'technology_and_technique', 'conclusion']:
        if section in enhanced and enhanced[section]:
            if section not in original or not original[section]:
                report['quality_improvements'].append(
                    f"✅ Added missing section: {section}"
                )
    
    # 5. Reference additions
    def count_references(data):
        count = 0
        def traverse(obj):
            nonlocal count
            if isinstance(obj, dict):
                if 'reference' in obj and obj['reference']:
                    count += 1
                for v in obj.values():
                    traverse(v)
            elif isinstance(obj, list):
                for item in obj:
                    traverse(item)
        traverse(data)
        return count
    
    orig_refs = count_references(original)
    enhanced_refs = count_references(enhanced)
    if enhanced_refs > orig_refs:
        report['quality_improvements'].append(
            f"✅ Added {enhanced_refs - orig_refs} inline references"
        )
    
    # 6. Definitions enrichment
    orig_defs = len(original.get('definitions', []))
    enhanced_defs = len(enhanced.get('definitions', []))
    if enhanced_defs > orig_defs:
        report['quality_improvements'].append(
            f"✅ Enriched definitions: {orig_defs} → {enhanced_defs}"
        )
    
    return report


def print_comparison(report: Dict):
    """Print formatted comparison report"""
    
    print("\n" + "="*60)
    print("📊 EXTRACTION COMPARISON REPORT")
    print("="*60)
    
    if report['sections_added']:
        print("\n✨ New Sections Added:")
        for section in report['sections_added']:
            print(f"  • {section}")
    
    print("\n📈 Content Metrics:")
    print(f"{'Section':<25} {'Original':<10} {'Enhanced':<10} {'Change':<10}")
    print("-"*55)
    
    for section in report['metrics']['original'].keys():
        orig = report['metrics']['original'].get(section, 0)
        enh = report['metrics']['enhanced'].get(section, 0)
        change = enh - orig
        symbol = "↑" if change > 0 else "→" if change == 0 else "↓"
        print(f"{section:<25} {orig:<10} {enh:<10} {symbol}{abs(change):<9}")
    
    if report['quality_improvements']:
        print("\n🎯 Quality Improvements:")
        for improvement in report['quality_improvements']:
            print(f"  {improvement}")
    
    # Calculate overall improvement score
    total_improvements = (
        len(report['sections_added']) + 
        len(report['quality_improvements'])
    )
    
    print("\n" + "="*60)
    print(f"⭐ Total Improvements: {total_improvements}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Compare original and enhanced extractions'
    )
    
    parser.add_argument(
        'original',
        type=Path,
        help='Path to original extraction JSON'
    )
    
    parser.add_argument(
        'enhanced',
        type=Path,
        help='Path to enhanced extraction JSON'
    )
    
    parser.add_argument(
        '--save-report',
        type=Path,
        help='Save comparison report to JSON file'
    )
    
    args = parser.parse_args()
    
    # Load JSONs
    with open(args.original, 'r', encoding='utf-8') as f:
        original = json.load(f)
    
    with open(args.enhanced, 'r', encoding='utf-8') as f:
        enhanced = json.load(f)
    
    # Generate comparison
    report = compare_extractions(original, enhanced)
    
    # Print report
    print_comparison(report)
    
    # Save if requested
    if args.save_report:
        with open(args.save_report, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)
        print(f"\n📄 Report saved to: {args.save_report}")


if __name__ == "__main__":
    main()