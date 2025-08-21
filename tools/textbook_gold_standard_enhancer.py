#!/usr/bin/env python3
"""
Gold Standard Textbook Extraction Enhancer
==========================================
Transforms production textbook extractions into gold-standard quality JSON
based on the enhanced structure from ChatGPT analysis.

Key Enhancements:
1. Separates risk models from diagnostic approaches
2. Adds missing narrative sections (guideline adherence, technology/technique, conclusion)
3. Enriches tables with clinical interpretation
4. Adds inline references to all items
5. Reduces redundancy through intelligent consolidation
6. Structures performance metrics consistently
7. Organizes clinical pearls and cases properly
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import argparse
from dataclasses import dataclass, field, asdict
from collections import defaultdict
import hashlib

# Import OpenAI for targeted extraction of missing sections
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class PerformanceMetrics:
    """Standardized performance metrics"""
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    ppv: Optional[float] = None
    npv: Optional[float] = None
    accuracy: Optional[float] = None
    auc: Optional[float] = None
    unit: str = "proportion"
    reference: Optional[str] = None

@dataclass
class EnhancementConfig:
    """Configuration for enhancement process"""
    add_clinical_interpretation: bool = True
    extract_missing_sections: bool = True
    consolidate_duplicates: bool = True
    add_inline_references: bool = True
    normalize_performance_metrics: bool = True
    extract_guideline_adherence: bool = True
    extract_technology_technique: bool = True
    model: str = "gpt-4o"  # or gpt-5
    verbose: bool = False

class TextbookGoldStandardEnhancer:
    """Enhances textbook extractions to gold-standard quality"""
    
    def __init__(self, config: EnhancementConfig = None):
        self.config = config or EnhancementConfig()
        self.client = OpenAI() if os.getenv("OPENAI_API_KEY") else None
        
        # Pattern for detecting risk models vs diagnostic approaches
        self.risk_model_patterns = [
            r"mayo", r"swensen", r"herder", r"brock", r"treat", 
            r"lung.?rads", r"risk.?(model|calculator|score)",
            r"prediction.?model", r"malignancy.?calculator"
        ]
        
        # Pattern for detecting performance metrics
        self.metric_patterns = {
            'sensitivity': r'sensitivity[:\s]+([0-9.]+)%?',
            'specificity': r'specificity[:\s]+([0-9.]+)%?',
            'ppv': r'(?:ppv|positive.?predictive)[:\s]+([0-9.]+)%?',
            'npv': r'(?:npv|negative.?predictive)[:\s]+([0-9.]+)%?',
            'accuracy': r'accuracy[:\s]+([0-9.]+)%?',
            'auc': r'(?:auc|area.?under|c.?statistic)[:\s]+([0-9.]+)'
        }
    
    def _call_llm_json(self, system: str, user: str) -> Any:
        """
        Call OpenAI with strict JSON output, routing GPT-5 to Responses API
        and others to Chat Completions.
        """
        if not self.client:
            return None
        
        model = self.config.model
        use_responses = model.lower().startswith("gpt-5") and not model.lower().startswith("gpt-5-chat")
        
        try:
            if use_responses:
                # GPT-5 via Responses API — enforce JSON via instructions, not response_format
                combined = f"{system}\n\n{user}\n\nReturn one valid JSON object only. No prose outside the JSON."
                resp = self.client.responses.create(
                    model=model,
                    input=combined
                )
                content = getattr(resp, "output_text", None) or str(resp)
            else:
                # Chat Completions API
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user}
                    ],
                    response_format={"type": "json_object"},
                    temperature=0
                )
                content = resp.choices[0].message.content
            
            # Strip accidental fences
            if content and content.strip().startswith("```"):
                stripped = content.strip().strip("`")
                content = stripped[4:].lstrip() if stripped.lower().startswith("json") else stripped
            
            return json.loads(content)
        except Exception:
            return None
        
    def enhance(self, 
                input_json: Dict[str, Any], 
                source_text: Optional[str] = None,
                adobe_json: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Main enhancement pipeline
        
        Args:
            input_json: Original extraction JSON
            source_text: Full chapter text (for extracting missing sections)
            adobe_json: Adobe Extract JSON (for table enrichment)
        
        Returns:
            Enhanced gold-standard JSON
        """
        
        if self.config.verbose:
            print("🚀 Starting gold-standard enhancement...")
        
        # Start with deep copies of input for safety and reference
        original = json.loads(json.dumps(input_json))
        enhanced = json.loads(json.dumps(input_json))
        
        # Step 1: Separate risk models from diagnostic approaches
        if self.config.consolidate_duplicates:
            enhanced = self._separate_risk_models(enhanced)
        
        # Step 2: Extract missing narrative sections
        if self.config.extract_missing_sections and source_text:
            enhanced = self._extract_missing_sections(enhanced, source_text)
        
        # Step 3: Add clinical interpretations to tables
        if self.config.add_clinical_interpretation:
            enhanced = self._add_clinical_interpretations(enhanced, source_text)
        
        # Step 4: Normalize performance metrics and move numeric fields into a unified
        #         "performance" object with proportion units
        if self.config.normalize_performance_metrics:
            enhanced = self._normalize_performance_metrics(enhanced)
            enhanced = self._move_metrics_into_performance(enhanced)
            enhanced = self._fill_performance_from_text(enhanced)
        
        # Step 5: Add inline references
        if self.config.add_inline_references:
            enhanced = self._add_inline_references(enhanced)
        
        # Step 6: Consolidate duplicates
        if self.config.consolidate_duplicates:
            enhanced = self._consolidate_duplicates(enhanced)
        
        # Step 7: Structure clinical pearls and cases
        enhanced = self._structure_clinical_content(enhanced)

        # Step 8: Post-normalization cleanup and integrity checks
        enhanced = self._ensure_chapter_metadata_integrity(original, enhanced)
        enhanced = self._normalize_page_ranges(enhanced)
        enhanced = self._fix_risk_model_references(enhanced)
        enhanced = self._standardize_definitions(enhanced)
        enhanced = self._populate_guideline_orgs(enhanced)
        enhanced = self._enforce_source_excerpt_requirement(enhanced)
        enhanced = self._normalize_paths(enhanced)
        # Avoid stray top-level reference field
        if isinstance(enhanced, dict) and 'reference' in enhanced:
            enhanced.pop('reference', None)

        # Step 9: Add extraction metadata
        enhanced = self._add_metadata(enhanced)
        
        if self.config.verbose:
            print("✅ Enhancement complete!")
        
        return enhanced

    # ---------- Integrity and normalization helpers ----------

    def _ensure_chapter_metadata_integrity(self, original: Dict, enhanced: Dict) -> Dict:
        """Ensure authors/title not dropped; restore from original if missing."""
        try:
            emd = enhanced.setdefault('chapter_metadata', {})
            omd = original.get('chapter_metadata', {})
            # Restore authors if dropped or empty
            if not emd.get('authors') and omd.get('authors'):
                emd['authors'] = omd['authors']
        except Exception:
            pass
        # Clean conclusion points
        if isinstance(enhanced.get('conclusion'), dict):
            points = enhanced['conclusion'].get('points')
            if isinstance(points, list):
                cleaned = []
                for p in points:
                    if not isinstance(p, str):
                        continue
                    q = p
                    # Remove dot leaders and excessive punctuation
                    q = re.sub(r'[•·◦]+', ' ', q)
                    q = re.sub(r'\.{3,}|…{1,}', ' ', q)
                    # Drop lines that look like headers or references listings
                    if re.search(r'\bReferences\b', q, re.I):
                        continue
                    if re.match(r'^\s*\d+\s*$', q):
                        continue
                    q = q.strip()
                    if len(q) >= 20:
                        cleaned.append(q)
                enhanced['conclusion']['points'] = cleaned[:5]
        return enhanced

    def _normalize_page_ranges(self, data: Dict) -> Dict:
        """Convert all page_range strings like '1-6' to objects {start,end}."""
        def fix(obj: Any) -> Any:
            if isinstance(obj, dict):
                if 'page_range' in obj and isinstance(obj['page_range'], str):
                    s = obj['page_range']
                    m = re.match(r'\s*(\d+)\s*(?:-\s*(\d+))?\s*$', s)
                    if m:
                        start = int(m.group(1))
                        end = int(m.group(2)) if m.group(2) else start
                        obj['page_range'] = {'start': start, 'end': end}
                return {k: fix(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [fix(v) for v in obj]
            return obj
        return fix(data)

    def _fix_risk_model_references(self, data: Dict) -> Dict:
        """Ensure risk_models.reference is null if empty string."""
        rms = data.get('risk_models', [])
        for rm in rms:
            if isinstance(rm, dict):
                if 'reference' in rm and (rm['reference'] is None or str(rm['reference']).strip() == ''):
                    rm['reference'] = None
        return data

    def _standardize_definitions(self, data: Dict) -> Dict:
        """Normalize definitions array to two shapes: quoted vs glossary."""
        defs = data.get('definitions')
        if not isinstance(defs, list):
            return data
        normalized = []
        for d in defs:
            if not isinstance(d, dict):
                continue
            if 'term' in d and 'definition' in d:
                item = dict(d)
                item.setdefault('present_in_source', False)
                item.setdefault('added_by', 'enhancer')
                normalized.append(item)
            elif 'content' in d:
                item = dict(d)
                # Ensure keys exist for consistency
                item.setdefault('reference', item.get('reference', '[Chapter Text]'))
                # Ensure page_range object type if present as string
                pr = item.get('page_range')
                if isinstance(pr, str):
                    m = re.match(r'\s*(\d+)\s*(?:-\s*(\d+))?\s*$', pr)
                    if m:
                        item['page_range'] = {
                            'start': int(m.group(1)),
                            'end': int(m.group(2)) if m.group(2) else int(m.group(1))
                        }
                normalized.append(item)
            else:
                # Pass through unknown shape
                normalized.append(d)
        data['definitions'] = normalized
        return data

    def _move_metrics_into_performance(self, data: Dict) -> Dict:
        """Sweep all dicts and relocate metric keys into a nested performance object."""
        METRIC_KEYS = {'sensitivity','specificity','ppv','npv','accuracy','auc'}
        def sweep(obj: Any) -> Any:
            if isinstance(obj, dict):
                # Recurse first
                new_obj = {k: sweep(v) for k, v in obj.items()}
                # If any metric present at this level, consolidate into performance
                if any(k in new_obj for k in METRIC_KEYS):
                    perf = {}
                    notes = []
                    for k in list(METRIC_KEYS):
                        if k in new_obj:
                            v = new_obj.pop(k)
                            # Convert strings like "76% (both approaches)" to proportion and capture trailing text
                            if isinstance(v, str):
                                m = re.search(r'([0-9.]+)', v)
                                if m:
                                    num = float(m.group(1))
                                    if num > 1:
                                        num = num/100.0
                                    perf[k] = {'value': num, 'unit': 'proportion'}
                                extra = v[m.end():].strip() if m else v
                                if extra:
                                    notes.append(f"{k}: {extra}")
                            elif isinstance(v, (int,float)):
                                num = float(v)
                                if num > 1:
                                    num = num/100.0
                                perf[k] = {'value': num, 'unit': 'proportion'}
                            elif isinstance(v, dict) and 'value' in v:
                                # Already normalized
                                perf[k] = v
                    if perf:
                        # Merge with existing performance if present
                        if isinstance(new_obj.get('performance'), dict):
                            new_obj['performance'].update(perf)
                        else:
                            new_obj['performance'] = perf
                        if notes:
                            new_obj['performance'].setdefault('notes', '; '.join(notes))
                return new_obj
            elif isinstance(obj, list):
                return [sweep(v) for v in obj]
            return obj
        return sweep(data)

    def _fill_performance_from_text(self, data: Dict) -> Dict:
        """Populate performance from interpretation text if missing and parsable."""
        items = data.get('diagnostic_approaches', [])
        for a in items:
            if isinstance(a, dict) and not a.get('performance'):
                it = a.get('interpretation')
                if isinstance(it, str):
                    sens = re.search(r'sens(?:itivity)?[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?', it, re.I)
                    spec = re.search(r'spec(?:ificity)?[^0-9%]*([0-9]+(?:\.[0-9]+)?)%?', it, re.I)
                    if sens or spec:
                        a['performance'] = {}
                        if sens:
                            val = float(sens.group(1))
                            a['performance']['sensitivity'] = {'value': (val/100.0 if val>1 else val), 'unit': 'proportion'}
                        if spec:
                            val = float(spec.group(1))
                            a['performance']['specificity'] = {'value': (val/100.0 if val>1 else val), 'unit': 'proportion'}
        return data

    def _populate_guideline_orgs(self, data: Dict) -> Dict:
        """Fill source_organization in guidelines when obvious (ACCP/BTS/Fleischner)."""
        org_map = {
            'accp': 'American College of Chest Physicians',
            'bts': 'British Thoracic Society',
            'fleischner': 'Fleischner Society'
        }
        for g in data.get('clinical_guidelines', []) or []:
            if isinstance(g, dict):
                if not g.get('source_organization') and isinstance(g.get('title'), (str, type(None))):
                    t = (g.get('title') or '')
                    for k, full in org_map.items():
                        if k in t.lower():
                            g['source_organization'] = full
                            break
        return data

    def _enforce_source_excerpt_requirement(self, data: Dict) -> Dict:
        """Ensure any item with present_in_source True has non-empty source_excerpt; otherwise flip to False."""
        def walk(obj: Any) -> Any:
            if isinstance(obj, dict):
                if obj.get('present_in_source') is True and not obj.get('source_excerpt'):
                    obj['present_in_source'] = False
                return {k: walk(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [walk(v) for v in obj]
            return obj
        return walk(data)

    def _normalize_paths(self, data: Dict) -> Dict:
        """Standardize path separators in extraction metadata to forward slashes."""
        md = data.get('extraction_metadata')
        if isinstance(md, dict):
            for k in ['source_pdf', 'adobe_json']:
                if isinstance(md.get(k), str):
                    md[k] = md[k].replace('\\', '/')
        return data
    
    def _separate_risk_models(self, data: Dict) -> Dict:
        """Separate risk prediction models from diagnostic approaches"""
        
        if 'diagnostic_approaches' not in data:
            return data
        
        # Initialize risk_models if not present
        if 'risk_models' not in data:
            data['risk_models'] = []
        
        # Check each diagnostic approach
        new_diagnostic = []
        for approach in data['diagnostic_approaches']:
            name = approach.get('name', '').lower()
            
            # Check if this is a risk model
            is_risk_model = any(re.search(pattern, name) for pattern in self.risk_model_patterns)
            
            if is_risk_model:
                # Transform to risk model format
                risk_model = self._transform_to_risk_model(approach)
                data['risk_models'].append(risk_model)
            else:
                new_diagnostic.append(approach)
        
        data['diagnostic_approaches'] = new_diagnostic
        return data
    
    def _transform_to_risk_model(self, approach: Dict) -> Dict:
        """Transform a diagnostic approach into risk model format"""
        
        model = {
            'model_name': approach.get('name', ''),
            'setting': approach.get('purpose', ''),
            'predictors': [],
            'reference': approach.get('reference') or None
        }
        
        # Extract predictors from criteria_or_scoring
        if 'criteria_or_scoring' in approach:
            criteria = approach['criteria_or_scoring']
            if isinstance(criteria, list):
                model['predictors'] = criteria
            elif isinstance(criteria, str):
                # Parse comma-separated predictors
                model['predictors'] = [p.strip() for p in criteria.split(',')]
        
        # Extract performance if present
        if 'performance' in approach:
            model['performance'] = approach['performance']
        
        # Extract cohort info if in text
        if 'interpretation' in approach:
            interp_text = approach.get('interpretation')
            if not isinstance(interp_text, str):
                interp_text = ''
            cohort_match = re.search(r'n\s*=\s*(\d+)', interp_text)
            if cohort_match:
                model['cohort'] = {'n': int(cohort_match.group(1))}
            
            prev_match = re.search(r'prevalence[:\s]+([0-9.]+)%?', interp_text)
            if prev_match:
                if 'cohort' not in model:
                    model['cohort'] = {}
                model['cohort']['prevalence_malignancy'] = f"{prev_match.group(1)}%"
        
        return model
    
    def _extract_missing_sections(self, data: Dict, source_text: str) -> Dict:
        """Extract missing narrative sections using GPT"""
        
        if not self.client:
            print("⚠️ OpenAI client not available, skipping section extraction")
            return data
        
        # Check for guideline adherence section
        if self.config.extract_guideline_adherence and 'guideline_adherence' not in data:
            data['guideline_adherence'] = self._extract_guideline_adherence(source_text)
        
        # Check for technology and technique section
        if self.config.extract_technology_technique and 'technology_and_technique' not in data:
            data['technology_and_technique'] = self._extract_technology_technique(source_text)
        
        # Check for proper conclusion
        if 'conclusion' not in data or not data.get('conclusion'):
            data['conclusion'] = self._extract_conclusion(source_text)
        
        # Check for treatment algorithms
        if 'treatment_algorithms' not in data:
            data['treatment_algorithms'] = self._extract_algorithms(source_text)
        
        return data
    
    def _extract_guideline_adherence(self, text: str) -> Dict:
        """Extract guideline adherence section"""
        
        prompt = """
        Extract information about guideline adherence from this chapter text.
        Look for:
        1. Problems with guideline adherence in practice
        2. Statistics on over-evaluation or under-evaluation
        3. Discordance between guidelines and actual practice
        4. Implications for patient care
        
        Return as JSON:
        {
            "problems_observed": [
                {
                    "finding": "main finding",
                    "details": "specific statistics or examples",
                    "reference": "citation if available"
                }
            ],
            "implication": "overall clinical implication"
        }
        
        If no guideline adherence discussion found, return empty structure.
        """
        
        try:
            result = self._call_llm_json(
                "You are a medical textbook content extractor.",
                f"{prompt}\n\nText:\n{text[:8000]}"
            )
            return result if result and result.get('problems_observed') else {}
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Error extracting guideline adherence: {e}")
            return {}
    
    def _extract_technology_technique(self, text: str) -> List[Dict]:
        """Extract technology and technique section"""
        
        prompt = """
        Extract information about diagnostic technology and techniques from this chapter.
        Look for:
        1. Procedural techniques and their diagnostic yields
        2. Technology comparisons (e.g., bronchoscopy vs TTNB)
        3. Clinical trials comparing modalities
        4. Tips for improving procedural performance
        5. Patient selection criteria
        
        Return as JSON array:
        [
            {
                "topic": "brief topic name",
                "summary": "key findings and statistics",
                "reference": "citation if available"
            }
        ]
        
        Focus on specific yield percentages, safety data, and comparative effectiveness.
        """
        
        try:
            result = self._call_llm_json(
                "You are a medical textbook content extractor.",
                f"{prompt}\n\nText:\n{text[:8000]}"
            )
            return result if isinstance(result, list) else []
            
        except Exception as e:
            if self.config.verbose:
                print(f"⚠️ Error extracting technology/technique: {e}")
            return []
    
    def _extract_conclusion(self, text: str) -> Dict:
        """Extract chapter conclusion"""
        
        # First try to find conclusion section in text
        conclusion_pattern = r'(?:conclusion|summary|key\s+points?|take[\s-]?home)[\s:]*(.{100,2000})'
        match = re.search(conclusion_pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            conclusion_text = match.group(1)
            
            # Parse into bullet points
            points = []
            sentences = re.split(r'[.!?]\s+', conclusion_text)
            for sent in sentences[:5]:  # Take first 5 sentences
                if len(sent) > 20:  # Skip very short fragments
                    points.append(sent.strip())
            
            return {
                "points": points,
                "reference": "[Chapter Conclusion]"
            }
        
        return {}
    
    def _extract_algorithms(self, text: str) -> List[Dict]:
        """Extract treatment/diagnostic algorithms"""
        
        algorithms = []
        
        # Look for figure references to algorithms
        algorithm_pattern = r'(?:fig(?:ure)?\.?\s*\d+)[^.]*(?:algorithm|flowchart|decision\s+tree)'
        matches = re.finditer(algorithm_pattern, text, re.IGNORECASE)
        
        for match in matches:
            # Extract figure number
            fig_match = re.search(r'fig(?:ure)?\.?\s*(\d+)', match.group(), re.IGNORECASE)
            if fig_match:
                algorithms.append({
                    "figure_id": f"Fig. {fig_match.group(1)}",
                    "type": "algorithm",
                    "summary": match.group(),
                    "reference": "[Chapter Text]"
                })
        
        return algorithms
    
    def _add_clinical_interpretations(self, data: Dict, source_text: Optional[str]) -> Dict:
        """Add clinical interpretations to tables"""
        
        if 'tables' not in data:
            return data
        
        for table in data['tables']:
            if 'clinical_interpretation' not in table or not table['clinical_interpretation']:
                # Generate interpretation based on table content
                interpretation = self._generate_table_interpretation(table, source_text)
                if interpretation:
                    table['clinical_interpretation'] = interpretation
        
        return data
    
    def _generate_table_interpretation(self, table: Dict, source_text: Optional[str]) -> str:
        """Generate clinical interpretation for a table"""
        
        title = table.get('title', '').lower()
        
        # Common interpretations based on table type
        if 'differential' in title or 'diagnosis' in title:
            return "The differential spans neoplastic, infectious, inflammatory, vascular, and congenital etiologies; clinical and imaging context is essential to triage toward surveillance vs invasive testing."
        
        elif 'benign' in title and 'malignant' in title:
            return "Classic patterns support surveillance while concerning features increase suspicion and often prompt PET/biopsy."
        
        elif 'prediction' in title or 'model' in title:
            return "Choose models aligned with context (screening vs incidental vs surgical). Consider adding volumetry or PET to refine probability."
        
        elif 'guideline' in title or 'accp' in title.lower() or 'bts' in title.lower():
            return "Guidelines differ in risk thresholds and surveillance duration; choose based on local practice and patient population."
        
        elif 'performance' in title or 'sensitivity' in title:
            return "Performance varies by population and technique; consider local expertise and patient factors when selecting modality."
        
        return ""
    
    def _normalize_performance_metrics(self, data: Dict) -> Dict:
        """Normalize all performance metrics to consistent format"""
        
        def normalize_metrics(obj: Any) -> Any:
            """Recursively normalize metrics in any object"""
            
            if isinstance(obj, dict):
                # Check if this dict contains performance metrics
                if any(key in obj for key in ['sensitivity', 'specificity', 'ppv', 'npv']):
                    normalized = {}
                    
                    for metric, pattern in self.metric_patterns.items():
                        if metric in obj:
                            value = obj[metric]
                            
                            # Convert to proportion if percentage
                            if isinstance(value, str):
                                # Extract numeric value
                                match = re.search(r'([0-9.]+)', value)
                                if match:
                                    num_value = float(match.group(1))
                                    # Convert percentage to proportion
                                    if num_value > 1:
                                        num_value = num_value / 100
                                    normalized[metric] = {
                                        "value": num_value,
                                        "unit": "proportion"
                                    }
                            elif isinstance(value, (int, float)):
                                # Convert to proportion if needed
                                if value > 1:
                                    value = value / 100
                                normalized[metric] = {
                                    "value": value,
                                    "unit": "proportion"
                                }
                            elif isinstance(value, dict):
                                # Already structured
                                normalized[metric] = value
                    
                    # Preserve other fields
                    for key, value in obj.items():
                        if key not in self.metric_patterns:
                            normalized[key] = normalize_metrics(value)
                    
                    return normalized
                else:
                    # Recursively process dict values
                    return {k: normalize_metrics(v) for k, v in obj.items()}
                    
            elif isinstance(obj, list):
                return [normalize_metrics(item) for item in obj]
            else:
                return obj
        
        return normalize_metrics(data)
    
    def _add_inline_references(self, data: Dict) -> Dict:
        """Add inline references to items"""
        
        # Get references list if available
        references = data.get('references', [])
        
        def add_reference(obj: Any, depth: int = 0) -> Any:
            """Recursively add references where missing"""
            
            if depth > 10:  # Prevent infinite recursion
                return obj
                
            if isinstance(obj, dict):
                # Heuristic: only leaf-like content objects should get a reference
                contentish_keys = {'name','title','content','term','figure_id','guideline','drug_name'}
                has_leaf_keys = any(k in obj for k in contentish_keys)
                has_provenance = any(k in obj for k in ['source_page','source_excerpt','page','page_range'])
                if 'reference' not in obj and (has_leaf_keys or has_provenance):
                    # Look for bracketed citation in immediate string fields only
                    immediate = ' '.join([v for v in obj.values() if isinstance(v, str)])
                    citation_match = re.search(r'\[(\d+)\]', immediate)
                    if citation_match:
                        obj['reference'] = f"[{citation_match.group(1)}]"
                    elif has_provenance:
                        obj['reference'] = "[Chapter Text]"
                
                # Recursively process dict values
                return {k: add_reference(v, depth+1) for k, v in obj.items()}
                
            elif isinstance(obj, list):
                return [add_reference(item, depth+1) for item in obj]
            else:
                return obj
        
        return add_reference(data)
    
    def _consolidate_duplicates(self, data: Dict) -> Dict:
        """Consolidate duplicate entries across sections"""
        
        # Track unique items by content hash
        seen_hashes = defaultdict(list)
        
        def get_content_hash(obj: Dict) -> str:
            """Get hash of content for deduplication"""
            # Use name/title as primary key
            key = obj.get('name') or obj.get('title') or obj.get('model_name', '')
            return hashlib.md5(key.lower().encode()).hexdigest()
        
        # Check diagnostic approaches vs risk models
        if 'diagnostic_approaches' in data and 'risk_models' in data:
            diag_hashes = {get_content_hash(d): d for d in data['diagnostic_approaches']}
            risk_hashes = {get_content_hash(r): r for r in data['risk_models']}
            
            # Remove duplicates from diagnostic approaches
            data['diagnostic_approaches'] = [
                d for d in data['diagnostic_approaches'] 
                if get_content_hash(d) not in risk_hashes
            ]
        
        # Check guidelines for duplicate models
        if 'clinical_guidelines' in data:
            for guideline in data['clinical_guidelines']:
                if 'recommendations' in guideline:
                    # Deduplicate recommendations
                    seen = set()
                    unique_recs = []
                    for rec in guideline['recommendations']:
                        rec_text = json.dumps(rec, sort_keys=True)
                        if rec_text not in seen:
                            seen.add(rec_text)
                            unique_recs.append(rec)
                    guideline['recommendations'] = unique_recs
        
        return data
    
    def _structure_clinical_content(self, data: Dict) -> Dict:
        """Structure clinical pearls and cases properly"""
        
        # Ensure clinical_pearls is list of dicts
        if 'clinical_pearls' in data:
            pearls = data['clinical_pearls']
            if isinstance(pearls, list):
                structured_pearls = []
                for pearl in pearls:
                    if isinstance(pearl, str):
                        structured_pearls.append({
                            "content": pearl,
                            "reference": "[Chapter Text]"
                        })
                    elif isinstance(pearl, dict):
                        if 'content' not in pearl and 'text' in pearl:
                            pearl['content'] = pearl.pop('text')
                        structured_pearls.append(pearl)
                data['clinical_pearls'] = structured_pearls
        
        # Structure clinical cases
        if 'clinical_cases' in data:
            cases = data['clinical_cases']
            if isinstance(cases, list):
                structured_cases = []
                for case in cases:
                    if isinstance(case, str):
                        structured_cases.append({
                            "scenario": case,
                            "teaching_point": "Clinical application",
                            "reference": "[Chapter Text]"
                        })
                    elif isinstance(case, dict):
                        if 'scenario' not in case and 'description' in case:
                            case['scenario'] = case.pop('description')
                        structured_cases.append(case)
                data['clinical_cases'] = structured_cases
        
        return data
    
    def _add_metadata(self, data: Dict) -> Dict:
        """Add extraction metadata"""
        
        if 'extraction_metadata' not in data:
            data['extraction_metadata'] = {}
        
        data['extraction_metadata'].update({
            'schema_version': 'gold_standard_v1.0',
            'enhancement_version': '1.0',
            'enhancements_applied': [
                'risk_model_separation',
                'clinical_interpretation_addition',
                'performance_metric_normalization',
                'inline_reference_addition',
                'duplicate_consolidation',
                'missing_section_extraction'
            ]
        })
        
        return data


def main():
    """CLI interface for enhancement"""
    
    parser = argparse.ArgumentParser(
        description='Enhance textbook extraction to gold-standard quality'
    )
    
    parser.add_argument(
        'input_json',
        type=Path,
        help='Path to input extraction JSON'
    )
    
    parser.add_argument(
        '--source-text',
        type=Path,
        help='Path to source text file (for extracting missing sections)'
    )
    
    parser.add_argument(
        '--adobe-json',
        type=Path,
        help='Path to Adobe Extract JSON (for enrichment)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path for enhanced JSON (default: adds _enhanced suffix)'
    )
    
    parser.add_argument(
        '--model',
        default='gpt-4o',
        choices=['gpt-4o', 'gpt-5'],
        help='Model to use for extraction'
    )
    
    parser.add_argument(
        '--no-missing-sections',
        action='store_true',
        help='Skip extraction of missing sections'
    )
    
    parser.add_argument(
        '--no-consolidation',
        action='store_true',
        help='Skip duplicate consolidation'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Verbose output'
    )
    
    args = parser.parse_args()
    
    # Load input JSON
    with open(args.input_json, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    # Load source text if provided
    source_text = None
    if args.source_text and args.source_text.exists():
        with open(args.source_text, 'r', encoding='utf-8') as f:
            source_text = f.read()
    
    # Load Adobe JSON if provided
    adobe_json = None
    if args.adobe_json and args.adobe_json.exists():
        with open(args.adobe_json, 'r', encoding='utf-8') as f:
            adobe_json = json.load(f)
    
    # Configure enhancement
    config = EnhancementConfig(
        extract_missing_sections=not args.no_missing_sections,
        consolidate_duplicates=not args.no_consolidation,
        model=args.model,
        verbose=args.verbose
    )
    
    # Run enhancement
    enhancer = TextbookGoldStandardEnhancer(config)
    enhanced_data = enhancer.enhance(input_data, source_text, adobe_json)
    
    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = args.input_json.parent / f"{args.input_json.stem}_enhanced.json"
    
    # Save enhanced JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Enhanced extraction saved to: {output_path}")
    
    # Print summary statistics
    print("\n📊 Enhancement Summary:")
    print(f"  - Risk models: {len(enhanced_data.get('risk_models', []))}")
    print(f"  - Diagnostic approaches: {len(enhanced_data.get('diagnostic_approaches', []))}")
    print(f"  - Guidelines: {len(enhanced_data.get('clinical_guidelines', []))}")
    print(f"  - Tables: {len(enhanced_data.get('tables', []))}")
    
    if 'guideline_adherence' in enhanced_data and enhanced_data['guideline_adherence']:
        print(f"  - ✅ Guideline adherence section added")
    
    if 'technology_and_technique' in enhanced_data and enhanced_data['technology_and_technique']:
        print(f"  - ✅ Technology/technique section added ({len(enhanced_data['technology_and_technique'])} items)")
    
    if 'conclusion' in enhanced_data and enhanced_data['conclusion']:
        print(f"  - ✅ Conclusion section present")


if __name__ == "__main__":
    main()
