#!/usr/bin/env python3
"""
Prepare the complete knowledge base for Bronchmonkey NLP/RAG system
Creates master indices and prepares data for search
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from datetime import datetime

class KnowledgeBaseBuilder:
    def __init__(self):
        self.base_dir = Path("/mnt/c/Users/russe/OneDrive/07_Technology_Tools/IP_chat2")
        self.migrated_dir = self.base_dir / "data/migrated_extracted"
        self.textbook_dir = self.base_dir / "data/textbook_extractions/Principles_Practices"
        self.indices_dir = self.base_dir / "data/indices"
        
        # Create indices directory if needed
        self.indices_dir.mkdir(parents=True, exist_ok=True)
        
    def create_migrated_index(self) -> Dict[str, Any]:
        """Create master index for migrated research articles"""
        print("Creating index for migrated research articles...")
        
        articles = []
        categories = {}
        procedures_set = set()
        conditions_set = set()
        
        json_files = list(self.migrated_dir.glob("*.json"))
        
        for file_path in json_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                article_info = {
                    "filename": file_path.name,
                    "document_id": data.get('source', {}).get('document_id', ''),
                    "title": "",
                    "year": None,
                    "authors": [],
                    "study_type": "",
                    "key_findings": [],
                    "procedures": [],
                    "conditions": []
                }
                
                # Extract metadata
                if 'document' in data:
                    doc = data['document']
                    if 'metadata' in doc:
                        meta = doc['metadata']
                        article_info['title'] = meta.get('title', '')
                        article_info['year'] = meta.get('year', None)
                        article_info['authors'] = meta.get('authors', [])[:3]  # First 3 authors
                    
                    # Extract study type from sections
                    if 'sections' in doc:
                        methods = doc['sections'].get('methods', '')
                        if 'randomized' in methods.lower():
                            article_info['study_type'] = 'RCT'
                        elif 'systematic review' in methods.lower():
                            article_info['study_type'] = 'Systematic Review'
                        elif 'meta-analysis' in methods.lower():
                            article_info['study_type'] = 'Meta-Analysis'
                        elif 'retrospective' in methods.lower():
                            article_info['study_type'] = 'Retrospective'
                        elif 'prospective' in methods.lower():
                            article_info['study_type'] = 'Prospective'
                        else:
                            article_info['study_type'] = 'Observational'
                
                # Extract clinical data if available
                if 'clinical_extraction' in data:
                    clinical = data['clinical_extraction']
                    article_info['key_findings'] = clinical.get('key_findings', [])[:5]
                    
                    # Extract procedures from text
                    title_lower = article_info['title'].lower()
                    if 'ebus' in title_lower:
                        article_info['procedures'].append('EBUS')
                        procedures_set.add('EBUS')
                    if 'bronchoscopy' in title_lower:
                        article_info['procedures'].append('Bronchoscopy')
                        procedures_set.add('Bronchoscopy')
                    if 'cryotherapy' in title_lower or 'cryobiopsy' in title_lower:
                        article_info['procedures'].append('Cryotherapy')
                        procedures_set.add('Cryotherapy')
                    if 'stent' in title_lower:
                        article_info['procedures'].append('Airway Stenting')
                        procedures_set.add('Airway Stenting')
                    if 'valve' in title_lower:
                        article_info['procedures'].append('Endobronchial Valve')
                        procedures_set.add('Endobronchial Valve')
                    
                    # Extract conditions
                    if 'emphysema' in title_lower:
                        article_info['conditions'].append('Emphysema')
                        conditions_set.add('Emphysema')
                    if 'cancer' in title_lower or 'malignant' in title_lower:
                        article_info['conditions'].append('Lung Cancer')
                        conditions_set.add('Lung Cancer')
                    if 'pneumothorax' in title_lower:
                        article_info['conditions'].append('Pneumothorax')
                        conditions_set.add('Pneumothorax')
                
                # Categorize by study type
                study_type = article_info['study_type']
                categories[study_type] = categories.get(study_type, 0) + 1
                
                articles.append(article_info)
                
            except Exception as e:
                print(f"  Error processing {file_path.name}: {e}")
        
        # Create the index
        index = {
            "source": "Research Articles Database",
            "extraction_date": datetime.now().isoformat(),
            "total_articles": len(articles),
            "articles": sorted(articles, key=lambda x: x.get('year', 0), reverse=True),
            "categories": categories,
            "unique_procedures": sorted(list(procedures_set)),
            "unique_conditions": sorted(list(conditions_set)),
            "statistics": {
                "total_files": len(json_files),
                "successfully_indexed": len(articles),
                "rcts": len([a for a in articles if a['study_type'] == 'RCT']),
                "systematic_reviews": len([a for a in articles if 'Systematic' in a['study_type']]),
                "with_clinical_data": len([a for a in articles if a['key_findings']])
            }
        }
        
        # Save the index
        index_path = self.indices_dir / "migrated_articles_index.json"
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)
        
        print(f"  Created index with {len(articles)} articles")
        print(f"  Saved to: {index_path}")
        
        return index
    
    def create_combined_knowledge_base(self) -> Dict[str, Any]:
        """Create a unified knowledge base combining research and textbook content"""
        print("\nCreating combined knowledge base...")
        
        # Load the textbook index
        textbook_index_path = self.textbook_dir / "MASTER_INDEX.json"
        with open(textbook_index_path, 'r', encoding='utf-8') as f:
            textbook_index = json.load(f)
        
        # Load or create the migrated articles index
        migrated_index_path = self.indices_dir / "migrated_articles_index.json"
        if migrated_index_path.exists():
            with open(migrated_index_path, 'r', encoding='utf-8') as f:
                migrated_index = json.load(f)
        else:
            migrated_index = self.create_migrated_index()
        
        # Create combined index
        combined = {
            "name": "Bronchmonkey Complete Knowledge Base",
            "version": "2.0",
            "created": datetime.now().isoformat(),
            "sources": {
                "research_articles": {
                    "count": migrated_index['total_articles'],
                    "path": "data/migrated_extracted",
                    "index": "data/indices/migrated_articles_index.json"
                },
                "textbook_chapters": {
                    "count": textbook_index['total_chapters'],
                    "path": "data/textbook_extractions/Principles_Practices",
                    "index": "data/textbook_extractions/Principles_Practices/MASTER_INDEX.json"
                }
            },
            "total_documents": migrated_index['total_articles'] + textbook_index['total_chapters'],
            "procedures": {
                "from_research": migrated_index['unique_procedures'],
                "from_textbook": textbook_index['search_index']['procedures'],
                "combined": sorted(list(set(
                    migrated_index['unique_procedures'] + 
                    textbook_index['search_index']['procedures']
                )))
            },
            "conditions": {
                "from_research": migrated_index['unique_conditions'],
                "from_textbook": textbook_index['search_index']['common_conditions'],
                "combined": sorted(list(set(
                    migrated_index['unique_conditions'] + 
                    textbook_index['search_index']['common_conditions']
                )))
            },
            "search_config": {
                "vector_weight": 0.5,
                "keyword_weight": 0.3,
                "structured_weight": 0.2,
                "chunk_size": 500,
                "overlap": 50,
                "top_k": 10
            }
        }
        
        # Save combined index
        combined_path = self.indices_dir / "combined_knowledge_base.json"
        with open(combined_path, 'w', encoding='utf-8') as f:
            json.dump(combined, f, indent=2)
        
        print(f"  Total documents: {combined['total_documents']}")
        print(f"  Unique procedures: {len(combined['procedures']['combined'])}")
        print(f"  Unique conditions: {len(combined['conditions']['combined'])}")
        print(f"  Saved to: {combined_path}")
        
        return combined
    
    def create_search_chunks(self):
        """Create search-optimized chunks from all documents"""
        print("\nCreating search chunks...")
        
        chunks = []
        chunk_id = 0
        
        # Process research articles
        print("  Processing research articles...")
        for file_path in self.migrated_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Create chunks from different sections
                if 'document' in data and 'sections' in data['document']:
                    sections = data['document']['sections']
                    title = data['document'].get('metadata', {}).get('title', '')
                    
                    # Abstract chunk
                    if 'abstract' in sections and sections['abstract']:
                        chunks.append({
                            'chunk_id': f"research_{chunk_id}",
                            'source_file': file_path.name,
                            'source_type': 'research',
                            'title': title,
                            'section': 'abstract',
                            'content': sections['abstract'][:2000],  # Limit size
                            'metadata': {
                                'year': data['document'].get('metadata', {}).get('year'),
                                'authors': data['document'].get('metadata', {}).get('authors', [])[:3]
                            }
                        })
                        chunk_id += 1
                    
                    # Results chunk
                    if 'results' in sections and sections['results']:
                        chunks.append({
                            'chunk_id': f"research_{chunk_id}",
                            'source_file': file_path.name,
                            'source_type': 'research',
                            'title': title,
                            'section': 'results',
                            'content': sections['results'][:2000],
                            'metadata': {}
                        })
                        chunk_id += 1
                    
                    # Clinical extraction chunk
                    if 'clinical_extraction' in data:
                        clinical = data['clinical_extraction']
                        if clinical.get('key_findings'):
                            chunks.append({
                                'chunk_id': f"research_{chunk_id}",
                                'source_file': file_path.name,
                                'source_type': 'research',
                                'title': title,
                                'section': 'key_findings',
                                'content': ' '.join(clinical['key_findings']),
                                'metadata': {}
                            })
                            chunk_id += 1
                            
            except Exception as e:
                print(f"    Error processing {file_path.name}: {e}")
        
        # Process textbook chapters
        print("  Processing textbook chapters...")
        for file_path in self.textbook_dir.glob("*.json"):
            if file_path.name in ['MASTER_INDEX.json', 'EXTRACTION_ANALYSIS_REPORT.md']:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Skip supplementary table files
                if file_path.name.endswith('_table.json'):
                    continue
                
                # Handle title that might be a dict or string
                title_data = data.get('chapter_metadata', {}).get('title', '')
                if isinstance(title_data, dict):
                    title = title_data.get('value', '')
                else:
                    title = title_data or ''
                
                # Key points chunk
                if 'chapter_metadata' in data and 'key_points' in data['chapter_metadata']:
                    key_points = data['chapter_metadata']['key_points']
                    if key_points:
                        chunks.append({
                            'chunk_id': f"textbook_{chunk_id}",
                            'source_file': file_path.name,
                            'source_type': 'textbook',
                            'title': title,
                            'section': 'key_points',
                            'content': ' '.join(key_points),
                            'metadata': {
                                'chapter': data['chapter_metadata'].get('chapter_number')
                            }
                        })
                        chunk_id += 1
                
                # Clinical procedures chunks
                if 'clinical_procedures' in data:
                    for proc in data['clinical_procedures'][:3]:  # First 3 procedures
                        if 'steps' in proc:
                            chunks.append({
                                'chunk_id': f"textbook_{chunk_id}",
                                'source_file': file_path.name,
                                'source_type': 'textbook',
                                'title': str(title) if title else '',
                                'section': 'procedure',
                                'content': f"{proc.get('name', '')}: {' '.join(proc['steps'][:5])}",
                                'metadata': {
                                    'procedure_name': proc.get('name', '')
                                }
                            })
                            chunk_id += 1
                            
            except Exception as e:
                print(f"    Error processing {file_path.name}: {e}")
        
        # Save chunks
        chunks_path = self.indices_dir / "search_chunks.json"
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump({
                'total_chunks': len(chunks),
                'created': datetime.now().isoformat(),
                'chunks': chunks
            }, f, indent=2)
        
        print(f"  Created {len(chunks)} search chunks")
        print(f"  Saved to: {chunks_path}")
        
        return chunks
    
    def create_quick_lookup(self):
        """Create a quick lookup index for common queries"""
        print("\nCreating quick lookup index...")
        
        lookup = {
            "diagnostic_yields": {},
            "complication_rates": {},
            "procedure_steps": {},
            "indications": {},
            "contraindications": {}
        }
        
        # Process research articles for yields and complications
        for file_path in self.migrated_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'clinical_extraction' in data:
                    clinical = data['clinical_extraction']
                    title = data.get('document', {}).get('metadata', {}).get('title', '')
                    
                    # Extract diagnostic yields
                    if clinical.get('diagnostic_yields'):
                        for proc, yields in clinical['diagnostic_yields'].items():
                            if proc not in lookup['diagnostic_yields']:
                                lookup['diagnostic_yields'][proc] = []
                            lookup['diagnostic_yields'][proc].append({
                                'source': title[:50],
                                'data': yields
                            })
                    
                    # Extract complication rates
                    if clinical.get('complication_rates'):
                        for complication, rate_info in clinical['complication_rates'].items():
                            if complication not in lookup['complication_rates']:
                                lookup['complication_rates'][complication] = []
                            lookup['complication_rates'][complication].append({
                                'source': title[:50],
                                'rate': rate_info.get('rate', 'N/A')
                            })
                            
            except Exception as e:
                continue
        
        # Process textbook for procedure steps
        for file_path in self.textbook_dir.glob("*.json"):
            if file_path.name.endswith('_table.json') or file_path.name == 'MASTER_INDEX.json':
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                title = data.get('chapter_metadata', {}).get('title', '')
                
                # Extract procedure steps
                if 'clinical_procedures' in data:
                    for proc in data['clinical_procedures']:
                        proc_name = proc.get('name', '')
                        if proc_name and 'steps' in proc:
                            lookup['procedure_steps'][proc_name] = {
                                'source': title,
                                'steps': proc['steps']
                            }
                            
            except Exception as e:
                continue
        
        # Save lookup index
        lookup_path = self.indices_dir / "quick_lookup.json"
        with open(lookup_path, 'w', encoding='utf-8') as f:
            json.dump(lookup, f, indent=2)
        
        print(f"  Created quick lookup with:")
        print(f"    - {len(lookup['diagnostic_yields'])} procedures with yields")
        print(f"    - {len(lookup['complication_rates'])} complications tracked")
        print(f"    - {len(lookup['procedure_steps'])} procedures with steps")
        print(f"  Saved to: {lookup_path}")
        
        return lookup
    
    def run_full_preparation(self):
        """Run the complete knowledge base preparation"""
        print("="*60)
        print("BRONCHMONKEY KNOWLEDGE BASE PREPARATION")
        print("="*60)
        
        # Step 1: Create indices
        migrated_index = self.create_migrated_index()
        
        # Step 2: Create combined knowledge base
        combined_kb = self.create_combined_knowledge_base()
        
        # Step 3: Create search chunks
        chunks = self.create_search_chunks()
        
        # Step 4: Create quick lookup
        lookup = self.create_quick_lookup()
        
        print("\n" + "="*60)
        print("PREPARATION COMPLETE!")
        print("="*60)
        print(f"✅ Indexed {migrated_index['total_articles']} research articles")
        print(f"✅ Indexed 41 textbook chapters")
        print(f"✅ Created {len(chunks)} search chunks")
        print(f"✅ Built quick lookup indices")
        print("\nKnowledge base is ready for use!")
        print("\nNext step: Run 'python chatbot_app.py' to start Bronchmonkey")
        
        return True


if __name__ == "__main__":
    builder = KnowledgeBaseBuilder()
    builder.run_full_preparation()