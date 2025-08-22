# Practical Clinical Extraction Refactor for IPchat
## Focus: Granular Clinical Information without Meta-Analysis Complexity

## Overview
This refactor creates a streamlined extraction system that captures detailed clinical information (diagnostic yields, complication rates, methodologies, clinical pearls) without the overhead of full meta-analysis capabilities.

## Key Extraction Goals
- **Diagnostic Yields**: Specific percentages for procedures
- **Complication Rates**: Detailed rates with management strategies  
- **Methodology**: Equipment, techniques, patient selection
- **Clinical Pearls**: Tips, tricks, and learning points
- **Practical Guidance**: Indications, contraindications, troubleshooting

## Phase 1: Simplified Directory Structure

```bash
# Create focused directory structure
mkdir -p ipchat/extraction
mkdir -p ipchat/processing
mkdir -p ipchat/retrieval
mkdir -p ipchat/storage
mkdir -p data/raw/research
mkdir -p data/raw/textbooks
mkdir -p data/extracted
mkdir -p data/chunks
mkdir -p data/indices
```

## Phase 2: Core Clinical Extractor

### 2.1 Create Clinical Data Extractor
Create `ipchat/extraction/clinical_extractor.py`:

```python
"""
Clinical Data Extractor for Interventional Pulmonology
Focuses on actionable clinical information without meta-analysis complexity
"""

import json
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

@dataclass
class ClinicalExtraction:
    """Structured clinical data extraction"""
    document_id: str
    title: str
    document_type: str  # 'research' or 'textbook'
    
    # Key clinical data
    diagnostic_yields: Dict[str, Any]
    complication_rates: Dict[str, Any]
    methodology: Dict[str, Any]
    clinical_pearls: List[str]
    
    # Practical information
    indications: List[str]
    contraindications: List[str]
    patient_selection: Dict[str, Any]
    equipment: List[str]
    
    # Summary for quick reference
    key_findings: List[str]
    practical_tips: List[str]
    
    # Quality metrics
    extraction_confidence: float
    has_numerical_data: bool

class ClinicalDataExtractor:
    """Extract granular clinical information from medical documents"""
    
    def __init__(self, use_gpt5: bool = False):
        """
        Initialize extractor
        Args:
            use_gpt5: If True, use GPT-5 API; if False, use Claude Code internally
        """
        self.use_gpt5 = use_gpt5
        self.procedure_keywords = [
            'ebus', 'tbna', 'bronchoscopy', 'navigational',
            'cryobiopsy', 'bal', 'thoracentesis', 'pleuroscopy'
        ]
        
    def extract(self, content: str, document_type: str = "research") -> ClinicalExtraction:
        """
        Extract clinical information from document
        
        Args:
            content: Document text
            document_type: 'research' or 'textbook'
            
        Returns:
            ClinicalExtraction object with structured data
        """
        
        doc_id = self._generate_doc_id(content)
        
        # Extract different types of clinical information
        diagnostic_yields = self._extract_diagnostic_yields(content)
        complications = self._extract_complications(content)
        methodology = self._extract_methodology(content)
        pearls = self._extract_clinical_pearls(content)
        
        # Extract practical information
        indications = self._extract_indications(content)
        contraindications = self._extract_contraindications(content)
        patient_selection = self._extract_patient_selection(content)
        equipment = self._extract_equipment(content)
        
        # Generate summaries
        key_findings = self._extract_key_findings(content, diagnostic_yields, complications)
        practical_tips = self._extract_practical_tips(content)
        
        # Calculate confidence
        has_numerical = bool(diagnostic_yields or complications)
        confidence = self._calculate_confidence(
            diagnostic_yields, complications, methodology, pearls
        )
        
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
        """Extract diagnostic yield information with specific values"""
        
        yields = {}
        
        # Pattern matching for yields
        yield_patterns = [
            r'diagnostic yield[^\d]*(\d+\.?\d*)\s*%',
            r'sensitivity[^\d]*(\d+\.?\d*)\s*%',
            r'specificity[^\d]*(\d+\.?\d*)\s*%',
            r'accuracy[^\d]*(\d+\.?\d*)\s*%'
        ]
        
        for pattern in yield_patterns:
            matches = re.finditer(pattern, content.lower())
            for match in matches:
                # Get context around the match
                start = max(0, match.start() - 100)
                end = min(len(content), match.end() + 100)
                context = content[start:end]
                
                # Identify procedure
                procedure = self._identify_procedure(context)
                
                # Store yield data
                if procedure not in yields:
                    yields[procedure] = {}
                
                metric = pattern.split('[')[0].strip()
                yields[procedure][metric] = {
                    'value': float(match.group(1)),
                    'context': context.strip()
                }
        
        return yields
    
    def _extract_complications(self, content: str) -> Dict[str, Any]:
        """Extract complication rates and management"""
        
        complications = {}
        
        # Common complications to look for
        complication_terms = [
            'pneumothorax', 'bleeding', 'hemorrhage', 'infection',
            'perforation', 'hypoxia', 'mortality', 'adverse event'
        ]
        
        for term in complication_terms:
            # Find mentions with rates
            pattern = f'{term}[^\d]*(\d+\.?\d*)\s*%'
            matches = re.finditer(pattern, content.lower())
            
            for match in matches:
                rate = float(match.group(1))
                
                # Get management info if available
                start = max(0, match.start() - 200)
                end = min(len(content), match.end() + 200)
                context = content[start:end]
                
                management = self._extract_management(context)
                
                complications[term] = {
                    'rate': rate,
                    'management': management,
                    'context': context.strip()
                }
        
        return complications
    
    def _extract_methodology(self, content: str) -> Dict[str, Any]:
        """Extract procedural methodology and techniques"""
        
        methodology = {}
        
        # Look for methodology sections
        method_keywords = ['technique', 'procedure', 'method', 'approach', 'protocol']
        
        for keyword in method_keywords:
            if keyword in content.lower():
                # Extract surrounding context
                pattern = f'{keyword}[^.]*\\.'
                matches = re.finditer(pattern, content.lower())
                
                for match in matches:
                    # Get extended context
                    start = max(0, match.start() - 300)
                    end = min(len(content), match.end() + 300)
                    context = content[start:end]
                    
                    # Extract specific details
                    procedure = self._identify_procedure(context)
                    if procedure:
                        if procedure not in methodology:
                            methodology[procedure] = {
                                'technique': [],
                                'equipment': [],
                                'positioning': None,
                                'sedation': None
                            }
                        
                        methodology[procedure]['technique'].append(context.strip())
        
        return methodology
    
    def _extract_clinical_pearls(self, content: str) -> List[str]:
        """Extract clinical pearls and learning points"""
        
        pearls = []
        
        # Keywords that indicate clinical pearls
        pearl_keywords = [
            'tip', 'trick', 'pearl', 'key point', 'important',
            'remember', 'note', 'pitfall', 'avoid', 'recommend'
        ]
        
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(keyword in sentence_lower for keyword in pearl_keywords):
                # Clean and add pearl
                pearl = sentence.strip()
                if 20 < len(pearl) < 500:  # Reasonable length
                    pearls.append(pearl)
        
        # Limit to most relevant pearls
        return pearls[:20]
    
    def _extract_indications(self, content: str) -> List[str]:
        """Extract indications for procedures"""
        
        indications = []
        
        # Find indications section
        if 'indication' in content.lower():
            # Get content after "indications"
            idx = content.lower().find('indication')
            section = content[idx:idx+1000]
            
            # Extract bullet points or numbered items
            lines = section.split('\n')
            for line in lines:
                if line.strip() and (line.strip()[0] in '•-123456789' or 'nodule' in line.lower() or 'mass' in line.lower()):
                    indications.append(line.strip())
        
        return indications[:10]
    
    def _extract_contraindications(self, content: str) -> List[str]:
        """Extract contraindications"""
        
        contraindications = []
        
        if 'contraindication' in content.lower():
            idx = content.lower().find('contraindication')
            section = content[idx:idx+1000]
            
            lines = section.split('\n')
            for line in lines:
                if line.strip() and (line.strip()[0] in '•-123456789' or 'coagulopathy' in line.lower()):
                    contraindications.append(line.strip())
        
        return contraindications[:10]
    
    def _extract_patient_selection(self, content: str) -> Dict[str, Any]:
        """Extract patient selection criteria"""
        
        selection = {
            'inclusion_criteria': [],
            'exclusion_criteria': [],
            'ideal_candidate': None
        }
        
        # Look for patient selection information
        if 'patient selection' in content.lower() or 'inclusion criteria' in content.lower():
            # Extract relevant sections
            patterns = {
                'inclusion': r'inclusion[^:]*:(.*?)(?:exclusion|$)',
                'exclusion': r'exclusion[^:]*:(.*?)(?:inclusion|$)'
            }
            
            for key, pattern in patterns.items():
                match = re.search(pattern, content.lower(), re.DOTALL)
                if match:
                    criteria = match.group(1)[:500]
                    selection[f'{key}_criteria'] = criteria.strip()
        
        return selection
    
    def _extract_equipment(self, content: str) -> List[str]:
        """Extract equipment and tools mentioned"""
        
        equipment = []
        
        equipment_keywords = [
            'needle', 'scope', 'catheter', 'guidewire', 'forceps',
            'bronchoscope', 'ultrasound', 'probe', 'stent'
        ]
        
        for keyword in equipment_keywords:
            if keyword in content.lower():
                # Get specific model/type if mentioned
                pattern = f'\\b[\\w\\s]*{keyword}[\\w\\s]*\\b'
                matches = re.finditer(pattern, content.lower())
                for match in matches:
                    equipment_item = match.group().strip()
                    if equipment_item not in equipment:
                        equipment.append(equipment_item)
        
        return equipment[:15]
    
    def _extract_key_findings(self, content: str, yields: Dict, complications: Dict) -> List[str]:
        """Extract key findings from the document"""
        
        findings = []
        
        # Add yield findings
        for procedure, yield_data in yields.items():
            for metric, data in yield_data.items():
                finding = f"{procedure.upper()} {metric}: {data['value']}%"
                findings.append(finding)
        
        # Add complication findings
        for complication, data in complications.items():
            finding = f"{complication.capitalize()} rate: {data['rate']}%"
            findings.append(finding)
        
        # Look for conclusion/summary statements
        if 'conclusion' in content.lower():
            idx = content.lower().find('conclusion')
            conclusion = content[idx:idx+500]
            sentences = conclusion.split('.')
            for sentence in sentences[:3]:
                if len(sentence.strip()) > 20:
                    findings.append(sentence.strip())
        
        return findings[:10]
    
    def _extract_practical_tips(self, content: str) -> List[str]:
        """Extract practical tips and recommendations"""
        
        tips = []
        
        tip_patterns = [
            r'recommend[^.]*\.',
            r'suggest[^.]*\.',
            r'should[^.]*\.',
            r'important to[^.]*\.'
        ]
        
        for pattern in tip_patterns:
            matches = re.finditer(pattern, content.lower())
            for match in matches:
                tip = content[match.start():match.end()].strip()
                if 20 < len(tip) < 300:
                    tips.append(tip)
        
        return tips[:10]
    
    def _identify_procedure(self, text: str) -> str:
        """Identify which procedure is being discussed"""
        
        text_lower = text.lower()
        for procedure in self.procedure_keywords:
            if procedure in text_lower:
                return procedure
        return 'general'
    
    def _extract_management(self, context: str) -> str:
        """Extract management strategy from context"""
        
        management_keywords = ['managed', 'treated', 'resolved', 'conservative', 'surgical']
        
        for keyword in management_keywords:
            if keyword in context.lower():
                # Get sentence containing keyword
                sentences = context.split('.')
                for sentence in sentences:
                    if keyword in sentence.lower():
                        return sentence.strip()
        
        return "Not specified"
    
    def _extract_title(self, content: str) -> str:
        """Extract document title"""
        
        lines = content.split('\n')
        for line in lines[:10]:
            if 10 < len(line) < 200:
                return line.strip()
        return "Unknown Title"
    
    def _generate_doc_id(self, content: str) -> str:
        """Generate unique document ID"""
        
        import hashlib
        return hashlib.md5(content[:1000].encode()).hexdigest()[:12]
    
    def _calculate_confidence(self, yields: Dict, complications: Dict, 
                            methodology: Dict, pearls: List) -> float:
        """Calculate extraction confidence score"""
        
        score = 0.5  # Base score
        
        if yields:
            score += 0.2
        if complications:
            score += 0.15
        if methodology:
            score += 0.1
        if len(pearls) > 5:
            score += 0.05
        
        return min(score, 1.0)
```

### 2.2 Create Practical Chunker
Create `ipchat/processing/practical_chunker.py`:

```python
"""
Practical document chunking for clinical information retrieval
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import re

@dataclass
class ClinicalChunk:
    """A chunk optimized for clinical information retrieval"""
    chunk_id: str
    content: str
    chunk_type: str  # 'diagnostic', 'complications', 'methodology', 'pearls', 'general'
    metadata: Dict[str, Any]
    source_document: str
    relevance_keywords: List[str]

class PracticalChunker:
    """Create chunks optimized for clinical Q&A"""
    
    def __init__(self, chunk_size: int = 800, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_document(self, 
                      document: Dict[str, Any],
                      extraction: 'ClinicalExtraction') -> List[ClinicalChunk]:
        """
        Create chunks from document with clinical metadata
        
        Args:
            document: Raw document dict with 'content' and 'id'
            extraction: ClinicalExtraction object with structured data
            
        Returns:
            List of ClinicalChunk objects
        """
        
        chunks = []
        doc_id = document['id']
        content = document['content']
        
        # Create specialized chunks for different information types
        
        # 1. Diagnostic yield chunks
        if extraction.diagnostic_yields:
            for procedure, yields in extraction.diagnostic_yields.items():
                chunk_content = f"Diagnostic Yields for {procedure.upper()}:\n"
                for metric, data in yields.items():
                    chunk_content += f"- {metric}: {data['value']}%\n"
                    chunk_content += f"  Context: {data.get('context', '')[:200]}\n"
                
                chunks.append(ClinicalChunk(
                    chunk_id=f"{doc_id}_yield_{procedure}",
                    content=chunk_content,
                    chunk_type='diagnostic',
                    metadata={
                        'procedure': procedure,
                        'has_numerical_data': True,
                        'extraction_confidence': extraction.extraction_confidence
                    },
                    source_document=doc_id,
                    relevance_keywords=[procedure, 'yield', 'diagnostic', 'sensitivity', 'specificity']
                ))
        
        # 2. Complication chunks
        if extraction.complication_rates:
            chunk_content = "Complications and Management:\n"
            for complication, data in extraction.complication_rates.items():
                chunk_content += f"- {complication}: {data['rate']}%\n"
                chunk_content += f"  Management: {data.get('management', 'Not specified')}\n"
            
            chunks.append(ClinicalChunk(
                chunk_id=f"{doc_id}_complications",
                content=chunk_content,
                chunk_type='complications',
                metadata={
                    'has_rates': True,
                    'complications_listed': list(extraction.complication_rates.keys())
                },
                source_document=doc_id,
                relevance_keywords=['complication', 'adverse', 'safety', 'management']
            ))
        
        # 3. Methodology chunks
        if extraction.methodology:
            for procedure, method_data in extraction.methodology.items():
                chunk_content = f"Methodology for {procedure.upper()}:\n"
                if method_data.get('technique'):
                    chunk_content += "Technique:\n"
                    for technique in method_data['technique'][:3]:
                        chunk_content += f"- {technique[:200]}\n"
                
                chunks.append(ClinicalChunk(
                    chunk_id=f"{doc_id}_method_{procedure}",
                    content=chunk_content,
                    chunk_type='methodology',
                    metadata={
                        'procedure': procedure,
                        'has_technique': bool(method_data.get('technique'))
                    },
                    source_document=doc_id,
                    relevance_keywords=[procedure, 'technique', 'method', 'approach', 'protocol']
                ))
        
        # 4. Clinical pearls chunk
        if extraction.clinical_pearls:
            chunk_content = "Clinical Pearls and Tips:\n"
            for i, pearl in enumerate(extraction.clinical_pearls[:10], 1):
                chunk_content += f"{i}. {pearl}\n"
            
            chunks.append(ClinicalChunk(
                chunk_id=f"{doc_id}_pearls",
                content=chunk_content,
                chunk_type='pearls',
                metadata={
                    'pearl_count': len(extraction.clinical_pearls)
                },
                source_document=doc_id,
                relevance_keywords=['tip', 'pearl', 'trick', 'recommendation', 'learning']
            ))
        
        # 5. Practical information chunk
        if extraction.indications or extraction.contraindications:
            chunk_content = ""
            
            if extraction.indications:
                chunk_content += "Indications:\n"
                for indication in extraction.indications[:5]:
                    chunk_content += f"- {indication}\n"
            
            if extraction.contraindications:
                chunk_content += "\nContraindications:\n"
                for contra in extraction.contraindications[:5]:
                    chunk_content += f"- {contra}\n"
            
            chunks.append(ClinicalChunk(
                chunk_id=f"{doc_id}_criteria",
                content=chunk_content,
                chunk_type='general',
                metadata={
                    'has_indications': bool(extraction.indications),
                    'has_contraindications': bool(extraction.contraindications)
                },
                source_document=doc_id,
                relevance_keywords=['indication', 'contraindication', 'patient', 'selection', 'criteria']
            ))
        
        # 6. Key findings summary chunk
        if extraction.key_findings:
            chunk_content = f"Key Findings from {extraction.title}:\n"
            for finding in extraction.key_findings:
                chunk_content += f"• {finding}\n"
            
            chunks.append(ClinicalChunk(
                chunk_id=f"{doc_id}_summary",
                content=chunk_content,
                chunk_type='general',
                metadata={
                    'is_summary': True,
                    'document_type': extraction.document_type
                },
                source_document=doc_id,
                relevance_keywords=['summary', 'findings', 'conclusion', 'results']
            ))
        
        # 7. Standard text chunks for remaining content
        text_chunks = self._create_standard_chunks(content, doc_id)
        chunks.extend(text_chunks)
        
        return chunks
    
    def _create_standard_chunks(self, content: str, doc_id: str) -> List[ClinicalChunk]:
        """Create standard overlapping chunks for general content"""
        
        chunks = []
        sentences = content.split('.')
        
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < self.chunk_size:
                current_chunk += sentence + "."
            else:
                # Save current chunk
                if current_chunk:
                    chunks.append(ClinicalChunk(
                        chunk_id=f"{doc_id}_text_{chunk_index}",
                        content=current_chunk,
                        chunk_type='general',
                        metadata={'chunk_index': chunk_index},
                        source_document=doc_id,
                        relevance_keywords=self._extract_keywords(current_chunk)
                    ))
                    chunk_index += 1
                
                # Start new chunk with overlap
                sentences_in_chunk = current_chunk.split('.')
                overlap_sentences = sentences_in_chunk[-2:] if len(sentences_in_chunk) > 2 else sentences_in_chunk
                current_chunk = '. '.join(overlap_sentences) + ". " + sentence + "."
        
        # Add final chunk
        if current_chunk:
            chunks.append(ClinicalChunk(
                chunk_id=f"{doc_id}_text_{chunk_index}",
                content=current_chunk,
                chunk_type='general',
                metadata={'chunk_index': chunk_index},
                source_document=doc_id,
                relevance_keywords=self._extract_keywords(current_chunk)
            ))
        
        return chunks
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract relevant keywords from text"""
        
        keywords = []
        
        # Clinical procedure keywords
        procedures = ['ebus', 'bronchoscopy', 'tbna', 'bal', 'cryobiopsy']
        for proc in procedures:
            if proc in text.lower():
                keywords.append(proc)
        
        # Clinical concept keywords
        concepts = ['diagnostic', 'complication', 'technique', 'indication', 'contraindication']
        for concept in concepts:
            if concept in text.lower():
                keywords.append(concept)
        
        return keywords[:10]
```

## Phase 3: Practical Storage and Retrieval

### 3.1 Create Clinical Index Manager
Create `ipchat/retrieval/clinical_index.py`:

```python
"""
Index management for clinical information retrieval
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import pickle
import json
from pathlib import Path
from typing import List, Dict, Any
import sqlite3

class ClinicalIndexManager:
    """Manage indices for clinical information retrieval"""
    
    def __init__(self, index_dir: Path = Path("data/indices")):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize embedder (using lightweight model)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Initialize indices
        self.vector_index = None
        self.bm25_index = None
        self.chunk_metadata = []
        
        # SQLite for structured queries
        self.db_path = self.index_dir / "clinical_data.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for structured clinical data"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Table for diagnostic yields
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS diagnostic_yields (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                procedure TEXT,
                metric TEXT,
                value REAL,
                context TEXT
            )
        """)
        
        # Table for complications
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS complications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                complication_type TEXT,
                rate REAL,
                management TEXT
            )
        """)
        
        # Table for clinical pearls
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS clinical_pearls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id TEXT,
                pearl TEXT,
                category TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def add_clinical_extraction(self, extraction: 'ClinicalExtraction'):
        """Add structured clinical data to database"""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add diagnostic yields
        for procedure, yields in extraction.diagnostic_yields.items():
            for metric, data in yields.items():
                cursor.execute("""
                    INSERT INTO diagnostic_yields (document_id, procedure, metric, value, context)
                    VALUES (?, ?, ?, ?, ?)
                """, (extraction.document_id, procedure, metric, data['value'], data.get('context', '')))
        
        # Add complications
        for comp_type, data in extraction.complication_rates.items():
            cursor.execute("""
                INSERT INTO complications (document_id, complication_type, rate, management)
                VALUES (?, ?, ?, ?)
            """, (extraction.document_id, comp_type, data['rate'], data.get('management', '')))
        
        # Add clinical pearls
        for pearl in extraction.clinical_pearls:
            cursor.execute("""
                INSERT INTO clinical_pearls (document_id, pearl, category)
                VALUES (?, ?, ?)
            """, (extraction.document_id, pearl, 'general'))
        
        conn.commit()
        conn.close()
    
    def create_indices(self, chunks: List['ClinicalChunk']):
        """Create vector and BM25 indices from chunks"""
        
        # Prepare texts
        texts = [chunk.content for chunk in chunks]
        
        # Create vector embeddings
        embeddings = self.embedder.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.vector_index = faiss.IndexFlatL2(dimension)
        self.vector_index.add(embeddings.astype('float32'))
        
        # Create BM25 index
        tokenized_texts = [text.lower().split() for text in texts]
        self.bm25_index = BM25Okapi(tokenized_texts)
        
        # Store metadata
        self.chunk_metadata = [
            {
                'chunk_id': chunk.chunk_id,
                'chunk_type': chunk.chunk_type,
                'source_document': chunk.source_document,
                'metadata': chunk.metadata,
                'relevance_keywords': chunk.relevance_keywords
            }
            for chunk in chunks
        ]
        
        # Save indices
        self._save_indices()
    
    def search(self, query: str, k: int = 10, search_type: str = 'hybrid') -> List[Dict[str, Any]]:
        """
        Search for relevant chunks
        
        Args:
            query: Search query
            k: Number of results
            search_type: 'vector', 'keyword', or 'hybrid'
            
        Returns:
            List of relevant chunks with metadata
        """
        
        results = []
        
        if search_type in ['vector', 'hybrid']:
            # Vector search
            query_embedding = self.embedder.encode([query])
            distances, indices = self.vector_index.search(query_embedding.astype('float32'), k)
            
            for idx, distance in zip(indices[0], distances[0]):
                if idx < len(self.chunk_metadata):
                    result = self.chunk_metadata[idx].copy()
                    result['score'] = float(1 / (1 + distance))
                    result['search_type'] = 'vector'
                    results.append(result)
        
        if search_type in ['keyword', 'hybrid']:
            # BM25 search
            tokenized_query = query.lower().split()
            bm25_scores = self.bm25_index.get_scores(tokenized_query)
            
            # Get top k indices
            top_indices = np.argsort(bm25_scores)[-k:][::-1]
            
            for idx in top_indices:
                if idx < len(self.chunk_metadata):
                    result = self.chunk_metadata[idx].copy()
                    result['score'] = float(bm25_scores[idx])
                    result['search_type'] = 'keyword'
                    results.append(result)
        
        # Merge and deduplicate for hybrid search
        if search_type == 'hybrid':
            # Deduplicate by chunk_id
            seen = set()
            deduplicated = []
            for result in sorted(results, key=lambda x: x['score'], reverse=True):
                if result['chunk_id'] not in seen:
                    seen.add(result['chunk_id'])
                    deduplicated.append(result)
            results = deduplicated[:k]
        
        return results
    
    def query_clinical_data(self, query_type: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Query structured clinical data from database
        
        Args:
            query_type: 'yields', 'complications', 'pearls'
            **kwargs: Query parameters
            
        Returns:
            List of results
        """
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        results = []
        
        if query_type == 'yields':
            procedure = kwargs.get('procedure', '%')
            cursor.execute("""
                SELECT * FROM diagnostic_yields
                WHERE procedure LIKE ?
                ORDER BY value DESC
            """, (f'%{procedure}%',))
            
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
        
        elif query_type == 'complications':
            comp_type = kwargs.get('complication_type', '%')
            cursor.execute("""
                SELECT * FROM complications
                WHERE complication_type LIKE ?
                ORDER BY rate DESC
            """, (f'%{comp_type}%',))
            
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
        
        elif query_type == 'pearls':
            keyword = kwargs.get('keyword', '%')
            cursor.execute("""
                SELECT * FROM clinical_pearls
                WHERE pearl LIKE ?
                LIMIT 20
            """, (f'%{keyword}%',))
            
            columns = [desc[0] for desc in cursor.description]
            for row in cursor.fetchall():
                results.append(dict(zip(columns, row)))
        
        conn.close()
        return results
    
    def _save_indices(self):
        """Save indices to disk"""
        
        # Save FAISS index
        faiss_path = self.index_dir / "clinical.faiss"
        faiss.write_index(self.vector_index, str(faiss_path))
        
        # Save BM25 index
        bm25_path = self.index_dir / "clinical_bm25.pkl"
        with open(bm25_path, 'wb') as f:
            pickle.dump(self.bm25_index, f)
        
        # Save metadata
        metadata_path = self.index_dir / "chunk_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.chunk_metadata, f)
    
    def load_indices(self):
        """Load indices from disk"""
        
        # Load FAISS index
        faiss_path = self.index_dir / "clinical.faiss"
        if faiss_path.exists():
            self.vector_index = faiss.read_index(str(faiss_path))
        
        # Load BM25 index
        bm25_path = self.index_dir / "clinical_bm25.pkl"
        if bm25_path.exists():
            with open(bm25_path, 'rb') as f:
                self.bm25_index = pickle.load(f)
        
        # Load metadata
        metadata_path = self.index_dir / "chunk_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.chunk_metadata = json.load(f)
```

## Phase 4: Complete Processing Pipeline

### 4.1 Create Practical Pipeline
Create `ipchat/pipeline/practical_pipeline.py`:

```python
"""
Practical clinical information processing pipeline
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from ipchat.extraction.clinical_extractor import ClinicalDataExtractor, ClinicalExtraction
from ipchat.processing.practical_chunker import PracticalChunker
from ipchat.retrieval.clinical_index import ClinicalIndexManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PracticalClinicalPipeline:
    """Complete pipeline for clinical information extraction and retrieval"""
    
    def __init__(self, data_dir: Path = Path("data")):
        self.data_dir = Path(data_dir)
        self.extractor = ClinicalDataExtractor()
        self.chunker = PracticalChunker()
        self.index_manager = ClinicalIndexManager()
        
        # Setup directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        
        dirs = [
            self.data_dir / "raw" / "research",
            self.data_dir / "raw" / "textbooks",
            self.data_dir / "extracted",
            self.data_dir / "chunks",
            self.data_dir / "indices"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def process_document(self, 
                        pdf_path: str,
                        document_type: str = "research") -> Dict[str, Any]:
        """
        Process a single document through the pipeline
        
        Args:
            pdf_path: Path to PDF file
            document_type: 'research' or 'textbook'
            
        Returns:
            Processing results
        """
        
        logger.info(f"Processing document: {pdf_path}")
        
        # Extract text from PDF (placeholder - implement with pdfplumber)
        text_content = self._extract_pdf_text(pdf_path)
        
        # Extract clinical information
        extraction = self.extractor.extract(text_content, document_type)
        
        # Save extraction
        extraction_path = self.data_dir / "extracted" / f"{extraction.document_id}.json"
        with open(extraction_path, 'w') as f:
            json.dump(asdict(extraction), f, indent=2)
        
        # Add to database
        self.index_manager.add_clinical_extraction(extraction)
        
        # Create chunks
        document = {
            'id': extraction.document_id,
            'content': text_content
        }
        chunks = self.chunker.chunk_document(document, extraction)
        
        # Save chunks
        chunks_path = self.data_dir / "chunks" / f"{extraction.document_id}.json"
        with open(chunks_path, 'w') as f:
            json.dump([chunk.__dict__ for chunk in chunks], f, indent=2)
        
        # Update indices
        self.index_manager.create_indices(chunks)
        
        logger.info(f"Document processed: {extraction.document_id}")
        
        return {
            'document_id': extraction.document_id,
            'title': extraction.title,
            'extraction_confidence': extraction.extraction_confidence,
            'chunks_created': len(chunks),
            'has_numerical_data': extraction.has_numerical_data,
            'diagnostic_yields': len(extraction.diagnostic_yields),
            'complications': len(extraction.complication_rates),
            'clinical_pearls': len(extraction.clinical_pearls)
        }
    
    def query(self, 
             query_text: str,
             query_type: str = "general") -> Dict[str, Any]:
        """
        Query the clinical information system
        
        Args:
            query_text: User query
            query_type: 'general', 'yields', 'complications', 'pearls'
            
        Returns:
            Query results
        """
        
        results = {}
        
        if query_type == "general":
            # Hybrid search through chunks
            chunks = self.index_manager.search(query_text, k=10, search_type='hybrid')
            results['chunks'] = chunks
            
        elif query_type == "yields":
            # Query diagnostic yields
            yields = self.index_manager.query_clinical_data(
                'yields',
                procedure=query_text
            )
            results['yields'] = yields
            
        elif query_type == "complications":
            # Query complications
            complications = self.index_manager.query_clinical_data(
                'complications',
                complication_type=query_text
            )
            results['complications'] = complications
            
        elif query_type == "pearls":
            # Query clinical pearls
            pearls = self.index_manager.query_clinical_data(
                'pearls',
                keyword=query_text
            )
            results['pearls'] = pearls
        
        return results
    
    def _extract_pdf_text(self, pdf_path: str) -> str:
        """Extract text from PDF (placeholder)"""
        
        # Implement with pdfplumber or PyPDF2
        # For now, return placeholder
        return f"[PDF content from {pdf_path}]"
    
    def batch_process(self, 
                     pdf_directory: str,
                     document_type: str = "research",
                     max_documents: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Process multiple documents
        
        Args:
            pdf_directory: Directory containing PDFs
            document_type: 'research' or 'textbook'
            max_documents: Maximum number to process
            
        Returns:
            List of processing results
        """
        
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if max_documents:
            pdf_files = pdf_files[:max_documents]
        
        results = []
        
        for i, pdf_file in enumerate(pdf_files, 1):
            logger.info(f"Processing {i}/{len(pdf_files)}: {pdf_file.name}")
            
            try:
                result = self.process_document(str(pdf_file), document_type)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                results.append({
                    'document_id': None,
                    'title': pdf_file.name,
                    'error': str(e)
                })
        
        # Save batch summary
        summary_path = self.data_dir / "batch_summary.json"
        with open(summary_path, 'w') as f:
            json.dump({
                'total_processed': len(results),
                'successful': sum(1 for r in results if 'error' not in r),
                'failed': sum(1 for r in results if 'error' in r),
                'results': results
            }, f, indent=2)
        
        logger.info(f"Batch processing complete: {len(results)} documents")
        
        return results
```

## Phase 5: Migration Script

Create `tools/scripts/migrate_to_practical.py`:

```python
#!/usr/bin/env python3
"""
Migration script to practical clinical extraction system
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from ipchat.pipeline.practical_pipeline import PracticalClinicalPipeline
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Migrate to practical clinical extraction')
    parser.add_argument('--research-dir', default='Studies', 
                       help='Directory with research papers')
    parser.add_argument('--textbook-dir', default='Textbooks/Chapter pdfs',
                       help='Directory with textbook chapters')
    parser.add_argument('--max-docs', type=int, default=5,
                       help='Maximum documents to process (for testing)')
    parser.add_argument('--full-run', action='store_true',
                       help='Process all documents')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = PracticalClinicalPipeline()
    
    # Determine document limit
    max_docs = None if args.full_run else args.max_docs
    
    # Process research papers
    if Path(args.research_dir).exists():
        logger.info(f"Processing research papers from {args.research_dir}")
        research_results = pipeline.batch_process(
            args.research_dir,
            document_type='research',
            max_documents=max_docs
        )
        logger.info(f"Processed {len(research_results)} research papers")
    
    # Process textbook chapters
    if Path(args.textbook_dir).exists():
        logger.info(f"Processing textbook chapters from {args.textbook_dir}")
        textbook_results = pipeline.batch_process(
            args.textbook_dir,
            document_type='textbook',
            max_documents=max_docs
        )
        logger.info(f"Processed {len(textbook_results)} textbook chapters")
    
    logger.info("Migration complete!")
    
    # Test queries
    logger.info("\nTesting queries...")
    
    test_queries = [
        ("What is the diagnostic yield of EBUS?", "yields"),
        ("Pneumothorax complications", "complications"),
        ("Clinical tips for bronchoscopy", "pearls")
    ]
    
    for query_text, query_type in test_queries:
        results = pipeline.query(query_text, query_type)
        logger.info(f"Query: {query_text}")
        logger.info(f"Results: {len(results.get(query_type, []))} items found")

if __name__ == "__main__":
    main()
```

## Summary

This practical refactor provides:

1. **Clinical Focus**: Extracts diagnostic yields, complication rates, methodologies, and clinical pearls
2. **Structured Storage**: SQLite for numerical data, FAISS for semantic search
3. **Practical Chunking**: Specialized chunks for different clinical information types
4. **Hybrid Search**: Combines vector and keyword search for best results
5. **Simple Architecture**: Easy to maintain and extend

The system is optimized for your ~1000 documents and provides granular clinical information without the complexity of full meta-analysis capabilities.
