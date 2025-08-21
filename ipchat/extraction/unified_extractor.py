"""
Simplified unified extractor for both research articles and textbooks.
Focuses on extracting only what's needed for RAG retrieval.
"""

import json
from typing import Dict, Any, Optional, List
from pathlib import Path
import openai
from dataclasses import dataclass, asdict

@dataclass
class ExtractedDocument:
    """Simplified extracted document structure"""
    document_id: str
    title: str
    document_type: str  # 'research' or 'textbook'
    
    # For research articles
    population: Optional[str] = None
    intervention: Optional[str] = None
    comparator: Optional[str] = None
    outcomes: Optional[Dict[str, Any]] = None
    key_findings: Optional[List[str]] = None
    
    # For textbook chapters
    procedures: Optional[List[Dict[str, str]]] = None
    indications: Optional[List[str]] = None
    contraindications: Optional[List[str]] = None
    algorithms: Optional[List[Dict[str, str]]] = None
    
    # Common fields
    summary: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    source_pages: Optional[List[int]] = None

class UnifiedExtractor:
    """Single extractor for all document types"""
    
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI()
    
    def extract(self, 
                content: str, 
                document_type: str,
                document_metadata: Dict[str, Any] = None) -> ExtractedDocument:
        """
        Extract key information from document content.
        
        Args:
            content: Document text content
            document_type: 'research' or 'textbook'
            document_metadata: Additional metadata about the document
            
        Returns:
            ExtractedDocument with relevant fields populated
        """
        prompt = self._get_prompt(document_type)
        
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Extract information from this document:\n\n{content[:15000]}"}  # Limit context
            ],
            response_format={"type": "json_object"}
        )
        
        extracted_data = json.loads(response.choices[0].message.content)
        
        # Create ExtractedDocument instance
        doc = ExtractedDocument(
            document_id=document_metadata.get('id', 'unknown'),
            title=document_metadata.get('title', extracted_data.get('title', 'Unknown')),
            document_type=document_type,
            **{k: v for k, v in extracted_data.items() if k in ExtractedDocument.__annotations__}
        )
        
        return doc
    
    def _get_prompt(self, document_type: str) -> str:
        """Get extraction prompt based on document type"""
        if document_type == 'research':
            return self._research_prompt()
        elif document_type == 'textbook':
            return self._textbook_prompt()
        else:
            raise ValueError(f"Unknown document type: {document_type}")
    
    def _research_prompt(self) -> str:
        return """You are a medical information extractor for interventional pulmonology research.
        
Extract ONLY the following information if explicitly present. Return NULL for missing fields.

Return a JSON object with these fields:
- population: Patient population studied (string or null)
- intervention: Primary intervention or procedure (string or null)
- comparator: Comparison group or control (string or null)
- outcomes: Key measured outcomes with values (object or null)
- key_findings: List of 3-5 most important findings (array of strings or null)
- summary: 2-3 sentence summary of the study (string)

Do not invent or infer information. Only extract what is explicitly stated."""

    def _textbook_prompt(self) -> str:
        return """You are a medical information extractor for interventional pulmonology textbooks.
        
Extract ONLY the following information if explicitly present. Return NULL for missing fields.

Return a JSON object with these fields:
- procedures: Array of procedures with name and brief description (array or null)
- indications: List of clinical indications (array of strings or null)
- contraindications: List of contraindications (array of strings or null)
- algorithms: Diagnostic or treatment algorithms with name and steps (array or null)
- summary: 2-3 sentence chapter summary (string)

Do not invent information. Only extract what is explicitly stated."""

    def batch_extract(self, documents: List[Dict[str, Any]], 
                     output_dir: Path) -> List[ExtractedDocument]:
        """Process multiple documents"""
        results = []
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for doc in documents:
            try:
                extracted = self.extract(
                    content=doc['content'],
                    document_type=doc['type'],
                    document_metadata=doc.get('metadata', {})
                )
                
                # Save individual result
                output_file = output_dir / f"{extracted.document_id}.json"
                with open(output_file, 'w') as f:
                    json.dump(asdict(extracted), f, indent=2)
                
                results.append(extracted)
                print(f"✓ Extracted: {extracted.title[:50]}...")
                
            except Exception as e:
                print(f"✗ Failed to extract {doc.get('metadata', {}).get('title', 'Unknown')}: {e}")
        
        return results