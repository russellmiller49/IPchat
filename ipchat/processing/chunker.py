"""
Intelligent chunking system for documents.
Uses semantic boundaries and adaptive sizing.
"""

from typing import List, Dict, Any
import nltk
from dataclasses import dataclass
import hashlib

@dataclass
class Chunk:
    """Represents a document chunk"""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict[str, Any]
    token_count: int
    chunk_index: int
    total_chunks: int

class SemanticChunker:
    """Semantic-aware document chunker"""
    
    def __init__(self, 
                 target_chunk_size: int = 400,
                 overlap_size: int = 50,
                 min_chunk_size: int = 100,
                 max_chunk_size: int = 600):
        self.target_chunk_size = target_chunk_size
        self.overlap_size = overlap_size
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        
        # Download required NLTK data
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def chunk_document(self, 
                      document: Dict[str, Any],
                      extracted_data: Dict[str, Any] = None) -> List[Chunk]:
        """
        Chunk a document intelligently based on semantic boundaries.
        
        Args:
            document: Document with 'content' and 'metadata'
            extracted_data: Extracted structured data to enhance chunks
            
        Returns:
            List of Chunk objects
        """
        content = document['content']
        doc_id = document.get('id', self._generate_id(content))
        
        # Split into paragraphs first
        paragraphs = self._split_paragraphs(content)
        
        # Create semantic chunks
        chunks = []
        current_chunk = []
        current_tokens = 0
        
        for para in paragraphs:
            para_tokens = self._count_tokens(para)
            
            if current_tokens + para_tokens > self.max_chunk_size and current_chunk:
                # Save current chunk and start new one
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Add overlap
                if self.overlap_size > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]
                    current_tokens = self._count_tokens(current_chunk[0])
                else:
                    current_chunk = []
                    current_tokens = 0
            
            current_chunk.append(para)
            current_tokens += para_tokens
            
            # Check if we've reached target size
            if current_tokens >= self.target_chunk_size:
                chunk_text = '\n\n'.join(current_chunk)
                chunks.append(chunk_text)
                
                # Overlap for next chunk
                if self.overlap_size > 0 and len(current_chunk) > 1:
                    current_chunk = current_chunk[-1:]
                    current_tokens = self._count_tokens(current_chunk[0])
                else:
                    current_chunk = []
                    current_tokens = 0
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append('\n\n'.join(current_chunk))
        
        # Create Chunk objects with metadata
        chunk_objects = []
        for i, chunk_text in enumerate(chunks):
            chunk_id = f"{doc_id}_chunk_{i}"
            
            # Add extracted data as metadata if available
            chunk_metadata = document.get('metadata', {}).copy()
            if extracted_data:
                chunk_metadata['extracted_summary'] = extracted_data.get('summary')
                chunk_metadata['document_type'] = extracted_data.get('document_type')
            
            chunk_obj = Chunk(
                chunk_id=chunk_id,
                document_id=doc_id,
                content=chunk_text,
                metadata=chunk_metadata,
                token_count=self._count_tokens(chunk_text),
                chunk_index=i,
                total_chunks=len(chunks)
            )
            chunk_objects.append(chunk_obj)
        
        return chunk_objects
    
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        # Split on double newlines or common section markers
        paragraphs = text.split('\n\n')
        
        # Further split very long paragraphs
        result = []
        for para in paragraphs:
            if self._count_tokens(para) > self.max_chunk_size:
                # Split by sentences
                sentences = nltk.sent_tokenize(para)
                current = []
                current_tokens = 0
                
                for sent in sentences:
                    sent_tokens = self._count_tokens(sent)
                    if current_tokens + sent_tokens > self.target_chunk_size and current:
                        result.append(' '.join(current))
                        current = []
                        current_tokens = 0
                    current.append(sent)
                    current_tokens += sent_tokens
                
                if current:
                    result.append(' '.join(current))
            else:
                result.append(para)
        
        return [p.strip() for p in result if p.strip()]
    
    def _count_tokens(self, text: str) -> int:
        """Approximate token count (rough estimate)"""
        # Rough approximation: ~4 characters per token
        return len(text) // 4
    
    def _generate_id(self, content: str) -> str:
        """Generate document ID from content hash"""
        return hashlib.md5(content.encode()).hexdigest()[:8]

class HierarchicalChunker(SemanticChunker):
    """Enhanced chunker with hierarchical structure preservation"""
    
    def chunk_with_hierarchy(self, 
                            document: Dict[str, Any],
                            extracted_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Create chunks while preserving document hierarchy.
        Returns both chunks and a hierarchy map.
        """
        chunks = self.chunk_document(document, extracted_data)
        
        # Create hierarchy map
        hierarchy = {
            'document_id': document.get('id'),
            'title': document.get('title'),
            'total_chunks': len(chunks),
            'sections': self._extract_sections(document['content']),
            'chunk_map': {
                chunk.chunk_id: {
                    'index': chunk.chunk_index,
                    'tokens': chunk.token_count,
                    'preview': chunk.content[:100] + '...'
                }
                for chunk in chunks
            }
        }
        
        return {
            'chunks': chunks,
            'hierarchy': hierarchy
        }
    
    def _extract_sections(self, content: str) -> List[Dict[str, Any]]:
        """Extract section headers from content"""
        # Simple heuristic: lines that are likely headers
        lines = content.split('\n')
        sections = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            if (line and 
                len(line) < 100 and 
                (line.isupper() or 
                 line.endswith(':') or
                 any(line.startswith(p) for p in ['#', '##', '###']))):
                sections.append({
                    'title': line.replace('#', '').strip(),
                    'line_number': i
                })
        
        return sections