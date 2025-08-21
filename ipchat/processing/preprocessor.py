"""
Document preprocessing utilities.
"""

import re
from typing import Dict, Any, Optional

class DocumentPreprocessor:
    """Preprocess documents before extraction and chunking"""
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and normalize text.
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might break processing
        text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Remove excessive newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def extract_metadata(text: str) -> Dict[str, Any]:
        """
        Extract metadata from document text.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Try to extract title (first line or heading)
        lines = text.split('\n')
        if lines:
            potential_title = lines[0].strip()
            if len(potential_title) < 200:
                metadata['title'] = potential_title
        
        # Extract DOI if present
        doi_pattern = r'10\.\d{4,}(?:\.\d+)*\/[-._;()\/:a-zA-Z0-9]+'
        doi_match = re.search(doi_pattern, text)
        if doi_match:
            metadata['doi'] = doi_match.group()
        
        # Extract year if present
        year_pattern = r'\b(19|20)\d{2}\b'
        year_matches = re.findall(year_pattern, text[:1000])  # Look in first 1000 chars
        if year_matches:
            metadata['year'] = year_matches[0]
        
        # Detect document type heuristically
        research_keywords = ['abstract', 'methods', 'results', 'conclusion', 'participants', 'study']
        textbook_keywords = ['chapter', 'section', 'introduction', 'summary', 'key points', 'learning objectives']
        
        text_lower = text[:5000].lower()  # Check first 5000 chars
        
        research_score = sum(1 for kw in research_keywords if kw in text_lower)
        textbook_score = sum(1 for kw in textbook_keywords if kw in text_lower)
        
        if research_score > textbook_score:
            metadata['detected_type'] = 'research'
        elif textbook_score > research_score:
            metadata['detected_type'] = 'textbook'
        else:
            metadata['detected_type'] = 'unknown'
        
        return metadata
    
    @staticmethod
    def segment_document(text: str) -> Dict[str, str]:
        """
        Segment document into sections.
        
        Args:
            text: Document text
            
        Returns:
            Dictionary of section_name -> section_content
        """
        sections = {}
        
        # Common section headers
        section_patterns = [
            r'(?i)^abstract[:\s]*',
            r'(?i)^introduction[:\s]*',
            r'(?i)^methods?[:\s]*',
            r'(?i)^results?[:\s]*',
            r'(?i)^discussion[:\s]*',
            r'(?i)^conclusion[:\s]*',
            r'(?i)^references?[:\s]*',
        ]
        
        # Split by section headers
        current_section = 'introduction'
        current_content = []
        
        for line in text.split('\n'):
            # Check if line matches a section header
            for pattern in section_patterns:
                if re.match(pattern, line.strip()):
                    # Save previous section
                    if current_content:
                        sections[current_section] = '\n'.join(current_content)
                    
                    # Start new section
                    current_section = re.sub(pattern, '', line.strip()).lower()
                    if not current_section:
                        current_section = line.strip().lower().replace(':', '')
                    current_content = []
                    break
            else:
                current_content.append(line)
        
        # Save last section
        if current_content:
            sections[current_section] = '\n'.join(current_content)
        
        # If no sections found, return whole document
        if not sections:
            sections['full_text'] = text
        
        return sections