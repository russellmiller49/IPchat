
from __future__ import annotations
from typing import List, Optional, Union, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime

SCHEMA_VERSION = "1.0.0"

class Citation(BaseModel):
    ref_id: Optional[str] = None
    locator: Optional[str] = None

class ProvenanceSpan(BaseModel):
    page: Union[int, str, None] = None
    paragraph_index: Optional[int] = None
    char_start: Optional[int] = None
    char_end: Optional[int] = None

class ProvenanceLabel(BaseModel):
    page: Union[int, str, None] = None
    label: Optional[str] = None

class SectionNode(BaseModel):
    id: str
    title: str
    level: int
    page_start: Union[int, str]
    page_end: Union[int, str]
    anchor: Optional[str] = None
    hash: Optional[str] = None
    children: List["SectionNode"] = Field(default_factory=list)

SectionNode.model_rebuild()

class TextUnit(BaseModel):
    id: str
    unit_type: str = Field(description="paragraph | list_item | caption | note | quote | definition")
    section_id: str
    order: int
    text: str
    tokens: Optional[int] = None
    page_range: List[Union[int, str]] = Field(default_factory=list)
    bbox: Optional[List[float]] = None
    citations: List[Citation] = Field(default_factory=list)
    provenance: ProvenanceSpan

class TableColumn(BaseModel):
    id: str
    name: str
    unit: Optional[str] = None

class TableRow(BaseModel):
    row_header: Optional[str] = None
    cells: List[Union[str, float, int, None]]

class Figure(BaseModel):
    id: str
    number: Union[int, str]
    title: Optional[str] = None
    caption: Optional[str] = None
    file_ref: Optional[str] = None
    page: Union[int, str]
    bbox: Optional[List[float]] = None
    alt_text: Optional[str] = None
    ocr_text: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    data_extracted: Dict[str, Any] = Field(default_factory=dict)
    provenance: Optional[ProvenanceLabel] = None

class Table(BaseModel):
    id: str
    number: Union[int, str]
    title: Optional[str] = None
    caption: Optional[str] = None
    page: Union[int, str]
    bbox: Optional[List[float]] = None
    type: Optional[str] = Field(default=None, description="demographics | outcomes | labs | drugs | other")
    columns: List[TableColumn]
    rows: List[TableRow]
    footnotes: List[str] = Field(default_factory=list)
    units: Optional[str] = None
    normalized: bool = False
    provenance: Optional[ProvenanceLabel] = None

class Box(BaseModel):
    id: str
    box_type: str = Field(description="clinical_pearl | pitfall | key_points | algorithm | checklist | summary")
    title: Optional[str] = None
    items: List[str]
    page: Union[int, str]
    bbox: Optional[List[float]] = None
    provenance: Optional[ProvenanceLabel] = None

class EquationVar(BaseModel):
    symbol: Optional[str] = None
    name: Optional[str] = None
    units: Optional[str] = None
    description: Optional[str] = None

class Equation(BaseModel):
    id: str
    latex: str
    display: str = Field(description="inline | block")
    variables: List[EquationVar] = Field(default_factory=list)
    page: Union[int, str]
    bbox: Optional[List[float]] = None

class Case(BaseModel):
    id: str
    title: str
    vignette: str
    questions: List[Dict[str, Any]] = Field(default_factory=list)
    answers: List[str] = Field(default_factory=list)
    image_refs: List[str] = Field(default_factory=list)
    takeaways: List[str] = Field(default_factory=list)

class Reference(BaseModel):
    id: str
    citation_text: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    isbn: Optional[str] = None
    url: Optional[str] = None
    year: Optional[Union[int, str]] = None
    authors: List[str] = Field(default_factory=list)
    title: Optional[str] = None
    venue: Optional[str] = None
    pages: Optional[str] = None

class OntologyCode(BaseModel):
    system: str
    code: str

class Mention(BaseModel):
    text: str
    unit_id: Optional[str] = None
    span: Optional[Dict[str, int]] = None

class Concept(BaseModel):
    id: str
    name: str
    type: Optional[str] = None
    ontology: List[OntologyCode] = Field(default_factory=list)
    aliases: List[str] = Field(default_factory=list)
    definition: Optional[str] = None
    section_ids: List[str] = Field(default_factory=list)
    mentions: List[Mention] = Field(default_factory=list)

class Relation(BaseModel):
    subject_id: str
    predicate: str
    object_id: str
    evidence_unit_ids: List[str] = Field(default_factory=list)
    confidence: Optional[float] = None

class Nugget(BaseModel):
    question: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)

class Chunk(BaseModel):
    id: str
    unit_ids: List[str]
    text: str
    tokens: Optional[int] = None
    embedding_ref: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)
    hierarchy_path: List[str] = Field(default_factory=list)

class SourceBlock(BaseModel):
    document_id: str
    book_title: str
    edition: Optional[str] = None
    volume: Optional[str] = None
    publisher: Optional[str] = None
    isbn13: Optional[str] = None
    isbn10: Optional[str] = None
    year: Optional[int] = None
    chapter_number: Optional[Union[int, str]] = None
    chapter_pages: Optional[Dict[str, Union[int, str]]] = None
    source_url: Optional[str] = None
    file_sha256: Optional[str] = None
    license: Optional[str] = None
    rights_holder: Optional[str] = None
    access: Optional[str] = None
    ingest_date: Optional[str] = None

class DocumentBlock(BaseModel):
    chapter_title: str
    authors: List[str]
    editors: List[str] = Field(default_factory=list)
    affiliations: List[str] = Field(default_factory=list)
    abstract: Optional[str] = None
    learning_objectives: List[str] = Field(default_factory=list)
    keywords: List[str] = Field(default_factory=list)
    summary_key_points: List[str] = Field(default_factory=list)
    disclaimer: Optional[str] = None

class StructureBlock(BaseModel):
    toc_path: List[str] = Field(default_factory=list)
    section_tree: List[SectionNode] = Field(default_factory=list)

class ContentBlock(BaseModel):
    text_units: List[TextUnit]
    figures: List[Figure] = Field(default_factory=list)
    tables: List[Table] = Field(default_factory=list)
    boxes: List[Box] = Field(default_factory=list)
    equations: List[Equation] = Field(default_factory=list)
    cases: List[Case] = Field(default_factory=list)
    references: List[Reference] = Field(default_factory=list)

class EntitiesBlock(BaseModel):
    concepts: List[Concept] = Field(default_factory=list)
    relations: List[Relation] = Field(default_factory=list)

class PedagogyBlock(BaseModel):
    questions: List[Dict[str, Any]] = Field(default_factory=list)
    bloom_levels: List[str] = Field(default_factory=list)
    difficulty: Optional[str] = None
    estimated_time_minutes: Optional[float] = None

class MedicalExtension(BaseModel):
    procedures: List[Dict[str, Any]] = Field(default_factory=list)
    indications: List[str] = Field(default_factory=list)
    contraindications: List[str] = Field(default_factory=list)
    complications: List[str] = Field(default_factory=list)
    drugs: List[Dict[str, Any]] = Field(default_factory=list)
    guidelines: List[Dict[str, Any]] = Field(default_factory=list)

class RetrievalBlock(BaseModel):
    keywords: List[str]
    summary_tldr: str
    nuggets: List[Nugget] = Field(default_factory=list)
    chunking: Optional[Dict[str, Any]] = None
    chunks: List[Chunk] = Field(default_factory=list)

class Versioning(BaseModel):
    extraction_tool: str
    model: str
    model_version: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    revision: str = "r1"
    notes: Optional[str] = None

class QCFlag(BaseModel):
    code: Optional[str] = None
    message: Optional[str] = None
    severity: Optional[str] = None

class AuditBlock(BaseModel):
    reviewer: Optional[str] = None
    qc_flags: List[QCFlag] = Field(default_factory=list)
    coverage: Optional[Dict[str, Any]] = None

class RightsBlock(BaseModel):
    license: Optional[str] = None
    rights_holder: Optional[str] = None
    access: Optional[str] = None
    terms: Optional[str] = None

class ChapterDoc(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: SourceBlock
    document: DocumentBlock
    structure: StructureBlock
    content: ContentBlock
    entities: Optional[EntitiesBlock] = None
    pedagogy: Optional[PedagogyBlock] = None
    medical_extension: Optional[MedicalExtension] = None
    retrieval: RetrievalBlock
    versioning: Versioning
    rights: Optional[RightsBlock] = None
    audit: Optional[AuditBlock] = None

if __name__ == "__main__":
    import json
    schema = ChapterDoc.model_json_schema()
    with open("textbook_chapter.schema.json", "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    print("Wrote textbook_chapter.schema.json")
