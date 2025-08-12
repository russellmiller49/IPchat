from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Literal

class ProcedureStep(BaseModel):
    step_number: int
    description: str
    critical_points: List[str] = []

class Procedure(BaseModel):
    name: str
    indications: List[str] = []
    contraindications: List[str] = []
    required_equipment: List[str] = []
    steps: List[ProcedureStep] = []
    complications: List[str] = []
    page_range: List[int] = []

class AlgorithmDecision(BaseModel):
    question: str
    options: List[str] = []
    actions: List[str] = []

class Algorithm(BaseModel):
    id: Optional[str] = None
    title: str
    purpose: Optional[str] = None
    decision_points: List[AlgorithmDecision] = []
    pages: List[int] = []

class ClinicalGuideline(BaseModel):
    guideline: str
    category: Optional[str] = None
    recommendation_grade: Optional[str] = None
    evidence_level: Optional[str] = None
    details: Optional[str] = None
    page: Optional[int] = None

class DrugInfo(BaseModel):
    drug_name: str
    drug_class: Optional[str] = None
    indications: List[str] = []
    dosage: Optional[str] = None
    contraindications: List[str] = []
    side_effects: List[str] = []
    page: Optional[int] = None

class TableBlock(BaseModel):
    table_id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    headers: List[str] = []
    rows: List[List[str]] = []
    footnotes: List[str] = []
    clinical_relevance: Optional[str] = None
    page: Optional[int] = None
    source_xlsx: Optional[str] = None
    bbox: Optional[List[float]] = None

class FigureBlock(BaseModel):
    figure_id: Optional[str] = None
    title: Optional[str] = None
    caption: Optional[str] = None
    kind: Optional[Literal["photo","fluoro","endoscopic","diagram","algorithm"]] = None
    clinical_significance: Optional[str] = None
    page: Optional[int] = None
    asset_path: Optional[str] = None
    bbox: Optional[List[float]] = None

class BoxBlock(BaseModel):
    box_id: Optional[str] = None
    title: Optional[str] = None
    kind: Optional[Literal["tip","warning","pearl","note"]] = None
    content: str
    page: Optional[int] = None

class ClinicalCase(BaseModel):
    case_id: Optional[str] = None
    presentation: str
    history: Optional[str] = None
    examination_findings: Optional[str] = None
    investigations: List[str] = []
    diagnosis: Optional[str] = None
    management: Optional[str] = None
    outcome: Optional[str] = None
    learning_points: List[str] = []
    page_range: List[int] = []

class Definition(BaseModel):
    term: str
    definition: str
    context: Optional[str] = None
    page: Optional[int] = None

class Reference(BaseModel):
    citation: str
    doi: Optional[str] = None
    pmid: Optional[str] = None
    ref_type: Optional[str] = None
    page: Optional[int] = None

class ChapterMetadata(BaseModel):
    title: str
    authors: List[str] = []
    chapter_number: Optional[str] = None
    learning_objectives: List[Dict[str, str]] = []  # {"objective","page"}
    key_points: List[Dict[str, str]] = []           # {"point","page"}

class SummaryBlock(BaseModel):
    chapter_summary: Optional[str] = None
    clinical_pearls: List[str] = []
    practice_recommendations: List[str] = []
    future_directions: Optional[str] = None

class Provenance(BaseModel):
    pdf_path: str
    adobe_json_path: str
    file_hash: str
    model: str
    extracted_at: str

class TextbookChapter(BaseModel):
    chapter_metadata: ChapterMetadata
    clinical_content: Dict[str, List]         # {"procedures":[...], "algorithms":[...], "clinical_guidelines":[...], "drug_information":[...]}
    structured_data: Dict[str, List]          # {"tables":[...], "figures":[...], "boxes":[...]}
    clinical_cases: List[ClinicalCase] = []
    definitions: List[Definition] = []
    references: List[Reference] = []
    summary: SummaryBlock
    provenance: Provenance