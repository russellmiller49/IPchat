"""
Evaluation benchmarks for the IP chatbot.
"""

from typing import List, Dict, Any
from dataclasses import dataclass
import json
from pathlib import Path

@dataclass
class BenchmarkQuestion:
    """A benchmark question with validated answer"""
    question_id: str
    question: str
    question_type: str  # 'factual', 'procedural', 'diagnostic', 'comparative'
    expected_answer: str
    required_citations: List[str]
    difficulty: str  # 'easy', 'medium', 'hard'
    source_documents: List[str]

class IPBenchmark:
    """Benchmark dataset for interventional pulmonology questions"""
    
    def __init__(self):
        self.questions = self._load_benchmark_questions()
    
    def _load_benchmark_questions(self) -> List[BenchmarkQuestion]:
        """Load or create benchmark questions"""
        
        # Start with a curated set of essential questions
        questions = [
            BenchmarkQuestion(
                question_id="q001",
                question="What is the diagnostic yield of EBUS-TBNA for mediastinal lymph nodes?",
                question_type="factual",
                expected_answer="The diagnostic yield of EBUS-TBNA for mediastinal lymph nodes ranges from 85-95% in most studies.",
                required_citations=["research_papers"],
                difficulty="easy",
                source_documents=["ebus_studies.pdf"]
            ),
            BenchmarkQuestion(
                question_id="q002",
                question="What are the contraindications for bronchial thermoplasty?",
                question_type="procedural",
                expected_answer="Contraindications include: active respiratory infection, FEV1 <60% predicted, recent asthma exacerbation, bleeding disorders",
                required_citations=["textbook", "guidelines"],
                difficulty="medium",
                source_documents=["thermoplasty_chapter.pdf"]
            ),
            BenchmarkQuestion(
                question_id="q003",
                question="Compare the pneumothorax rates between CT-guided biopsy and navigational bronchoscopy",
                question_type="comparative",
                expected_answer="CT-guided biopsy: 15-25% pneumothorax rate. Navigational bronchoscopy: 2-5% pneumothorax rate.",
                required_citations=["comparative_studies"],
                difficulty="hard",
                source_documents=["nav_bronch_studies.pdf", "ct_biopsy_studies.pdf"]
            ),
            BenchmarkQuestion(
                question_id="q004",
                question="What is the recommended approach for malignant central airway obstruction?",
                question_type="procedural",
                expected_answer="Multimodality approach: tumor debulking (laser, electrocautery, cryotherapy) followed by stent placement if needed.",
                required_citations=["textbook", "guidelines"],
                difficulty="medium",
                source_documents=["cao_management.pdf"]
            ),
            BenchmarkQuestion(
                question_id="q005",
                question="What are the indications for endobronchial valve placement?",
                question_type="procedural",
                expected_answer="Severe emphysema with hyperinflation, intact interlobar fissures, heterogeneous disease distribution, no collateral ventilation.",
                required_citations=["research_papers", "guidelines"],
                difficulty="medium",
                source_documents=["valve_therapy.pdf"]
            ),
            BenchmarkQuestion(
                question_id="q006",
                question="What is the success rate of pleurodesis with talc for malignant pleural effusion?",
                question_type="factual",
                expected_answer="Talc pleurodesis success rate is 80-90% for malignant pleural effusions when performed via thoracoscopy.",
                required_citations=["research_papers"],
                difficulty="easy",
                source_documents=["pleurodesis_studies.pdf"]
            ),
            BenchmarkQuestion(
                question_id="q007",
                question="How do you manage a bronchopleural fistula?",
                question_type="procedural",
                expected_answer="Options include: endobronchial valves, glue/sealants, surgical repair, conservative management with chest drainage.",
                required_citations=["textbook", "case_series"],
                difficulty="hard",
                source_documents=["bpf_management.pdf"]
            ),
            BenchmarkQuestion(
                question_id="q008",
                question="What is the diagnostic algorithm for peripheral pulmonary nodules?",
                question_type="diagnostic",
                expected_answer="Risk assessment (size, characteristics) → CT surveillance vs tissue diagnosis → Navigation bronchoscopy/EBUS/CT-guided biopsy based on location.",
                required_citations=["guidelines", "textbook"],
                difficulty="medium",
                source_documents=["ppn_algorithm.pdf"]
            ),
            BenchmarkQuestion(
                question_id="q009",
                question="What are the complications of transbronchial lung cryobiopsy?",
                question_type="factual",
                expected_answer="Major complications: pneumothorax (10-15%), moderate-severe bleeding (1-5%), death (<1%). Minor: mild bleeding, transient hypoxemia.",
                required_citations=["research_papers", "systematic_reviews"],
                difficulty="medium",
                source_documents=["cryobiopsy_safety.pdf"]
            ),
            BenchmarkQuestion(
                question_id="q010",
                question="When should you use rigid bronchoscopy over flexible bronchoscopy?",
                question_type="comparative",
                expected_answer="Rigid bronchoscopy preferred for: massive hemoptysis, foreign body removal, large tumor debulking, stent placement, better airway control.",
                required_citations=["textbook", "expert_consensus"],
                difficulty="medium",
                source_documents=["rigid_vs_flexible.pdf"]
            )
        ]
        
        return questions
    
    def save_benchmark(self, output_path: Path):
        """Save benchmark to JSON file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = [
            {
                'question_id': q.question_id,
                'question': q.question,
                'question_type': q.question_type,
                'expected_answer': q.expected_answer,
                'required_citations': q.required_citations,
                'difficulty': q.difficulty,
                'source_documents': q.source_documents
            }
            for q in self.questions
        ]
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_benchmark(self, input_path: Path) -> List[BenchmarkQuestion]:
        """Load benchmark from JSON file"""
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        questions = []
        for item in data:
            questions.append(BenchmarkQuestion(**item))
        
        return questions
    
    def get_questions_by_type(self, question_type: str) -> List[BenchmarkQuestion]:
        """Get questions filtered by type"""
        return [q for q in self.questions if q.question_type == question_type]
    
    def get_questions_by_difficulty(self, difficulty: str) -> List[BenchmarkQuestion]:
        """Get questions filtered by difficulty"""
        return [q for q in self.questions if q.difficulty == difficulty]