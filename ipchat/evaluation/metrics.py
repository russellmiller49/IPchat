"""
Evaluation metrics for the chatbot system.
"""

from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json

class RetrievalMetrics:
    """Metrics for evaluating retrieval quality"""
    
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate precision at k"""
        retrieved_k = retrieved[:k]
        relevant_in_retrieved = len(set(retrieved_k) & set(relevant))
        return relevant_in_retrieved / k if k > 0 else 0.0
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate recall at k"""
        retrieved_k = retrieved[:k]
        relevant_in_retrieved = len(set(retrieved_k) & set(relevant))
        return relevant_in_retrieved / len(relevant) if relevant else 0.0
    
    @staticmethod
    def mean_reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        """Calculate MRR"""
        for i, doc in enumerate(retrieved):
            if doc in relevant:
                return 1.0 / (i + 1)
        return 0.0
    
    @staticmethod
    def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        """Calculate NDCG at k"""
        def dcg_at_k(scores, k):
            scores = scores[:k]
            if not scores:
                return 0.0
            return scores[0] + sum(scores[i] / np.log2(i + 2) for i in range(1, len(scores)))
        
        # Create relevance scores (1 if relevant, 0 otherwise)
        scores = [1 if doc in relevant else 0 for doc in retrieved[:k]]
        ideal_scores = [1] * min(len(relevant), k) + [0] * (k - min(len(relevant), k))
        
        dcg = dcg_at_k(scores, k)
        idcg = dcg_at_k(ideal_scores, k)
        
        return dcg / idcg if idcg > 0 else 0.0

class AnswerQualityMetrics:
    """Metrics for evaluating answer quality"""
    
    @staticmethod
    def citation_accuracy(answer: str, expected_citations: List[str]) -> float:
        """Check if answer includes proper citations"""
        citations_found = sum(1 for citation in expected_citations if citation in answer)
        return citations_found / len(expected_citations) if expected_citations else 1.0
    
    @staticmethod
    def answer_completeness(answer: str, expected_points: List[str]) -> float:
        """Check if answer covers expected key points"""
        points_covered = sum(1 for point in expected_points if point.lower() in answer.lower())
        return points_covered / len(expected_points) if expected_points else 1.0
    
    @staticmethod
    def factual_accuracy(answer: str, ground_truth: str, threshold: float = 0.8) -> float:
        """
        Compare semantic similarity between answer and ground truth.
        Requires embeddings - simplified version here.
        """
        # Simplified: check for key terms overlap
        answer_terms = set(answer.lower().split())
        truth_terms = set(ground_truth.lower().split())
        
        intersection = answer_terms & truth_terms
        union = answer_terms | truth_terms
        
        return len(intersection) / len(union) if union else 0.0
    
    @staticmethod
    def answer_relevance(answer: str, question: str) -> float:
        """
        Check if answer is relevant to the question.
        Simplified version using keyword overlap.
        """
        # Extract key terms from question
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'as', 'are', 'was', 'were', 'what', 'when', 'where', 'who', 'why', 'how'}
        question_words = question_words - stop_words
        
        # Check how many question terms appear in answer
        if not question_words:
            return 1.0
        
        relevance = len(question_words & answer_words) / len(question_words)
        return min(relevance * 1.5, 1.0)  # Boost score but cap at 1.0

class SystemEvaluator:
    """Complete system evaluation"""
    
    def __init__(self, retrieval_metrics: RetrievalMetrics = None, answer_metrics: AnswerQualityMetrics = None):
        self.retrieval_metrics = retrieval_metrics or RetrievalMetrics()
        self.answer_metrics = answer_metrics or AnswerQualityMetrics()
        self.results = []
    
    def evaluate_query(self, 
                      query: str,
                      retrieved_docs: List[str],
                      generated_answer: str,
                      ground_truth: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate a single query"""
        
        results = {
            'query': query,
            'retrieval_precision@5': self.retrieval_metrics.precision_at_k(
                retrieved_docs, ground_truth.get('relevant_docs', []), 5
            ),
            'retrieval_recall@5': self.retrieval_metrics.recall_at_k(
                retrieved_docs, ground_truth.get('relevant_docs', []), 5
            ),
            'mrr': self.retrieval_metrics.mean_reciprocal_rank(
                retrieved_docs, ground_truth.get('relevant_docs', [])
            ),
            'ndcg@5': self.retrieval_metrics.ndcg_at_k(
                retrieved_docs, ground_truth.get('relevant_docs', []), 5
            ),
            'citation_accuracy': self.answer_metrics.citation_accuracy(
                generated_answer, ground_truth.get('required_citations', [])
            ),
            'answer_completeness': self.answer_metrics.answer_completeness(
                generated_answer, ground_truth.get('key_points', [])
            ),
            'factual_accuracy': self.answer_metrics.factual_accuracy(
                generated_answer, ground_truth.get('expected_answer', '')
            ),
            'answer_relevance': self.answer_metrics.answer_relevance(
                generated_answer, query
            )
        }
        
        # Calculate overall score
        results['overall_score'] = np.mean([
            results['retrieval_precision@5'],
            results['mrr'],
            results['citation_accuracy'],
            results['factual_accuracy'],
            results['answer_relevance']
        ])
        
        self.results.append(results)
        return results
    
    def get_aggregate_metrics(self) -> Dict[str, float]:
        """Get aggregate metrics across all evaluated queries"""
        if not self.results:
            return {}
        
        aggregate = {}
        metrics = [k for k in self.results[0].keys() if k != 'query']
        
        for metric in metrics:
            values = [r[metric] for r in self.results]
            aggregate[f'mean_{metric}'] = np.mean(values)
            aggregate[f'std_{metric}'] = np.std(values)
            aggregate[f'min_{metric}'] = np.min(values)
            aggregate[f'max_{metric}'] = np.max(values)
        
        return aggregate
    
    def save_results(self, output_path: str):
        """Save evaluation results"""
        with open(output_path, 'w') as f:
            json.dump({
                'individual_results': self.results,
                'aggregate_metrics': self.get_aggregate_metrics()
            }, f, indent=2)
    
    def generate_report(self) -> str:
        """Generate a human-readable evaluation report"""
        if not self.results:
            return "No evaluation results available."
        
        metrics = self.get_aggregate_metrics()
        
        report = "=" * 60 + "\n"
        report += "EVALUATION REPORT\n"
        report += "=" * 60 + "\n\n"
        
        report += f"Total Queries Evaluated: {len(self.results)}\n\n"
        
        report += "RETRIEVAL PERFORMANCE:\n"
        report += "-" * 30 + "\n"
        report += f"Precision@5: {metrics.get('mean_retrieval_precision@5', 0):.3f} (±{metrics.get('std_retrieval_precision@5', 0):.3f})\n"
        report += f"Recall@5: {metrics.get('mean_retrieval_recall@5', 0):.3f} (±{metrics.get('std_retrieval_recall@5', 0):.3f})\n"
        report += f"MRR: {metrics.get('mean_mrr', 0):.3f} (±{metrics.get('std_mrr', 0):.3f})\n"
        report += f"NDCG@5: {metrics.get('mean_ndcg@5', 0):.3f} (±{metrics.get('std_ndcg@5', 0):.3f})\n\n"
        
        report += "ANSWER QUALITY:\n"
        report += "-" * 30 + "\n"
        report += f"Citation Accuracy: {metrics.get('mean_citation_accuracy', 0):.3f} (±{metrics.get('std_citation_accuracy', 0):.3f})\n"
        report += f"Answer Completeness: {metrics.get('mean_answer_completeness', 0):.3f} (±{metrics.get('std_answer_completeness', 0):.3f})\n"
        report += f"Factual Accuracy: {metrics.get('mean_factual_accuracy', 0):.3f} (±{metrics.get('std_factual_accuracy', 0):.3f})\n"
        report += f"Answer Relevance: {metrics.get('mean_answer_relevance', 0):.3f} (±{metrics.get('std_answer_relevance', 0):.3f})\n\n"
        
        report += "OVERALL PERFORMANCE:\n"
        report += "-" * 30 + "\n"
        report += f"Overall Score: {metrics.get('mean_overall_score', 0):.3f} (±{metrics.get('std_overall_score', 0):.3f})\n"
        
        report += "\n" + "=" * 60 + "\n"
        
        return report