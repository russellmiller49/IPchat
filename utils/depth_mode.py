"""
Depth Mode utilities for nuanced medical evidence synthesis
Implements multi-query expansion, reranking, and contrastive analysis
"""
from typing import List, Dict, Any, Tuple
import re
from functools import lru_cache
import os

# Only import if available
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    CrossEncoder = None

# Initialize reranker if available
RERANKER = None
if RERANKER_AVAILABLE and os.getenv("ENABLE_RERANKER", "1") == "1":
    try:
        RERANKER = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", max_length=512)
    except Exception as e:
        print(f"Could not load reranker: {e}")

def expand_queries(query: str, chat_complete_fn) -> List[str]:
    """
    Generate 3-5 query variations for better recall
    Uses gpt-5-mini for fast, cheap expansion
    """
    system_prompt = (
        "You are a medical search query optimizer. Generate query variations "
        "that capture different aspects, synonyms, and related terms."
    )
    
    user_prompt = f"""Generate 4 search query variations for this medical question:
"{query}"

Include:
- Medical synonyms and acronyms
- Alternative phrasings
- Related outcome terms
- Specific intervention names

Output one query per line, no numbering or bullets."""

    try:
        response = chat_complete_fn(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=200
        )
        
        # Parse response into queries
        lines = response.strip().split("\n")
        queries = [line.strip() for line in lines if line.strip() and len(line.strip()) > 10]
        
        # Always include original
        return [query] + queries[:4]
    except Exception as e:
        print(f"Query expansion failed: {e}")
        return [query]

def multi_query_search(
    queries: List[str], 
    search_fn, 
    k_each: int = 20
) -> List[Dict]:
    """
    Search with multiple queries and merge results
    Deduplicates by chunk_id, keeping highest scores
    """
    all_hits = []
    seen_chunks = {}
    
    for q in queries:
        try:
            hits = search_fn(q, k=k_each)
            for hit in hits:
                chunk_id = hit.get("chunk_id", "")
                if not chunk_id:
                    continue
                    
                # Keep highest scoring version
                if chunk_id not in seen_chunks or hit.get("score", 0) > seen_chunks[chunk_id].get("score", 0):
                    seen_chunks[chunk_id] = hit
        except Exception as e:
            print(f"Search failed for query '{q}': {e}")
    
    # Return sorted by score
    return sorted(seen_chunks.values(), key=lambda x: x.get("score", 0), reverse=True)

def rerank_hits(query: str, hits: List[Dict], top_n: int = 15) -> List[Dict]:
    """
    Rerank hits using cross-encoder for better relevance
    Falls back to original order if reranker unavailable
    """
    if not RERANKER or not hits:
        return hits[:top_n]
    
    try:
        # Prepare query-text pairs
        pairs = []
        valid_hits = []
        
        for hit in hits:
            text = hit.get("text", "")
            if text:
                pairs.append([query, text[:512]])  # Limit text length
                valid_hits.append(hit)
        
        if not pairs:
            return hits[:top_n]
        
        # Get reranking scores
        scores = RERANKER.predict(pairs)
        
        # Add rerank scores to hits
        for hit, score in zip(valid_hits, scores):
            hit["rerank_score"] = float(score)
        
        # Sort by rerank score
        reranked = sorted(valid_hits, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_n]
        
    except Exception as e:
        print(f"Reranking failed: {e}")
        return hits[:top_n]

def enforce_breadth(hits: List[Dict], min_docs: int = 5) -> List[Dict]:
    """
    Ensure we have evidence from multiple distinct documents
    Returns hits covering at least min_docs different sources
    """
    seen_docs = {}
    selected = []
    
    # First pass: get at least one hit per document
    for hit in hits:
        doc_id = hit.get("document_id", "")
        if doc_id and doc_id not in seen_docs:
            seen_docs[doc_id] = True
            selected.append(hit)
            if len(seen_docs) >= min_docs:
                break
    
    # If we don't have enough docs, add more hits from existing docs
    if len(seen_docs) < min_docs:
        for hit in hits:
            if hit not in selected:
                selected.append(hit)
                if len(selected) >= min_docs * 2:  # Reasonable limit
                    break
    
    return selected

def get_contrastive_prompt() -> str:
    """
    Returns the system prompt for nuanced, contrastive synthesis
    """
    return """You are a medical evidence synthesis expert specializing in interventional pulmonology.

When presenting evidence, follow this structured approach:

1. **Bottom Line** (2-3 sentences): State the overall finding and strength of evidence.

2. **Key Numbers**: Present specific outcomes with sample sizes
   - Report as: Study (Year): X% (n/N) or X% [95% CI: Y-Z]
   - Include timepoints when relevant

3. **Where Studies Disagree**: Explicitly note conflicting findings
   - State which studies disagree and why (design, population, endpoints)
   - Don't hide minority findings

4. **Applicability & Limitations**:
   - Population characteristics that affect generalizability
   - Study quality issues (bias, sample size, indirectness)
   - Important exclusion criteria

5. **Clinical Takeaway**: 2-4 practical bullets for clinicians

CRITICAL RULES:
- Cite every claim with (Author Year)
- Report numerical ranges when studies vary
- State "no direct evidence" if applicable
- Distinguish RCT from observational data
- Note if evidence is indirect or low quality
- Never invent numbers or percentages"""

def get_concise_prompt() -> str:
    """
    Returns the standard concise synthesis prompt
    """
    return """You are a medical evidence expert assistant specializing in interventional pulmonology.
Provide accurate, evidence-based answers using ONLY the provided research context.
Cite specific studies inline like (Author Year). Use numbers when available.
Be concise but thorough."""

def build_enhanced_context(
    hits: List[Dict],
    get_chunk_fn,
    include_neighbors: bool = False
) -> Tuple[str, Dict[str, str]]:
    """
    Build context with optional neighbor chunks for continuity
    Returns (context_string, citation_map)
    """
    context_parts = []
    citation_map = {}
    seen_chunks = set()
    
    for i, hit in enumerate(hits, 1):
        chunk_id = hit.get("chunk_id", "")
        if chunk_id in seen_chunks:
            continue
        seen_chunks.add(chunk_id)
        
        doc_id = hit.get("document_id", "Unknown")
        chunk_text = get_chunk_fn(chunk_id)
        
        if not chunk_text:
            continue
        
        # Get citation key
        from utils.citations import extract_author_year
        author, year = extract_author_year(doc_id)
        cite_key = f"{author} {year}" if author and year else doc_id
        citation_map[f"Source_{i}"] = cite_key
        
        # Add main chunk
        context_parts.append(f"[{cite_key}]:\n{chunk_text[:1000]}")
        
        # Optionally add neighbor chunks (not implemented here for simplicity)
        # Would require chunk ordering information
    
    return "\n\n".join(context_parts), citation_map

def critique_and_improve(
    query: str,
    context: str,
    draft_answer: str,
    chat_complete_fn,
    model: str = "gpt-5"
) -> str:
    """
    Second pass to critique and improve the answer for nuance
    """
    critique_prompt = f"""Review this medical evidence synthesis for accuracy and nuance:

QUESTION: {query}

DRAFT ANSWER:
{draft_answer}

ORIGINAL CONTEXT:
{context[:2000]}...

Please check for:
1. Numeric accuracy - verify all numbers match the source
2. Missing opposing evidence or minority findings
3. Overgeneralizations that need qualification
4. Unsupported causal claims
5. Missing applicability caveats
6. Incomplete citations

Provide a CORRECTED version that addresses any issues found. 
If the draft is accurate, return it with minor improvements for clarity."""

    try:
        improved = chat_complete_fn(
            model=model,
            messages=[
                {"role": "system", "content": "You are a medical evidence quality reviewer."},
                {"role": "user", "content": critique_prompt}
            ],
            temperature=0.1,
            max_tokens=1200
        )
        return improved
    except Exception as e:
        print(f"Critique pass failed: {e}")
        return draft_answer

def verify_numeric_claims(answer: str, context: str) -> List[str]:
    """
    Check if numeric claims in answer appear in context
    Returns list of warnings for unverified claims
    """
    warnings = []
    
    # Find all percentages and fractions in answer
    percent_pattern = r'\d+(?:\.\d+)?%'
    fraction_pattern = r'\d+/\d+'
    
    percentages = re.findall(percent_pattern, answer)
    fractions = re.findall(fraction_pattern, answer)
    
    # Check each numeric claim
    for num in percentages + fractions:
        if num not in context:
            # Try to find similar numbers (within 1% for percentages)
            if '%' in num:
                base_num = float(num.rstrip('%'))
                found_similar = False
                for match in re.findall(percent_pattern, context):
                    context_num = float(match.rstrip('%'))
                    if abs(base_num - context_num) < 1.0:
                        found_similar = True
                        break
                if not found_similar:
                    warnings.append(f"Unverified claim: {num}")
            else:
                warnings.append(f"Unverified claim: {num}")
    
    return warnings

def compute_effect_summaries(structured_data: List[Dict]) -> str:
    """
    Compute risk differences and relative risks from structured data
    Provides pre-calculated summaries to prevent LLM math errors
    """
    summaries = []
    
    for item in structured_data:
        try:
            # Example structure - adapt to your actual schema
            if "events" in item and "total" in item:
                events_a = item.get("events_intervention", 0)
                total_a = item.get("total_intervention", 0)
                events_b = item.get("events_control", 0)
                total_b = item.get("total_control", 0)
                
                if total_a > 0 and total_b > 0:
                    rate_a = events_a / total_a
                    rate_b = events_b / total_b
                    risk_diff = rate_a - rate_b
                    
                    # Relative risk with continuity correction for zero events
                    if events_b == 0:
                        rr = "undefined (no control events)"
                    else:
                        rr = rate_a / rate_b
                        rr = f"{rr:.2f}"
                    
                    summary = (
                        f"Study {item.get('study_id', 'Unknown')}: "
                        f"{events_a}/{total_a} vs {events_b}/{total_b} "
                        f"→ Risk Difference: {risk_diff*100:.1f}%, RR: {rr}"
                    )
                    summaries.append(summary)
        except Exception:
            continue
    
    if summaries:
        return "\n\nComputed Effect Sizes:\n" + "\n".join(summaries)
    return ""

# Configuration for depth mode
class DepthConfig:
    """Configuration for depth mode parameters"""
    
    def __init__(self, depth_enabled: bool = False):
        if depth_enabled:
            # Deep mode settings
            self.k_each = 20  # Hits per query
            self.top_k_final = 30  # After reranking
            self.min_docs = 5  # Minimum distinct documents
            self.use_multi_query = True
            self.use_reranker = True
            self.use_critique = True
            self.model = "gpt-5"
            self.max_tokens = 1200
            self.answer_style = "contrastive"
            self.include_neighbors = True
        else:
            # Fast mode settings
            self.k_each = 10
            self.top_k_final = 10
            self.min_docs = 3
            self.use_multi_query = False
            self.use_reranker = False
            self.use_critique = False
            self.model = "gpt-5-mini"
            self.max_tokens = 900
            self.answer_style = "concise"
            self.include_neighbors = False