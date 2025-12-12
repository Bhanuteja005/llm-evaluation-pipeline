"""Relevance and completeness evaluator using embeddings and semantic search."""

from dataclasses import dataclass
from typing import List, Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.models import VectorData
from src.utils import compute_batch_cosine_similarity, get_logger

logger = get_logger(__name__)


@dataclass
class RelevanceResult:
    """Result from relevance evaluation."""

    relevance_score: float
    completeness_score: float
    top_k_similarities: List[float]
    top_k_context_ids: List[int]
    supporting_sources: List[str]
    explanation: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "relevance_score": round(self.relevance_score, 3),
            "completeness_score": round(self.completeness_score, 3),
            "top_k_similarities": [round(s, 3) for s in self.top_k_similarities],
            "top_k_context_ids": self.top_k_context_ids,
            "supporting_sources": self.supporting_sources,
            "explanation": self.explanation,
        }


class RelevanceEvaluator:
    """Evaluates relevance and completeness of LLM responses."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize relevance evaluator.

        Args:
            model_name: Name of sentence transformer model
        """
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.context_vectors: List[VectorData] = []

    def build_index(self, context_vectors: List[VectorData]) -> None:
        """
        Build FAISS index from context vectors.

        Args:
            context_vectors: List of context vectors to index
        """
        if not context_vectors:
            raise ValueError("No context vectors provided")

        logger.info(f"Building FAISS index for {len(context_vectors)} context vectors")

        # Store context vectors
        self.context_vectors = context_vectors

        # Extract texts
        texts = [v.text for v in context_vectors]

        # Generate embeddings
        embeddings = self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

        # Normalize embeddings for cosine similarity (using inner product)
        faiss.normalize_L2(embeddings)

        # Build FAISS index
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        self.index.add(embeddings.astype("float32"))

        logger.info(f"FAISS index built with {self.index.ntotal} vectors")

    def evaluate(
        self, response_text: str, user_message: str, top_k: Optional[int] = None
    ) -> RelevanceResult:
        """
        Evaluate relevance and completeness of response.

        Args:
            response_text: LLM response text
            user_message: Original user message
            top_k: Number of top contexts to retrieve

        Returns:
            RelevanceResult with scores and explanations
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")

        top_k = top_k or settings.top_k_contexts

        # Encode response
        response_embedding = self.model.encode(
            [response_text], convert_to_numpy=True, show_progress_bar=False
        )
        faiss.normalize_L2(response_embedding)

        # Search for top-k similar contexts
        similarities, indices = self.index.search(response_embedding.astype("float32"), top_k)

        # Get results
        top_similarities = similarities[0].tolist()
        top_indices = indices[0].tolist()

        # Get corresponding context data
        top_contexts = [self.context_vectors[idx] for idx in top_indices]
        top_context_ids = [ctx.id for ctx in top_contexts]
        supporting_sources = list(set([ctx.source_url for ctx in top_contexts]))

        # Calculate relevance score (weighted average of top-k similarities)
        relevance_score = self._calculate_relevance_score(top_similarities)

        # Calculate completeness score
        completeness_score = self._calculate_completeness_score(
            response_text, user_message, top_contexts
        )

        # Generate explanation
        explanation = self._generate_explanation(
            relevance_score, completeness_score, top_similarities
        )

        return RelevanceResult(
            relevance_score=relevance_score,
            completeness_score=completeness_score,
            top_k_similarities=top_similarities,
            top_k_context_ids=top_context_ids,
            supporting_sources=supporting_sources,
            explanation=explanation,
        )

    def _calculate_relevance_score(self, similarities: List[float]) -> float:
        """
        Calculate relevance score from similarities.

        Uses weighted average with more weight on top results.
        """
        if not similarities:
            return 0.0

        # Exponentially decaying weights
        weights = [0.4, 0.3, 0.2, 0.07, 0.03][: len(similarities)]
        weights = weights + [0.0] * (len(similarities) - len(weights))

        weighted_sum = sum(s * w for s, w in zip(similarities, weights))
        return min(1.0, max(0.0, weighted_sum))

    def _calculate_completeness_score(
        self, response_text: str, user_message: str, contexts: List[VectorData]
    ) -> float:
        """
        Calculate completeness score.

        Checks if response addresses key aspects of the question.
        """
        # Extract key terms from user message
        user_terms = set(user_message.lower().split())
        response_lower = response_text.lower()

        # Check for question-specific completeness
        completeness_factors = []

        # 1. Does response contain key terms from question?
        important_terms = [t for t in user_terms if len(t) > 4]
        if important_terms:
            term_coverage = sum(1 for term in important_terms if term in response_lower) / len(
                important_terms
            )
            completeness_factors.append(term_coverage)

        # 2. Length factor (very short responses are likely incomplete)
        length_score = min(1.0, len(response_text) / 200)
        completeness_factors.append(length_score)

        # 3. Presence of specifics (numbers, dates, URLs suggest detailed answer)
        has_specifics = any(
            [
                any(char.isdigit() for char in response_text),
                "http" in response_text.lower(),
                "$" in response_text or "rs" in response_text.lower(),
            ]
        )
        completeness_factors.append(0.8 if has_specifics else 0.4)

        # Average of factors
        return sum(completeness_factors) / len(completeness_factors) if completeness_factors else 0.5

    def _generate_explanation(
        self, relevance_score: float, completeness_score: float, similarities: List[float]
    ) -> List[str]:
        """Generate human-readable explanation."""
        explanations = []

        # Relevance
        if relevance_score >= 0.8:
            explanations.append("Response is highly relevant to available context.")
        elif relevance_score >= 0.6:
            explanations.append("Response is moderately relevant to available context.")
        else:
            explanations.append("Response has low relevance to available context.")

        # Completeness
        if completeness_score >= 0.7:
            explanations.append("Response appears complete and addresses key aspects.")
        elif completeness_score >= 0.5:
            explanations.append("Response is partially complete.")
        else:
            explanations.append("Response may be incomplete or too brief.")

        # Best match
        if similarities:
            best_sim = similarities[0]
            explanations.append(f"Best context match similarity: {best_sim:.3f}")

        return explanations
