"""Factual accuracy and hallucination detection evaluator."""

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import settings
from src.extractors import Claim, ClaimExtractor
from src.models import VectorData
from src.utils import compute_batch_cosine_similarity, get_logger

logger = get_logger(__name__)


@dataclass
class ClaimVerification:
    """Result of verifying a single claim."""

    claim_text: str
    claim_type: str
    verdict: str  # 'verified', 'weak_support', 'unverified', 'contradiction'
    confidence: float
    supporting_context_id: Optional[int]
    supporting_text: Optional[str]


@dataclass
class FactualResult:
    """Result from factual accuracy evaluation."""

    hallucination_rate: float
    total_claims: int
    verified_claims: int
    unverified_claims: int
    contradicted_claims: int
    claim_verifications: List[ClaimVerification]
    explanation: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "hallucination_rate": round(self.hallucination_rate, 3),
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "unverified_claims": self.unverified_claims,
            "contradicted_claims": self.contradicted_claims,
            "claim_verifications": [
                {
                    "claim_text": cv.claim_text,
                    "claim_type": cv.claim_type,
                    "verdict": cv.verdict,
                    "confidence": round(cv.confidence, 3),
                    "supporting_context_id": cv.supporting_context_id,
                }
                for cv in self.claim_verifications
            ],
            "explanation": self.explanation,
        }


class FactualEvaluator:
    """Evaluates factual accuracy and detects hallucinations."""

    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize factual evaluator.

        Args:
            model_name: Name of sentence transformer model
        """
        self.model_name = model_name or settings.embedding_model
        logger.info(f"Loading embedding model for factual evaluation: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.claim_extractor = ClaimExtractor()
        self.context_vectors: List[VectorData] = []
        self.context_embeddings: Optional[np.ndarray] = None

    def set_context(self, context_vectors: List[VectorData]) -> None:
        """
        Set context vectors for verification.

        Args:
            context_vectors: List of context vectors
        """
        if not context_vectors:
            raise ValueError("No context vectors provided")

        logger.info(f"Setting context with {len(context_vectors)} vectors")
        self.context_vectors = context_vectors

        # Precompute embeddings for all context texts
        texts = [v.text for v in context_vectors]
        self.context_embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False
        )

        logger.info("Context embeddings computed")

    def evaluate(self, response_text: str) -> FactualResult:
        """
        Evaluate factual accuracy of response.

        Args:
            response_text: LLM response text to evaluate

        Returns:
            FactualResult with verification details
        """
        if self.context_embeddings is None:
            raise ValueError("Context not set. Call set_context() first.")

        # Extract claims from response
        claims = self.claim_extractor.extract_claims(response_text)

        if not claims:
            logger.warning("No claims extracted from response")
            return FactualResult(
                hallucination_rate=0.0,
                total_claims=0,
                verified_claims=0,
                unverified_claims=0,
                contradicted_claims=0,
                claim_verifications=[],
                explanation=["No specific factual claims found in response."],
            )

        # Verify each claim
        verifications = []
        for claim in claims:
            verification = self._verify_claim(claim)
            verifications.append(verification)

        # Calculate statistics
        verified = sum(1 for v in verifications if v.verdict == "verified")
        weak_support = sum(1 for v in verifications if v.verdict == "weak_support")
        unverified = sum(1 for v in verifications if v.verdict == "unverified")
        contradicted = sum(1 for v in verifications if v.verdict == "contradiction")

        # Calculate hallucination rate
        # Hallucinations = unverified + contradicted claims
        hallucinations = unverified + contradicted
        hallucination_rate = hallucinations / len(claims) if claims else 0.0

        # Generate explanation
        explanation = self._generate_explanation(
            len(claims), verified, weak_support, unverified, contradicted
        )

        return FactualResult(
            hallucination_rate=hallucination_rate,
            total_claims=len(claims),
            verified_claims=verified,
            unverified_claims=unverified,
            contradicted_claims=contradicted,
            claim_verifications=verifications,
            explanation=explanation,
        )

    def _verify_claim(self, claim: Claim) -> ClaimVerification:
        """
        Verify a single claim against context.

        Args:
            claim: Claim to verify

        Returns:
            ClaimVerification result
        """
        # Encode claim
        claim_embedding = self.model.encode(
            [claim.text], convert_to_numpy=True, show_progress_bar=False
        )

        # Compute similarity with all context vectors
        similarities = compute_batch_cosine_similarity(claim_embedding, self.context_embeddings)

        # Find best match
        best_idx = int(np.argmax(similarities))
        best_similarity = float(similarities[best_idx])
        best_context = self.context_vectors[best_idx]

        # Determine verdict based on similarity threshold
        if best_similarity >= settings.factual_verification_threshold:
            verdict = "verified"
        elif best_similarity >= settings.factual_weak_threshold:
            verdict = "weak_support"
        else:
            # Check for potential contradiction
            if self._check_contradiction(claim.text, best_context.text):
                verdict = "contradiction"
            else:
                verdict = "unverified"

        return ClaimVerification(
            claim_text=claim.text,
            claim_type=claim.claim_type,
            verdict=verdict,
            confidence=best_similarity,
            supporting_context_id=best_context.id if verdict != "unverified" else None,
            supporting_text=best_context.text[:200] if verdict == "verified" else None,
        )

    def _check_contradiction(self, claim_text: str, context_text: str) -> bool:
        """
        Check if claim contradicts context (simple heuristic).

        Args:
            claim_text: Claim text
            context_text: Context text

        Returns:
            True if contradiction detected
        """
        # Simple contradiction detection (can be improved)
        claim_lower = claim_text.lower()
        context_lower = context_text.lower()

        # Check for negation patterns
        negation_patterns = [
            ("not", "is"),
            ("no", "yes"),
            ("cannot", "can"),
            ("never", "always"),
        ]

        for neg, pos in negation_patterns:
            if neg in claim_lower and pos in context_lower:
                return True
            if pos in claim_lower and neg in context_lower:
                return True

        return False

    def _generate_explanation(
        self,
        total: int,
        verified: int,
        weak: int,
        unverified: int,
        contradicted: int,
    ) -> List[str]:
        """Generate human-readable explanation."""
        explanations = []

        explanations.append(f"Extracted {total} factual claims from response.")

        if verified > 0:
            explanations.append(
                f"{verified} claims strongly verified by context ({verified/total*100:.1f}%)."
            )

        if weak > 0:
            explanations.append(f"{weak} claims have weak support in context.")

        if unverified > 0:
            explanations.append(
                f"{unverified} claims could not be verified from context (potential hallucination)."
            )

        if contradicted > 0:
            explanations.append(f"{contradicted} claims contradict available context.")

        # Overall assessment
        hallucination_rate = (unverified + contradicted) / total if total > 0 else 0
        if hallucination_rate == 0:
            explanations.append("No hallucinations detected.")
        elif hallucination_rate < 0.2:
            explanations.append("Low hallucination rate - response is mostly accurate.")
        elif hallucination_rate < 0.5:
            explanations.append("Moderate hallucination rate - some unverified claims present.")
        else:
            explanations.append(
                "High hallucination rate - response contains many unverified claims."
            )

        return explanations
