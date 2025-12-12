"""Scoring and aggregation of evaluation results."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.evaluators.factual import FactualResult
from src.evaluators.latency_cost import LatencyCostResult
from src.evaluators.relevance import RelevanceResult
from src.utils import clamp, get_logger

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Complete evaluation report combining all metrics."""

    # Metadata
    chat_id: int
    turn: int
    user_message: str
    model_response: str
    timestamp: str
    provider: str

    # Individual results
    relevance: RelevanceResult
    factual: FactualResult
    latency_cost: LatencyCostResult

    # Aggregate scores
    overall_quality_score: float
    passed_thresholds: bool

    # Summary
    summary: List[str]

    def to_dict(self) -> Dict:
        """Convert report to dictionary for JSON serialization."""
        return {
            "metadata": {
                "chat_id": self.chat_id,
                "turn": self.turn,
                "timestamp": self.timestamp,
                "provider": self.provider,
            },
            "input": {
                "user_message": self.user_message,
                "model_response": self.model_response,
            },
            "metrics": {
                "relevance": self.relevance.to_dict(),
                "factual_accuracy": self.factual.to_dict(),
                "latency_cost": self.latency_cost.to_dict(),
            },
            "aggregate": {
                "overall_quality_score": round(self.overall_quality_score, 3),
                "passed_thresholds": self.passed_thresholds,
            },
            "summary": self.summary,
        }


class Scorer:
    """Aggregates evaluation results and computes scores."""

    def __init__(
        self,
        relevance_weight: float = 0.45,
        factual_weight: float = 0.35,
        completeness_weight: float = 0.20,
    ):
        """
        Initialize scorer with weights.

        Args:
            relevance_weight: Weight for relevance score
            factual_weight: Weight for factual accuracy score
            completeness_weight: Weight for completeness score
        """
        # Normalize weights
        total = relevance_weight + factual_weight + completeness_weight
        self.relevance_weight = relevance_weight / total
        self.factual_weight = factual_weight / total
        self.completeness_weight = completeness_weight / total

        logger.info(
            f"Scorer initialized with weights: "
            f"relevance={self.relevance_weight:.2f}, "
            f"factual={self.factual_weight:.2f}, "
            f"completeness={self.completeness_weight:.2f}"
        )

    def compute_quality_score(
        self,
        relevance: RelevanceResult,
        factual: FactualResult,
    ) -> float:
        """
        Compute overall quality score.

        Args:
            relevance: Relevance evaluation result
            factual: Factual evaluation result

        Returns:
            Quality score (0 to 1)
        """
        # Factual accuracy score (inverse of hallucination rate)
        factual_accuracy = 1.0 - factual.hallucination_rate

        # Weighted combination
        quality_score = (
            self.relevance_weight * relevance.relevance_score
            + self.factual_weight * factual_accuracy
            + self.completeness_weight * relevance.completeness_score
        )

        return clamp(quality_score, 0.0, 1.0)

    def create_report(
        self,
        chat_id: int,
        turn: int,
        user_message: str,
        model_response: str,
        provider: str,
        relevance: RelevanceResult,
        factual: FactualResult,
        latency_cost: LatencyCostResult,
    ) -> EvaluationReport:
        """
        Create comprehensive evaluation report.

        Args:
            chat_id: Chat ID
            turn: Turn number
            user_message: User's message
            model_response: Model's response
            provider: LLM provider name
            relevance: Relevance evaluation result
            factual: Factual evaluation result
            latency_cost: Latency/cost evaluation result

        Returns:
            Complete evaluation report
        """
        # Compute overall score
        overall_score = self.compute_quality_score(relevance, factual)

        # Check if thresholds passed
        passed = self._check_thresholds(relevance, factual, latency_cost)

        # Generate summary
        summary = self._generate_summary(
            overall_score, relevance, factual, latency_cost, passed
        )

        # Create timestamp
        timestamp = datetime.utcnow().isoformat() + "Z"

        return EvaluationReport(
            chat_id=chat_id,
            turn=turn,
            user_message=user_message,
            model_response=model_response,
            timestamp=timestamp,
            provider=provider,
            relevance=relevance,
            factual=factual,
            latency_cost=latency_cost,
            overall_quality_score=overall_score,
            passed_thresholds=passed,
            summary=summary,
        )

    def _check_thresholds(
        self,
        relevance: RelevanceResult,
        factual: FactualResult,
        latency_cost: LatencyCostResult,
    ) -> bool:
        """Check if all thresholds are passed."""
        from src.config import settings

        checks = [
            relevance.relevance_score >= settings.relevance_threshold,
            factual.hallucination_rate <= 0.3,  # Allow up to 30% hallucination
            not latency_cost.latency_flag,
            not latency_cost.cost_flag,
        ]

        return all(checks)

    def _generate_summary(
        self,
        overall_score: float,
        relevance: RelevanceResult,
        factual: FactualResult,
        latency_cost: LatencyCostResult,
        passed: bool,
    ) -> List[str]:
        """Generate executive summary."""
        summary = []

        # Overall assessment
        if overall_score >= 0.8:
            summary.append("🟢 Excellent response quality")
        elif overall_score >= 0.6:
            summary.append("🟡 Good response quality")
        elif overall_score >= 0.4:
            summary.append("🟠 Fair response quality - improvements needed")
        else:
            summary.append("🔴 Poor response quality - significant issues detected")

        summary.append(f"Overall quality score: {overall_score:.3f}")

        # Key findings
        if relevance.relevance_score < 0.6:
            summary.append("⚠ Low relevance to available context")

        if factual.hallucination_rate > 0.3:
            summary.append(
                f"⚠ High hallucination rate: {factual.hallucination_rate*100:.1f}% of claims unverified"
            )

        if latency_cost.latency_flag:
            summary.append(f"⚠ Latency exceeded threshold: {latency_cost.latency_ms:.0f}ms")

        if latency_cost.cost_flag:
            summary.append(
                f"⚠ Cost exceeded threshold: ${latency_cost.estimated_cost_usd:.6f}"
            )

        # Pass/fail
        if passed:
            summary.append("✓ All thresholds passed")
        else:
            summary.append("✗ Some thresholds not met")

        return summary
