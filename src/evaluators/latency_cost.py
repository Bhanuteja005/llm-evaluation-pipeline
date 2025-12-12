"""Latency and cost evaluator."""

from dataclasses import dataclass
from typing import List

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class LatencyCostResult:
    """Result from latency and cost evaluation."""

    latency_ms: float
    token_count: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    latency_flag: bool  # True if exceeds threshold
    cost_flag: bool  # True if exceeds threshold
    explanation: List[str]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "latency_ms": round(self.latency_ms, 2),
            "token_count": self.token_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "latency_flag": self.latency_flag,
            "cost_flag": self.cost_flag,
            "explanation": self.explanation,
        }


class LatencyCostEvaluator:
    """Evaluates latency and cost metrics."""

    def __init__(
        self,
        max_latency_ms: float = None,
        max_cost_usd: float = None,
    ):
        """
        Initialize latency/cost evaluator.

        Args:
            max_latency_ms: Maximum acceptable latency in milliseconds
            max_cost_usd: Maximum acceptable cost per call in USD
        """
        self.max_latency_ms = max_latency_ms or settings.max_latency_ms
        self.max_cost_usd = max_cost_usd or settings.max_cost_usd

        logger.info(
            f"Initialized LatencyCostEvaluator: "
            f"max_latency={self.max_latency_ms}ms, max_cost=${self.max_cost_usd}"
        )

    def evaluate(
        self,
        latency_ms: float,
        token_count: int,
        input_tokens: int,
        output_tokens: int,
        estimated_cost_usd: float,
    ) -> LatencyCostResult:
        """
        Evaluate latency and cost metrics.

        Args:
            latency_ms: Actual latency in milliseconds
            token_count: Total token count
            input_tokens: Input token count
            output_tokens: Output token count
            estimated_cost_usd: Estimated cost in USD

        Returns:
            LatencyCostResult with metrics and flags
        """
        # Check thresholds
        latency_flag = latency_ms > self.max_latency_ms
        cost_flag = estimated_cost_usd > self.max_cost_usd

        # Generate explanation
        explanation = self._generate_explanation(
            latency_ms,
            latency_flag,
            estimated_cost_usd,
            cost_flag,
            token_count,
        )

        return LatencyCostResult(
            latency_ms=latency_ms,
            token_count=token_count,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=estimated_cost_usd,
            latency_flag=latency_flag,
            cost_flag=cost_flag,
            explanation=explanation,
        )

    def _generate_explanation(
        self,
        latency_ms: float,
        latency_flag: bool,
        cost_usd: float,
        cost_flag: bool,
        token_count: int,
    ) -> List[str]:
        """Generate human-readable explanation."""
        explanations = []

        # Latency
        if latency_flag:
            explanations.append(
                f"⚠ Latency ({latency_ms:.0f}ms) exceeds threshold ({self.max_latency_ms}ms)."
            )
        else:
            if latency_ms < self.max_latency_ms * 0.5:
                explanations.append(f"✓ Excellent latency: {latency_ms:.0f}ms")
            else:
                explanations.append(
                    f"✓ Acceptable latency: {latency_ms:.0f}ms (under {self.max_latency_ms}ms threshold)"
                )

        # Cost
        if cost_flag:
            explanations.append(
                f"⚠ Cost (${cost_usd:.6f}) exceeds threshold (${self.max_cost_usd:.6f})."
            )
        else:
            explanations.append(
                f"✓ Cost within budget: ${cost_usd:.6f} (under ${self.max_cost_usd:.6f} threshold)"
            )

        # Token efficiency
        explanations.append(f"Total tokens used: {token_count}")

        # Projected costs at scale
        cost_per_1k = cost_usd * 1000
        cost_per_1m = cost_usd * 1_000_000
        explanations.append(
            f"Projected cost at scale: ${cost_per_1k:.2f}/1K calls, ${cost_per_1m:.2f}/1M calls"
        )

        return explanations

    def compute_percentiles(self, latencies: List[float]) -> dict:
        """
        Compute latency percentiles.

        Args:
            latencies: List of latency measurements

        Returns:
            Dictionary with p50, p95, p99 latencies
        """
        if not latencies:
            return {"p50": 0, "p95": 0, "p99": 0}

        sorted_latencies = sorted(latencies)
        n = len(sorted_latencies)

        p50_idx = int(n * 0.50)
        p95_idx = int(n * 0.95)
        p99_idx = int(n * 0.99)

        return {
            "p50": sorted_latencies[p50_idx],
            "p95": sorted_latencies[p95_idx],
            "p99": sorted_latencies[min(p99_idx, n - 1)],
        }
