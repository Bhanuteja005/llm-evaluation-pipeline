"""Tests for factual evaluator."""

import pytest

from src.evaluators.factual import FactualEvaluator
from src.models import VectorData


@pytest.fixture
def sample_context_vectors():
    """Sample context vectors for testing."""
    return [
        VectorData(
            id=1,
            source_url="https://example.com/hotels",
            text="Gopal Mansion offers rooms for Rs 800 per night. "
            "Hotel Supreme is 5 minutes from the clinic. "
            "Happy Home Hotel charges Rs 1400 for single rooms.",
            tokens=25,
            created_at="2025-01-01T00:00:00Z",
        ),
        VectorData(
            id=2,
            source_url="https://example.com/ivf",
            text="IVF cycle costs Rs 3,00,000 including ICSI. "
            "The clinic is located in Colaba, Mumbai. "
            "Success rate is approximately 45% for women under 35.",
            tokens=28,
            created_at="2025-01-01T00:00:00Z",
        ),
    ]


def test_factual_evaluator_initialization():
    """Test evaluator initialization."""
    evaluator = FactualEvaluator()
    assert evaluator.model is not None
    assert evaluator.claim_extractor is not None


def test_set_context(sample_context_vectors):
    """Test setting context."""
    evaluator = FactualEvaluator()
    evaluator.set_context(sample_context_vectors)

    assert len(evaluator.context_vectors) == len(sample_context_vectors)
    assert evaluator.context_embeddings is not None


def test_evaluate_verified_claims(sample_context_vectors):
    """Test evaluation with verified claims."""
    evaluator = FactualEvaluator()
    evaluator.set_context(sample_context_vectors)

    # Response with claims that match context
    response_text = "The IVF cycle costs Rs 3,00,000 and the clinic is in Colaba."

    result = evaluator.evaluate(response_text)

    assert result.total_claims > 0
    assert result.hallucination_rate <= 0.5  # Should have low hallucination


def test_evaluate_unverified_claims(sample_context_vectors):
    """Test evaluation with unverified claims."""
    evaluator = FactualEvaluator()
    evaluator.set_context(sample_context_vectors)

    # Response with claims NOT in context
    response_text = (
        "The clinic is open 24/7 and has 100 doctors on staff. "
        "We guarantee 100% success rate."
    )

    result = evaluator.evaluate(response_text)

    assert result.total_claims > 0
    # Many claims should be unverified
    assert result.unverified_claims > 0


def test_evaluate_no_claims():
    """Test evaluation with no extractable claims."""
    evaluator = FactualEvaluator()
    evaluator.set_context(
        [
            VectorData(
                id=1,
                source_url="https://example.com",
                text="Some context",
                tokens=2,
                created_at="2025-01-01T00:00:00Z",
            )
        ]
    )

    response_text = "Hello there"

    result = evaluator.evaluate(response_text)

    assert result.total_claims == 0
    assert result.hallucination_rate == 0.0
