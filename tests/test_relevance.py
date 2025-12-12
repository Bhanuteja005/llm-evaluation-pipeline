"""Tests for relevance evaluator."""

import pytest

from src.evaluators.relevance import RelevanceEvaluator
from src.models import VectorData


@pytest.fixture
def sample_context_vectors():
    """Sample context vectors for testing."""
    return [
        VectorData(
            id=1,
            source_url="https://example.com/hotels",
            text="Gopal Mansion offers air-conditioned rooms for Rs 800 per night. "
            "Hotel Supreme is a 5-minute walk from the clinic.",
            tokens=20,
            created_at="2025-01-01T00:00:00Z",
        ),
        VectorData(
            id=2,
            source_url="https://example.com/ivf-cost",
            text="A complete IVF cycle costs approximately Rs 3,00,000. "
            "This includes ICSI and blastocyst transfer. Medications cost Rs 1,45,000 more.",
            tokens=25,
            created_at="2025-01-01T00:00:00Z",
        ),
        VectorData(
            id=3,
            source_url="https://example.com/donor-eggs",
            text="Donor egg IVF is recommended for women with low ovarian reserve. "
            "We use carefully screened donors from our egg bank with high success rates.",
            tokens=22,
            created_at="2025-01-01T00:00:00Z",
        ),
    ]


def test_relevance_evaluator_initialization():
    """Test evaluator initialization."""
    evaluator = RelevanceEvaluator()
    assert evaluator.model is not None
    assert evaluator.index is None


def test_build_index(sample_context_vectors):
    """Test building FAISS index."""
    evaluator = RelevanceEvaluator()
    evaluator.build_index(sample_context_vectors)

    assert evaluator.index is not None
    assert evaluator.index.ntotal == len(sample_context_vectors)
    assert len(evaluator.context_vectors) == len(sample_context_vectors)


def test_evaluate_relevance(sample_context_vectors):
    """Test relevance evaluation."""
    evaluator = RelevanceEvaluator()
    evaluator.build_index(sample_context_vectors)

    response_text = (
        "I recommend staying at Gopal Mansion which offers affordable rooms at Rs 800 per night. "
        "It's a great option for patients visiting the clinic."
    )
    user_message = "What hotels are available near the clinic?"

    result = evaluator.evaluate(response_text, user_message, top_k=3)

    assert result.relevance_score >= 0.0
    assert result.relevance_score <= 1.0
    assert result.completeness_score >= 0.0
    assert result.completeness_score <= 1.0
    assert len(result.top_k_similarities) == 3
    assert len(result.top_k_context_ids) == 3
    assert len(result.explanation) > 0


def test_evaluate_high_relevance(sample_context_vectors):
    """Test evaluation with highly relevant response."""
    evaluator = RelevanceEvaluator()
    evaluator.build_index(sample_context_vectors)

    # Response that closely matches context
    response_text = (
        "A complete IVF cycle costs approximately Rs 3,00,000 including ICSI and blastocyst transfer."
    )
    user_message = "How much does IVF cost?"

    result = evaluator.evaluate(response_text, user_message)

    # Should have high relevance
    assert result.relevance_score > 0.6


def test_evaluate_low_relevance(sample_context_vectors):
    """Test evaluation with low relevance response."""
    evaluator = RelevanceEvaluator()
    evaluator.build_index(sample_context_vectors)

    # Response not related to context
    response_text = "The weather is nice today. Would you like to discuss something else?"
    user_message = "What is the cost of IVF?"

    result = evaluator.evaluate(response_text, user_message)

    # Should have lower relevance
    assert result.relevance_score < 0.8
