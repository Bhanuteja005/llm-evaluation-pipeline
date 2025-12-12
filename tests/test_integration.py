"""Integration tests for the full pipeline."""

import json
import tempfile
from pathlib import Path

import pytest

from src.evaluators.factual import FactualEvaluator
from src.evaluators.latency_cost import LatencyCostEvaluator
from src.evaluators.relevance import RelevanceEvaluator
from src.ingest import load_inputs
from src.llm_client import MockLLMClient
from src.prompt_builder import build_prompt
from src.scoring import Scorer


@pytest.fixture
def sample_conversation_file():
    """Create a temporary conversation file."""
    conversation_data = {
        "chat_id": 78128,
        "user_id": 77096,
        "conversation_turns": [
            {
                "turn": 1,
                "sender_id": 1,
                "role": "AI/Chatbot",
                "message": "Hello, how can I help you?",
                "created_at": "2025-01-01T10:00:00.000000Z",
            },
            {
                "turn": 2,
                "sender_id": 77096,
                "role": "User",
                "message": "What hotels are available near the clinic?",
                "created_at": "2025-01-01T10:01:00.000000Z",
            },
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(conversation_data, f)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


@pytest.fixture
def sample_context_file():
    """Create a temporary context vectors file."""
    context_data = {
        "status": "success",
        "status_code": 200,
        "message": "Success",
        "data": {
            "vector_data": [
                {
                    "id": 1,
                    "source_url": "https://example.com/hotels",
                    "text": "Gopal Mansion offers air-conditioned rooms for Rs 800 per night. "
                    "You can book rooms in advance by sending an email.",
                    "tokens": 20,
                    "created_at": "2025-01-01T00:00:00.000Z",
                },
                {
                    "id": 2,
                    "source_url": "https://example.com/hotels",
                    "text": "Hotel Supreme is a 5-minute walk from the clinic. "
                    "They offer special packages for our patients starting at Rs 30,000 for 10 nights.",
                    "tokens": 25,
                    "created_at": "2025-01-01T00:00:00.000Z",
                },
            ],
            "sources": {
                "message_id": 12345,
                "vector_ids": ["1", "2"],
                "vectors_used": [1, 2],
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(context_data, f)
        temp_path = f.name

    yield temp_path
    Path(temp_path).unlink()


def test_full_pipeline_mock(sample_conversation_file, sample_context_file):
    """Test complete pipeline with mock LLM."""
    # Load inputs
    conversation, context_vectors_data = load_inputs(
        sample_conversation_file, sample_context_file
    )

    assert conversation.chat_id == 78128
    context_vectors = context_vectors_data.get_vector_data()
    assert len(context_vectors) == 2

    # Build prompt
    prompt = build_prompt(conversation, context_vectors)
    assert len(prompt) > 0
    assert "hotel" in prompt.lower()

    # Generate response with mock LLM
    llm_client = MockLLMClient()
    llm_response = llm_client.generate(prompt)

    assert llm_response.text is not None
    assert llm_response.latency_ms > 0
    assert llm_response.token_count > 0

    # Initialize evaluators
    relevance_evaluator = RelevanceEvaluator()
    relevance_evaluator.build_index(context_vectors)

    factual_evaluator = FactualEvaluator()
    factual_evaluator.set_context(context_vectors)

    latency_cost_evaluator = LatencyCostEvaluator()

    # Run evaluations
    latest_message = conversation.get_latest_user_message()
    assert latest_message is not None

    relevance_result = relevance_evaluator.evaluate(llm_response.text, latest_message.message)
    assert 0 <= relevance_result.relevance_score <= 1
    assert 0 <= relevance_result.completeness_score <= 1

    factual_result = factual_evaluator.evaluate(llm_response.text)
    assert 0 <= factual_result.hallucination_rate <= 1

    latency_cost_result = latency_cost_evaluator.evaluate(
        llm_response.latency_ms,
        llm_response.token_count,
        llm_response.input_tokens,
        llm_response.output_tokens,
        llm_response.estimated_cost_usd(),
    )
    assert latency_cost_result.latency_ms > 0

    # Create report
    scorer = Scorer()
    report = scorer.create_report(
        chat_id=conversation.chat_id,
        turn=latest_message.turn,
        user_message=latest_message.message,
        model_response=llm_response.text,
        provider=llm_response.provider,
        relevance=relevance_result,
        factual=factual_result,
        latency_cost=latency_cost_result,
    )

    assert report.chat_id == 78128
    assert 0 <= report.overall_quality_score <= 1
    assert isinstance(report.passed_thresholds, bool)
    assert len(report.summary) > 0

    # Test serialization
    report_dict = report.to_dict()
    assert "metadata" in report_dict
    assert "metrics" in report_dict
    assert "aggregate" in report_dict


def test_scorer_quality_calculation():
    """Test quality score calculation."""
    from src.evaluators.factual import ClaimVerification, FactualResult
    from src.evaluators.relevance import RelevanceResult

    scorer = Scorer()

    # Mock results
    relevance = RelevanceResult(
        relevance_score=0.8,
        completeness_score=0.7,
        top_k_similarities=[0.9, 0.7, 0.5],
        top_k_context_ids=[1, 2, 3],
        supporting_sources=["source1"],
        explanation=["Good relevance"],
    )

    factual = FactualResult(
        hallucination_rate=0.1,
        total_claims=10,
        verified_claims=9,
        unverified_claims=1,
        contradicted_claims=0,
        claim_verifications=[],
        explanation=["Low hallucination"],
    )

    quality_score = scorer.compute_quality_score(relevance, factual)

    assert 0 <= quality_score <= 1
    # With high relevance and low hallucination, score should be good
    assert quality_score > 0.6
