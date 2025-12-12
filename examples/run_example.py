#!/usr/bin/env python3
"""
Quick example script demonstrating the evaluation pipeline.
Run with: python examples/run_example.py
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import settings
from src.evaluators.factual import FactualEvaluator
from src.evaluators.latency_cost import LatencyCostEvaluator
from src.evaluators.relevance import RelevanceEvaluator
from src.llm_client import get_llm_client
from src.models import Conversation, ConversationTurn, ContextVectors, VectorData
from src.prompt_builder import build_prompt
from src.scoring import Scorer


def main():
    """Run example evaluation."""
    print("🚀 LLM Evaluation Pipeline - Example\n")

    # Create sample conversation
    conversation = Conversation(
        chat_id=12345,
        user_id=67890,
        conversation_turns=[
            ConversationTurn(
                turn=1,
                sender_id=1,
                role="AI/Chatbot",
                message="Hello! How can I help you today?",
                created_at="2025-01-01T10:00:00Z",
            ),
            ConversationTurn(
                turn=2,
                sender_id=67890,
                role="User",
                message="What hotels are available near your clinic?",
                created_at="2025-01-01T10:01:00Z",
            ),
        ],
    )

    # Create sample context vectors
    context_vectors = [
        VectorData(
            id=1,
            source_url="https://example.com/hotels",
            text="We have partnerships with several hotels near the clinic. "
            "Gopal Mansion offers air-conditioned rooms for Rs 800 per night. "
            "Hotel Supreme is a 5-minute walk from the clinic and offers special "
            "packages for patients at Rs 30,000 for 10 nights including breakfast.",
            tokens=50,
            created_at="2025-01-01T00:00:00Z",
        ),
        VectorData(
            id=2,
            source_url="https://example.com/location",
            text="Our clinic is located in Colaba, Mumbai. The area has many "
            "accommodation options ranging from budget to luxury hotels. "
            "We can help arrange bookings for your stay.",
            tokens=30,
            created_at="2025-01-01T00:00:00Z",
        ),
    ]

    print(f"📝 Chat ID: {conversation.chat_id}")
    print(f"💬 User Question: {conversation.get_latest_user_message().message}\n")

    # Build prompt
    print("🔨 Building prompt...")
    prompt = build_prompt(conversation, context_vectors)

    # Generate response
    print("🤖 Generating LLM response...")
    llm_client = get_llm_client()
    llm_response = llm_client.generate(prompt)

    print(f"✓ Response generated in {llm_response.latency_ms:.0f}ms")
    print(f"📄 Response: {llm_response.text}\n")

    # Initialize evaluators
    print("⚙️  Initializing evaluators...")
    relevance_evaluator = RelevanceEvaluator()
    relevance_evaluator.build_index(context_vectors)

    factual_evaluator = FactualEvaluator()
    factual_evaluator.set_context(context_vectors)

    latency_cost_evaluator = LatencyCostEvaluator()

    # Run evaluations
    print("📊 Running evaluations...\n")

    latest_message = conversation.get_latest_user_message()

    relevance_result = relevance_evaluator.evaluate(llm_response.text, latest_message.message)
    print(f"  Relevance Score: {relevance_result.relevance_score:.3f}")
    print(f"  Completeness Score: {relevance_result.completeness_score:.3f}")

    factual_result = factual_evaluator.evaluate(llm_response.text)
    print(f"  Hallucination Rate: {factual_result.hallucination_rate:.3f}")
    print(f"  Verified Claims: {factual_result.verified_claims}/{factual_result.total_claims}")

    latency_cost_result = latency_cost_evaluator.evaluate(
        llm_response.latency_ms,
        llm_response.token_count,
        llm_response.input_tokens,
        llm_response.output_tokens,
        llm_response.estimated_cost_usd(),
    )
    print(f"  Latency: {latency_cost_result.latency_ms:.0f}ms")
    print(f"  Estimated Cost: ${latency_cost_result.estimated_cost_usd:.6f}")

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

    print(f"\n🎯 Overall Quality Score: {report.overall_quality_score:.3f}")
    print(f"✅ Passed Thresholds: {report.passed_thresholds}")

    print("\n📋 Summary:")
    for item in report.summary:
        print(f"  • {item}")

    print("\n✅ Evaluation complete!")


if __name__ == "__main__":
    main()
