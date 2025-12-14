# LLM Evaluation Pipeline

A production-grade evaluation pipeline for assessing LLM responses across three key dimensions:
1. **Relevance & Completeness** - Using semantic similarity with context
2. **Factual Accuracy** - Detecting hallucinations through claim verification
3. **Latency & Cost** - Performance and efficiency metrics

## Features

- 🎯 **Comprehensive Evaluation**: Multi-dimensional scoring with detailed explanations
- 🔍 **Semantic Search**: FAISS-based vector similarity for relevance checking
- 🧠 **Claim Extraction**: Automated factual claim detection and verification
- 📊 **SQLite Persistence**: Store and query evaluation results
- 🎨 **Rich CLI**: Beautiful terminal output with tables and panels
- 🐳 **Docker Support**: Containerized deployment ready
- 🧪 **Well-Tested**: Comprehensive test suite included
- 🔌 **Pluggable LLMs**: Mock or real LLM providers (OpenAI)

## Architecture

```
Input JSONs → Ingest → Prompt Builder → LLM Client → Evaluators → Scorer → Persistence
                                                          ↓
                                                    ┌──────────┐
                                                    │Relevance │
                                                    │Factual   │
                                                    │Latency   │
                                                    └──────────┘
```


## Quick Start

### Usage

```bash


# Run evaluation
python -m src.cli evaluate \
  -c samples/sample-chat-conversation-01.json \
  -x samples/sample_context_vectors-01.json \
  -o results.json
```

### View Statistics

```bash
# Overall statistics
python -m src.cli stats

# Statistics for specific chat
python -m src.cli stats --chat-id 78128
```

## Input Format

### Conversation JSON

```json
{
  "chat_id": 78128,
  "user_id": 77096,
  "conversation_turns": [
    {
      "turn": 1,
      "sender_id": 1,
      "role": "AI/Chatbot",
      "message": "How can I help?",
      "created_at": "2025-01-01T10:00:00.000000Z"
    },
    {
      "turn": 2,
      "sender_id": 77096,
      "role": "User",
      "message": "What is the cost of IVF?",
      "created_at": "2025-01-01T10:01:00.000000Z"
    }
  ]
}
```

### Context Vectors JSON

```json
{
  "status": "success",
  "status_code": 200,
  "message": "Success",
  "data": {
    "vector_data": [
      {
        "id": 1,
        "source_url": "https://example.com/article",
        "text": "IVF costs approximately Rs 3,00,000...",
        "tokens": 50,
        "created_at": "2024-01-01T00:00:00.000Z"
      }
    ]
  }
}
```

## Output Format

```json
{
  "metadata": {
    "chat_id": 78128,
    "turn": 2,
    "timestamp": "2025-12-12T10:00:00.000000Z",
    "provider": "mock"
  },
  "metrics": {
    "relevance": {
      "relevance_score": 0.856,
      "completeness_score": 0.724,
      "top_k_context_ids": [1, 2, 3]
    },
    "factual_accuracy": {
      "hallucination_rate": 0.125,
      "verified_claims": 7,
      "total_claims": 8
    },
    "latency_cost": {
      "latency_ms": 245.3,
      "estimated_cost_usd": 0.000543
    }
  },
  "aggregate": {
    "overall_quality_score": 0.782,
    "passed_thresholds": true
  }
}
```

## Project Structure

```
llm-eval-pipeline/
├── src/
│   ├── cli.py              # CLI entrypoint
│   ├── config.py           # Configuration
│   ├── models.py           # Pydantic models
│   ├── ingest.py           # Input loading
│   ├── llm_client.py       # LLM wrappers
│   ├── prompt_builder.py   # Prompt construction
│   ├── extractors.py       # Claim extraction
│   ├── evaluators/
│   │   ├── relevance.py    # Relevance evaluation
│   │   ├── factual.py      # Factual checking
│   │   └── latency_cost.py # Performance metrics
│   ├── scoring.py          # Score aggregation
│   ├── persistence.py      # Database storage
│   └── utils.py            # Utilities
├── tests/
│   ├── test_ingest.py
│   ├── test_relevance.py
│   ├── test_factual.py
│   └── test_integration.py
├── samples/                # Sample input files
├── requirements.txt
├── Dockerfile
└── README.md
```
