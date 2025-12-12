"""Tests for ingest module."""

import json
import tempfile
from pathlib import Path

import pytest

from src.ingest import load_context_vectors, load_conversation, load_inputs


def test_load_conversation_valid():
    """Test loading valid conversation JSON."""
    conversation_data = {
        "chat_id": 12345,
        "user_id": 67890,
        "conversation_turns": [
            {
                "turn": 1,
                "sender_id": 1,
                "role": "AI/Chatbot",
                "message": "Hello, how can I help?",
                "created_at": "2025-01-01T10:00:00.000000Z",
            },
            {
                "turn": 2,
                "sender_id": 67890,
                "role": "User",
                "message": "I need help with IVF",
                "created_at": "2025-01-01T10:01:00.000000Z",
            },
        ],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(conversation_data, f)
        temp_path = f.name

    try:
        conversation = load_conversation(temp_path)
        assert conversation.chat_id == 12345
        assert conversation.user_id == 67890
        assert len(conversation.conversation_turns) == 2
        assert conversation.conversation_turns[0].role == "AI/Chatbot"
    finally:
        Path(temp_path).unlink()


def test_load_conversation_missing_file():
    """Test loading non-existent file."""
    with pytest.raises(FileNotFoundError):
        load_conversation("nonexistent.json")


def test_load_context_vectors_valid():
    """Test loading valid context vectors JSON."""
    context_data = {
        "status": "success",
        "status_code": 200,
        "message": "Success",
        "data": {
            "vector_data": [
                {
                    "id": 1,
                    "source_url": "https://example.com",
                    "text": "Sample context text",
                    "tokens": 10,
                    "created_at": "2025-01-01T10:00:00.000Z",
                }
            ],
            "sources": {
                "message_id": 123,
                "vector_ids": ["1"],
                "vectors_used": [1],
            },
        },
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(context_data, f)
        temp_path = f.name

    try:
        context_vectors = load_context_vectors(temp_path)
        assert context_vectors.status == "success"
        assert context_vectors.status_code == 200
        vector_data = context_vectors.get_vector_data()
        assert len(vector_data) == 1
        assert vector_data[0].id == 1
    finally:
        Path(temp_path).unlink()


def test_get_latest_user_message():
    """Test extracting latest user message."""
    from src.models import Conversation, ConversationTurn

    conversation = Conversation(
        chat_id=1,
        user_id=2,
        conversation_turns=[
            ConversationTurn(
                turn=1,
                sender_id=1,
                role="AI/Chatbot",
                message="Hi",
                created_at="2025-01-01T10:00:00Z",
            ),
            ConversationTurn(
                turn=2,
                sender_id=2,
                role="User",
                message="Hello",
                created_at="2025-01-01T10:01:00Z",
            ),
            ConversationTurn(
                turn=3,
                sender_id=2,
                role="User",
                message="Can you help?",
                created_at="2025-01-01T10:02:00Z",
            ),
        ],
    )

    latest = conversation.get_latest_user_message()
    assert latest is not None
    assert latest.message == "Can you help?"
    assert latest.turn == 3
