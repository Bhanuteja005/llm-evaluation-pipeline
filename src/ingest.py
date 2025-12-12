"""Input JSON file ingestion and validation."""

import json
from pathlib import Path
from typing import Tuple

from src.models import Conversation, ContextVectors


def load_json_file(file_path: str) -> dict:
    """
    Load and parse a JSON file.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_conversation(file_path: str) -> Conversation:
    """
    Load and validate a conversation JSON file.

    Args:
        file_path: Path to the conversation JSON file

    Returns:
        Validated Conversation object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON structure is invalid
    """
    data = load_json_file(file_path)
    try:
        return Conversation(**data)
    except Exception as e:
        raise ValueError(f"Invalid conversation JSON structure: {e}") from e


def load_context_vectors(file_path: str) -> ContextVectors:
    """
    Load and validate a context vectors JSON file.

    Args:
        file_path: Path to the context vectors JSON file

    Returns:
        Validated ContextVectors object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If JSON structure is invalid
    """
    data = load_json_file(file_path)
    try:
        return ContextVectors(**data)
    except Exception as e:
        raise ValueError(f"Invalid context vectors JSON structure: {e}") from e


def load_inputs(conversation_path: str, context_path: str) -> Tuple[Conversation, ContextVectors]:
    """
    Load both input files.

    Args:
        conversation_path: Path to conversation JSON
        context_path: Path to context vectors JSON

    Returns:
        Tuple of (Conversation, ContextVectors)
    """
    conversation = load_conversation(conversation_path)
    context_vectors = load_context_vectors(context_path)
    return conversation, context_vectors
