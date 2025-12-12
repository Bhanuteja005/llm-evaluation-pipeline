"""Prompt builder for constructing LLM prompts from conversation and context."""

from typing import List

from src.config import settings
from src.models import Conversation, ConversationTurn, VectorData


def build_prompt(
    conversation: Conversation,
    context_vectors: List[VectorData],
    max_context_length: int = 2000,
) -> str:
    """
    Build a prompt combining conversation history and relevant context.

    Args:
        conversation: The conversation object
        context_vectors: List of relevant context vectors
        max_context_length: Maximum characters for context section

    Returns:
        Formatted prompt string
    """
    # Get latest user message
    latest_user_message = conversation.get_latest_user_message()
    if not latest_user_message:
        raise ValueError("No user message found in conversation")

    # Get conversation history (last few turns)
    history = conversation.get_conversation_history(max_turns=5)

    # Build conversation context
    conversation_context = _build_conversation_context(history)

    # Build knowledge context from vectors
    knowledge_context = _build_knowledge_context(context_vectors, max_context_length)

    # Construct final prompt
    prompt = f"""You are a helpful AI assistant for a medical clinic specializing in infertility treatment and IVF.

CONVERSATION HISTORY:
{conversation_context}

RELEVANT KNOWLEDGE BASE:
{knowledge_context}

CURRENT USER QUESTION:
{latest_user_message.message}

Please provide a helpful, accurate, and empathetic response based on the knowledge base provided. If the knowledge base doesn't contain relevant information, politely suggest booking a consultation or contacting the clinic directly.

RESPONSE:"""

    return prompt


def _build_conversation_context(turns: List[ConversationTurn]) -> str:
    """
    Build formatted conversation history.

    Args:
        turns: List of conversation turns

    Returns:
        Formatted conversation string
    """
    context_lines = []
    for turn in turns:
        role = "User" if turn.role.lower() == "user" else "Assistant"
        # Truncate long messages
        message = turn.message[:500] + "..." if len(turn.message) > 500 else turn.message
        context_lines.append(f"[{role}]: {message}")

    return "\n".join(context_lines)


def _build_knowledge_context(vectors: List[VectorData], max_length: int = 2000) -> str:
    """
    Build formatted knowledge base context from vectors.

    Args:
        vectors: List of context vectors
        max_length: Maximum total characters

    Returns:
        Formatted knowledge context
    """
    if not vectors:
        return "[No relevant knowledge base articles found]"

    context_parts = []
    current_length = 0

    for i, vector in enumerate(vectors, 1):
        # Format each context piece
        header = f"\n[Context {i} - Source: {vector.source_url}]\n"
        text = vector.text

        # Calculate length
        piece_length = len(header) + len(text)

        # Check if adding this would exceed max length
        if current_length + piece_length > max_length and context_parts:
            break

        context_parts.append(header + text)
        current_length += piece_length

    return "\n".join(context_parts)


def build_simple_prompt(user_message: str, context_texts: List[str]) -> str:
    """
    Build a simpler prompt for basic use cases.

    Args:
        user_message: The user's question
        context_texts: List of context text snippets

    Returns:
        Formatted prompt
    """
    context = "\n\n".join(context_texts[:settings.top_k_contexts])

    prompt = f"""Given the following context information:

{context}

Please answer this question:
{user_message}

Provide a helpful and accurate answer based on the context."""

    return prompt
