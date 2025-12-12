"""Data models for conversation and context inputs."""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ConversationTurn(BaseModel):
    """A single turn in a conversation."""

    turn: int
    sender_id: int
    role: str
    message: str
    created_at: str

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        """Validate datetime string."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {v}") from e
        return v


class Conversation(BaseModel):
    """A complete conversation with multiple turns."""

    chat_id: int
    user_id: int
    conversation_turns: List[ConversationTurn]

    def get_latest_user_message(self) -> Optional[ConversationTurn]:
        """Get the most recent user message."""
        user_turns = [turn for turn in self.conversation_turns if turn.role.lower() == "user"]
        return user_turns[-1] if user_turns else None

    def get_conversation_history(self, max_turns: int = 10) -> List[ConversationTurn]:
        """Get recent conversation history."""
        return self.conversation_turns[-max_turns:]


class VectorData(BaseModel):
    """A single context vector with metadata."""

    id: int
    source_url: Optional[str] = None  # Some items may not have source_url
    text: Optional[str] = None  # Some items may not have text
    tokens: int
    created_at: str
    source_type: Optional[int] = None

    @field_validator("created_at")
    @classmethod
    def validate_datetime(cls, v: str) -> str:
        """Validate datetime string."""
        try:
            datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError as e:
            raise ValueError(f"Invalid datetime format: {v}") from e
        return v


class ContextSources(BaseModel):
    """Metadata about context sources used."""

    message_id: int
    vector_ids: List[str]
    vectors_info: List[Any] = Field(default_factory=list)
    vectors_used: List[int]
    final_response: List[str] = Field(default_factory=list)


class ContextVectors(BaseModel):
    """Complete context vectors data."""

    status: str
    status_code: int
    message: str
    data: Dict[str, Any]

    def get_vector_data(self) -> List[VectorData]:
        """Extract and validate vector data."""
        vector_data_raw = self.data.get("vector_data", [])
        # Filter out items without text
        return [
            VectorData(**item)
            for item in vector_data_raw
            if item.get("text")
        ]

    def get_sources(self) -> Optional[ContextSources]:
        """Extract sources metadata."""
        sources_raw = self.data.get("sources")
        if sources_raw:
            return ContextSources(**sources_raw)
        return None
