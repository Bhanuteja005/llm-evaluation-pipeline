"""LLM client wrapper supporting mock and real providers."""

import time
from abc import ABC, abstractmethod
from typing import Dict, Optional

from src.config import settings
from src.utils import get_logger

logger = get_logger(__name__)


class LLMResponse:
    """Response from LLM with metadata."""

    def __init__(
        self,
        text: str,
        token_count: int,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        provider: str,
    ):
        self.text = text
        self.token_count = token_count
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.latency_ms = latency_ms
        self.provider = provider

    def estimated_cost_usd(self) -> float:
        """Calculate estimated cost based on token usage."""
        input_cost = self.input_tokens * settings.cost_per_input_token
        output_cost = self.output_tokens * settings.cost_per_output_token
        return input_cost + output_cost

    def to_dict(self) -> Dict[str, any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "token_count": self.token_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "latency_ms": self.latency_ms,
            "provider": self.provider,
            "estimated_cost_usd": self.estimated_cost_usd(),
        }


class LLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> LLMResponse:
        """Generate response from LLM."""
        pass

    def count_tokens(self, text: str) -> int:
        """Count tokens in text (default implementation)."""
        # Simple approximation: ~4 characters per token
        return len(text) // 4


class MockLLMClient(LLMClient):
    """Mock LLM client for testing and offline use."""

    def __init__(self) -> None:
        """Initialize mock client."""
        logger.info("Initialized MockLLMClient")

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate deterministic mock response.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate (ignored in mock)

        Returns:
            LLMResponse with mock data
        """
        start_time = time.perf_counter()

        # Simulate processing delay
        time.sleep(0.1)

        # Generate deterministic response based on prompt
        response_text = self._generate_mock_response(prompt)

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        input_tokens = self.count_tokens(prompt)
        output_tokens = self.count_tokens(response_text)

        logger.debug(f"Mock LLM generated {output_tokens} tokens in {latency_ms:.2f}ms")

        return LLMResponse(
            text=response_text,
            token_count=input_tokens + output_tokens,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            provider="mock",
        )

    def _generate_mock_response(self, prompt: str) -> str:
        """Generate deterministic mock response based on prompt content."""
        prompt_lower = prompt.lower()

        # Pattern matching for common queries
        if "hotel" in prompt_lower or "accommodation" in prompt_lower:
            return (
                "Based on the information provided, I recommend Gopal Mansion "
                "which offers air-conditioned rooms for Rs 800 per night. "
                "You can also consider Hotel Supreme which is a 5-minute walk from the clinic."
            )
        elif "cost" in prompt_lower or "price" in prompt_lower or "ivf" in prompt_lower:
            return (
                "A complete IVF cycle at our clinic costs approximately Rs 3,00,000, "
                "which includes ICSI and blastocyst transfer. Medications typically cost "
                "about Rs 1,45,000 more. Please book a consultation for a personalized quote."
            )
        elif "donor" in prompt_lower and "egg" in prompt_lower:
            return (
                "Donor egg IVF is an excellent option for women with low ovarian reserve. "
                "We use carefully screened donors and frozen eggs from our egg bank with "
                "high success rates. The donor eggs undergo thorough genetic screening."
            )
        else:
            return (
                "Thank you for your question. Based on the context provided, "
                "I recommend scheduling a consultation to discuss your specific situation. "
                "You can book an appointment through our website or call our clinic directly."
            )


class GeminiClient(LLMClient):
    """Google Gemini LLM client."""

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None) -> None:
        """
        Initialize Gemini client.

        Args:
            api_key: Gemini API key (uses settings if not provided)
            model: Model name (uses settings if not provided)
        """
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError(
                "Google Generative AI package not installed. Install with: pip install google-generativeai"
            )

        self.api_key = api_key or settings.gemini_api_key
        self.model_name = model or settings.gemini_model

        if not self.api_key:
            raise ValueError("Gemini API key not provided")

        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

        logger.info(f"Initialized GeminiClient with model: {self.model_name}")

    def generate(self, prompt: str, max_tokens: Optional[int] = None) -> LLMResponse:
        """
        Generate response from Gemini.

        Args:
            prompt: Input prompt
            max_tokens: Max tokens to generate

        Returns:
            LLMResponse with Gemini data
        """
        start_time = time.perf_counter()

        max_tokens = max_tokens or settings.gemini_max_tokens

        try:
            # Configure generation
            generation_config = {
                "max_output_tokens": max_tokens,
                "temperature": 0.7,
            }

            response = self.model.generate_content(
                prompt,
                generation_config=generation_config,
            )

            end_time = time.perf_counter()
            latency_ms = (end_time - start_time) * 1000

            response_text = response.text if hasattr(response, 'text') else ""

            # Estimate token counts (Gemini doesn't provide exact counts in basic API)
            input_tokens = self.count_tokens(prompt)
            output_tokens = self.count_tokens(response_text)
            total_tokens = input_tokens + output_tokens

            logger.debug(f"Gemini generated ~{output_tokens} tokens in {latency_ms:.2f}ms")

            return LLMResponse(
                text=response_text,
                token_count=total_tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                latency_ms=latency_ms,
                provider="gemini",
            )

        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise

    def count_tokens(self, text: str) -> int:
        """Count tokens (approximation for Gemini)."""
        # Approximate: ~4 characters per token
        return len(text) // 4


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """
    Factory function to get appropriate LLM client.

    Args:
        provider: Provider name ('mock' or 'gemini')

    Returns:
        LLMClient instance
    """
    provider = provider or settings.llm_provider

    if provider == "mock":
        return MockLLMClient()
    elif provider == "gemini":
        return GeminiClient()
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
