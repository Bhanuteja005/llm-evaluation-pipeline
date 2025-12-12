"""Claim extraction for factual verification."""

import re
from typing import List, Optional, Set

from src.utils import get_logger

logger = get_logger(__name__)


class Claim:
    """Represents an extracted factual claim."""

    def __init__(self, text: str, claim_type: str, span: tuple[int, int]):
        self.text = text
        self.claim_type = claim_type  # e.g., 'price', 'location', 'fact', 'number'
        self.span = span  # Start and end positions in original text

    def __repr__(self) -> str:
        return f"Claim(type={self.claim_type}, text='{self.text}')"


class ClaimExtractor:
    """Extracts factual claims from text for verification."""

    def __init__(self) -> None:
        """Initialize claim extractor."""
        # Patterns for different claim types
        self.patterns = {
            "price": [
                r"(?:Rs\.?|₹)\s*[\d,]+(?:\.\d{2})?",  # Indian Rupees
                r"\$\s*[\d,]+(?:\.\d{2})?",  # US Dollars
                r"(?:costs?|price[ds]?)\s+(?:approximately|about|around)?\s*(?:Rs\.?|₹|\$)\s*[\d,]+",
            ],
            "phone": [
                r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{2,3}\)?[-.\s]?\d{3,4}[-.\s]?\d{4}",  # Phone numbers
            ],
            "email": [
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email addresses
            ],
            "url": [
                r"https?://[^\s]+",  # URLs
                r"www\.[^\s]+",
            ],
            "percentage": [
                r"\d+(?:\.\d+)?%",  # Percentages
                r"\d+(?:\.\d+)?\s*percent",
            ],
            "measurement": [
                r"\d+(?:\.\d+)?\s*(?:mg|ml|kg|cm|mm|m|km|hrs?|mins?|days?|weeks?|months?|years?)",
            ],
            "date": [
                r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b",  # Dates
                r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
            ],
            "time": [
                r"\b\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?\b",  # Time
            ],
        }

        # Compile patterns
        self.compiled_patterns = {
            claim_type: [re.compile(pattern, re.IGNORECASE) for pattern in patterns]
            for claim_type, patterns in self.patterns.items()
        }

    def extract_claims(self, text: str, min_claim_length: int = 10) -> List[Claim]:
        """
        Extract factual claims from text.

        Args:
            text: Input text
            min_claim_length: Minimum length for a claim

        Returns:
            List of extracted claims
        """
        claims: List[Claim] = []
        seen_spans: Set[tuple[int, int]] = set()

        # Extract pattern-based claims
        for claim_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                for match in pattern.finditer(text):
                    span = (match.start(), match.end())
                    if span not in seen_spans:
                        claims.append(Claim(match.group(), claim_type, span))
                        seen_spans.add(span)

        # Extract sentence-level factual claims
        sentences = self._split_sentences(text)
        for sentence in sentences:
            if len(sentence) >= min_claim_length:
                # Check if sentence contains factual indicators
                if self._is_factual_sentence(sentence):
                    # Find position in original text
                    start = text.find(sentence)
                    if start != -1:
                        span = (start, start + len(sentence))
                        if span not in seen_spans:
                            claims.append(Claim(sentence, "fact", span))
                            seen_spans.add(span)

        logger.debug(f"Extracted {len(claims)} claims from text")
        return claims

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences (simple implementation)."""
        # Simple sentence splitting
        sentences = re.split(r"[.!?]+", text)
        return [s.strip() for s in sentences if s.strip()]

    def _is_factual_sentence(self, sentence: str) -> bool:
        """
        Determine if a sentence contains factual claims.

        A sentence is considered factual if it:
        - Contains specific numbers, names, or measurements
        - Makes definitive statements
        - Contains references to specific entities
        """
        sentence_lower = sentence.lower()

        # Factual indicators
        factual_indicators = [
            r"\b(?:is|are|was|were|costs?|located|offers?|provides?|includes?)\b",
            r"\b(?:doctor|dr\.?|clinic|hospital|treatment|procedure)\b",
            r"\b(?:available|required|recommended|necessary)\b",
            r"\d+",  # Contains numbers
        ]

        # Check for indicators
        for indicator in factual_indicators:
            if re.search(indicator, sentence_lower):
                # Exclude questions and uncertain statements
                if not re.search(r"\?|maybe|perhaps|possibly|might|could|would", sentence_lower):
                    return True

        return False

    def extract_named_entities_simple(self, text: str) -> List[Claim]:
        """
        Simple named entity extraction without spaCy dependency.

        Extracts:
        - Capitalized sequences (potential names/places)
        - Common entity patterns
        """
        claims = []

        # Pattern for capitalized words (potential proper nouns)
        capitalized_pattern = r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b"
        for match in re.finditer(capitalized_pattern, text):
            # Filter out sentence starters
            if match.start() == 0 or text[match.start() - 1] in ".!?":
                continue
            span = (match.start(), match.end())
            claims.append(Claim(match.group(), "entity", span))

        return claims
