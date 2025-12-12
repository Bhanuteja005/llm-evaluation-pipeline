"""Utility functions for the evaluation pipeline."""

import hashlib
import logging
import time
from functools import wraps
from typing import Any, Callable, List, TypeVar

import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


def timeit(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to measure function execution time."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug(f"{func.__name__} took {(end - start) * 1000:.2f}ms")
        return result

    return wrapper


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors.

    Args:
        vec1: First vector
        vec2: Second vector

    Returns:
        Cosine similarity score (0 to 1)
    """
    if vec1.ndim == 1:
        vec1 = vec1.reshape(1, -1)
    if vec2.ndim == 1:
        vec2 = vec2.reshape(1, -1)

    dot_product = np.dot(vec1, vec2.T)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    similarity = dot_product / (norm1 * norm2)
    return float(np.clip(similarity, 0, 1))


def compute_batch_cosine_similarity(
    query_vector: np.ndarray, context_vectors: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between a query vector and multiple context vectors.

    Args:
        query_vector: Query vector (1D or 2D)
        context_vectors: Context vectors (2D array)

    Returns:
        Array of similarity scores
    """
    if query_vector.ndim == 1:
        query_vector = query_vector.reshape(1, -1)

    # Normalize vectors
    query_norm = query_vector / np.linalg.norm(query_vector, axis=1, keepdims=True)
    context_norm = context_vectors / np.linalg.norm(context_vectors, axis=1, keepdims=True)

    # Compute dot product
    similarities = np.dot(query_norm, context_norm.T).flatten()

    return np.clip(similarities, 0, 1)


def normalize_text(text: str) -> str:
    """
    Normalize text for comparison.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    return " ".join(text.lower().strip().split())


def compute_text_hash(text: str) -> str:
    """
    Compute SHA-256 hash of text.

    Args:
        text: Input text

    Returns:
        Hex digest of hash
    """
    return hashlib.sha256(text.encode()).hexdigest()


def extract_keywords(text: str, top_n: int = 10) -> List[str]:
    """
    Extract top keywords from text using simple frequency.

    Args:
        text: Input text
        top_n: Number of keywords to extract

    Returns:
        List of keywords
    """
    # Simple word frequency (production would use TF-IDF)
    words = normalize_text(text).split()
    # Filter out common stop words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "is",
        "are",
        "was",
        "were",
        "been",
        "be",
        "have",
        "has",
        "had",
    }
    filtered = [w for w in words if w not in stop_words and len(w) > 3]

    # Count frequency
    freq: dict[str, int] = {}
    for word in filtered:
        freq[word] = freq.get(word, 0) + 1

    # Sort by frequency
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]


def clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp a value between min and max."""
    return max(min_val, min(max_val, value))
