"""Test configuration."""

import pytest


@pytest.fixture(scope="session")
def test_database_url():
    """Use in-memory SQLite for tests."""
    return "sqlite:///:memory:"
