"""Tests for validation utilities."""

from src.validation import is_non_empty


def test_is_non_empty_returns_true_for_text() -> None:
    """Non-empty text should pass validation."""
    assert is_non_empty("dataset")


def test_is_non_empty_returns_false_for_whitespace() -> None:
    """Whitespace-only text should fail validation."""
    assert not is_non_empty("   ")