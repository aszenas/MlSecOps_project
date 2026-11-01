"""Basic validation utilities."""


def is_non_empty(value: str) -> bool:
    """Return whether a string contains a non-whitespace value."""
    return bool(value.strip())