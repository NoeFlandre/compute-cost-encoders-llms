from __future__ import annotations

from collections.abc import Mapping


def _mapping_field(
    document: Mapping[str, object],
    field: str,
    *,
    required: bool = False,
    error_context: str = "document",
) -> Mapping[str, object]:
    """Return a mapping field, optionally requiring it to be present and typed."""

    value = document.get(field)
    if isinstance(value, Mapping):
        return value
    if required:
        raise ValueError(f"{error_context} field is not an object: {field}")
    return {}
