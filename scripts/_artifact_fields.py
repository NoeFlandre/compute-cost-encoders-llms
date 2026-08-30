from __future__ import annotations

from collections.abc import Mapping

from compute_cost_encoders_llms.benchmark._mappings import _mapping_field


def _mapping_value(document: Mapping[str, object], field: str) -> Mapping[str, object]:
    return _mapping_field(
        document,
        field,
        required=True,
        error_context="merged artifact",
    )


def _text_value(document: Mapping[str, object], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"merged artifact field is not text: {field}")
    return value


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"merged artifact field is not an object: {field}")
    return value
