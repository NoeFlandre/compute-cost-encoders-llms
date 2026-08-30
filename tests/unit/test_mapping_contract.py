from __future__ import annotations

import pytest

from compute_cost_encoders_llms.benchmark._mappings import _mapping_field


def test_mapping_field_supports_optional_and_required_modes() -> None:
    assert _mapping_field({"nested": {"value": 1}}, "nested") == {"value": 1}
    assert _mapping_field({}, "nested") == {}
    assert _mapping_field({"nested": None}, "nested") == {}

    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: nested$",
    ):
        _mapping_field(
            {"nested": None},
            "nested",
            required=True,
            error_context="merged artifact",
        )


def test_mapping_field_uses_document_as_default_error_context() -> None:
    with pytest.raises(ValueError, match=r"^document field is not an object: value$"):
        _mapping_field({}, "value", required=True)
