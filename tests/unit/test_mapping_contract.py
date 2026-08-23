from __future__ import annotations

import pytest
import scripts.render_report as render_report_module

import compute_cost_encoders_llms.benchmark.reporting as reporting_module


def test_mapping_field_is_shared_with_explicit_optional_and_required_modes() -> None:
    mapping_field = reporting_module._mapping_field

    assert mapping_field is render_report_module._mapping_field
    assert mapping_field({"nested": {"value": 1}}, "nested") == {"value": 1}
    assert mapping_field({}, "nested") == {}
    assert mapping_field({"nested": None}, "nested") == {}

    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: nested$",
    ):
        render_report_module._mapping_field(
            {"nested": None},
            "nested",
            required=True,
            error_context="merged artifact",
        )
