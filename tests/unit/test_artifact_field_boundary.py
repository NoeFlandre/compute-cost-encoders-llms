from __future__ import annotations

import pytest
import scripts._artifact_fields as fields_module
import scripts.grid5000.checkpoint_metadata as checkpoint_module
import scripts.render_report as report_module
from scripts._artifact_fields import _as_mapping, _mapping_value, _text_value


def test_shared_artifact_helpers_are_owned_by_neutral_module() -> None:
    for name in ("_as_mapping", "_mapping_value", "_text_value"):
        assert getattr(report_module, name) is getattr(fields_module, name)
        assert getattr(checkpoint_module, name) is getattr(fields_module, name)


def test_artifact_helpers_preserve_merged_artifact_contract() -> None:
    document = {"nested": {"value": 1}, "name": "encoder"}
    assert _mapping_value(document, "nested") == {"value": 1}
    assert _text_value(document, "name") == "encoder"
    assert _as_mapping(document["nested"], "nested") == {"value": 1}

    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: missing$",
    ):
        _mapping_value(document, "missing")
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not text: name$",
    ):
        _text_value({"name": ""}, "name")
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: nested$",
    ):
        _as_mapping(None, "nested")
