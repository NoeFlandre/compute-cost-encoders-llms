from __future__ import annotations

from pathlib import Path

import scripts.render_report as report_module


def test_report_script_does_not_expose_unrelated_mapping_helper() -> None:
    assert not hasattr(report_module, "_mapping_field")


def test_report_script_imports_latex_from_the_owning_module() -> None:
    source = Path(report_module.__file__).read_text(encoding="utf-8")

    assert (
        "from compute_cost_encoders_llms.benchmark.latex import render_latex_document"
    ) in source
    assert (
        "from compute_cost_encoders_llms.benchmark.reporting import "
        "render_latex_document"
    ) not in source
