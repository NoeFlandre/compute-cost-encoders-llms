from __future__ import annotations

import scripts.grid5000.checkpoint_metadata as checkpoint_module
import scripts.render_report as report_module


def test_report_builder_is_owned_by_checkpoint_module() -> None:
    assert (
        report_module.build_checkpoint_metadata
        is checkpoint_module.build_checkpoint_metadata
    )
