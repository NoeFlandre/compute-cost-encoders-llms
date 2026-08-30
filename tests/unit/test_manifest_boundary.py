from __future__ import annotations

import compute_cost_encoders_llms.benchmark.cli as cli_module
import compute_cost_encoders_llms.benchmark.manifest as manifest_module


def test_cli_manifest_builder_is_owned_by_manifest_module() -> None:
    assert cli_module.build_manifest is manifest_module.build_manifest
