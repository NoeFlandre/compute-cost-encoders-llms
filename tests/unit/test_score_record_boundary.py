from __future__ import annotations

import compute_cost_encoders_llms.benchmark.cli as cli_module
import compute_cost_encoders_llms.benchmark.measurement as measurement_module


def test_cli_score_record_is_owned_by_measurement_module() -> None:
    assert cli_module.score_record is measurement_module.score_record
