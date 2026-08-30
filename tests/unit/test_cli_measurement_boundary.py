from __future__ import annotations

import compute_cost_encoders_llms.benchmark.cli as cli_module
from compute_cost_encoders_llms.benchmark.encoder import EncoderScore
from compute_cost_encoders_llms.benchmark.measurement import score_record


def test_measure_records_excludes_warmups_and_indexes_results() -> None:
    score = EncoderScore(
        logprobs={"yes": -0.1, "no": -2.2},
        tokenization_ms=1.0,
        model_ms=2.0,
        logprob_ms=0.1,
        text_to_logprob_ms=3.1,
        input_tokens=12,
    )
    calls = 0

    def operation() -> EncoderScore:
        nonlocal calls
        calls += 1
        return score

    records = cli_module._measure_records(
        "encoder",
        operation,
        warmups=1,
        repetitions=2,
    )

    assert calls == 3
    assert records == [
        score_record("encoder", 0, score),
        score_record("encoder", 1, score),
    ]
