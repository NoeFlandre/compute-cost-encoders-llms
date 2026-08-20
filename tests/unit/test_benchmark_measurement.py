from __future__ import annotations

import json
from typing import cast

import pytest
from scripts.render_report import merge_artifacts

from compute_cost_encoders_llms.benchmark.measurement import (
    MeasurementError,
    choose_decision,
    measure_repetitions,
    summarize_latencies,
    validate_measurement,
)
from compute_cost_encoders_llms.benchmark.reporting import (
    build_summary,
    json_line,
    render_latex_document,
    render_latex_summary,
)


def test_summarize_latencies_returns_deterministic_quantiles() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["median"] == pytest.approx(2.5)
    assert summary["p05"] == pytest.approx(1.15)
    assert summary["p95"] == pytest.approx(3.85)


def test_summarize_latencies_handles_single_and_invalid_samples() -> None:
    assert summarize_latencies([2.0])["stdev"] == 0.0
    with pytest.raises(MeasurementError, match="latencies"):
        summarize_latencies([])
    with pytest.raises(MeasurementError, match="latencies"):
        summarize_latencies([-1.0])


def test_validate_measurement_requires_text_to_logprob_timing() -> None:
    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"yes": -0.1, "no": -2.2},
    }

    assert validate_measurement(record) == record

    with pytest.raises(MeasurementError, match="non-negative"):
        validate_measurement({**record, "model_ms": -1.0})
    with pytest.raises(MeasurementError, match="model must be non-empty"):
        validate_measurement({**record, "model": ""})
    with pytest.raises(MeasurementError, match="repetition must be non-negative"):
        validate_measurement({**record, "repetition": -1})
    with pytest.raises(MeasurementError, match="logprobs must contain"):
        validate_measurement({**record, "logprobs": {"yes": -1.0}})


def test_json_line_is_sorted_and_serializable() -> None:
    assert json.loads(json_line({"z": 1, "a": 2})) == {"a": 2, "z": 1}


def test_render_latex_summary_escapes_report_values() -> None:
    rendered = render_latex_summary(
        {"model": "qwen_27b", "median": 12.5, "decision": "yes_no"}
    )

    assert "qwen\\_27b" in rendered
    assert "12.500" in rendered
    assert "yes\\_no" in rendered


def test_choose_decision_uses_stable_label_order_for_ties() -> None:
    assert choose_decision({"yes": -1.0, "no": -1.0}) == "yes"
    assert choose_decision({"yes": -0.1, "no": -1.0}) == "yes"
    with pytest.raises(MeasurementError, match="yes and no"):
        choose_decision({"yes": -1.0})
    with pytest.raises(MeasurementError, match="finite"):
        choose_decision({"yes": float("nan"), "no": -1.0})


def test_measure_repetitions_excludes_warmups() -> None:
    calls: list[int] = []

    def operation() -> int:
        calls.append(len(calls))
        return len(calls)

    results = measure_repetitions(operation, warmups=2, repetitions=3)

    assert [result.value for result in results] == [3, 4, 5]
    assert len(calls) == 5
    assert all(result.elapsed_ms >= 0 for result in results)


def test_build_summary_groups_models_and_counts_decisions() -> None:
    records = [
        {
            "model": "encoder",
            "repetition": 0,
            "tokenization_ms": 1.0,
            "model_ms": 2.0,
            "logprob_ms": 0.1,
            "text_to_logprob_ms": 3.1,
            "logprobs": {"yes": -0.1, "no": -2.2},
        },
        {
            "model": "encoder",
            "repetition": 1,
            "tokenization_ms": 1.1,
            "model_ms": 2.1,
            "logprob_ms": 0.1,
            "text_to_logprob_ms": 3.3,
            "logprobs": {"yes": -0.2, "no": -2.0},
        },
    ]

    summary = build_summary(records)

    assert summary["models"][0]["model"] == "encoder"
    assert summary["models"][0]["latency"]["count"] == 2
    assert summary["models"][0]["decision_counts"] == {"yes": 2, "no": 0}


def test_render_latex_document_contains_revisions_and_results() -> None:
    document = render_latex_document(
        {
            "source_commit": "a" * 40,
            "llama_cpp_revision": "d" * 40,
            "models": {
                "encoder": {"id": "jhu-clsp/mmBERT-base", "revision": "b" * 40},
                "llm": {"id": "ggml-org/Qwen3.6-27B-GGUF", "revision": "c" * 40},
            },
            "protocol": {"warmups": 8, "repetitions": 64, "prompt_cache": False},
            "example": {"sentence": "A park is land use.", "labels": ["yes", "no"]},
        },
        {
            "models": [
                {
                    "model": "encoder",
                    "latency": {"count": 1, "median": 1.2, "p05": 1.2, "p95": 1.2},
                    "mean_logprobs": {"yes": -0.2, "no": -1.4},
                    "decision_counts": {"yes": 1, "no": 0},
                }
            ]
        },
    )

    assert "Binary Land-Use Logprob Benchmark" in document
    assert "jhu-clsp/mmBERT-base" in document
    assert "a" * 40 in document
    assert "A park is land use." in document
    assert "1.200" in document
    assert "mean log" in document
    assert "-0.200" in document


def test_merge_artifacts_requires_matching_source_commits() -> None:
    manifest = {"source_commit": "a" * 40, "example": {"sentence": "x"}}
    encoder = {"models": [{"model": "encoder"}]}
    llm = {"models": [{"model": "llm"}]}

    merged = merge_artifacts(manifest, manifest, encoder, llm)

    summary = cast(dict[str, object], merged["summary"])
    assert summary["models"] == [{"model": "encoder"}, {"model": "llm"}]

    with pytest.raises(ValueError, match="source commit"):
        merge_artifacts(manifest, {"source_commit": "b" * 40}, encoder, llm)
