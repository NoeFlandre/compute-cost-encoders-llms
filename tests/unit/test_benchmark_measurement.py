from __future__ import annotations

import json
import locale
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

import pytest
from scripts.render_report import merge_artifacts

import compute_cost_encoders_llms.benchmark.measurement as measurement_module
import compute_cost_encoders_llms.benchmark.reporting as reporting_module
from compute_cost_encoders_llms.benchmark.encoder import EncoderScore, _variant_logprob
from compute_cost_encoders_llms.benchmark.latex import (
    _comparison_section,
    _count_text,
    _decision_text,
    _model_summaries,
)
from compute_cost_encoders_llms.benchmark.measurement import (
    MeasurementError,
    MeasurementRecord,
    _required_float,
    choose_decision,
    measure_repetitions,
    score_record,
    summarize_latencies,
    validate_measurement,
)
from compute_cost_encoders_llms.benchmark.reporting import (
    ModelSummary,
    build_summary,
    json_line,
    render_latex_document,
    render_latex_summary,
    write_json,
    write_jsonl,
)


def test_summarize_latencies_returns_deterministic_quantiles() -> None:
    summary = summarize_latencies([1.0, 2.0, 3.0, 4.0])

    assert summary["count"] == 4
    assert summary["median"] == pytest.approx(2.5)
    assert summary["p05"] == pytest.approx(1.15)
    assert summary["p95"] == pytest.approx(3.85)
    assert summarize_latencies([1.0, 3.0])["stdev"] == pytest.approx(2**0.5)


def test_summarize_latencies_handles_single_and_invalid_samples() -> None:
    assert summarize_latencies([2.0])["stdev"] == 0.0
    assert summarize_latencies([0.0])["minimum"] == 0.0
    with pytest.raises(
        MeasurementError,
        match=r"^latencies must contain finite non-negative values$",
    ):
        summarize_latencies([])
    with pytest.raises(
        MeasurementError,
        match=r"^latencies must contain finite non-negative values$",
    ):
        summarize_latencies([-1.0])


def test_score_record_normalizes_backend_score() -> None:
    score = EncoderScore(
        logprobs={"yes": -0.1, "no": -2.2},
        tokenization_ms=1.0,
        model_ms=2.0,
        logprob_ms=0.1,
        text_to_logprob_ms=3.1,
        input_tokens=12,
    )

    record = score_record("encoder", 3, score)

    assert record["model"] == "encoder"
    assert record["repetition"] == 3
    assert record["input_tokens"] == 12
    assert record["logprobs"] == {"yes": -0.1, "no": -2.2}
    assert set(record) == {
        "model",
        "repetition",
        "tokenization_ms",
        "model_ms",
        "logprob_ms",
        "text_to_logprob_ms",
        "input_tokens",
        "logprobs",
        "decision",
    }
    assert record["decision"] == "yes"


def test_model_summary_uses_one_shared_candidate_label_order(monkeypatch) -> None:
    calls: list[str] = []

    def labels() -> tuple[str, str]:
        calls.append("called")
        return ("yes", "no")

    monkeypatch.setattr(reporting_module, "candidate_labels", labels, raising=False)
    record = cast(
        MeasurementRecord,
        {
            "model": "encoder",
            "repetition": 0,
            "tokenization_ms": 1.0,
            "model_ms": 2.0,
            "logprob_ms": 0.1,
            "text_to_logprob_ms": 3.1,
            "logprobs": {"yes": -0.1, "no": -2.2},
        },
    )

    summary = reporting_module._model_summary("encoder", [record])

    assert calls == ["called"]
    assert summary["mean_logprobs"] == {"yes": -0.1, "no": -2.2}
    assert summary["decision_counts"] == {"yes": 1, "no": 0}


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
    assert validate_measurement({**record, "model_ms": 0.0})["model_ms"] == 0.0

    with pytest.raises(
        MeasurementError,
        match=r"^model_ms must be non-negative$",
    ):
        validate_measurement({**record, "model_ms": -1.0})
    with pytest.raises(MeasurementError, match=r"^model must be non-empty$"):
        validate_measurement({**record, "model": ""})
    with pytest.raises(
        MeasurementError,
        match=r"^repetition must be non-negative$",
    ):
        validate_measurement({**record, "repetition": -1})
    with pytest.raises(
        MeasurementError,
        match=r"^logprobs must contain yes and no$",
    ):
        validate_measurement({**record, "logprobs": {"yes": -1.0}})
    with pytest.raises(
        MeasurementError,
        match=r"^missing measurement field: model$",
    ):
        validate_measurement({})


def test_logprob_validation_helpers_preserve_binary_finite_contract() -> None:
    has_binary_keys = getattr(measurement_module, "_has_binary_logprob_keys", None)
    has_finite = getattr(measurement_module, "_has_finite_binary_logprobs", None)
    assert callable(has_binary_keys)
    assert callable(has_finite)

    assert has_binary_keys({"yes": -0.1, "no": -2.2}) is True
    assert has_binary_keys({"yes": -0.1}) is False
    assert has_binary_keys(["yes", "no"]) is False
    assert has_finite({"yes": -0.1, "no": -2.2}) is True
    assert has_finite({"yes": float("nan"), "no": -2.2}) is False
    assert has_finite({"yes": "not a number", "no": -2.2}) is False


def test_measurement_validation_uses_shared_candidate_labels(monkeypatch) -> None:
    labels = ("positive", "negative")
    monkeypatch.setattr(measurement_module, "candidate_labels", lambda: labels)
    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"positive": -0.1, "negative": -2.2},
        "decision": "positive",
    }

    assert validate_measurement(record) == record


def test_json_line_is_sorted_and_serializable(tmp_path) -> None:
    assert json_line({"z": 1, "a": 2}) == '{"a":2,"z":1}'
    assert json_line({"é": "é"}) == '{"é":"é"}'

    document_path = tmp_path / "document.json"
    write_json(document_path, {"z": 1, "a": 2})
    assert document_path.read_text() == '{\n  "a": 2,\n  "z": 1\n}\n'

    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"yes": -0.1, "no": -2.2},
    }
    jsonl_path = tmp_path / "records.jsonl"
    write_jsonl(jsonl_path, [record])
    assert jsonl_path.read_text() == json_line(record) + "\n"


def test_validated_reporting_helpers_reuse_one_validation_pass(
    monkeypatch, tmp_path
) -> None:
    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"yes": -0.1, "no": -2.2},
    }
    second_record = {**record, "repetition": 1}
    calls: list[object] = []
    original_validate = reporting_module.validate_measurement

    def capture_validate(value: Mapping[str, object]) -> object:
        calls.append(value)
        return original_validate(value)

    monkeypatch.setattr(reporting_module, "validate_measurement", capture_validate)
    validate_records = getattr(reporting_module, "_validated_records", None)
    build_validated_summary = getattr(
        reporting_module, "_summary_from_validated_records", None
    )
    write_validated_jsonl = getattr(reporting_module, "_write_validated_jsonl", None)
    assert callable(validate_records)
    assert callable(build_validated_summary)
    assert callable(write_validated_jsonl)

    validated = validate_records([record, second_record])
    summary = build_validated_summary(validated)
    output = tmp_path / "measurements.jsonl"
    write_validated_jsonl(output, validated)

    assert len(calls) == 2
    assert summary["models"][0]["model"] == "encoder"
    assert summary["models"][0]["latency"]["count"] == 2
    assert (
        output.read_text()
        == "\n".join((json_line(record), json_line(second_record))) + "\n"
    )
    with pytest.raises(MeasurementError, match=r"^duplicate measurement repetition$"):
        build_validated_summary([validated[0], validated[0]])
    with pytest.raises(MeasurementError, match=r"^no measurements$"):
        build_validated_summary([])


def test_write_measurement_artifacts_validates_once_and_writes_both_outputs(
    monkeypatch, tmp_path
) -> None:
    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"yes": -0.1, "no": -2.2},
    }
    second = cast(MeasurementRecord, {**record, "repetition": 1})
    expected_summary = build_summary([record, second])
    calls: list[object] = []
    original_validate = reporting_module.validate_measurement

    def capture_validate(value: Mapping[str, object]) -> object:
        calls.append(value)
        return original_validate(value)

    monkeypatch.setattr(reporting_module, "validate_measurement", capture_validate)

    reporting_module.write_measurement_artifacts(tmp_path, [record, second])

    assert len(calls) == 2
    assert (tmp_path / "measurements.jsonl").read_text() == "\n".join(
        (json_line(record), json_line(second))
    ) + "\n"
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "measurements.jsonl",
        "summary.json",
    ]
    assert json.loads((tmp_path / "summary.json").read_text()) == expected_summary


def test_validated_reporting_grouping_preserves_identity_and_errors() -> None:
    record = cast(
        MeasurementRecord,
        {
            "model": "encoder",
            "repetition": 0,
            "tokenization_ms": 1.0,
            "model_ms": 2.0,
            "logprob_ms": 0.1,
            "text_to_logprob_ms": 3.1,
            "logprobs": {"yes": -0.1, "no": -2.2},
        },
    )
    second = cast(MeasurementRecord, {**record, "repetition": 1})

    grouped = reporting_module._group_validated_records([record, second])

    assert list(grouped) == ["encoder"]
    assert len(grouped["encoder"]) == 2
    assert reporting_module._summary_from_grouped(grouped)["models"][0]["model"] == (
        "encoder"
    )
    with pytest.raises(MeasurementError, match=r"^duplicate measurement repetition$"):
        reporting_module._group_validated_records([record, record])
    with pytest.raises(MeasurementError, match=r"^no measurements$"):
        reporting_module._group_validated_records([])


def test_reporting_writers_pass_explicit_utf8_encoding() -> None:
    class RecordingPath:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def write_text(self, _content: str, **kwargs: object) -> int:
            self.calls.append(kwargs)
            return 0

    path = RecordingPath()
    write_json(cast(Path, path), {"é": "é"})
    reporting_module._write_validated_jsonl(
        cast(Path, path),
        [
            cast(
                MeasurementRecord,
                {
                    "model": "encoder",
                    "repetition": 0,
                    "tokenization_ms": 1.0,
                    "model_ms": 2.0,
                    "logprob_ms": 0.1,
                    "text_to_logprob_ms": 3.1,
                    "logprobs": {"yes": -0.1, "no": -2.2},
                },
            )
        ],
    )

    assert [call["encoding"] for call in path.calls] == ["utf-8", "utf-8"]


def test_write_json_preserves_unicode_under_ascii_locale(tmp_path) -> None:
    previous_locale = locale.setlocale(locale.LC_CTYPE)
    try:
        locale.setlocale(locale.LC_CTYPE, "C")
        document_path = tmp_path / "unicode.json"
        write_json(document_path, {"é": "é"})
    finally:
        locale.setlocale(locale.LC_CTYPE, previous_locale)

    assert document_path.read_text(encoding="utf-8") == '{\n  "é": "é"\n}\n'


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
    assert choose_decision({"yes": -2.2, "no": -0.1}) == "no"
    with pytest.raises(
        MeasurementError,
        match=r"^logprobs must contain yes and no$",
    ):
        choose_decision({"yes": -1.0})
    with pytest.raises(MeasurementError, match=r"^logprobs must be finite$"):
        choose_decision({"yes": float("nan"), "no": -1.0})


def test_choose_decision_uses_shared_candidate_labels(monkeypatch) -> None:
    calls: list[str] = []

    def labels() -> tuple[str, str]:
        calls.append("called")
        return ("yes", "no")

    monkeypatch.setattr(measurement_module, "candidate_labels", labels, raising=False)

    assert measurement_module.choose_decision({"yes": -1.0, "no": -2.0}) == "yes"
    assert calls == ["called"]


def test_measure_repetitions_excludes_warmups(monkeypatch) -> None:
    calls: list[int] = []

    def operation() -> int:
        calls.append(len(calls))
        return len(calls)

    results = measure_repetitions(operation, warmups=2, repetitions=3)

    assert [result.value for result in results] == [3, 4, 5]
    assert len(calls) == 5
    assert all(result.elapsed_ms >= 0 for result in results)

    assert measure_repetitions(operation, warmups=0, repetitions=1)
    with pytest.raises(
        MeasurementError,
        match=r"^warmups must be non-negative and repetitions positive$",
    ):
        measure_repetitions(operation, warmups=-1, repetitions=1)
    with pytest.raises(
        MeasurementError,
        match=r"^warmups must be non-negative and repetitions positive$",
    ):
        measure_repetitions(operation, warmups=0, repetitions=0)

    ticks = iter((100, 1_000_100))
    monkeypatch.setattr(measurement_module.time, "perf_counter_ns", lambda: next(ticks))
    timed = measure_repetitions(lambda: "value", warmups=0, repetitions=1)
    assert timed[0].elapsed_ms == 1.0


def test_measure_repetitions_resolves_the_clock_once(monkeypatch) -> None:
    class CountingTime:
        def __init__(self) -> None:
            self.accesses = 0
            self.ticks = iter((100, 1_000_100))

        @property
        def perf_counter_ns(self) -> Callable[[], int]:
            self.accesses += 1
            return lambda: next(self.ticks)

    clock = CountingTime()
    monkeypatch.setattr(measurement_module, "time", clock)

    timed = measure_repetitions(lambda: "value", warmups=0, repetitions=1)

    assert timed[0].elapsed_ms == 1.0
    assert clock.accesses == 1


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
    tokenization = summary["models"][0]["tokenization"]
    model_time = summary["models"][0]["model_time"]
    assert tokenization is not None
    assert model_time is not None
    assert tokenization["count"] == 2
    assert model_time["count"] == 2
    assert summary["models"][0]["logprob_time"]["count"] == 2
    assert summary["models"][0]["mean_logprobs"] == {
        "yes": pytest.approx(-0.15),
        "no": pytest.approx(-2.1),
    }
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
            "example": {
                "sentence": "A park is land use.",
                "question": "Is this sentence relevant for a land use description?",
                "labels": ["yes", "no"],
            },
        },
        {
            "models": [
                {
                    "model": "encoder",
                    "latency": {"count": 1, "median": 1.2, "p05": 1.2, "p95": 1.2},
                    "mean_logprobs": {"yes": -0.2, "no": -1.4},
                    "decision_counts": {"yes": 1, "no": 0},
                },
                {
                    "model": "llm",
                    "latency": {"count": 1, "median": 2.4, "p05": 2.4, "p95": 2.4},
                    "mean_logprobs": {"yes": -0.3, "no": -1.1},
                    "decision_counts": {"yes": 1, "no": 0},
                },
            ]
        },
    )

    assert "Binary Land-Use Logprob Benchmark" in document
    assert "jhu-clsp/mmBERT-base" in document
    assert "a" * 40 in document
    assert "A park is land use." in document
    assert "Is this sentence relevant for a land use description?" in document
    assert "1.200" in document
    assert "mean log" in document
    assert "-0.200" in document
    assert "Timing decomposition" in document
    assert "Comparison" in document
    assert "Reproducibility" in document
    assert "Interpretation" not in document
    assert "without fine-tuning or multi" not in document
    assert document.index("Reproducibility") > document.index("Comparison")


def test_report_helpers_fail_closed_on_incomplete_values() -> None:
    assert _model_summaries({}) == []
    assert _model_summaries({"models": [None]}) == []
    assert _count_text(0) == "0"
    assert _count_text(-1) == "--"
    assert _count_text(True) == "--"
    assert _decision_text({"yes": 0, "no": 1}) == "no"
    assert _decision_text({"yes": 1}) == "--"
    encoder = {
        "model": "encoder",
        "latency": {"median": 1.0},
        "mean_logprobs": {"yes": -1.0, "no": -2.0},
        "decision_counts": {"yes": 1, "no": 0},
    }
    llm = {**encoder, "model": "llm", "latency": {"median": "missing"}}
    encoder_summary = cast(ModelSummary, encoder)
    llm_summary = cast(ModelSummary, llm)
    assert _comparison_section([encoder_summary]) == []
    assert _comparison_section([encoder_summary, llm_summary]) == [
        "\\section*{Comparison}",
        "Comparison unavailable.",
    ]
    with pytest.raises(ValueError, match="must not be empty"):
        _variant_logprob([0.0], (), 0.0)


def test_merge_artifacts_requires_matching_source_commits() -> None:
    manifest = {
        "source_commit": "a" * 40,
        "example": {"sentence": "x"},
        "run_id": "encoder-run",
    }
    llm_manifest = {**manifest, "run_id": "llm-run"}
    encoder = {
        "models": [
            {
                "model": "encoder",
                "latency": {
                    "count": 1,
                    "minimum": 1.0,
                    "median": 1.0,
                    "p05": 1.0,
                    "p95": 1.0,
                    "maximum": 1.0,
                    "mean": 1.0,
                    "stdev": 0.0,
                },
                "mean_logprobs": {"yes": -0.1, "no": -2.2},
                "decision_counts": {"yes": 1, "no": 0},
            }
        ]
    }
    llm = {"models": [{**encoder["models"][0], "model": "llm"}]}

    merged = merge_artifacts(manifest, llm_manifest, encoder, llm)

    summary = cast(dict[str, object], merged["summary"])
    models = cast(list[dict[str, object]], summary["models"])
    assert [model["model"] for model in models] == ["encoder", "llm"]
    merged_manifest = cast(dict[str, object], merged["manifest"])
    assert merged_manifest["backend"] == "both"
    assert merged_manifest["run_ids"] == ["encoder-run", "llm-run"]

    with pytest.raises(ValueError, match="source commit"):
        merge_artifacts(
            manifest,
            {**llm_manifest, "source_commit": "b" * 40},
            encoder,
            llm,
        )


def test_validate_measurement_rejects_nonfinite_scores_and_inconsistent_decisions() -> (
    None
):
    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"yes": -0.1, "no": -2.2},
    }

    with pytest.raises(MeasurementError, match=r"^logprobs must be finite$"):
        validate_measurement({**record, "logprobs": {"yes": float("nan"), "no": -2.2}})
    with pytest.raises(
        MeasurementError,
        match=r"^decision is inconsistent with logprobs$",
    ):
        validate_measurement({**record, "decision": "no"})
    decision_no = {
        **record,
        "logprobs": {"yes": -2.2, "no": -0.1},
        "decision": "no",
    }
    assert validate_measurement(decision_no) == decision_no
    with pytest.raises(
        MeasurementError,
        match=r"^decision must be yes or no$",
    ):
        validate_measurement({**record, "decision": "maybe"})
    with pytest.raises(
        MeasurementError,
        match=r"^input_tokens must be non-negative$",
    ):
        validate_measurement({**record, "input_tokens": True})
    with pytest.raises(
        MeasurementError,
        match=r"^text_to_logprob_ms must cover component timings$",
    ):
        validate_measurement({**record, "model_ms": 4.0})
    with pytest.raises(
        MeasurementError,
        match=r"^text_to_logprob_ms must cover component timings$",
    ):
        validate_measurement({**record, "logprob_ms": 4.0})
    equal_component = {**record, "model_ms": record["text_to_logprob_ms"]}
    assert validate_measurement(equal_component) == equal_component
    with pytest.raises(MeasurementError, match=r"^logprob_ms must be finite$"):
        validate_measurement({**record, "logprob_ms": None})
    with pytest.raises(MeasurementError, match=r"^timing must be finite$"):
        _required_float(float("nan"))


def test_summary_preserves_unmeasured_timing_as_none_and_rejects_empty_input() -> None:
    record = {
        "model": "llm",
        "repetition": 0,
        "tokenization_ms": None,
        "model_ms": None,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"yes": -0.1, "no": -2.2},
    }

    summary = build_summary([record])

    assert summary["models"][0]["tokenization"] is None
    assert summary["models"][0]["model_time"] is None
    with pytest.raises(MeasurementError, match="duplicate"):
        build_summary([record, record])
    with pytest.raises(MeasurementError, match="no measurements"):
        build_summary([])
