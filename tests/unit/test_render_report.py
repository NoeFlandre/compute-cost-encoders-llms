from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import cast

import pytest
import scripts.render_report as report_module
from scripts.render_report import (
    _as_mapping,
    _checkpoint_metrics,
    _non_negative_count,
    _positive_count,
    _read_json,
    _require_binary_keys,
    _validate_decision_counts,
    _validate_latency_order,
    _validate_latency_summary,
    _validate_mean_logprobs,
    _validated_decision_counts,
    _validated_summary_models,
    build_checkpoint_metadata,
    main,
    merge_artifacts,
    render_report,
)


def test_build_checkpoint_metadata_is_publishable_and_reproducible() -> None:
    merged = {
        "manifest": {
            "source_commit": "a" * 40,
            "seed": 7,
            "protocol": {"repetitions": 64},
        },
        "summary": {
            "models": [
                {
                    "model": "encoder",
                    "latency": {"count": 64, "median": 2.5},
                    "decision_counts": {"yes": 64, "no": 0},
                }
            ]
        },
    }

    metadata = build_checkpoint_metadata(
        merged,
        config_revision="sha256:config",
        dataset_revision="made-up-landuse-example-v1",
        model_revision="b" * 40,
        artifact_prefix="//runs/example//",
    )

    assert metadata == {
        "artifact_uri": (
            "hf://buckets/NoeFlandre/compute-cost-encoders-llms/runs/example"
        ),
        "complete": True,
        "config_revision": "sha256:config",
        "dataset_revision": "made-up-landuse-example-v1",
        "metrics": {
            "encoder": {
                "decision_counts": {"no": 0, "yes": 64},
                "median_text_to_logprob_ms": 2.5,
            }
        },
        "model_revision": "b" * 40,
        "seed": 7,
        "source_commit": "a" * 40,
        "step": 64,
    }

    x_prefixed = build_checkpoint_metadata(
        merged,
        config_revision="config",
        dataset_revision="dataset",
        model_revision="model",
        artifact_prefix="XrunsX",
    )
    assert x_prefixed["artifact_uri"] == (
        "hf://buckets/NoeFlandre/compute-cost-encoders-llms/XrunsX"
    )


def test_build_checkpoint_metadata_rejects_empty_artifact_prefix() -> None:
    with pytest.raises(
        ValueError,
        match=r"^artifact prefix must be non-empty and traversal-free$",
    ):
        build_checkpoint_metadata(
            {"manifest": {}, "summary": {"models": []}},
            config_revision="sha256:config",
            dataset_revision="dataset",
            model_revision="model",
            artifact_prefix="",
        )
    with pytest.raises(
        ValueError,
        match=r"^artifact prefix must be non-empty and traversal-free$",
    ):
        build_checkpoint_metadata(
            {"manifest": {}, "summary": {"models": []}},
            config_revision="sha256:config",
            dataset_revision="dataset",
            model_revision="model",
            artifact_prefix="../runs/example",
        )

    valid_shape = {
        "manifest": {
            "source_commit": "a" * 40,
            "seed": 7,
            "protocol": {"repetitions": 2},
        },
        "summary": {"models": []},
    }
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not a non-negative integer: seed$",
    ):
        build_checkpoint_metadata(
            {**valid_shape, "manifest": {**valid_shape["manifest"], "seed": -1}},
            config_revision="sha256:config",
            dataset_revision="dataset",
            model_revision="model",
            artifact_prefix="runs/example",
        )
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not a non-negative integer: seed$",
    ):
        build_checkpoint_metadata(
            {**valid_shape, "manifest": {**valid_shape["manifest"], "seed": True}},
            config_revision="sha256:config",
            dataset_revision="dataset",
            model_revision="model",
            artifact_prefix="runs/example",
        )
    zero_seed = build_checkpoint_metadata(
        {
            **valid_shape,
            "manifest": {**valid_shape["manifest"], "seed": 0},
            "summary": {
                "models": [
                    {
                        "model": "encoder",
                        "latency": {"median": 1.0},
                        "decision_counts": {"yes": 1, "no": 0},
                    }
                ]
            },
        },
        config_revision="sha256:config",
        dataset_revision="dataset",
        model_revision="model",
        artifact_prefix="runs/example",
    )
    assert zero_seed["seed"] == 0
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not text: source_commit$",
    ):
        build_checkpoint_metadata(
            {
                **valid_shape,
                "manifest": {**valid_shape["manifest"], "source_commit": ""},
            },
            config_revision="sha256:config",
            dataset_revision="dataset",
            model_revision="model",
            artifact_prefix="runs/example",
        )
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: manifest$",
    ):
        build_checkpoint_metadata(
            {"manifest": [], "summary": {}},
            config_revision="sha256:config",
            dataset_revision="dataset",
            model_revision="model",
            artifact_prefix="runs/example",
        )
    with pytest.raises(
        ValueError,
        match=r"^merged summary must contain model results$",
    ):
        _checkpoint_metrics({"models": []})
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: model result$",
    ):
        _checkpoint_metrics({"models": [None]})


def test_render_report_writes_latex_and_checkpoint(tmp_path, monkeypatch) -> None:
    manifest = {
        "source_commit": "a" * 40,
        "seed": 7,
        "llama_cpp_revision": "d" * 40,
        "protocol": {"warmups": 1, "repetitions": 2, "prompt_cache": False},
        "example": {
            "sentence": "A park is land use.",
            "question": "Is this sentence relevant for a land use description?",
            "labels": ["yes", "no"],
        },
        "models": {
            "encoder": {"id": "encoder", "revision": "b" * 40},
            "llm": {"id": "llm", "revision": "c" * 40},
        },
    }
    summary = {
        "models": [
            {
                "model": model,
                "latency": {
                    "count": 2,
                    "minimum": 0.9,
                    "median": 1.0,
                    "p05": 0.9,
                    "p95": 1.1,
                    "maximum": 1.1,
                    "mean": 1.0,
                    "stdev": 0.1,
                },
                "mean_logprobs": {"yes": -0.1, "no": -1.0},
                "decision_counts": {"yes": 2, "no": 0},
            }
            for model in ("encoder", "llm")
        ]
    }
    for backend in ("encoder", "llm"):
        backend_dir = tmp_path / backend
        backend_dir.mkdir()
        (backend_dir / "manifest.json").write_text(json.dumps(manifest))
        (backend_dir / "summary.json").write_text(json.dumps(summary))

    monkeypatch.setenv("GRID5000_CONFIG_REVISION", "sha256:config")
    monkeypatch.setenv("GRID5000_DATASET_REVISION", "made-up-landuse-example-v1")
    monkeypatch.setenv("GRID5000_MODEL_REVISION", "c" * 40)
    monkeypatch.setenv("GRID5000_ARTIFACT_PREFIX", "runs/example")
    output = tmp_path / "report.tex"
    checkpoint = tmp_path / "checkpoint.json"

    render_report(tmp_path / "encoder", tmp_path / "llm", output, checkpoint=checkpoint)

    assert "Binary Land-Use Logprob Benchmark" in output.read_text()
    assert "Is this sentence relevant for a land use description?" in output.read_text()
    checkpoint_text = checkpoint.read_text()
    checkpoint_document = json.loads(checkpoint_text)
    assert checkpoint_document["complete"] is True
    assert checkpoint_document["config_revision"] == "sha256:config"
    assert checkpoint_document["dataset_revision"] == "made-up-landuse-example-v1"
    assert checkpoint_document["model_revision"] == "c" * 40
    assert (
        checkpoint_text
        == json.dumps(checkpoint_document, indent=2, sort_keys=True) + "\n"
    )

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]")
    with pytest.raises(ValueError, match="not an object"):
        _read_json(invalid)


def test_render_report_reads_canonical_artifact_names(tmp_path, monkeypatch) -> None:
    manifest = {"source_commit": "a" * 40, "example": {"sentence": "x"}}
    summary = {
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
                "mean_logprobs": {"yes": -0.1, "no": -1.0},
                "decision_counts": {"yes": 1, "no": 0},
            }
        ]
    }
    paths: list[Path] = []

    def read_json(path: Path) -> Mapping[str, object]:
        paths.append(path)
        return manifest if path.name == "manifest.json" else summary

    monkeypatch.setattr(report_module, "_read_json", read_json)
    render_report(
        tmp_path / "encoder",
        tmp_path / "llm",
        tmp_path / "report.tex",
    )
    assert paths == [
        tmp_path / "encoder" / "manifest.json",
        tmp_path / "llm" / "manifest.json",
        tmp_path / "encoder" / "summary.json",
        tmp_path / "llm" / "summary.json",
    ]


def test_read_json_passes_explicit_utf8_encoding() -> None:
    class RecordingPath:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def read_text(self, **kwargs: object) -> str:
            self.calls.append(kwargs)
            return "{}"

    path = RecordingPath()

    assert _read_json(cast(Path, path)) == {}
    assert path.calls == [{"encoding": "utf-8"}]


def test_render_report_passes_explicit_utf8_encoding_to_outputs(monkeypatch) -> None:
    class RecordingPath:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict[str, object]]] = []

        def write_text(self, content: str, **kwargs: object) -> None:
            self.calls.append((content, kwargs))

    output = RecordingPath()
    checkpoint = RecordingPath()
    monkeypatch.setattr(report_module, "_read_json", lambda _path: {})
    monkeypatch.setattr(
        report_module,
        "merge_artifacts",
        lambda *_args: {"manifest": {}, "summary": {}},
    )
    monkeypatch.setattr(
        report_module,
        "render_latex_document",
        lambda _manifest, _summary: "latex",
    )
    monkeypatch.setattr(
        report_module,
        "build_checkpoint_metadata",
        lambda *_args, **_kwargs: {"complete": True},
    )
    monkeypatch.setenv("GRID5000_CONFIG_REVISION", "config")
    monkeypatch.setenv("GRID5000_DATASET_REVISION", "dataset")
    monkeypatch.setenv("GRID5000_MODEL_REVISION", "model")
    monkeypatch.setenv("GRID5000_ARTIFACT_PREFIX", "runs/example")

    render_report(
        Path("encoder"),
        Path("llm"),
        cast(Path, output),
        checkpoint=cast(Path, checkpoint),
    )

    assert output.calls == [("latex", {"encoding": "utf-8"})]
    assert checkpoint.calls == [('{\n  "complete": true\n}\n', {"encoding": "utf-8"})]


def test_merge_artifacts_rejects_incomplete_backend_summaries() -> None:
    manifest = {"source_commit": "a" * 40, "example": {"sentence": "x"}}
    complete = {
        "models": [
            {
                "model": "encoder",
                "latency": {"count": 1, "median": 1.0},
                "decision_counts": {"yes": 1, "no": 0},
            }
        ]
    }

    with pytest.raises(ValueError, match="model results"):
        merge_artifacts(manifest, manifest, {"models": []}, complete)
    with pytest.raises(ValueError, match="latency"):
        merge_artifacts(
            manifest,
            manifest,
            {"models": [{"model": "encoder"}]},
            complete,
        )
    duplicate_model = {
        **complete["models"][0],
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
        "mean_logprobs": {"yes": -0.1, "no": -1.0},
    }
    with pytest.raises(ValueError, match="duplicate"):
        merge_artifacts(
            manifest,
            manifest,
            {"models": [duplicate_model, duplicate_model]},
            complete,
        )


def test_merge_artifacts_preserves_backend_runtime_contract(monkeypatch) -> None:
    encoder_manifest = {
        "source_commit": "a" * 40,
        "example": {"sentence": "x"},
        "runtime": {"device": "cuda"},
    }
    llm_manifest = {
        **encoder_manifest,
        "runtime": {"device": "cpu"},
    }
    encoder_model = {
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
        "mean_logprobs": {"yes": -0.1, "no": -1.0},
        "decision_counts": {"yes": 1, "no": 0},
    }
    llm_model = {**encoder_model, "model": "llm"}
    merged = merge_artifacts(
        encoder_manifest,
        llm_manifest,
        {"models": [encoder_model]},
        {"models": [llm_model]},
    )
    manifest = cast(dict[str, object], merged["manifest"])
    assert manifest["runtime_by_backend"] == {
        "encoder": {"device": "cuda"},
        "llm": {"device": "cpu"},
    }

    without_runtime = merge_artifacts(
        {key: value for key, value in encoder_manifest.items() if key != "runtime"},
        {key: value for key, value in llm_manifest.items() if key != "runtime"},
        {"models": [encoder_model]},
        {"models": [llm_model]},
    )
    without_runtime_manifest = cast(dict[str, object], without_runtime["manifest"])
    assert without_runtime_manifest["runtime_by_backend"] == {
        "encoder": {},
        "llm": {},
    }

    captured: list[tuple[object, str]] = []

    def capture_models(value: object, backend: str) -> list[dict[str, object]]:
        captured.append((value, backend))
        return []

    monkeypatch.setattr(report_module, "_validated_summary_models", capture_models)
    assert merge_artifacts(encoder_manifest, llm_manifest, {}, {})["summary"] == {
        "models": []
    }
    assert captured == [([], "encoder"), ([], "llm")]


def test_merge_artifacts_rejects_identity_and_model_contract_violations() -> None:
    manifest = {"source_commit": "a" * 40, "example": {"sentence": "x"}}
    complete = {
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
                "mean_logprobs": {"yes": -0.1, "no": -1.0},
                "decision_counts": {"yes": 1, "no": 0},
            }
        ]
    }
    with pytest.raises(
        ValueError,
        match=r"^source commit differs between backend runs$",
    ):
        merge_artifacts(
            manifest,
            {**manifest, "source_commit": "b" * 40},
            complete,
            complete,
        )
    with pytest.raises(
        ValueError,
        match=r"^example differs between backend runs$",
    ):
        merge_artifacts(
            manifest,
            {**manifest, "example": {"sentence": "y"}},
            complete,
            complete,
        )
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: encoder model result$",
    ):
        merge_artifacts(
            manifest,
            manifest,
            {"models": [None]},
            complete,
        )


def test_validate_mean_logprobs_uses_shared_candidate_labels(monkeypatch) -> None:
    calls: list[str] = []

    def labels() -> tuple[str, str]:
        calls.append("called")
        return ("yes", "no")

    monkeypatch.setattr(report_module, "candidate_labels", labels, raising=False)

    report_module._validate_mean_logprobs({"mean_logprobs": {"yes": -1.0, "no": -2.0}})
    assert calls == ["called"]


def test_render_report_validation_contracts() -> None:
    with pytest.raises(
        ValueError,
        match=r"^latency field is invalid: minimum$",
    ):
        _validate_latency_summary(
            {"latency": {"count": 1}, "mean_logprobs": {}, "decision_counts": {}}
        )
    with pytest.raises(
        ValueError,
        match=r"^latency quantiles are inconsistent$",
    ):
        _validate_latency_order(
            {
                "minimum": 1.0,
                "p05": 3.0,
                "median": 2.0,
                "p95": 4.0,
                "maximum": 5.0,
                "mean": 3.0,
            }
        )
    with pytest.raises(
        ValueError,
        match=r"^latency mean is inconsistent$",
    ):
        _validate_latency_order(
            {
                "minimum": 1.0,
                "p05": 1.0,
                "median": 1.0,
                "p95": 1.0,
                "maximum": 2.0,
                "mean": 3.0,
            }
        )
    with pytest.raises(
        ValueError,
        match=r"^mean_logprobs must contain yes and no$",
    ):
        _validate_mean_logprobs(
            {"mean_logprobs": {"yes": -1.0}, "latency": {}, "decision_counts": {}}
        )
    with pytest.raises(
        ValueError,
        match=r"^mean_logprobs must be finite$",
    ):
        _validate_mean_logprobs({"mean_logprobs": {"yes": float("nan"), "no": -1.0}})
    with pytest.raises(
        ValueError,
        match=r"^decision_counts must sum to latency count$",
    ):
        _validate_decision_counts(
            {"latency": {"count": 2}, "decision_counts": {"yes": 1, "no": 0}}
        )
    with pytest.raises(
        ValueError,
        match=r"^decision_counts must contain yes and no$",
    ):
        _validated_decision_counts({"yes": 1})
    with pytest.raises(
        ValueError,
        match=r"^mean_logprobs must contain yes and no$",
    ):
        _require_binary_keys({"yes": -1.0}, "mean_logprobs")
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: field$",
    ):
        _as_mapping(None, "field")
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: encoder model result$",
    ):
        _validated_summary_models([None], "encoder")


def test_merge_artifacts_rejects_inconsistent_latency_order() -> None:
    manifest = {"source_commit": "a" * 40, "example": {"sentence": "x"}}
    model = {
        "model": "encoder",
        "latency": {
            "count": 2,
            "minimum": 1.0,
            "median": 2.0,
            "p05": 3.0,
            "p95": 4.0,
            "maximum": 5.0,
            "mean": 3.0,
            "stdev": 1.0,
        },
        "mean_logprobs": {"yes": -0.1, "no": -1.0},
        "decision_counts": {"yes": 2, "no": 0},
    }

    with pytest.raises(
        ValueError,
        match=r"^latency quantiles are inconsistent$",
    ):
        merge_artifacts(
            manifest,
            manifest,
            {"models": [model]},
            {"models": [{**model, "model": "llm"}]},
        )

    valid_order = {
        **model,
        "latency": {**model["latency"], "p05": 1.5, "mean": 3.0},
    }
    invalid_mean = {
        **valid_order,
        "latency": {**valid_order["latency"], "mean": 10.0},
    }
    with pytest.raises(ValueError, match=r"^latency mean is inconsistent$"):
        merge_artifacts(
            manifest,
            manifest,
            {"models": [invalid_mean]},
            {"models": [{**valid_order, "model": "llm"}]},
        )

    invalid_score = {
        **valid_order,
        "mean_logprobs": {"yes": "1", "no": -1.0},
    }
    with pytest.raises(ValueError, match=r"^mean_logprobs must be finite$"):
        merge_artifacts(
            manifest,
            manifest,
            {"models": [invalid_score]},
            {"models": [{**valid_order, "model": "llm"}]},
        )

    negative_latency = {
        **valid_order,
        "latency": {**valid_order["latency"], "minimum": -1.0},
    }
    with pytest.raises(
        ValueError,
        match=r"^latency field is invalid: minimum$",
    ):
        merge_artifacts(
            manifest,
            manifest,
            {"models": [negative_latency]},
            {"models": [{**valid_order, "model": "llm"}]},
        )

    equal_values = {
        **model,
        "latency": {
            "count": 2,
            "minimum": 1.0,
            "median": 1.0,
            "p05": 1.0,
            "p95": 1.0,
            "maximum": 1.0,
            "mean": 1.0,
            "stdev": 1.0,
        },
    }
    equal_values["decision_counts"] = {"yes": 2, "no": 0}
    assert merge_artifacts(
        manifest,
        manifest,
        {"models": [equal_values]},
        {"models": [{**equal_values, "model": "llm"}]},
    )["summary"]


@pytest.mark.parametrize("value", [True, 0, "1"])
def test_positive_latency_count_rejects_invalid_values(value: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"^latency count must be a positive integer$",
    ):
        _positive_count(value)


@pytest.mark.parametrize("value", [True, -1, "1"])
def test_decision_count_rejects_invalid_values(value: object) -> None:
    with pytest.raises(
        ValueError,
        match=r"^decision_counts must be non-negative integers$",
    ):
        _non_negative_count(value)


def test_render_report_cli_contract(monkeypatch) -> None:
    calls: list[tuple[Path, Path, Path, Path | None]] = []

    def capture(
        encoder_dir: Path,
        llm_dir: Path,
        output: Path,
        checkpoint: Path | None = None,
    ) -> None:
        calls.append((encoder_dir, llm_dir, output, checkpoint))

    monkeypatch.setattr(report_module, "render_report", capture)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "render_report",
            "--encoder-dir",
            "encoder",
            "--llm-dir",
            "llm",
            "--output",
            "report.tex",
            "--checkpoint",
            "checkpoint.json",
        ],
    )
    main()
    assert calls == [
        (Path("encoder"), Path("llm"), Path("report.tex"), Path("checkpoint.json"))
    ]


@pytest.mark.parametrize("missing", ["--encoder-dir", "--llm-dir", "--output"])
def test_render_report_cli_requires_core_paths(monkeypatch, missing: str) -> None:
    argv = ["render_report"]
    values = {
        "--encoder-dir": "encoder",
        "--llm-dir": "llm",
        "--output": "report.tex",
    }
    for option, value in values.items():
        if option != missing:
            argv.extend([option, value])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit):
        main()


def test_render_report_cli_help_uses_module_description(monkeypatch, capsys) -> None:
    monkeypatch.setattr(sys, "argv", ["render_report", "--help"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
    assert "Merge verified backend artifacts" in capsys.readouterr().out
