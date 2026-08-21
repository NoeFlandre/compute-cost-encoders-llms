from __future__ import annotations

import json

import pytest
from scripts.render_report import (
    _checkpoint_metrics,
    _non_negative_count,
    _positive_count,
    _read_json,
    build_checkpoint_metadata,
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


def test_build_checkpoint_metadata_rejects_empty_artifact_prefix() -> None:
    with pytest.raises(ValueError, match="artifact prefix"):
        build_checkpoint_metadata(
            {"manifest": {}, "summary": {"models": []}},
            config_revision="sha256:config",
            dataset_revision="dataset",
            model_revision="model",
            artifact_prefix="",
        )
    with pytest.raises(ValueError, match="artifact prefix"):
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
    with pytest.raises(ValueError, match="non-negative integer"):
        build_checkpoint_metadata(
            {**valid_shape, "manifest": {**valid_shape["manifest"], "seed": -1}},
            config_revision="sha256:config",
            dataset_revision="dataset",
            model_revision="model",
            artifact_prefix="runs/example",
        )
    with pytest.raises(ValueError, match="non-negative integer"):
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
    with pytest.raises(ValueError, match="text"):
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
    with pytest.raises(ValueError, match="object"):
        build_checkpoint_metadata(
            {"manifest": [], "summary": {}},
            config_revision="sha256:config",
            dataset_revision="dataset",
            model_revision="model",
            artifact_prefix="runs/example",
        )
    with pytest.raises(ValueError, match="model results"):
        _checkpoint_metrics({"models": []})


def test_render_report_writes_latex_and_checkpoint(tmp_path, monkeypatch) -> None:
    manifest = {
        "source_commit": "a" * 40,
        "seed": 7,
        "llama_cpp_revision": "d" * 40,
        "protocol": {"warmups": 1, "repetitions": 2, "prompt_cache": False},
        "example": {"sentence": "A park is land use.", "labels": ["yes", "no"]},
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
    assert json.loads(checkpoint.read_text())["complete"] is True

    invalid = tmp_path / "invalid.json"
    invalid.write_text("[]")
    with pytest.raises(ValueError, match="not an object"):
        _read_json(invalid)


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

    with pytest.raises(ValueError, match="quantiles"):
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
    with pytest.raises(ValueError, match="mean"):
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
    with pytest.raises(ValueError, match="mean_logprobs"):
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
    with pytest.raises(ValueError, match="latency field"):
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
    with pytest.raises(ValueError, match="positive integer"):
        _positive_count(value)


@pytest.mark.parametrize("value", [True, -1, "1"])
def test_decision_count_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="non-negative integers"):
        _non_negative_count(value)
