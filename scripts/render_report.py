#!/usr/bin/env python3
"""Merge verified backend artifacts into a standalone LaTeX report."""

from __future__ import annotations

import argparse
import itertools
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import TypeGuard

from compute_cost_encoders_llms.benchmark._numerics import _is_finite_number
from compute_cost_encoders_llms.benchmark.example import candidate_labels
from compute_cost_encoders_llms.benchmark.latex import render_latex_document
from scripts._artifact_fields import _as_mapping, _mapping_value, _text_value
from scripts.grid5000.checkpoint_metadata import build_checkpoint_metadata


def _read_json(path: Path) -> Mapping[str, object]:
    document = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(document, Mapping):
        raise ValueError(f"JSON document is not an object: {path}")
    return document


def merge_artifacts(
    encoder_manifest: Mapping[str, object],
    llm_manifest: Mapping[str, object],
    encoder_summary: Mapping[str, object],
    llm_summary: Mapping[str, object],
) -> dict[str, object]:
    """Combine two backend runs only when their protocol identities match."""

    if encoder_manifest.get("source_commit") != llm_manifest.get("source_commit"):
        raise ValueError("source commit differs between backend runs")
    if encoder_manifest.get("example") != llm_manifest.get("example"):
        raise ValueError("example differs between backend runs")
    manifest = dict(encoder_manifest)
    manifest["backend"] = "both"
    manifest["run_ids"] = [encoder_manifest.get("run_id"), llm_manifest.get("run_id")]
    manifest["runtime_by_backend"] = {
        "encoder": encoder_manifest.get("runtime", {}),
        "llm": llm_manifest.get("runtime", {}),
    }
    encoder_models = encoder_summary.get("models", [])
    llm_models = llm_summary.get("models", [])
    validated_encoder = _validated_summary_models(encoder_models, "encoder")
    validated_llm = _validated_summary_models(llm_models, "llm")
    return {
        "manifest": manifest,
        "summary": {"models": validated_encoder + validated_llm},
    }


def _validated_summary_models(
    value: object,
    backend: str,
) -> list[Mapping[str, object]]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{backend} summary must contain model results")
    models: list[Mapping[str, object]] = []
    names: set[str] = set()
    for item in value:
        model = _as_mapping(item, f"{backend} model result")
        name = _text_value(model, "model")
        if name in names:
            raise ValueError(f"{backend} summary contains duplicate models")
        names.add(name)
        _validate_latency_summary(model)
        _validate_mean_logprobs(model)
        _validate_decision_counts(model)
        models.append(model)
    return models


def _validate_latency_summary(model: Mapping[str, object]) -> None:
    latency = _mapping_value(model, "latency")
    _positive_count(latency.get("count"))
    fields = ("minimum", "median", "p05", "p95", "maximum", "mean", "stdev")
    values = {
        field: _validated_latency_value(latency.get(field), field) for field in fields
    }
    _validate_latency_order(values)


def _positive_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("latency count must be a positive integer")
    return value


def _validated_latency_value(value: object, field: str) -> float:
    if not _is_finite_non_negative(value):
        raise ValueError(f"latency field is invalid: {field}")
    return float(value)


def _validate_latency_order(values: Mapping[str, float]) -> None:
    ordered = [
        values[field] for field in ("minimum", "p05", "median", "p95", "maximum")
    ]
    if any(left > right for left, right in itertools.pairwise(ordered)):
        raise ValueError("latency quantiles are inconsistent")
    if not values["minimum"] <= values["mean"] <= values["maximum"]:
        raise ValueError("latency mean is inconsistent")


def _validate_mean_logprobs(model: Mapping[str, object]) -> None:
    scores = _mapping_value(model, "mean_logprobs")
    _require_binary_keys(scores, "mean_logprobs")
    for label in candidate_labels():
        if not _is_finite_number(scores[label]):
            raise ValueError("mean_logprobs must be finite")


def _validate_decision_counts(model: Mapping[str, object]) -> None:
    counts = _mapping_value(model, "decision_counts")
    values = _validated_decision_counts(counts)
    latency = _mapping_value(model, "latency")
    if sum(values) != latency["count"]:
        raise ValueError("decision_counts must sum to latency count")


def _require_binary_keys(values: Mapping[str, object], field: str) -> None:
    if set(values) != {"yes", "no"}:
        raise ValueError(f"{field} must contain yes and no")


def _validated_decision_counts(counts: Mapping[str, object]) -> tuple[int, int]:
    _require_binary_keys(counts, "decision_counts")
    return (
        _non_negative_count(counts["yes"]),
        _non_negative_count(counts["no"]),
    )


def _non_negative_count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("decision_counts must be non-negative integers")
    return value


def _is_finite_non_negative(value: object) -> TypeGuard[int | float]:
    return _is_finite_number(value) and float(value) >= 0


def render_report(
    encoder_dir: Path,
    llm_dir: Path,
    output: Path,
    checkpoint: Path | None = None,
) -> None:
    encoder_manifest = _read_json(encoder_dir / "manifest.json")
    llm_manifest = _read_json(llm_dir / "manifest.json")
    encoder_summary = _read_json(encoder_dir / "summary.json")
    llm_summary = _read_json(llm_dir / "summary.json")
    merged = merge_artifacts(
        encoder_manifest,
        llm_manifest,
        encoder_summary,
        llm_summary,
    )
    manifest = _mapping_value(merged, "manifest")
    summary = _mapping_value(merged, "summary")
    output.write_text(render_latex_document(manifest, summary), encoding="utf-8")
    if checkpoint is not None:
        metadata = build_checkpoint_metadata(
            merged,
            config_revision=os.environ["GRID5000_CONFIG_REVISION"],
            dataset_revision=os.environ["GRID5000_DATASET_REVISION"],
            model_revision=os.environ["GRID5000_MODEL_REVISION"],
            artifact_prefix=os.environ["GRID5000_ARTIFACT_PREFIX"],
        )
        checkpoint.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--llm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    render_report(args.encoder_dir, args.llm_dir, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
