#!/usr/bin/env python3
"""Validate the metadata required to resume a Grid5000 checkpoint safely."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from scripts._artifact_fields import _as_mapping, _mapping_value, _text_value

PROJECT_BUCKET_PREFIX = "hf://buckets/NoeFlandre/compute-cost-encoders-llms/"
PROJECT_BUCKET_URI = PROJECT_BUCKET_PREFIX.rstrip("/")
REQUIRED_TEXT_FIELDS = (
    "source_commit",
    "config_revision",
    "dataset_revision",
    "model_revision",
    "artifact_uri",
)


def build_checkpoint_metadata(
    merged: Mapping[str, object],
    *,
    config_revision: str,
    dataset_revision: str,
    model_revision: str,
    artifact_prefix: str,
) -> dict[str, object]:
    """Build the complete metadata required by the Grid5000 publisher."""

    prefix = artifact_prefix.strip("/")
    if not prefix or ".." in prefix:
        raise ValueError("artifact prefix must be non-empty and traversal-free")
    manifest = _mapping_value(merged, "manifest")
    summary = _mapping_value(merged, "summary")
    protocol = _mapping_value(manifest, "protocol")
    source_commit = _text_value(manifest, "source_commit")
    seed = _integer_value(manifest, "seed")
    step = _integer_value(protocol, "repetitions")
    return {
        "source_commit": source_commit,
        "config_revision": config_revision,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "seed": seed,
        "step": step,
        "metrics": _checkpoint_metrics(summary),
        "complete": True,
        "artifact_uri": f"{PROJECT_BUCKET_URI}/{prefix}",
    }


def _integer_value(document: Mapping[str, object], field: str) -> int:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"merged artifact field is not a non-negative integer: {field}"
        )
    return value


def _checkpoint_metrics(summary: Mapping[str, object]) -> dict[str, object]:
    models = summary.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("merged summary must contain model results")
    metrics: dict[str, object] = {}
    for model in models:
        model_mapping = _as_mapping(model, "model result")
        model_name = _text_value(model_mapping, "model")
        latency = _mapping_value(model_mapping, "latency")
        decisions = _mapping_value(model_mapping, "decision_counts")
        metrics[model_name] = {
            "median_text_to_logprob_ms": latency.get("median"),
            "decision_counts": dict(decisions),
        }
    return metrics


def _required_text_errors(metadata: Mapping[str, object]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_TEXT_FIELDS:
        value = metadata.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} is required")
    return errors


def _non_negative_integer_error(
    metadata: Mapping[str, object], field: str
) -> str | None:
    value = metadata.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return f"{field} must be a non-negative integer"
    return None


def _integer_errors(metadata: Mapping[str, object]) -> list[str]:
    errors = []
    for field in ("seed", "step"):
        error = _non_negative_integer_error(metadata, field)
        if error is not None:
            errors.append(error)
    return errors


def _metrics_error(metadata: Mapping[str, object]) -> str | None:
    metrics = metadata.get("metrics")
    if not isinstance(metrics, Mapping) or not metrics:
        return "metrics must be a non-empty object"
    return None


def _complete_error(metadata: Mapping[str, object]) -> str | None:
    if metadata.get("complete") is not True:
        return "complete must be true"
    return None


def _artifact_uri_error(metadata: Mapping[str, object]) -> str | None:
    artifact_uri = metadata.get("artifact_uri")
    if not isinstance(artifact_uri, str) or not artifact_uri:
        return None
    if artifact_uri.startswith(PROJECT_BUCKET_PREFIX):
        return None
    return "artifact_uri must use the project Hugging Face bucket"


def _integrity_errors(metadata: Mapping[str, object]) -> list[str]:
    errors = []
    for error in (
        _metrics_error(metadata),
        _complete_error(metadata),
        _artifact_uri_error(metadata),
    ):
        if error is not None:
            errors.append(error)
    return errors


def validate_metadata(metadata: object) -> list[str]:
    """Return deterministic validation errors for checkpoint metadata."""
    if not isinstance(metadata, Mapping):
        return ["metadata must be a JSON object"]

    return (
        _required_text_errors(metadata)
        + _integer_errors(metadata)
        + _integrity_errors(metadata)
    )


def validate_file(path: Path) -> list[str]:
    """Return validation errors for a JSON metadata file."""
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return [f"could not read JSON metadata: {error}"]
    return validate_metadata(metadata)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata", type=Path)
    args = parser.parse_args(argv)

    errors = validate_file(args.metadata)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(f"Valid checkpoint metadata: {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
