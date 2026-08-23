#!/usr/bin/env python3
"""Validate the metadata required to resume a Grid5000 checkpoint safely."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_BUCKET_PREFIX = "hf://buckets/NoeFlandre/compute-cost-encoders-llms/"
REQUIRED_TEXT_FIELDS = (
    "source_commit",
    "config_revision",
    "dataset_revision",
    "model_revision",
    "artifact_uri",
)


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
