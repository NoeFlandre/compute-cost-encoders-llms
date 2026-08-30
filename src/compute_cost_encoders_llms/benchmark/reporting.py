from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypedDict

from ._numerics import _number_value
from .latex import (  # noqa: F401
    render_latex_document,
    render_latex_summary,
)
from .measurement import (
    LatencySummary,
    MeasurementError,
    MeasurementRecord,
    choose_decision,
    summarize_latencies,
    validate_measurement,
)


class ModelSummary(TypedDict):
    model: str
    latency: LatencySummary
    tokenization: LatencySummary | None
    model_time: LatencySummary | None
    logprob_time: LatencySummary
    mean_logprobs: dict[str, float]
    decision_counts: dict[str, int]


class SummaryDocument(TypedDict):
    models: list[ModelSummary]


class _JsonOptions(TypedDict, total=False):
    ensure_ascii: bool
    indent: int
    separators: tuple[str, str]
    sort_keys: bool


def _json_options(*, compact: bool) -> _JsonOptions:
    if not isinstance(compact, bool):
        raise TypeError("compact must be a boolean")
    options: _JsonOptions = {
        "ensure_ascii": False,
        "sort_keys": True,
    }
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    return options


def json_line(record: Mapping[str, object]) -> str:
    """Serialize one stable JSONL record."""

    return json.dumps(record, **_json_options(compact=True))


def build_summary(records: Iterable[Mapping[str, object]]) -> SummaryDocument:
    """Group validated measurements into deterministic model summaries."""

    return _summary_from_grouped(_group_records(records))


def _summary_from_validated_records(
    records: Iterable[MeasurementRecord],
) -> SummaryDocument:
    """Build a summary without revalidating records checked by the caller."""

    return _summary_from_grouped(_group_validated_records(records))


def _summary_from_grouped(
    grouped: Mapping[str, list[MeasurementRecord]],
) -> SummaryDocument:
    return {
        "models": [_model_summary(model, grouped[model]) for model in sorted(grouped)]
    }


def _group_records(
    records: Iterable[Mapping[str, object]],
) -> dict[str, list[MeasurementRecord]]:
    return _group_validated_records(validate_measurement(record) for record in records)


def _validated_records(
    records: Iterable[Mapping[str, object]],
) -> list[MeasurementRecord]:
    """Validate a run's records once so downstream artifact writers can reuse them."""

    return [validate_measurement(record) for record in records]


def _group_validated_records(
    records: Iterable[MeasurementRecord],
) -> dict[str, list[MeasurementRecord]]:
    grouped: dict[str, list[MeasurementRecord]] = {}
    identities: set[tuple[str, int]] = set()
    for record in records:
        model = str(record["model"])
        identity = (model, record["repetition"])
        if identity in identities:
            raise MeasurementError("duplicate measurement repetition")
        identities.add(identity)
        grouped.setdefault(model, []).append(record)
    if not grouped:
        raise MeasurementError("no measurements")
    return grouped


def _model_summary(
    model: str,
    records: list[MeasurementRecord],
) -> ModelSummary:
    decisions = {"yes": 0, "no": 0}
    for record in records:
        decisions[choose_decision(record["logprobs"])] += 1
    return {
        "model": model,
        "latency": summarize_latencies(_timing_values(records, "text_to_logprob_ms")),
        "tokenization": _optional_latency_summary(
            _timing_values(records, "tokenization_ms")
        ),
        "model_time": _optional_latency_summary(_timing_values(records, "model_ms")),
        "logprob_time": summarize_latencies(_timing_values(records, "logprob_ms")),
        "mean_logprobs": {
            "yes": _mean_score(records, "yes"),
            "no": _mean_score(records, "no"),
        },
        "decision_counts": decisions,
    }


def _timing_values(
    records: list[MeasurementRecord],
    field: str,
) -> list[float]:
    values: list[float] = []
    for record in records:
        value = _number_value(record.get(field))
        if value is not None:
            values.append(value)
    return values


def _mean_score(records: list[MeasurementRecord], label: str) -> float:
    return sum(float(record["logprobs"][label]) for record in records) / len(records)


def _optional_latency_summary(
    values: list[float],
) -> LatencySummary | None:
    return summarize_latencies(values) if values else None


def write_json(path: Path, document: Mapping[str, object]) -> None:
    """Write one deterministic JSON document."""

    path.write_text(
        json.dumps(document, **_json_options(compact=False)) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """Write validated records as deterministic JSONL."""

    _write_validated_jsonl(path, (validate_measurement(record) for record in records))


def write_measurement_artifacts(
    output_dir: Path, records: Iterable[Mapping[str, object]]
) -> None:
    """Write validated measurement and summary artifacts for one run."""

    validated = _validated_records(records)
    _write_validated_jsonl(output_dir / "measurements.jsonl", validated)
    write_json(output_dir / "summary.json", _summary_from_validated_records(validated))


def _write_validated_jsonl(path: Path, records: Iterable[MeasurementRecord]) -> None:
    """Write records after validation has already been completed."""

    lines = [json_line(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
