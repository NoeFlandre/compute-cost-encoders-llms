from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import NotRequired, TypedDict, cast


class MeasurementError(ValueError):
    """Raised when a measurement record cannot be trusted."""


class MeasurementRecord(TypedDict):
    model: str
    repetition: int
    tokenization_ms: float
    model_ms: float
    logprob_ms: float
    text_to_logprob_ms: float
    logprobs: dict[str, float]
    input_tokens: NotRequired[int]
    decision: NotRequired[str]


@dataclass(frozen=True, slots=True)
class TimedValue[ValueT]:
    value: ValueT
    elapsed_ms: float


def measure_repetitions[ValueT](
    operation: Callable[[], ValueT],
    *,
    warmups: int,
    repetitions: int,
) -> list[TimedValue[ValueT]]:
    """Run warmups, then return timed values for measured repetitions."""

    if warmups < 0 or repetitions < 1:
        raise MeasurementError("warmups must be non-negative and repetitions positive")
    for _ in range(warmups):
        operation()
    results: list[TimedValue[ValueT]] = []
    for _ in range(repetitions):
        start = time.perf_counter_ns()
        value = operation()
        elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
        results.append(TimedValue(value=value, elapsed_ms=elapsed_ms))
    return results


def choose_decision(logprobs: Mapping[str, float]) -> str:
    """Choose the highest binary score, preferring ``yes`` on a tie."""

    labels = ("yes", "no")
    if set(logprobs) != set(labels):
        raise MeasurementError("logprobs must contain yes and no")
    if any(not math.isfinite(float(logprobs[label])) for label in labels):
        raise MeasurementError("logprobs must be finite")
    return max(labels, key=lambda label: float(logprobs[label]))


_TIMING_FIELDS = (
    "tokenization_ms",
    "model_ms",
    "logprob_ms",
    "text_to_logprob_ms",
)


def validate_measurement(record: Mapping[str, object]) -> MeasurementRecord:
    """Validate and copy one model timing and score record."""

    _require_measurement_fields(record)
    _validate_identity(record)
    _validate_timings(record)
    _validate_logprobs(record)
    return cast(MeasurementRecord, dict(record))


def _require_measurement_fields(record: Mapping[str, object]) -> None:
    for field in ("model", "repetition", "logprobs", *_TIMING_FIELDS):
        if field not in record:
            raise MeasurementError(f"missing measurement field: {field}")


def _validate_identity(record: Mapping[str, object]) -> None:
    if not isinstance(record["model"], str) or not record["model"]:
        raise MeasurementError("model must be non-empty")
    if not isinstance(record["repetition"], int) or record["repetition"] < 0:
        raise MeasurementError("repetition must be non-negative")


def _validate_timings(record: Mapping[str, object]) -> None:
    for field in _TIMING_FIELDS:
        value = record[field]
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise MeasurementError(f"{field} must be finite")
        if float(value) < 0:
            raise MeasurementError(f"{field} must be non-negative")


def _validate_logprobs(record: Mapping[str, object]) -> None:
    logprobs = record["logprobs"]
    if not isinstance(logprobs, Mapping) or set(logprobs) != {"yes", "no"}:
        raise MeasurementError("logprobs must contain yes and no")


def _quantile(values: list[float], probability: float) -> float:
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    weight = position - lower
    return values[lower] + weight * (values[upper] - values[lower])


class LatencySummary(TypedDict):
    count: int
    minimum: float
    median: float
    p05: float
    p95: float
    maximum: float
    mean: float
    stdev: float


def summarize_latencies(latencies: Iterable[float]) -> LatencySummary:
    """Summarize non-negative latency samples with deterministic quantiles."""

    values = _validated_latencies(latencies)
    return cast(
        LatencySummary,
        {
            "count": len(values),
            "minimum": values[0],
            "median": _quantile(values, 0.5),
            "p05": _quantile(values, 0.05),
            "p95": _quantile(values, 0.95),
            "maximum": values[-1],
            "mean": statistics.fmean(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        },
    )


def _validated_latencies(latencies: Iterable[float]) -> list[float]:
    values = sorted(float(value) for value in latencies)
    if not values:
        raise MeasurementError("latencies must contain finite non-negative values")
    if _has_invalid_latency(values):
        raise MeasurementError("latencies must contain finite non-negative values")
    return values


def _has_invalid_latency(values: Iterable[float]) -> bool:
    return any(value < 0 or not math.isfinite(value) for value in values)
