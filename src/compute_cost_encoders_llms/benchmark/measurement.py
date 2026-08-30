from __future__ import annotations

import math
import statistics
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import NotRequired, Protocol, TypedDict, TypeGuard

from ._numerics import _is_finite_number
from .example import candidate_labels


class MeasurementError(ValueError):
    """Raised when a measurement record cannot be trusted."""


class MeasurementRecord(TypedDict):
    model: str
    repetition: int
    tokenization_ms: float | None
    model_ms: float | None
    logprob_ms: float
    text_to_logprob_ms: float
    logprobs: dict[str, float]
    input_tokens: NotRequired[int | None]
    decision: NotRequired[str]


class ScoreLike(Protocol):
    @property
    def tokenization_ms(self) -> float | None: ...

    @property
    def model_ms(self) -> float | None: ...

    @property
    def logprob_ms(self) -> float: ...

    @property
    def text_to_logprob_ms(self) -> float: ...

    @property
    def input_tokens(self) -> int | None: ...

    @property
    def logprobs(self) -> dict[str, float]: ...


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
    clock = time.perf_counter_ns
    for _ in range(warmups):
        operation()
    results: list[TimedValue[ValueT]] = []
    for _ in range(repetitions):
        start = clock()
        value = operation()
        elapsed_ms = (clock() - start) / 1_000_000
        results.append(TimedValue(value=value, elapsed_ms=elapsed_ms))
    return results


def choose_decision(logprobs: Mapping[str, float]) -> str:
    """Choose the highest binary score, preferring ``yes`` on a tie."""

    labels = candidate_labels()
    if set(logprobs) != set(labels):
        raise MeasurementError("logprobs must contain yes and no")
    if any(not math.isfinite(float(logprobs[label])) for label in labels):
        raise MeasurementError("logprobs must be finite")
    return max(labels, key=lambda label: float(logprobs[label]))


def score_record(model: str, repetition: int, score: ScoreLike) -> MeasurementRecord:
    """Normalize either backend result into the common measurement schema."""

    return {
        "model": model,
        "repetition": repetition,
        "tokenization_ms": score.tokenization_ms,
        "model_ms": score.model_ms,
        "logprob_ms": score.logprob_ms,
        "text_to_logprob_ms": score.text_to_logprob_ms,
        "input_tokens": score.input_tokens,
        "logprobs": score.logprobs,
        "decision": choose_decision(score.logprobs),
    }


_TIMING_FIELDS = (
    "tokenization_ms",
    "model_ms",
    "logprob_ms",
    "text_to_logprob_ms",
)
_OPTIONAL_TIMING_FIELDS = ("tokenization_ms", "model_ms")


def validate_measurement(record: Mapping[str, object]) -> MeasurementRecord:
    """Validate and copy one model timing and score record."""

    assert _require_measurement_fields(record)
    _validate_identity(record)
    _validate_timings(record)
    logprobs = _validate_logprobs(record)
    _validate_optional_fields(record, logprobs)
    return {**record}


def _require_measurement_fields(
    record: Mapping[str, object],
) -> TypeGuard[MeasurementRecord]:
    for field in ("model", "repetition", "logprobs", *_TIMING_FIELDS):
        if field not in record:
            raise MeasurementError(f"missing measurement field: {field}")
    return True


def _validate_identity(record: Mapping[str, object]) -> None:
    if not _is_non_empty_text(record["model"]):
        raise MeasurementError("model must be non-empty")
    if not _is_non_negative_integer(record["repetition"]):
        raise MeasurementError("repetition must be non-negative")


def _is_non_empty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_non_negative_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_timings(record: Mapping[str, object]) -> None:
    for field in _TIMING_FIELDS:
        _validate_timing_field(field, record[field])
    _validate_total_covers_components(record)


def _validate_timing_field(field: str, value: object) -> None:
    if value is None and field in _OPTIONAL_TIMING_FIELDS:
        return
    if not _is_finite_number(value):
        raise MeasurementError(f"{field} must be finite")
    if float(value) < 0:
        raise MeasurementError(f"{field} must be non-negative")


def _validate_total_covers_components(record: Mapping[str, object]) -> None:
    total = _required_float(record["text_to_logprob_ms"])
    fields = ("logprob_ms", *_OPTIONAL_TIMING_FIELDS)
    for field in fields:
        value = record[field]
        if value is not None and total < _required_float(value):
            raise MeasurementError("text_to_logprob_ms must cover component timings")


def _required_float(value: object) -> float:
    if not _is_finite_number(value):
        raise MeasurementError("timing must be finite")
    return float(value)


def _validate_logprobs(record: Mapping[str, object]) -> dict[str, float]:
    logprobs = record["logprobs"]
    if not _has_binary_logprob_keys(logprobs):
        raise MeasurementError("logprobs must contain yes and no")
    if not _has_finite_binary_logprobs(logprobs):
        raise MeasurementError("logprobs must be finite")
    return {label: float(logprobs[label]) for label in ("yes", "no")}


def _has_binary_logprob_keys(
    value: object,
) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping) and set(value) == {"yes", "no"}


def _has_finite_binary_logprobs(
    logprobs: Mapping[str, object],
) -> TypeGuard[Mapping[str, int | float]]:
    return all(_is_finite_number(logprobs[label]) for label in ("yes", "no"))


def _validate_optional_fields(
    record: Mapping[str, object], logprobs: Mapping[str, float]
) -> None:
    _validate_decision(record, logprobs)
    _validate_input_tokens(record)


def _validate_decision(
    record: Mapping[str, object], scores: Mapping[str, float]
) -> None:
    if "decision" not in record:
        return
    decision = record["decision"]
    if not isinstance(decision, str) or decision not in {"yes", "no"}:
        raise MeasurementError("decision must be yes or no")
    if decision != choose_decision(scores):
        raise MeasurementError("decision is inconsistent with logprobs")


def _validate_input_tokens(record: Mapping[str, object]) -> None:
    if "input_tokens" not in record:
        return
    if record["input_tokens"] is None:
        return
    if not _is_non_negative_integer(record["input_tokens"]):
        raise MeasurementError("input_tokens must be non-negative")


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
    return LatencySummary(
        count=len(values),
        minimum=values[0],
        median=_quantile(values, 0.5),
        p05=_quantile(values, 0.05),
        p95=_quantile(values, 0.95),
        maximum=values[-1],
        mean=statistics.fmean(values),
        stdev=statistics.stdev(values) if len(values) > 1 else 0.0,
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
