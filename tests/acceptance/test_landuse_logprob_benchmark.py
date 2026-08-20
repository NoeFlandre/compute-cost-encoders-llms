from __future__ import annotations

from typing import cast

import pytest
from pytest_bdd import given, scenarios, then, when

from compute_cost_encoders_llms.benchmark.example import candidate_labels
from compute_cost_encoders_llms.benchmark.measurement import (
    MeasurementRecord,
    validate_measurement,
)

scenarios("features/landuse_logprob_benchmark.feature")


@pytest.fixture
def context() -> dict[str, object]:
    return {}


@given("the approved binary land-use example")
def approved_example() -> None:
    assert candidate_labels() == ("yes", "no")


@when("the benchmark records encoder and LLM log probabilities")
def record_scores(context: dict[str, object]) -> None:
    context["records"] = [
        validate_measurement(
            {
                "model": model,
                "repetition": 0,
                "tokenization_ms": 1.0,
                "model_ms": 2.0,
                "logprob_ms": 0.1,
                "text_to_logprob_ms": 3.1,
                "logprobs": {"yes": -0.1, "no": -2.2},
            }
        )
        for model in ("encoder", "llm")
    ]


@then("both records contain yes and no scores")
def records_contain_scores(context: dict[str, object]) -> None:
    records = cast(list[MeasurementRecord], context["records"])
    assert all(set(record["logprobs"]) == {"yes", "no"} for record in records)


@then("each record contains text-to-logprob timing")
def records_contain_timing(context: dict[str, object]) -> None:
    records = cast(list[MeasurementRecord], context["records"])
    assert all(record["text_to_logprob_ms"] > 0 for record in records)


@given("no OAR job allocation is present")
def no_oar_job(monkeypatch) -> None:
    monkeypatch.delenv("OAR_JOB_ID", raising=False)
    monkeypatch.delenv("OAR_NODEFILE", raising=False)


@when("the benchmark guard is evaluated")
def evaluate_guard(context: dict[str, object]) -> None:
    context["guard"] = not bool(
        __import__("os").environ.get("OAR_JOB_ID")
        and __import__("os").environ.get("OAR_NODEFILE")
    )


@then("execution is refused before model loading")
def execution_refused(context: dict[str, object]) -> None:
    assert context["guard"] is True
