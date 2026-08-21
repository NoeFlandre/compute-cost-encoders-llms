from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypedDict, cast

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


def json_line(record: Mapping[str, object]) -> str:
    """Serialize one stable JSONL record."""

    return json.dumps(record, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def build_summary(records: Iterable[Mapping[str, object]]) -> SummaryDocument:
    """Group validated measurements into deterministic model summaries."""

    grouped = _group_records(records)
    return {
        "models": [_model_summary(model, grouped[model]) for model in sorted(grouped)]
    }


def _group_records(
    records: Iterable[Mapping[str, object]],
) -> dict[str, list[MeasurementRecord]]:
    grouped: dict[str, list[MeasurementRecord]] = {}
    identities: set[tuple[str, int]] = set()
    for record in records:
        validated = validate_measurement(record)
        model = str(validated["model"])
        identity = (model, validated["repetition"])
        if identity in identities:
            raise MeasurementError("duplicate measurement repetition")
        identities.add(identity)
        grouped.setdefault(model, []).append(validated)
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
    return [
        float(cast(int | float, value))
        for record in records
        if (value := cast(Mapping[str, object], record)[field]) is not None
    ]


def _mean_score(records: list[MeasurementRecord], label: str) -> float:
    return sum(float(record["logprobs"][label]) for record in records) / len(records)


def _optional_latency_summary(
    values: list[float],
) -> LatencySummary | None:
    return summarize_latencies(values) if values else None


def write_json(path: Path, document: Mapping[str, object]) -> None:
    """Write one deterministic JSON document."""

    path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )


def write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    """Write validated records as deterministic JSONL."""

    lines = [json_line(validate_measurement(record)) for record in records]
    path.write_text("\n".join(lines) + "\n")


def _latex_escape(value: object) -> str:
    escaped = str(value)
    for source, replacement in (
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
    ):
        escaped = escaped.replace(source, replacement)
    return escaped


def render_latex_summary(summary: Mapping[str, object]) -> str:
    """Render one compact LaTeX summary table."""

    model = _latex_escape(cast(str, summary["model"]))
    median = float(cast(float, summary["median"]))
    decision = _latex_escape(cast(str, summary["decision"]))
    return_value = (
        "\\begin{tabular}{lr}\nModel & "
        + model
        + r" \\"
        + "\nMedian ms & "
        + f"{median:.3f}"
        + r" \\"
        + "\nDecision & "
        + decision
        + r" \\"
        + "\n\\end{tabular}"
    )
    return return_value


def render_latex_document(
    manifest: Mapping[str, object], summary: Mapping[str, object]
) -> str:
    """Render the complete reproducibility report as a standalone LaTeX file."""

    models = cast(Mapping[str, Mapping[str, str]], manifest["models"])
    example = cast(Mapping[str, str], manifest["example"])
    protocol = cast(Mapping[str, object], manifest["protocol"])
    model_summaries = cast(list[ModelSummary], summary["models"])
    rows = [
        r"\documentclass[11pt]{article}",
        r"\usepackage[margin=1in]{geometry}",
        r"\usepackage{booktabs}",
        r"\title{Binary Land-Use Logprob Benchmark}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Objective}",
        "Measure the cost of mapping one land-use sentence to binary yes/no "
        "log probabilities without fine-tuning or multi-token decoding.",
        r"\section*{Reproducibility}",
        "Source commit: \\texttt{"
        + _latex_escape(cast(str, manifest["source_commit"]))
        + "}\\\\",
        "Encoder: "
        + _latex_escape(models["encoder"]["id"])
        + " at \\texttt{"
        + _latex_escape(models["encoder"]["revision"])
        + "}\\\\",
        "LLM: "
        + _latex_escape(models["llm"]["id"])
        + " at \\texttt{"
        + _latex_escape(models["llm"]["revision"])
        + "}\\\\",
        "llama.cpp revision: \\texttt{"
        + _latex_escape(cast(str, manifest["llama_cpp_revision"]))
        + "}\\\\",
        "Protocol: "
        + _latex_escape(protocol["warmups"])
        + " warmups, "
        + _latex_escape(protocol["repetitions"])
        + " repetitions, prompt cache disabled.\\\\",
        r"\section*{Example}",
        _latex_escape(example["sentence"]),
        r"\section*{Results}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & $n$ & Median ms & P05 ms & P95 ms & mean log $p(yes)$ & "
        r"mean log $p(no)$ & Decision \\ ",
        r"\midrule",
    ]
    for model_summary in model_summaries:
        latency = model_summary["latency"]
        logprobs = model_summary["mean_logprobs"]
        decisions = model_summary["decision_counts"]
        decision = "yes" if decisions["yes"] >= decisions["no"] else "no"
        rows.append(
            f"{_latex_escape(model_summary['model'])} & {latency['count']} & "
            f"{float(latency['median']):.3f} & {float(latency['p05']):.3f} & "
            f"{float(latency['p95']):.3f} & {logprobs['yes']:.3f} & "
            f"{logprobs['no']:.3f} & {_latex_escape(decision)} \\\\"
        )
    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\section*{Interpretation}",
            "The log probabilities are native scores from different model "
            "objectives. This report compares timing and the selected binary "
            "decision; it does not claim that the probabilities are calibrated "
            "or directly comparable.",
            r"\end{document}",
        ]
    )
    return "\n".join(rows) + "\n"
