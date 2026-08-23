from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TypedDict

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
        "ensure_ascii": not bool(1),
        "sort_keys": True,
    }
    if compact:
        options["separators"] = (",", ":")
    else:
        options["indent"] = 2
    if options["ensure_ascii"] is not False:
        raise ValueError("JSON must preserve Unicode")
    return options


def json_line(record: Mapping[str, object]) -> str:
    """Serialize one stable JSONL record."""

    return json.dumps(record, **_json_options(compact=True))


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

    path.write_text(json.dumps(document, **_json_options(compact=False)) + "\n")


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

    model = _latex_escape(summary["model"])
    median = _number_value(summary["median"])
    if median is None:
        raise TypeError("median must be numeric")
    decision = _latex_escape(summary["decision"])
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

    models = _mapping_field(manifest, "models")
    example = _mapping_field(manifest, "example")
    protocol = _mapping_field(manifest, "protocol")
    model_summaries = _model_summaries(summary)
    rows = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.8in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{titlesec}",
        r"\titlespacing*{\section}{0pt}{1.35ex}{0.85ex}",
        r"\title{Binary Land-Use Logprob Benchmark}",
        r"\date{}",
        r"\begin{document}",
        r"\maketitle",
        r"\section*{Objective}",
        "Measure the cost of mapping one land-use sentence to binary yes/no "
        "log probabilities.",
        r"\section*{Example}",
        "Target sentence: ``"
        + _latex_escape(example.get("sentence", "not captured"))
        + "''\\\\",
        "Question: " + _latex_escape(example.get("question", "not captured")),
        r"\section*{Results}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & $n$ & Median ms & P05 ms & P95 ms & mean log $p(yes)$ & "
        r"mean log $p(no)$ & Decision \\ ",
        r"\midrule",
    ]
    for model_summary in model_summaries:
        latency = _mapping_field(model_summary, "latency")
        logprobs = _mapping_field(model_summary, "mean_logprobs")
        decisions = _mapping_field(model_summary, "decision_counts")
        decision = _decision_text(decisions)
        rows.append(
            f"{_latex_escape(model_summary.get('model', 'unknown'))} & "
            f"{_count_text(latency.get('count'))} & "
            f"{_number_text(latency.get('median'))} & "
            f"{_number_text(latency.get('p05'))} & "
            f"{_number_text(latency.get('p95'))} & "
            f"{_number_text(logprobs.get('yes'))} & "
            f"{_number_text(logprobs.get('no'))} & "
            f"{_latex_escape(decision)} \\\\"
        )
    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    rows.extend(_timing_section(model_summaries))
    rows.extend(_runtime_section(manifest, models))
    rows.extend(_comparison_section(model_summaries))
    rows.extend(_reproducibility_section(manifest, models, protocol))
    rows.append(r"\end{document}")
    return "\n".join(rows) + "\n"


def _mapping_field(document: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = document.get(field)
    return value if isinstance(value, Mapping) else {}


def _model_summaries(summary: Mapping[str, object]) -> list[ModelSummary]:
    value = summary.get("models")
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _number_text(value: object) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{float(value):.3f}"
    return "--"


def _count_text(value: object) -> str:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return str(value)
    return "--"


def _decision_text(decisions: Mapping[str, object]) -> str:
    yes = decisions.get("yes")
    no = decisions.get("no")
    if isinstance(yes, int) and isinstance(no, int):
        return "yes" if yes >= no else "no"
    return "--"


def _timing_section(model_summaries: list[ModelSummary]) -> list[str]:
    rows = [
        r"\section*{Timing decomposition}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Tokenization ms & Model ms & Logprob ms \\",
        r"\midrule",
    ]
    for model_summary in model_summaries:
        rows.append(
            f"{_latex_escape(model_summary.get('model', 'unknown'))} & "
            f"{_timing_text(model_summary, 'tokenization')} & "
            f"{_timing_text(model_summary, 'model_time')} & "
            f"{_timing_text(model_summary, 'logprob_time')} \\\\"
        )
    rows.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\smallskip",
            "The encoder measures text encoding and logprob evaluation directly; "
            "it has no decoding phase.",
            "The LLM measurement includes the local server request and one",
            "generated token used to expose the binary logprobs.",
        ]
    )
    return rows


def _timing_text(model_summary: Mapping[str, object], field: str) -> str:
    value = model_summary.get(field)
    if not isinstance(value, Mapping):
        return "--"
    return _number_text(value.get("median"))


def _runtime_section(
    manifest: Mapping[str, object], models: Mapping[str, object]
) -> list[str]:
    encoder_runtime = _backend_runtime(manifest, "encoder")
    llm_runtime = _backend_runtime(manifest, "llm")
    encoder_id = _model_id(models, "encoder")
    llm_id = _model_id(models, "llm")
    return [
        r"\section*{Runtime}",
        _runtime_line("Encoder", encoder_id, encoder_runtime) + r"\\",
        _runtime_line("LLM", llm_id, llm_runtime) + ".",
    ]


def _backend_runtime(
    manifest: Mapping[str, object], backend: str
) -> Mapping[str, object]:
    by_backend = _mapping_field(manifest, "runtime_by_backend")
    selected = by_backend.get(backend)
    if isinstance(selected, Mapping):
        return selected
    return _mapping_field(manifest, "runtime")


def _runtime_line(label: str, model_id: str, runtime: Mapping[str, object]) -> str:
    cuda = _mapping_field(runtime, "cuda")
    details = [
        f"{label} ({model_id}): {_hardware_text(cuda)}",
        f"dtype {_latex_escape(runtime.get('dtype', 'not captured'))}",
        f"Python {_latex_escape(runtime.get('python', 'not captured'))}",
    ]
    details.extend(_runtime_package_details(runtime))
    details.extend(_runtime_cuda_details(cuda))
    return "; ".join(details)


def _hardware_text(cuda: Mapping[str, object]) -> str:
    gpu_text = str(cuda.get("gpu", "not captured"))
    capability = cuda.get("capability")
    if isinstance(capability, list) and len(capability) == 2:
        return f"{gpu_text} (capability {capability[0]}.{capability[1]})"
    return gpu_text


def _runtime_package_details(runtime: Mapping[str, object]) -> list[str]:
    details: list[str] = []
    for field, title in (("torch", "Torch"), ("transformers", "Transformers")):
        value = runtime.get(field)
        if value is not None:
            details.append(f"{title} {_latex_escape(value)}")
    return details


def _runtime_cuda_details(cuda: Mapping[str, object]) -> list[str]:
    details: list[str] = []
    cuda_runtime = cuda.get("runtime")
    if cuda_runtime is not None:
        details.append(f"CUDA {_latex_escape(cuda_runtime)}")
    driver = cuda.get("driver")
    details.append(
        "Driver " + _latex_escape(driver)
        if driver is not None
        else "Driver not captured"
    )
    return details


def _model_id(models: Mapping[str, object], backend: str) -> str:
    return str(_mapping_field(models, backend).get("id", backend))


def _comparison_section(model_summaries: list[ModelSummary]) -> list[str]:
    by_model = {
        str(model_summary.get("model")): model_summary
        for model_summary in model_summaries
    }
    encoder = by_model.get("encoder")
    llm = by_model.get("llm")
    if encoder is None or llm is None:
        return []
    comparison = _comparison_values(encoder, llm)
    if comparison is None:
        return [r"\section*{Comparison}", "Comparison unavailable."]
    return _comparison_lines(comparison)


def _comparison_values(
    encoder: ModelSummary, llm: ModelSummary
) -> (
    tuple[
        float,
        float,
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
    ]
    | None
):
    encoder_latency = _mapping_field(encoder, "latency")
    llm_latency = _mapping_field(llm, "latency")
    encoder_median = _number_value(encoder_latency.get("median"))
    llm_median = _number_value(llm_latency.get("median"))
    if encoder_median is None or llm_median is None or encoder_median <= 0:
        return None
    return (
        encoder_median,
        llm_median,
        _mapping_field(encoder, "decision_counts"),
        _mapping_field(llm, "decision_counts"),
        _mapping_field(encoder, "mean_logprobs"),
        _mapping_field(llm, "mean_logprobs"),
    )


def _comparison_lines(
    values: tuple[
        float,
        float,
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
        Mapping[str, object],
    ],
) -> list[str]:
    (
        encoder_median,
        llm_median,
        encoder_decisions,
        llm_decisions,
        encoder_scores,
        llm_scores,
    ) = values
    ratio = llm_median / encoder_median
    slower = llm_median - encoder_median
    return [
        r"\section*{Comparison}",
        f"The LLM median is {ratio:.3f}\\(\\times\\) the encoder median, or "
        f"{slower:.3f} ms slower per sentence.",
        "Encoder decisions: "
        + _decision_counts_text(encoder_decisions)
        + "; LLM decisions: "
        + _decision_counts_text(llm_decisions)
        + ".",
        "Mean score margins (no--yes): encoder "
        + _margin_text(encoder_scores)
        + "; LLM "
        + _margin_text(llm_scores)
        + ". These native scores are model-specific and are not calibrated "
        + "across models.",
    ]


def _number_value(value: object) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _decision_counts_text(decisions: Mapping[str, object]) -> str:
    return (
        f"{_count_text(decisions.get('yes'))} yes / "
        f"{_count_text(decisions.get('no'))} no"
    )


def _margin_text(scores: Mapping[str, object]) -> str:
    yes = _number_value(scores.get("yes"))
    no = _number_value(scores.get("no"))
    return f"{no - yes:.3f}" if yes is not None and no is not None else "--"


def _reproducibility_section(
    manifest: Mapping[str, object],
    models: Mapping[str, object],
    protocol: Mapping[str, object],
) -> list[str]:
    run_ids = manifest.get("run_ids")
    if isinstance(run_ids, list) and run_ids:
        run = ", ".join(str(item) for item in run_ids)
    else:
        run = str(manifest.get("run_id", "not captured"))
    return [
        r"\section*{Reproducibility}",
        "Run: \\texttt{" + _latex_escape(run) + "}\\\\",
        "Source commit: \\texttt{"
        + _latex_escape(manifest.get("source_commit", "not captured"))
        + "}\\\\",
        "Encoder: "
        + _latex_escape(_model_id(models, "encoder"))
        + " at \\texttt{"
        + _latex_escape(_revision(models, "encoder"))
        + "}\\\\",
        "LLM: "
        + _latex_escape(_model_id(models, "llm"))
        + " at \\texttt{"
        + _latex_escape(_revision(models, "llm"))
        + "}\\\\",
        "llama.cpp revision: \\texttt{"
        + _latex_escape(manifest.get("llama_cpp_revision", "not captured"))
        + "}\\\\",
        "Protocol: "
        + _latex_escape(protocol.get("warmups", "not captured"))
        + " warmups, "
        + _latex_escape(protocol.get("repetitions", "not captured"))
        + " repetitions, prompt cache disabled; Qwen chat template via "
        + "\\texttt{/apply-template}, thinking disabled.\\\\",
    ]


def _revision(models: Mapping[str, object], backend: str) -> str:
    return str(_mapping_field(models, backend).get("revision", "not captured"))
