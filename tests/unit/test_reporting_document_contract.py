from __future__ import annotations

import ast
from pathlib import Path
from typing import cast

import pytest

import compute_cost_encoders_llms.benchmark.reporting as reporting_module
from compute_cost_encoders_llms.benchmark.reporting import (
    MeasurementError,
    ModelSummary,
    _comparison_values,
    _decision_text,
    _group_records,
    _hardware_text,
    _latex_escape,
    _margin_text,
    _model_id,
    _number_text,
    _runtime_line,
    json_line,
    render_latex_document,
    render_latex_summary,
    write_json,
)


def test_json_options_declares_unicode_policy_as_literal() -> None:
    source = Path(reporting_module.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.AnnAssign)
        and isinstance(node.target, ast.Name)
        and node.target.id == "options"
    )
    assert isinstance(assignment.value, ast.Dict)
    ensure_ascii = next(
        value
        for key, value in zip(
            assignment.value.keys, assignment.value.values, strict=True
        )
        if isinstance(key, ast.Constant) and key.value == "ensure_ascii"
    )
    assert isinstance(ensure_ascii, ast.Constant)
    assert ensure_ascii.value is False


def test_number_text_uses_shared_numeric_conversion(monkeypatch) -> None:
    monkeypatch.setattr(reporting_module, "_number_value", lambda _value: 12.5)

    assert reporting_module._number_text(object()) == "12.500"


def test_render_latex_document_preserves_complete_public_report_contract() -> None:
    manifest: dict[str, object] = {
        "source_commit": "a" * 40,
        "llama_cpp_revision": "d" * 40,
        "run_ids": ["encoder-run", "llm-run"],
        "protocol": {"warmups": 8, "repetitions": 64, "prompt_cache": False},
        "example": {
            "sentence": (
                "A public park with grass, trees, and walking paths occupies "
                "the parcel."
            ),
            "question": "Is this sentence relevant for a land use description?",
            "labels": ["yes", "no"],
        },
        "models": {
            "encoder": {"id": "jhu-clsp/mmBERT-base", "revision": "b" * 40},
            "llm": {
                "id": "ggml-org/Qwen3.6-27B-GGUF",
                "revision": "c" * 40,
            },
        },
        "runtime_by_backend": {
            "encoder": {
                "dtype": "float16",
                "python": "3.12.8",
                "torch": "2.6.0",
                "transformers": "4.55.4",
                "cuda": {
                    "gpu": "NVIDIA A100",
                    "capability": [8, 0],
                    "runtime": "12.4",
                    "driver": "550.54",
                },
            },
            "llm": {
                "dtype": "Q4_K_M",
                "python": "3.12.8",
                "cuda": {
                    "gpu": "NVIDIA A100",
                    "capability": [8, 0],
                    "runtime": "12.4",
                    "driver": "550.54",
                },
            },
        },
    }
    summary: dict[str, object] = {
        "models": [
            {
                "model": "encoder",
                "latency": {
                    "count": 64,
                    "median": 10.5,
                    "p05": 10.4,
                    "p95": 10.6,
                },
                "tokenization": {
                    "count": 64,
                    "median": 2.1,
                    "p05": 2.09,
                    "p95": 2.11,
                },
                "model_time": {
                    "count": 64,
                    "median": 7.8,
                    "p05": 7.79,
                    "p95": 7.81,
                },
                "logprob_time": {
                    "count": 64,
                    "median": 0.6,
                    "p05": 0.59,
                    "p95": 0.61,
                },
                "mean_logprobs": {"yes": -0.2, "no": -1.4},
                "decision_counts": {"yes": 64, "no": 0},
            },
            {
                "model": "llm",
                "latency": {
                    "count": 64,
                    "median": 123.4,
                    "p05": 123.3,
                    "p95": 123.5,
                },
                "tokenization": None,
                "model_time": None,
                "logprob_time": {
                    "count": 64,
                    "median": 4.2,
                    "p05": 4.19,
                    "p95": 4.21,
                },
                "mean_logprobs": {"yes": -0.3, "no": -1.1},
                "decision_counts": {"yes": 64, "no": 0},
            },
        ]
    }

    expected_lines = [
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
        "Target sentence: ``A public park with grass, trees, and walking paths "
        "occupies the parcel.''\\\\",
        "Question: Is this sentence relevant for a land use description?",
        r"\section*{Results}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Model & $n$ & Median ms & P05 ms & P95 ms & mean log $p(yes)$ & "
        r"mean log $p(no)$ & Decision \\ ",
        r"\midrule",
        r"encoder & 64 & 10.500 & 10.400 & 10.600 & -0.200 & -1.400 & yes \\",
        r"llm & 64 & 123.400 & 123.300 & 123.500 & -0.300 & -1.100 & yes \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\section*{Timing decomposition}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Model & Tokenization ms & Model ms & Logprob ms \\",
        r"\midrule",
        r"encoder & 2.100 & 7.800 & 0.600 \\",
        r"llm & -- & -- & 4.200 \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\smallskip",
        "The encoder measures text encoding and logprob evaluation directly; "
        "it has no decoding phase.",
        "The LLM measurement includes the local server request and one",
        "generated token used to expose the binary logprobs.",
        r"\section*{Runtime}",
        "Encoder (jhu-clsp/mmBERT-base): NVIDIA A100 (capability 8.0); dtype "
        "float16; Python 3.12.8; Torch 2.6.0; Transformers 4.55.4; CUDA "
        "12.4; Driver 550.54\\\\",
        "LLM (ggml-org/Qwen3.6-27B-GGUF): NVIDIA A100 (capability 8.0); dtype "
        r"Q4\_K\_M; Python 3.12.8; CUDA 12.4; Driver 550.54.",
        r"\section*{Comparison}",
        r"The LLM median is 11.752\(\times\) the encoder median, or "
        r"112.900 ms slower per sentence.",
        "Encoder decisions: 64 yes / 0 no; LLM decisions: 64 yes / 0 no.",
        "Mean score margins (no--yes): encoder -1.200; LLM -0.800. These "
        "native scores are model-specific and are not calibrated across models.",
        r"\section*{Reproducibility}",
        r"Run: \texttt{encoder-run, llm-run}\\",
        r"Source commit: \texttt{aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa}\\",
        r"Encoder: jhu-clsp/mmBERT-base at "
        r"\texttt{bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb}\\",
        r"LLM: ggml-org/Qwen3.6-27B-GGUF at "
        r"\texttt{cccccccccccccccccccccccccccccccccccccccc}\\",
        r"llama.cpp revision: \texttt{" + "d" * 40 + r"}\\",
        r"Protocol: 8 warmups, 64 repetitions, prompt cache disabled; Qwen chat "
        r"template via \texttt{/apply-template}, thinking disabled.\\",
        r"\end{document}",
    ]

    assert render_latex_document(manifest, summary) == "\n".join(expected_lines) + "\n"


def test_render_latex_document_preserves_missing_metadata_defaults() -> None:
    manifest: dict[str, object] = {
        "run_id": "single-run",
        "models": {
            "encoder": {"id": "enc"},
            "llm": {"id": "llm"},
        },
        "runtime": {
            "dtype": "float32",
            "python": "3.12.8",
            "cuda": {"gpu": "CPU"},
        },
    }
    summary: dict[str, object] = {
        "models": [
            {
                "model": "encoder",
                "latency": {
                    "count": 1,
                    "median": 1.0,
                    "p05": 1.0,
                    "p95": 1.0,
                },
                "logprob_time": {"median": 0.1},
                "mean_logprobs": {"yes": -0.1, "no": -1.0},
                "decision_counts": {"yes": 1, "no": 0},
            },
            {
                "latency": {
                    "count": 1,
                    "median": 2.0,
                    "p05": 2.0,
                    "p95": 2.0,
                },
                "logprob_time": {"median": 0.2},
                "mean_logprobs": {"yes": -0.2, "no": -0.8},
                "decision_counts": {"yes": 1, "no": 0},
            },
        ]
    }

    document = render_latex_document(manifest, summary)

    assert r"Run: \texttt{single-run}\\" in document
    assert r"Source commit: \texttt{not captured}\\" in document
    assert r"Encoder: enc at \texttt{not captured}\\" in document
    assert r"LLM: llm at \texttt{not captured}\\" in document
    assert "Target sentence: ``not captured''\\" in document
    assert "Question: not captured" in document
    assert r"llama.cpp revision: \texttt{not captured}\\" in document
    assert "Protocol: not captured warmups, not captured repetitions" in document
    assert (
        "Encoder (enc): CPU; dtype float32; Python 3.12.8; Driver not captured"
        in document
    )
    assert "LLM (llm): CPU; dtype float32; Python 3.12.8;" in document
    assert "encoder & -- & -- & 0.100" in document
    assert "unknown & 1 & 2.000 & 2.000 & 2.000 & -0.200 & -0.800 & yes" in document
    assert "unknown & -- & -- & 0.200" in document
    assert r"\section*{Comparison}" not in document
    assert "Comparison unavailable." not in document

    missing_run_manifest = dict(manifest)
    missing_run_manifest.pop("run_id")
    missing_run_document = render_latex_document(missing_run_manifest, summary)
    assert r"Run: \texttt{not captured}\\" in missing_run_document

    non_list_run_manifest = dict(manifest)
    non_list_run_manifest["run_ids"] = "invalid"
    non_list_document = render_latex_document(non_list_run_manifest, summary)
    assert r"Run: \texttt{single-run}\\" in non_list_document


def test_reporting_helpers_preserve_exact_formatting_and_edge_cases() -> None:
    expected_summary = "\n".join(
        [
            r"\begin{tabular}{lr}",
            r"Model & encoder \\",
            r"Median ms & 12.500 \\",
            r"Decision & yes \\",
            r"\end{tabular}",
        ]
    )
    summary_input = {"model": "encoder", "median": 12.5, "decision": "yes"}
    assert render_latex_summary(summary_input) == expected_summary
    assert _latex_escape(chr(92) + "&%$#_{}") == r"\textbackslash\{\}\&\%\$\#\_\{\}"

    assert _hardware_text({}) == "not captured"
    assert _hardware_text({"gpu": "A100", "capability": [8, 0]}) == (
        "A100 (capability 8.0)"
    )
    assert _hardware_text({"gpu": "A100", "capability": [8]}) == "A100"
    assert _hardware_text({"gpu": "A100", "capability": (8, 0)}) == "A100"
    assert _runtime_line("Encoder", "enc", {}) == (
        "Encoder (enc): not captured; dtype not captured; Python not captured; "
        "Driver not captured"
    )
    assert _model_id({}, "encoder") == "encoder"

    assert _number_text("missing") == "--"
    assert _margin_text({"yes": -1.0}) == "--"
    assert _margin_text({"no": -1.0}) == "--"
    assert _decision_text({"yes": 1, "no": 1}) == "yes"
    zero_encoder = cast(ModelSummary, {"latency": {"median": 0.0}})
    unit_encoder = cast(ModelSummary, {"latency": {"median": 1.0}})
    positive_llm = cast(ModelSummary, {"latency": {"median": 1.0}})
    assert _comparison_values(zero_encoder, positive_llm) is None
    assert _comparison_values(unit_encoder, positive_llm) is not None

    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"yes": -0.1, "no": -2.2},
    }
    with pytest.raises(MeasurementError, match=r"^no measurements$"):
        _group_records([])
    with pytest.raises(MeasurementError, match=r"^duplicate measurement repetition$"):
        _group_records([record, record])
    with pytest.raises(TypeError, match=r"^median must be numeric$"):
        render_latex_summary(
            {"model": "encoder", "median": "missing", "decision": "yes"}
        )


def test_json_serialization_preserves_unescaped_unicode(tmp_path: Path) -> None:
    path = tmp_path / "unicode.json"

    write_json(path, {"é": "é"})

    assert json_line({"é": "é"}) == '{"é":"é"}'
    assert path.read_text() == '{\n  "é": "é"\n}\n'
