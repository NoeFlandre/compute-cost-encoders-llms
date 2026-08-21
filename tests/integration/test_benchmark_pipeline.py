from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from scripts.render_report import render_report

import compute_cost_encoders_llms.benchmark.cli as cli_module
from compute_cost_encoders_llms.benchmark.example import (
    candidate_label_forms,
    candidate_labels,
)
from compute_cost_encoders_llms.benchmark.llm import LlamaClient
from compute_cost_encoders_llms.benchmark.runtime import build_runtime_metadata


class FakeTensor:
    def __init__(self, value: object) -> None:
        self.value = value

    def __getitem__(self, index: int | slice) -> FakeTensor:
        return FakeTensor(cast(Sequence[object], self.value)[index])

    def to(self, _device: str) -> FakeTensor:
        return self

    def tolist(self) -> object:
        return self.value

    def detach(self) -> FakeTensor:
        return self

    def float(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self


class FakeTokenizer:
    mask_token = "<mask>"
    mask_token_id = 99

    def __call__(self, text: str, **_kwargs: object) -> dict[str, object]:
        candidate_ids = {
            form: [index]
            for index, form in enumerate(
                form
                for label in candidate_labels()
                for form in candidate_label_forms(label)
            )
        }
        if text in candidate_ids:
            return {"input_ids": candidate_ids[text]}
        return {
            "input_ids": FakeTensor([[10, 99, 11]]),
            "attention_mask": FakeTensor([[1, 1, 1]]),
        }


class FakeTorch:
    class _InferenceMode:
        def __enter__(self) -> FakeTorch._InferenceMode:
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

    @staticmethod
    def inference_mode() -> FakeTorch._InferenceMode:
        return FakeTorch._InferenceMode()


class FakeModel:
    def __call__(self, **_inputs: object) -> SimpleNamespace:
        return SimpleNamespace(
            logits=FakeTensor(
                [
                    [
                        [0.0] * 7,
                        [0.0, 2.0, -1.0, 1.5, -2.0, 1.0, -3.0],
                        [0.0] * 7,
                    ]
                ]
            )
        )


def _write_config(path: Path) -> Path:
    path.write_text(
        """
[benchmark]
encoder_revision = "c5955035435e2bf121cde7f3c8863ef52ff35d82"
llm_revision = "8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8"
llama_cpp_revision = "6503355df0eb4f65875012523263c302fe0088c1"
device = "cpu"
repetitions = 2
warmups = 1
""".strip()
    )
    return path


def _fake_encoder_loader(_config: object) -> cli_module.LoadedEncoder:
    return cli_module.LoadedEncoder(
        tokenizer=FakeTokenizer(),
        model=FakeModel(),
        torch_module=FakeTorch(),
        runtime=build_runtime_metadata(
            torch_module=None,
            dtype="float32",
            llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
            llm_filename="Qwen3.6-27B-Q4_K_M.gguf",
            dependency_lock_sha256=None,
        ),
    )


def _fake_llama_client(_url: str) -> LlamaClient:
    def request(url: str, _payload: object, _timeout: float) -> dict[str, object]:
        if url.endswith("/apply-template"):
            return {"prompt": "rendered prompt"}
        return {
            "completion_probabilities": [
                {
                    "probs": [
                        {
                            "token": "yes",
                            "logprob": -0.25,
                            "top_logprobs": [{"token": "no", "logprob": -1.25}],
                        }
                    ]
                }
            ],
            "usage": {"prompt_tokens": 4},
        }

    return LlamaClient(_url, request=request)


def test_config_backend_measurement_report_pipeline(tmp_path, monkeypatch) -> None:
    config_path = _write_config(tmp_path / "benchmark.toml")
    monkeypatch.setattr(cli_module, "_load_encoder", _fake_encoder_loader)
    monkeypatch.setattr(cli_module, "LlamaClient", _fake_llama_client)

    encoder_dir = tmp_path / "encoder"
    llm_dir = tmp_path / "llm"
    cli_module.run(config_path, encoder_dir, "encoder", "run-001")
    cli_module.run(config_path, llm_dir, "llm", "run-001")

    monkeypatch.setenv("GRID5000_CONFIG_REVISION", "sha256:config")
    monkeypatch.setenv("GRID5000_DATASET_REVISION", "made-up-landuse-example-v1")
    monkeypatch.setenv("GRID5000_MODEL_REVISION", "c" * 40)
    monkeypatch.setenv("GRID5000_ARTIFACT_PREFIX", "runs/run-001")
    report = tmp_path / "report.tex"
    checkpoint = tmp_path / "checkpoint.json"
    render_report(encoder_dir, llm_dir, report, checkpoint=checkpoint)

    encoder_manifest = json.loads((encoder_dir / "manifest.json").read_text())
    llm_measurement = json.loads(
        (llm_dir / "measurements.jsonl").read_text().splitlines()[0]
    )
    report_text = report.read_text()

    assert (
        encoder_manifest["models"]["encoder"]["revision"]
        == "c5955035435e2bf121cde7f3c8863ef52ff35d82"
    )
    assert encoder_manifest["runtime"]["dtype"] == "float32"
    assert encoder_manifest["runtime"]["python"]
    assert encoder_manifest["runtime"]["llama_cpp_revision"] == (
        "6503355df0eb4f65875012523263c302fe0088c1"
    )
    assert len(encoder_manifest["dependency_lock_sha256"]) == 64
    assert llm_measurement["tokenization_ms"] is None
    assert "encoder" in report_text
    assert "llm" in report_text
    assert json.loads(checkpoint.read_text())["complete"] is True
