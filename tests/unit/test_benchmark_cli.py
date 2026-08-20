from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

import compute_cost_encoders_llms.benchmark.cli as cli_module
from compute_cost_encoders_llms.benchmark.cli import (
    _load_encoder,
    _source_commit,
    build_manifest,
    score_record,
)
from compute_cost_encoders_llms.benchmark.config import BenchmarkConfig
from compute_cost_encoders_llms.benchmark.encoder import EncoderScore


def test_score_record_normalizes_backend_score() -> None:
    score = EncoderScore(
        logprobs={"yes": -0.1, "no": -2.2},
        tokenization_ms=1.0,
        model_ms=2.0,
        logprob_ms=0.1,
        text_to_logprob_ms=3.1,
        input_tokens=12,
    )

    record = score_record("encoder", 3, score)

    assert record["model"] == "encoder"
    assert record["repetition"] == 3
    assert record["input_tokens"] == 12
    assert record["logprobs"] == {"yes": -0.1, "no": -2.2}


def test_build_manifest_captures_reproducibility_fields() -> None:
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
    )

    manifest = build_manifest(
        config,
        run_id="smoke-001",
        source_commit="a" * 40,
        hardware={"gpu": "test"},
    )

    assert manifest["run_id"] == "smoke-001"
    assert manifest["source_commit"] == "a" * 40
    models = cast(dict[str, dict[str, str]], manifest["models"])
    assert models["llm"]["revision"] == config.llm_revision
    assert manifest["hardware"] == {"gpu": "test"}


def test_source_commit_accepts_grid5000_override(monkeypatch) -> None:
    monkeypatch.setenv("GRID5000_SOURCE_COMMIT", "b" * 40)

    assert _source_commit() == "b" * 40


def test_load_encoder_uses_pinned_cuda_model(monkeypatch) -> None:
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
        device="cuda",
    )
    tokenizer = object()

    class Model:
        def __init__(self) -> None:
            self.device = ""
            self.evaluated = False

        def to(self, device: str) -> None:
            self.device = device

        def eval(self) -> None:
            self.evaluated = True

    model = Model()
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        bfloat16="bfloat16",
        float32="float32",
    )
    transformers_module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: tokenizer
        ),
        AutoModelForMaskedLM=SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: model
        ),
    )
    modules = {"torch": torch_module, "transformers": transformers_module}
    monkeypatch.setattr(cli_module.importlib, "import_module", modules.__getitem__)

    loaded_tokenizer, loaded_model, loaded_torch = _load_encoder(config)

    assert loaded_tokenizer is tokenizer
    assert loaded_model is model
    assert loaded_torch is torch_module
    assert model.device == "cuda"
    assert model.evaluated is True


def test_load_encoder_rejects_missing_dependencies(monkeypatch) -> None:
    def missing(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(cli_module.importlib, "import_module", missing)
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
    )

    with pytest.raises(RuntimeError, match="dependencies"):
        _load_encoder(config)


def test_load_encoder_rejects_unavailable_cuda(monkeypatch) -> None:
    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        bfloat16="bfloat16",
        float32="float32",
    )
    transformers_module = SimpleNamespace()
    modules = {"torch": torch_module, "transformers": transformers_module}
    monkeypatch.setattr(cli_module.importlib, "import_module", modules.__getitem__)
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
        device="cuda",
    )

    with pytest.raises(RuntimeError, match="CUDA"):
        _load_encoder(config)
