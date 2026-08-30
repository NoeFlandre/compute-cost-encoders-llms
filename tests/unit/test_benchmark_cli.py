from __future__ import annotations

import hashlib
from collections.abc import Callable
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar, cast

import pytest

import compute_cost_encoders_llms.benchmark.cli as cli_module
import compute_cost_encoders_llms.benchmark.runtime as runtime_module
from compute_cost_encoders_llms.benchmark.cli import (
    LoadedEncoder,
    _as_model,
    _as_tokenizer,
    _as_torch,
    _dependency_lock_digest,
    _encoder_records,
    _hardware,
    _llm_records,
    _load_encoder,
    _optional_torch_module,
    _source_commit,
    run,
)
from compute_cost_encoders_llms.benchmark.config import BenchmarkConfig
from compute_cost_encoders_llms.benchmark.encoder import (
    EncoderScore,
    ModelLike,
    TokenizerLike,
    TorchLike,
)
from compute_cost_encoders_llms.benchmark.llm import LlamaScore
from compute_cost_encoders_llms.benchmark.manifest import build_manifest
from compute_cost_encoders_llms.benchmark.measurement import (
    TimedValue,
    score_record,
)
from compute_cost_encoders_llms.benchmark.runtime import (
    CudaApi,
    _cuda,
    _device_name,
    _driver_version,
    build_runtime_metadata,
    quantization_from_filename,
    select_encoder_precision,
)


class RecordingModel:
    def __init__(self) -> None:
        self.device = ""
        self.evaluated = False

    def to(self, device: str) -> None:
        self.device = device

    def eval(self) -> None:
        self.evaluated = True

    def __call__(self, **_inputs: object) -> object:
        return object()


class RecordingTokenizer:
    mask_token = "<mask>"
    mask_token_id = 99

    def __call__(self, _text: str, **_kwargs: object) -> dict[str, object]:
        return {}


class RecordingLoader:
    def __init__(self, value: object) -> None:
        self.value = value
        self.calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def __call__(self, *args: object, **kwargs: object) -> object:
        self.calls.append((args, kwargs))
        return self.value


class RecordingLlamaClient:
    instances: ClassVar[list[str]] = []
    seeds: ClassVar[list[int]] = []
    result: ClassVar[LlamaScore]

    def __init__(self, base_url: str) -> None:
        self.instances.append(base_url)

    def score(self, seed: int) -> LlamaScore:
        self.seeds.append(seed)
        return self.result


def measure_encoder_once(
    operation: Callable[[], EncoderScore],
    *,
    warmups: int,
    repetitions: int,
    expected_warmups: int,
    expected_repetitions: int,
) -> list[TimedValue[EncoderScore]]:
    assert warmups == expected_warmups
    assert repetitions == expected_repetitions
    return [TimedValue(value=operation(), elapsed_ms=4.0)]


def measure_llm_once(
    operation: Callable[[], LlamaScore],
    *,
    warmups: int,
    repetitions: int,
    expected_warmups: int,
    expected_repetitions: int,
) -> list[TimedValue[LlamaScore]]:
    assert warmups == expected_warmups
    assert repetitions == expected_repetitions
    return [TimedValue(value=operation(), elapsed_ms=5.0)]


def capture_loaded_encoder(
    calls: list[tuple[BenchmarkConfig, str | None]],
    loaded: LoadedEncoder,
    value: BenchmarkConfig,
    dependency_lock_sha256: str | None = None,
) -> LoadedEncoder:
    calls.append((value, dependency_lock_sha256))
    return loaded


def capture_encoder_score(
    calls: list[tuple[object, object, object, str]],
    score: EncoderScore,
    tokenizer: object,
    model: object,
    torch_module: object,
    device: str,
    *,
    candidate_token_ids: object = None,
) -> EncoderScore:
    calls.append((tokenizer, model, torch_module, device))
    return score


def capture_quantization(calls: list[str], filename: str) -> str:
    calls.append(filename)
    return "Q4_K_M"


def capture_runtime_metadata(
    calls: list[dict[str, object]], **kwargs: object
) -> dict[str, object]:
    calls.append(kwargs)
    return {"runtime": "captured"}


def capture_config(
    calls: dict[str, object], config: BenchmarkConfig, path: Path
) -> BenchmarkConfig:
    calls["config_path"] = path
    return config


def capture_lock_digest(calls: dict[str, object], path: Path) -> str:
    calls["digest_path"] = path
    return "lock-sha"


def capture_backend_records(
    calls: dict[str, object],
    records: list[dict[str, object]],
    actual_config: BenchmarkConfig,
    digest: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    calls["backend_args"] = (actual_config, digest)
    return records, {"dtype": "test"}


def fail_if_backend_called(
    *_args: object, **_kwargs: object
) -> tuple[list[dict[str, object]], dict[str, object]]:
    raise AssertionError("run selected the wrong backend")


def capture_manifest(
    calls: dict[str, object],
    actual_config: BenchmarkConfig,
    **kwargs: object,
) -> dict[str, object]:
    calls["manifest_args"] = (actual_config, kwargs)
    return {"manifest": True}


def capture_json(
    calls: list[tuple[Path, object]], path: Path, document: object
) -> None:
    calls.append((path, document))


def capture_measurement_artifacts(
    calls: list[tuple[Path, object]], path: Path, document: object
) -> None:
    calls.append((path, document))


def test_build_manifest_captures_reproducibility_fields(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
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
    assert manifest["schema_version"] == 2
    assert set(manifest) == {
        "schema_version",
        "run_id",
        "source_commit",
        "seed",
        "example",
        "models",
        "llama_cpp_revision",
        "runtime",
        "dependency_lock_sha256",
        "protocol",
        "hardware",
    }
    assert manifest["example"] == {
        "sentence": (
            "A public park with grass, trees, and walking paths occupies the parcel."
        ),
        "question": "Is this sentence relevant for a land use description?",
        "labels": ("yes", "no"),
        "label_forms": {
            "yes": ("yes", "Yes", "YES"),
            "no": ("no", "No", "NO"),
        },
    }
    protocol = cast(dict[str, object], manifest["protocol"])
    assert protocol == {
        "warmups": config.warmups,
        "repetitions": config.repetitions,
        "batch_size": 1,
        "generated_tokens": 1,
        "prompt_cache": False,
        "encoder_answer_marker": "Answer: <mask>",
        "llm_template_endpoint": "/apply-template",
        "llm_reasoning": False,
    }
    models = cast(dict[str, dict[str, str]], manifest["models"])
    assert models == {
        "encoder": {
            "id": config.encoder_model,
            "revision": config.encoder_revision,
        },
        "llm": {
            "id": config.llm_model,
            "revision": config.llm_revision,
            "filename": config.llm_filename,
        },
    }
    assert manifest["runtime"] == {}
    assert manifest["dependency_lock_sha256"] is None
    assert manifest["hardware"] == {"gpu": "test"}
    hardware_snapshot = _hardware()
    assert set(hardware_snapshot) == {"platform", "python", "cuda_visible_devices"}
    assert hardware_snapshot["cuda_visible_devices"] == "0"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert _hardware()["cuda_visible_devices"] == ""


def test_source_commit_accepts_grid5000_override(monkeypatch) -> None:
    monkeypatch.setenv("GRID5000_SOURCE_COMMIT", "b" * 40)

    assert _source_commit() == "b" * 40


def test_source_commit_uses_exact_git_probe_when_not_overridden(monkeypatch) -> None:
    monkeypatch.delenv("GRID5000_SOURCE_COMMIT", raising=False)
    calls: list[tuple[list[str], dict[str, object]]] = []

    def check_output(command: list[str], **kwargs: object) -> str:
        calls.append((command, kwargs))
        assert command == ["git", "rev-parse", "HEAD"]
        assert kwargs == {"text": True, "encoding": "utf-8"}
        return "c" * 40 + "\n"

    monkeypatch.setattr(cli_module.subprocess, "check_output", check_output)

    assert _source_commit() == "c" * 40
    assert calls == [
        (["git", "rev-parse", "HEAD"], {"text": True, "encoding": "utf-8"})
    ]


def test_optional_torch_module_imports_exact_name_or_returns_none(monkeypatch) -> None:
    sentinel = object()
    calls: list[str] = []

    def import_module(name: str) -> object:
        calls.append(name)
        return sentinel

    monkeypatch.setattr(cli_module.importlib, "import_module", import_module)
    assert _optional_torch_module() is sentinel
    assert calls == ["torch"]

    def missing(_name: str) -> object:
        raise ImportError("not installed")

    monkeypatch.setattr(cli_module.importlib, "import_module", missing)
    assert _optional_torch_module() is None


def test_load_encoder_uses_pinned_cuda_model(monkeypatch) -> None:
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
        device="cuda",
    )
    tokenizer = RecordingTokenizer()

    model = RecordingModel()
    tokenizer_loader = RecordingLoader(tokenizer)
    model_loader = RecordingLoader(model)
    runtime_calls: list[dict[str, object]] = []

    torch_module = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: True),
        bfloat16="bfloat16",
        float32="float32",
        inference_mode=lambda: None,
    )
    transformers_module = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer_loader),
        AutoModelForMaskedLM=SimpleNamespace(from_pretrained=model_loader),
    )
    modules = {"torch": torch_module, "transformers": transformers_module}
    monkeypatch.setattr(cli_module.importlib, "import_module", modules.__getitem__)
    monkeypatch.setattr(
        cli_module,
        "build_runtime_metadata",
        partial(capture_runtime_metadata, runtime_calls),
    )

    loaded = _load_encoder(config, "lock-sha")

    assert loaded.tokenizer is tokenizer
    assert loaded.model is model
    assert loaded.torch_module is torch_module
    assert loaded.runtime == {"runtime": "captured"}
    assert model.device == "cuda"
    assert model.evaluated is True
    assert tokenizer_loader.calls == [
        (
            (config.encoder_model,),
            {"revision": config.encoder_revision, "trust_remote_code": False},
        )
    ]
    assert model_loader.calls == [
        (
            (config.encoder_model,),
            {
                "revision": config.encoder_revision,
                "trust_remote_code": False,
                "torch_dtype": "float32",
            },
        )
    ]
    assert runtime_calls == [
        {
            "torch_module": torch_module,
            "dtype": "float32",
            "llama_cpp_revision": config.llama_cpp_revision,
            "llm_filename": config.llm_filename,
            "dependency_lock_sha256": "lock-sha",
        }
    ]


def test_load_encoder_rejects_missing_dependencies(monkeypatch) -> None:
    def missing(_name: str) -> object:
        raise ImportError("missing")

    monkeypatch.setattr(cli_module.importlib, "import_module", missing)
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
    )

    with pytest.raises(
        RuntimeError,
        match=r"^encoder dependencies are not installed$",
    ):
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


def test_loaded_encoder_adapters_fail_closed_with_stable_errors() -> None:
    with pytest.raises(
        RuntimeError,
        match=r"^loaded tokenizer does not implement the encoder interface$",
    ):
        _as_tokenizer(object())
    with pytest.raises(
        RuntimeError,
        match=r"^loaded model does not implement the encoder interface$",
    ):
        _as_model(object())
    with pytest.raises(
        RuntimeError,
        match=r"^loaded torch module does not implement the encoder interface$",
    ):
        _as_torch(object())


def test_encoder_records_preserve_config_inputs_and_runtime_metadata(
    monkeypatch,
) -> None:
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
        device="cuda",
        repetitions=3,
        warmups=2,
    )
    tokenizer = cast(TokenizerLike, object())
    model = cast(ModelLike, object())
    torch_module = cast(TorchLike, object())
    loaded = LoadedEncoder(
        tokenizer=tokenizer,
        model=model,
        torch_module=torch_module,
        runtime={"dtype": "float32", "dependency_lock_sha256": "lock-sha"},
    )
    load_calls: list[tuple[BenchmarkConfig, str | None]] = []
    score_calls: list[tuple[object, object, object, str]] = []

    score = EncoderScore(
        logprobs={"yes": -0.1, "no": -2.2},
        tokenization_ms=1.0,
        model_ms=2.0,
        logprob_ms=0.1,
        text_to_logprob_ms=3.1,
        input_tokens=12,
    )

    monkeypatch.setattr(
        cli_module,
        "_load_encoder",
        partial(capture_loaded_encoder, load_calls, loaded),
    )
    monkeypatch.setattr(
        cli_module,
        "build_candidate_token_ids",
        lambda _tokenizer: {"yes": (1, 3, 5), "no": (2, 4, 6)},
    )
    monkeypatch.setattr(
        cli_module,
        "score_transformers_once",
        partial(
            capture_encoder_score,
            score_calls,
            score,
        ),
    )
    monkeypatch.setattr(
        cli_module,
        "measure_repetitions",
        partial(
            measure_encoder_once,
            expected_warmups=config.warmups,
            expected_repetitions=config.repetitions,
        ),
    )

    records, runtime = _encoder_records(config, "lock-sha")

    assert load_calls == [(config, "lock-sha")]
    assert score_calls == [(tokenizer, model, torch_module, "cuda")]
    assert records == [{**score_record("encoder", 0, score)}]
    assert runtime == {"dtype": "float32", "dependency_lock_sha256": "lock-sha"}


def test_encoder_records_pass_precomputed_candidate_ids_to_repetitions(
    monkeypatch,
) -> None:
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
        repetitions=1,
        warmups=0,
    )
    tokenizer = cast(TokenizerLike, object())
    loaded = LoadedEncoder(
        tokenizer=tokenizer,
        model=cast(ModelLike, object()),
        torch_module=cast(TorchLike, object()),
        runtime={},
    )
    candidate_token_ids: dict[str, tuple[int, ...]] = {
        "yes": (1, 3, 5),
        "no": (2, 4, 6),
    }
    score = EncoderScore(
        logprobs={"yes": -0.1, "no": -2.2},
        tokenization_ms=1.0,
        model_ms=2.0,
        logprob_ms=0.1,
        text_to_logprob_ms=3.1,
        input_tokens=12,
    )
    prepare_calls: list[object] = []
    seen: list[object] = []

    def capture_score(
        _tokenizer: object,
        _model: object,
        _torch_module: object,
        _device: str,
        *,
        candidate_token_ids: object = None,
    ) -> EncoderScore:
        seen.append(candidate_token_ids)
        return score

    def prepare_candidate_ids(_tokenizer: object) -> dict[str, tuple[int, ...]]:
        prepare_calls.append(_tokenizer)
        return candidate_token_ids

    monkeypatch.setattr(cli_module, "_load_encoder", lambda *_args: loaded)
    monkeypatch.setattr(
        cli_module,
        "build_candidate_token_ids",
        prepare_candidate_ids,
        raising=False,
    )
    monkeypatch.setattr(cli_module, "score_transformers_once", capture_score)
    monkeypatch.setattr(
        cli_module,
        "measure_repetitions",
        partial(
            measure_encoder_once,
            expected_warmups=config.warmups,
            expected_repetitions=config.repetitions,
        ),
    )

    _encoder_records(config, None)

    assert prepare_calls == [tokenizer]
    assert seen == [candidate_token_ids]


def test_llm_records_preserve_seed_and_runtime_metadata(monkeypatch) -> None:
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
        llama_base_url="http://llama.test:8080",
        repetitions=2,
        warmups=1,
    )
    score = LlamaScore(
        logprobs={"yes": -0.1, "no": -2.2},
        tokenization_ms=None,
        model_ms=2.0,
        logprob_ms=0.1,
        text_to_logprob_ms=3.1,
        input_tokens=12,
    )
    client_calls: list[str] = []
    seed_calls: list[int] = []
    RecordingLlamaClient.instances = client_calls
    RecordingLlamaClient.seeds = seed_calls
    RecordingLlamaClient.result = score

    torch_sentinel = object()
    quantization_calls: list[str] = []
    metadata_calls: list[dict[str, object]] = []

    monkeypatch.setattr(cli_module, "LlamaClient", RecordingLlamaClient)
    monkeypatch.setattr(
        cli_module,
        "measure_repetitions",
        partial(
            measure_llm_once,
            expected_warmups=config.warmups,
            expected_repetitions=config.repetitions,
        ),
    )
    monkeypatch.setattr(cli_module, "_optional_torch_module", lambda: torch_sentinel)
    monkeypatch.setattr(
        cli_module,
        "quantization_from_filename",
        partial(capture_quantization, quantization_calls),
    )
    monkeypatch.setattr(
        cli_module,
        "build_runtime_metadata",
        partial(capture_runtime_metadata, metadata_calls),
    )

    records, runtime = _llm_records(config, "lock-sha")

    assert client_calls == [config.llama_base_url]
    assert seed_calls == [config.seed]
    assert quantization_calls == [config.llm_filename]
    assert metadata_calls == [
        {
            "torch_module": torch_sentinel,
            "dtype": "Q4_K_M",
            "llama_cpp_revision": config.llama_cpp_revision,
            "llm_filename": config.llm_filename,
            "dependency_lock_sha256": "lock-sha",
        }
    ]
    assert records == [{**score_record("llm", 0, score)}]
    assert runtime == {"runtime": "captured"}


@pytest.mark.parametrize("backend", ["encoder", "llm"])
def test_run_writes_canonical_artifacts_and_manifest_contract(
    tmp_path, monkeypatch, backend: str
) -> None:
    config = BenchmarkConfig(
        encoder_revision="c5955035435e2bf121cde7f3c8863ef52ff35d82",
        llm_revision="8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8",
        llama_cpp_revision="6503355df0eb4f65875012523263c302fe0088c1",
    )
    config_path = tmp_path / "config.toml"
    config_path.write_text("placeholder")
    output_dir = tmp_path / "nested" / backend / "run"
    records: list[dict[str, object]] = [
        {
            "model": backend,
            "repetition": 0,
            "tokenization_ms": 1.0,
            "model_ms": 2.0,
            "logprob_ms": 0.1,
            "text_to_logprob_ms": 3.1,
            "logprobs": {"yes": -0.1, "no": -2.2},
        }
    ]
    calls: dict[str, object] = {}
    json_calls: list[tuple[Path, object]] = []
    artifact_calls: list[tuple[Path, object]] = []

    monkeypatch.setattr(
        cli_module.BenchmarkConfig,
        "from_toml",
        staticmethod(partial(capture_config, calls, config)),
    )
    monkeypatch.setattr(
        cli_module,
        "_dependency_lock_digest",
        partial(capture_lock_digest, calls),
    )
    monkeypatch.setattr(
        cli_module,
        "_encoder_records" if backend == "encoder" else "_llm_records",
        partial(capture_backend_records, calls, records),
    )
    monkeypatch.setattr(
        cli_module,
        "_llm_records" if backend == "encoder" else "_encoder_records",
        fail_if_backend_called,
    )
    monkeypatch.setattr(cli_module, "_source_commit", lambda: "a" * 40)
    monkeypatch.setattr(cli_module, "_hardware", lambda: {"gpu": "test"})
    monkeypatch.setattr(cli_module, "build_manifest", partial(capture_manifest, calls))
    monkeypatch.setattr(cli_module, "_config_digest", lambda path: "config-sha")
    monkeypatch.setattr(
        cli_module,
        "write_measurement_artifacts",
        partial(capture_measurement_artifacts, artifact_calls),
    )
    monkeypatch.setattr(cli_module, "write_json", partial(capture_json, json_calls))

    run(config_path, output_dir, backend, "run-001")

    assert output_dir.is_dir()
    assert calls["config_path"] == config_path
    assert calls["digest_path"] == config_path
    assert calls["backend_args"] == (config, "lock-sha")
    assert calls["manifest_args"] == (
        config,
        {
            "run_id": "run-001",
            "source_commit": "a" * 40,
            "hardware": {"gpu": "test"},
            "runtime": {"dtype": "test"},
            "dependency_lock_sha256": "lock-sha",
        },
    )
    assert json_calls == [
        (
            output_dir / "manifest.json",
            {
                "manifest": True,
                "config_sha256": "config-sha",
                "backend": backend,
            },
        )
    ]
    assert artifact_calls == [(output_dir, records)]

    run(config_path, output_dir, backend, "run-001")


def test_run_rejects_unsupported_backend_before_entering_a_backend(
    tmp_path, monkeypatch
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        "encoder_revision = 'c5955035435e2bf121cde7f3c8863ef52ff35d82'\n"
        "llm_revision = '8a7ee08e8b9bfb857107ecc25a5599d2f38b76f8'\n"
        "llama_cpp_revision = '6503355df0eb4f65875012523263c302fe0088c1'\n"
    )

    def unexpected_backend(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("backend execution must not start")

    monkeypatch.setattr(cli_module, "_encoder_records", unexpected_backend)
    monkeypatch.setattr(cli_module, "_llm_records", unexpected_backend)

    with pytest.raises(ValueError, match=r"^unsupported backend: invalid$"):
        run(config_path, tmp_path / "output", "invalid", "run-001")


def test_dependency_lock_digest_selects_each_candidate_in_order(
    tmp_path, monkeypatch
) -> None:
    cwd = tmp_path / "cwd"
    config_parent = tmp_path / "config"
    cwd.mkdir()
    config_parent.mkdir()
    monkeypatch.chdir(cwd)
    cwd_lock = cwd / "uv.lock"
    cwd_lock.write_text("cwd-lock")
    config_lock = config_parent / "uv.lock"
    config_lock.write_text("config-lock")
    config_path = config_parent / "benchmark.toml"

    assert (
        _dependency_lock_digest(config_path) == hashlib.sha256(b"cwd-lock").hexdigest()
    )
    cwd_lock.unlink()
    assert (
        _dependency_lock_digest(config_path)
        == hashlib.sha256(b"config-lock").hexdigest()
    )


def test_dependency_lock_digest_uses_repo_fallback_and_can_return_none(
    tmp_path, monkeypatch
) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.chdir(empty)
    config_path = empty / "benchmark.toml"
    repository_lock = (
        cli_module.Path(cli_module.__file__).resolve().parents[3] / "uv.lock"
    )
    expected = hashlib.sha256(repository_lock.read_bytes()).hexdigest()
    assert _dependency_lock_digest(config_path) == expected

    fallback_root = tmp_path / "repository"
    module_path = fallback_root / "src" / "package" / "benchmark" / "cli.py"
    module_path.parent.mkdir(parents=True)
    fallback_lock = fallback_root / "uv.lock"
    fallback_lock.write_text("fallback-lock")
    monkeypatch.setattr(cli_module, "__file__", str(module_path))
    assert (
        _dependency_lock_digest(config_path)
        == hashlib.sha256(b"fallback-lock").hexdigest()
    )

    monkeypatch.setattr(cli_module, "__file__", str(tmp_path / "isolated.py"))
    assert _dependency_lock_digest(config_path) is None


class FakeCuda:
    def __init__(self, *, bf16: bool, capability: tuple[int, int]) -> None:
        self._bf16 = bf16
        self._capability = capability

    def is_available(self) -> bool:
        return True

    def is_bf16_supported(self) -> bool:
        return self._bf16

    def get_device_capability(self, _device: int = 0) -> tuple[int, int]:
        return self._capability


class FakePrecisionTorch:
    bfloat16 = object()
    float16 = object()
    float32 = object()

    def __init__(self, cuda: FakeCuda) -> None:
        self.cuda = cuda


def test_cuda_precision_prefers_bfloat16_only_when_supported() -> None:
    precision = select_encoder_precision(
        FakePrecisionTorch(FakeCuda(bf16=True, capability=(7, 0))), "cuda"
    )

    assert precision.name == "bfloat16"
    assert precision.torch_dtype is FakePrecisionTorch.bfloat16


def test_cuda_boolean_checks_share_safe_probe(monkeypatch) -> None:
    calls: list[str] = []
    cuda = cast(CudaApi, SimpleNamespace())

    def probe(_cuda: CudaApi, attribute: str) -> bool:
        calls.append(attribute)
        return True

    monkeypatch.setattr(runtime_module, "_safe_cuda_bool", probe, raising=False)

    assert runtime_module._cuda_available(cuda) is True
    assert runtime_module._bf16_supported(cuda) is True
    assert calls == ["is_available", "is_bf16_supported"]


def test_cuda_precision_rejects_unavailable_cuda() -> None:
    def fail() -> bool:
        raise RuntimeError("unavailable")

    for cuda in (
        SimpleNamespace(),
        SimpleNamespace(is_available=object()),
        SimpleNamespace(is_available=fail),
        SimpleNamespace(is_available=lambda: False),
    ):
        torch_module = SimpleNamespace(cuda=cuda, float32=object())
        with pytest.raises(
            RuntimeError,
            match=r"^CUDA is required for the Grid5000 encoder benchmark$",
        ):
            select_encoder_precision(torch_module, "cuda")


def test_cpu_precision_uses_fp32() -> None:
    torch_module = SimpleNamespace(float32=object())

    precision = select_encoder_precision(torch_module, "cpu")

    assert precision.name == "float32"
    assert precision.torch_dtype is torch_module.float32
    with pytest.raises(RuntimeError, match="float32"):
        select_encoder_precision(SimpleNamespace(), "cpu")


def test_fp16_support_reuses_normalized_device_capability(monkeypatch) -> None:
    cuda = object()
    calls: list[CudaApi] = []

    def capability(value: CudaApi) -> list[int] | None:
        calls.append(value)
        return [5, 3]

    monkeypatch.setattr(runtime_module, "_device_capability", capability)

    assert runtime_module._fp16_supported(cuda) is True
    assert calls == [cuda]


def test_cuda_precision_falls_back_to_fp16_then_fp32() -> None:
    fp16 = select_encoder_precision(
        FakePrecisionTorch(FakeCuda(bf16=False, capability=(7, 0))), "cuda"
    )
    fp32 = select_encoder_precision(
        FakePrecisionTorch(FakeCuda(bf16=False, capability=(3, 5))), "cuda"
    )

    assert fp16.name == "float16"
    assert fp32.name == "float32"
    assert fp16.torch_dtype is FakePrecisionTorch.float16
    assert fp32.torch_dtype is FakePrecisionTorch.float32

    boundary = select_encoder_precision(
        FakePrecisionTorch(FakeCuda(bf16=False, capability=(5, 3))), "cuda"
    )
    assert boundary.name == "float16"

    def fail() -> bool:
        raise RuntimeError("unavailable")

    failed_bf16 = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=fail,
            get_device_capability=lambda: (3, 5),
        ),
        bfloat16=object(),
        float16=object(),
        float32=object(),
    )
    assert select_encoder_precision(failed_bf16, "cuda").name == "float32"

    missing_capability = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: False,
        ),
        bfloat16=object(),
        float16=object(),
        float32=object(),
    )
    assert select_encoder_precision(missing_capability, "cuda").name == "float32"

    def fail_capability() -> tuple[int, int]:
        raise RuntimeError("unavailable")

    failed_capability = SimpleNamespace(
        cuda=SimpleNamespace(
            is_available=lambda: True,
            is_bf16_supported=lambda: False,
            get_device_capability=fail_capability,
        ),
        bfloat16=object(),
        float16=object(),
        float32=object(),
    )
    assert select_encoder_precision(failed_capability, "cuda").name == "float32"


def test_quantization_from_filename_extracts_gguf_label() -> None:
    assert quantization_from_filename("Qwen3.6-27B-Q4_K_M.gguf") == "Q4_K_M"
    assert quantization_from_filename("model-f16.gguf") is None


def test_runtime_metadata_records_available_gpu_and_driver(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(
        runtime_module,
        "_package_version",
        lambda package: {"torch": "2.6.0", "transformers": "4.55.0"}.get(package),
    )
    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=lambda: "Test GPU",
        get_device_capability=lambda: (8, 0),
        driver_version=55000,
    )
    torch_module = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda="12.4"),
    )

    metadata = build_runtime_metadata(
        torch_module=torch_module,
        dtype="bfloat16",
        llama_cpp_revision="llama-revision",
        llm_filename="model.Q4_K_M.gguf",
        dependency_lock_sha256="lock-sha",
    )

    hardware = cast(dict[str, object], metadata["cuda"])
    assert set(hardware) == {
        "available",
        "gpu",
        "capability",
        "runtime",
        "driver",
        "visible_devices",
    }
    assert metadata["python"]
    assert metadata["platform"]
    assert set(metadata) == {
        "python",
        "platform",
        "torch",
        "transformers",
        "llama_cpp_revision",
        "llm_filename",
        "dtype",
        "dependency_lock_sha256",
        "cuda",
    }
    assert metadata["torch"] == "2.6.0"
    assert metadata["transformers"] == "4.55.0"
    assert metadata["llama_cpp_revision"] == "llama-revision"
    assert metadata["llm_filename"] == "model.Q4_K_M.gguf"
    assert metadata["dtype"] == "bfloat16"
    assert metadata["dependency_lock_sha256"] == "lock-sha"
    assert hardware["available"] is True
    assert hardware["gpu"] == "Test GPU"
    assert hardware["capability"] == [8, 0]
    assert hardware["runtime"] == "12.4"
    assert hardware["driver"] == 55000
    assert hardware["visible_devices"] == "0"


def test_runtime_metadata_uses_driver_fallback_and_nulls_failed_observations() -> None:
    def fail() -> None:
        raise RuntimeError("unavailable")

    cuda = SimpleNamespace(
        is_available=lambda: True,
        get_device_name=fail,
        get_device_capability=fail,
    )
    torch_module = SimpleNamespace(
        cuda=cuda,
        version=SimpleNamespace(cuda=None),
        _C=SimpleNamespace(_cuda_getDriverVersion=lambda: 55001),
    )

    metadata = build_runtime_metadata(
        torch_module=torch_module,
        dtype=None,
        llama_cpp_revision="llama-revision",
        llm_filename="model.gguf",
        dependency_lock_sha256=None,
    )

    hardware = cast(dict[str, object], metadata["cuda"])
    assert set(hardware) == {
        "available",
        "gpu",
        "capability",
        "runtime",
        "driver",
        "visible_devices",
    }
    assert hardware["gpu"] is None
    assert hardware["capability"] is None
    assert hardware["runtime"] is None
    assert hardware["driver"] == 55001


def test_runtime_metadata_driver_and_device_helpers_fail_closed() -> None:
    cuda = SimpleNamespace()
    assert _cuda(SimpleNamespace(cuda=cuda)) is cuda
    with pytest.raises(RuntimeError, match=r"^torch does not expose CUDA$"):
        _cuda(SimpleNamespace())
    assert _driver_version(SimpleNamespace(), cast(CudaApi, SimpleNamespace())) is None
    assert (
        _driver_version(
            SimpleNamespace(
                _C=SimpleNamespace(_cuda_getDriverVersion=lambda: "invalid")
            ),
            cast(CudaApi, SimpleNamespace()),
        )
        is None
    )
    assert _device_name(cast(CudaApi, SimpleNamespace())) is None
    assert (
        _device_name(cast(CudaApi, SimpleNamespace(get_device_name=lambda: 7))) is None
    )


def test_device_capability_returns_none_when_attribute_is_missing() -> None:
    assert runtime_module._device_capability(object()) is None


def test_cuda_boundary_declares_dynamic_object_contract() -> None:
    assert getattr(runtime_module.CudaApi, "__value__", None) is object


def test_cuda_adapter_preserves_identity_and_error_message() -> None:
    cuda = SimpleNamespace()
    assert _cuda(SimpleNamespace(cuda=cuda)) is cuda
    with pytest.raises(RuntimeError, match=r"^torch does not expose CUDA$"):
        _cuda(SimpleNamespace())


def test_unavailable_cuda_metadata_has_stable_schema(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    helper = getattr(runtime_module, "_unavailable_cuda_metadata", None)

    assert callable(helper)
    assert helper() == {
        "available": None,
        "gpu": None,
        "capability": None,
        "runtime": None,
        "driver": None,
        "visible_devices": "0",
    }


@pytest.mark.parametrize("torch_module", [None, SimpleNamespace()])
def test_runtime_metadata_uses_nulls_when_torch_is_unavailable(
    torch_module: object | None,
    monkeypatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    metadata = build_runtime_metadata(
        torch_module=torch_module,
        dtype=None,
        llama_cpp_revision="llama-revision",
        llm_filename="model.gguf",
        dependency_lock_sha256=None,
    )

    hardware = cast(dict[str, object], metadata["cuda"])
    assert set(hardware) == {
        "available",
        "gpu",
        "capability",
        "runtime",
        "driver",
        "visible_devices",
    }
    assert hardware["available"] is None
    assert hardware["gpu"] is None
    assert hardware["capability"] is None
    assert hardware["runtime"] is None
    assert hardware["driver"] is None
    assert hardware["visible_devices"] == "0"

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    default_metadata = build_runtime_metadata(
        torch_module=torch_module,
        dtype=None,
        llama_cpp_revision="llama-revision",
        llm_filename="model.gguf",
        dependency_lock_sha256=None,
    )
    default_hardware = cast(dict[str, object], default_metadata["cuda"])
    assert default_hardware["visible_devices"] == ""

    unavailable_cuda = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    unavailable_metadata = build_runtime_metadata(
        torch_module=unavailable_cuda,
        dtype=None,
        llama_cpp_revision="llama-revision",
        llm_filename="model.gguf",
        dependency_lock_sha256=None,
    )
    unavailable_hardware = cast(dict[str, object], unavailable_metadata["cuda"])
    assert set(unavailable_hardware) == set(hardware)
    assert unavailable_hardware["available"] is False
    assert unavailable_hardware["visible_devices"] == ""


@pytest.mark.parametrize("torch_module", [None, SimpleNamespace()])
def test_runtime_metadata_defaults_visible_devices_to_empty(
    torch_module: object | None,
    monkeypatch,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    metadata = build_runtime_metadata(
        torch_module=torch_module,
        dtype=None,
        llama_cpp_revision="llama-revision",
        llm_filename="model.gguf",
        dependency_lock_sha256=None,
    )

    hardware = cast(dict[str, object], metadata["cuda"])
    assert hardware["visible_devices"] == ""
