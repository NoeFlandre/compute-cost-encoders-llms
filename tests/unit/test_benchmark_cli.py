from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

import compute_cost_encoders_llms.benchmark.cli as cli_module
import compute_cost_encoders_llms.benchmark.runtime as runtime_module
from compute_cost_encoders_llms.benchmark.cli import (
    _hardware,
    _load_encoder,
    _source_commit,
    build_manifest,
    score_record,
)
from compute_cost_encoders_llms.benchmark.config import BenchmarkConfig
from compute_cost_encoders_llms.benchmark.encoder import EncoderScore
from compute_cost_encoders_llms.benchmark.runtime import (
    CudaApi,
    _device_name,
    _driver_version,
    build_runtime_metadata,
    quantization_from_filename,
    select_encoder_precision,
)


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
    assert manifest["schema_version"] == 1
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
        "labels": ("yes", "no"),
    }
    models = cast(dict[str, dict[str, str]], manifest["models"])
    assert models["llm"]["revision"] == config.llm_revision
    assert manifest["hardware"] == {"gpu": "test"}
    hardware_snapshot = _hardware()
    assert set(hardware_snapshot) == {"platform", "python", "cuda_visible_devices"}
    assert hardware_snapshot["cuda_visible_devices"] == "0"


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

    loaded = _load_encoder(config)

    assert loaded.tokenizer is tokenizer
    assert loaded.model is model
    assert loaded.torch_module is torch_module
    assert loaded.runtime["dtype"] == "float32"
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
        with pytest.raises(RuntimeError, match="CUDA"):
            select_encoder_precision(torch_module, "cuda")


def test_cpu_precision_uses_fp32() -> None:
    torch_module = SimpleNamespace(float32=object())

    precision = select_encoder_precision(torch_module, "cpu")

    assert precision.name == "float32"
    assert precision.torch_dtype is torch_module.float32
    with pytest.raises(RuntimeError, match="float32"):
        select_encoder_precision(SimpleNamespace(), "cpu")


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
