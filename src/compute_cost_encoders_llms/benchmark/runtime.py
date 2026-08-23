from __future__ import annotations

import importlib.metadata
import os
import platform
import re
from dataclasses import dataclass

type CudaApi = object


@dataclass(frozen=True, slots=True)
class EncoderPrecision:
    """Torch dtype selected from the requested device capabilities."""

    torch_dtype: object
    name: str


def select_encoder_precision(torch_module: object, device: str) -> EncoderPrecision:
    """Select a safe encoder dtype without assuming BF16 support."""

    if not device.startswith("cuda"):
        return EncoderPrecision(_dtype(torch_module, "float32"), "float32")

    cuda = _cuda(torch_module)
    if not _cuda_available(cuda):
        raise RuntimeError("CUDA is required for the Grid5000 encoder benchmark")
    if _bf16_supported(cuda):
        return EncoderPrecision(_dtype(torch_module, "bfloat16"), "bfloat16")
    if _fp16_supported(cuda):
        return EncoderPrecision(_dtype(torch_module, "float16"), "float16")
    return EncoderPrecision(_dtype(torch_module, "float32"), "float32")


def build_runtime_metadata(
    *,
    torch_module: object | None,
    dtype: str | None,
    llama_cpp_revision: str,
    llm_filename: str,
    dependency_lock_sha256: str | None,
) -> dict[str, object]:
    """Capture stable runtime facts, using null for unavailable observations."""

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": _package_version("torch"),
        "transformers": _package_version("transformers"),
        "llama_cpp_revision": llama_cpp_revision,
        "llm_filename": llm_filename,
        "dtype": dtype,
        "dependency_lock_sha256": dependency_lock_sha256,
        "cuda": _cuda_metadata(torch_module),
    }


def quantization_from_filename(filename: str) -> str | None:
    """Return a GGUF quantization label when it is encoded in the filename."""

    match = re.search(r"Q\d(?:_[A-Z0-9]+)+", filename)
    return match.group(0) if match else None


def _dtype(torch_module: object, name: str) -> object:
    dtype = getattr(torch_module, name, None)
    if dtype is None:
        raise RuntimeError(f"torch does not expose {name}")
    return dtype


def _cuda(torch_module: object) -> CudaApi:
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None:
        raise RuntimeError("torch does not expose CUDA")
    return cuda


def _cuda_available(cuda: CudaApi) -> bool:
    checker = getattr(cuda, "is_available", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except (AttributeError, RuntimeError, TypeError):
        return False


def _bf16_supported(cuda: CudaApi) -> bool:
    checker = getattr(cuda, "is_bf16_supported", None)
    if not callable(checker):
        return False
    try:
        return bool(checker())
    except (AttributeError, RuntimeError, TypeError):
        return False


def _fp16_supported(cuda: CudaApi) -> bool:
    getter = getattr(cuda, "get_device_capability", None)
    if not callable(getter):
        return False
    try:
        major, minor = getter()
        capability = (int(major), int(minor))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return False
    return capability >= (5, 3)


def _unavailable_cuda_metadata() -> dict[str, object]:
    return {
        "available": None,
        "gpu": None,
        "capability": None,
        "runtime": None,
        "driver": None,
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _cuda_metadata(torch_module: object | None) -> dict[str, object]:
    if torch_module is None:
        return _unavailable_cuda_metadata()

    try:
        cuda = _cuda(torch_module)
    except RuntimeError:
        return _unavailable_cuda_metadata()
    available = _cuda_available(cuda)
    capability = _device_capability(cuda) if available else None
    return {
        "available": available,
        "gpu": _device_name(cuda) if available else None,
        "capability": capability,
        "runtime": _cuda_runtime(torch_module),
        "driver": _driver_version(torch_module, cuda),
        "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _device_capability(cuda: CudaApi) -> list[int] | None:
    getter = getattr(cuda, "get_device_capability", None)
    if not callable(getter):
        return None
    try:
        major, minor = getter()
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None
    return [int(major), int(minor)]


def _device_name(cuda: CudaApi) -> str | None:
    getter = getattr(cuda, "get_device_name", None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except (AttributeError, RuntimeError, TypeError):
        return None
    return value if isinstance(value, str) else None


def _cuda_runtime(torch_module: object) -> str | None:
    version = getattr(torch_module, "version", None)
    value = getattr(version, "cuda", None)
    return value if isinstance(value, str) else None


def _driver_version(torch_module: object, cuda: CudaApi) -> int | None:
    value = getattr(cuda, "driver_version", None)
    if isinstance(value, int):
        return value
    torch_c = getattr(torch_module, "_C", None)
    getter = getattr(torch_c, "_cuda_getDriverVersion", None)
    if not callable(getter):
        return None
    try:
        value = getter()
    except (AttributeError, RuntimeError, TypeError):
        return None
    return value if isinstance(value, int) else None


def _package_version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None
