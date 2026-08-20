from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar


class ConfigError(ValueError):
    """Raised when benchmark configuration is unsafe or incomplete."""


_IMMUTABLE_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    """Immutable settings required to reproduce one benchmark run."""

    encoder_revision: str
    llm_revision: str
    llama_cpp_revision: str
    encoder_model: str = "jhu-clsp/mmBERT-base"
    llm_model: str = "ggml-org/Qwen3.6-27B-GGUF"
    llm_filename: str = "Qwen3.6-27B-Q4_K_M.gguf"
    llama_base_url: str = "http://127.0.0.1:8080"
    device: str = "cuda"
    repetitions: int = 128
    warmups: int = 8
    seed: int = 7

    _TOML_FIELDS: ClassVar[tuple[str, ...]] = (
        "encoder_revision",
        "llm_revision",
        "llama_cpp_revision",
        "encoder_model",
        "llm_model",
        "llm_filename",
        "llama_base_url",
        "device",
        "repetitions",
        "warmups",
        "seed",
    )

    @classmethod
    def from_toml(cls, path: Path) -> BenchmarkConfig:
        with path.open("rb") as stream:
            document: dict[str, Any] = tomllib.load(stream)
        values = document.get("benchmark", document)
        if not isinstance(values, dict):
            raise ConfigError("benchmark TOML section must be an object")
        selected = {
            field: values[field] for field in cls._TOML_FIELDS if field in values
        }
        return cls(**selected)

    def __post_init__(self) -> None:
        _validate_revision("encoder_revision", self.encoder_revision)
        _validate_revision("llm_revision", self.llm_revision)
        if not self.llama_cpp_revision or self.llama_cpp_revision == "main":
            raise ConfigError("llama_cpp_revision must be pinned")
        _validate_run_settings(self.repetitions, self.warmups)


def _validate_revision(field_name: str, revision: str) -> None:
    if not _IMMUTABLE_REVISION.fullmatch(revision):
        raise ConfigError(f"{field_name} must be a 40-character revision")


def _validate_run_settings(repetitions: int, warmups: int) -> None:
    if repetitions < 1:
        raise ConfigError("repetitions must be positive")
    if warmups < 0:
        raise ConfigError("warmups must be non-negative")
