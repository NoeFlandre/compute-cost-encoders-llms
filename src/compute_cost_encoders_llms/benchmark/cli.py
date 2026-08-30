from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from .config import BenchmarkConfig
from .encoder import (
    ModelLike,
    TokenizerLike,
    TorchLike,
    build_candidate_token_ids,
    score_transformers_once,
)
from .llm import LlamaClient
from .manifest import build_manifest
from .measurement import (
    MeasurementRecord,
    ScoreLike,
    measure_repetitions,
    score_record,
)
from .reporting import (
    write_json,
    write_measurement_artifacts,
)
from .runtime import (
    build_runtime_metadata,
    quantization_from_filename,
    select_encoder_precision,
)


@dataclass(frozen=True, slots=True)
class LoadedEncoder:
    """Lazy-loaded encoder resources and the facts observed while loading."""

    tokenizer: TokenizerLike
    model: ModelLike
    torch_module: TorchLike
    runtime: Mapping[str, object]


def _as_tokenizer(value: object) -> TokenizerLike:
    if not isinstance(value, TokenizerLike):
        raise RuntimeError("loaded tokenizer does not implement the encoder interface")
    return value


def _as_model(value: object) -> ModelLike:
    if not isinstance(value, ModelLike):
        raise RuntimeError("loaded model does not implement the encoder interface")
    return value


def _as_torch(value: object) -> TorchLike:
    if not isinstance(value, TorchLike):
        raise RuntimeError(
            "loaded torch module does not implement the encoder interface"
        )
    return value


def _source_commit() -> str:
    configured = os.environ.get("GRID5000_SOURCE_COMMIT")
    if configured:
        return configured
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True, encoding="utf-8"
    ).strip()


def _hardware() -> dict[str, object]:
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def _config_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_encoder(
    config: BenchmarkConfig,
    dependency_lock_sha256: str | None = None,
) -> LoadedEncoder:
    try:
        torch_module = importlib.import_module("torch")
        transformers_module = importlib.import_module("transformers")
    except ImportError as error:
        raise RuntimeError("encoder dependencies are not installed") from error
    precision = select_encoder_precision(torch_module, config.device)
    tokenizer = transformers_module.AutoTokenizer.from_pretrained(
        config.encoder_model,
        revision=config.encoder_revision,
        trust_remote_code=False,
    )
    model = transformers_module.AutoModelForMaskedLM.from_pretrained(
        config.encoder_model,
        revision=config.encoder_revision,
        trust_remote_code=False,
        torch_dtype=precision.torch_dtype,
    )
    model.to(config.device)
    model.eval()
    runtime = build_runtime_metadata(
        torch_module=torch_module,
        dtype=precision.name,
        llama_cpp_revision=config.llama_cpp_revision,
        llm_filename=config.llm_filename,
        dependency_lock_sha256=dependency_lock_sha256,
    )
    return LoadedEncoder(
        _as_tokenizer(tokenizer),
        _as_model(model),
        _as_torch(torch_module),
        runtime,
    )


def _measure_records(
    model: str,
    operation: Callable[[], ScoreLike],
    *,
    warmups: int,
    repetitions: int,
) -> list[MeasurementRecord]:
    timed = measure_repetitions(
        operation,
        warmups=warmups,
        repetitions=repetitions,
    )
    return [score_record(model, index, item.value) for index, item in enumerate(timed)]


def _encoder_records(
    config: BenchmarkConfig,
    dependency_lock_sha256: str | None,
) -> tuple[list[MeasurementRecord], Mapping[str, object]]:
    loaded = _load_encoder(config, dependency_lock_sha256)
    candidate_token_ids = build_candidate_token_ids(loaded.tokenizer)
    records = _measure_records(
        "encoder",
        lambda: score_transformers_once(
            loaded.tokenizer,
            loaded.model,
            loaded.torch_module,
            config.device,
            candidate_token_ids=candidate_token_ids,
        ),
        warmups=config.warmups,
        repetitions=config.repetitions,
    )
    return records, loaded.runtime


def _llm_records(
    config: BenchmarkConfig,
    dependency_lock_sha256: str | None,
) -> tuple[list[MeasurementRecord], Mapping[str, object]]:
    client = LlamaClient(config.llama_base_url)
    records = _measure_records(
        "llm",
        lambda: client.score(config.seed),
        warmups=config.warmups,
        repetitions=config.repetitions,
    )
    runtime = build_runtime_metadata(
        torch_module=_optional_torch_module(),
        dtype=quantization_from_filename(config.llm_filename),
        llama_cpp_revision=config.llama_cpp_revision,
        llm_filename=config.llm_filename,
        dependency_lock_sha256=dependency_lock_sha256,
    )
    return records, runtime


def run(config_path: Path, output_dir: Path, backend: str, run_id: str) -> None:
    """Run one backend and write its manifest, measurements, and summary."""

    config = BenchmarkConfig.from_toml(config_path)
    if backend not in {"encoder", "llm"}:
        raise ValueError(f"unsupported backend: {backend}")
    output_dir.mkdir(parents=True, exist_ok=True)
    dependency_lock_sha256 = _dependency_lock_digest(config_path)
    records, runtime = (
        _encoder_records(config, dependency_lock_sha256)
        if backend == "encoder"
        else _llm_records(config, dependency_lock_sha256)
    )
    manifest = build_manifest(
        config,
        run_id=run_id,
        source_commit=_source_commit(),
        hardware=_hardware(),
        runtime=runtime,
        dependency_lock_sha256=dependency_lock_sha256,
    )
    manifest["config_sha256"] = _config_digest(config_path)
    manifest["backend"] = backend
    write_json(output_dir / "manifest.json", manifest)
    write_measurement_artifacts(output_dir, records)


def _dependency_lock_digest(config_path: Path) -> str | None:
    candidates = (
        Path.cwd() / "uv.lock",
        config_path.parent / "uv.lock",
        Path(__file__).resolve().parents[3] / "uv.lock",
    )
    for candidate in candidates:
        if candidate.name == "uv.lock" and candidate.is_file():
            return _config_digest(candidate)
    return None


def _optional_torch_module() -> object | None:
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=("encoder", "llm"), required=True)
    parser.add_argument("--run-id", required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    run(args.config, args.output_dir, args.backend, args.run_id)


if __name__ == "__main__":
    main()
