from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from .config import BenchmarkConfig
from .encoder import (
    EncoderScore,
    ModelLike,
    TokenizerLike,
    TorchLike,
    score_transformers_once,
)
from .example import LANDUSE_SENTENCE, candidate_labels
from .llm import LlamaClient, LlamaScore
from .measurement import MeasurementRecord, choose_decision, measure_repetitions
from .reporting import build_summary, write_json, write_jsonl
from .runtime import (
    build_runtime_metadata,
    quantization_from_filename,
    select_encoder_precision,
)


@dataclass(frozen=True, slots=True)
class LoadedEncoder:
    """Lazy-loaded encoder resources and the facts observed while loading."""

    tokenizer: object
    model: object
    torch_module: object
    runtime: Mapping[str, object]


def score_record(
    model: str, repetition: int, score: EncoderScore | LlamaScore
) -> MeasurementRecord:
    """Normalize either backend result into the common measurement schema."""

    return {
        "model": model,
        "repetition": repetition,
        "tokenization_ms": score.tokenization_ms,
        "model_ms": score.model_ms,
        "logprob_ms": score.logprob_ms,
        "text_to_logprob_ms": score.text_to_logprob_ms,
        "input_tokens": score.input_tokens,
        "logprobs": score.logprobs,
        "decision": choose_decision(score.logprobs),
    }


def build_manifest(
    config: BenchmarkConfig,
    *,
    run_id: str,
    source_commit: str,
    hardware: Mapping[str, object],
    runtime: Mapping[str, object] | None = None,
    dependency_lock_sha256: str | None = None,
) -> dict[str, object]:
    """Build the reproducibility manifest for a run."""

    return {
        "schema_version": 1,
        "run_id": run_id,
        "source_commit": source_commit,
        "seed": config.seed,
        "example": {"sentence": LANDUSE_SENTENCE, "labels": candidate_labels()},
        "models": {
            "encoder": {
                "id": config.encoder_model,
                "revision": config.encoder_revision,
            },
            "llm": {
                "id": config.llm_model,
                "revision": config.llm_revision,
                "filename": config.llm_filename,
            },
        },
        "llama_cpp_revision": config.llama_cpp_revision,
        "runtime": dict(runtime or {}),
        "dependency_lock_sha256": dependency_lock_sha256,
        "protocol": {
            "warmups": config.warmups,
            "repetitions": config.repetitions,
            "batch_size": 1,
            "generated_tokens": 1,
            "prompt_cache": False,
        },
        "hardware": dict(hardware),
    }


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


def _load_encoder(config: BenchmarkConfig) -> LoadedEncoder:
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
        dependency_lock_sha256=None,
    )
    return LoadedEncoder(tokenizer, model, torch_module, runtime)


def _encoder_records(
    config: BenchmarkConfig,
    dependency_lock_sha256: str | None,
) -> tuple[list[MeasurementRecord], Mapping[str, object]]:
    loaded = _load_encoder(config)
    timed = measure_repetitions(
        lambda: score_transformers_once(
            cast(TokenizerLike, loaded.tokenizer),
            cast(ModelLike, loaded.model),
            cast(TorchLike, loaded.torch_module),
            config.device,
        ),
        warmups=config.warmups,
        repetitions=config.repetitions,
    )
    runtime = dict(loaded.runtime)
    runtime["dependency_lock_sha256"] = dependency_lock_sha256
    return (
        [
            score_record("encoder", index, item.value)
            for index, item in enumerate(timed)
        ],
        runtime,
    )


def _llm_records(
    config: BenchmarkConfig,
    dependency_lock_sha256: str | None,
) -> tuple[list[MeasurementRecord], Mapping[str, object]]:
    client = LlamaClient(config.llama_base_url)
    timed = measure_repetitions(
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
    return (
        [score_record("llm", index, item.value) for index, item in enumerate(timed)],
        runtime,
    )


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
    write_jsonl(output_dir / "measurements.jsonl", records)
    write_json(output_dir / "summary.json", build_summary(records))


def _dependency_lock_digest(config_path: Path) -> str | None:
    candidates = (
        Path.cwd() / "uv.lock",
        config_path.parent / "uv.lock",
        Path(__file__).resolve().parents[3] / "uv.lock",
    )
    for candidate in candidates:
        if candidate.is_file():
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
