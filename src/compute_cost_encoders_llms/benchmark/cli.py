from __future__ import annotations

import argparse
import hashlib
import importlib
import os
import platform
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .config import BenchmarkConfig
from .encoder import EncoderScore, score_transformers_once
from .example import LANDUSE_SENTENCE, candidate_labels
from .llm import LlamaClient, LlamaScore
from .measurement import MeasurementRecord, choose_decision, measure_repetitions
from .reporting import build_summary, write_json, write_jsonl


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


def _load_encoder(config: BenchmarkConfig) -> tuple[Any, Any, Any]:
    try:
        torch: Any = importlib.import_module("torch")
        transformers: Any = importlib.import_module("transformers")
    except ImportError as error:
        raise RuntimeError("encoder dependencies are not installed") from error
    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the Grid5000 encoder benchmark")
    dtype = torch.bfloat16 if config.device.startswith("cuda") else torch.float32
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.encoder_model,
        revision=config.encoder_revision,
        trust_remote_code=False,
    )
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        config.encoder_model,
        revision=config.encoder_revision,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    model.to(config.device)
    model.eval()
    return tokenizer, model, torch


def _encoder_records(config: BenchmarkConfig) -> list[MeasurementRecord]:
    tokenizer, model, torch_module = _load_encoder(config)
    timed = measure_repetitions(
        lambda: score_transformers_once(tokenizer, model, torch_module, config.device),
        warmups=config.warmups,
        repetitions=config.repetitions,
    )
    return [
        score_record("encoder", index, item.value) for index, item in enumerate(timed)
    ]


def _llm_records(config: BenchmarkConfig) -> list[MeasurementRecord]:
    client = LlamaClient(config.llama_base_url)
    timed = measure_repetitions(
        lambda: client.score(config.seed),
        warmups=config.warmups,
        repetitions=config.repetitions,
    )
    return [score_record("llm", index, item.value) for index, item in enumerate(timed)]


def run(config_path: Path, output_dir: Path, backend: str, run_id: str) -> None:
    """Run one backend and write its manifest, measurements, and summary."""

    config = BenchmarkConfig.from_toml(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        config,
        run_id=run_id,
        source_commit=_source_commit(),
        hardware=_hardware(),
    )
    manifest["config_sha256"] = _config_digest(config_path)
    records = _encoder_records(config) if backend == "encoder" else _llm_records(config)
    manifest["backend"] = backend
    write_json(output_dir / "manifest.json", manifest)
    write_jsonl(output_dir / "measurements.jsonl", records)
    write_json(output_dir / "summary.json", build_summary(records))


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
