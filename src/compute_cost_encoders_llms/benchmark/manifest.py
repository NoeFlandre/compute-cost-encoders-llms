from __future__ import annotations

from collections.abc import Mapping

from .config import BenchmarkConfig
from .example import (
    LANDUSE_QUESTION,
    LANDUSE_SENTENCE,
    candidate_label_forms,
    candidate_labels,
)


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
        "schema_version": 2,
        "run_id": run_id,
        "source_commit": source_commit,
        "seed": config.seed,
        "example": {
            "sentence": LANDUSE_SENTENCE,
            "question": LANDUSE_QUESTION,
            "labels": candidate_labels(),
            "label_forms": {
                label: candidate_label_forms(label) for label in candidate_labels()
            },
        },
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
            "encoder_answer_marker": "Answer: <mask>",
            "llm_template_endpoint": "/apply-template",
            "llm_reasoning": False,
        },
        "hardware": dict(hardware),
    }
