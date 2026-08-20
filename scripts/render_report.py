#!/usr/bin/env python3
"""Merge verified backend artifacts into a standalone LaTeX report."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from compute_cost_encoders_llms.benchmark.reporting import render_latex_document

PROJECT_BUCKET_URI = "hf://buckets/NoeFlandre/compute-cost-encoders-llms"


def _read_json(path: Path) -> Mapping[str, object]:
    document = json.loads(path.read_text())
    if not isinstance(document, Mapping):
        raise ValueError(f"JSON document is not an object: {path}")
    return document


def merge_artifacts(
    encoder_manifest: Mapping[str, object],
    llm_manifest: Mapping[str, object],
    encoder_summary: Mapping[str, object],
    llm_summary: Mapping[str, object],
) -> dict[str, object]:
    """Combine two backend runs only when their protocol identities match."""

    if encoder_manifest.get("source_commit") != llm_manifest.get("source_commit"):
        raise ValueError("source commit differs between backend runs")
    if encoder_manifest.get("example") != llm_manifest.get("example"):
        raise ValueError("example differs between backend runs")
    manifest = dict(encoder_manifest)
    manifest["backend"] = "both"
    manifest["run_ids"] = [encoder_manifest.get("run_id"), llm_manifest.get("run_id")]
    encoder_models = encoder_summary.get("models", [])
    llm_models = llm_summary.get("models", [])
    if not isinstance(encoder_models, list) or not isinstance(llm_models, list):
        raise ValueError("backend summaries must contain model lists")
    return {"manifest": manifest, "summary": {"models": encoder_models + llm_models}}


def build_checkpoint_metadata(
    merged: Mapping[str, object],
    *,
    config_revision: str,
    dataset_revision: str,
    model_revision: str,
    artifact_prefix: str,
) -> dict[str, object]:
    """Build the complete metadata required by the Grid5000 publisher."""

    prefix = artifact_prefix.strip("/")
    if not prefix or ".." in prefix:
        raise ValueError("artifact prefix must be non-empty and traversal-free")
    manifest = _mapping_value(merged, "manifest")
    summary = _mapping_value(merged, "summary")
    protocol = _mapping_value(manifest, "protocol")
    source_commit = _text_value(manifest, "source_commit")
    seed = _integer_value(manifest, "seed")
    step = _integer_value(protocol, "repetitions")
    return {
        "source_commit": source_commit,
        "config_revision": config_revision,
        "dataset_revision": dataset_revision,
        "model_revision": model_revision,
        "seed": seed,
        "step": step,
        "metrics": _checkpoint_metrics(summary),
        "complete": True,
        "artifact_uri": f"{PROJECT_BUCKET_URI}/{prefix}",
    }


def _mapping_value(document: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = document.get(field)
    if not isinstance(value, Mapping):
        raise ValueError(f"merged artifact field is not an object: {field}")
    return value


def _text_value(document: Mapping[str, object], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"merged artifact field is not text: {field}")
    return value


def _integer_value(document: Mapping[str, object], field: str) -> int:
    value = document.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"merged artifact field is not a non-negative integer: {field}"
        )
    return value


def _checkpoint_metrics(summary: Mapping[str, object]) -> dict[str, object]:
    models = summary.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("merged summary must contain model results")
    metrics: dict[str, object] = {}
    for model in models:
        model_mapping = _as_mapping(model, "model result")
        model_name = _text_value(model_mapping, "model")
        latency = _mapping_value(model_mapping, "latency")
        decisions = _mapping_value(model_mapping, "decision_counts")
        metrics[model_name] = {
            "median_text_to_logprob_ms": latency.get("median"),
            "decision_counts": dict(decisions),
        }
    return metrics


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"merged artifact field is not an object: {field}")
    return value


def render_report(
    encoder_dir: Path,
    llm_dir: Path,
    output: Path,
    checkpoint: Path | None = None,
) -> None:
    encoder_manifest = _read_json(encoder_dir / "manifest.json")
    llm_manifest = _read_json(llm_dir / "manifest.json")
    encoder_summary = _read_json(encoder_dir / "summary.json")
    llm_summary = _read_json(llm_dir / "summary.json")
    merged = merge_artifacts(
        encoder_manifest,
        llm_manifest,
        encoder_summary,
        llm_summary,
    )
    manifest = _mapping_value(merged, "manifest")
    summary = _mapping_value(merged, "summary")
    output.write_text(render_latex_document(manifest, summary))
    if checkpoint is not None:
        metadata = build_checkpoint_metadata(
            merged,
            config_revision=os.environ["GRID5000_CONFIG_REVISION"],
            dataset_revision=os.environ["GRID5000_DATASET_REVISION"],
            model_revision=os.environ["GRID5000_MODEL_REVISION"],
            artifact_prefix=os.environ["GRID5000_ARTIFACT_PREFIX"],
        )
        checkpoint.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--encoder-dir", type=Path, required=True)
    parser.add_argument("--llm-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()
    render_report(args.encoder_dir, args.llm_dir, args.output, args.checkpoint)


if __name__ == "__main__":
    main()
