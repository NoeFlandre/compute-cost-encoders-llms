from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]


def _run_entrypoint(arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    pythonpath = [str(PROJECT_ROOT / "src"), str(PROJECT_ROOT)]
    if environment.get("PYTHONPATH"):
        pythonpath.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return subprocess.run(
        [sys.executable, *arguments],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )


def test_benchmark_cli_module_executes_help_entrypoint() -> None:
    result = _run_entrypoint(
        ["-m", "compute_cost_encoders_llms.benchmark.cli", "--help"]
    )

    assert result.returncode == 0, result.stderr
    assert all(
        option in result.stdout
        for option in ("--config", "--output-dir", "--backend", "--run-id")
    )


def test_checkpoint_metadata_script_executes_valid_file(tmp_path) -> None:
    metadata_path = tmp_path / "checkpoint.json"
    metadata_path.write_text(
        json.dumps(
            {
                "source_commit": "commit-001",
                "config_revision": "sha256:config-001",
                "dataset_revision": "dataset@revision",
                "model_revision": "model@revision",
                "seed": 0,
                "step": 0,
                "metrics": {"encoder": {"median_text_to_logprob_ms": 1.0}},
                "complete": True,
                "artifact_uri": (
                    "hf://buckets/NoeFlandre/compute-cost-encoders-llms/runs/run-001"
                ),
            }
        ),
        encoding="utf-8",
    )

    result = _run_entrypoint(
        ["scripts/grid5000/checkpoint_metadata.py", str(metadata_path)]
    )

    assert result.returncode == 0, result.stderr
    assert "Valid checkpoint metadata" in result.stdout


def test_report_script_executes_help_entrypoint() -> None:
    result = _run_entrypoint(["scripts/render_report.py", "--help"])

    assert result.returncode == 0, result.stderr
    assert all(
        option in result.stdout
        for option in ("--encoder-dir", "--llm-dir", "--output", "--checkpoint")
    )
