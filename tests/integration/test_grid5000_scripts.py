from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
SCRIPT_DIR = PROJECT_ROOT / "scripts" / "grid5000"


def _install_fake_command(bin_dir: Path, name: str, body: str = "") -> None:
    variable = name.upper().replace("-", "_")
    script = f"""#!/usr/bin/env bash
set -euo pipefail
count_file="$FAKE_LOG_DIR/{name}.count"
count=0
if [[ -f "$count_file" ]]; then count="$(<"$count_file")"; fi
count=$((count + 1))
printf '%s' "$count" > "$count_file"
printf '%s\\0' "$@" > "$FAKE_LOG_DIR/{name}.$count"
"""
    if body:
        script += body.rstrip() + "\n"
    script += f"""
if [[ -n "${{FAKE_{variable}_STDOUT:-}}" ]]; then
    printf '%s' "${{FAKE_{variable}_STDOUT}}"
fi
exit "${{FAKE_{variable}_EXIT:-0}}"
"""
    path = bin_dir / name
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _read_invocations(log_dir: Path, name: str) -> list[list[str]]:
    paths = sorted(
        (
            path
            for path in log_dir.glob(f"{name}.*")
            if path.name.rsplit(".", maxsplit=1)[1].isdigit()
        ),
        key=lambda path: int(path.name.rsplit(".", maxsplit=1)[1]),
    )
    return [
        [argument.decode() for argument in path.read_bytes().split(b"\0")[:-1]]
        for path in paths
    ]


def _run_script(
    tmp_path: Path,
    script_name: str,
    args: Sequence[str],
    fake_commands: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    fake_bodies: Mapping[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / "bin"
    log_dir = tmp_path / "logs"
    bin_dir.mkdir()
    log_dir.mkdir()
    for name in fake_commands:
        _install_fake_command(bin_dir, name, (fake_bodies or {}).get(name, ""))

    child_env = os.environ.copy()
    child_env.update(env or {})
    child_env["FAKE_LOG_DIR"] = str(log_dir)
    child_env["PATH"] = os.pathsep.join((str(bin_dir), child_env["PATH"]))
    result = subprocess.run(
        ["bash", str(SCRIPT_DIR / script_name), *args],
        cwd=PROJECT_ROOT,
        env=child_env,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result, log_dir


def test_submit_executes_policy_check_and_forwards_a_safe_job_request(tmp_path) -> None:
    result, log_dir = _run_script(
        tmp_path,
        "submit.sh",
        [
            "./scripts/grid5000/run.sh",
            "--",
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "package",
            "--label",
            "value with spaces",
        ],
        ("usagepolicycheck", "oarstat", "oarsub"),
        env={
            "GRID5000_RUN_ID": "run-001",
            "GRID5000_RESOURCES": "host=1/gpu=1,walltime=0:30",
            "GRID5000_QUEUE": "gpu",
            "GRID5000_PROPERTIES": "gpu=V100",
            "GRID5000_TYPE": "besteffort",
            "FAKE_OARSTAT_STDOUT": "an unrelated active job",
        },
    )

    assert result.returncode == 0, result.stderr
    assert _read_invocations(log_dir, "usagepolicycheck") == [["-t"], ["-t"]]
    assert _read_invocations(log_dir, "oarstat") == [["-u"]]
    assert _read_invocations(log_dir, "oarsub") == [
        [
            "-n",
            "compute-cost-encoders-llms-run-001",
            "-l",
            "host=1/gpu=1,walltime=0:30",
            "-q",
            "gpu",
            "-p",
            "gpu=V100",
            "-t",
            "besteffort",
            "./scripts/grid5000/run.sh",
            "--",
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "package",
            "--label",
            "value with spaces",
        ]
    ]


def test_submit_rejects_an_active_duplicate_before_submission(tmp_path) -> None:
    result, log_dir = _run_script(
        tmp_path,
        "submit.sh",
        ["./scripts/grid5000/run.sh", "--", "uv", "run", "--locked", "python"],
        ("usagepolicycheck", "oarstat", "oarsub"),
        env={
            "GRID5000_RUN_ID": "run-duplicate",
            "FAKE_OARSTAT_STDOUT": "compute-cost-encoders-llms-run-duplicate",
        },
    )

    assert result.returncode != 0
    assert "refusing a duplicate" in result.stderr
    assert _read_invocations(log_dir, "oarsub") == []


def test_run_exports_reproducibility_context_and_executes_locked_uv(tmp_path) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[benchmark]\nrepetitions = 1\n", encoding="utf-8")
    nodefile = tmp_path / "nodefile"
    nodefile.write_text("node-001\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    fake_bodies = {
        "git": 'if [[ "$*" == "rev-parse HEAD" ]]; then printf "%s" "commit-001"; fi\n',
        "uv": (
            "{\n"
            '    printf "%s\\n" "$GRID5000_SOURCE_COMMIT"\n'
            '    printf "%s\\n" "$GRID5000_CONFIG_REVISION"\n'
            '    printf "%s\\n" "$UV_CACHE_DIR"\n'
            '    printf "%s\\n" "$HF_HOME"\n'
            '    printf "%s\\n" "$PYTHONHASHSEED"\n'
            '} > "$FAKE_LOG_DIR/uv.env"\n'
        ),
    }

    result, log_dir = _run_script(
        tmp_path,
        "run.sh",
        [
            "--",
            "uv",
            "run",
            "--locked",
            "python",
            "-m",
            "package",
            "value with spaces",
        ],
        ("git", "uv"),
        env={
            "OAR_JOB_ID": "123",
            "OAR_NODEFILE": str(nodefile),
            "GRID5000_RUN_ID": "run-001",
            "GRID5000_CONFIG_PATH": str(config),
            "GRID5000_DATASET_REVISION": "dataset@revision",
            "GRID5000_MODEL_REVISION": "model@revision",
            "GRID5000_CHECKPOINT_INTERVAL": "100",
            "GRID5000_ARTIFACT_PREFIX": "runs/run-001",
            "GRID5000_CACHE_ROOT": str(cache_root),
            "UV_CACHE_DIR": "",
            "HF_HOME": "",
            "PYTHONHASHSEED": "",
        },
        fake_bodies=fake_bodies,
    )

    assert result.returncode == 0, result.stderr
    assert _read_invocations(log_dir, "uv") == [
        ["run", "--locked", "python", "-m", "package", "value with spaces"]
    ]
    assert (cache_root / "uv").is_dir()
    assert (cache_root / "huggingface").is_dir()
    exported = (log_dir / "uv.env").read_text(encoding="utf-8").splitlines()
    assert exported[0] == "commit-001"
    assert exported[1].startswith("sha256:")
    assert exported[2] == str(cache_root / "uv")
    assert exported[3] == str(cache_root / "huggingface")
    assert exported[4] == "0"


def test_publish_validates_metadata_and_syncs_to_the_project_bucket(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "checkpoint.json").write_text("{}", encoding="utf-8")

    result, log_dir = _run_script(
        tmp_path,
        "publish.sh",
        [str(artifact_dir), "runs/run-001"],
        ("uv", "usagepolicycheck", "hf"),
        env={
            "GRID5000_PUBLISH_CONFIRM": "yes",
            "HF_BUCKET_URI": "hf://buckets/NoeFlandre/compute-cost-encoders-llms",
        },
    )

    assert result.returncode == 0, result.stderr
    assert _read_invocations(log_dir, "uv") == [
        [
            "run",
            "--locked",
            "python",
            "scripts/grid5000/checkpoint_metadata.py",
            str(artifact_dir / "checkpoint.json"),
        ]
    ]
    assert _read_invocations(log_dir, "usagepolicycheck") == [["-t"], ["-t"]]
    assert _read_invocations(log_dir, "hf") == [
        [
            "buckets",
            "sync",
            str(artifact_dir),
            "hf://buckets/NoeFlandre/compute-cost-encoders-llms/runs/run-001",
        ]
    ]


def test_publish_requires_explicit_confirmation_before_invoking_external_tools(
    tmp_path,
) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "checkpoint.json").write_text("{}", encoding="utf-8")

    result, log_dir = _run_script(
        tmp_path,
        "publish.sh",
        [str(artifact_dir), "runs/run-001"],
        ("uv", "usagepolicycheck", "hf"),
        env={
            "HF_BUCKET_URI": "hf://buckets/NoeFlandre/compute-cost-encoders-llms",
        },
    )

    assert result.returncode != 0
    assert "GRID5000_PUBLISH_CONFIRM=yes" in result.stderr
    assert _read_invocations(log_dir, "uv") == []
    assert _read_invocations(log_dir, "hf") == []


def test_benchmark_encoder_branch_forwards_paths_and_exposes_checkout_package(
    tmp_path,
) -> None:
    config = tmp_path / "config.toml"
    config.write_text("[benchmark]\nrepetitions = 1\n", encoding="utf-8")
    output_dir = tmp_path / "output"
    fake_bodies = {
        "uv": 'printf "%s" "$PYTHONPATH" > "$FAKE_LOG_DIR/uv.env"\n',
    }

    result, log_dir = _run_script(
        tmp_path,
        "benchmark.sh",
        ["encoder", str(config), str(output_dir), "run-001"],
        ("uv",),
        fake_bodies=fake_bodies,
    )

    assert result.returncode == 0, result.stderr
    assert (output_dir / "encoder").is_dir()
    assert _read_invocations(log_dir, "uv") == [
        [
            "run",
            "--locked",
            "--extra",
            "benchmark",
            "--no-dev",
            "python",
            "-m",
            "compute_cost_encoders_llms.benchmark.cli",
            "--config",
            str(config),
            "--output-dir",
            str(output_dir / "encoder"),
            "--backend",
            "encoder",
            "--run-id",
            "run-001-encoder",
        ]
    ]
    pythonpath = (log_dir / "uv.env").read_text(encoding="utf-8")
    assert str(PROJECT_ROOT / "src") in pythonpath.split(os.pathsep)
    assert str(PROJECT_ROOT) in pythonpath.split(os.pathsep)
