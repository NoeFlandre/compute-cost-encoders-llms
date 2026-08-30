# Grid5000 Shell Contract Tests Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace source-text-only confidence in the critical Grid5000 wrappers with executable contract tests that verify their real exit status, command forwarding, environment propagation, and safety guards in isolated subprocesses.

**Architecture:** Add an integration test module that invokes each wrapper with `bash` and a temporary `PATH` containing deterministic fake external commands. The fakes record one invocation per file and can emit controlled output, allowing tests to assert exact argument vectors without contacting OAR, Hugging Face, `uv`, or a model server. Keep the existing static tests for immutable configuration and release metadata; the new tests cover runtime semantics. No production script change is expected unless a red contract test demonstrates a documented behavior mismatch.

**Tech Stack:** Bash, Python 3.12, `subprocess`, `pathlib`, pytest, Ruff, ty, import-linter, crap4py, and mutmut.

---

### Task 1: Establish the current Grid5000 test baseline

**Files:**
- Read: `scripts/grid5000/submit.sh`
- Read: `scripts/grid5000/run.sh`
- Read: `scripts/grid5000/publish.sh`
- Read: `scripts/grid5000/benchmark.sh`
- Read: `tests/unit/test_grid5000_guardrails.py`
- Test: `tests/integration/test_grid5000_scripts.py`

- [x] **Step 1: Confirm the existing static guardrail tests pass before editing.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_grid5000_guardrails.py -q
```

Expected: the existing source-text checks pass, and no executable shell-contract integration module exists yet.

### Task 2: Add a deterministic subprocess harness and submission contracts

**Files:**
- Create: `tests/integration/test_grid5000_scripts.py`

- [x] **Step 1: Add helpers that install recording fake commands and invoke a wrapper in an isolated environment.**

The helper must write executable Bash fakes under `tmp_path/bin`, prepend that directory to `PATH`, store each invocation as NUL-delimited arguments under `tmp_path/logs`, and return both `subprocess.CompletedProcess[str]` and the log directory. Use `subprocess.run` with `capture_output=True`, `text=True`, `check=False`, `timeout=10`, and `cwd=PROJECT_ROOT`.

```python
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
    script += body
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
```

- [x] **Step 2: Add the executable contract for `submit.sh`.**

The test must run with fake `usagepolicycheck`, `oarstat`, and `oarsub` commands, configure queue, properties, and type overrides, then assert policy checks run before and on exit, the active-job query receives `-u`, and `oarsub` receives the exact options and command arguments including an argument containing spaces.

```python
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
```

- [x] **Step 3: Add the duplicate-job rejection contract.**

Run `submit.sh` with `oarstat` output containing the exact generated job name. Assert the wrapper exits nonzero, reports the duplicate, and never invokes `oarsub`.

```python
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
```

### Task 3: Add runner, publisher, and benchmark execution contracts

**Files:**
- Modify: `tests/integration/test_grid5000_scripts.py`

- [x] **Step 1: Test `run.sh` environment recording and exact command forwarding.**

Create a config file and non-empty OAR nodefile. Fake `git` so the cleanliness probes succeed and `rev-parse HEAD` returns a deterministic commit; fake `uv` to record its arguments and the exported source commit, configuration digest, cache paths, and hash seed. Assert the wrapper creates both cache directories and forwards the command unchanged.

```python
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
        ["--", "uv", "run", "--locked", "python", "-m", "package", "value with spaces"],
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
```

- [x] **Step 2: Test `publish.sh` validation and exact HF destination forwarding.**

Create a checkpoint file and run with explicit confirmation, fake `uv`, `usagepolicycheck`, and `hf`. Assert the metadata validator command is attempted first, policy checks run before and on exit, and `hf buckets sync` receives the artifact directory and project bucket destination as separate arguments.

```python
def test_publish_validates_metadata_and_syncs_to_the_project_bucket(tmp_path) -> None:
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir()
    (artifact_dir / "checkpoint.json").write_text("{}", encoding="utf-8")

    result, log_dir = _run_script(
        tmp_path,
        "publish.sh",
        [str(artifact_dir), "runs/run-001"],
        ("uv", "usagepolicycheck", "hf"),
        env={"GRID5000_PUBLISH_CONFIRM": "yes"},
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
    assert _read_invocations(log_dir, "hf") == [
        [
            "buckets",
            "sync",
            str(artifact_dir),
            "hf://buckets/NoeFlandre/compute-cost-encoders-llms/runs/run-001",
        ]
    ]
```

- [x] **Step 3: Test the encoder branch of `benchmark.sh` without loading a model.**

Create a config file, fake `uv`, and invoke the encoder backend. Assert the output directory is created, the Python benchmark module receives the exact paths and run ID, and the wrapper adds the checkout source and `src` directory to `PYTHONPATH`.

```python
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
```

- [x] **Step 4: Run the new integration module and correct only genuine contract failures.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/integration/test_grid5000_scripts.py -q
```

Expected: all executable wrapper contracts pass. If a test fails, first correct the fake-command harness or assertion when it does not reflect the documented shell contract; only a confirmed production mismatch may receive a minimal script patch, followed by a red-to-green rerun.

### Task 4: Run all quality gates and publish the scoped change

**Files:**
- Modify: `tests/integration/test_grid5000_scripts.py`
- Create: `docs/superpowers/plans/2026-08-30-grid5000-shell-contract-tests.md`

- [x] **Step 1: Run formatting, linting, typing, and focused tests.**

Run:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check tests/integration/test_grid5000_scripts.py
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check tests/integration/test_grid5000_scripts.py
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_grid5000_guardrails.py tests/integration/test_grid5000_scripts.py -q
```

Expected: every command exits 0 and the test output contains no warnings from the new harness.

- [x] **Step 2: Run the complete repository QA gate.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Expected: unit, integration, and acceptance tests pass; coverage remains at least 99%; Ruff, ty, import-linter, and CRAP pass; mutation testing reports no surviving, suspicious, timed-out, or unchecked mutants.

- [x] **Step 3: Review and commit only the executable-test plan and test module.**

Run:

```bash
git diff --check
git diff -- tests/integration/test_grid5000_scripts.py docs/superpowers/plans/2026-08-30-grid5000-shell-contract-tests.md
git status --short
git add docs/superpowers/plans/2026-08-30-grid5000-shell-contract-tests.md tests/integration/test_grid5000_scripts.py
git commit -m "test: exercise grid5000 shell contracts"
```

Expected: no Grid5000 production script changes are staged unless a documented contract failure required one; unrelated files and generated QA artifacts remain unstaged.

- [x] **Step 4: Push and verify the exact remote commit.**

Run:

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

Expected: the push succeeds, the worktree is clean, and the local, tracking, and remote `main` references resolve to the same commit.
