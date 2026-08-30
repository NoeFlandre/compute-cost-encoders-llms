from __future__ import annotations

import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
QA_SCRIPT = PROJECT_ROOT / "scripts" / "qa.sh"


def _install_fake_uv(bin_dir: Path) -> None:
    script = """#!/usr/bin/env bash
set -euo pipefail
count_file="$QA_LOG_DIR/uv.count"
count=0
if [[ -f "$count_file" ]]; then count="$(<"$count_file")"; fi
count=$((count + 1))
printf '%s' "$count" > "$count_file"
printf '%s\\0' "$@" > "$QA_LOG_DIR/uv.$count"
if [[ "${1:-}" == "run" && "${2:-}" == "mutmut" && "${3:-}" == "results" ]]; then
    printf '%s' "${QA_MUTATION_RESULTS:-}"
fi
"""
    path = bin_dir / "uv"
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)


def _run_qa(
    tmp_path: Path, mutation_results: str
) -> tuple[subprocess.CompletedProcess[str], Path]:
    bin_dir = tmp_path / "bin"
    log_dir = tmp_path / "logs"
    bin_dir.mkdir()
    log_dir.mkdir()
    _install_fake_uv(bin_dir)

    environment = os.environ.copy()
    environment["QA_LOG_DIR"] = str(log_dir)
    environment["QA_MUTATION_RESULTS"] = mutation_results
    environment["PATH"] = os.pathsep.join((str(bin_dir), environment["PATH"]))
    result = subprocess.run(
        ["bash", str(QA_SCRIPT)],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    return result, log_dir


def _read_uv_invocations(log_dir: Path) -> list[list[str]]:
    paths = sorted(
        (
            path
            for path in log_dir.glob("uv.*")
            if path.name.rsplit(".", maxsplit=1)[1].isdigit()
        ),
        key=lambda path: int(path.name.rsplit(".", maxsplit=1)[1]),
    )
    return [
        [argument.decode() for argument in path.read_bytes().split(b"\0")[:-1]]
        for path in paths
    ]


def test_qa_rejects_timed_out_mutation_results(tmp_path: Path) -> None:
    result, _ = _run_qa(tmp_path, "src/example.py:10: timeout\n")

    assert result.returncode != 0
    assert "Mutation testing found" in result.stderr


def test_qa_rejects_mutation_results_with_no_tests(tmp_path: Path) -> None:
    result, _ = _run_qa(tmp_path, "src/example.py:10: no tests\n")

    assert result.returncode != 0
    assert "Mutation testing found" in result.stderr


def test_qa_rejects_an_empty_mutation_inventory(tmp_path: Path) -> None:
    result, _ = _run_qa(tmp_path, "")

    assert result.returncode != 0
    assert "Mutation testing produced no mutants" in result.stderr


def test_qa_runs_all_stages_with_an_all_killed_inventory(tmp_path: Path) -> None:
    result, log_dir = _run_qa(tmp_path, "src/example.py:10: killed\n")

    assert result.returncode == 0, result.stderr
    assert _read_uv_invocations(log_dir) == [
        ["run", "ruff", "check", "."],
        ["run", "ruff", "format", "--check", "."],
        ["run", "ty", "check", "src", "tests", "scripts"],
        [
            "run",
            "pytest",
            "tests/unit",
            "--cov=src",
            "--cov=scripts",
            "--cov-branch",
            "--cov-report=term-missing",
            "--cov-report=lcov:coverage.lcov",
        ],
        [
            "run",
            "pytest",
            "tests/integration",
            "--cov=src",
            "--cov=scripts",
            "--cov-branch",
            "--cov-append",
            "--cov-report=term-missing",
            "--cov-report=lcov:coverage.lcov",
        ],
        [
            "run",
            "pytest",
            "tests/acceptance",
            "--cov=src",
            "--cov=scripts",
            "--cov-branch",
            "--cov-append",
            "--cov-report=term-missing",
            "--cov-report=lcov:coverage.lcov",
        ],
        ["run", "lint-imports", "--no-cache"],
        [
            "run",
            "crap4py",
            "src",
            "scripts",
            "--lcov",
            "coverage.lcov",
            "--max-crap",
            "5.99",
            "--max-workers",
            "1",
        ],
        ["run", "mutmut", "run"],
        ["run", "mutmut", "results", "--all", "true"],
    ]
