# CLI Contract Coverage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the benchmark command-line boundary directly testable and specify its accepted, rejected, and delegated invocations without changing the existing shell interface.

**Architecture:** Keep `_parser()` as the single source of command-line argument definitions. Add an optional argument vector to `main`; `None` preserves `argparse`'s existing process-argument behavior, while tests can supply arguments without mutating process-global state. Contract tests will cover successful parsing, required options, invalid choices, help output, and exact delegation to `run`.

**Tech Stack:** Python 3.12, `argparse`, pytest, Ruff, ty, import-linter, crap4py, and mutmut.

---

### Task 1: Establish the CLI test baseline

**Files:**
- Read: `src/compute_cost_encoders_llms/benchmark/cli.py:232-247`
- Read: `tests/unit/test_benchmark_cli.py`

- [ ] **Step 1: Run the focused CLI tests before editing.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_cli.py -q
```

Expected: the existing CLI unit tests pass and no parser or `main` contract tests are collected yet.

### Task 2: Add parser and delegation contract tests (RED)

**Files:**
- Modify: `tests/unit/test_benchmark_cli.py`

- [ ] **Step 1: Add tests for the parser contract and injectable `main` arguments.**

Add the following imports and tests near the top-level CLI tests:

```python
import sys


def test_parser_parses_all_benchmark_options(tmp_path) -> None:
    config_path = tmp_path / "benchmark.toml"
    output_dir = tmp_path / "output"

    args = cli_module._parser().parse_args(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--backend",
            "llm",
            "--run-id",
            "run-001",
        ]
    )

    assert args.config == config_path
    assert args.output_dir == output_dir
    assert args.backend == "llm"
    assert args.run_id == "run-001"


@pytest.mark.parametrize(
    "omitted",
    ["--config", "--output-dir", "--backend", "--run-id"],
)
def test_parser_requires_each_benchmark_option(omitted, tmp_path, capsys) -> None:
    values = {
        "--config": str(tmp_path / "benchmark.toml"),
        "--output-dir": str(tmp_path / "output"),
        "--backend": "encoder",
        "--run-id": "run-001",
    }
    argv = [
        item
        for option, value in values.items()
        if option != omitted
        for item in (option, value)
    ]

    with pytest.raises(SystemExit) as error:
        cli_module._parser().parse_args(argv)

    assert error.value.code == 2
    assert f"the following arguments are required: {omitted}" in capsys.readouterr().err


def test_parser_rejects_an_unsupported_backend(tmp_path, capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli_module._parser().parse_args(
            [
                "--config",
                str(tmp_path / "benchmark.toml"),
                "--output-dir",
                str(tmp_path / "output"),
                "--backend",
                "unsupported",
                "--run-id",
                "run-001",
            ]
        )

    assert error.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_parser_help_lists_the_benchmark_options(capsys) -> None:
    with pytest.raises(SystemExit) as error:
        cli_module._parser().parse_args(["--help"])

    assert error.value.code == 0
    help_text = capsys.readouterr().out
    assert all(
        option in help_text
        for option in ("--config", "--output-dir", "--backend", "--run-id")
    )


def test_parser_does_not_derive_help_from_the_module_docstring(monkeypatch) -> None:
    monkeypatch.setattr(cli_module, "__doc__", "unrelated module documentation")

    assert cli_module._parser().description is None


def test_main_delegates_supplied_arguments_to_run(monkeypatch, tmp_path) -> None:
    calls = []
    config_path = tmp_path / "benchmark.toml"
    output_dir = tmp_path / "output"

    def capture_run(config, output, backend, run_id) -> None:
        calls.append((config, output, backend, run_id))

    monkeypatch.setattr(cli_module, "run", capture_run)

    cli_module.main(
        [
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--backend",
            "encoder",
            "--run-id",
            "run-002",
        ]
    )

    assert calls == [(config_path, output_dir, "encoder", "run-002")]


def test_main_without_arguments_reads_process_arguments(monkeypatch, tmp_path) -> None:
    calls = []
    config_path = tmp_path / "benchmark.toml"
    output_dir = tmp_path / "output"

    def capture_run(config, output, backend, run_id) -> None:
        calls.append((config, output, backend, run_id))

    monkeypatch.setattr(cli_module, "run", capture_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--config",
            str(config_path),
            "--output-dir",
            str(output_dir),
            "--backend",
            "llm",
            "--run-id",
            "run-003",
        ],
    )

    cli_module.main()

    assert calls == [(config_path, output_dir, "llm", "run-003")]
```

- [ ] **Step 2: Run the new tests and confirm the intended RED failure.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_cli.py -q
```

Expected: parser tests pass against the current parser, while `test_main_delegates_supplied_arguments_to_run` fails because the current `main()` does not accept an argument vector. The failure must be a `TypeError` from the test call, not a collection or assertion error.

### Task 3: Add the smallest compatible testability seam (GREEN)

**Files:**
- Modify: `src/compute_cost_encoders_llms/benchmark/cli.py:241-244`

- [ ] **Step 1: Pass an optional argument vector through `main`.**

Add `Sequence` to the `collections.abc` imports and update the entry point:

```python
from collections.abc import Callable, Mapping, Sequence


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    run(args.config, args.output_dir, args.backend, args.run_id)
```

Calling `main()` still passes `None` to `argparse`, which reads `sys.argv` exactly as before.

- [ ] **Step 2: Remove the redundant module-docstring coupling.**

Update `_parser()` to construct the default parser directly:

```python
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
```

This preserves the current help output because `cli.py` has no module docstring, while making the parser independent of mutable module metadata.

- [ ] **Step 3: Run the focused tests.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_cli.py -q
```

Expected: all CLI unit tests pass.

### Task 4: Refactor and validate the CLI boundary

**Files:**
- Modify: `src/compute_cost_encoders_llms/benchmark/cli.py`
- Modify: `tests/unit/test_benchmark_cli.py`

- [ ] **Step 1: Run formatting, linting, and type checks on changed files.**

Run:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check src/compute_cost_encoders_llms/benchmark/cli.py tests/unit/test_benchmark_cli.py
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check src/compute_cost_encoders_llms/benchmark/cli.py tests/unit/test_benchmark_cli.py
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
```

Expected: every command exits 0.

- [ ] **Step 2: Run the full repository QA gate.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Expected: unit, integration, and acceptance tests pass; coverage remains at least 99%; import-linter passes; CRAP remains below 6; mutation testing reports no surviving or suspicious mutants. The formerly untested CLI mutations should be killed by the parser and delegation contracts, and the redundant docstring mutation should no longer exist.

- [ ] **Step 3: Review the exact diff and commit only the scoped files.**

Run:

```bash
git diff --check
git diff -- src/compute_cost_encoders_llms/benchmark/cli.py tests/unit/test_benchmark_cli.py
git status --short
git add src/compute_cost_encoders_llms/benchmark/cli.py tests/unit/test_benchmark_cli.py
git commit -m "test: cover benchmark cli contract"
```

Expected: the diff contains only the optional `argv` seam, the redundant parser-argument cleanup, and CLI contract tests; generated QA artifacts and unrelated work remain unstaged.

- [ ] **Step 4: Push and verify synchronization.**

Run:

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

Expected: push succeeds, the worktree is clean, and all three commit references resolve to the same commit.
