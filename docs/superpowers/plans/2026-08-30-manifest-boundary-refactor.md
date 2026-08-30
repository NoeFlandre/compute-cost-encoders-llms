# Benchmark Manifest Boundary Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move pure benchmark manifest construction out of CLI orchestration
while preserving every existing manifest value, signature, import path, and
runtime behavior.

**Architecture:** Add `benchmark/manifest.py` as the canonical owner of the
pure `build_manifest` contract. `benchmark/cli.py` will retain an exact
import-level facade and continue to own backend loading, measurement,
runtime observation, artifact writing, and command-line parsing.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, import-linter, crap4py,
mutmut.

---

### Task 1: Establish the manifest ownership boundary with a red test

**Files:**
- Create: `tests/unit/test_manifest_boundary.py`
- Read: `src/compute_cost_encoders_llms/benchmark/cli.py`
- Read: `src/compute_cost_encoders_llms/benchmark/manifest.py`

- [ ] **Step 1: Write the failing boundary test**

Create the test before adding the new production module:

```python
from __future__ import annotations

import compute_cost_encoders_llms.benchmark.cli as cli_module
import compute_cost_encoders_llms.benchmark.manifest as manifest_module


def test_cli_manifest_builder_is_owned_by_manifest_module() -> None:
    assert cli_module.build_manifest is manifest_module.build_manifest
```

- [ ] **Step 2: Run the focused test and verify the expected red**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_manifest_boundary.py -q
```

Expected result: collection fails with `ModuleNotFoundError` because
`benchmark.manifest` does not exist yet. Do not add production code before
capturing this failure.

### Task 2: Move the pure manifest implementation to its canonical module

**Files:**
- Create: `src/compute_cost_encoders_llms/benchmark/manifest.py`
- Modify: `src/compute_cost_encoders_llms/benchmark/cli.py`

- [ ] **Step 1: Add the minimal canonical implementation**

Create `manifest.py` with the existing function body and only the imports it
needs:

```python
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
```

Do not add validation, normalization, I/O, environment access, or new
abstractions.

- [ ] **Step 2: Replace the CLI implementation with a compatibility import**

In `cli.py`, remove the `LANDUSE_QUESTION`, `LANDUSE_SENTENCE`,
`candidate_label_forms`, and `candidate_labels` import block and delete the
old `build_manifest` function. Import the canonical callable with the other
local imports:

```python
from .manifest import build_manifest
```

Leave all callers, including `run()` and the command-line entry point,
unchanged so `cli_module.build_manifest` remains available and identical to
the canonical implementation.

- [ ] **Step 3: Run the focused suite to reach green**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_manifest_boundary.py tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_contract.py tests/integration/test_benchmark_pipeline.py -q
```

Expected result: all selected tests pass, including exact manifest values,
CLI orchestration, and the new ownership assertion.

### Task 3: Make test ownership explicit after green

**Files:**
- Modify: `tests/unit/test_benchmark_cli.py`
- Modify: `tests/unit/test_manifest_boundary.py`

- [ ] **Step 1: Import the behavior test from the canonical owner**

Remove `build_manifest` from the `benchmark.cli` import list in
`test_benchmark_cli.py` and add:

```python
from compute_cost_encoders_llms.benchmark.manifest import build_manifest
```

Keep the existing `cli_module.build_manifest` monkeypatch in the `run()`
test; that test verifies the CLI orchestration seam, while the boundary test
covers the compatibility identity.

- [ ] **Step 2: Keep the boundary test minimal and behavior-focused**

Retain exactly one identity assertion in `test_manifest_boundary.py`. Do not
duplicate the detailed manifest fixture, because the existing exact-schema
test remains the regression oracle for values, nested structures, and copied
mappings.

- [ ] **Step 3: Run focused static and behavioral checks**

Run:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_manifest_boundary.py tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_contract.py tests/integration/test_benchmark_pipeline.py -q
```

Expected result: no formatting, lint, type, or focused test failures.

### Task 4: Run the complete quality and mutation verification

**Files:**
- Modify none unless a gate identifies a direct regression in this refactor.

- [ ] **Step 1: Run the complete repository gate**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Require clean Ruff, formatting, ty, unit, integration, acceptance,
architecture, CRAP, and mutation stages. The baseline has 166 unit tests, 1
integration test, 6 acceptance tests, 99% coverage, CRAP maximum 5.0, and a
fresh mutation campaign with 3,175 killed and 49 no-test CLI mutants.

- [ ] **Step 2: Inspect mutation categories explicitly**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run mutmut results
```

There must be no `survived` or `suspicious` entries. If the moved source is
not represented by the mutation workspace, move the generated ignored
`mutants/` directory to a recoverable temporary path and rerun the same full
campaign; do not report stale results.

- [ ] **Step 3: Review the final diff and scope**

Run:

```bash
git diff --check
git status --short --branch
git diff --stat
```

Confirm only the manifest module, CLI facade, focused test ownership change,
boundary test, design document, and plan document are present.

### Task 5: Commit and push the validated refactor

**Files:**
- `docs/superpowers/specs/2026-08-30-manifest-boundary-design.md`
- `docs/superpowers/plans/2026-08-30-manifest-boundary-refactor.md`
- `src/compute_cost_encoders_llms/benchmark/manifest.py`
- `src/compute_cost_encoders_llms/benchmark/cli.py`
- `tests/unit/test_manifest_boundary.py`
- `tests/unit/test_benchmark_cli.py`

- [ ] **Step 1: Stage only the approved paths**

Run:

```bash
git add docs/superpowers/specs/2026-08-30-manifest-boundary-design.md docs/superpowers/plans/2026-08-30-manifest-boundary-refactor.md src/compute_cost_encoders_llms/benchmark/manifest.py src/compute_cost_encoders_llms/benchmark/cli.py tests/unit/test_manifest_boundary.py tests/unit/test_benchmark_cli.py
git diff --cached --check
```

- [ ] **Step 2: Commit with a focused Conventional Commit message**

Run:

```bash
git commit -m "refactor: isolate benchmark manifest construction"
```

- [ ] **Step 3: Push and verify remote synchronization**

Run:

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

The local branch must be clean and the local `HEAD`, tracking ref, and remote
advertised `main` commit must agree.
