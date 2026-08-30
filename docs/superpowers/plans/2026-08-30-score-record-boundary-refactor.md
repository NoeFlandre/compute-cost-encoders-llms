# Score Record Boundary Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move pure backend-score normalization into the measurement module while preserving the `benchmark.cli.score_record` compatibility path and every generated record.

**Architecture:** `benchmark.measurement` will own `ScoreLike`, `MeasurementRecord`, `choose_decision`, and `score_record`. `benchmark.cli` will import `score_record` as an exact facade and retain only orchestration, loading, runtime observation, and artifact-writing responsibilities.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, import-linter, crap4py, mutmut.

---

### Task 1: Establish the score-record ownership boundary with a red test

**Files:**
- Create: `tests/unit/test_score_record_boundary.py`
- Read: `src/compute_cost_encoders_llms/benchmark/cli.py`
- Read: `src/compute_cost_encoders_llms/benchmark/measurement.py`

- [ ] **Step 1: Write the failing boundary test**

Create the new test file before changing production code:

```python
from __future__ import annotations

import compute_cost_encoders_llms.benchmark.cli as cli_module
import compute_cost_encoders_llms.benchmark.measurement as measurement_module


def test_cli_score_record_is_owned_by_measurement_module() -> None:
    assert cli_module.score_record is measurement_module.score_record
```

- [ ] **Step 2: Run the test to verify the expected red**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_score_record_boundary.py -q
```

Expected result: the test fails with an `AttributeError` because
`benchmark.measurement` does not yet expose `score_record`. Do not change
production code until this failure is observed.

### Task 2: Move score normalization into the measurement module

**Files:**
- Modify: `src/compute_cost_encoders_llms/benchmark/measurement.py`
- Modify: `src/compute_cost_encoders_llms/benchmark/cli.py`

- [ ] **Step 1: Add the minimal structural score protocol and canonical function**

In `measurement.py`, import `Protocol` from `typing` and add the following
protocol after `MeasurementRecord`:

```python
class ScoreLike(Protocol):
    @property
    def tokenization_ms(self) -> float | None: ...

    @property
    def model_ms(self) -> float | None: ...

    @property
    def logprob_ms(self) -> float: ...

    @property
    def text_to_logprob_ms(self) -> float: ...

    @property
    def input_tokens(self) -> int | None: ...

    @property
    def logprobs(self) -> dict[str, float]: ...
```

Add the existing `score_record` implementation after `choose_decision`:

```python
def score_record(model: str, repetition: int, score: ScoreLike) -> MeasurementRecord:
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
```

Do not add validation, conversion, normalization, or new behavior.

- [ ] **Step 2: Replace the CLI implementation with the compatibility import**

In `cli.py`:

1. Remove `EncoderScore` from the encoder import list.
2. Remove `LlamaScore` from the Llama import.
3. Remove `choose_decision` from the measurement import.
4. Add `score_record` to the measurement import.
5. Delete the old `score_record` function body.

The resulting measurement import must retain the existing record and timing
imports while exposing the compatibility name:

```python
from .measurement import MeasurementRecord, measure_repetitions, score_record
```

Leave every `score_record(...)` caller, `_encoder_records`, `_llm_records`,
`run`, and the command-line entry point unchanged.

- [ ] **Step 3: Run the focused suite to reach green**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_score_record_boundary.py tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_measurement.py tests/integration/test_benchmark_pipeline.py -q
```

Expected result: the boundary test and all selected behavior tests pass,
including encoder and Llama record normalization, CLI orchestration, and the
integration pipeline.

### Task 3: Make canonical test ownership explicit after green

**Files:**
- Modify: `tests/unit/test_benchmark_cli.py`
- Modify: `tests/unit/test_benchmark_measurement.py`

- [ ] **Step 1: Move the detailed behavior test to the canonical owner**

Remove the detailed `test_score_record_normalizes_backend_score` test from
`test_benchmark_cli.py`. Add the test to `test_benchmark_measurement.py` and
import `score_record` from the measurement module:

```python
from compute_cost_encoders_llms.benchmark.measurement import (
    TimedValue,
    score_record,
)
```

Keep the existing detailed assertions intact. The CLI test retains its
remaining uses of the compatibility import for orchestration assertions, and
the boundary test remains the sole exact callable-identity assertion.

- [ ] **Step 2: Run focused static and behavioral checks**

Run each command separately:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_score_record_boundary.py tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_measurement.py tests/integration/test_benchmark_pipeline.py -q
```

Expected result: no diff, formatting, lint, type, collection, or focused
behavior failures.

### Task 4: Run complete quality and mutation verification

**Files:**
- Modify none unless a check identifies a direct regression in this refactor.

- [ ] **Step 1: Run the complete repository gate**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Require passing Ruff, formatting, ty, unit, integration, acceptance,
architecture, CRAP, and mutation stages. The pre-refactor baseline is 167 unit
tests, 1 integration test, 6 acceptance tests, 99% coverage, CRAP maximum 5.0,
and 3,175 killed mutants with 49 known no-test CLI parser/main mutants. After
the boundary test is added, the complete suite exercises 168 unit tests (175
tests total).

- [ ] **Step 2: Inspect mutation categories explicitly**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run mutmut results
```

Require no `survived`, `suspicious`, or timeout entries. If the mutation
workspace is stale after moving the function, move the exact generated
`mutants/` directory to `/private/tmp/compute-cost-encoders-llms-mutants-score-record`
and rerun the complete mutation campaign; do not use stale results.

- [ ] **Step 3: Review the final pre-commit scope**

Run:

```bash
git diff --check
git status --short --branch
git diff --stat
```

Confirm that only the score-record design and plan documents, the measurement
owner, the CLI compatibility import, the boundary test, and the test import
move are present. Do not stage generated coverage or mutation artifacts.

### Task 5: Commit, push, and verify synchronization

**Files:**
- `docs/superpowers/specs/2026-08-30-score-record-boundary-design.md`
- `docs/superpowers/plans/2026-08-30-score-record-boundary-refactor.md`
- `src/compute_cost_encoders_llms/benchmark/measurement.py`
- `src/compute_cost_encoders_llms/benchmark/cli.py`
- `tests/unit/test_score_record_boundary.py`
- `tests/unit/test_benchmark_cli.py`

- [ ] **Step 1: Stage only the approved paths**

Run:

```bash
git add docs/superpowers/specs/2026-08-30-score-record-boundary-design.md docs/superpowers/plans/2026-08-30-score-record-boundary-refactor.md src/compute_cost_encoders_llms/benchmark/measurement.py src/compute_cost_encoders_llms/benchmark/cli.py tests/unit/test_score_record_boundary.py tests/unit/test_benchmark_cli.py
git diff --cached --check
```

- [ ] **Step 2: Commit with a focused Conventional Commit message**

Run:

```bash
git commit -m "refactor: isolate score record normalization"
```

- [ ] **Step 3: Run the post-commit regression suite**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit tests/integration tests/acceptance -q
```

Require all 175 tests to pass after the commit, not only before it.

- [ ] **Step 4: Push and verify local/tracking/remote state**

Run each command separately:

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

The worktree must be clean and the local `HEAD`, local tracking ref, and
remote `main` must report the same commit.
