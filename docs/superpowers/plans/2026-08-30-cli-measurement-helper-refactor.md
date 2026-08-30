# CLI Measurement Helper Refactor Implementation Plan

> **For agentic workers:** Use the red → green → refactor cycle for every
> implementation step, and run the complete verification gate before commit.

**Goal:** Remove duplicated timing-to-record conversion from the encoder and
LLM CLI paths without changing backend behavior, measurement semantics, or
artifact output.

**Architecture:** Keep backend loading and runtime metadata collection in
`benchmark.cli`. Add a private `_measure_records` helper there for the shared
`measure_repetitions` and `score_record` orchestration. Both backend-specific
functions will supply only their operation and model label.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, import-linter, crap4py,
mutmut.

## Task 1: Establish the helper contract with a red test

**Files:**

- Create: `tests/unit/test_cli_measurement_boundary.py`
- Read: `src/compute_cost_encoders_llms/benchmark/cli.py`

### Step 1: Write the failing test

Add a focused test that calls the intended private helper with a small fake
score, one warmup, and two measured repetitions. Assert that warmups are not
returned, the operation is called three times, and the measured values become
ordered records with the supplied model label and repetition indexes.

### Step 2: Run the test and verify red

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_cli_measurement_boundary.py -q
```

Expected: collection fails because `_measure_records` does not yet exist.
Do not add production code until this red state is observed.

## Task 2: Implement the smallest shared orchestration helper

**Files:**

- Modify: `src/compute_cost_encoders_llms/benchmark/cli.py`

### Step 1: Add the minimal helper

Define `_measure_records` beside the backend record functions. It accepts a
model label, a zero-argument callable returning `ScoreLike`, and keyword
`warmups` and `repetitions` values. It must delegate to the existing
`measure_repetitions` and return `[score_record(model, index, item.value) ...]`
in the existing order.

Add only the type imports required by the signature. Do not change timing,
validation, score conversion, or backend behavior.

### Step 2: Delegate both backend paths

Replace the duplicated `measure_repetitions` plus list-comprehension blocks in
`_encoder_records` and `_llm_records` with `_measure_records` calls. Preserve
the existing lambdas, candidate-token-ID reuse, warmup/repetition config, seed,
runtime construction, and return shape exactly.

### Step 3: Run focused tests to reach green

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_cli_measurement_boundary.py tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_measurement.py tests/integration/test_benchmark_pipeline.py -q
```

Expected: the new helper test and all existing CLI, measurement, and pipeline
tests pass.

## Task 3: Refactor after green and check the exact diff

**Files:**

- Modify: `src/compute_cost_encoders_llms/benchmark/cli.py`
- Modify: `tests/unit/test_cli_measurement_boundary.py`

### Step 1: Simplify only if the green implementation reveals duplication

Keep the helper private and local to `cli.py`; do not introduce a new public
measurement API or alter existing compatibility imports. Keep the focused test
behavioral rather than asserting implementation details beyond the helper's
contract.

### Step 2: Run focused static checks

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
```

Expected: no whitespace, formatting, lint, or type errors.

## Task 4: Run the complete quality and mutation gates

**Files:** No further changes unless a gate exposes a direct regression.

### Step 1: Run the full QA script

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Require all unit, integration, and acceptance tests; coverage; import linting;
CRAP below 6; and mutation testing with no surviving or suspicious mutants.
Known no-test CLI parser/main mutants remain a separate baseline category.

### Step 2: Inspect mutation categories explicitly

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run mutmut results
```

If a survivor or suspicious mutant appears, add a red characterization test,
make the smallest fix, rerun the affected tests, and repeat the complete gate.

## Task 5: Commit, push, and verify synchronization

### Step 1: Review exact scope

```bash
git status --short --branch
git diff --check
git diff --stat
```

Stage only the design, plan, helper implementation, and focused test paths.

### Step 2: Commit

```bash
git add docs/superpowers/specs/2026-08-30-cli-measurement-helper-design.md docs/superpowers/plans/2026-08-30-cli-measurement-helper-refactor.md src/compute_cost_encoders_llms/benchmark/cli.py tests/unit/test_cli_measurement_boundary.py
git commit -m "refactor: share CLI measurement record conversion"
```

### Step 3: Run a fresh post-commit regression suite

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit tests/integration tests/acceptance -q
```

### Step 4: Push and verify the refs

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

Require a clean worktree and matching local, tracking, and remote commit IDs.
