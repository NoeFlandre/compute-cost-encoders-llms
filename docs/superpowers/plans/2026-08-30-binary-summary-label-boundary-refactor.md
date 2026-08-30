# Binary Summary Label Boundary Refactor Implementation Plan

> **For agentic workers:** Use the red → green → refactor cycle for every
> implementation step, and run the complete verification gate before commit.

**Goal:** Remove duplicated binary-label knowledge from the reporting summary
builder without changing any benchmark output or compatibility surface.

**Architecture:** Keep `candidate_labels()` as the protocol owner in
`benchmark.example`. Import it into `benchmark.reporting`, capture its stable
ordered result once in `_model_summary`, and use that result for both decision
counts and mean log-probability mappings.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, import-linter, crap4py,
mutmut.

## Task 1: Establish the shared-label contract with a red test

**Files:**

- Modify: `tests/unit/test_benchmark_measurement.py`
- Read: `src/compute_cost_encoders_llms/benchmark/reporting.py`

### Step 1: Write the failing test

Add a focused test that replaces the reporting module's label provider,
builds one valid measurement summary, and asserts the provider is consulted
once while the existing binary values remain correct.

### Step 2: Run the test and verify red

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_measurement.py -q
```

Expected: the new assertion fails because `_model_summary` currently has no
shared-label lookup.

## Task 2: Implement the smallest refactor

**Files:**

- Modify: `src/compute_cost_encoders_llms/benchmark/reporting.py`

### Step 1: Import the canonical provider

Import `candidate_labels` from `benchmark.example`.

### Step 2: Reuse one local ordered label tuple

In `_model_summary`, capture `labels = candidate_labels()`, initialize the
decision mapping from `labels`, and build `mean_logprobs` by iterating over
`labels`. Preserve all existing calls, values, ordering, and return keys.

### Step 3: Run focused tests to reach green

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_measurement.py tests/unit/test_benchmark_cli.py tests/unit/test_render_report.py -q
```

## Task 3: Refactor review and focused quality checks

Keep the change local if the green implementation is already clear. Run:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
```

## Task 4: Run complete quality and mutation gates

Run `UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache
./scripts/qa.sh`, then inspect `uv run mutmut results`. Require all existing
tests, coverage, import linting, CRAP below 6, and no surviving or suspicious
mutants. Known no-test mutations remain a separately reported baseline
category.

## Task 5: Review, commit, push, and verify synchronization

Review only the design, plan, reporting implementation, and focused test
paths. Commit with a Conventional Commit message, run the complete test suite
again after committing, push `main`, and verify the worktree plus local,
tracking, and remote `main` SHAs match.
