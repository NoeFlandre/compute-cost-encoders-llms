# Measurement Label Snapshot Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make one measurement logprob validation pass use one canonical binary-label snapshot while preserving all existing values, errors, helper calls, and public interfaces.

**Architecture:** Keep candidate_labels() as the single source of label truth. Capture its tuple once in _validate_logprobs, pass that tuple to the two private predicates that validate keys and numeric values, and retain optional defaults so direct helper callers continue to resolve the canonical labels themselves.

**Tech Stack:** Python 3.12, pytest, Ruff, ty, import-linter, crap4py, mutmut, uv.

---

### Task 1: Establish the current contract

**Files:**
- Read: src/compute_cost_encoders_llms/benchmark/measurement.py:179-199
- Read: tests/unit/test_benchmark_measurement.py:162-190

- [ ] **Step 1: Run the focused baseline tests**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_measurement.py -q
```

Expected: all measurement unit tests pass before the refactor.

- [ ] **Step 2: Confirm the complete baseline gate**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Expected: Ruff, ty, all unit/integration/acceptance tests, import-linter, CRAP, and mutation testing pass with zero surviving or suspicious mutants.

### Task 2: Add the failing snapshot regression test

**Files:**
- Modify: tests/unit/test_benchmark_measurement.py:162-190

- [ ] **Step 1: Add a test that changes the provider after the first snapshot**

Add this test after test_logprob_validation_helpers_preserve_binary_finite_contract:

```python
def test_logprob_validation_reuses_one_candidate_label_snapshot(monkeypatch) -> None:
    labels = ("positive", "negative")
    calls = 0

    def changing_labels() -> tuple[str, str]:
        nonlocal calls
        calls += 1
        return labels if calls == 1 else ("yes", "no")

    monkeypatch.setattr(measurement_module, "candidate_labels", changing_labels)
    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"positive": -0.1, "negative": -2.2},
    }

    assert validate_measurement(record) == record
    assert calls == 1
```

- [ ] **Step 2: Run the new test and verify the expected RED failure**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_measurement.py::test_logprob_validation_reuses_one_candidate_label_snapshot -q
```

Expected: FAIL with MeasurementError: logprobs must contain yes and no, proving the existing helper calls do not share the first label snapshot.

### Task 3: Thread the snapshot through the private predicates

**Files:**
- Modify: src/compute_cost_encoders_llms/benchmark/measurement.py:179-199

- [ ] **Step 1: Pass one label tuple from _validate_logprobs**

Change the logprob validation to this implementation while preserving its existing error messages:

```python
def _validate_logprobs(record: Mapping[str, object]) -> dict[str, float]:
    logprobs = record["logprobs"]
    labels = candidate_labels()
    if not _has_binary_logprob_keys(logprobs, labels=labels):
        raise MeasurementError("logprobs must contain yes and no")
    if not _has_finite_binary_logprobs(logprobs, labels=labels):
        raise MeasurementError("logprobs must be finite")
    return {label: float(logprobs[label]) for label in labels}
```

- [ ] **Step 2: Let each private predicate accept an optional explicit snapshot**

Use these signatures and implementations so existing one-argument helper callers retain the canonical-label default:

```python
def _has_binary_logprob_keys(
    value: object,
    *,
    labels: tuple[str, str] | None = None,
) -> TypeGuard[Mapping[str, object]]:
    expected = candidate_labels() if labels is None else labels
    return isinstance(value, Mapping) and set(value) == set(expected)


def _has_finite_binary_logprobs(
    logprobs: Mapping[str, object],
    *,
    labels: tuple[str, str] | None = None,
) -> TypeGuard[Mapping[str, int | float]]:
    expected = candidate_labels() if labels is None else labels
    return all(_is_finite_number(logprobs[label]) for label in expected)
```

### Task 4: Verify focused behavior and static quality

**Files:**
- Test: tests/unit/test_benchmark_measurement.py
- Source: src/compute_cost_encoders_llms/benchmark/measurement.py

- [ ] **Step 1: Run the focused tests**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_measurement.py -q
```

Expected: every measurement test passes, including the new snapshot regression test.

- [ ] **Step 2: Run focused static checks**

Run:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check src/compute_cost_encoders_llms/benchmark/measurement.py tests/unit/test_benchmark_measurement.py
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check src/compute_cost_encoders_llms/benchmark/measurement.py tests/unit/test_benchmark_measurement.py
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
```

Expected: all commands exit successfully.

### Task 5: Run the full quality gate and review the diff

**Files:**
- Review: src/compute_cost_encoders_llms/benchmark/measurement.py
- Review: tests/unit/test_benchmark_measurement.py

- [ ] **Step 1: Run the complete QA gate**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Expected: full tests, coverage, import-linter, CRAP under 6, and mutation testing pass with zero surviving or suspicious mutants.

- [ ] **Step 2: Inspect the exact change set**

Run:

```bash
git diff --check
git diff --stat
git diff -- src/compute_cost_encoders_llms/benchmark/measurement.py tests/unit/test_benchmark_measurement.py
```

Expected: only the planned label-snapshot source and test changes are present; no output or error text changes are introduced.

### Task 6: Commit, publish, and verify synchronization

**Files:**
- Commit: src/compute_cost_encoders_llms/benchmark/measurement.py
- Commit: tests/unit/test_benchmark_measurement.py

- [ ] **Step 1: Commit the validated refactor**

Run:

```bash
git add src/compute_cost_encoders_llms/benchmark/measurement.py tests/unit/test_benchmark_measurement.py
git commit -m "refactor: snapshot measurement candidate labels"
```

Expected: one focused Conventional Commit containing only the implementation and regression test.

- [ ] **Step 2: Push the current branch**

Run:

```bash
git push origin main
```

Expected: the new commit is accepted by origin/main.

- [ ] **Step 3: Verify local, tracking, and remote state**

Run:

```bash
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

Expected: a clean worktree and identical commit IDs for local HEAD, origin/main, and the remote main ref.
