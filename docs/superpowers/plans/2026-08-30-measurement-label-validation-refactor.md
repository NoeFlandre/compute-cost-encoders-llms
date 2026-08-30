# Measurement Label Validation Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove residual hard-coded binary-label knowledge from measurement validation without changing current records, errors, or outputs.

**Architecture:** Keep `candidate_labels()` in `benchmark.example` as the protocol owner. Replace only the repeated label literals in `benchmark.measurement` validation paths with that existing helper, retaining all function names, signatures, error messages, and return shapes.

**Tech Stack:** Python 3.12, pytest, pytest-cov, Ruff, ty, import-linter, crap4py, mutmut, uv.

---

### Task 1: Make measurement validation use the canonical label contract

**Files:**
- Modify: `src/compute_cost_encoders_llms/benchmark/measurement.py:179-218` — replace duplicated binary-label literals in validation helpers with `candidate_labels()`.
- Test: `tests/unit/test_benchmark_measurement.py:162-174` — add one contract test for a non-default canonical label pair while preserving existing default-label tests.

- [ ] **Step 1: Write the failing test.**

Add this test after `test_logprob_validation_helpers_preserve_binary_finite_contract`:

```python
def test_measurement_validation_uses_shared_candidate_labels(monkeypatch) -> None:
    labels = ("positive", "negative")
    monkeypatch.setattr(measurement_module, "candidate_labels", lambda: labels)
    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"positive": -0.1, "negative": -2.2},
        "decision": "positive",
    }

    assert validate_measurement(record) == record
```

- [ ] **Step 2: Run the focused test to verify the red state.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_measurement.py::test_measurement_validation_uses_shared_candidate_labels -q
```

Expected: `FAIL`, because `_has_binary_logprob_keys` still compares against
the hard-coded `{"yes", "no"}` set and raises `MeasurementError`.

- [ ] **Step 3: Implement the minimal refactor.**

Change only the repeated label sources in `measurement.py`:

```python
def _validate_logprobs(record: Mapping[str, object]) -> dict[str, float]:
    logprobs = record["logprobs"]
    labels = candidate_labels()
    if not _has_binary_logprob_keys(logprobs):
        raise MeasurementError("logprobs must contain yes and no")
    if not _has_finite_binary_logprobs(logprobs):
        raise MeasurementError("logprobs must be finite")
    return {label: float(logprobs[label]) for label in labels}


def _has_binary_logprob_keys(
    value: object,
) -> TypeGuard[Mapping[str, object]]:
    return isinstance(value, Mapping) and set(value) == set(candidate_labels())


def _has_finite_binary_logprobs(
    logprobs: Mapping[str, object],
) -> TypeGuard[Mapping[str, int | float]]:
    return all(_is_finite_number(logprobs[label]) for label in candidate_labels())


def _validate_decision(
    record: Mapping[str, object], scores: Mapping[str, float]
) -> None:
    if "decision" not in record:
        return
    decision = record["decision"]
    if not isinstance(decision, str) or decision not in candidate_labels():
        raise MeasurementError("decision must be yes or no")
    if decision != choose_decision(scores):
        raise MeasurementError("decision is inconsistent with logprobs")
```

- [ ] **Step 4: Run the focused test and unit suite to verify green.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_measurement.py::test_measurement_validation_uses_shared_candidate_labels tests/unit/test_benchmark_measurement.py -q
```

Expected: the new test and the complete measurement unit module pass with no
warnings or failures.

- [ ] **Step 5: Run focused static checks and inspect the exact diff.**

Run:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
```

Expected: all commands exit 0; only the planned measurement source and unit
test files are modified.

- [ ] **Step 6: Run the complete quality gate.**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Expected: all unit, integration, and acceptance tests pass; coverage remains at
least 99%; import-linter reports 0 broken contracts; CRAP remains below 6 for
every relevant function; mutation testing reports no `survived` or
`suspicious` entries. Record the exact counts from the command output.

- [ ] **Step 7: Request an independent review of the final diff.**

Compare the implementation commit with its parent and ask the reviewer to
check label-contract reuse, error/output compatibility, test quality, and scope
creep. Fix any Critical or Important finding, then rerun the relevant gates.

- [ ] **Step 8: Commit and publish the validated change.**

Stage only the planned source and test files and commit with:

```bash
git add src/compute_cost_encoders_llms/benchmark/measurement.py tests/unit/test_benchmark_measurement.py
git commit -m "refactor: reuse canonical measurement labels"
git push origin main
```

After pushing, verify `git status --short --branch`, `git rev-parse HEAD`,
`git rev-parse origin/main`, and `git ls-remote origin refs/heads/main`; all
three refs must match and the worktree must be clean.
