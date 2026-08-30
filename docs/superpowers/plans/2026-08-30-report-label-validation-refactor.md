# Report Label Validation Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make report-artifact validation consume the canonical binary-label definition without changing public behavior, serialized output, or existing helper callers.

**Architecture:** Keep `scripts/render_report.py` as the report-validation boundary. Each validation path captures `candidate_labels()` once and passes that tuple through the existing private helpers; the helpers retain their current two-argument call shape by making the captured labels optional. The canonical labels remain `("yes", "no")`, so all production errors, report text, and artifact schemas remain unchanged.

**Tech Stack:** Python 3.12, pytest, uv, Ruff, ty, import-linter, crap4py, and mutmut.

---

### Task 1: Add canonical-label regression coverage

**Files:**
- Modify: `tests/unit/test_render_report.py:529`
- Test: `tests/unit/test_render_report.py`

- [ ] **Step 1: Write the failing test**

Add this test after `test_validate_mean_logprobs_uses_shared_candidate_labels`:

```python
def test_summary_validation_uses_shared_candidate_labels(monkeypatch) -> None:
    labels = ("positive", "negative")
    monkeypatch.setattr(report_module, "candidate_labels", lambda: labels)
    model = {
        "model": "encoder",
        "latency": {
            "count": 1,
            "minimum": 1.0,
            "median": 1.0,
            "p05": 1.0,
            "p95": 1.0,
            "maximum": 1.0,
            "mean": 1.0,
            "stdev": 0.0,
        },
        "mean_logprobs": {"positive": -0.1, "negative": -2.2},
        "decision_counts": {"positive": 1, "negative": 0},
    }

    assert report_module._validated_summary_models([model], "encoder") == [model]
```

- [ ] **Step 2: Run the test to verify it fails for the intended reason**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_render_report.py::test_summary_validation_uses_shared_candidate_labels -q
```

Expected result before the implementation change: `FAIL` with `ValueError: mean_logprobs must contain yes and no`, because `_require_binary_keys` still owns a duplicated literal label set.

### Task 2: Thread one canonical label tuple through report validation

**Files:**
- Modify: `scripts/render_report.py:110-137`
- Test: `tests/unit/test_render_report.py`

- [ ] **Step 1: Capture labels once when validating mean scores**

Replace `_validate_mean_logprobs` with:

```python
def _validate_mean_logprobs(model: Mapping[str, object]) -> None:
    scores = _mapping_value(model, "mean_logprobs")
    labels = candidate_labels()
    _require_binary_keys(scores, "mean_logprobs", labels=labels)
    for label in labels:
        if not _is_finite_number(scores[label]):
            raise ValueError("mean_logprobs must be finite")
```

- [ ] **Step 2: Capture labels once when validating decision counts**

Replace `_validate_decision_counts` with:

```python
def _validate_decision_counts(model: Mapping[str, object]) -> None:
    counts = _mapping_value(model, "decision_counts")
    labels = candidate_labels()
    values = _validated_decision_counts(counts, labels=labels)
    latency = _mapping_value(model, "latency")
    if sum(values) != latency["count"]:
        raise ValueError("decision_counts must sum to latency count")
```

- [ ] **Step 3: Make binary-key validation use the canonical tuple while preserving direct callers**

Replace the two helper functions with:

```python
def _require_binary_keys(
    values: Mapping[str, object],
    field: str,
    *,
    labels: tuple[str, str] | None = None,
) -> None:
    expected = candidate_labels() if labels is None else labels
    if set(values) != set(expected):
        raise ValueError(f"{field} must contain yes and no")


def _validated_decision_counts(
    counts: Mapping[str, object],
    *,
    labels: tuple[str, str] | None = None,
) -> tuple[int, int]:
    expected = candidate_labels() if labels is None else labels
    _require_binary_keys(counts, "decision_counts", labels=expected)
    return (
        _non_negative_count(counts[expected[0]]),
        _non_negative_count(counts[expected[1]]),
    )
```

The optional keyword keeps existing direct calls valid while allowing each higher-level validator to use one stable label snapshot for both key validation and value access.

- [ ] **Step 4: Run focused tests to verify the green state**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_render_report.py::test_summary_validation_uses_shared_candidate_labels tests/unit/test_render_report.py -q
```

Expected result: all tests in `test_render_report.py` pass, including the new canonical-label contract.

### Task 3: Run complete verification and publish the reviewed change

**Files:**
- Verify: `scripts/qa.sh`
- Verify: `src/compute_cost_encoders_llms/benchmark/measurement.py`
- Verify: `scripts/render_report.py`
- Verify: `tests/unit/test_render_report.py`

- [ ] **Step 1: Run static and formatting checks**

Run:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
```

Expected result: each command exits with status 0.

- [ ] **Step 2: Run the complete repository quality gate**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Expected result: unit, integration, and acceptance tests pass; import-linter reports no broken contracts; CRAP remains below 6; and mutmut reports no surviving or suspicious mutants.

- [ ] **Step 3: Review the exact diff and commit the scoped implementation**

Run:

```bash
git diff --check
git diff -- scripts/render_report.py tests/unit/test_render_report.py
git add scripts/render_report.py tests/unit/test_render_report.py
git commit -m "refactor: reuse canonical report labels"
```

The commit must contain only the report validator and its regression test; the existing error strings and report outputs must remain byte-for-byte unchanged for canonical `yes`/`no` artifacts.

- [ ] **Step 4: Request independent review and push after a clean result**

Compare the implementation commit with its parent and request a read-only review for behavior preservation, helper compatibility, TDD coverage, and scope. After Critical and Important findings are absent, run the full test suite once more, then run:

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

Expected result: the worktree is clean and the local, tracking, and remote `main` revisions are identical.
