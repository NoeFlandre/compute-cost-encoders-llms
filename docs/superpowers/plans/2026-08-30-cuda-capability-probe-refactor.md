# CUDA Capability Probe Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove duplicated CUDA capability probing from `_fp16_supported`
without changing precision selection, runtime metadata, or failure behavior.

**Architecture:** Keep `_device_capability` as the single guarded capability
adapter. Make `_fp16_supported` consume its `list[int] | None` result and
compare a tuple only for the threshold decision.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, import-linter, crap4py,
mutmut.

---

### Task 1: Add the capability-reuse contract

**Files:**

- Modify: `tests/unit/test_benchmark_cli.py`
- Read: `src/compute_cost_encoders_llms/benchmark/runtime.py`

- [ ] **Step 1: Write the failing test**

Add this test beside the existing CUDA precision tests:

```python
def test_fp16_support_reuses_normalized_device_capability(monkeypatch) -> None:
    cuda = cast(CudaApi, object())
    calls: list[CudaApi] = []

    def capability(value: CudaApi) -> list[int] | None:
        calls.append(value)
        return [5, 3]

    monkeypatch.setattr(runtime_module, "_device_capability", capability)

    assert runtime_module._fp16_supported(cuda) is True
    assert calls == [cuda]
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_cli.py::test_fp16_support_reuses_normalized_device_capability -q
```

Expected: FAIL because the current `_fp16_supported` probes the CUDA object
directly and never calls `_device_capability`.

### Task 2: Reuse the normalized capability adapter

**Files:**

- Modify: `src/compute_cost_encoders_llms/benchmark/runtime.py:82-96`

- [ ] **Step 1: Implement the minimal refactor**

Replace the duplicated getter lookup, call, conversion, and exception block in
`_fp16_supported` with:

```python
def _fp16_supported(cuda: CudaApi) -> bool:
    capability = _device_capability(cuda)
    return capability is not None and tuple(capability) >= (5, 3)
```

Do not alter `_device_capability` or the precision-selection branches.

- [ ] **Step 2: Run the focused runtime tests**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_cli.py -q
```

Expected: all runtime and CLI unit tests pass, including the new reuse test,
the `(5, 3)` boundary, lower capabilities, missing getters, and failing
getters.

### Task 3: Refactor verification

- [ ] **Step 1: Run focused static checks**

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
```

Expected: no whitespace, format, lint, or type errors.

### Task 4: Full quality gates

- [ ] **Step 1: Run the repository QA script**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Require all unit, integration, and acceptance tests; 99% coverage; clean
import-linter output; CRAP below 6; and no surviving or suspicious mutants.
Known no-test mutations remain a separately reported baseline category.

- [ ] **Step 2: Inspect mutation results explicitly**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run mutmut results
```

Confirm that no surviving, suspicious, or timed-out mutants are reported.

### Task 5: Review, commit, push, and post-push verification

- [ ] **Step 1: Review the exact scope**

```bash
git status --short --branch
git diff --check
git diff --stat
```

Only the design, plan, runtime implementation, and focused test paths may be
staged.

- [ ] **Step 2: Commit the validated change**

```bash
git add docs/superpowers/specs/2026-08-30-cuda-capability-probe-design.md docs/superpowers/plans/2026-08-30-cuda-capability-probe-refactor.md src/compute_cost_encoders_llms/benchmark/runtime.py tests/unit/test_benchmark_cli.py
git commit -m "refactor: reuse CUDA capability probe"
```

- [ ] **Step 3: Run a fresh post-commit suite**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit tests/integration tests/acceptance -q
```

Expected: all tests pass after the commit.

- [ ] **Step 4: Push and verify all refs**

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

Require a clean worktree and matching local, tracking, and remote `main`
commit IDs.
