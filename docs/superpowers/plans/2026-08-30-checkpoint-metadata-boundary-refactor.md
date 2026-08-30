# Checkpoint Metadata Boundary Refactor Implementation Plan

> **For agentic workers:** Execute this plan task by task, maintaining the
> red → green → refactor cycle and stopping only for a real external blocker.

**Goal:** Move checkpoint-metadata construction out of the report script while
preserving all existing report imports, outputs, validation errors, and JSON
artifacts.

**Architecture:** `scripts/grid5000/checkpoint_metadata.py` will own the full
checkpoint metadata contract: pure construction plus publication validation.
`scripts/render_report.py` will retain JSON loading, backend validation and
merging, LaTeX rendering, and CLI orchestration. It will import the existing
`build_checkpoint_metadata` name as a compatibility facade.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, import-linter, crap4py,
mutmut.

---

### Task 1: Establish the canonical-owner boundary with a red test

**Files:**
- Create: `tests/unit/test_checkpoint_metadata_boundary.py`
- Read: `scripts/render_report.py`
- Read: `scripts/grid5000/checkpoint_metadata.py`

- [x] **Step 1: Write the failing test**

Add a focused test that imports both modules and asserts the public builder
facade is the exact callable owned by the Grid5000 checkpoint module:

```python
import scripts.grid5000.checkpoint_metadata as checkpoint_module
import scripts.render_report as report_module


def test_report_builder_is_owned_by_checkpoint_module() -> None:
    assert (
        report_module.build_checkpoint_metadata
        is checkpoint_module.build_checkpoint_metadata
    )
```

- [x] **Step 2: Run the focused test and verify red**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_checkpoint_metadata_boundary.py -q
```

Expected: collection or assertion failure because the current implementation
still lives in `scripts/render_report.py`. Do not add production code before
capturing this red result.

### Task 2: Move the pure checkpoint contract to its canonical module

**Files:**
- Modify: `scripts/grid5000/checkpoint_metadata.py`
- Modify: `scripts/render_report.py`

- [x] **Step 1: Move the implementation without behavior edits**

Move `PROJECT_BUCKET_URI`, `build_checkpoint_metadata`, `_mapping_value`,
`_text_value`, `_integer_value`, `_checkpoint_metrics`, and `_as_mapping` into
`scripts/grid5000/checkpoint_metadata.py`. Derive the URI from the existing
`PROJECT_BUCKET_PREFIX` with `rstrip("/")` so the canonical bucket string is
defined once.

Preserve function signatures, validation order, error text, returned key
values, metric ordering, and JSON-compatible value types exactly. Keep the
builder free of file I/O and environment access.

- [x] **Step 2: Add the report compatibility facade**

Import `build_checkpoint_metadata` and the helper aliases still needed by
report validation into `scripts/render_report.py`. Remove the old definitions
and any imports that are unused after the move. Do not change the CLI or the
call to the builder in `render_report`.

- [x] **Step 3: Run focused tests to reach green**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_checkpoint_metadata_boundary.py tests/unit/test_render_report.py tests/unit/test_grid5000_checkpoint.py -q
```

Expected: the new boundary test, existing report behavior tests, and existing
checkpoint validator tests all pass.

### Task 3: Refactor tests to document ownership after green

**Files:**
- Modify: `tests/unit/test_render_report.py`
- Modify: `tests/unit/test_checkpoint_metadata_boundary.py`

- [x] **Step 1: Move builder-helper imports to the canonical owner**

Import `_as_mapping` and `_checkpoint_metrics` from
`scripts.grid5000.checkpoint_metadata` in the tests that exercise those
private helpers. Keep `build_checkpoint_metadata` imported from
`scripts.render_report` in the compatibility tests so the old public import
path remains explicitly covered.

- [x] **Step 2: Strengthen the boundary contract**

Keep the identity assertion and add a direct canonical-module import check if
needed for readability. Do not duplicate behavior fixtures; the existing
exact metadata and failure-path tests remain the regression oracle.

- [x] **Step 3: Run focused static and behavioral checks**

Run:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_checkpoint_metadata_boundary.py tests/unit/test_render_report.py tests/unit/test_grid5000_checkpoint.py -q
```

Expected: no formatting, lint, type, or focused test failures.

### Task 4: Run the complete quality and mutation verification

**Files:**
- Modify none unless a gate identifies a direct regression in this refactor.

- [x] **Step 1: Run the complete repository gate**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

This must pass Ruff, formatting, ty, all unit/integration/acceptance tests,
import-linter, CRAP, and mutation testing. The added boundary test increases
the unit-test count by one; no existing test may be removed or weakened.

- [x] **Step 2: Inspect mutation categories explicitly**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run mutmut results
```

There must be no `survived` or `suspicious` mutants. If mutation cache state
does not cover the moved source, rebuild the mutation workspace and rerun the
same full campaign rather than reporting stale results.

- [x] **Step 3: Verify the final diff and worktree**

Run:

```bash
git diff --check
git status --short --branch
```

Confirm only the design, plan, canonical checkpoint module, report facade, and
focused boundary/test ownership changes are present.

### Task 5: Commit and push the validated refactor

**Files:**
- `docs/superpowers/specs/2026-08-30-checkpoint-metadata-boundary-design.md`
- `docs/superpowers/plans/2026-08-30-checkpoint-metadata-boundary-refactor.md`
- `scripts/grid5000/checkpoint_metadata.py`
- `scripts/render_report.py`
- `tests/unit/test_checkpoint_metadata_boundary.py`
- `tests/unit/test_render_report.py`

- [x] **Step 1: Stage only the approved paths**

Run:

```bash
git add docs/superpowers/specs/2026-08-30-checkpoint-metadata-boundary-design.md docs/superpowers/plans/2026-08-30-checkpoint-metadata-boundary-refactor.md scripts/grid5000/checkpoint_metadata.py scripts/render_report.py tests/unit/test_checkpoint_metadata_boundary.py tests/unit/test_render_report.py
git diff --cached --check
```

- [x] **Step 2: Commit with a focused Conventional Commit message**

Run:

```bash
git commit -m "refactor: isolate checkpoint metadata construction"
```

- [x] **Step 3: Push and verify remote synchronization**

Run:

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

The local branch must be clean and `HEAD`, `origin/main`, and the remote
advertised commit must agree.
