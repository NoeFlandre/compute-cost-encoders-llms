# Report Script Import Boundary Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the report script's stale private mapping alias and make its LaTeX dependency point directly to the module that owns rendering, without changing behavior or compatibility.

**Architecture:** `scripts/render_report.py` will retain its artifact-field helpers from `scripts._artifact_fields` and import `render_latex_document` from `benchmark.latex`. `benchmark.reporting` will continue to re-export both LaTeX functions for existing callers; no public reporting surface changes.

**Tech Stack:** Python 3.12, pytest, Ruff, ty, import-linter, crap4py, mutmut, uv.

---

## Files

- Create: `tests/unit/test_report_script_import_boundary.py` — structural tests for the script's owned dependencies.
- Modify: `scripts/render_report.py` — remove the unused mapping alias and import the LaTeX implementation from its owner.
- Modify: `tests/unit/test_mapping_contract.py` — test `_mapping_field` at its owning module instead of through a stale script alias.
- Create: `docs/superpowers/specs/2026-08-30-report-script-import-boundary-design.md` — committed design contract.

### Task 1: Add the report import boundary test

**Files:**
- Create: `tests/unit/test_report_script_import_boundary.py`

- [ ] **Step 1: Write the failing tests**

```python
from __future__ import annotations

from pathlib import Path

import scripts.render_report as report_module


def test_report_script_does_not_expose_unrelated_mapping_helper() -> None:
    assert not hasattr(report_module, "_mapping_field")


def test_report_script_imports_latex_from_the_owning_module() -> None:
    source = Path(report_module.__file__).read_text(encoding="utf-8")

    assert (
        "from compute_cost_encoders_llms.benchmark.latex import "
        "render_latex_document"
    ) in source
    assert (
        "from compute_cost_encoders_llms.benchmark.reporting import "
        "render_latex_document"
    ) not in source
```

- [ ] **Step 2: Run the focused test to verify the red state**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_report_script_import_boundary.py -q
```

Expected: failure because the existing script still exposes `_mapping_field` and
imports rendering through `benchmark.reporting`.

### Task 2: Apply the minimal import refactor

**Files:**
- Modify: `scripts/render_report.py:14-19`

- [ ] **Step 1: Remove the stale import and point rendering at its owner**

Change the imports to:

```python
from compute_cost_encoders_llms.benchmark._numerics import _is_finite_number
from compute_cost_encoders_llms.benchmark.example import candidate_labels
from compute_cost_encoders_llms.benchmark.latex import render_latex_document
from scripts._artifact_fields import _as_mapping, _mapping_value, _text_value
from scripts.grid5000.checkpoint_metadata import build_checkpoint_metadata
```

Do not change any function body, CLI argument, output path, artifact key, error
message, or compatibility export in `benchmark.reporting`.

- [ ] **Step 2: Run the focused boundary test to verify green**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_report_script_import_boundary.py -q
```

Expected: both tests pass.

### Task 3: Move mapping-helper assertions to the owning module

**Files:**
- Modify: `tests/unit/test_mapping_contract.py`

- [ ] **Step 1: Remove the test's dependency on `scripts.render_report._mapping_field`**

Use the owning import directly:

```python
from __future__ import annotations

import pytest

from compute_cost_encoders_llms.benchmark._mappings import _mapping_field


def test_mapping_field_supports_optional_and_required_modes() -> None:
    assert _mapping_field({"nested": {"value": 1}}, "nested") == {"value": 1}
    assert _mapping_field({}, "nested") == {}
    assert _mapping_field({"nested": None}, "nested") == {}

    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: nested$",
    ):
        _mapping_field(
            {"nested": None},
            "nested",
            required=True,
            error_context="merged artifact",
        )


def test_mapping_field_uses_document_as_default_error_context() -> None:
    with pytest.raises(ValueError, match=r"^document field is not an object: value$"):
        _mapping_field({}, "value", required=True)
```

- [ ] **Step 2: Run the focused regression suite**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_report_script_import_boundary.py tests/unit/test_mapping_contract.py tests/unit/test_latex_boundary.py tests/unit/test_reporting_document_contract.py tests/unit/test_render_report.py tests/unit/test_grid5000_checkpoint.py -q
```

Expected: all collected tests pass, including the existing reporting re-export
contract and exact report/checkpoint behavior tests.

### Task 4: Run static checks and inspect the diff

**Files:** all changed files.

- [ ] **Step 1: Run formatting, linting, typing, and whitespace checks**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
git diff --check
```

Expected: every command exits successfully with no formatting, lint, typing, or
whitespace errors.

- [ ] **Step 2: Inspect the staged scope before commit**

```bash
git diff -- scripts/render_report.py tests/unit/test_mapping_contract.py tests/unit/test_report_script_import_boundary.py
```

Expected: only import ownership and its tests changed; `reporting.py` remains
untouched.

### Task 5: Run the complete quality gauntlet

**Files:** no additional changes unless a gate identifies a real regression.

- [ ] **Step 1: Run the full project QA script**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Expected: Ruff, formatting, ty, unit/integration/acceptance coverage, import
linting, CRAP below 6, and mutation testing all pass with no surviving or
suspicious mutants. The known CLI parser/main no-test mutations may remain
classified as `no tests`, as on the established baseline.

### Task 6: Commit, push, and verify

**Files:** the implementation and tests above.

- [ ] **Step 1: Commit the validated implementation**

```bash
git add scripts/render_report.py tests/unit/test_mapping_contract.py tests/unit/test_report_script_import_boundary.py
git commit -m "refactor: align report script imports"
```

- [ ] **Step 2: Run a fresh post-commit regression suite**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit tests/integration tests/acceptance -q
```

Expected: every collected test passes.

- [ ] **Step 3: Push and verify the current branch**

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

Expected: push succeeds, the worktree is clean, and all three SHA values match.
