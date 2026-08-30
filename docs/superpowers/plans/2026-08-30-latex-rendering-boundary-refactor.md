# LaTeX Rendering Boundary Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move the pure LaTeX presentation layer out of `reporting.py` while preserving every existing reporting import, rendered string, error, and artifact contract.

**Architecture:** `benchmark/reporting.py` will retain measurement validation, grouping, summary construction, deterministic JSON/JSONL writing, and artifact writing. A new `benchmark/latex.py` will own LaTeX escaping, formatting, and document assembly; `reporting.py` will re-export only the two existing public renderer functions so callers do not change.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, import-linter, crap4py, mutmut.

---

### Task 1: Establish the module-boundary contract with a red test

**Files:**
- Create: `tests/unit/test_latex_boundary.py`
- Read: `src/compute_cost_encoders_llms/benchmark/reporting.py`

- [ ] **Step 1: Write the failing test**

Create a test that imports the current reporting facade and the intended
dedicated renderer module, then asserts the two existing renderer names are
the exact callables provided by the new module:

```python
import compute_cost_encoders_llms.benchmark.latex as latex_module
import compute_cost_encoders_llms.benchmark.reporting as reporting_module


def test_reporting_reexports_latex_renderers() -> None:
    assert reporting_module.render_latex_document is latex_module.render_latex_document
    assert reporting_module.render_latex_summary is latex_module.render_latex_summary
```

- [ ] **Step 2: Run the focused test and verify the expected red state**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_latex_boundary.py -q
```

Expected: collection fails with `ModuleNotFoundError` for
`compute_cost_encoders_llms.benchmark.latex`, proving the new boundary is not
already present and the test is meaningful.

### Task 2: Create the dedicated LaTeX module with the minimum move

**Files:**
- Create: `src/compute_cost_encoders_llms/benchmark/latex.py`
- Modify: `src/compute_cost_encoders_llms/benchmark/reporting.py`

- [ ] **Step 1: Add the new module imports and moved implementation**

Create `latex.py` with this exact import boundary:

```python
from __future__ import annotations

import itertools
from collections.abc import Mapping

from ._mappings import _mapping_field
from ._numerics import _is_finite_number
from .example import candidate_labels
```

Move the existing implementation block beginning at
`reporting.py::_latex_escape` and ending at `reporting.py::_revision` into the
new module without changing any function body or output expression. The
moved definitions, in their existing order, are:

```text
_latex_escape
render_latex_summary
render_latex_document
_model_summaries
_number_text
_count_text
_decision_text
_timing_section
_timing_text
_runtime_section
_backend_runtime
_runtime_line
_hardware_text
_runtime_package_details
_runtime_cuda_details
_model_id
_comparison_section
_comparison_values
_comparison_lines
_number_value
_decision_counts_text
_margin_text
_reproducibility_section
_revision
```

Replace the moved `ModelSummary` annotations with
`Mapping[str, object]` annotations in `latex.py`; the runtime values are
already mappings, and this avoids importing the summary-builder module back
into the presentation module. Keep `from __future__ import annotations` so
this type-only refinement has no runtime effect.

- [ ] **Step 2: Add the compatibility re-exports**

Add this import to `reporting.py` alongside its existing relative imports:

```python
from .latex import render_latex_document, render_latex_summary
```

Do not add a reverse import from `latex.py` to `reporting.py`. Keep
`ModelSummary`, `SummaryDocument`, all measurement helpers, JSON writers, and
`write_measurement_artifacts` in `reporting.py`.

- [ ] **Step 3: Run the boundary and exact-output tests for green**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_latex_boundary.py tests/unit/test_benchmark_measurement.py tests/unit/test_reporting_document_contract.py tests/unit/test_render_report.py -q
```

Expected: the new boundary test and all existing LaTeX/document contract
tests pass with unchanged rendered strings and errors.

### Task 3: Refactor the source boundary after green

**Files:**
- Modify: `src/compute_cost_encoders_llms/benchmark/reporting.py`
- Modify: `tests/unit/test_reporting_document_contract.py`

- [ ] **Step 1: Remove the old implementation block and unused imports**

After Task 2 is green, delete only the moved definitions from
`reporting.py`. Remove any import that is unused after the deletion; retain
the reporting imports needed by summary construction and artifact writing.
Update the private `_number_text` contract test to import the helper from
`benchmark.latex` because it now tests the module that owns that helper. Do
not change tests that import the public `render_latex_document` or
`render_latex_summary` from `reporting`.

- [ ] **Step 2: Run the focused suite again**

Run the Task 2 command again. Expected: all focused tests pass, including the
identity re-export test and byte/string-level formatting assertions.

- [ ] **Step 3: Run a diff and static sanity check**

Run:

```bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
```

Expected: no whitespace errors, formatting changes, lint errors, or type
errors.

### Task 4: Run the complete quality gates and mutation campaign

**Files:**
- Modify none unless a verified gate failure identifies a direct regression in the refactor.

- [ ] **Step 1: Run the full deterministic project gate**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
```

Expected: Ruff, formatting, ty, 165 unit tests, 1 integration test, 6
acceptance tests, import-linter, and CRAP all pass. Mutation testing must
report `survived: 0` and `suspicious: 0`; existing `no tests` entries must be
reported separately and not misrepresented as killed mutants.

- [ ] **Step 2: Verify the mutation result categories explicitly**

Run:

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run mutmut results
```

Inspect the complete output. Any `survived` or `suspicious` result is a
blocking regression: inspect it, add a red characterization test, rerun that
test red, make the smallest fix, rerun green, and repeat the full gate.

### Task 5: Commit and publish the verified refactor

**Files:**
- Add: `docs/superpowers/plans/2026-08-30-latex-rendering-boundary-refactor.md`
- Add: `tests/unit/test_latex_boundary.py`
- Add: `src/compute_cost_encoders_llms/benchmark/latex.py`
- Modify: `src/compute_cost_encoders_llms/benchmark/reporting.py`
- Modify: `tests/unit/test_reporting_document_contract.py`

- [ ] **Step 1: Review the exact scope and stage explicit paths**

Run:

```bash
git status --short --branch
git diff --check
git add docs/superpowers/plans/2026-08-30-latex-rendering-boundary-refactor.md tests/unit/test_latex_boundary.py src/compute_cost_encoders_llms/benchmark/latex.py src/compute_cost_encoders_llms/benchmark/reporting.py tests/unit/test_reporting_document_contract.py
git diff --cached --check
```

Expected: only the plan, boundary test, new renderer module, reporting facade,
and direct test import update are staged.

- [ ] **Step 2: Commit the refactor**

Run:

```bash
git commit -m "refactor: isolate LaTeX report rendering"
```

- [ ] **Step 3: Push and verify synchronization**

Run:

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

Expected: push succeeds, the worktree is clean, and all three commit IDs
match.
