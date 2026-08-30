
# Artifact Field Boundary Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Give shared merged-artifact field validation a neutral owner while preserving the existing private helper paths, errors, report output, and checkpoint behavior.

**Architecture:** Add scripts/_artifact_fields.py for _mapping_value, _text_value, and _as_mapping. scripts.render_report and scripts.grid5000.checkpoint_metadata will import those exact callables from the neutral module; their current module-level names remain compatibility aliases. The existing benchmark _mapping_field utility remains the single implementation of required/optional mapping extraction.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, import-linter, crap4py, mutmut.

---

### Task 1: Establish the neutral ownership boundary with a red test

**Files:**
- Create: tests/unit/test_artifact_field_boundary.py
- Read: scripts/render_report.py
- Read: scripts/grid5000/checkpoint_metadata.py

- [ ] **Step 1: Write the failing boundary test**

Create the test before adding the neutral module:

~~~python
from __future__ import annotations

import scripts._artifact_fields as fields_module
import scripts.grid5000.checkpoint_metadata as checkpoint_module
import scripts.render_report as report_module


def test_shared_artifact_helpers_are_owned_by_neutral_module() -> None:
    for name in ("_as_mapping", "_mapping_value", "_text_value"):
        assert getattr(report_module, name) is getattr(fields_module, name)
        assert getattr(checkpoint_module, name) is getattr(fields_module, name)
~~~

- [ ] **Step 2: Run the test to verify the expected red**

Run:

~~~bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_artifact_field_boundary.py -q
~~~

Expected result: collection fails with ModuleNotFoundError because
scripts._artifact_fields does not exist yet. Do not add production code until
this failure is observed.

### Task 2: Add the neutral helper module and preserve both import paths

**Files:**
- Create: scripts/_artifact_fields.py
- Modify: scripts/grid5000/checkpoint_metadata.py
- Modify: scripts/render_report.py

- [ ] **Step 1: Add the minimal canonical helper implementations**

Create scripts/_artifact_fields.py with the existing behavior and error
messages:

~~~python
from __future__ import annotations

from collections.abc import Mapping

from compute_cost_encoders_llms.benchmark._mappings import _mapping_field


def _mapping_value(document: Mapping[str, object], field: str) -> Mapping[str, object]:
    return _mapping_field(
        document,
        field,
        required=True,
        error_context="merged artifact",
    )


def _text_value(document: Mapping[str, object], field: str) -> str:
    value = document.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"merged artifact field is not text: {field}")
    return value


def _as_mapping(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"merged artifact field is not an object: {field}")
    return value
~~~

- [ ] **Step 2: Replace the checkpoint helper definitions with aliases**

In scripts/grid5000/checkpoint_metadata.py, replace the local helper definitions
and add:

~~~python
from scripts._artifact_fields import _as_mapping, _mapping_value, _text_value
~~~

Keep the existing Mapping import because the checkpoint builder and validator
still use it in their annotations. Delete only the local implementations of
_mapping_value, _text_value, and _as_mapping. Leave all builder and
metadata-validator callers unchanged.

- [ ] **Step 3: Replace the report helper imports with neutral aliases**

In scripts/render_report.py, keep the existing benchmark _mapping_field
compatibility import and replace the checkpoint helper imports with:

~~~python
from scripts._artifact_fields import _as_mapping, _mapping_value, _text_value
from scripts.grid5000.checkpoint_metadata import build_checkpoint_metadata
~~~

Do not change the report validation functions, the
build_checkpoint_metadata compatibility name, the CLI parser, or output
writing.

- [ ] **Step 4: Run focused tests to reach green**

Run:

~~~bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_artifact_field_boundary.py tests/unit/test_mapping_contract.py tests/unit/test_render_report.py tests/unit/test_grid5000_checkpoint.py -q
~~~

Expected result: the new identity test and all existing mapping, report, and
checkpoint tests pass with unchanged error text and artifacts.

### Task 3: Make canonical test ownership explicit after green

**Files:**
- Modify: tests/unit/test_render_report.py
- Modify: tests/unit/test_artifact_field_boundary.py

- [ ] **Step 1: Import direct helper behavior from the neutral owner**

In tests/unit/test_render_report.py, replace the checkpoint-module _as_mapping
import with:

~~~python
from scripts._artifact_fields import _as_mapping
from scripts.grid5000.checkpoint_metadata import (
    _checkpoint_metrics,
    build_checkpoint_metadata,
)
~~~

Keep the existing _as_mapping failure assertion and all checkpoint-builder
compatibility tests intact. The boundary test remains responsible for proving
both historical module paths are exact aliases to the neutral owner.

- [ ] **Step 2: Add direct neutral-helper behavior checks**

Extend test_artifact_field_boundary.py with these focused assertions:

~~~python
import pytest

from scripts._artifact_fields import _as_mapping, _mapping_value, _text_value


def test_artifact_helpers_preserve_merged_artifact_contract() -> None:
    document = {"nested": {"value": 1}, "name": "encoder"}
    assert _mapping_value(document, "nested") == {"value": 1}
    assert _text_value(document, "name") == "encoder"
    assert _as_mapping(document["nested"], "nested") == {"value": 1}

    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: missing$",
    ):
        _mapping_value(document, "missing")
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not text: name$",
    ):
        _text_value({"name": ""}, "name")
    with pytest.raises(
        ValueError,
        match=r"^merged artifact field is not an object: nested$",
    ):
        _as_mapping(None, "nested")
~~~

- [ ] **Step 3: Run focused static and behavioral checks**

Run each command separately:

~~~bash
git diff --check
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ruff check .
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run ty check src tests scripts
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_artifact_field_boundary.py tests/unit/test_mapping_contract.py tests/unit/test_render_report.py tests/unit/test_grid5000_checkpoint.py -q
~~~

Expected result: no diff, formatting, lint, type, collection, or focused
behavior failures.

### Task 4: Run complete quality and mutation verification

**Files:**
- Modify none unless a gate identifies a direct regression in this refactor.

- [ ] **Step 1: Run the complete repository gate**

Run:

~~~bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
~~~

Require Ruff, formatting, ty, unit, integration, acceptance, import-linter,
CRAP, and mutation stages to pass. The baseline is 168 unit tests, 1
integration test, 6 acceptance tests, 99% coverage, CRAP maximum 5.0, and no
surviving or suspicious mutants. The two boundary checks increase the unit
count to 170 without weakening or removing existing tests.

- [ ] **Step 2: Inspect mutation categories explicitly**

Run:

~~~bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run mutmut results
~~~

Require no survived, suspicious, or timeout entries. If the mutation workspace
retains stale statuses for moved helpers, run the affected mutant names
explicitly after confirming mutmut tests-for-mutant maps them to the neutral
tests; never report stale results.

- [ ] **Step 3: Review the final pre-commit scope**

Run:

~~~bash
git diff --check
git status --short --branch
git diff --stat
~~~

Confirm only the design and plan documents, the neutral helper module, two
consumer import changes, the boundary test, and the direct test-owner change
are present. Do not stage coverage or mutation artifacts.

### Task 5: Commit, push, and verify synchronization

**Files:**
- docs/superpowers/specs/2026-08-30-artifact-field-boundary-design.md
- docs/superpowers/plans/2026-08-30-artifact-field-boundary-refactor.md
- scripts/_artifact_fields.py
- scripts/grid5000/checkpoint_metadata.py
- scripts/render_report.py
- tests/unit/test_artifact_field_boundary.py
- tests/unit/test_render_report.py

- [ ] **Step 1: Stage only the approved paths**

Run:

~~~bash
git add docs/superpowers/specs/2026-08-30-artifact-field-boundary-design.md docs/superpowers/plans/2026-08-30-artifact-field-boundary-refactor.md scripts/_artifact_fields.py scripts/grid5000/checkpoint_metadata.py scripts/render_report.py tests/unit/test_artifact_field_boundary.py tests/unit/test_render_report.py
git diff --cached --check
~~~

- [ ] **Step 2: Commit with a focused Conventional Commit message**

Run:

~~~bash
git commit -m "refactor: decouple artifact field validation"
~~~

- [ ] **Step 3: Run the post-commit regression suite**

Run:

~~~bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit tests/integration tests/acceptance -q
~~~

Require all 177 tests to pass after the commit, not only before it.

- [ ] **Step 4: Push and verify local/tracking/remote state**

Run each command separately:

~~~bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
~~~

The local branch must be clean and HEAD, origin/main, and the remote advertised
commit must agree.
