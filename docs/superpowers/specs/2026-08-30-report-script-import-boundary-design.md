# Report Script Import Boundary Design

**Date:** 2026-08-30

**Goal:** Remove stale private coupling from the report-rendering script without
changing its CLI, output, exceptions, or supported import paths.

## Context

`scripts/render_report.py` still imports
`compute_cost_encoders_llms.benchmark._mappings._mapping_field` only to expose a
private module attribute. The script no longer calls that helper: its artifact
field access goes through `scripts._artifact_fields`. The unused import makes a
generic package helper look like part of the script's interface and leaves a
misleading dependency behind.

The script also imports `render_latex_document` from `benchmark.reporting`,
although `benchmark.latex` owns the implementation. `reporting.py` must continue
to re-export that function because existing callers use the historical import
path, but the report script should depend on the owning presentation module
directly.

## Design

1. Delete the unused `_mapping_field` import from `scripts/render_report.py`.
2. Import `render_latex_document` directly from
   `compute_cost_encoders_llms.benchmark.latex`.
3. Keep `reporting.render_latex_document` and
   `reporting.render_latex_summary` unchanged as compatibility aliases.
4. Update the boundary tests to assert that the report script no longer exposes
   the unrelated private mapping helper and that its rendering import points to
   the LaTeX owner. Keep the existing mapping helper behavior tests at the
   helper's owning module.

## Compatibility and error behavior

The change affects imports only. `render_report`, `merge_artifacts`, `main`, all
CLI options, all artifact keys, all error messages, and all rendered LaTeX bytes
remain unchanged. The historical `benchmark.reporting` rendering imports remain
available and are covered by the existing reporting contract tests.

## Verification

The focused boundary tests will be written first and must fail because the old
script still exposes `_mapping_field`. After the minimal import change they must
pass along with the focused report, mapping, LaTeX, and checkpoint tests. The
complete `scripts/qa.sh` gauntlet must then pass, including coverage, import
linting, CRAP, and mutation testing, followed by a fresh post-commit test run.
