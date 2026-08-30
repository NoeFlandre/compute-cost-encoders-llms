# LaTeX Rendering Boundary Design

## Goal

Separate LaTeX presentation from measurement aggregation and artifact I/O so
each module has one clear responsibility, without changing any public
reporting functions, rendered output, errors, or JSON artifacts.

## Current Context

`src/compute_cost_encoders_llms/benchmark/reporting.py` currently owns four
related but distinct concerns:

1. validating and grouping measurement records;
2. building summary documents;
3. writing deterministic JSON and JSONL artifacts; and
4. rendering complete LaTeX reports.

The first three concerns form the measurement reporting boundary. The LaTeX
functions are a pure presentation layer with their own formatting helpers and
do not need to know how measurements are validated or persisted. Keeping both
layers in one 565-line module makes changes harder to locate and tests harder
to target, even though the current complexity and coverage gates pass.

## Options Considered

### Keep the module unchanged

This has no migration risk, but preserves the mixed-responsibility boundary
and makes future reporting changes less cohesive.

### Extract the pure LaTeX layer (recommended)

Move `render_latex_summary`, `render_latex_document`, and their private
formatting helpers to `benchmark/latex.py`. `reporting.py` will re-export the
two existing render functions from that module, so current imports and call
signatures remain valid. Existing exact-output tests continue to protect the
rendered document contract, while a boundary test proves the implementation
lives in the dedicated module.

This is a small, reversible file-boundary change. It does not introduce a new
document model, change data flow, alter validation, or add runtime behavior.

### Introduce a typed intermediate report object

This could make the renderer's input more explicit, but it would add a new
schema and conversion layer without a current consumer need. It is therefore
YAGNI for this cleanup.

## Design

Create `src/compute_cost_encoders_llms/benchmark/latex.py` containing:

- LaTeX escaping and numeric/count/decision formatting;
- the summary-table renderer;
- timing, runtime, comparison, and reproducibility sections; and
- the complete document renderer.

The new module depends only on the shared mapping/numeric/example helpers and
standard-library types. It will not import `reporting.py`, preventing a
circular dependency.

Keep measurement validation, grouping, summary construction, JSON/JSONL
serialization, and artifact writing in `reporting.py`. Import the two moved
public render functions into `reporting.py` so this existing compatibility
surface remains unchanged:

```python
from .latex import render_latex_document, render_latex_summary
```

The renderer receives the same `Mapping[str, object]` values and returns the
same strings. No callers change, and no output path or artifact ordering
changes.

## Testing and TDD Sequence

1. Add a focused boundary test that imports both modules and asserts
   `reporting.render_latex_document` and `reporting.render_latex_summary` are
   the same callables exported by `benchmark.latex`. Run it first and confirm
   it fails because `benchmark.latex` does not yet exist.
2. Add the new module and compatibility imports with the smallest possible
   move. Run the boundary test and the existing exact LaTeX/reporting tests.
3. Refactor only after green: remove the moved implementation and unused
   imports from `reporting.py`, then rerun the focused tests.
4. Run the full project quality gauntlet: Ruff, formatting, ty, unit,
   integration, acceptance, import-linter, CRAP, and mutation testing.

The existing acceptance and exact-output tests are the regression oracle for
behavior. The mutation gate must report no `survived` or `suspicious` mutants;
any `no tests` entries remain explicitly reported rather than being treated as
kills.

## Compatibility and Rollback

The only import-level change is internal module ownership. Existing imports
from `compute_cost_encoders_llms.benchmark.reporting` continue to resolve.
If any focused or full gate detects a changed output, exception, or import
contract, restore the moved functions in `reporting.py` and revert the new
module without changing callers or fixtures.
