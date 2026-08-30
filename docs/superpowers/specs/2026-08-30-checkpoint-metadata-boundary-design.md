# Checkpoint Metadata Boundary Design

## Goal

Separate checkpoint-metadata construction from report orchestration while
preserving every existing report output, validation error, file format, and
import path.

## Current Context

`scripts/render_report.py` currently owns five responsibilities:

1. reading backend JSON artifacts;
2. validating and merging backend summaries;
3. constructing publishable checkpoint metadata;
4. rendering the merged report; and
5. parsing the report CLI.

The repository already has a `scripts/grid5000/checkpoint_metadata.py` module
whose domain is the checkpoint metadata contract, but construction remains in
the report script. That split makes checkpoint changes harder to locate and
leaves the report script responsible for a concern that does not belong to
LaTeX/report orchestration.

## Options Considered

### Keep the module unchanged

This has no migration risk, but preserves the mixed responsibility and makes
the existing checkpoint contract unnecessarily dependent on report code.

### Add a separate checkpoint-artifact module

This would separate construction from validation, but it would introduce a
third module for one small metadata contract and require another home for
shared field-validation helpers. That is more structure than the current
code needs.

### Extend the existing checkpoint metadata module (recommended)

Move checkpoint metadata construction and its field/metric helpers into
`scripts/grid5000/checkpoint_metadata.py`. The module then owns the complete
checkpoint metadata contract: building it from a verified merged artifact and
validating the resulting JSON before publication. `render_report.py` keeps
artifact loading, backend validation/merging, report rendering, and CLI
orchestration.

Keep `scripts.render_report.build_checkpoint_metadata` as an import-level
compatibility facade, and retain the helper aliases needed by the report
validation code. Existing callers therefore keep the same public function,
signature, outputs, and exceptions while the implementation has one canonical
owner.

## Design

Move these definitions from `scripts/render_report.py` to
`scripts/grid5000/checkpoint_metadata.py` without changing their behavior:

- `PROJECT_BUCKET_URI` (derived from the existing project bucket prefix);
- `build_checkpoint_metadata`;
- `_mapping_value`;
- `_text_value`;
- `_integer_value`;
- `_checkpoint_metrics`; and
- `_as_mapping`.

`render_report.py` will import the builder and the helpers it still uses for
merged-artifact validation. Its public builder name remains available through
that import. The existing validator CLI and its deterministic error ordering
remain unchanged. The checkpoint module will not import `render_report.py`, so
the dependency remains one-way and acyclic.

The metadata builder remains pure: it only validates the supplied merged
mapping and returns the same JSON-compatible dictionary. No file I/O,
environment access, or report rendering is added to the checkpoint module.

## Testing and TDD Sequence

1. Add a boundary test importing both modules and asserting that
   `scripts.render_report.build_checkpoint_metadata` is the exact callable
   owned by `scripts.grid5000.checkpoint_metadata`. Run it first and confirm
   the test is red because the canonical owner is not yet exported.
2. Move the builder and helpers to the checkpoint module, add the compatibility
   imports, and run the focused report/checkpoint tests until green.
3. After green, update the builder-helper tests to import their canonical
   owner and remove the old implementation from `render_report.py`. Run the
   focused suite again to verify unchanged output and errors.
4. Run the full project gate: Ruff, formatting, ty, unit, integration,
   acceptance, import-linter, CRAP, and mutation testing. Mutation results
   must contain no `survived` or `suspicious` mutants; any `no tests` entries
   are reported separately.

## Compatibility and Rollback

No command-line interface, public function signature, JSON key, artifact URI,
exception text, or report output changes. The old import path remains a
facade. If any focused or complete gate detects a behavior change, restore the
implementation in `render_report.py` and remove the moved definitions without
changing callers or fixtures.
