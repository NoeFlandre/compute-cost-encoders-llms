# Artifact Field Boundary Design

## Goal

Remove the hidden dependency from `scripts/render_report.py` to the
Grid’5000 checkpoint implementation while preserving every existing helper
alias, validation error, report output, and checkpoint artifact.

## Current Context

The report script validates merged backend summaries and currently imports
private `_as_mapping`, `_mapping_value`, and `_text_value` helpers from
`scripts/grid5000/checkpoint_metadata.py`. The checkpoint script therefore
acts as an implementation dependency for report rendering even though the
three helpers only describe generic merged-artifact fields.

The existing `compute_cost_encoders_llms.benchmark._mappings._mapping_field`
helper already provides the shared required/optional mapping behavior. The
two script modules also expose the private helper names because existing tests
and callers use those module-level paths.

## Options Considered

### Keep the current imports

This has no code churn, but preserves the reversed responsibility: report
validation remains coupled to a Grid’5000 publisher module and private helper
ownership is unclear.

### Duplicate the helpers in the report module

This would remove the import but create two copies of the same field
validation logic, making error-message drift and future fixes more likely.

### Add a neutral artifact-field module (recommended)

Create `scripts/_artifact_fields.py` with the three generic helpers. It will
reuse `_mapping_field` for required mapping extraction, while owning the
merged-artifact text and object checks. Both existing modules import the
helpers from this neutral owner, so their current private names remain exact
aliases. The report module will import `build_checkpoint_metadata` from the
checkpoint module only; its validation helpers will no longer depend on that
module.

## Design

`scripts/_artifact_fields.py` will provide:

```python
def _mapping_value(
    document: Mapping[str, object], field: str
) -> Mapping[str, object]: ...


def _text_value(document: Mapping[str, object], field: str) -> str: ...


def _as_mapping(value: object, field: str) -> Mapping[str, object]: ...
```

The implementation will preserve the current behavior exactly:

- `_mapping_value` requires a mapping and raises
  `merged artifact field is not an object: ` followed by the supplied field
  name otherwise;
- `_text_value` requires a non-empty string and raises
  `merged artifact field is not text: ` followed by the supplied field name
  otherwise; and
- `_as_mapping` requires a mapping and raises
  `merged artifact field is not an object: ` followed by the supplied field
  name otherwise.

`scripts.grid5000.checkpoint_metadata` will import all three names from the
neutral module, retaining its current module-level compatibility paths.
`scripts.render_report` will import the same three names from the neutral
module and retain its current `build_checkpoint_metadata` import. No public
function, command-line argument, artifact key, output ordering, or exception
text changes.

## Testing and TDD Sequence

1. Add a boundary test that imports the neutral module and both consumers and
   asserts each consumer exposes the exact canonical helper. Run it before
   adding the module and record the expected collection failure.
2. Add the minimal neutral module and replace the two consumer imports. Run
   the boundary, report, checkpoint, and mapping tests until green.
3. Run formatting, linting, typing, and the complete repository quality gate.
   Mutation results must contain no `survived` or `suspicious` entries; known
   no-test CLI parser/main entries remain reported separately.

## Compatibility and Rollback

The old private module paths remain aliases to the canonical functions, so
existing imports and monkeypatch targets continue to resolve. If any focused
or complete check detects changed behavior, restore the original imports and
remove the neutral module without altering callers or data artifacts.
