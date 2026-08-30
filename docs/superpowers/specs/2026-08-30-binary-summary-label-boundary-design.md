# Binary summary label boundary design

## Context

`benchmark.reporting._model_summary` constructs two binary-label mappings with
separate hardcoded `yes` and `no` entries. The decision selector already uses
the canonical `candidate_labels()` contract from the example definition. The
duplicated label knowledge makes the summary builder easier to drift out of
alignment with the benchmark's binary protocol.

## Decision

Capture `candidate_labels()` once at the start of `_model_summary` and use that
stable ordered tuple to initialize decision counts and construct mean
log-probability values. Keep the existing record loop, summary keys, numeric
calculations, insertion order, and error behavior unchanged.

## Invariants

- The summary still contains the same `yes` and `no` keys in the same order.
- Every record is classified exactly once with the existing `choose_decision`.
- Mean scores use the same records and labels as before.
- No public API, artifact schema, or validation behavior changes.
- The canonical label provider is evaluated once per model summary.

## Verification

First add a focused unit test proving `_model_summary` consults the shared
label provider and uses its result for the binary summary. Run that test red
before adding the import or implementation. Then implement the smallest local
refactor, run focused reporting and measurement tests, and finish with the
full repository QA, mutation, review, commit, push, and post-commit checks.

## Alternatives considered

- Keeping literal mappings preserves the duplication and allows summary fields
  to diverge from the canonical label order.
- Moving label definitions into a new module would broaden the change without
  a concrete need; `example.py` already owns this protocol value.
