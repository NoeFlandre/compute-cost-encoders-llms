# Reporting Artifact Boundary Design

## Goal

Improve the benchmark artifact-writing boundary without changing public APIs,
artifact bytes, error behavior, or backend execution behavior.

## Scope

The CLI currently coordinates three private reporting helpers: it validates
records, writes JSONL, and builds/writes the summary. That spreads one artifact
workflow across modules and makes the CLI depend on reporting internals.

The refactor introduces one cohesive reporting operation,
`write_measurement_artifacts(output_dir, records)`. It materializes and
validates records once, writes `measurements.jsonl`, then writes the
deterministic `summary.json`. The existing public `build_summary`, `write_jsonl`,
`write_json`, and serialization behavior remain available and unchanged.

The CLI continues to write `manifest.json` before measurement artifacts, so
failure ordering and partial-output behavior remain unchanged. The new helper
does not change record validation, ordering, duplicate detection, JSON options,
or summary contents.

The mutation-quality pass will add focused characterization tests for the
remaining observable contracts: missing CUDA capability access, dependency-lock
propagation, unsupported backends, mapping error context, serialization options,
and explicit UTF-8 file boundaries. The unreachable defensive JSON branch will
be removed because `ensure_ascii` is a literal invariant in the same function.

## Testing

Each source change follows red -> green -> refactor:

1. Add a contract test and run it to observe the expected failure.
2. Implement the smallest behavior-preserving change and rerun the focused
   test plus the affected unit tests.
3. Refactor only after green, then rerun the focused tests.

The final checks are the repository's ordered QA gauntlet, an explicit
`mutmut results` inspection with no surviving or suspicious mutations, and
local/remote SHA verification after publication.
