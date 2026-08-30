# Benchmark Manifest Boundary Design

## Goal

Separate reproducibility-manifest construction from backend loading and CLI
orchestration while preserving every existing manifest value, function
signature, import path, and command-line behavior.

## Current Context

`src/compute_cost_encoders_llms/benchmark/cli.py` currently owns backend
loading, measurement execution, artifact writing, runtime probing, and the
pure `build_manifest` function. The manifest builder only transforms a
`BenchmarkConfig` and supplied provenance mappings into a JSON-compatible
dictionary, so it does not need to depend on the CLI module or backend
implementations.

The repository has already separated other pure contracts from orchestration:
report rendering and checkpoint metadata each have canonical modules with
compatibility facades where older imports are part of the supported surface.
The manifest builder is the remaining small pure contract inside the CLI.

## Options Considered

### Keep the module unchanged

This has no migration risk, but leaves a pure reproducibility contract coupled
to backend execution and makes manifest changes harder to locate and test in
isolation.

### Extract all CLI support helpers at once

Moving manifest construction, source-commit discovery, hardware probing,
dependency hashing, and parser creation together would make `cli.py` smaller,
but it would introduce several new boundaries in one change and expand the
regression surface without a concrete need.

### Extract only the pure manifest contract (recommended)

Move `build_manifest` and its fixed-example imports to
`benchmark/manifest.py`. Keep `cli.py` responsible for loading, measuring,
runtime observation, and writing artifacts. Re-import `build_manifest` in
`cli.py` so the existing `benchmark.cli.build_manifest` path remains the exact
same callable. This is the smallest cohesive boundary and follows YAGNI.

## Design

`benchmark/manifest.py` will expose one function:

```python
def build_manifest(
    config: BenchmarkConfig,
    *,
    run_id: str,
    source_commit: str,
    hardware: Mapping[str, object],
    runtime: Mapping[str, object] | None = None,
    dependency_lock_sha256: str | None = None,
) -> dict[str, object]: ...
```

The implementation and its imports move without behavior edits. The returned
schema, nested values, tuple-valued labels, protocol constants, handling of
`None`, and copy behavior for supplied mappings remain byte-for-byte
equivalent after JSON serialization. The new module imports only the config
type and fixed benchmark-example functions; it will not import `cli.py` or
perform I/O, environment access, subprocess calls, or model loading.

`benchmark.cli` will import `build_manifest` from the new module. Existing
callers continue to use the old import path, while the identity is protected
by a focused boundary test. No CLI argument, output file, manifest field, or
runtime observation changes.

## Testing and TDD Sequence

1. Add a focused boundary test asserting that
   `benchmark.cli.build_manifest` is the exact callable owned by
   `benchmark.manifest`. Run it before moving production code and record the
   expected collection/attribute failure.
2. Move the existing implementation verbatim, add the compatibility import,
   and run the boundary test plus the existing CLI/contract tests until green.
3. Move the manifest behavior test import to the canonical module while
   retaining the facade identity test. Run focused formatting, lint, type, and
   behavior checks.
4. Run the complete project gate: Ruff, formatting, ty, unit, integration,
   acceptance, import-linter, CRAP, and mutation testing. Require no
   surviving or suspicious mutants; report existing no-test CLI mutants
   separately.

## Compatibility and Rollback

No public signature, manifest key, value, JSON representation, exception,
CLI option, artifact path, or backend behavior changes. The compatibility
import can be reverted independently if any focused or complete gate detects
a difference, restoring the original builder body in `cli.py` without changing
its callers.
