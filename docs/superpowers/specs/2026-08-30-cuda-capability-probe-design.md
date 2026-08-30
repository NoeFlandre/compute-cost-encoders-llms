# CUDA capability probe design

## Context

`benchmark.runtime` has two helpers that independently call
`cuda.get_device_capability()`: `_fp16_supported` converts the result to a
tuple for comparison, while `_device_capability` performs the same guarded
call and conversion for runtime metadata. Maintaining both defensive probe
implementations duplicates failure handling and makes the capability contract
harder to evolve safely.

## Decision

Have `_fp16_supported` call `_device_capability` and compare the returned
capability to `(5, 3)` only when a capability was observed. Keep
`_device_capability` returning `list[int] | None`, retain all existing caught
exceptions and conversions, and leave the precision-selection policy
unchanged.

## Invariants

- Missing, malformed, or failing capability probes still report FP16 as
  unsupported.
- Capability `(5, 3)` and newer still report FP16 as supported.
- Capability below `(5, 3)` still reports FP16 as unsupported.
- The capability getter is invoked at most once for an FP16 check.
- Runtime metadata keeps its existing list-shaped capability value.
- No public API, output, exception, or device-selection behavior changes.

## Verification

First add a focused unit test proving `_fp16_supported` consumes the existing
normalized capability helper, and run it red before changing production code.
Then make the smallest reuse refactor, run the runtime/CLI tests and static
checks, and finish with the complete QA, mutation, review, commit, push, and
post-push ref checks.

## Alternatives considered

- Keeping two probes preserves unnecessary duplication and separate failure
  handling.
- Introducing a generic safe-call abstraction would broaden the change beyond
  the concrete duplicated capability boundary.
- Changing `_device_capability` to return a tuple would alter an existing
  private helper's observed shape without providing a needed benefit.
