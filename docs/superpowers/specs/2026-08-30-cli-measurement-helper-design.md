# CLI measurement helper design

## Context

`benchmark.cli` currently contains two backend-specific functions,
`_encoder_records` and `_llm_records`, that each repeat the same orchestration:
run `measure_repetitions`, enumerate the timed values, and normalize each value
with `score_record`. The duplicated code makes the two paths unnecessarily
easy to drift apart and obscures the boundary between backend execution and
common measurement handling.

## Decision

Add one private `_measure_records` helper in `benchmark.cli` with the following
responsibilities:

1. Execute the supplied zero-argument backend operation through
   `measure_repetitions` using the configured warmups and repetitions.
2. Convert each measured value to a `MeasurementRecord` with the supplied model
   label and its zero-based repetition index.

`_encoder_records` and `_llm_records` will delegate only this shared work to the
helper. They will retain ownership of backend loading, operation construction,
and runtime metadata. The existing imports of `measure_repetitions` and
`score_record` remain in `cli.py`, preserving their current monkeypatch and
module compatibility surface.

## Invariants

- Warmups are still excluded from returned records.
- The operation is invoked exactly as often as before.
- Returned records preserve their model labels, ordering, indexes, and schema.
- Encoder candidate token IDs continue to be built once and reused by every
  encoder operation.
- LLM seed handling and runtime metadata collection are unchanged.
- No public API, output artifact, or measurement validation behavior changes.

## Verification

First add a focused unit test for `_measure_records` that proves warmups are
excluded and measured scores become ordered normalized records. Run that test
red before adding the helper. Then implement the smallest helper and run the
focused test plus the existing CLI and measurement tests green. Finally run
the repository QA script, including coverage, type checking, import linting,
CRAP analysis, and mutation testing.

## Alternatives considered

- Moving the helper into `measurement.py` would broaden the measurement module's
  API for a CLI-only orchestration concern.
- Keeping two copies would preserve the current duplication and leave the
  backend paths vulnerable to divergence.

The private CLI helper is the smallest cohesive change that removes the
duplication without introducing a new public abstraction.
