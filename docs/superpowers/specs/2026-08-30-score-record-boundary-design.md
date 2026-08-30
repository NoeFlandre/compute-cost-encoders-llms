# Score Record Boundary Design

## Goal

Separate backend-score normalization from CLI orchestration while preserving
the existing `score_record` signature, import path, record schema, and
benchmark behavior.

## Current Context

`benchmark/cli.py` currently owns backend loading, repetition measurement,
artifact writing, runtime observation, command-line parsing, and the pure
`score_record` adapter. The adapter only reads the common score attributes and
calls `choose_decision` to produce a `MeasurementRecord`; it does not need
subprocesses, environment access, model loading, or CLI state.

`benchmark/measurement.py` already owns `MeasurementRecord`,
`MeasurementError`, `choose_decision`, and measurement validation. Moving the
adapter there gives the pure record contract one canonical home. The CLI will
re-import the function so existing callers of `benchmark.cli.score_record`
continue to receive the same callable.

## Options Considered

### Keep the module unchanged

This has no migration risk, but leaves a pure measurement transformation
coupled to backend orchestration and forces readers to inspect the CLI to find
the record-construction contract.

### Extract all remaining CLI support helpers

Moving score normalization together with source-commit discovery, hardware
probing, dependency hashing, and parser creation would reduce the CLI size,
but would introduce multiple new boundaries without a concrete requirement.
It would also increase the compatibility and regression surface.

### Move only score normalization to the measurement module (recommended)

Add a small structural `ScoreLike` protocol and `score_record` function to
`benchmark/measurement.py`. Remove the implementation from `cli.py` and
import it there as a compatibility facade. The protocol keeps measurement
code independent of the concrete encoder and Llama backends while retaining
the existing runtime behavior and type contract.

## Design

`benchmark.measurement` will expose the following pure contract:

```python
class ScoreLike(Protocol):
    @property
    def tokenization_ms(self) -> float | None: ...

    @property
    def model_ms(self) -> float | None: ...

    @property
    def logprob_ms(self) -> float: ...

    @property
    def text_to_logprob_ms(self) -> float: ...

    @property
    def input_tokens(self) -> int | None: ...

    @property
    def logprobs(self) -> dict[str, float]: ...


def score_record(
    model: str, repetition: int, score: ScoreLike
) -> MeasurementRecord: ...
```

The implementation will be moved without changing its returned dictionary,
decision calculation, or exception behavior. `benchmark.cli.score_record`
will be the exact imported callable from `benchmark.measurement`; no caller,
CLI option, output artifact, JSON representation, backend operation, or public
compatibility path changes.

## Testing and TDD Sequence

1. Add a focused boundary test asserting that
   `benchmark.cli.score_record` is the exact callable owned by
   `benchmark.measurement`. Run it before changing production code and record
   the expected collection/attribute failure because the new owner does not
   exist yet.
2. Add the structural protocol and move the existing function body into
   `measurement.py`; import it in `cli.py` and run the boundary, score-record,
   CLI, and pipeline tests.
3. Move the detailed score-record behavior test to the canonical measurement
   import while retaining the CLI identity assertion. Run focused formatting,
   linting, typing, and behavior checks.
4. Run the complete project gate: Ruff, formatting, ty, unit, integration,
   acceptance, import-linter, CRAP, and a fresh mutation campaign. Require no
   surviving or suspicious mutants; report the known no-test CLI parser/main
   mutants separately.

## Compatibility and Rollback

No public signature, record key, value, exception, ordering, CLI option,
artifact path, or backend behavior changes. If any focused or complete gate
detects a difference, restore the original function body in `cli.py` and
remove the measurement owner; callers and outputs remain unchanged.
