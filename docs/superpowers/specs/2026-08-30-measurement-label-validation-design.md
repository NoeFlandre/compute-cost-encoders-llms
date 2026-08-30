# Measurement Label Validation Design

## Context

`compute_cost_encoders_llms.benchmark.measurement` already imports the canonical
`candidate_labels()` protocol helper for `choose_decision`, but its validation
helpers still repeat the binary labels as `("yes", "no")` and `{"yes", "no"}`.
That leaves the validation boundary with duplicated protocol knowledge that can
drift from the decision boundary.

## Decision

Use `candidate_labels()` in `_validate_logprobs`,
`_has_binary_logprob_keys`, `_has_finite_binary_logprobs`, and
`_validate_decision`. Keep the existing helper names, signatures, error text,
return shapes, and default label order. No new abstraction or public API is
needed; the existing example module remains the single owner of the labels.

The refactor is behavior-preserving for the current `("yes", "no")` labels.
It additionally makes all validation paths follow the same canonical contract
if the protocol labels evolve. A regression test will patch the module-local
canonical lookup to a different binary pair and validate a complete record,
proving that key, finite-score, normalization, and decision checks use it.

## Alternatives considered

- Keep the literals: smallest textual diff, but preserves duplicated protocol
  knowledge and allows validation to drift.
- Introduce a new label-set abstraction: unnecessary because `candidate_labels`
  already provides the required stable contract.
- Change the public record schema: out of scope and would risk compatibility.

## Compatibility contract

- Existing `yes`/`no` inputs produce the same normalized records and decisions.
- Existing validation errors remain unchanged for malformed default-label input.
- No exported names, function signatures, serialized fields, or artifact output
  formats change.
- The change remains limited to `measurement.py` and its unit contract test.
