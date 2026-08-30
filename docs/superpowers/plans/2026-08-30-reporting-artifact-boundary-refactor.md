# Reporting Artifact Boundary Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce CLI/reporting coupling and close the measured mutation-quality gaps without changing benchmark behavior or artifact contracts.

**Architecture:** Keep validation and serialization in `benchmark.reporting`, but expose one cohesive `write_measurement_artifacts` operation for the CLI. It validates once, writes the existing JSONL artifact, and writes the existing summary artifact in the same order. Public `build_summary`, `write_jsonl`, `write_json`, and render functions remain unchanged.

**Tech Stack:** Python 3.12, uv, pytest, Ruff, ty, import-linter, crap4py, mutmut.

---

### Task 1: Add the cohesive artifact-writing contract

**Files:**
- Modify: `tests/unit/test_benchmark_measurement.py`
- Modify: `tests/unit/test_benchmark_cli.py`
- Modify: `src/compute_cost_encoders_llms/benchmark/reporting.py`
- Modify: `src/compute_cost_encoders_llms/benchmark/cli.py`

- [x] **Step 1: Write the failing test**

Add a test that calls the new helper with two valid records, records calls to
`reporting.validate_measurement`, and asserts two validations plus exact
`measurements.jsonl` bytes and a JSON summary equal to `build_summary`.

```python
def test_write_measurement_artifacts_validates_once_and_writes_both_outputs(
    monkeypatch, tmp_path
) -> None:
    record = {
        "model": "encoder",
        "repetition": 0,
        "tokenization_ms": 1.0,
        "model_ms": 2.0,
        "logprob_ms": 0.1,
        "text_to_logprob_ms": 3.1,
        "logprobs": {"yes": -0.1, "no": -2.2},
    }
    second = {**record, "repetition": 1}
    calls: list[object] = []
    original_validate = reporting_module.validate_measurement

    def capture_validate(value: Mapping[str, object]) -> object:
        calls.append(value)
        return original_validate(value)

    monkeypatch.setattr(reporting_module, "validate_measurement", capture_validate)
    reporting_module.write_measurement_artifacts(tmp_path, [record, second])

    assert len(calls) == 2
    assert (tmp_path / "measurements.jsonl").read_text() == "\n".join(
        (json_line(record), json_line(second))
    ) + "\n"
    assert json.loads((tmp_path / "summary.json").read_text()) == build_summary(
        [record, second]
    )
```

- [x] **Step 2: Run the focused test to verify it fails**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_measurement.py -k write_measurement_artifacts -q
```

Expected: FAIL with `AttributeError` because the helper does not yet exist.

- [x] **Step 3: Implement the minimal helper and CLI delegation**

Add this function after `write_jsonl` in `reporting.py`:

```python
def write_measurement_artifacts(
    output_dir: Path, records: Iterable[Mapping[str, object]]
) -> None:
    """Write validated measurement and summary artifacts for one run."""

    validated = _validated_records(records)
    _write_validated_jsonl(output_dir / "measurements.jsonl", validated)
    write_json(output_dir / "summary.json", _summary_from_validated_records(validated))
```

In `cli.py`, import `write_measurement_artifacts` and replace the three
private artifact calls in `run` with:

```python
write_measurement_artifacts(output_dir, records)
```

- [x] **Step 4: Run the focused and affected tests**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_measurement.py tests/unit/test_benchmark_cli.py -q
```

Expected: all tests pass, including the new helper contract and the existing
CLI manifest/artifact contract.

- [x] **Step 5: Refactor only after green**

Move the validation-count assertion into the cohesive reporting test. Update
the CLI test to assert that `cli.write_measurement_artifacts` receives the
output directory and records, while retaining byte-level artifact assertions
in the reporting test.

- [x] **Step 6: Re-run the affected tests**

Run the Step 4 command and expect all tests to pass.

### Task 2: Remove unreachable defensive code and lock down its invariant

**Files:**
- Modify: `src/compute_cost_encoders_llms/benchmark/reporting.py`
- Modify: `tests/unit/test_reporting_document_contract.py`

- [x] **Step 1: Write the failing quality test**

Add this focused contract. The option assertions characterize the existing
serialization modes; the source assertion is the red quality expectation for
removing the unreachable defensive branch:

```python
def test_json_options_preserves_modes_and_rejects_non_boolean() -> None:
    assert reporting_module._json_options(compact=True) == {
        "ensure_ascii": False,
        "sort_keys": True,
        "separators": (",", ":"),
    }
    assert reporting_module._json_options(compact=False) == {
        "ensure_ascii": False,
        "sort_keys": True,
        "indent": 2,
    }
    with pytest.raises(TypeError, match=r"^compact must be a boolean$"):
        reporting_module._json_options(compact=1)
    source = Path(reporting_module.__file__).read_text(encoding="utf-8")
    assert 'if options["ensure_ascii"] is not False' not in source
```

- [x] **Step 2: Run the focused test to verify red**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_reporting_document_contract.py -k json_options -q
```

Expected: FAIL on the assertion that the unreachable branch is absent. If it
does not collect, fix only the test import/setup before touching source.

- [x] **Step 3: Make the minimal refactor**

Delete only the `if options["ensure_ascii"] is not False` branch and its
exception. Keep the literal `ensure_ascii=False` option unchanged.

- [x] **Step 4: Verify green**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_reporting_document_contract.py tests/unit/test_benchmark_measurement.py -q
```

### Task 3: Enforce the zero-survivor mutation gate

**Files:**
- Modify: `tests/acceptance/test_project_quality.py`
- Modify: `scripts/qa.sh`

- [x] **Step 1: Write the failing acceptance contract**

Add an assertion that the QA script captures `mutmut results` and rejects both
surviving and suspicious statuses:

```python
def test_qa_rejects_surviving_or_suspicious_mutants() -> None:
    qa_script = (PROJECT_ROOT / "scripts" / "qa.sh").read_text()

    assert 'mutation_results="$(uv run mutmut results)"' in qa_script
    assert "grep -Eq ': (survived|suspicious)$'" in qa_script
```

- [x] **Step 2: Run the acceptance contract to verify red**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/acceptance/test_project_quality.py -k mutation -q
```

Expected: FAIL because the current QA script does not inspect mutmut results.

- [x] **Step 3: Implement the minimal fail-closed check**

Append this immediately after the existing `mutmut run` command in
`scripts/qa.sh`:

```bash
mutation_results="$(uv run mutmut results)"
printf '%s\\n' "$mutation_results"
if grep -Eq ': (survived|suspicious)$' <<<"$mutation_results"; then
    printf '%s\\n' "Mutation testing found surviving or suspicious mutants." >&2
    exit 1
fi
```

- [x] **Step 4: Run the acceptance contract to verify green**

Run the Step 2 command and expect all project-quality acceptance tests to pass.

### Task 4: Add focused tests for baseline mutation survivors

**Files:**
- Modify: `tests/unit/test_benchmark_cli.py`
- Modify: `tests/unit/test_benchmark_measurement.py`
- Modify: `tests/unit/test_reporting_document_contract.py`
- Modify: `tests/unit/test_mapping_contract.py`
- Modify: `tests/unit/test_render_report.py`
- Modify: `tests/unit/test_grid5000_checkpoint.py`

- [x] **Step 1: Add characterization tests before source changes**

Cover the currently unobserved contracts with exact assertions:

```python
def test_mapping_field_uses_document_as_default_error_context() -> None:
    with pytest.raises(ValueError, match=r"^document field is not an object: value$"):
        _mapping_field({}, "value", required=True)


def test_device_capability_returns_none_when_attribute_is_missing() -> None:
    assert runtime_module._device_capability(object()) is None
```

Also add tests that call `_load_encoder(config, "lock-sha")` and assert the
runtime metadata receives `"lock-sha"); reject an invalid backend before either
backend-record function is called; exercise a synthetic module path whose
`parents[3] / "uv.lock"` is the fallback candidate; and assert explicit
`encoding="utf-8"` at the reporting, report-rendering, and checkpoint-file
boundaries with small recording path doubles. Retain the existing Unicode byte
assertions.

- [x] **Step 2: Run the focused tests and inspect failures**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run pytest tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_measurement.py tests/unit/test_reporting_document_contract.py tests/unit/test_mapping_contract.py tests/unit/test_render_report.py tests/unit/test_grid5000_checkpoint.py -q
```

Expected: only the new assertions fail, and failures identify missing test
coverage or an actually broken contract rather than collection errors.

- [x] **Step 3: Implement only necessary behavior-preserving adjustments**

Keep all existing outputs/errors and add no new feature. Use the smallest
source change only if a new test demonstrates an unguarded behavior path.

- [x] **Step 4: Run the focused tests again**

Run the Step 2 command and expect all tests to pass.

### Task 5: Full verification and publication

**Files:**
- Modify only the source, test, design, and plan paths named above.

- [x] **Step 1: Run the complete checks**

```bash
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache ./scripts/qa.sh
UV_CACHE_DIR=/private/tmp/compute-cost-encoders-llms-uv-cache uv run mutmut results
```

Expected: Ruff, ty, all tests, import-linter, and CRAP pass; no mutation is
`survived` or `suspicious`. Any `no tests` entries are reported separately
from actual survivors.

- [x] **Step 2: Review and commit explicit paths**

```bash
git diff --check
git status --short --branch
git add docs/superpowers/plans/2026-08-30-reporting-artifact-boundary-refactor.md src/compute_cost_encoders_llms/benchmark/cli.py src/compute_cost_encoders_llms/benchmark/reporting.py tests/unit/test_benchmark_cli.py tests/unit/test_benchmark_measurement.py tests/unit/test_reporting_document_contract.py tests/unit/test_mapping_contract.py tests/unit/test_render_report.py tests/unit/test_grid5000_checkpoint.py
git commit -m "refactor: simplify measurement artifact writing"
```

- [x] **Step 3: Push and verify synchronization**

```bash
git push origin main
git status --short --branch
git rev-parse HEAD
git rev-parse origin/main
git ls-remote origin refs/heads/main
```

Expected: clean worktree and matching local, tracking, and remote SHAs.
