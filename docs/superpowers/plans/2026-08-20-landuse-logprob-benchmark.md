# Binary Land-Use Logprob Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement and run a reproducible Grid’5000 benchmark that measures zero-shot binary land-use log probabilities from mmBERT and Qwen3.6-27B-GGUF.

**Architecture:** A small Python benchmark package will separate configuration, example prompts, encoder scoring, llama.cpp scoring, measurement, and reporting. A reserved-node shell entry point will run the package, write a manifest and JSONL measurements, and a LaTeX renderer will turn the verified summary into a report.

**Tech Stack:** uv, Python 3.12, Ruff, ty, pytest/pytest-bdd, import-linter, crap4py, mutmut, Transformers/PyTorch, llama.cpp server, Docker, Grid’5000 OAR, Hugging Face Bucket, LaTeX.

---

### Task 1: Establish failing benchmark contracts

**Files:**
- Create: `tests/unit/test_benchmark_contract.py`
- Create: `tests/unit/test_benchmark_measurement.py`
- Create: `tests/acceptance/features/landuse_logprob_benchmark.feature`
- Create: `tests/acceptance/test_landuse_logprob_benchmark.py`

- [ ] **Step 1: Write tests for the fixed land-use example, binary label contract, and prompt rendering.**
- [ ] **Step 2: Write tests for rejecting missing revisions, unsupported labels, negative timings, and incomplete records.**
- [ ] **Step 3: Write Gherkin scenarios for deterministic paired scores, reserved-node enforcement, and complete artifact publication.**
- [ ] **Step 4: Run the focused tests and verify they fail because the benchmark package does not yet exist.**

Run: `uv run pytest tests/unit/test_benchmark_contract.py tests/unit/test_benchmark_measurement.py tests/acceptance/test_landuse_logprob_benchmark.py -q`

Expected: collection or import failures for the missing benchmark modules.

### Task 2: Implement configuration and example contracts

**Files:**
- Create: `src/compute_cost_encoders_llms/benchmark/__init__.py`
- Create: `src/compute_cost_encoders_llms/benchmark/config.py`
- Create: `src/compute_cost_encoders_llms/benchmark/example.py`
- Modify: `tests/unit/test_benchmark_contract.py`

- [ ] **Step 1: Implement the immutable configuration and fixed sentence/label contract required by the failing tests.**
- [ ] **Step 2: Run the focused contract tests and verify they pass.**
- [ ] **Step 3: Refactor only names and duplication while keeping the tests green.**

### Task 3: Implement model scoring boundaries

**Files:**
- Create: `src/compute_cost_encoders_llms/benchmark/encoder.py`
- Create: `src/compute_cost_encoders_llms/benchmark/llm.py`
- Modify: `tests/unit/test_benchmark_contract.py`

- [ ] **Step 1: Add failing protocol tests for masked-token candidate scoring and llama.cpp candidate logprob parsing.**
- [ ] **Step 2: Implement the smallest dependency-injected scoring boundaries; unit tests must use deterministic fake backends.**
- [ ] **Step 3: Add fail-closed validation for candidate-token availability and incomplete llama.cpp responses.**
- [ ] **Step 4: Run focused unit tests and verify green.**

### Task 4: Implement timing and report schemas

**Files:**
- Create: `src/compute_cost_encoders_llms/benchmark/measurement.py`
- Create: `src/compute_cost_encoders_llms/benchmark/reporting.py`
- Modify: `tests/unit/test_benchmark_measurement.py`

- [ ] **Step 1: Add failing tests for warmup exclusion, monotonic timing, repeated records, summary quantiles, and deterministic JSON serialization.**
- [ ] **Step 2: Implement timing and summary functions with no model-specific logic.**
- [ ] **Step 3: Run focused tests and verify green.**
- [ ] **Step 4: Refactor to keep every public function below the CRAP threshold.**

### Task 5: Add command-line orchestration and Grid’5000 entry point

**Files:**
- Create: `src/compute_cost_encoders_llms/benchmark/cli.py`
- Create: `scripts/grid5000/benchmark.sh`
- Create: `configs/landuse-logprob.toml`
- Modify: `tests/acceptance/test_landuse_logprob_benchmark.py`
- Modify: `scripts/qa.sh`

- [ ] **Step 1: Add failing acceptance tests for configuration loading, manifest creation, and refusal outside an OAR job.**
- [ ] **Step 2: Implement the CLI and shell wrapper using the existing `scripts/grid5000/run.sh` contract.**
- [ ] **Step 3: Add immutable model/runtime/config fields to the manifest and write JSONL plus summary artifacts.**
- [ ] **Step 4: Run unit and acceptance tests and verify green.**

### Task 6: Add runtime image and LaTeX report generation

**Files:**
- Create: `Dockerfile.grid5000`
- Create: `scripts/render_report.py`
- Create: `reports/landuse-logprob-report.tex`
- Create: `tests/unit/test_report_rendering.py`
- Modify: `docs/grid5000.md`
- Modify: `README.md`

- [ ] **Step 1: Add failing tests for escaping report values and rendering a complete summary.**
- [ ] **Step 2: Implement the minimal LaTeX renderer and a report template with methods, environment, measurements, limitations, and results tables.**
- [ ] **Step 3: Build the Grid’5000 image with pinned Python, Transformers/PyTorch, and llama.cpp dependencies.**
- [ ] **Step 4: Run report tests, `uv build`, and strict MkDocs.**

### Task 7: Run local quality gates and commit the implementation

**Files:**
- Modify only the files listed in Tasks 1–6.

- [ ] **Step 1: Run `./scripts/qa.sh` in the required order and require zero surviving mutants.**
- [ ] **Step 2: Run `uv build` and `uv run --locked mkdocs build --strict`.**
- [ ] **Step 3: Inspect the diff and confirm protected unrelated files are untouched.**
- [ ] **Step 4: Commit the exact implementation paths on `main` with a Conventional Commit.**
- [ ] **Step 5: Push `main` and verify the remote SHA and GitHub Actions result.**

### Task 8: Execute the Grid’5000 benchmark and publish results

**Files/artifacts:**
- Runtime artifacts under a temporary Grid’5000 run directory.
- Published results under `runs/<run-id>/` in the project HF bucket.
- LaTeX/PDF report attached to a GitHub release.

- [ ] **Step 1: Run `usagepolicycheck -t` on the site frontend and verify the exact source commit, model revisions, hardware, and configuration.**
- [ ] **Step 2: Submit one bounded smoke run with a unique run ID and the smallest suitable GPU allocation.**
- [ ] **Step 3: Inspect the smoke manifest and timings, then submit the single bounded measurement run without duplicating an active job.**
- [ ] **Step 4: Verify every measurement, summary, and report input before publication.**
- [ ] **Step 5: Run the post-submission policy check, publish only complete artifacts to the HF bucket, and cancel any remaining job.**
- [ ] **Step 6: Compile the LaTeX report, verify the PDF, and publish the report and metadata in a GitHub release.**
- [ ] **Step 7: Verify the HF inventory, release assets, remote refs, exact job IDs, and clean `main` checkout.**
