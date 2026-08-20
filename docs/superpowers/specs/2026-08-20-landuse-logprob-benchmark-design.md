# Binary Land-Use Logprob Benchmark Design

## Goal

Measure the time and resource cost of mapping one made-up land-use sentence to
binary `yes`/`no` log probabilities with `jhu-clsp/mmBERT-base` and
`ggml-org/Qwen3.6-27B-GGUF` using the `Q4_K_M` file, without fine-tuning or
multi-token decoding.

## Scope

The benchmark uses one deterministic example:

> A public park with grass, trees, and walking paths occupies the parcel.

The decision labels are `yes` and `no`, where `yes` means the sentence
describes land use. The encoder uses a masked-token template and reads the
masked-position logits. The causal LLM uses the same semantic prompt and reads
next-token log probabilities for `yes` and `no`. The two probabilities are
model-native scores and are compared as decisions and timings, not as
calibrated probabilities across architectures.

Model loading and text tokenization are reported separately from the primary
warm `text -> logprob` measurement. The LLM benchmark performs only the
minimum final-logit evaluation needed to obtain candidate scores; it does not
measure a generated answer sequence.

## Architecture

Small modules provide stable boundaries:

- `config.py` validates the immutable benchmark configuration.
- `example.py` owns the made-up sentence, label contract, and prompt templates.
- `encoder.py` owns masked-language-model loading and scoring.
- `llm.py` owns the llama.cpp HTTP/logprob boundary.
- `measurement.py` owns monotonic timing and repeated warm measurements.
- `reporting.py` owns JSON output and summary calculations.
- `cli.py` provides the Grid’5000 entry point.

The runtime dependencies remain optional to the base package and are installed
in a pinned Grid’5000 Docker image. All benchmark artifacts are written first
to a run directory and then published to the project HF bucket. Source and
reproducibility metadata remain in GitHub and the LaTeX report is attached to a
release.

## Measurement protocol

1. Resolve and record full immutable revisions and hashes for both models and
   the llama.cpp runtime.
2. Validate that `yes` and `no` are supported candidate tokens for both
   tokenizers; fail closed if the chosen scoring path cannot expose them.
3. Record hardware, driver, runtime, configuration, source commit, and OAR job
   ID.
4. Load each model once and report load time separately.
5. Run a bounded warm-up, then repeated batch-one measurements.
6. Record tokenization time, model computation time, logprob extraction time,
   end-to-end latency, candidate scores, peak GPU memory, and available power
   telemetry.
7. Write raw JSONL measurements and a deterministic summary containing median,
   p5, p95, mean, standard deviation, and the selected decision.

Encoder and LLM runs use the same Grid’5000 hardware class and container
configuration, but separate model processes. No local model benchmark is
allowed. The Grid’5000 wrapper requires an OAR reservation and `uv run
--locked`; policy checks run before and after submission.

## Testing and quality

Unit tests use deterministic fake model boundaries and never download models.
Acceptance tests cover the complete scoring contract, artifact schema, and
reserved-node guard. The ordered gates are Ruff, ty, unit tests, acceptance
tests, architecture checks, CRAP below 6, and mutation testing with zero
surviving mutants. MkDocs, Docker, and `uv build` are release checks.

## Failure and safety behavior

- Reject mutable or missing model revisions.
- Reject labels that cannot be scored through the selected logprob interface.
- Reject execution outside a reserved OAR node.
- Disable prompt/KV reuse between independent measurements.
- Never publish incomplete manifests or credential-like files.
- Keep model caches in temporary Grid’5000 storage and publish only results,
  logs, metadata, and the report.
