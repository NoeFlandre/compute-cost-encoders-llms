# Land-use logprob benchmark

The benchmark compares one out-of-the-box masked-language-model pass with one
out-of-the-box llama.cpp next-token request for this made-up binary example:

> A public park with grass, trees, and walking paths occupies the parcel.

Both backends score the labels `yes` and `no`. The encoder fills one mask and
does not decode. Fixed candidate label forms are tokenized once during encoder
setup and reused across repetitions; prompt tokenization remains per
repetition. The LLM requests exactly one prediction with prompt caching
disabled; its prompt evaluation and one-token prediction are included in the
text-to-logprob measurement. No model is fine-tuned and no dataset is used.

The run pins the model revisions in
[`configs/landuse-logprob.toml`](https://github.com/NoeFlandre/compute-cost-encoders-llms/blob/main/configs/landuse-logprob.toml).
It writes raw measurements, summaries, a complete checkpoint manifest, and a
standalone LaTeX report to the public project HF bucket.

All model computation runs on a reserved Grid’5000 node. See the
[Grid’5000 operating contract](grid5000.md) for policy checks, bounded
resources, checkpoint metadata, publication, and cleanup.

The two models have different training objectives, so their native log
probabilities are reported for decision inspection but are not treated as
calibrated, directly comparable probabilities. Timing is compared from input
text to the available binary logprob result.

When a backend does not expose a timing component, the raw measurement and
summary contain `null`; zero is never used as a placeholder. Each manifest
also records the selected dtype, Python/Torch/Transformers versions, CUDA GPU,
runtime and driver observations, llama.cpp revision, model revisions, source
commit, configuration digest, and `uv.lock` digest.

## Completed Grid’5000 run

Run `landuse-logprob-20260820T214923Z` used job `4055299` on
`abacus28-1.rennes.grid5000.fr` (Tesla V100-SXM2-32GB), with 8 warmups and 64
measured repetitions. The source commit was
`3e69807b2faa62673b474be0f20422ea69066d96`; prompt caching was disabled and
one generated token was requested for the LLM.

| Backend | Median text → logprob | P95 text → logprob | Median model time | Decision |
| --- | ---: | ---: | ---: | --- |
| mmBERT encoder | 74.395 ms | 75.827 ms | 23.271 ms | no |
| Qwen3.6-27B Q4_K_M | 201.471 ms | 202.583 ms | 197.456 ms | no |

The LLM path was 2.71× slower at the median and 2.67× slower at P95 for this
single sentence and this hardware. Both backends selected `no` for the made-up
example; this is a timing experiment, not an accuracy evaluation, and one
example cannot support a model-selection conclusion. The complete measurements,
checkpoint, LaTeX source, and compiled PDF are stored in the public HF bucket.
