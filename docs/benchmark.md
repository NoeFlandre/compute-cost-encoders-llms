# Land-use logprob benchmark

The benchmark compares one out-of-the-box masked-language-model pass with one
out-of-the-box llama.cpp next-token request for this made-up binary example:

> A public park with grass, trees, and walking paths occupies the parcel.

Both backends score the labels `yes` and `no`. The encoder fills one mask and
does not decode. The LLM requests exactly one prediction with prompt caching
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
