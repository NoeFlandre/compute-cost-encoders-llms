# Correct Land-Use Logprob Protocol and Rerun

## Goal

Correct the zero-shot binary benchmark so the encoder and Qwen LLM answer the same land-use question with explicit answer framing, Qwen chat-template rendering with reasoning disabled, and fail-closed candidate-logprob handling. Then run the complete quality gate, one bounded Grid’5000 experiment, publish verified artifacts to the public Hugging Face bucket, remove remote scratch data, and generate the agreed one-page PDF.

## Architecture

Keep the existing deep backend boundaries. `example.py` owns the shared question and answer framing; `encoder.py` scores all valid single-token case/spacing variants by exact vocabulary IDs; `llm.py` renders the user message through llama.cpp’s `/apply-template` endpoint with `enable_thinking=false`, then scores one uncached next-token distribution and aggregates only exact yes/no token forms. Measurements use nullable fields for unavailable observations, and manifests record the prompt protocol. The existing reporting layout remains one page, with reproducibility last and no interpretation section.

## Tech Stack

Python 3.12, uv, Ruff, ty, pytest/pytest-bdd, import-linter, crap4py, mutmut, Transformers/PyTorch, llama.cpp server, OAR/Grid’5000, Hugging Face Bucket, and LaTeX.

## TDD execution steps

1. Add failing unit and integration contracts for the explicit encoder answer marker, chat-template payload, disabled reasoning, case/space candidate aggregation, nullable LLM input-token metadata, and the config-to-report pipeline. Run the focused tests and observe the failures.
2. Implement the smallest protocol correction in the example, encoder, and llama.cpp client modules. Run the focused tests until green, then refactor only duplication and typing.
3. Add the manifest/report protocol metadata and preserve the approved one-page report shape. Verify the report tests and PDF compilation locally with deterministic fixtures.
4. Run the full local QA procedure, including package/CLI smoke checks and the repository’s existing mutation stage. Do not edit the delegated protected files; report any pre-existing mutation limitation accurately.
5. Commit only the confirmed implementation and test paths, merge/push the verified result to `main`, and verify the remote SHA.
6. On a Grid’5000 frontend, run `usagepolicycheck -t`, reserve the smallest suitable single GPU node with a short walltime, run the encoder and LLM measurements once with unique IDs, verify artifacts and checkpoints, run the post-run policy check, and cancel/clean any remaining project-owned remote processes.
7. Publish only verified experiment artifacts to the public HF bucket, compile and inspect the one-page PDF, attach it with source/config/model metadata to a GitHub release, and verify HF, GitHub, local, and remote-storage state.
