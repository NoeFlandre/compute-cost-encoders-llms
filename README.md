# compute-cost-encoders-llms

Minimal starting point for measuring compute costs for encoder models and large language models.

## Project boundaries

- Source code will live in this repository and be synchronized on `main`.
- Data, checkpoints, logs, and run artifacts will live in the public Hugging Face bucket below.
- The public documentation remains intentionally minimal until the first code slice.

## Data bucket

[Public Hugging Face bucket](https://huggingface.co/buckets/NoeFlandre/compute-cost-encoders-llms):
`hf://buckets/NoeFlandre/compute-cost-encoders-llms`

The bucket is not mirrored into Git. Use `hf buckets sync` for data workflows.

## Grid’5000

All compute workloads run on reserved Grid’5000 nodes. Local work is limited to
development checks and control-plane tasks. Read the [Grid’5000 operating
contract](docs/grid5000.md) before submitting a job.

## QA

Run the deterministic quality gauntlet with:

```bash
./scripts/qa.sh
```

The order is Ruff, ty, unit tests, acceptance tests, architecture checks, CRAP,
and mutation testing. CRAP must remain below 6.

## Intended stack

The project uses uv, Ruff, ty, Docker, MkDocs, Grid'5000, pytest with strict RED → GREEN → REFACTOR TDD, executable Gherkin acceptance scenarios, unit and acceptance testing, mutation testing, and a CRAP score below 6. Prefer deep modules with small stable interfaces, low cyclomatic complexity, and explicit dependency boundaries without circular imports.

The same ordered QA gauntlet runs locally and in GitHub Actions.

## Releases

Versioned tags (`vMAJOR.MINOR.PATCH`) trigger QA, package building, and a GitHub
release with the source and wheel distributions. Data and compute artifacts are
published to the HF bucket, not bundled into GitHub releases. See the
[release history](https://github.com/NoeFlandre/compute-cost-encoders-llms/releases).

## License

Apache-2.0. See [LICENSE](LICENSE).
