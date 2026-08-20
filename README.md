# compute-cost-encoders-llms

Minimal starting point for measuring compute costs for encoder models and large language models.

## Project boundaries

- Source code will live in this repository and be synchronized on `main`.
- Data, checkpoints, logs, and run artifacts will live in the public Hugging Face bucket below.
- The public documentation remains intentionally minimal until the first code slice.

## Data bucket

[Public Hugging Face bucket](https://huggingface.co/buckets/NoeFlandre/compute-cost-encoders-llms):
`hf://buckets/NoeFlandre/compute-cost-encoders-llms`

The bucket is not mirrored into Git. Use Hugging Face bucket commands or `hf sync` for data workflows.

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

## License

Apache-2.0. See [LICENSE](LICENSE).
