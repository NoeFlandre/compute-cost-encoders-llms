# compute-cost-encoders-llms

Minimal starting point for measuring compute costs for encoder models and large language models.

## Project boundaries

- Source code will live in this repository and be synchronized on `main`.
- Data, checkpoints, logs, and run artifacts will live in the private Hugging Face bucket below.
- The public documentation remains intentionally minimal until the first code slice.

## Data bucket

`hf://buckets/NoeFlandre/compute-cost-encoders-llms`

The bucket is not mirrored into Git. Use Hugging Face bucket commands or `hf sync` for data workflows.

## Intended stack

The project will use uv, Ruff, ty, Docker, MkDocs, Grid'5000, pytest with RED → GREEN TDD, mutation testing, and a CRAP score below 6. These tools will be configured alongside the first implementation.
