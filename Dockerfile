FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /workspace

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project

COPY . .

CMD ["uv", "run", "--locked", "pytest"]
