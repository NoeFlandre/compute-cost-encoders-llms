#!/usr/bin/env bash
set -euo pipefail

# QA_STAGE: ruff
uv run ruff check .
uv run ruff format --check .

# QA_STAGE: ty
uv run ty check src tests scripts

# QA_STAGE: tests
uv run pytest tests/unit --cov=src --cov=scripts --cov-branch --cov-report=term-missing --cov-report=lcov:coverage.lcov
uv run pytest tests/integration --cov=src --cov=scripts --cov-branch --cov-append --cov-report=term-missing --cov-report=lcov:coverage.lcov

# QA_STAGE: acceptance tests
uv run pytest tests/acceptance --cov=src --cov=scripts --cov-branch --cov-append --cov-report=term-missing --cov-report=lcov:coverage.lcov

# QA_STAGE: architecture checks
PYTHONPATH=src uv run lint-imports --no-cache

# QA_STAGE: CRAP
uv run crap4py src scripts --lcov coverage.lcov --max-crap 5.99 --max-workers 1

# QA_STAGE: mutation tests
if find src scripts -type f -name "*.py" ! -name "__init__.py" -print -quit | grep -q .; then
    uv run mutmut run
    mutation_results="$(uv run mutmut results)"
    printf '%s\n' "$mutation_results"
    if grep -Eq ': (survived|suspicious)$' <<<"$mutation_results"; then
        printf '%s\n' "Mutation testing found surviving or suspicious mutants." >&2
        exit 1
    fi
else
    printf '%s\n' "No implementation modules yet; mutation stage is vacuously satisfied."
fi
