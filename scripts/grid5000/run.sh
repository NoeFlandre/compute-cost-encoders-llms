#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

if (($# < 5)) || [[ "${1:-}" != "--" ]]; then
    die "usage: $0 -- uv run --locked command [args ...]"
fi
shift

[[ "${1:-}" == "uv" && "${2:-}" == "run" && "${3:-}" == "--locked" ]] || die "compute must run through uv run --locked"
[[ -n "${OAR_JOB_ID:-}" && -n "${OAR_NODEFILE:-}" ]] || die "compute must run inside a reserved OAR job"
[[ -s "$OAR_NODEFILE" ]] || die "OAR_NODEFILE is empty"

run_id="${GRID5000_RUN_ID:-}"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "GRID5000_RUN_ID must be set"
: "${GRID5000_CONFIG_PATH:?GRID5000_CONFIG_PATH must identify the committed configuration}"
: "${GRID5000_DATASET_REVISION:?GRID5000_DATASET_REVISION must be pinned}"
: "${GRID5000_MODEL_REVISION:?GRID5000_MODEL_REVISION must be pinned}"
: "${GRID5000_CHECKPOINT_INTERVAL:?GRID5000_CHECKPOINT_INTERVAL must be set}"
: "${GRID5000_ARTIFACT_PREFIX:?GRID5000_ARTIFACT_PREFIX must be set}"
[[ -f "$GRID5000_CONFIG_PATH" ]] || die "configuration file does not exist"
[[ "$GRID5000_CHECKPOINT_INTERVAL" =~ ^[1-9][0-9]*$ ]] || die "checkpoint interval must be a positive integer"
[[ "$GRID5000_ARTIFACT_PREFIX" != /* && "$GRID5000_ARTIFACT_PREFIX" != *..* ]] || die "artifact prefix must be relative and traversal-free"

git diff --quiet || die "source checkout has unstaged changes"
git diff --cached --quiet || die "source checkout has staged changes"
source_commit="$(git rev-parse HEAD)" || die "could not record source commit"

if command -v sha256sum >/dev/null 2>&1; then
    config_revision="$(sha256sum "$GRID5000_CONFIG_PATH" | awk '{print $1}')"
elif command -v shasum >/dev/null 2>&1; then
    config_revision="$(shasum -a 256 "$GRID5000_CONFIG_PATH" | awk '{print $1}')"
else
    die "no SHA-256 utility is available"
fi

cache_root="${GRID5000_CACHE_ROOT:-/tmp/compute-cost-encoders-llms-${run_id}}"
[[ "$cache_root" != "/" && "$cache_root" != "$HOME" ]] || die "refusing a broad cache path"
mkdir -p "$cache_root/uv" "$cache_root/huggingface"

export GRID5000_SOURCE_COMMIT="$source_commit"
export GRID5000_CONFIG_REVISION="sha256:${config_revision}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$cache_root/uv}"
export HF_HOME="${HF_HOME:-$cache_root/huggingface}"
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"

exec "$@"
