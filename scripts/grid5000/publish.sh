#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

artifact_dir="${1:-}"
remote_prefix="${2:-${GRID5000_ARTIFACT_PREFIX:-}}"
[[ -n "$artifact_dir" && -d "$artifact_dir" ]] || die "provide an artifact directory"
artifact_path="$(cd -- "$artifact_dir" && pwd -P)"
[[ "$artifact_path" != "/" && "$artifact_path" != "$HOME" ]] || die "refusing a broad artifact directory"
[[ -n "$remote_prefix" ]] || die "provide a relative remote artifact prefix"
[[ "$remote_prefix" != /* && "$remote_prefix" != *..* ]] || die "remote prefix must be relative and traversal-free"
[[ "${GRID5000_PUBLISH_CONFIRM:-}" == "yes" ]] || die "set GRID5000_PUBLISH_CONFIRM=yes to publish"

bucket_uri="${HF_BUCKET_URI:-hf://buckets/NoeFlandre/compute-cost-encoders-llms}"
[[ "$bucket_uri" == "hf://buckets/NoeFlandre/compute-cost-encoders-llms" ]] || die "HF_BUCKET_URI must be the project bucket"

for command in hf usagepolicycheck uv; do
    command -v "$command" >/dev/null 2>&1 || die "missing required command: $command"
done

metadata_path="${GRID5000_METADATA_PATH:-$artifact_dir/checkpoint.json}"
[[ -f "$metadata_path" ]] || die "complete checkpoint metadata is required"
uv run --locked python scripts/grid5000/checkpoint_metadata.py "$metadata_path"

if find "$artifact_dir" -type f \( -name ".env" -o -name ".env.*" -o -name "*.pem" -o -name "*.key" \) -print -quit | grep -q .; then
    die "refusing to publish possible credential files"
fi

usagepolicycheck -t
trap 'usagepolicycheck -t' EXIT

hf buckets sync "$artifact_dir" "${bucket_uri%/}/${remote_prefix#/}"
