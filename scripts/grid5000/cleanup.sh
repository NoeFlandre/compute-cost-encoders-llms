#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

run_id="${GRID5000_RUN_ID:-}"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "GRID5000_RUN_ID must be set"
[[ "$run_id" != *..* ]] || die "GRID5000_RUN_ID must not contain traversal"
temp_root="${GRID5000_TEMP_ROOT:-/tmp/compute-cost-encoders-llms-${run_id}}"
[[ "$temp_root" == /tmp/compute-cost-encoders-llms-* ]] || die "cleanup is restricted to project-owned /tmp paths"
[[ "$temp_root" != *..* ]] || die "cleanup path must not contain traversal"
[[ ! -L "$temp_root" ]] || die "refusing to remove a symlink"
[[ "${GRID5000_CLEANUP_CONFIRM:-}" == yes ]] || die "set GRID5000_CLEANUP_CONFIRM=yes to clean up"

if [[ -e "$temp_root" ]]; then
    printf 'Removing confirmed project-owned temporary path: %s\n' "$temp_root"
    rm -rf -- "$temp_root"
fi
