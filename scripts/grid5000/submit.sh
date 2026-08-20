#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

if (($# == 0)); then
    die "usage: GRID5000_RUN_ID=... $0 command [args ...]"
fi

for command in usagepolicycheck oarsub oarstat; do
    command -v "$command" >/dev/null 2>&1 || die "missing Grid’5000 command: $command"
done

run_id="${GRID5000_RUN_ID:-}"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "GRID5000_RUN_ID must be a safe unique identifier"

job_name="compute-cost-encoders-llms-${run_id}"
resources="${GRID5000_RESOURCES:-host=1/gpu=1,walltime=0:30}"
[[ "$resources" == *"walltime="* ]] || die "GRID5000_RESOURCES must set walltime"
queue_args=()
if [[ -n "${GRID5000_QUEUE:-}" ]]; then
    queue_args=(-q "$GRID5000_QUEUE")
fi
property_args=()
if [[ -n "${GRID5000_PROPERTIES:-}" ]]; then
    property_args=(-p "$GRID5000_PROPERTIES")
fi

usagepolicycheck -t
trap 'usagepolicycheck -t' EXIT

active_jobs="$(oarstat -u 2>/dev/null)" || die "could not inspect existing OAR jobs"
if grep -Fq "$job_name" <<<"$active_jobs"; then
    die "a job with run ID '$run_id' is already active; refusing a duplicate"
fi

oarsub -n "$job_name" -l "$resources" "${queue_args[@]}" "${property_args[@]}" "$@"
