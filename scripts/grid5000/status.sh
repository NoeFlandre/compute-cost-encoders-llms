#!/usr/bin/env bash
set -euo pipefail

job_id="${1:-}"
[[ "$job_id" =~ ^[0-9]+$ ]] || {
    printf '%s\n' "usage: $0 JOB_ID" >&2
    exit 1
}

command -v oarstat >/dev/null 2>&1 || {
    printf '%s\n' "error: run this script on a Grid’5000 frontend." >&2
    exit 1
}

exec oarstat -j "$job_id"
