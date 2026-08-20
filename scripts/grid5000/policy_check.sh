#!/usr/bin/env bash
set -euo pipefail

if ! command -v usagepolicycheck >/dev/null 2>&1; then
    printf '%s\n' "error: run this script on a Grid’5000 site frontend." >&2
    exit 1
fi

exec usagepolicycheck -t
