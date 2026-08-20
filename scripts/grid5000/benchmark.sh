#!/usr/bin/env bash
set -euo pipefail

die() {
    printf 'error: %s\n' "$*" >&2
    exit 1
}

backend="${1:-}"
config_path="${2:-}"
output_dir="${3:-}"
run_id="${4:-${GRID5000_RUN_ID:-}}"
[[ "$backend" == "encoder" || "$backend" == "llm" || "$backend" == "both" ]] || die "backend must be encoder, llm, or both"
[[ -f "$config_path" ]] || die "benchmark configuration does not exist"
[[ -n "$output_dir" && "$output_dir" != / && "$output_dir" != *..* ]] || die "unsafe output directory"
[[ "$run_id" =~ ^[A-Za-z0-9._-]+$ ]] || die "run ID must be set and traversal-free"

run_backend() {
    local selected_backend="$1"
    mkdir -p "$output_dir/$selected_backend"
    # The Python request sets cache_prompt=false for independent measurements.
    uv run --locked python -m compute_cost_encoders_llms.benchmark.cli \
        --config "$config_path" \
        --output-dir "$output_dir/$selected_backend" \
        --backend "$selected_backend" \
        --run-id "$run_id-$selected_backend"
}

ensure_qwen_model() {
    : "${GRID5000_LLM_REVISION:?GRID5000_LLM_REVISION must be pinned}"
    local model_dir="${HF_HOME:-/tmp/compute-cost-encoders-llms-huggingface}/qwen"
    local filename="${GRID5000_LLM_FILENAME:-Qwen3.6-27B-Q4_K_M.gguf}"
    if [[ -z "${GRID5000_QWEN_MODEL_PATH:-}" ]]; then
        mkdir -p "$model_dir"
        uv run --locked hf download ggml-org/Qwen3.6-27B-GGUF \
            --revision "$GRID5000_LLM_REVISION" \
            --include "$filename" \
            --local-dir "$model_dir"
        export GRID5000_QWEN_MODEL_PATH="$model_dir/$filename"
    fi
}

server_pid=""
stop_server() {
    if [[ -n "$server_pid" ]]; then
        kill "$server_pid" 2>/dev/null || true
        wait "$server_pid" 2>/dev/null || true
    fi
}

start_server() {
    : "${LLAMA_SERVER_BIN:?LLAMA_SERVER_BIN must identify a pinned llama-server binary}"
    ensure_qwen_model
    : "${GRID5000_QWEN_MODEL_PATH:?GRID5000_QWEN_MODEL_PATH must identify the Q4_K_M file}"
    [[ -f "$GRID5000_QWEN_MODEL_PATH" ]] || die "Qwen GGUF file does not exist"
    command -v curl >/dev/null 2>&1 || die "curl is required for the local server health check"
    local port="${LLAMA_SERVER_PORT:-8080}"
    local log_path="$output_dir/llama-server.log"
    "$LLAMA_SERVER_BIN" \
        --model "$GRID5000_QWEN_MODEL_PATH" \
        --host 127.0.0.1 \
        --port "$port" \
        --n-gpu-layers "${LLAMA_GPU_LAYERS:-999}" \
        --ctx-size "${LLAMA_CONTEXT_SIZE:-2048}" \
        --parallel 1 \
        --no-webui \
        >"$log_path" 2>&1 &
    server_pid=$!
    trap stop_server EXIT
    for _attempt in $(seq 1 180); do
        if curl --fail --silent "http://127.0.0.1:${port}/health" >/dev/null; then
            return
        fi
        kill -0 "$server_pid" 2>/dev/null || die "llama-server exited during startup"
        sleep 1
    done
    die "llama-server did not become healthy"
}

render_report_and_checkpoint() {
    : "${GRID5000_CONFIG_REVISION:?GRID5000_CONFIG_REVISION must be set}"
    : "${GRID5000_DATASET_REVISION:?GRID5000_DATASET_REVISION must be pinned}"
    : "${GRID5000_MODEL_REVISION:?GRID5000_MODEL_REVISION must be pinned}"
    : "${GRID5000_ARTIFACT_PREFIX:?GRID5000_ARTIFACT_PREFIX must be set}"
    uv run --locked python scripts/render_report.py \
        --encoder-dir "$output_dir/encoder" \
        --llm-dir "$output_dir/llm" \
        --output "$output_dir/landuse-logprob-report.tex" \
        --checkpoint "$output_dir/checkpoint.json"
}

case "$backend" in
    encoder)
        run_backend encoder
        ;;
    llm)
        start_server
        run_backend llm
        ;;
    both)
        run_backend encoder
        start_server
        run_backend llm
        render_report_and_checkpoint
        ;;
esac
