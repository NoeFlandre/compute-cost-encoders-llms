from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]
GRID5000_DIR = PROJECT_ROOT / "scripts" / "grid5000"


def read_guardrail(name: str) -> str:
    return (GRID5000_DIR / name).read_text()


def test_submission_checks_policy_and_rejects_duplicate_jobs() -> None:
    script = read_guardrail("submit.sh")

    assert script.count("usagepolicycheck -t") >= 2
    assert "oarstat -u" in script
    assert "oarsub" in script
    assert "host=1/gpu=1,walltime=0:30" in script


def test_submission_can_scope_queue_and_gpu_properties() -> None:
    script = read_guardrail("submit.sh")

    assert "GRID5000_QUEUE" in script
    assert "GRID5000_PROPERTIES" in script


def test_compute_runner_requires_reserved_nodes_and_locked_uv() -> None:
    script = read_guardrail("run.sh")

    for required in (
        "OAR_JOB_ID",
        "OAR_NODEFILE",
        "GRID5000_CONFIG_PATH",
        "GRID5000_DATASET_REVISION",
        "GRID5000_MODEL_REVISION",
        "GRID5000_CHECKPOINT_INTERVAL",
        "uv",
        "run",
        "--locked",
    ):
        assert required in script


def test_artifact_publishing_requires_explicit_confirmation() -> None:
    script = read_guardrail("publish.sh")

    assert "GRID5000_PUBLISH_CONFIRM" in script
    assert "hf buckets sync" in script
    assert "HF_BUCKET_URI" in script


def test_benchmark_entrypoint_requires_pinned_runtime_and_runs_locked_uv() -> None:
    script = read_guardrail("benchmark.sh")

    for required in (
        "GRID5000_LLM_REVISION",
        "GRID5000_QWEN_MODEL_PATH",
        "LLAMA_SERVER_BIN",
        "uv run --locked",
        "--n-gpu-layers",
        "cache_prompt",
        "hf download",
        "--checkpoint",
        "GRID5000_ARTIFACT_PREFIX",
    ):
        assert required in script

    assert script.count("uv run --locked --extra benchmark --no-dev") == 3


def test_benchmark_entrypoint_exposes_the_checkout_package_to_uv() -> None:
    script = read_guardrail("benchmark.sh")

    assert 'PYTHONPATH="$project_root/src:$project_root' in script


def test_grid5000_image_pins_llama_cpp_and_exposes_cuda_server() -> None:
    dockerfile = (PROJECT_ROOT / "Dockerfile.grid5000").read_text()

    assert "6503355df0eb4f65875012523263c302fe0088c1" in dockerfile
    assert "llama-server" in dockerfile
    assert "CUDA" in dockerfile


def test_release_workflow_builds_and_attaches_versioned_artifacts() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text()

    assert 'tags: ["v*.*.*"]' in workflow
    assert "uv build" in workflow
    assert "gh release create" in workflow
