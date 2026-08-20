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


def test_release_workflow_builds_and_attaches_versioned_artifacts() -> None:
    workflow = (PROJECT_ROOT / ".github" / "workflows" / "release.yml").read_text()

    assert 'tags: ["v*.*.*"]' in workflow
    assert "uv build" in workflow
    assert "gh release create" in workflow
