from pathlib import Path

from pytest_bdd import given, scenarios, then, when

PROJECT_ROOT = Path(__file__).parents[2]

scenarios("features/project_quality.feature")


@given("the project quality runner", target_fixture="qa_runner")
def project_quality_runner() -> Path:
    return PROJECT_ROOT / "scripts" / "qa.sh"


@when("I inspect the declared QA stages", target_fixture="qa_stages")
def inspect_declared_qa_stages(qa_runner: Path) -> list[str]:
    if not qa_runner.exists():
        return []

    prefix = "# QA_STAGE: "
    return [
        line.removeprefix(prefix)
        for line in qa_runner.read_text().splitlines()
        if line.startswith(prefix)
    ]


@then(
    "the stages are ordered as ruff, ty, tests, acceptance tests, "
    "architecture checks, CRAP, mutation tests"
)
def assert_qa_stage_order(qa_stages: list[str]) -> None:
    assert qa_stages == [
        "ruff",
        "ty",
        "tests",
        "acceptance tests",
        "architecture checks",
        "CRAP",
        "mutation tests",
    ]


def test_qa_runs_the_integration_test_layer() -> None:
    qa_script = (PROJECT_ROOT / "scripts" / "qa.sh").read_text()

    assert "pytest tests/integration" in qa_script


def test_qa_requires_a_nonempty_all_killed_mutation_inventory() -> None:
    qa_script = (PROJECT_ROOT / "scripts" / "qa.sh").read_text()

    assert 'mutation_results="$(uv run mutmut results --all true)"' in qa_script
    assert 'if [[ -z "$mutation_results" ]]; then' in qa_script
    assert "grep -Evq ': killed$'" in qa_script


def test_ci_declares_package_install_and_cli_smoke_tests() -> None:
    for workflow_name in ("qa.yml", "release.yml"):
        workflow = (PROJECT_ROOT / ".github" / "workflows" / workflow_name).read_text()
        assert "Package, install, and CLI smoke tests" in workflow
        assert "uv build --out-dir" in workflow
        assert "--isolated --no-project --with" in workflow
        assert "benchmark.cli --help" in workflow
