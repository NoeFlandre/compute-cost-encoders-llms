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
