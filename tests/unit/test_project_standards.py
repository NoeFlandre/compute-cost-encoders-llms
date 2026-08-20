import re
import tomllib
from pathlib import Path

PROJECT_ROOT = Path(__file__).parents[2]


def load_project_metadata() -> dict:
    return tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())


def test_dev_dependency_group_declares_the_quality_toolchain() -> None:
    metadata = load_project_metadata()
    required = {
        "crap4py",
        "import-linter",
        "mkdocs-material",
        "mutmut",
        "pytest",
        "pytest-bdd",
        "pytest-cov",
        "ruff",
        "ty",
    }
    declared = {
        re.split(r"[<>=!~]", dependency.split("[", maxsplit=1)[0], maxsplit=1)[0]
        for dependency in metadata["dependency-groups"]["dev"]
    }

    assert required <= declared


def test_static_analysis_defaults_are_explicit() -> None:
    metadata = load_project_metadata()

    assert metadata["tool"]["ruff"]["target-version"] == "py312"
    assert metadata["tool"]["ruff"]["src"] == ["src", "tests", "scripts"]
    assert metadata["tool"]["ty"]["environment"]["python-version"] == "3.12"
    assert metadata["tool"]["pytest"]["ini_options"]["pythonpath"] == [
        ".",
        "src",
    ]


def test_project_contains_the_apache_license() -> None:
    assert load_project_metadata()["project"]["license"] == "Apache-2.0"
    license_path = PROJECT_ROOT / "LICENSE"
    assert license_path.exists(), "The GitHub repository must include LICENSE."

    license_text = license_path.read_text()
    assert "Apache License" in license_text
    assert "Version 2.0" in license_text
    assert "http://www.apache.org/licenses/LICENSE-2.0" in license_text


def test_mutation_and_crap_limits_are_declared() -> None:
    metadata = load_project_metadata()
    mutmut = metadata["tool"]["mutmut"]

    assert mutmut["source_paths"] == ["src", "scripts"]
    assert mutmut["mutate_only_covered_lines"] is False
    assert mutmut["use_git_change_detection"] is False
    assert mutmut["on_dependency_change"] == "rerun"
    assert "--max-crap 5.99" in (PROJECT_ROOT / "scripts" / "qa.sh").read_text()
