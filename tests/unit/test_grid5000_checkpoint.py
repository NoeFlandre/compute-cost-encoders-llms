import json
from pathlib import Path
from typing import cast

import pytest
from scripts.grid5000.checkpoint_metadata import main, validate_file, validate_metadata

PROJECT_ROOT = Path(__file__).parents[2]


VALID_METADATA = {
    "source_commit": "8a0b93914083e507ffec4aace67ff89ab71dfa5d",
    "config_revision": "sha256:config-revision",
    "seed": 7,
    "dataset_revision": "dataset@revision",
    "model_revision": "model@revision",
    "step": 120,
    "metrics": {"loss": 0.25},
    "complete": True,
    "artifact_uri": ("hf://buckets/NoeFlandre/compute-cost-encoders-llms/runs/example"),
}


def test_complete_checkpoint_metadata_is_accepted() -> None:
    assert validate_metadata(VALID_METADATA) == []


def test_non_object_metadata_is_rejected() -> None:
    assert validate_metadata([]) == ["metadata must be a JSON object"]


def test_non_text_revision_is_reported_as_missing() -> None:
    metadata = {**VALID_METADATA, "source_commit": None}

    errors = validate_metadata(metadata)

    assert "source_commit is required" in errors


def test_incomplete_checkpoint_metadata_reports_resume_safety_errors() -> None:
    metadata = {**VALID_METADATA, "complete": False, "step": -1}

    errors = validate_metadata(metadata)

    assert "complete must be true" in errors
    assert "step must be a non-negative integer" in errors


def test_non_integer_seed_is_rejected() -> None:
    metadata = {**VALID_METADATA, "seed": "seven"}

    errors = validate_metadata(metadata)

    assert "seed must be a non-negative integer" in errors


def test_zero_seed_and_step_are_valid_non_negative_integers() -> None:
    metadata = {**VALID_METADATA, "seed": 0, "step": 0}

    assert validate_metadata(metadata) == []


def test_checkpoint_metadata_reports_missing_integrity_fields() -> None:
    metadata = {**VALID_METADATA, "metrics": {}, "complete": False}

    errors = validate_metadata(metadata)

    assert "metrics must be a non-empty object" in errors
    assert "complete must be true" in errors


def test_checkpoint_metadata_requires_the_project_bucket() -> None:
    metadata = {**VALID_METADATA, "artifact_uri": "s3://unapproved/location"}

    errors = validate_metadata(metadata)

    assert "artifact_uri must use the project Hugging Face bucket" in errors


def test_empty_artifact_uri_is_only_reported_as_required() -> None:
    metadata = {**VALID_METADATA, "artifact_uri": ""}

    errors = validate_metadata(metadata)

    assert "artifact_uri is required" in errors
    assert "artifact_uri must use the project Hugging Face bucket" not in errors


def test_checkpoint_metadata_file_reports_invalid_json(tmp_path: Path) -> None:
    metadata_path = tmp_path / "checkpoint.json"
    metadata_path.write_text("{")

    errors = validate_file(metadata_path)

    assert errors[0].startswith("could not read JSON metadata:")


def test_validate_file_passes_explicit_utf8_encoding() -> None:
    class RecordingPath:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def read_text(self, **kwargs: object) -> str:
            self.calls.append(kwargs)
            return json.dumps(VALID_METADATA)

    path = RecordingPath()

    assert validate_file(cast(Path, path)) == []
    assert path.calls == [{"encoding": "utf-8"}]


def test_checkpoint_metadata_cli_accepts_a_valid_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata_path = tmp_path / "checkpoint.json"
    metadata_path.write_text(json.dumps(VALID_METADATA))

    assert main([str(metadata_path)]) == 0
    assert "Valid checkpoint metadata" in capsys.readouterr().out


def test_checkpoint_metadata_cli_rejects_an_incomplete_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    metadata_path = tmp_path / "checkpoint.json"
    metadata_path.write_text(json.dumps({**VALID_METADATA, "complete": False}))

    assert main([str(metadata_path)]) == 1
    assert "complete must be true" in capsys.readouterr().err


def test_checkpoint_metadata_cli_help_describes_the_validator(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit):
        main(["--help"])

    assert "Validate the metadata" in capsys.readouterr().out


def test_checkpoint_metadata_cli_contract_is_documented() -> None:
    assert (PROJECT_ROOT / "docs" / "grid5000.md").exists(), (
        "Grid5000 operating instructions must be public documentation."
    )
