"""Inspect canonical artifacts and explicitly migrate the two legacy formats."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .artifact import (
    ARCHITECTURES,
    TOKENIZER_FILES,
    ArtifactError,
    seal_directory,
    validate_directory,
)


def add_provenance_arguments(parser) -> None:
    parser.add_argument("--base-model-id", required=True)
    parser.add_argument(
        "--base-revision", required=True, help="Immutable base model revision"
    )
    parser.add_argument(
        "--model-license", required=True, help="Base model license identifier or URL"
    )
    parser.add_argument(
        "--calibration-record",
        type=Path,
        help="JSON record with dataset revision/license, seed and calibration parameters",
    )


def calibration_record(args) -> dict:
    if args.calibration_record:
        value = json.loads(args.calibration_record.read_text())
        if not isinstance(value, dict):
            raise ArtifactError("Calibration record must be a JSON object")
        return value
    return {
        "status": "not-recorded",
        "note": "No calibration evidence supplied; no accuracy claim is made",
    }


def migrate(
    source: Path,
    destination: Path,
    *,
    original_hidden_size: int,
    base_model_id: str,
    base_revision: str,
    model_license: str,
    calibration: dict,
) -> None:
    if destination.exists():
        raise ArtifactError(
            "Destination already exists; migration never overwrites artifacts"
        )
    config = json.loads((source / "config.json").read_text())
    if "slicegpt_artifact" in config:
        raise ArtifactError(
            "Input already has artifact metadata; use validate, not a legacy migration"
        )
    architectures = config.get("architectures", [])
    if len(architectures) != 1 or architectures[0] not in ARCHITECTURES:
        raise ArtifactError("Not a supported legacy SliceGPT checkpoint")
    family = ARCHITECTURES[architectures[0]]
    if config.get("model_type") not in {family, f"slicegpt_{family}"}:
        raise ArtifactError("Unknown legacy model_type")
    config["model_type"] = family
    config.pop("auto_map", None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".slicegpt-migrate-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "model"
        staging.mkdir()
        (staging / "config.json").write_text(json.dumps(config, indent=2))
        for name in ("model.safetensors", *TOKENIZER_FILES):
            if (source / name).is_file():
                shutil.copyfile(source / name, staging / name)
        seal_directory(
            staging,
            original_hidden_size=original_hidden_size,
            base_model_id=base_model_id,
            base_revision=base_revision,
            model_license=model_license,
            calibration=calibration,
        )
        staging.rename(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate")
    validate.add_argument("path", type=Path)
    migration = commands.add_parser("migrate")
    migration.add_argument("source", type=Path)
    migration.add_argument("destination", type=Path)
    migration.add_argument("--original-hidden-size", type=int, required=True)
    add_provenance_arguments(migration)
    args = parser.parse_args()
    if args.command == "validate":
        config = validate_directory(args.path)
        print(
            json.dumps(
                {
                    "schema_version": config["slicegpt_artifact"]["schema_version"],
                    "architecture": config["architectures"][0],
                    "valid": True,
                }
            )
        )
    else:
        migrate(
            args.source,
            args.destination,
            original_hidden_size=args.original_hidden_size,
            base_model_id=args.base_model_id,
            base_revision=args.base_revision,
            model_license=args.model_license,
            calibration=calibration_record(args),
        )
        print(f"Migrated and validated {args.destination}")


if __name__ == "__main__":
    main()
