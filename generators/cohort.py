"""Bind fixed generator recipes to externally prepared runtime inputs."""
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def external_path(path, label="Runtime path"):
    path = Path(path).resolve()
    if path == ROOT or ROOT in path.parents:
        raise ValueError(f"{label} must be outside the code repository: {path}")
    return path


def runtime_data(data_dir=None):
    value = data_dir or os.environ.get("TWO_SAMPLE_DATA_DIR")
    if not value:
        raise ValueError("Supply --data-dir or TWO_SAMPLE_DATA_DIR; no data are bundled")
    return external_path(value, "Data directory")


def training_configuration(root=ROOT):
    config = json.loads((Path(root) / "generators/training_config.json").read_text())
    if config["model_order"] != ["icfm", "dt", "d", "mbgan"] or config["generated_rows"] != 1000:
        raise ValueError("The four-generator order and 1000-sample protocol must be retained")
    return config


def cohort_binding(root=ROOT, verify_files=True, data_dir=None):
    data = runtime_data(data_dir)
    config = training_configuration(root)
    manifest = json.loads((data / config["cohort_manifest"]).read_text())
    if manifest.get("status") != "verified" or not manifest.get("cohort_id"):
        raise ValueError("External cohort must pass preprocessing verification first")
    for name in ("training_rows", "test_rows", "taxa"):
        if not isinstance(manifest.get(name), int) or manifest[name] < 2:
            raise ValueError(f"Invalid runtime cohort dimension: {name}")
        config[name] = manifest[name]
    config["cohort_id"] = manifest["cohort_id"]
    config["models"]["dt"]["nodes"] = config["taxa"] - 1
    if manifest["observed_rows"] != config["training_rows"] + config["test_rows"]:
        raise ValueError("Cohort row count differs from the split")
    required = {"sample_train.csv", "sample_test.csv", "provenance/samples.csv",
                "provenance/taxa.txt", "tree/childNode.csv", "tree/treeTaxa.csv", "tree/outTaxa.csv"}
    if not required.issubset(manifest["files"]):
        raise ValueError("Runtime manifest must bind real matrices, identities, taxa and tree assets")
    for relative, digest in manifest["files"].items():
        name = Path(relative)
        if name.is_absolute() or ".." in name.parts:
            raise ValueError(f"Unsafe runtime cohort path: {relative}")
        path = data / name
        if not path.is_file() or not path.resolve().is_relative_to(data):
            raise ValueError(f"Missing or external runtime cohort file: {relative}")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"Invalid runtime hash: {relative}")
        if verify_files and hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Runtime input changed: {relative}")
    return config, manifest


def expected_rows(config):
    return {"train": config["training_rows"], "test": config["test_rows"],
            **{key: config["generated_rows"] for key in config["model_order"]}}
