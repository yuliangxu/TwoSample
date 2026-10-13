"""Check final case-study inputs, generator assets, and reproduced samples.

This integrity check does not fit BATTS or compare with superseded synthetic
cohorts. The pinned file list is generators/manifest.json.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("icfm", "dt", "d", "mbgan")
TOLERANCES = {"icfm": 1e-6, "dt": 1e-8, "d": 1e-9, "mbgan": 1e-6}


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def verify_bundle():
    """Require every fixed data, checkpoint, tree and vendor file to match."""
    manifest = json.loads((REPO / "generators/manifest.json").read_text())
    verified = {}
    for entry in manifest["files"]:
        path = REPO / entry["path"]
        checksum = sha256(path)
        if checksum != entry["sha256"]:
            raise ValueError(f"Bundled checksum mismatch: {entry['path']}")
        verified[entry["path"]] = checksum
    return verified


def load(path, rows):
    values = np.loadtxt(path, delimiter=",")
    if values.shape != (rows, 123):
        raise ValueError(f"Unexpected matrix shape: {path}: {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError(f"Invalid nonnegative compositions: {path}")
    if np.max(np.abs(values.sum(axis=1) - 1)) > 1e-5:
        raise ValueError(f"Rows do not sum to one: {path}")
    return values


def compare_sample(path, model):
    values = load(path, 1000)
    reference_path = REPO / "data" / f"sample_{model}.csv"
    reference = load(reference_path, 1000)
    difference = float(np.max(np.abs(values - reference)))
    checksum, reference_checksum = sha256(path), sha256(reference_path)
    return {
        "shape": list(values.shape),
        "sha256": checksum,
        "selected_csv_sha256": reference_checksum,
        "exact_csv_match": checksum == reference_checksum,
        "exact_numeric_match": bool(np.array_equal(values, reference)),
        "max_absolute_difference": difference,
        "absolute_tolerance": TOLERANCES[model],
        "max_row_sum_error": float(np.max(np.abs(values.sum(axis=1) - 1))),
        "passed": difference <= TOLERANCES[model],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO / "data")
    parser.add_argument("--output", type=Path, default=REPO / "output/generators/validation")
    parser.add_argument("--check-bundled-hashes", action="store_true",
                        help="Compatibility flag; bundled hashes are always verified")
    args = parser.parse_args()
    verified = verify_bundle()
    real_data = {}
    for split, rows in [("train", 1166), ("test", 292)]:
        path = REPO / "data" / f"sample_{split}.csv"
        values = load(path, rows)
        real_data[split] = {"shape": list(values.shape), "sha256": sha256(path)}
    models = {key: compare_sample(args.data_dir / f"sample_{key}.csv", key)
              for key in MODEL_ORDER}
    if not all(model["passed"] for model in models.values()):
        raise ValueError("A synthetic matrix exceeds its fixed-cohort replay tolerance")
    report = {
        "status": "complete",
        "model_order": list(MODEL_ORDER),
        "bundle_files_verified": len(verified),
        "real_data": real_data,
        "models": models,
        "scope": "Fixed input integrity and numerical replay; not a model-quality or historical-training claim.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "validation.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
