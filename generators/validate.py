"""Verify external cohort/tree assets and newly generated runtime evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from .cohort import cohort_binding, expected_rows, external_path, runtime_data
from .tree import TreeTransform

REPO = Path(__file__).resolve().parents[1]
MODEL_ORDER = ("icfm", "dt", "d", "mbgan")


def sha256(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load(path, rows, taxa):
    values = np.loadtxt(path, delimiter=",")
    if values.shape != (rows, taxa):
        raise ValueError(f"Unexpected matrix shape: {path}: {values.shape}")
    if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
        raise ValueError(f"Invalid nonnegative compositions: {path}")
    if np.max(np.abs(values.sum(axis=1) - 1)) > 1e-6:
        raise ValueError(f"Rows do not sum to one: {path}")
    return values


def verify_bundle(data_dir=None):
    """Verify user-supplied inputs; the repository contains no data or tree."""
    data = runtime_data(data_dir)
    config, cohort = cohort_binding(REPO, data_dir=data)
    for split in ("train", "test"):
        load(data / f"sample_{split}.csv", config[f"{split if split == 'test' else 'training'}_rows"], config["taxa"])
    tree = TreeTransform(data / "tree")
    taxa = (data / "provenance/taxa.txt").read_text().splitlines()
    if tree.out_taxa != taxa or tree.n_taxa != config["taxa"]:
        raise ValueError("Runtime tree output order differs from the matrix taxonomy")
    return cohort["files"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Optional new external verification JSON")
    parser.add_argument("--inputs-only", action="store_true")
    args = parser.parse_args()
    data = runtime_data(args.data_dir)
    verified = verify_bundle(data)
    config, _ = cohort_binding(REPO, data_dir=data)
    report = dict(status="verified", cohort_id=config["cohort_id"], runtime_files_verified=len(verified))
    if not args.inputs_only:
        inputs = {}
        for key, count in expected_rows(config).items():
            path = data / f"sample_{key}.csv"
            values = load(path, count, config["taxa"])
            inputs[path.name] = dict(shape=list(values.shape), sha256=sha256(path))
        sys.path.insert(0, str(REPO / "scripts"))
        from prepare_run import validate_generation
        manifest = validate_generation(data / "generation_manifest.json", inputs, REPO, data_dir=data)
        report.update(models=manifest["models"], independent_full_training_runs=2)
    if args.output:
        target = external_path(args.output, "Verification output")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
