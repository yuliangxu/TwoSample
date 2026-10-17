#!/usr/bin/env python3
"""Independently validate corrected cohort identities, filters, split and values."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
EXPECTED = {"HMP_2019_ibdmdb": 1491, "HMP_2019_t2d": 277,
            "HallAB_2017": 209, "HanniganGD_2017": 75}


def rows(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row.get("source_studies", row.get("study")), row["sample_id"]


def external_path(path, label):
    resolved = Path(path).expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the code repository: {resolved}")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="External runtime data directory")
    parser.add_argument("--raw-source", action="store_true", help="Also reconstruct both matrices independently from pinned sources using R")
    parser.add_argument("--rscript", default="Rscript")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data_dir = external_path(args.data_dir, "Runtime data")
    if args.output is not None:
        args.output = external_path(args.output, "Verification output")
    prov = data_dir / "provenance"
    manifest = json.loads((prov / "cohort_manifest.json").read_text())
    assert manifest["status"] == "verified"
    assert manifest["cohort_id"] == "corrected_four_study_2052_20261015"
    assert (manifest["observed_rows"], manifest["training_rows"], manifest["test_rows"], manifest["taxa"]) == (2052, 1641, 411, 123)
    for name, expected_hash in manifest["files"].items():
        relative = Path(name)
        assert not relative.is_absolute() and ".." not in relative.parts
        assert sha(data_dir / relative) == expected_hash, f"Cohort-bound file changed: {name}"
    order = rows(prov / "cohort_order.csv")
    samples = rows(prov / "samples.csv")
    assert len(order) == len(samples) == len({key(r) for r in samples}) == 2052
    assert len({r["sample_id"] for r in samples}) == 2052
    assert Counter(r["source_studies"] for r in samples) == EXPECTED
    assert [int(r["cohort_row"]) for r in order] == list(range(1, 2053))
    historical_columns = [int(r["historical_filtered_column"]) for r in order]
    assert historical_columns == sorted(set(historical_columns))
    taxa = (prov / "taxa.txt").read_text().splitlines()
    assert len(taxa) == len(set(taxa)) == 123

    collection = rows(prov / "collection_sample_filter.csv")
    metadata = rows(prov / "collection_metadata_ids.csv")
    assert len(collection) == len(metadata) == 22584
    assert {key(r) for r in collection} == {key(r) for r in metadata}
    means = np.asarray([float(r["retained_mean_percent"]) for r in collection])
    total = np.asarray([float(r["total_abundance_percent"]) for r in collection])
    complete = total >= 99
    eligibility = complete & (means >= means.mean())
    assert complete.sum() == 22001 and eligibility.sum() == 16824
    selected = {key(row) for row, flag in zip(collection, eligibility) if flag and row["study"] in EXPECTED}
    assert selected == {key(r) for r in samples} == {key(r) for r in order}
    assert all((r["eligible"] == "TRUE") == bool(flag) for r, flag in zip(collection, eligibility))
    taxon_filter = rows(prov / "collection_taxon_filter.csv")
    assert {r["taxon"] for r in taxon_filter if float(r["mean_abundance_percent"]) >= 0.1} == set(taxa)
    summary = rows(prov / "preprocessing_summary.csv")[0]
    assert abs(float(summary["retained_mean_percent_cutoff"]) - means.mean()) < 1e-12

    train_indices, test_indices = train_test_split(np.arange(2052), test_size=0.2, random_state=1)
    checks = {r["split"]: r for r in rows(prov / "numeric_checksums.csv")}
    matrices = {}
    for split, indices in (("train", train_indices), ("test", test_indices)):
        actual = np.loadtxt(data_dir / f"sample_{split}.csv", delimiter=",")
        assert actual.shape == (len(indices), 123)
        assert np.isfinite(actual).all() and (actual >= 0).all()
        assert np.max(np.abs(actual.sum(axis=1) - 1)) < 1e-7
        assert np.array_equal(actual, actual.astype("<f4").astype(np.float64))
        bound_samples = [r for r in samples if r["split"] == split]
        assert [int(r["csv_row"]) for r in bound_samples] == list(range(1, len(indices) + 1))
        assert [int(r["cohort_row"]) - 1 for r in bound_samples] == indices.tolist()
        assert [key(r) for r in bound_samples] == [key(order[i]) for i in indices]
        numeric = hashlib.md5(actual.astype("<f4").tobytes(order="C")).hexdigest()
        assert numeric == checks[split]["md5_float32_le_row_major"]
        matrices[split] = actual
    assert set(train_indices).isdisjoint(test_indices)
    assert set(train_indices) | set(test_indices) == set(range(2052))

    overlap = json.loads((prov / "participant_overlap.json").read_text())
    if overlap["metadata_available"]:
        participant_rows = rows(prov / "participant_metadata.csv")
        lookup = {(r["study_name"], r["sample_id"]): r["subject_id"] for r in participant_rows}
        assert len(lookup) == len(participant_rows) == 2052
        assert all(r["subject_id"] and r["subject_id"] == lookup[key(r)] for r in samples)
        subjects = {split: {(r["source_studies"], r["subject_id"]) for r in samples if r["split"] == split}
                    for split in ("train", "test")}
        shared = subjects["train"] & subjects["test"]
        assert len(shared) == overlap["shared_participants"]
        assert len(subjects["train"] | subjects["test"]) == overlap["unique_participants"]

    raw_replay = False
    if args.raw_source:
        with tempfile.TemporaryDirectory(prefix="corrected-cohort-replay-") as temporary:
            output = Path(temporary) / "replayed"
            subprocess.run([args.rscript, str(ROOT / "scripts/reprocess_microbiome.R"), "--data-dir", str(data_dir), "--output", str(output)], check=True)
            for split, actual in matrices.items():
                recovered = np.loadtxt(output / f"sample_{split}.csv", delimiter=",")
                assert np.array_equal(actual, recovered), f"Raw source mismatch in {split}"
        raw_replay = True
    result = {"status": "verified", "cohort_id": manifest["cohort_id"],
              "cohort_manifest_sha256": sha(prov / "cohort_manifest.json"),
              "bound_files_verified": len(manifest["files"]),
              "all_eligible_four_study_ids": 2052, "study_counts": EXPECTED,
              "training_shape": list(matrices["train"].shape), "test_shape": list(matrices["test"].shape),
              "split_seed_1_reproduced": True, "raw_source_numeric_replay": raw_replay,
              "participant_overlap": overlap}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
