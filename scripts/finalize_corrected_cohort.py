#!/usr/bin/env python3
"""Build runtime-only corrected data from an external full-collection preparation."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
COHORT_ID = "corrected_four_study_2052_20261015"
EXPECTED_COUNTS = {"HMP_2019_ibdmdb": 1491, "HMP_2019_t2d": 277,
                   "HallAB_2017": 209, "HanniganGD_2017": 75}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def external_path(path, label):
    resolved = Path(path).expanduser().resolve()
    if resolved == ROOT or ROOT in resolved.parents:
        raise ValueError(f"{label} must be outside the code repository: {resolved}")
    return resolved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("preparation", type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New runtime data directory (never the source checkout)")
    parser.add_argument("--tree-dir", required=True, type=Path, help="External exact childNode/treeTaxa/outTaxa CSV assets")
    parser.add_argument("--legacy-reference", type=Path, help="Optional external normalized taxa-by-observation CSV for an extra value check")
    parser.add_argument("--participant-metadata", required=True, type=Path,
                        help="CSV with study_name, sample_id, subject_id joined by IDs")
    args = parser.parse_args()
    prep = external_path(args.preparation, "Preparation input")
    output = external_path(args.output, "Runtime data output")
    args.tree_dir = external_path(args.tree_dir, "Tree input")
    args.participant_metadata = external_path(args.participant_metadata, "Participant metadata")
    if args.legacy_reference is not None:
        args.legacy_reference = external_path(args.legacy_reference, "Legacy reference")
    if output.exists():
        raise FileExistsError("Choose a new runtime data directory: " + str(output))
    prov = output / "provenance"
    assert (prep / "PREPARATION_COMPLETE").is_file()
    order = read_csv(prep / "cohort_order.csv")
    assert len(order) == 2052 and len({r["sample_id"] for r in order}) == 2052
    x64 = np.fromfile(prep / "cohort_normalized_float64_le.bin", dtype="<f8").reshape(2052, 123)
    assert np.isfinite(x64).all() and (x64 >= 0).all()
    assert np.max(np.abs(x64.sum(axis=1) - 1)) < 1e-12
    x = x64.astype("<f4")

    taxa = (prep / "taxa.txt").read_text().splitlines()
    assert len(taxa) == len(set(taxa)) == 123
    tree_dir = args.tree_dir.resolve()
    tree_names = ("childNode.csv", "treeTaxa.csv", "outTaxa.csv")
    tree_taxa = [row[1] for row in list(csv.reader((tree_dir / "treeTaxa.csv").open()))[1:]]
    output_taxa = [row[1] for row in list(csv.reader((tree_dir / "outTaxa.csv").open()))[1:]]
    assert output_taxa == taxa and len(tree_taxa) == len(set(tree_taxa)) == 123
    assert set(tree_taxa) == set(taxa)
    children = list(csv.reader((tree_dir / "childNode.csv").open()))[1:]
    assert len(children) == 122
    seen = []
    for index, row in enumerate(children, 124):
        assert len(row) == 3 and row[0] == f"Node_{index}"
        for name in row[1:]:
            assert name.startswith("Node_")
            node = int(name.removeprefix("Node_"))
            assert 1 <= node <= 245 and (node <= 123 or node > index)
            seen.append(node)
    assert sorted(seen) == [i for i in range(1, 246) if i != 124]

    # Optional extra historical value check. Selection always uses source IDs.
    legacy = args.legacy_reference
    legacy_matches = False
    if legacy is not None:
        with legacy.open(newline="") as stream:
            reader = csv.reader(stream)
            header = next(reader)
            legacy_ids = header[1:]
            positions = {sid: j for j, sid in enumerate(legacy_ids)}
            selected_columns = [positions[row["sample_id"]] for row in order]
            legacy_taxa, values = [], []
            for row in reader:
                legacy_taxa.append(row[0])
                values.append([float(row[j + 1]) for j in selected_columns])
        assert legacy_taxa == taxa
        reference = np.asarray(values, dtype="<f4").T
        assert np.array_equal(x, reference), "Raw-cache reconstruction differs from original source values"

        legacy_matches = True

    prov.mkdir(parents=True)
    (output / "tree").mkdir()
    for name in tree_names:
        shutil.copyfile(tree_dir / name, output / "tree" / name)
    for source in prep.iterdir():
        if source.is_file() and source.name != "cohort_normalized_float64_le.bin":
            shutil.copyfile(source, prov / source.name)
    (prov / "tree_source.json").write_text(json.dumps({
        "source_directory": str(tree_dir),
        "files": {name: digest(tree_dir / name) for name in tree_names},
        "taxa": 123, "internal_nodes": 122, "validated": True}, indent=2) + "\n")

    participant_lookup = {}
    if args.participant_metadata:
        participant_rows = read_csv(args.participant_metadata)
        for row in participant_rows:
            key = (row["study_name"], row["sample_id"])
            assert key not in participant_lookup, f"Duplicated metadata key: {key}"
            participant_lookup[key] = row["subject_id"]
        assert all(participant_lookup.get((r["source_studies"], r["sample_id"])) for r in order)
        write_csv(prov / "participant_metadata.csv", [
            {"study_name": row["source_studies"], "sample_id": row["sample_id"],
             "subject_id": participant_lookup[(row["source_studies"], row["sample_id"])]}
            for row in order])
        (prov / "participant_metadata_source.json").write_text(json.dumps({
            "source": str(args.participant_metadata.resolve()),
            "source_sha256": digest(args.participant_metadata),
            "source_rows": len(participant_rows),
            "selected_rows": len(order),
            "join_keys": ["study_name", "sample_id"],
            "participant_key": ["study_name", "subject_id"]}, indent=2) + "\n")

    train, test = train_test_split(np.arange(len(order)), test_size=0.2, random_state=1)
    assert len(train) == 1641 and len(test) == 411
    manifests, numeric = [], []
    for split, indices in (("train", train), ("test", test)):
        values = x[indices]
        np.savetxt(output / f"sample_{split}.csv", values.astype(np.float64), delimiter=",", fmt="%.17g")
        numeric.append(dict(split=split, rows=len(indices), columns=len(taxa),
            md5_float32_le_row_major=hashlib.md5(values.astype("<f4").tobytes(order="C")).hexdigest()))
        for row_number, index in enumerate(indices, 1):
            row = order[int(index)]
            manifests.append(dict(split=split, csv_row=row_number, sample_id=row["sample_id"],
                source_studies=row["source_studies"], exact_candidates=1, source_samples=row["sample_id"],
                cohort_row=int(index) + 1, source_resource=row["source_resource"],
                subject_id=participant_lookup.get((row["source_studies"], row["sample_id"]), "")))
    write_csv(prov / "samples.csv", manifests)
    write_csv(prov / "numeric_checksums.csv", numeric)
    for name in ("collection_sample_filter.csv", "collection_taxon_filter.csv",
                 "collection_resource_counts.csv", "preprocessing_summary.csv"):
        shutil.copyfile(prep / name, prov / name)
    shutil.copyfile(prep / "cohort_order.csv", prov / "cohort_order.csv")
    study_counts = []
    for study, total in EXPECTED_COUNTS.items():
        rows = [r for r in manifests if r["source_studies"] == study]
        assert len(rows) == total
        study_counts.append(dict(study=study, all=total,
            train=sum(r["split"] == "train" for r in rows),
            test=sum(r["split"] == "test" for r in rows)))
    write_csv(prov / "study_counts.csv", study_counts)

    overlap = {"metadata_available": bool(participant_lookup), "split_unit": "observation",
               "participant_disjoint_split": False,
               "interpretation": "The split is by observation; repeated observations from a participant may occur in both sets."}
    if participant_lookup:
        def subjects(split):
            return {(r["source_studies"], r["subject_id"]) for r in manifests if r["split"] == split}
        shared = subjects("train") & subjects("test")
        overlap.update(unique_participants=len(subjects("train") | subjects("test")),
                       shared_participants=len(shared),
                       test_observations_with_training_participant=sum(
                           r["split"] == "test" and (r["source_studies"], r["subject_id"]) in shared
                           for r in manifests),
                       by_study={study: sum(key[0] == study for key in shared) for study in EXPECTED_COUNTS})
    else:
        overlap["limitation"] = "Participant metadata was unavailable during this preparation; overlap counts are not claimed."
    (prov / "participant_overlap.json").write_text(json.dumps(overlap, indent=2) + "\n")
    manifest = {
        "status": "verified", "cohort_id": COHORT_ID,
        "observed_rows": 2052, "training_rows": 1641, "test_rows": 411, "taxa": 123,
        "study_counts": EXPECTED_COUNTS,
        "split": {"method": "sklearn.model_selection.train_test_split", "test_size": 0.2,
                  "random_state": 1, "unit": "observation", "order": "provenance/cohort_order.csv"},
        "preprocessing": {"scope": "93-study collection before selecting four studies and splitting",
            "minimum_total_abundance_percent": 99,
            "minimum_taxon_mean_percent_among_complete_observations": 0.1,
            "minimum_retained_mean": "mean retained-taxon abundance across all 22584 collection observations",
            "zero_count_filter": None, "normalization": "divide each retained row by its sum in float64, then round to float32"},
        "source_verification": {"all_93_resource_sha256_verified": True,
            "all_2052_eligible_ids_selected": True, "source_study_sample_join": "study and sample_id",
            "raw_cache_vs_historical_filtered_matrix_float32_exact": legacy_matches,
            "verified_values": int(x.size)},
        "participant_metadata_available": bool(participant_lookup),
        "files": {}
    }
    bound = [output / "sample_train.csv", output / "sample_test.csv"]
    bound += [output / "tree" / name for name in tree_names]
    bound += [p for p in sorted(prov.iterdir()) if p.is_file() and p.name not in {"cohort_manifest.json", "README.md"}]
    manifest["files"] = {str(p.relative_to(output)): digest(p) for p in bound}
    (prov / "cohort_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"cohort_id": COHORT_ID, "study_counts": study_counts,
                      "participant_overlap": overlap, "input_sha256": {str(p.relative_to(output)): digest(p) for p in bound[:2]}}, indent=2))


if __name__ == "__main__":
    main()
