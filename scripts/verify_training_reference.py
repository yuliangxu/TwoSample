#!/usr/bin/env python3
"""Compare a newly regenerated data bank with the released reference results.

This read-only check uses the Python standard library and requires each bank's
generation manifest, training evidence, and all six composition matrices.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

from prepare_run import MODELS, relative_file, validate_generation

ROOT = Path(__file__).resolve().parents[1]
ROWS = {"train": 1166, "test": 292, **{key: 1000 for key in MODELS}}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1048576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validated_bank(directory):
    inputs = {}
    for key, expected_rows in ROWS.items():
        path = directory / f"sample_{key}.csv"
        count = 0
        with path.open(newline="") as stream:
            for count, row in enumerate(csv.reader(stream), 1):
                values = [float(value) for value in row]
                if (len(values) != 123 or any(not math.isfinite(v) or not 0 <= v <= 1 for v in values)
                        or abs(sum(values) - 1) > 1e-6):
                    raise ValueError(f"Invalid composition in {path}, row {count}")
        if count != expected_rows:
            raise ValueError(f"Wrong row count in {path}: {count}, expected {expected_rows}")
        inputs[path.name] = {"sha256": sha256(path), "shape": [count, 123]}
    manifest = validate_generation(directory / "generation_manifest.json", inputs, ROOT)
    metadata = {
        key: json.loads(relative_file(directory, manifest["models"][key]["training_metadata"]).read_text())
        for key in MODELS
    }
    evidence = json.loads(relative_file(directory, manifest["reproduction_evidence"]).read_text())
    for key in MODELS:
        parameter_hash = metadata[key]["final_parameter_sha256"]
        paired = evidence["models"][key]
        if (not isinstance(parameter_hash, str) or len(parameter_hash) != 64
                or paired["first_parameter_sha256"] != parameter_hash
                or paired["second_parameter_sha256"] != parameter_hash
                or paired["parameter_arrays_identical"] is not True):
            raise ValueError(f"{directory}: {key} parameter evidence is inconsistent")
    return inputs, manifest, metadata


def compare(generated_dir, reference_dir):
    generated, generated_manifest, generated_metadata = validated_bank(generated_dir)
    reference, reference_manifest, reference_metadata = validated_bank(reference_dir)
    for name in reference:
        if generated[name] != reference[name]:
            raise ValueError(f"Released reference mismatch: {name} bytes or shape differ")
    if generated_manifest["source_hashes"] != reference_manifest["source_hashes"]:
        raise ValueError("Released reference mismatch: generator source fingerprints differ")
    models = {}
    for key in MODELS:
        actual, expected = generated_metadata[key], reference_metadata[key]
        for field in ("configuration", "final_parameter_sha256"):
            if actual[field] != expected[field]:
                raise ValueError(f"Released reference mismatch: {key} {field} differs")
        native_actual, native_expected = actual["trainer_metadata"], expected["trainer_metadata"]
        if key == "icfm":
            fields = ("initial_state_sha256", "final_state_sha256")
        elif key == "mbgan":
            native_actual, native_expected = native_actual["completion"], native_expected["completion"]
            fields = ("final_generator_tensor_sha256", "final_critic_tensor_sha256",
                      "final_optimizer_tensor_sha256")
        else:
            fields = ()
        for field in fields:
            if native_actual.get(field) != native_expected.get(field) or not native_expected.get(field):
                raise ValueError(f"Released reference mismatch: {key} native {field} differs or is missing")
        models[key] = {"configuration_matches": True, "final_parameter_sha256": actual["final_parameter_sha256"],
                       "native_hashes_checked": list(fields)}
    return {"status": "passed", "generated_data_dir": str(generated_dir),
            "reference_data_dir": str(reference_dir), "csv_files_matched_exactly": len(reference),
            "model_order": list(MODELS), "models": models,
            "scope": "Exact match to the supplied released reference; no guarantee across hardware or numerical environments."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("generated_data_dir", type=Path)
    parser.add_argument("--reference-dir", type=Path, default=ROOT / "data")
    args = parser.parse_args()
    try:
        result = compare(args.generated_data_dir.resolve(), args.reference_dir.resolve())
    except (OSError, ValueError, KeyError, TypeError) as error:
        print(f"Reference verification failed: {error}", file=sys.stderr)
        print("Exact reproduction is scoped to the recorded hardware and software environment; "
              "different hardware or numerical libraries may change fitted values.", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
