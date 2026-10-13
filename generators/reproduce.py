"""Regenerate the four fixed synthetic cohorts used in the case study.

ICFM and MBGAN replay bundled checkpoints. DT and D refit the public 1,166-row
training matrix. Model order is ICFM, DT, D, MBGAN, as in the report.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .validate import MODEL_ORDER, compare_sample, sha256, verify_bundle

REPO = Path(__file__).resolve().parents[1]
SEEDS = {"icfm": 2, "dt": 1, "d": 1, "mbgan": 256}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "output/generators/reproduced")
    parser.add_argument("--models", nargs="+", choices=MODEL_ORDER, default=list(MODEL_ORDER))
    parser.add_argument("--tensorflow-python", default=sys.executable,
                        help="Python interpreter from the pinned MBGAN environment")
    parser.add_argument("--include-real", action="store_true",
                        help="Copy unchanged real train/test CSVs alongside the generated samples")
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error("Do not repeat model names")
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Output directory must be new or empty")
    verified = verify_bundle()
    out.mkdir(parents=True, exist_ok=True)
    (out / "checkpoints").mkdir()
    (out / "logs").mkdir()
    tf_python = args.tensorflow_python
    if "/" in tf_python:
        # Keep a venv symlink: resolving it can discard the environment.
        tf_python = os.path.abspath(tf_python)
    report = {
        "status": "running",
        "model_order": args.models,
        "models": {},
        "real_training_sha256": sha256(REPO / "data/sample_train.csv"),
        "bundle_files_verified": len(verified),
        "neural_mode": "Fixed recovered-checkpoint replay; original neural fitting membership is unverified.",
    }
    for key in args.models:
        csv_path = out / f"sample_{key}.csv"
        if key in {"d", "dt"}:
            module = "dirichlet" if key == "d" else "dirichlet_tree"
            command = [sys.executable, "-m", f"generators.{module}", "fit-sample",
                       "--train", str(REPO / "data/sample_train.csv"),
                       "--checkpoint", str(out / "checkpoints" / f"{key}.npz"),
                       "--output", str(csv_path), "--n-samples", "1000", "--seed", "1",
                       "--report", str(out / f"{key}_fit.json")]
        elif key == "icfm":
            command = [sys.executable, "-m", "generators.icfm", "sample",
                       "--output", str(csv_path), "--n", "1000", "--seed", "2",
                       "--device", "cpu", "--threads", "2"]
        else:
            command = [tf_python, "-m", "generators.mbgan", "sample",
                       "--output", str(csv_path), "--n", "1000", "--seed", "256",
                       "--batch-size", "32", "--threads", "2", "--device", "cpu"]
        print(f"Generating {key}", flush=True)
        with (out / "logs" / f"{key}.log").open("w") as log:
            subprocess.run(command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=True)
        result = compare_sample(csv_path, key)
        report["models"][key] = {"seed": SEEDS[key], **result}
        (out / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
        if not result["passed"]:
            raise RuntimeError(f"{key} exceeds the fixed-cohort replay tolerance; inspect {out}/logs")
    if args.include_real:
        for split in ["train", "test"]:
            shutil.copyfile(REPO / "data" / f"sample_{split}.csv", out / f"sample_{split}.csv")
    report["status"] = "complete"
    (out / "reproduction.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "COMPLETE").write_text("All requested generators passed fixed-cohort replay checks.\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
