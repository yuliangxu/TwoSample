"""Fit every generator twice from the real training data and verify reproduction.

No existing model checkpoint is an input. Each fit creates intermediate weights
used by its own sampling command. Full independent repeats must agree exactly.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
from .validate import MODEL_ORDER, load, sha256, verify_bundle

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / "generators/training_config.json"


def read(path):
    return json.loads(Path(path).read_text())


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def source_hashes():
    paths = [p for p in (REPO / "generators").rglob("*") if p.is_file()
             and "__pycache__" not in p.parts and "checkpoints" not in p.parts
             and p.suffix in {".py", ".json", ".csv", ".txt"}]
    return {str(p.relative_to(REPO)): sha256(p) for p in sorted(paths)}


def run_command(command, log, overrides=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2",
               TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1", PYTHONHASHSEED="0",
               CUDA_VISIBLE_DEVICES="")
    if overrides:
        env.update(overrides)
    with log.open("x") as f:
        subprocess.run([str(x) for x in command], cwd=REPO, env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True)


def single_run(out, tf_python):
    if out.exists():
        raise FileExistsError(f"Training requires a new directory: {out}")
    verify_bundle()
    out.mkdir(parents=True)
    config, sources = read(CONFIG), source_hashes()
    train = REPO / "data/sample_train.csv"
    train_sha = sha256(train)
    write(out / "run_inputs.json", {"configuration": config, "source_hashes": sources,
          "training_input_sha256": train_sha, "pretrained_model_inputs": [],
          "execution_environment": {name: os.environ.get(name) for name in
              ("CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "LD_LIBRARY_PATH")}})
    started = time.monotonic()
    for key in MODEL_ORDER:
        cfg, dest = config["models"][key], out / key
        sample = out / f"sample_{key}.csv"
        print(f"{out.name}: fitting {key} from training data", flush=True)
        if key == "icfm":
            run_command([sys.executable, "-m", "generators.icfm", "train", "--train", train,
                         "--output", dest / "model.pt", "--steps", cfg["steps"],
                         "--batch-size", cfg["batch_size"], "--lr", cfg["learning_rate"],
                         "--sigma", cfg["sigma"], "--zero-mass", cfg["zero_mass"],
                         "--clip-logratio", cfg["clip_logratio"], "--seed", cfg["training_seed"],
                         "--device", "cpu", "--threads", 2], out / "logs/icfm_train.log")
            run_command([sys.executable, "-m", "generators.icfm", "sample", "--checkpoint",
                         dest / "model.pt", "--output", sample, "--n", 1000,
                         "--seed", cfg["sampling_seed"], "--device", "cpu", "--threads", 2,
                         "--atol", cfg["sampling_atol"], "--rtol", cfg["sampling_rtol"]],
                        out / "logs/icfm_sample.log")
        elif key in {"d", "dt"}:
            module = "dirichlet" if key == "d" else "dirichlet_tree"
            run_command([sys.executable, "-m", f"generators.{module}", "fit-sample", "--train", train,
                         "--checkpoint", dest / "model.npz", "--output", sample, "--n-samples", 1000,
                         "--seed", cfg["sampling_seed"], "--report", dest / "training.json"],
                        out / f"logs/{key}_fit.log")
        else:
            run_command([tf_python, "-m", "generators.mbgan", "train", "--train", train,
                         "--taxonomy", REPO / "data/provenance/taxa.txt", "--output-dir", dest,
                         "--iterations", cfg["iterations"], "--batch-size", cfg["batch_size"],
                         "--n-critic", cfg["critic_updates"], "--seed", cfg["training_seed"],
                         "--device", cfg["device"], "--threads", cfg["threads"],
                         "--callback-interval", cfg["callback_interval"]], out / "logs/mbgan_train.log")
            run_command([tf_python, "-m", "generators.mbgan", "sample", "--checkpoint",
                         dest / "generator.npz", "--output", sample, "--n", 1000,
                         "--seed", cfg["sampling_seed"], "--batch-size", 32,
                         "--device", cfg["device"], "--threads", cfg["threads"]], out / "logs/mbgan_sample.log")
        load(sample, 1000)
        print(f"{out.name}: completed {key}: {sha256(sample)}", flush=True)
    if sources != source_hashes() or train_sha != sha256(train):
        raise RuntimeError("Sources or training data changed during fitting")
    write(out / "completion.json", {"completed": True, "seconds": time.monotonic() - started,
          "outputs": {k: sha256(out / f"sample_{k}.csv") for k in MODEL_ORDER},
          "training_input_sha256": train_sha, "source_hashes": sources})


def native_metadata(replica, key, cfg, train_sha):
    if key == "icfm":
        meta = read(replica / "icfm/model.pt.metadata.json")
        if (meta["training_csv_sha256"] != train_sha or meta["steps"] != cfg["steps"]
                or meta["seed"] != cfg["training_seed"] or not meta["parameters_changed"]
                or meta["status"] != "complete" or meta["batch_size"] != cfg["batch_size"]
                or meta["lr"] != cfg["learning_rate"] or meta["sigma"] != cfg["sigma"]
                or meta["device"] != cfg["device"] or meta["threads"] != cfg["threads"]
                or meta["initial_state_sha256"] == meta["final_state_sha256"]):
            raise RuntimeError("ICFM training provenance mismatch")
        return meta, meta["final_state_sha256"]
    if key in {"dt", "d"}:
        meta = read(replica / key / "training.json")
        verified = meta["all_parameters_verified"] if key == "d" else meta["all_nodes_verified"]
        if (meta["training_file_sha256"] != train_sha or not verified
                or meta["max_equation_residual"] >= 1e-8
                or meta["training_shape"] != [1166, 123]
                or meta["sampling"]["seed"] != cfg["sampling_seed"]):
            raise RuntimeError(f"{key}: training input mismatch")
        with np.load(replica / key / "model.npz", allow_pickle=False) as z:
            state = hashlib.sha256(z["parameters"].tobytes()).hexdigest()
        return meta, state
    native = read(replica / "mbgan/training_config.json")
    completion = read(replica / "mbgan/completion.json")
    if (native["train_sha256"] != train_sha or native["iterations"] != cfg["iterations"]
            or native["seed"] != cfg["training_seed"] or completion["status"] != "completed"
            or native["initialization"] != "random" or native["test_data_used"]
            or native["batch_size"] != cfg["batch_size"] or native["n_critic"] != cfg["critic_updates"]
            or native["learning_rate"] != cfg["learning_rate"] or native["device"] != cfg["device"]
            or native["threads"] != cfg["threads"] or not completion["all_step_losses_finite"]
            or not completion["all_checkpoint_tensors_finite"]
            or native["variant"] != cfg["variant"]
            or native["effective_gradient_penalty"]
            or native["callback_rng_policy"] != cfg["callback_rng_policy"]
            or native["callback_interval"] != cfg["callback_interval"]
            or completion["iterations"] != cfg["iterations"]
            or completion["max_generator_parameter_change"] <= 0
            or completion["max_critic_parameter_change"] <= 0):
        raise RuntimeError("MBGAN training provenance mismatch")
    with np.load(replica / "mbgan/generator.npz", allow_pickle=False) as z:
        h = hashlib.sha256()
        for name in sorted(n for n in z.files if n.startswith("weight_")):
            a = z[name]
            h.update(name.encode()); h.update(str(a.dtype).encode())
            h.update(str(a.shape).encode()); h.update(a.tobytes())
    return {"configuration": native, "completion": completion}, h.hexdigest()


def finalize(out):
    config, sources = read(CONFIG), source_hashes()
    train_sha = sha256(REPO / "data/sample_train.csv")
    replicas = [out / f"replicate_{i}" for i in (1, 2)]
    for replica in replicas:
        complete = read(replica / "completion.json")
        if not complete["completed"] or complete["source_hashes"] != sources:
            raise RuntimeError("Incomplete replicate or changed training sources")
    a_meta, b_meta = [read(r / "icfm/model.pt.metadata.json") for r in replicas]
    if (a_meta["losses"] != b_meta["losses"]
            or a_meta["initial_state_sha256"] != b_meta["initial_state_sha256"]):
        raise RuntimeError("ICFM independent training trajectories differ")
    subprocess.run([sys.executable, "-m", "generators.validate_mbgan_training",
                    str(replicas[0] / "mbgan"), str(replicas[1] / "mbgan"),
                    "--samples-a", str(replicas[0] / "sample_mbgan.csv"),
                    "--samples-b", str(replicas[1] / "sample_mbgan.csv"),
                    "--output", str(out / "mbgan_independent_validation.json")], cwd=REPO, check=True)
    data = out / "data"
    data.mkdir()
    for split in ("train", "test"):
        shutil.copy2(REPO / "data" / f"sample_{split}.csv", data)
    manifest = {"schema_version": 1, "mode": "from_training", "model_order": list(MODEL_ORDER),
        "training_input": {"sha256": train_sha, "shape": [1166, 123]},
        "test_input_sha256": sha256(data / "sample_test.csv"),
        "test_used_for_training": False, "test_used_for_seed_or_checkpoint_selection": False,
        "models": {}, "source_hashes": sources, "validation_artifacts": {},
        "reproduction_evidence": "evidence/reproduction.json",
        "mbgan_variant_choice": config["mbgan_variant_choice"]}
    reproduction = {"status": "complete", "method": "independent_rerun", "models": {}}
    for key in MODEL_ORDER:
        paths = [replica / f"sample_{key}.csv" for replica in replicas]
        a, b = [load(p, 1000) for p in paths]
        hashes = [sha256(p) for p in paths]
        metadata = [native_metadata(replica, key, config["models"][key], train_sha) for replica in replicas]
        if hashes[0] != hashes[1] or not np.array_equal(a, b) or metadata[0][1] != metadata[1][1]:
            raise RuntimeError(f"{key}: independent training runs did not reproduce exactly")
        shutil.copy2(paths[0], data)
        cfg = config["models"][key]
        complete = {"completed": True, "completed_steps": cfg.get("steps", cfg.get("iterations"))}
        initialization = "random" if key in {"icfm", "mbgan"} else "fit_from_training"
        normalized = {"model": key, "training_input_sha256": train_sha,
            "trained_from_scratch": True, "test_used_for_training": False, "test_used_for_seed_or_checkpoint_selection": False,
            "initialization": initialization, "configuration": cfg, "completion": complete,
            "trainer_metadata": metadata[0][0], "final_parameter_sha256": metadata[0][1]}
        evidence_path = f"evidence/{key}_training.json"
        write(data / evidence_path, normalized)
        manifest["validation_artifacts"][evidence_path] = sha256(data / evidence_path)
        reproduction["models"][key] = {"first_output_sha256": hashes[0], "second_output_sha256": hashes[1],
            "matched": True, "method": "independent_rerun", "exact_numeric_match": True,
            "max_absolute_difference": 0.0, "first_parameter_sha256": metadata[0][1],
            "second_parameter_sha256": metadata[1][1], "parameter_arrays_identical": True}
        manifest["models"][key] = {"initialization": initialization, "trained_from_scratch": True,
            "training_input_sha256": train_sha, "configuration": cfg, "completion": complete,
            "output": {"sha256": hashes[0], "shape": [1000, 123]}, "training_metadata": evidence_path,
            "verification": {"passed": True, "method": "independent_rerun", "output_sha256": hashes[1]}}
    write(data / "evidence/reproduction.json", reproduction)
    manifest["validation_artifacts"]["evidence/reproduction.json"] = sha256(data / "evidence/reproduction.json")
    runtime = os.environ.get("TWO_SAMPLE_RUNTIME_PROVENANCE")
    if runtime:
        target = data / "evidence/runtime_environment.json"
        shutil.copy2(runtime, target)
        manifest["validation_artifacts"]["evidence/runtime_environment.json"] = sha256(target)
    shutil.copy2(out / "mbgan_independent_validation.json", data / "evidence/mbgan_independent_validation.json")
    manifest["validation_artifacts"]["evidence/mbgan_independent_validation.json"] = sha256(
        data / "evidence/mbgan_independent_validation.json")
    write(data / "generation_manifest.json", manifest)
    for name in sources:
        target = out / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(REPO / name, target)
    (out / "COMPLETE").write_text("All four models fitted twice; parameters and samples match exactly.\n")
    print(f"Verified data and training provenance: {data}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO / "output/from_training")
    parser.add_argument("--tensorflow-python", default=sys.executable)
    parser.add_argument("--replica-workers", type=int, choices=(1, 2), default=2,
                        help="Concurrent independent CPU fits (default: 2).")
    parser.add_argument("--single-run", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="Verify two completed training runs without repeating them")
    args = parser.parse_args()
    tf_python = os.path.abspath(args.tensorflow_python) if "/" in args.tensorflow_python else args.tensorflow_python
    if args.single_run:
        single_run(args.single_run.resolve(), tf_python)
        return
    out = args.output.resolve()
    if args.finalize_existing:
        finalize(out)
        return
    if out.exists():
        parser.error("Output must be a new directory")
    verify_bundle()
    config = read(CONFIG)
    out.mkdir(parents=True)
    initial_sources = source_hashes()
    write(out / "execution.json", {"configuration": config, "source_hashes": initial_sources,
          "tensorflow_python": tf_python, "generator_python": sys.executable,
          "independent_full_runs": 2, "pretrained_model_inputs": [], "replica_workers": args.replica_workers,
          "device": "cpu"})
    def launch(i):
        command = [sys.executable, "-m", "generators.reproduce", "--single-run", out / f"replicate_{i}",
                   "--tensorflow-python", tf_python]
        run_command(command, out / f"replicate_{i}.log")
    with ThreadPoolExecutor(max_workers=args.replica_workers) as pool:
        list(pool.map(launch, (1, 2)))
    if initial_sources != source_hashes():
        raise RuntimeError("Training sources changed during execution")
    finalize(out)


if __name__ == "__main__":
    main()
