#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Adapted from zhanxw/MB-GAN and the recovered research implementation.
# Original code credits keras-team/keras-contrib/examples/improved_wgan.py.
# See licenses/MBGAN-NOTICE.md and the accompanying GPL/MIT license texts.
"""Replay the fixed MBGAN checkpoint used in the finalized case study."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKPOINT = HERE / "checkpoints/mbgan_generator.h5"
BUNDLED_SHA256 = "457c13dbba3afefd1c0a7979eb6854118d25806143ac5306bd1543740eaf840f"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def display_path(path):
    """Keep machine-specific paths out of generated public provenance."""
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


def configure_tf(device="cpu", threads=2, seed=None):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(threads))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
    import tensorflow as tf
    if device == "gpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("--device gpu requested, but TensorFlow has no visible GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    if seed is not None:
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
    return tf


def environment():
    versions = {"python": platform.python_version()}
    for name in ["tensorflow", "keras", "numpy", "h5py"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def load_historical_generator(tf, path):
    """Read the saved chain and weights without Keras 2 graph deserialization.

    The original Functional wrapper points to Sequential node 1, which Keras 3
    cannot deserialize. This reconstructs that chain without changing weights.
    """
    import h5py
    with h5py.File(path, "r") as f:
        saved = json.loads(f.attrs["model_config"])
        outer = saved["config"]["layers"]
        if [x["class_name"] for x in outer] != ["InputLayer", "Sequential"]:
            raise ValueError("Expected the recovered InputLayer/Sequential generator")
        nested = outer[1]["config"]
        model = tf.keras.Sequential(name=nested["name"])
        model.add(tf.keras.Input(shape=outer[0]["config"]["batch_input_shape"][1:]))
        group = f["model_weights"][nested["name"]]
        verified = 0
        for entry in nested["layers"]:
            kind, cfg = entry["class_name"], entry["config"]
            if kind == "InputLayer":
                continue
            common = dict(name=cfg["name"], dtype=cfg["dtype"], trainable=cfg["trainable"])
            if kind == "Dense":
                layer = tf.keras.layers.Dense(cfg["units"], activation=cfg["activation"],
                                              use_bias=cfg["use_bias"], **common)
                names = ["kernel", "bias"] if cfg["use_bias"] else ["kernel"]
            elif kind == "BatchNormalization":
                axis = cfg["axis"]
                if isinstance(axis, list):
                    if len(axis) != 1:
                        raise ValueError("Only the saved single-axis BN is supported")
                    axis = axis[0]
                layer = tf.keras.layers.BatchNormalization(axis=axis, momentum=cfg["momentum"],
                    epsilon=cfg["epsilon"], center=cfg["center"], scale=cfg["scale"], **common)
                names = (["gamma"] if cfg["scale"] else []) + (["beta"] if cfg["center"] else []) + ["moving_mean", "moving_variance"]
            elif kind == "Activation":
                layer = tf.keras.layers.Activation(cfg["activation"], **common)
                names = []
            else:
                raise ValueError("Unsupported saved layer: " + kind)
            model.add(layer)
            if names:
                arrays = [group[cfg["name"]][name+":0"][()] for name in names]
                layer.set_weights(arrays)
                if not all(np.array_equal(a,b) for a,b in zip(layer.get_weights(), arrays)):
                    raise ValueError("Checkpoint weight verification failed")
                verified += len(arrays)
        metadata = dict(checkpoint_keras_version=str(f.attrs.get("keras_version", "unknown")),
                        load_method="saved Sequential config plus exact HDF5 weights",
                        verified_weight_arrays=verified)
    return model, metadata


def load_generator(tf, path):
    path = Path(path)
    if path.resolve() == CHECKPOINT.resolve() and sha256(path) != BUNDLED_SHA256:
        raise ValueError("Bundled checkpoint hash differs from the validated checkpoint")
    if path.suffix.lower() not in {".h5", ".hdf5"}:
        raise ValueError("The case-study sampler requires the recovered HDF5 checkpoint")
    return load_historical_generator(tf, path)


def sample_generator(model, n=1000, seed=256, batch_size=32):
    if n < 1 or batch_size < 1:
        raise ValueError("n and batch_size must be positive")
    # RandomState preserves the exact original np.random.seed/normal algorithm.
    latent = np.random.RandomState(seed).normal(0, 1, (n, int(model.input_shape[-1])))
    x = model.predict(latent, batch_size=batch_size, verbose=0)
    if not np.isfinite(x).all() or (x < 0).any() or not np.allclose(x.sum(1), 1, atol=1e-6):
        raise RuntimeError("Generated values are not finite normalized compositions")
    return x


def write_new_json(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def sample_command(args):
    output = args.output
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Choose a new sample output path; existing files are preserved")
    if output.suffix.lower() not in {".npy", ".csv"}:
        raise ValueError("Sample output must end in .npy or .csv")
    tf = configure_tf(args.device, args.threads)
    model, metadata = load_generator(tf, args.checkpoint)
    x = sample_generator(model, args.n, args.seed, args.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".npy":
        with output.open("xb") as f:
            np.save(f, x)
    else:
        with output.open("x") as f:
            np.savetxt(f, x, delimiter=",")
    report = dict(mode="checkpoint sampling; no training", shape=list(x.shape), seed=args.seed,
                  batch_size=args.batch_size, device=args.device, checkpoint=display_path(args.checkpoint),
                  checkpoint_sha256=sha256(args.checkpoint), output_sha256=sha256(output),
                  max_row_sum_error=float(np.max(np.abs(x.astype(float).sum(1)-1))),
                  environment=environment(), **metadata)
    write_new_json(sidecar, report)
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sampling = commands.add_parser("sample", help="replay the fixed MBGAN checkpoint")
    sampling.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    sampling.add_argument("--output", type=Path, required=True)
    sampling.add_argument("--n", type=int, default=1000)
    sampling.add_argument("--seed", type=int, default=256)
    sampling.add_argument("--batch-size", type=int, default=32)
    sampling.add_argument("--threads", type=int, default=2)
    sampling.add_argument("--device", choices=["cpu", "gpu", "auto"], default="cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    sample_command(args)


if __name__ == "__main__":
    main()
