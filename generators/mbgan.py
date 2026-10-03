#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Adapted from zhanxw/MB-GAN and the recovered research implementation.
# Original code credits keras-team/keras-contrib/examples/improved_wgan.py.
# See licenses/MBGAN-NOTICE.md and the accompanying GPL/MIT license texts.
"""Portable MBGAN checkpoint sampling and an explicit new-training recipe.

Sampling replays the recovered generator; training does not claim to recreate
its historical weights. Run ``python generators/mbgan.py --help``.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CHECKPOINT = HERE / "checkpoints/mbgan_generator.h5"
TRAIN = ROOT / "data/sample_train.csv"
TAXA = ROOT / "data/provenance/taxa.txt"
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


def build_generator(tf, taxa=123, latent=100, width=512):
    """Original MBGAN chain, including its initial Dense ReLU before BN."""
    layers = [tf.keras.Input(shape=(latent,)),
              tf.keras.layers.Dense(width, activation="relu"),
              tf.keras.layers.BatchNormalization(momentum=0.8, epsilon=0.001),
              tf.keras.layers.Activation("relu")]
    for _ in range(2):
        layers += [tf.keras.layers.Dense(width),
                   tf.keras.layers.BatchNormalization(momentum=0.8, epsilon=0.001),
                   tf.keras.layers.Activation("relu")]
    layers += [tf.keras.layers.Dense(taxa), tf.keras.layers.Activation("softmax")]
    return tf.keras.Sequential(layers, name="mbgan_generator")


def phylogeny_matrix(taxa):
    """Leaf columns followed by taxonomy prefixes, as in recovered expand_phylo."""
    if len(taxa) != len(set(taxa)):
        raise ValueError("Taxon names must be unique")
    indices = {name: i for i, name in enumerate(taxa)}
    pairs = []
    for i, taxon in enumerate(taxa):
        parts = taxon.split("|")
        for depth in range(1, len(parts) + 1):
            name = "|".join(parts[:depth])
            if name not in indices:
                indices[name] = len(indices)
            pairs.append((i, indices[name]))
    matrix = np.zeros((len(taxa), len(indices)), dtype=np.float32)
    for i, j in pairs:
        matrix[i, j] = 1.
    return matrix


def build_critic(tf, matrix, width=256, dropout=0.25, t_pow=1000.):
    """Taxonomy aggregation, normalized log transform, three Dense blocks."""
    inputs = tf.keras.Input(shape=(matrix.shape[0],))
    transform = tf.constant(matrix, dtype=tf.float32)
    x = tf.keras.layers.Lambda(lambda z: tf.math.log1p(tf.matmul(z, transform)*t_pow)
                               / tf.math.log1p(tf.constant(t_pow, tf.float32)))(inputs)
    for _ in range(3):
        x = tf.keras.layers.Dense(width)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1)(x), name="mbgan_critic")


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
    if path.suffix.lower() in {".h5", ".hdf5"}:
        return load_historical_generator(tf, path)
    with np.load(path, allow_pickle=False) as data:
        config = json.loads(str(data["config"].item()))
        model = build_generator(tf, config["taxa"], config["latent"], config["generator_width"])
        model.set_weights([data[f"weight_{i:03d}"] for i in range(config["weight_count"])])
    return model, dict(load_method="new-training NumPy checkpoint", training_config=config)


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


def checked_gradients(tf, tape, loss, variables):
    gradients = tape.gradient(loss, variables)
    if any(g is None for g in gradients):
        raise RuntimeError("Disconnected training gradient")
    tf.debugging.assert_all_finite(loss, "Nonfinite training loss")
    for g in gradients:
        tf.debugging.assert_all_finite(g, "Nonfinite training gradient")
    return gradients, float(tf.linalg.global_norm(gradients))


def training_step(tf, generator, critic, opt_g, opt_d, rng, data, cfg):
    for _ in range(cfg["n_critic"]):
        real = tf.convert_to_tensor(data[rng.randint(len(data), size=cfg["batch_size"])])
        z = tf.convert_to_tensor(rng.normal(size=(cfg["batch_size"], cfg["latent"])).astype(np.float32))
        # Freeze generator weights and BN statistics during critic updates.
        fake = tf.stop_gradient(generator(z, training=False))
        alpha = tf.random.uniform((cfg["batch_size"], 1))
        interpolated = alpha*real + (1-alpha)*fake
        with tf.GradientTape() as critic_tape:
            real_score, fake_score = critic(real, training=True), critic(fake, training=True)
            with tf.GradientTape() as input_tape:
                input_tape.watch(interpolated)
                interpolated_score = critic(interpolated, training=True)
            input_gradient = input_tape.gradient(interpolated_score, interpolated)
            # Numerical epsilon prevents undefined sqrt derivative at zero norm.
            norm = tf.sqrt(tf.reduce_sum(tf.square(input_gradient), axis=1) + 1e-12)
            penalty = tf.reduce_mean(tf.square(norm-1.))
            wasserstein = tf.reduce_mean(fake_score)-tf.reduce_mean(real_score)
            d_loss = wasserstein + cfg["gradient_penalty_weight"]*penalty
        d_grad, d_norm = checked_gradients(tf, critic_tape, d_loss, critic.trainable_variables)
        opt_d.apply_gradients(zip(d_grad, critic.trainable_variables))
    z = tf.convert_to_tensor(rng.normal(size=(cfg["batch_size"], cfg["latent"])).astype(np.float32))
    with tf.GradientTape() as generator_tape:
        # Original training graph keeps critic dropout active in generator steps.
        fake = generator(z, training=True)
        g_loss = -tf.reduce_mean(critic(fake, training=True))
    g_grad, g_norm = checked_gradients(tf, generator_tape, g_loss, generator.trainable_variables)
    opt_g.apply_gradients(zip(g_grad, generator.trainable_variables))
    return dict(critic_loss=float(d_loss), generator_loss=float(g_loss),
                wasserstein_loss=float(wasserstein), gradient_penalty=float(penalty),
                critic_gradient_norm=d_norm, generator_gradient_norm=g_norm)


def train_command(args, smoke=False):
    if args.iterations < 1 or args.batch_size < 2 or args.n_critic < 1:
        raise ValueError("iterations/n_critic must be positive and batch_size must be >=2")
    if args.output_dir.exists():
        raise FileExistsError("Training requires a new output directory")
    if args.train.resolve() == (ROOT/"data/sample_test.csv").resolve():
        raise ValueError("The bundled real testing split cannot be used for training")
    x = np.loadtxt(args.train, delimiter=",").astype(np.float32)
    taxa = [line.strip() for line in args.taxonomy.read_text().splitlines() if line.strip()]
    if x.ndim != 2 or x.shape[1] != len(taxa) or not np.isfinite(x).all() or (x<0).any() or (x.sum(1)<=0).any():
        raise ValueError("Training must be a nonnegative finite matrix in supplied taxon order")
    x /= x.sum(axis=1, keepdims=True)
    cfg = dict(taxa=len(taxa), latent=100, generator_width=512, critic_width=256,
               critic_dropout=0.25, taxonomy_log_scale=1000., optimizer="RMSprop",
               learning_rate=5e-5, rho=0.9, optimizer_epsilon=1e-7,
               gradient_penalty_weight=10., batch_size=args.batch_size,
               n_critic=args.n_critic, n_generator=1, iterations=args.iterations, seed=args.seed,
               device=args.device, threads=args.threads, deterministic_ops=True,
               training_rows=len(x), train_file=display_path(args.train), train_sha256=sha256(args.train),
               taxonomy_file=display_path(args.taxonomy), taxonomy_sha256=sha256(args.taxonomy),
               training_claim="New train-only recipe adapted from recovered and upstream MBGAN; historical weights not retrained",
               test_data_used=False, gradient_penalty_epsilon=1e-12,
               generator_training_during_critic_steps=False, critic_dropout_during_generator_steps=True)
    tf = configure_tf(args.device, args.threads, args.seed)
    g = build_generator(tf, cfg["taxa"], cfg["latent"], cfg["generator_width"])
    matrix = phylogeny_matrix(taxa)
    d = build_critic(tf, matrix, cfg["critic_width"], cfg["critic_dropout"], cfg["taxonomy_log_scale"])
    options = dict(learning_rate=cfg["learning_rate"], rho=cfg["rho"], epsilon=cfg["optimizer_epsilon"])
    opt_g, opt_d = tf.keras.optimizers.RMSprop(**options), tf.keras.optimizers.RMSprop(**options)
    before_g = [v.numpy().copy() for v in g.trainable_variables]
    before_d = [v.numpy().copy() for v in d.trainable_variables]
    rng = np.random.RandomState(args.seed)
    args.output_dir.mkdir(parents=True)
    cfg["environment"] = environment()
    write_new_json(args.output_dir/"training_config.json", cfg)
    np.save(args.output_dir/"phylogeny_matrix.npy", matrix)
    started = time.monotonic()
    last = None
    with (args.output_dir/"training_history.jsonl").open("x") as log:
        for step in range(1, args.iterations+1):
            last = training_step(tf, g, d, opt_g, opt_d, rng, x, cfg)
            record = dict(iteration=step, **last)
            log.write(json.dumps(record)+"\n")
            if step == 1 or step == args.iterations or step % 1000 == 0:
                log.flush()
                print(json.dumps(record), flush=True)
    change_g = max(float(np.max(np.abs(v.numpy()-a))) for v,a in zip(g.trainable_variables,before_g))
    change_d = max(float(np.max(np.abs(v.numpy()-a))) for v,a in zip(d.trainable_variables,before_d))
    if change_g <= 0 or change_d <= 0:
        raise RuntimeError("Training did not update both generator and critic")
    weights = g.get_weights()
    saved_config = dict(cfg, weight_count=len(weights))
    np.savez_compressed(args.output_dir/"generator.npz", config=json.dumps(saved_config),
                        **{f"weight_{i:03d}": a for i,a in enumerate(weights)})
    d.save_weights(args.output_dir/"critic.weights.h5")
    # An optimizer checkpoint preserves training state; generator.npz is sufficient for sampling.
    tf.train.Checkpoint(generator=g, critic=d, optimizer_g=opt_g, optimizer_d=opt_d).write(str(args.output_dir/"training_state"))
    first = sample_generator(g, n=16, seed=256)
    restored, _ = load_generator(tf, args.output_dir/"generator.npz")
    second = sample_generator(restored, n=16, seed=256)
    if not np.array_equal(first, second):
        raise RuntimeError("Saved generator does not reproduce deterministic sampling")
    np.save(args.output_dir/"smoke_samples.npy", first)
    result = dict(status="completed", mode="smoke training" if smoke else "new training",
                  iterations=args.iterations, training_rows=len(x), elapsed_seconds=time.monotonic()-started,
                  all_step_gradients_finite=True, max_generator_parameter_change=change_g,
                  max_critic_parameter_change=change_d, deterministic_save_reload_equal=True,
                  final_losses=last, generator_sha256=sha256(args.output_dir/"generator.npz"),
                  historical_weights_reproduced=False)
    if smoke:
        historical, metadata = load_generator(tf, CHECKPOINT)
        a = sample_generator(historical, n=1000, seed=256)
        b = sample_generator(historical, n=1000, seed=256)
        ref = np.loadtxt(ROOT/"data/sample_mbgan.csv", delimiter=",")
        error = a.astype(float)-ref
        result["historical_replay"] = dict(checkpoint_sha256=sha256(CHECKPOINT),
            deterministic_equal=bool(np.array_equal(a,b)),
            max_abs_error=float(np.max(np.abs(error))), rmse=float(np.sqrt(np.mean(error**2))),
            allclose_rtol1e5_atol1e8=bool(np.allclose(a,ref,rtol=1e-5,atol=1e-8)), **metadata)
        if not result["historical_replay"]["deterministic_equal"] or not result["historical_replay"]["allclose_rtol1e5_atol1e8"]:
            raise RuntimeError("Historical replay validation failed")
    write_new_json(args.output_dir/"completion.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    sample = commands.add_parser("sample", help="sample a saved generator, without training")
    sample.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--n", type=int, default=1000)
    sample.add_argument("--seed", type=int, default=256)
    sample.add_argument("--batch-size", type=int, default=32)
    for name in ["train", "smoke-test"]:
        sub = commands.add_parser(name, help="fit a new model using only the supplied training CSV")
        sub.add_argument("--train", type=Path, default=TRAIN)
        sub.add_argument("--taxonomy", type=Path, default=TAXA)
        sub.add_argument("--output-dir", type=Path, required=True)
        sub.add_argument("--iterations", type=int, default=2 if name=="smoke-test" else 500000)
        sub.add_argument("--batch-size", type=int, default=32)
        sub.add_argument("--n-critic", type=int, default=5)
        sub.add_argument("--seed", type=int, default=1)
    for sub in commands.choices.values():
        sub.add_argument("--threads", type=int, default=2)
        sub.add_argument("--device", choices=["cpu","gpu","auto"], default="cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    if args.command == "sample":
        sample_command(args)
    else:
        train_command(args, smoke=args.command=="smoke-test")


if __name__ == "__main__":
    main()
