#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# Adapted from zhanxw/MB-GAN and the recovered research implementation.
# Original code credits keras-team/keras-contrib/examples/improved_wgan.py.
# See licenses/MBGAN-NOTICE.md and the accompanying GPL/MIT license texts.
"""Train MBGAN from random initialization and sample the resulting fitted model."""
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
TRAIN = ROOT / "data/sample_train.csv"
TAXA = ROOT / "data/provenance/taxa.txt"
SCHEMA = "mbgan_from_training_v1"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1048576), b""):
            h.update(chunk)
    return h.hexdigest()


def array_hash(arrays):
    """Hash tensor values independently of container compression or timestamps."""
    h = hashlib.sha256()
    for value in arrays:
        a = np.ascontiguousarray(value)
        h.update(str(a.dtype).encode())
        h.update(json.dumps(list(a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def display_path(path):
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return Path(path).name


def configure_tf(device="cpu", threads=2, seed=1):
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    # Specify the numerical execution route, rather than inherit ambient values.
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    import tensorflow as tf
    if device == "gpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("--device gpu requested, but TensorFlow has no visible GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def environment(tf):
    versions = {"python": platform.python_version(), "machine": platform.machine()}
    for name in ["tensorflow", "keras", "numpy", "h5py"]:
        versions[name] = importlib.metadata.version(name)
    versions["tensorflow_devices"] = [d.name for d in tf.config.list_physical_devices()]
    versions["onednn_enabled"] = True
    return versions


def build_generator(tf, taxa=123, latent=100, width=512):
    """Recovered architecture, including initial Dense ReLU before BN."""
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
    """Leaf columns followed by taxonomy prefixes in recovered expand_phylo order."""
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


def build_critic(tf, matrix, rng, width=256, dropout=0.25, t_pow=1000.):
    """Taxonomy-aware critic with dropout from the explicit training RNG."""
    class StreamDropout(tf.keras.layers.Layer):
        def call(self, inputs, training=False):
            if not training:
                return inputs
            keep = rng.uniform(tf.shape(inputs), dtype=inputs.dtype) >= dropout
            return tf.where(keep, inputs / (1. - dropout), tf.zeros_like(inputs))

    inputs = tf.keras.Input(shape=(matrix.shape[0],))
    transform = tf.constant(matrix, dtype=tf.float32)
    x = tf.keras.layers.Lambda(lambda z: tf.math.log1p(tf.matmul(z, transform)*t_pow)
                               / tf.math.log1p(tf.constant(t_pow, tf.float32)))(inputs)
    for _ in range(3):
        x = tf.keras.layers.Dense(width)(x)
        x = tf.keras.layers.LeakyReLU(negative_slope=0.2)(x)
        x = StreamDropout()(x)
    return tf.keras.Model(inputs, tf.keras.layers.Dense(1)(x), name="mbgan_critic")


def checked_gradients(tf, tape, loss, variables):
    gradients = tape.gradient(loss, variables)
    if any(g is None for g in gradients):
        raise RuntimeError("Disconnected training gradient")
    tf.debugging.assert_all_finite(loss, "Nonfinite training loss")
    for gradient in gradients:
        tf.debugging.assert_all_finite(gradient, "Nonfinite training gradient")
    return gradients, tf.linalg.global_norm(gradients)


def make_training_step(tf, generator, critic, opt_g, opt_d, rng, data, cfg,
                       compiled=True):
    """Same differentiated objective in eager and tf.function execution."""
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    batch, latent = cfg["batch_size"], cfg["latent"]

    def critic_step():
        indices = rng.uniform((batch,), maxval=len(data), dtype=tf.int32)
        real = tf.gather(data, indices)
        z = rng.normal((batch, latent), dtype=tf.float32)
        # Generator weights and moving BN statistics stay fixed here.
        fake = tf.stop_gradient(generator(z, training=False))
        alpha = rng.uniform((batch, 1), dtype=tf.float32)
        interpolated = alpha*real + (1-alpha)*fake
        with tf.GradientTape() as critic_tape:
            real_score = critic(real, training=True)
            fake_score = critic(fake, training=True)
            with tf.GradientTape() as input_tape:
                input_tape.watch(interpolated)
                interpolated_score = critic(interpolated, training=True)
            input_gradient = input_tape.gradient(interpolated_score, interpolated)
            norm = tf.sqrt(tf.reduce_sum(tf.square(input_gradient), axis=1)
                           + cfg["gradient_penalty_epsilon"])
            penalty = tf.reduce_mean(tf.square(norm - 1.))
            wasserstein = tf.reduce_mean(fake_score) - tf.reduce_mean(real_score)
            loss = wasserstein + cfg["gradient_penalty_weight"]*penalty
        grads, grad_norm = checked_gradients(tf, critic_tape, loss, critic.trainable_variables)
        opt_d.apply_gradients(zip(grads, critic.trainable_variables))
        return loss, wasserstein, penalty, grad_norm

    def generator_step():
        z = rng.normal((batch, latent), dtype=tf.float32)
        with tf.GradientTape() as generator_tape:
            fake = generator(z, training=True)
            loss = -tf.reduce_mean(critic(fake, training=True))
        grads, grad_norm = checked_gradients(tf, generator_tape, loss, generator.trainable_variables)
        opt_g.apply_gradients(zip(grads, generator.trainable_variables))
        return loss, grad_norm

    if compiled:
        critic_step = tf.function(critic_step, autograph=False, reduce_retracing=True)
        generator_step = tf.function(generator_step, autograph=False, reduce_retracing=True)

    def iteration():
        for _ in range(cfg["n_critic"]):
            d_loss, wasserstein, penalty, d_norm = critic_step()
        g_loss, g_norm = generator_step()
        return dict(critic_loss=d_loss, generator_loss=g_loss,
                    wasserstein_loss=wasserstein, gradient_penalty=penalty,
                    critic_gradient_norm=d_norm, generator_gradient_norm=g_norm)

    return tf.function(iteration, autograph=False) if compiled else iteration


def write_new_json(path, obj):
    with Path(path).open("x") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def load_generator(tf, path, require_completion=True):
    path = Path(path)
    if path.suffix.lower() != ".npz":
        raise ValueError("A newly trained generator.npz is required; historical HDF5 is unsupported")
    with np.load(path, allow_pickle=False) as data:
        cfg = json.loads(str(data["config"].item()))
        if (cfg.get("schema") != SCHEMA or cfg.get("initialization") != "random"
                or cfg.get("test_data_used") is not False):
            raise ValueError("Checkpoint lacks the required from-training provenance")
        model = build_generator(tf, cfg["taxa"], cfg["latent"], cfg["generator_width"])
        weights = [data[f"weight_{i:03d}"] for i in range(cfg["weight_count"])]
        if array_hash(weights) != cfg["final_generator_tensor_sha256"]:
            raise ValueError("Checkpoint tensor integrity failure")
        model.set_weights(weights)
    if require_completion:
        completion = json.loads((path.parent / "completion.json").read_text())
        if (completion.get("status") != "completed"
                or completion["iterations"] != cfg["iterations"]
                or completion["generator_sha256"] != sha256(path)):
            raise ValueError("Checkpoint does not match its completed training record")
        recorded = json.loads((path.parent / "training_config.json").read_text())
        if any(cfg.get(key) != value for key, value in recorded.items()):
            raise ValueError("Checkpoint and training configuration disagree")
    return model, cfg


def sample_generator(model, n=1000, seed=256, batch_size=32):
    if n < 1 or batch_size < 1:
        raise ValueError("n and batch_size must be positive")
    latent = np.random.RandomState(seed).normal(0, 1, (n, int(model.input_shape[-1])))
    x = np.concatenate([model(latent[start:start+batch_size], training=False).numpy()
                        for start in range(0, n, batch_size)], axis=0)
    if (not np.isfinite(x).all() or (x < 0).any()
            or not np.allclose(x.astype(float).sum(1), 1., atol=1e-6)):
        raise RuntimeError("Generated values are not finite normalized compositions")
    return x


def sample_command(args):
    output = args.output
    sidecar = output.with_suffix(output.suffix + ".json")
    if output.exists() or sidecar.exists():
        raise FileExistsError("Choose a new sample output path")
    if output.suffix.lower() not in {".npy", ".csv"}:
        raise ValueError("Sample output must end in .npy or .csv")
    tf = configure_tf(args.device, args.threads, args.seed)
    model, cfg = load_generator(tf, args.checkpoint)
    x = sample_generator(model, args.n, args.seed, args.batch_size)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".npy":
        with output.open("xb") as f:
            np.save(f, x)
    else:
        with output.open("x") as f:
            np.savetxt(f, x, delimiter=",")
    report = dict(mode="sampling from completed from-scratch training", shape=list(x.shape),
                  seed=args.seed, batch_size=args.batch_size, device=args.device,
                  threads=args.threads, checkpoint=display_path(args.checkpoint),
                  checkpoint_sha256=sha256(args.checkpoint), output_sha256=sha256(output),
                  source_sha256=sha256(__file__), training_config=cfg,
                  max_row_sum_error=float(np.max(np.abs(x.astype(float).sum(1)-1))),
                  environment=environment(tf))
    write_new_json(sidecar, report)
    print(json.dumps(report, indent=2))


def train_command(args):
    if args.iterations < 1 or args.batch_size < 2 or args.n_critic < 1 or args.log_every < 1:
        raise ValueError("iterations/n_critic/log_every must be positive; batch_size >= 2")
    if args.output_dir.exists():
        raise FileExistsError("Training requires a new output directory")
    if args.train.resolve() == (ROOT / "data/sample_test.csv").resolve():
        raise ValueError("The bundled testing split cannot be used for training")
    x = np.loadtxt(args.train, delimiter=",").astype(np.float32)
    taxa = [s.strip() for s in args.taxonomy.read_text().splitlines() if s.strip()]
    if (x.ndim != 2 or x.shape[1] != len(taxa) or not np.isfinite(x).all()
            or (x < 0).any() or (x.sum(1) <= 0).any()):
        raise ValueError("Training requires a finite nonnegative matrix in supplied taxon order")
    x /= x.sum(axis=1, keepdims=True)
    cfg = dict(schema=SCHEMA, initialization="random", taxa=len(taxa), latent=100,
               generator_width=512, critic_width=256, critic_dropout=0.25,
               taxonomy_log_scale=1000., optimizer="RMSprop", learning_rate=5e-5,
               rho=0.9, optimizer_epsilon=1e-7, optimizer_momentum=0.,
               optimizer_centered=False, gradient_penalty_weight=10.,
               gradient_penalty_epsilon=1e-12, batch_size=args.batch_size,
               n_critic=args.n_critic, n_generator=1, iterations=args.iterations,
               seed=args.seed, device=args.device, threads=args.threads,
               deterministic_ops=True, execution="eager" if args.eager else "tf.function",
               random_stream="tf.random.Generator PHILOX; explicit batch/noise/alpha/dropout draws",
               initialization_rng="tf.keras.utils.set_random_seed(seed); GlorotUniform dense weights",
               training_rows=len(x), train_file=display_path(args.train),
               train_sha256=sha256(args.train), normalized_train_tensor_sha256=array_hash([x]),
               taxonomy_file=display_path(args.taxonomy), taxonomy_sha256=sha256(args.taxonomy),
               source_sha256=sha256(__file__), test_data_used=False,
               checkpoint_selection="final prescribed iteration; no validation or test selection",
               generator_training_during_critic_steps=False,
               critic_dropout_during_generator_steps=True,
               training_preprocessing="float32; normalize each supplied training row to sum one")
    tf = configure_tf(args.device, args.threads, args.seed)
    rng = tf.random.Generator.from_seed(args.seed, alg="philox")
    g = build_generator(tf, cfg["taxa"], cfg["latent"], cfg["generator_width"])
    matrix = phylogeny_matrix(taxa)
    d = build_critic(tf, matrix, rng, cfg["critic_width"], cfg["critic_dropout"], cfg["taxonomy_log_scale"])
    options = dict(learning_rate=cfg["learning_rate"], rho=cfg["rho"], epsilon=cfg["optimizer_epsilon"],
                   momentum=cfg["optimizer_momentum"], centered=cfg["optimizer_centered"])
    opt_g, opt_d = tf.keras.optimizers.RMSprop(**options), tf.keras.optimizers.RMSprop(**options)
    opt_g.build(g.trainable_variables)
    opt_d.build(d.trainable_variables)
    before_g = [v.numpy().copy() for v in g.trainable_variables]
    before_d = [v.numpy().copy() for v in d.trainable_variables]
    cfg.update(environment=environment(tf), phylogeny_tensor_sha256=array_hash([matrix]),
               initial_generator_tensor_sha256=array_hash(g.get_weights()),
               initial_critic_tensor_sha256=array_hash(d.get_weights()))
    args.output_dir.mkdir(parents=True)
    write_new_json(args.output_dir / "training_config.json", cfg)
    np.save(args.output_dir / "phylogeny_matrix.npy", matrix)
    step_fn = make_training_step(tf, g, d, opt_g, opt_d, rng, x, cfg, compiled=not args.eager)
    started = time.monotonic()
    last = None
    warmup = None
    with (args.output_dir / "training_history.jsonl").open("x") as log:
        for step in range(1, args.iterations + 1):
            last = {key: float(value.numpy()) for key, value in step_fn().items()}
            elapsed = time.monotonic() - started
            record = dict(iteration=step, elapsed_seconds=elapsed, **last)
            log.write(json.dumps(record) + "\n")
            if step == min(10, args.iterations):
                warmup = elapsed
            if step == 1 or step == args.iterations or step % args.log_every == 0:
                log.flush()
                rate = (elapsed - warmup) / (step - 10) if warmup is not None and step > 10 else elapsed / step
                print(json.dumps(dict(record, estimated_remaining_seconds=rate*(args.iterations-step))), flush=True)
    change_g = max(float(np.max(np.abs(v.numpy()-a))) for v, a in zip(g.trainable_variables, before_g))
    change_d = max(float(np.max(np.abs(v.numpy()-a))) for v, a in zip(d.trainable_variables, before_d))
    if change_g <= 0 or change_d <= 0:
        raise RuntimeError("Training did not update both networks")
    weights = g.get_weights()
    final_g, final_d = array_hash(weights), array_hash(d.get_weights())
    saved_cfg = dict(cfg, weight_count=len(weights), final_generator_tensor_sha256=final_g)
    np.savez_compressed(args.output_dir / "generator.npz", config=json.dumps(saved_cfg, sort_keys=True),
                        **{f"weight_{i:03d}": a for i, a in enumerate(weights)})
    np.savez_compressed(args.output_dir / "critic.npz",
                        **{f"weight_{i:03d}": a for i, a in enumerate(d.get_weights())})
    np.save(args.output_dir / "rng_state.npy", rng.state.numpy())
    tf.train.Checkpoint(generator=g, critic=d, optimizer_g=opt_g, optimizer_d=opt_d,
                        training_rng=rng).write(str(args.output_dir / "training_state"))
    first = sample_generator(g, n=16, seed=256)
    restored, _ = load_generator(tf, args.output_dir / "generator.npz", require_completion=False)
    second = sample_generator(restored, n=16, seed=256)
    if not np.array_equal(first, second):
        raise RuntimeError("Saved generator does not reproduce deterministic sampling")
    np.save(args.output_dir / "reload_validation_samples.npy", first)
    result = dict(status="completed", mode="from-scratch training", iterations=args.iterations,
                  training_rows=len(x), elapsed_seconds=time.monotonic()-started,
                  warmed_iteration_seconds=(elapsed-warmup)/(args.iterations-10) if args.iterations>10 else None,
                  all_step_gradients_finite=True, max_generator_parameter_change=change_g,
                  max_critic_parameter_change=change_d, deterministic_save_reload_equal=True,
                  final_losses=last, generator_sha256=sha256(args.output_dir / "generator.npz"),
                  final_generator_tensor_sha256=final_g, final_critic_tensor_sha256=final_d,
                  final_optimizer_tensor_sha256=array_hash([v.numpy() for v in opt_g.variables + opt_d.variables]),
                  training_config_sha256=sha256(args.output_dir / "training_config.json"),
                  training_history_sha256=sha256(args.output_dir / "training_history.jsonl"),
                  source_sha256=sha256(__file__), test_data_used=False,
                  checkpoint_selection=cfg["checkpoint_selection"])
    if result["source_sha256"] != cfg["source_sha256"]:
        raise RuntimeError("Training source changed during execution")
    if sha256(args.train) != cfg["train_sha256"] or sha256(args.taxonomy) != cfg["taxonomy_sha256"]:
        raise RuntimeError("Training inputs changed during execution")
    write_new_json(args.output_dir / "completion.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit a fresh model from random initialization")
    train.add_argument("--train", type=Path, default=TRAIN)
    train.add_argument("--taxonomy", type=Path, default=TAXA)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--iterations", type=int, default=500000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--n-critic", type=int, default=5)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--log-every", type=int, default=1000)
    train.add_argument("--eager", action="store_true", help="uncompiled validation path; same objective")
    sample = commands.add_parser("sample", help="sample an explicitly supplied completed training run")
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--output", type=Path, required=True)
    sample.add_argument("--n", type=int, default=1000)
    sample.add_argument("--seed", type=int, default=256)
    sample.add_argument("--batch-size", type=int, default=32)
    for sub in commands.choices.values():
        sub.add_argument("--threads", type=int, default=2)
        sub.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    sample_command(args) if args.command == "sample" else train_command(args)


if __name__ == "__main__":
    main()
