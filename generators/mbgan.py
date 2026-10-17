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
import random
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
from .cohort import external_path
SCHEMA = "mbgan_historical_from_training_v2"
HISTORICAL_SOURCE = HERE / "vendor/historical_mbgan_model.py"
HISTORICAL_SOURCE_SHA256 = "9efc755e7fdcc18310c5fe4405319134f5d6e25ea50afb0a7b776b6971968e18"


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
    if device != "cpu":
        raise ValueError("This historical numerical recipe requires CPU execution")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["TF_NUM_INTRAOP_THREADS"] = str(threads)
    os.environ["TF_NUM_INTEROP_THREADS"] = "1"
    # Specify the numerical execution route, rather than inherit ambient values.
    os.environ["TF_ENABLE_ONEDNN_OPTS"] = "1"
    import tensorflow as tf
    import keras
    if tf.__version__ != "2.13.1" or keras.__version__ != "2.13.1":
        raise RuntimeError("The restored historical recipe requires TensorFlow/Keras 2.13.1")
    if device == "gpu" and not tf.config.list_physical_devices("GPU"):
        raise RuntimeError("--device gpu requested, but TensorFlow has no visible GPU")
    tf.config.threading.set_intra_op_parallelism_threads(threads)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    return tf


def environment(tf):
    versions = {"python": platform.python_version(), "machine": platform.machine()}
    import keras
    import h5py
    versions.update(tensorflow=tf.__version__, keras=keras.__version__,
                    numpy=np.__version__, h5py=h5py.__version__)
    versions["tensorflow_devices"] = [d.name for d in tf.config.list_physical_devices()]
    versions["onednn_enabled"] = True
    return versions


def historical_module():
    """Load the byte-preserved research source, with original upstream attribution.

    The source credits keras-team/keras-contrib improved_wgan.py and was used by
    the 2026-10-03 IBDMDB report. Its disconnected gradient-penalty layer is
    preserved deliberately; this is the historical local variant, not WGAN-GP.
    """
    if sha256(HISTORICAL_SOURCE) != HISTORICAL_SOURCE_SHA256:
        raise RuntimeError("Historical source differs from the audited original")
    from .vendor import historical_mbgan_model
    return historical_mbgan_model


def build_generator(tf, taxa=123, latent=100, width=512):
    return historical_module().build_generator((latent,), taxa, n_channels=width)


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


def optimizer_variables(optimizer):
    values = optimizer.variables
    return values() if callable(values) else values


def tf_random_stream_description(model, cfg):
    """Record seeded graph streams and their deterministic replay position.

    TensorFlow 2.13 stateful random kernels do not expose resumable counters as
    checkpoint variables. The seeded graph and completed update counts describe
    the stream position for full replay, not a checkpoint-resume guarantee.
    """
    functions = {}
    for label, graph in [("critic", model.critic_graph), ("generator", model.generator_graph)]:
        nodes = []
        for concrete in graph.train_function._list_all_concrete_functions_for_serialization():
            definition = concrete.graph.as_graph_def()
            groups = [("main", definition.node)]
            groups += [(f.signature.name, f.node_def) for f in definition.library.function]
            for scope, members in groups:
                for node in members:
                    if "Random" in node.op:
                        nodes.append(dict(scope=scope, name=node.name, op=node.op,
                                          seeds={k: int(node.attr[k].i) for k in ("seed", "seed2")
                                                 if k in node.attr}))
        functions[label] = nodes
    return dict(global_seed=cfg["seed"], graph_random_operations=functions,
                critic_updates=cfg["iterations"] * cfg["n_critic"],
                generator_updates=cfg["iterations"],
                exact_checkpoint_resume_supported=False,
                state_description="Seeded graph definition and completed call counts; replay from initialization")


def rng_description(model, cfg):
    state = np.random.get_state()
    return dict(python_state=random.getstate(),
                numpy_state=dict(bit_generator=state[0], keys=state[1].tolist(),
                                 position=state[2], has_gauss=state[3], cached_gaussian=state[4]),
                tensorflow=tf_random_stream_description(model, cfg))


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
    if (args.iterations < 1 or args.batch_size < 2 or args.n_critic < 1
            or args.log_every < 1 or args.callback_interval < 1):
        raise ValueError("Iteration and interval arguments must be positive; batch_size >= 2")
    if args.eager:
        raise ValueError("The historical recipe uses compiled Keras train_on_batch")
    if args.output_dir.exists():
        raise FileExistsError("Training requires a new output directory")
    if args.train.name == "sample_test.csv":
        raise ValueError("The testing split cannot be used for training")
    # Exact historical input path: CSV float64 values, no new normalization.
    x = np.loadtxt(args.train, delimiter=",")
    taxa = [s.strip() for s in args.taxonomy.read_text().splitlines() if s.strip()]
    if (x.ndim != 2 or x.shape[1] != len(taxa) or not np.isfinite(x).all()
            or (x < 0).any() or not np.allclose(x.sum(1), 1., atol=1e-6)):
        raise ValueError("Training requires finite normalized compositions in supplied taxon order")
    cfg = dict(schema=SCHEMA, variant="historical_local_no_penalty", initialization="random",
               taxa=len(taxa), latent=100, generator_width=512, critic_width=256,
               critic_dropout=0.25, taxonomy_log_scale=1000., optimizer="RMSprop",
               learning_rate=5e-5, rho=0.9, optimizer_epsilon=1e-7,
               optimizer_momentum=0., optimizer_centered=False,
               gradient_penalty_weight=0., effective_gradient_penalty=False,
               weight_clipping=False, compiled_loss_weights=[1, 1, 10],
               batch_size=args.batch_size, n_critic=args.n_critic, n_generator=1,
               iterations=args.iterations, seed=args.seed, device=args.device,
               threads=args.threads, deterministic_ops=True,
               execution="Keras 2.13.1 compiled train_on_batch",
               random_stream="NumPy MT19937 training draws; seeded TensorFlow/Keras stateful dropout",
               initialization_rng="tf.keras.utils.set_random_seed(seed); historical critic constructed before generator",
               callback_rng_policy="isolated_seeded_diagnostics",
               callback_interval=args.callback_interval,
               callback_seed_rule="(training seed + iteration) modulo 2**32; independent RandomState",
               training_rows=len(x), train_file=display_path(args.train),
               train_sha256=sha256(args.train), normalized_train_tensor_sha256=array_hash([x]),
               taxonomy_file=display_path(args.taxonomy), taxonomy_sha256=sha256(args.taxonomy),
               source_sha256=sha256(__file__), historical_source_sha256=sha256(HISTORICAL_SOURCE),
               historical_source_path=display_path(HISTORICAL_SOURCE), test_data_used=False,
               checkpoint_selection="final prescribed iteration; no validation or test selection",
               generator_training_during_critic_steps=False,
               critic_dropout_during_generator_steps=True,
               training_preprocessing="CSV float64 values, no additional normalization; Keras casts at model input")
    tf = configure_tf(args.device, args.threads, args.seed)
    historical = historical_module()
    matrix = phylogeny_matrix(taxa)
    model = historical.MBGAN("curated_comp_marginal",
        dict(ntaxa=len(taxa), latent_dim=100, generator=dict(n_channels=512),
             critic=dict(n_channels=256, dropout_rate=.25, tf_matrix=matrix, t_pow=1000.)),
        dict(generator=dict(optimizer=("RMSprop", {}), lr=.00005),
             critic=dict(loss_weights=[1, 1, 10], optimizer=("RMSprop", {}), lr=.00005)))
    if len(model.critic_graph.outputs) != 2 or model.critic_graph.losses:
        raise RuntimeError("Historical two-output unpenalized critic graph changed")
    g, d = model.generator, model.critic
    opt_g, opt_d = model.generator_graph.optimizer, model.critic_graph.optimizer
    before_g, before_d = g.get_weights(), d.get_weights()
    cfg.update(environment=environment(tf), phylogeny_tensor_sha256=array_hash([matrix]),
               initial_generator_tensor_sha256=array_hash(before_g),
               initial_critic_tensor_sha256=array_hash(before_d))
    args.output_dir.mkdir(parents=True)
    write_new_json(args.output_dir / "training_config.json", cfg)
    np.save(args.output_dir / "phylogeny_matrix.npy", matrix)
    valid, fake = -np.ones((args.batch_size, 1)), np.ones((args.batch_size, 1))
    started, warmup, last = time.monotonic(), None, None
    with (args.output_dir / "training_history.jsonl").open("x") as log, \
            (args.output_dir / "callback_history.jsonl").open("x") as callback_log:
        for step in range(1, args.iterations + 1):
            updates = []
            for _ in range(args.n_critic):
                real = x[np.random.randint(0, len(x), args.batch_size)]
                noise = np.random.normal(0, 1, (args.batch_size, cfg["latent"]))
                values = np.asarray(model.critic_graph.train_on_batch([real, noise], [valid, fake]), dtype=float)
                if not np.isfinite(values).all():
                    raise FloatingPointError(f"Nonfinite critic loss at iteration {step}")
                updates.append(values.tolist())
            noise = np.random.normal(0, 1, (args.batch_size, cfg["latent"]))
            g_loss = float(model.generator_graph.train_on_batch(noise, valid))
            if not np.isfinite(g_loss):
                raise FloatingPointError(f"Nonfinite generator loss at iteration {step}")
            elapsed = time.monotonic() - started
            last = dict(critic_loss=updates[-1][0], generator_loss=g_loss,
                        critic_update_losses=updates)
            record = dict(iteration=step, elapsed_seconds=elapsed, **last)
            log.write(json.dumps(record) + "\n")
            if step % args.callback_interval == 0:
                # Inference diagnostics never reseed or advance the training RNG.
                np_before, py_before = np.random.get_state(), random.getstate()
                samples = sample_generator(g, n=1000, seed=(args.seed + step) % 2**32)
                np.random.set_state(np_before)
                random.setstate(py_before)
                callback_log.write(json.dumps(dict(iteration=step, sample_tensor_sha256=array_hash([samples]),
                                                   mean_exact_zero_fraction=float(np.mean(samples == 0)))) + "\n")
                callback_log.flush()
            if step == min(10, args.iterations):
                warmup = elapsed
            if step == 1 or step == args.iterations or step % args.log_every == 0:
                log.flush()
                rate = (elapsed - warmup) / (step - 10) if warmup is not None and step > 10 else elapsed / step
                print(json.dumps(dict(record, estimated_remaining_seconds=rate * (args.iterations-step))), flush=True)
    change_g = max(float(np.max(np.abs(a-b))) for a, b in zip(g.get_weights(), before_g))
    change_d = max(float(np.max(np.abs(a-b))) for a, b in zip(d.get_weights(), before_d))
    if change_g <= 0 or change_d <= 0:
        raise RuntimeError("Training did not update both networks")
    weights, critic_weights = g.get_weights(), d.get_weights()
    optimizer_g = [v.numpy() for v in optimizer_variables(opt_g)]
    optimizer_d = [v.numpy() for v in optimizer_variables(opt_d)]
    if not all(np.isfinite(a).all() for a in weights + critic_weights + optimizer_g + optimizer_d):
        raise FloatingPointError("Nonfinite final model or optimizer tensor")
    final_g, final_d = array_hash(weights), array_hash(critic_weights)
    saved_cfg = dict(cfg, weight_count=len(weights), final_generator_tensor_sha256=final_g)
    np.savez_compressed(args.output_dir / "generator.npz", config=json.dumps(saved_cfg, sort_keys=True),
                        **{f"weight_{i:03d}": a for i, a in enumerate(weights)})
    np.savez_compressed(args.output_dir / "critic.npz",
                        **{f"weight_{i:03d}": a for i, a in enumerate(critic_weights)})
    np.savez_compressed(args.output_dir / "optimizer.npz",
                        **{f"generator_{i:03d}": a for i, a in enumerate(optimizer_g)},
                        **{f"critic_{i:03d}": a for i, a in enumerate(optimizer_d)})
    rng_record = rng_description(model, cfg)
    write_new_json(args.output_dir / "rng_state.json", rng_record)
    np.save(args.output_dir / "rng_state.npy", np.asarray(rng_record["numpy_state"]["keys"], dtype=np.uint32))
    tf.train.Checkpoint(generator=g, critic=d, optimizer_g=opt_g, optimizer_d=opt_d).write(
        str(args.output_dir / "training_state"))
    first = sample_generator(g, n=16, seed=256)
    restored, _ = load_generator(tf, args.output_dir / "generator.npz", require_completion=False)
    second = sample_generator(restored, n=16, seed=256)
    if not np.array_equal(first, second):
        raise RuntimeError("Saved generator does not reproduce deterministic sampling")
    np.save(args.output_dir / "reload_validation_samples.npy", first)
    result = dict(status="completed", mode="from-scratch training", iterations=args.iterations,
                  training_rows=len(x), elapsed_seconds=time.monotonic()-started,
                  warmed_iteration_seconds=(elapsed-warmup)/(args.iterations-10) if args.iterations>10 else None,
                  all_step_losses_finite=True, all_checkpoint_tensors_finite=True,
                  max_generator_parameter_change=change_g, max_critic_parameter_change=change_d,
                  deterministic_save_reload_equal=True, final_losses=last,
                  generator_sha256=sha256(args.output_dir / "generator.npz"),
                  final_generator_tensor_sha256=final_g, final_critic_tensor_sha256=final_d,
                  final_optimizer_tensor_sha256=array_hash(optimizer_g + optimizer_d),
                  optimizer_sha256=sha256(args.output_dir / "optimizer.npz"),
                  training_config_sha256=sha256(args.output_dir / "training_config.json"),
                  training_history_sha256=sha256(args.output_dir / "training_history.jsonl"),
                  callback_history_sha256=sha256(args.output_dir / "callback_history.jsonl"),
                  rng_state_sha256=sha256(args.output_dir / "rng_state.json"),
                  source_sha256=sha256(__file__), test_data_used=False,
                  checkpoint_selection=cfg["checkpoint_selection"])
    if result["source_sha256"] != cfg["source_sha256"] or sha256(HISTORICAL_SOURCE) != HISTORICAL_SOURCE_SHA256:
        raise RuntimeError("Training source changed during execution")
    if sha256(args.train) != cfg["train_sha256"] or sha256(args.taxonomy) != cfg["taxonomy_sha256"]:
        raise RuntimeError("Training inputs changed during execution")
    write_new_json(args.output_dir / "completion.json", result)
    print(json.dumps(result, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train", help="fit a fresh model from random initialization")
    train.add_argument("--train", type=Path, required=True)
    train.add_argument("--taxonomy", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--iterations", type=int, default=500000)
    train.add_argument("--batch-size", type=int, default=32)
    train.add_argument("--n-critic", type=int, default=5)
    train.add_argument("--seed", type=int, default=1)
    train.add_argument("--log-every", type=int, default=1000)
    train.add_argument("--callback-interval", type=int, default=10000)
    train.add_argument("--eager", action="store_true", help="rejected: historical recipe requires compiled Keras execution")
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
    external_path(args.output if args.command == "sample" else args.output_dir, "MBGAN output")
    if args.threads < 1:
        parser.error("--threads must be positive")
    sample_command(args) if args.command == "sample" else train_command(args)


if __name__ == "__main__":
    main()
