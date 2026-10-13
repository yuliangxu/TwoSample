#!/usr/bin/env python3
"""Compare independently trained MBGAN runs, including their complete trajectories."""
import argparse
import json
from pathlib import Path

import numpy as np

try:
    from .mbgan import array_hash, sha256, SCHEMA, HISTORICAL_SOURCE, HISTORICAL_SOURCE_SHA256
except ImportError:
    from mbgan import array_hash, sha256, SCHEMA, HISTORICAL_SOURCE, HISTORICAL_SOURCE_SHA256


def validate_run(path):
    config = json.loads((path / "training_config.json").read_text())
    result = json.loads((path / "completion.json").read_text())
    assert config["schema"] == SCHEMA
    assert config["initialization"] == "random" and not config["test_data_used"]
    assert result["status"] == "completed" and not result["test_data_used"]
    assert result["all_step_losses_finite"] and result["all_checkpoint_tensors_finite"]
    assert result["deterministic_save_reload_equal"]
    assert config["variant"] == "historical_local_no_penalty"
    assert config["callback_rng_policy"] == "isolated_seeded_diagnostics"
    assert config["effective_gradient_penalty"] is False and config["gradient_penalty_weight"] == 0
    assert config["historical_source_sha256"] == HISTORICAL_SOURCE_SHA256 == sha256(HISTORICAL_SOURCE)
    assert result["iterations"] == config["iterations"]
    for key, file in [("generator_sha256", "generator.npz"),
                      ("training_config_sha256", "training_config.json"),
                      ("training_history_sha256", "training_history.jsonl"),
                      ("optimizer_sha256", "optimizer.npz"),
                      ("callback_history_sha256", "callback_history.jsonl"),
                      ("rng_state_sha256", "rng_state.json")]:
        assert result[key] == sha256(path / file), file
    assert config["source_sha256"] == result["source_sha256"]
    assert config["source_sha256"] == sha256(Path(__file__).with_name("mbgan.py"))
    tensors = {}
    for family in ["generator", "critic"]:
        with np.load(path / (family + ".npz"), allow_pickle=False) as data:
            tensors[family] = [data[key] for key in sorted(data.files) if key.startswith("weight_")]
        assert all(np.isfinite(a).all() for a in tensors[family])
        assert array_hash(tensors[family]) == result[f"final_{family}_tensor_sha256"]
        assert config[f"initial_{family}_tensor_sha256"] != result[f"final_{family}_tensor_sha256"]
    with np.load(path / "optimizer.npz", allow_pickle=False) as data:
        names = sorted(k for k in data.files if k.startswith("generator_"))
        names += sorted(k for k in data.files if k.startswith("critic_"))
        tensors["optimizer"] = [data[k] for k in names]
    assert all(np.isfinite(a).all() for a in tensors["optimizer"])
    assert array_hash(tensors["optimizer"]) == result["final_optimizer_tensor_sha256"]
    return config, result, tensors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--samples-a", type=Path, required=True)
    parser.add_argument("--samples-b", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg_a, result_a, tensors_a = validate_run(args.run_a)
    cfg_b, result_b, tensors_b = validate_run(args.run_b)
    assert cfg_a == cfg_b, "Training configurations differ"
    for family in tensors_a:
        assert len(tensors_a[family]) == len(tensors_b[family])
        assert all(np.array_equal(a, b) for a, b in zip(tensors_a[family], tensors_b[family]))
    assert result_a["final_optimizer_tensor_sha256"] == result_b["final_optimizer_tensor_sha256"]
    assert np.array_equal(np.load(args.run_a / "rng_state.npy"), np.load(args.run_b / "rng_state.npy"))
    rng_a = json.loads((args.run_a / "rng_state.json").read_text())
    rng_b = json.loads((args.run_b / "rng_state.json").read_text())
    assert rng_a == rng_b, "Recorded RNG states or TensorFlow replay positions differ"
    assert (args.run_a / "callback_history.jsonl").read_bytes() == (args.run_b / "callback_history.jsonl").read_bytes()
    counts = 0
    with (args.run_a / "training_history.jsonl").open() as a, (args.run_b / "training_history.jsonl").open() as b:
        for counts, line_a in enumerate(a, 1):
            line_b = b.readline()
            assert line_b, "Second history is shorter"
            row_a, row_b = json.loads(line_a), json.loads(line_b)
            row_a.pop("elapsed_seconds")
            row_b.pop("elapsed_seconds")
            assert row_a == row_b, f"Different update at iteration {counts}"
            assert row_a["iteration"] == counts
            assert len(row_a["critic_update_losses"]) == cfg_a["n_critic"]
            assert np.isfinite(np.asarray(row_a["critic_update_losses"], dtype=float)).all()
            assert np.isfinite(row_a["critic_loss"]) and np.isfinite(row_a["generator_loss"])
        assert not b.readline(), "Second history is longer"
    assert counts == cfg_a["iterations"]
    samples_a, samples_b = np.loadtxt(args.samples_a, delimiter=","), np.loadtxt(args.samples_b, delimiter=",")
    assert samples_a.shape == (1000, cfg_a["taxa"])
    assert np.array_equal(samples_a, samples_b)
    assert sha256(args.samples_a) == sha256(args.samples_b)
    assert np.isfinite(samples_a).all() and (samples_a >= 0).all()
    assert np.allclose(samples_a.sum(1), 1, atol=1e-6)
    report = dict(status="passed", iterations=counts, training_rows=cfg_a["training_rows"],
                  source_sha256=cfg_a["source_sha256"],
                  initial_generator_tensor_sha256=cfg_a["initial_generator_tensor_sha256"],
                  initial_critic_tensor_sha256=cfg_a["initial_critic_tensor_sha256"],
                  final_generator_tensor_sha256=result_a["final_generator_tensor_sha256"],
                  final_critic_tensor_sha256=result_a["final_critic_tensor_sha256"],
                  final_optimizer_tensor_sha256=result_a["final_optimizer_tensor_sha256"],
                  all_loss_history_exact=True, all_generator_and_critic_tensors_exact=True,
                  all_optimizer_tensors_exact=True, callback_history_exact=True,
                  recorded_rng_state_and_replay_position_exact=True,
                  tensorflow_checkpoint_resume_supported=False, samples_shape=list(samples_a.shape),
                  samples_sha256=sha256(args.samples_a), samples_byte_exact=True,
                  warmed_iteration_seconds=[result_a["warmed_iteration_seconds"], result_b["warmed_iteration_seconds"]],
                  estimated_hours_500k=[r["warmed_iteration_seconds"]*500000/3600
                                        if r["warmed_iteration_seconds"] is not None else None
                                        for r in [result_a, result_b]])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
