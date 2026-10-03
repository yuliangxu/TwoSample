"""Train a new independent-CFM model or sample the bundled recovered ICFM model."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch
from torch import nn

from .tree import TreeTransform
from .vendor.torchdyn_numerics import integrate

PACKAGE = Path(__file__).resolve().parent
DEFAULT_TREE = PACKAGE / "assets" / "tree"
RECOVERED = PACKAGE / "checkpoints" / "icfm_recovered.pt"


class MLP(nn.Module):
    """Original time-dependent MLP: 123 -> 64 -> 64 -> 64 -> 122."""

    def __init__(self, dim: int = 122, width: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + 1, width), nn.SELU(),
            nn.Linear(width, width), nn.SELU(),
            nn.Linear(width, width), nn.SELU(),
            nn.Linear(width, dim),
        )

    def forward(self, x):
        return self.net(x)


class TimeWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        return self.model(torch.cat([x, t.repeat(x.shape[0])[:, None]], 1))


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def state_sha256(state) -> str:
    """Fingerprint tensors independently of torch.save archive filenames."""
    hasher = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        a = tensor.detach().cpu().contiguous().numpy()
        hasher.update(name.encode())
        hasher.update(str(a.dtype).encode())
        hasher.update(str(a.shape).encode())
        hasher.update(a.tobytes())
    return hasher.hexdigest()


def environment():
    return {"python": platform.python_version(),
            **{name: importlib.metadata.version(name) for name in ["torch", "numpy", "scipy", "pandas"]}}


def seed_everything(seed: int, device: str, threads: int):
    if threads < 1:
        raise ValueError("threads must be positive")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def tree_metadata(directory: Path):
    return {p.name: sha256(p) for p in sorted(directory.glob("*.csv"))}


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def output_guard(path: Path):
    if path.resolve() == RECOVERED.resolve():
        raise ValueError("The bundled recovered checkpoint cannot be overwritten")
    if path.exists() or path.with_suffix(path.suffix + ".metadata.json").exists():
        raise FileExistsError(f"Output exists: {path}; choose a new path")
    path.parent.mkdir(parents=True, exist_ok=True)


def load_model(checkpoint: Path, device: str):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "state_dict" in payload:
        state, metadata = payload["state_dict"], payload.get("metadata", {})
    else:
        state, metadata = payload, {"kind": "recovered state dictionary; historical training provenance unverified"}
    dim, width = state["net.6.weight"].shape
    model = MLP(dim=dim, width=width).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, metadata, dim


def sample(args):
    if args.n < 1 or args.atol <= 0 or args.rtol <= 0:
        raise ValueError("n, atol and rtol must be positive")
    if args.output.suffix.lower() not in {".npy", ".csv"}:
        raise ValueError("Sample output must end in .npy or .csv")
    output_guard(args.output)
    seed_everything(args.seed, args.device, args.threads)
    model, training_metadata, dim = load_model(args.checkpoint, args.device)
    current_tree_hashes = tree_metadata(args.tree)
    expected_tree_hashes = training_metadata.get("tree_csv_sha256")
    if expected_tree_hashes is not None and current_tree_hashes != expected_tree_hashes:
        raise ValueError("Tree metadata hashes differ from those recorded in the trained checkpoint")
    tree = TreeTransform.from_directory(args.tree)
    # Reseed after model construction, exactly as the historical sampling code.
    torch.manual_seed(args.seed)
    start = time.monotonic()
    initial = torch.randn(args.n, dim, device=args.device, dtype=torch.float32)
    with torch.no_grad():
        _, trajectory, nfe = integrate(TimeWrapper(model), initial,
            torch.linspace(0, 1, 2, device=args.device), args.atol, args.rtol)
    samples = tree.inverse(trajectory[-1].cpu().numpy(), mode="icfm_float32")
    if not np.isfinite(samples).all() or (samples < 0).any():
        raise RuntimeError("Generation returned invalid compositions")
    if args.output.suffix.lower() == ".npy":
        np.save(args.output, samples)
    else:
        np.savetxt(args.output, samples, delimiter=",", fmt="%.18e")
    metadata = {"operation": "sample", "checkpoint_name": args.checkpoint.name,
        "checkpoint_sha256": sha256(args.checkpoint), "state_sha256": state_sha256(model.state_dict()),
        "checkpoint_training_metadata": training_metadata, "tree_csv_sha256": current_tree_hashes,
        "n": args.n, "seed": args.seed, "device": args.device, "dtype": "float32",
        "threads": args.threads, "atol": args.atol, "rtol": args.rtol,
        "solver": "vendored torchdyn 1.0.6 dopri5 forward numerical path",
        "inverse": "icfm_float32: sigmoid float32, then float64 complement and tree multiplication",
        "nfe": nfe, "seconds": time.monotonic()-start, "shape": list(samples.shape),
        "max_row_sum_error": float(np.abs(samples.sum(1)-1).max()),
        "output_sha256": sha256(args.output), "environment": environment(),
        "source_sha256": sha256(Path(__file__)), "deterministic_algorithms": True,
        "replay_limit": "Bitwise reproducibility can depend on device, library versions, and floating-point implementation."}
    save_json(args.output.with_suffix(args.output.suffix + ".metadata.json"), metadata)
    print(json.dumps({"output": str(args.output), "sha256": metadata["output_sha256"], "shape": list(samples.shape)}))


def train(args):
    if args.steps < 1 or args.batch_size < 1 or args.lr <= 0 or args.sigma < 0:
        raise ValueError("steps, batch-size, and lr must be positive; sigma must be nonnegative")
    if args.zero_mass <= 0 or args.clip_logratio <= 0:
        raise ValueError("zero-mass and clip-logratio must be positive")
    output_guard(args.output)
    seed_everything(args.seed, args.device, args.threads)
    raw = np.loadtxt(args.train, delimiter=",")
    if raw.ndim != 2 or len(raw) < 2:
        raise ValueError("Training CSV must contain a sample-by-taxon matrix")
    tree = TreeTransform.from_directory(args.tree)
    logtree = tree.forward_icfm(raw, zero_mass=args.zero_mass, clip_logratio=args.clip_logratio)
    if not np.isfinite(logtree).all():
        raise ValueError("Training transformation contains non-finite values")
    target_bank = torch.as_tensor(logtree, dtype=torch.float32, device=args.device)
    model = MLP(dim=target_bank.shape[1]).to(args.device)
    initial_hash = state_sha256(model.state_dict())
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    losses = []
    start = time.monotonic()
    model.train()
    for step in range(args.steps):
        index = torch.randint(len(target_bank), (args.batch_size,), device=args.device)
        z1 = target_bank[index]
        z0 = torch.randn_like(z1)
        t = torch.rand(args.batch_size, 1, device=args.device)
        epsilon = torch.randn_like(z1)
        zt = (1-t)*z0 + t*z1 + args.sigma*epsilon
        velocity = z1-z0
        predicted = model(torch.cat([zt,t], dim=1))
        loss = torch.mean((predicted-velocity)**2)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at update {step+1}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    state = {name:value.detach().cpu() for name,value in model.state_dict().items()}
    final_hash = state_sha256(state)
    metadata = {"operation": "train", "kind": "new independent-CFM training recipe; not recovered historical settings",
        "training_csv_name": args.train.name, "training_csv_sha256": sha256(args.train),
        "training_shape": list(raw.shape), "training_inputs": "Only the supplied training CSV; no validation/test loading",
        "tree_csv_sha256": tree_metadata(args.tree),
        "preprocessing": {"method": "tree.forward_icfm", "zero_mass": args.zero_mass,
                          "clip_logratio": args.clip_logratio, "output_dtype": "float32",
                          "transformed_array_sha256": hashlib.sha256(target_bank.cpu().numpy().tobytes()).hexdigest()},
        "architecture": {"dimension": int(target_bank.shape[1]), "hidden_width": 64,
                         "hidden_layers": 3, "activation": "SELU", "time_input": True},
        "objective": "mean((v_theta((1-t)z0+t*z1+sigma*epsilon,t)-(z1-z0))**2)",
        "coupling": "independent standard-normal z0 and uniformly sampled training z1",
        "seed": args.seed, "steps": args.steps, "batch_size": args.batch_size,
        "minibatch_sampling": "with replacement; no epoch-based shuffle",
        "equivalent_training_passes": args.steps*args.batch_size/len(raw),
        "sigma": args.sigma, "optimizer": "Adam", "lr": args.lr, "weight_decay": 0.0,
        "optimizer_defaults": {"betas": [0.9,0.999], "eps": 1e-8, "amsgrad": False},
        "checkpoint_rule": "final update, no validation or model selection",
        "device": args.device, "threads": args.threads, "deterministic_algorithms": True,
        "initial_state_sha256": initial_hash, "final_state_sha256": final_hash,
        "parameters_changed": initial_hash != final_hash,
        "first_loss": losses[0], "final_loss": losses[-1], "losses": losses,
        "seconds": time.monotonic()-start, "environment": environment(), "source_sha256": sha256(Path(__file__))}
    torch.save({"format": "twosample.icfm.training.v1", "state_dict": state, "metadata": metadata}, args.output)
    metadata["checkpoint_sha256"] = sha256(args.output)
    save_json(args.output.with_suffix(args.output.suffix + ".metadata.json"), metadata)
    print(json.dumps({"output": str(args.output), "parameters_changed": metadata["parameters_changed"],
                      "final_state_sha256": final_hash, "first_loss": losses[0], "final_loss": losses[-1]}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["train", "sample"]:
        command = commands.add_parser(name)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--tree", type=Path, default=DEFAULT_TREE)
        command.add_argument("--seed", type=int, default=2 if name == "sample" else 20261003)
        command.add_argument("--device", default="cpu")
        command.add_argument("--threads", type=int, default=2)
    sampling = commands.choices["sample"]
    sampling.add_argument("--checkpoint", type=Path, default=RECOVERED)
    sampling.add_argument("--n", type=int, default=1000)
    sampling.add_argument("--atol", type=float, default=1e-4)
    sampling.add_argument("--rtol", type=float, default=1e-4)
    training = commands.choices["train"]
    training.add_argument("--train", type=Path, default=PACKAGE.parent/"data/sample_train.csv")
    training.add_argument("--steps", type=int, default=20000)
    training.add_argument("--batch-size", type=int, default=256)
    training.add_argument("--lr", type=float, default=1e-3)
    training.add_argument("--sigma", type=float, default=1e-4)
    training.add_argument("--zero-mass", type=float, default=1e-8)
    training.add_argument("--clip-logratio", type=float, default=float(np.log(1e6)))
    args = parser.parse_args()
    (train if args.command == "train" else sample)(args)


if __name__ == "__main__":
    main()
