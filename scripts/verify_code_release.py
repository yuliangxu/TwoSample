#!/usr/bin/env python3
"""Verify the selected source release without claiming completed training results."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    manifest = json.loads((ROOT / "code_release_manifest.json").read_text())
    require(manifest["schema"] == "twosample-selected-code-release-v1", "Unexpected schema")
    require(manifest["release_scope"] == "selected_code_and_preserved_real_inputs",
            "This verifier is for the source release")
    require(manifest["full_training_validation"] == "pending"
            and manifest["updated_results_included"] is False,
            "Full training/results require the completed-result verifier")
    require(manifest["pretrained_model_inputs"] == [], "Unexpected pretrained input")
    require(manifest["display_order"] == ["d", "dt", "icfm", "mbgan"], "Wrong figure order")
    for name, expected in manifest["files"].items():
        relative = Path(name)
        require(not relative.is_absolute() and ".." not in relative.parts,
                f"Unsafe manifest path: {name}")
        path = ROOT / relative
        require(path.is_file() and path.resolve().is_relative_to(ROOT.resolve()),
                f"Missing or external release file: {name}")
        require(digest(path) == expected, f"Changed release file: {name}")
    assets = json.loads((ROOT / "generators/manifest.json").read_text())
    for entry in assets["files"]:
        require(manifest["files"].get(entry["path"]) == entry["sha256"],
                f"Asset not bound to the release: {entry['path']}")
    config = json.loads((ROOT / "generators/training_config.json").read_text())
    cfg = config["models"]["mbgan"]
    require(cfg["variant"] == "historical_local_no_penalty"
            and cfg["device"] == "cpu" and cfg["iterations"] == 500000
            and cfg["callback_rng_policy"] == "isolated_seeded_diagnostics"
            and cfg["callback_interval"] == 10000
            and cfg["gradient_penalty_connected"] is False,
            "Selected MBGAN recipe differs")
    evidence = json.loads((ROOT / manifest["implementation_validation"]).read_text())
    require(evidence["status"] == "passed"
            and evidence["source_sha256"] == digest(ROOT / "generators/mbgan.py"),
            "Short validation does not identify this implementation")
    require(evidence["legacy_parity"] and all(value is True for value in evidence["legacy_parity"].values()),
            "Historical implementation parity was not verified")
    paired = evidence["paired_reproducibility"]
    require(paired["status"] == "passed" and paired["iterations"] == 120
            and paired["all_loss_history_exact"] and paired["all_optimizer_tensors_exact"]
            and paired["all_generator_and_critic_tensors_exact"] and paired["samples_byte_exact"],
            "Short independent reproduction failed")
    run = json.loads((ROOT / manifest["full_run_record"]).read_text())
    for name, expected in run["training_source_hashes"].items():
        require(manifest["files"].get(name) == expected,
                f"Published training code differs from the launched run: {name}")
    for name in ("data/generation_manifest.json", "results/final/COMPLETE", "release_manifest.json"):
        require(not (ROOT / name).exists(), f"Unexpected completed-result artifact: {name}")
    for key in ("d", "dt", "icfm", "mbgan"):
        require(not (ROOT / "data" / f"sample_{key}.csv").exists(),
                "This source release must not mix in a prior synthetic bank")
    print(f"Verified {len(manifest['files'])} source/input/evidence files and {len(assets['files'])} assets.")
    print("Historical MBGAN selected; 120-step parity and independent reproduction passed.")
    print("Full 500,000-iteration validation and updated analysis results remain pending.")


if __name__ == "__main__":
    main()
