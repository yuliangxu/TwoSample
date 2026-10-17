#!/usr/bin/env python3
"""Reject data, figures, checkpoints, and runtime artifacts in the release.

By default inspect the Git index (including staged changes). --revision checks
committed objects; --directory checks an unpacked source distribution. Runtime
outputs are deliberately not inspected by the Git modes because they are local
and ignored, never part of the published source tree.
"""
import argparse
import json
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SPECIAL = {
    ".gitignore", ".gitattributes", ".github/workflows/code-only.yml",
    "generators/training_config.json", "generators/vendor/SOURCE.json",
    "generators/requirements.txt", "generators/requirements-mbgan.txt",
    "requirements-preprocessing.txt",
    "generators/licenses/KERAS-CONTRIB-MIT.txt",
    "generators/licenses/MBGAN-GPL-3.0.txt",
    "generators/vendor/torchdyn_1_0_6/LICENSE",
}
ROOT_DOCS = {"README.md", "PREPROCESSING.md"}


def allowed(name):
    p = PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts:
        return False
    if name in SPECIAL or name in ROOT_DOCS:
        return True
    if p.parts[0] not in {"scripts", "generators", "hpc"}:
        return False
    if any(part in {"data", "assets", "output", "results", "checkpoints", "__pycache__"}
           for part in p.parts):
        return False
    return p.suffix in {".py", ".R", ".md", ".sh", ".slurm"}


def git(*args):
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--revision", help="Check a committed tree, e.g. HEAD")
    mode.add_argument("--directory", action="store_true", help="Check every source file on disk")
    args = parser.parse_args()
    files = []
    if args.directory:
        for p in sorted(ROOT.rglob("*")):
            if ".git" in p.relative_to(ROOT).parts:
                continue
            if p.is_symlink():
                raise SystemExit(f"Symlinks are not allowed: {p.relative_to(ROOT)}")
            if p.is_file():
                files.append((p.relative_to(ROOT).as_posix(), p.read_bytes()))
    else:
        if args.revision:
            records = git("ls-tree", "-rz", "--full-tree", args.revision).split(b"\0")
            for record in filter(None, records):
                metadata, name = record.split(b"\t", 1)
                permissions, kind, oid = metadata.decode().split()
                if kind != "blob" or permissions not in {"100644", "100755"}:
                    raise SystemExit(f"Non-source object: {name.decode()}")
                files.append((name.decode(), git("cat-file", "blob", oid)))
        else:
            for record in filter(None, git("ls-files", "--stage", "-z").split(b"\0")):
                metadata, name = record.split(b"\t", 1)
                permissions, oid, stage = metadata.decode().split()
                if stage != "0" or permissions not in {"100644", "100755"}:
                    raise SystemExit(f"Unmerged or non-source entry: {name.decode()}")
                files.append((name.decode(), git("cat-file", "blob", oid)))
    if not files:
        raise SystemExit("No source files found")
    errors = []
    for name, content in files:
        if not allowed(name):
            errors.append(f"Forbidden artifact or unrecognized source path: {name}")
        if len(content) > 1_000_000 or b"\0" in content:
            errors.append(f"Binary or oversized payload: {name}")
        try:
            content.decode("utf-8")
        except UnicodeDecodeError:
            errors.append(f"Non-text payload: {name}")
        if name.endswith(".json"):
            try:
                json.loads(content)
            except (ValueError, UnicodeDecodeError):
                errors.append(f"Invalid source configuration: {name}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"PASS: {len(files)} source, configuration, license, and documentation files; "
          "no data, checkpoints, or figure artifacts in the checked tree.")


if __name__ == "__main__":
    main()
