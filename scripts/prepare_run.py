#!/usr/bin/env python3
"""Freeze final analysis inputs and executable source before fitting."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '147e511d211919afefb8decac6333e66d24919db'
KEYS = ('train', 'test', 'icfm', 'dt', 'd', 'mbgan')

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    args = parser.parse_args()
    run, data = args.run_root.resolve(), args.data_dir.resolve()
    if run == ROOT or run == ROOT / 'data' or run == ROOT / 'results':
        parser.error('Use a dedicated output directory.')
    inputs = {}
    for key in KEYS:
        name = f'sample_{key}.csv'
        path = data / name
        with path.open(newline='') as stream:
            rows = [[float(v) for v in row] for row in csv.reader(stream)]
        expected = {'train': 1166, 'test': 292}.get(key, 1000)
        if len(rows) != expected or any(len(row) != 123 for row in rows):
            raise ValueError(f'{name}: expected ({expected},123)')
        if any(not math.isfinite(v) or v < 0 or v > 1 for row in rows for v in row):
            raise ValueError(f'{name}: invalid composition values')
        if max(abs(sum(row)-1) for row in rows) > 1e-6:
            raise ValueError(f'{name}: invalid row sums')
        inputs[name] = {'sha256': sha(path), 'shape': [expected,123]}
    scripts = sorted(p for p in (ROOT / 'scripts').rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    source_hashes = {'source/' + str(p.relative_to(ROOT)): sha(p) for p in scripts}
    manifest = {'BATTS_commit': COMMIT, 'BATTS_version': '0.3.0',
                'model_order': ['icfm','dt','d','mbgan'], 'inputs': inputs,
                'source_scripts': source_hashes,
                'preprocessing': 'Original pooled 1166/292 by 123 matrices and published split; retained-taxon normalization and float32 precision.',
                'generator_scope': 'D and DT fit original training rows. Fixed ICFM and MBGAN checkpoint replay; historical neural fitting membership unknown.'}
    run.mkdir(parents=True, exist_ok=True)
    target = run / 'input_manifest.json'
    if target.exists():
        prior = json.loads(target.read_text())
        if prior != manifest:
            raise RuntimeError('Existing run has different inputs or source. Choose a new output directory.')
        for name,item in inputs.items():
            if sha(run / 'inputs' / name) != item['sha256']:
                raise RuntimeError(f'Frozen input changed: {name}')
        for name,digest in source_hashes.items():
            if sha(run / name) != digest:
                raise RuntimeError(f'Frozen source changed: {name}')
        print('Existing frozen inputs/source verified.')
        return
    if any(run.glob('fits/*/*/*.rds')):
        raise RuntimeError('Existing fit files without a manifest; choose a new output directory.')
    (run / 'inputs').mkdir(exist_ok=True)
    for name in inputs:
        dest = run / 'inputs' / name
        if (data / name).resolve() != dest.resolve():
            shutil.copy2(data / name, dest)
    for src in scripts:
        dest = run / 'source' / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    target.write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Frozen six matrices and {len(scripts)} executable source files in {run}')

if __name__ == '__main__':
    main()
