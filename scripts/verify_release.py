#!/usr/bin/env python3
"""Verify the exact curated source, inputs, and final result exports."""
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

def main():
    manifest = json.loads((ROOT / 'release_manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if not path.is_file():
            raise RuntimeError(f'Missing release file: {name}')
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f'Changed release file: {name}')
    results = ROOT / 'results/final'
    for marker in ('ANALYSIS_COMPLETE', 'COMPLETE'):
        if not (results / marker).is_file():
            raise RuntimeError(f'Final analysis missing {marker}')
    validation = json.loads((results / 'analysis_validation.json').read_text())
    report = json.loads((results / 'report_manifest.json').read_text())
    if (results / 'BATTS_commit.txt').read_text().strip() != manifest['BATTS_commit']:
        raise RuntimeError('BATTS commit mismatch')
    print(f"Verified {len(manifest['files'])} source/input/result files, final completion markers and BATTS commit.")
    print(f"Model order: {', '.join(manifest['model_order'])}")
    print('Detailed numerical and PDF checks: results/final/analysis_validation.json and report_manifest.json')

if __name__ == '__main__':
    main()
