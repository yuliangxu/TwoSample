#!/usr/bin/env python3
"""Verify the exact curated source, inputs, and final result exports."""
import hashlib
import json
from pathlib import Path
import sys

from prepare_run import validate_generation

ROOT = Path(__file__).resolve().parents[1]

def verify_completed_report(results):
    def digest(name):
        return hashlib.sha256((results / name).read_bytes()).hexdigest()
    report = json.loads((results / 'report_manifest.json').read_text())
    complete = json.loads((results / 'COMPLETE').read_text())
    checks = {'case_study_report.pdf': 'sha256',
              'analysis_validation.json': 'analysis_validation_sha256',
              'ANALYSIS_COMPLETE': 'analysis_complete_sha256',
              'input_manifest.json': 'input_manifest_sha256',
              'settings.json': 'settings_sha256'}
    for name, field in checks.items():
        if digest(name) != report[field]:
            raise RuntimeError(f'Completed report provenance differs: {name}')
    if (complete['report_sha256'] != digest('case_study_report.pdf')
            or complete['report_manifest_sha256'] != digest('report_manifest.json')):
        raise RuntimeError('Report completion marker differs from the completed PDF or manifest')
    for figure in report['figures']:
        if digest(figure['source']) != figure['sha256']:
            raise RuntimeError(f"Report figure changed: {figure['source']}")
    return report


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
    verify_completed_report(results)
    if (results / 'BATTS_commit.txt').read_text().strip() != manifest['BATTS_commit']:
        raise RuntimeError('BATTS commit mismatch')
    if (manifest.get('pretrained_model_inputs') != []
            or manifest.get('independent_full_training_runs') != 2
            or not validation.get('all_generators_trained_from_scratch')
            or validation.get('independent_training_reproductions_verified') != 4):
        raise RuntimeError('Complete from-training reproduction was not verified')
    inputs = json.loads((results / 'input_manifest.json').read_text())['inputs']
    for name, entry in inputs.items():
        if hashlib.sha256((ROOT / 'data' / name).read_bytes()).hexdigest() != entry['sha256']:
            raise RuntimeError(f'Final input differs from fitted analysis: {name}')
    validate_generation(ROOT / 'data/generation_manifest.json', inputs, ROOT)
    print(f"Verified {len(manifest['files'])} source/input/result files, final completion markers and BATTS commit.")
    print(f"Model order: {', '.join(manifest['model_order'])}")
    print('Detailed numerical and PDF checks: results/final/analysis_validation.json and report_manifest.json')

if __name__ == '__main__':
    main()
