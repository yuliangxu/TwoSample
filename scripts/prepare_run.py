#!/usr/bin/env python3
"""Freeze analysis inputs, executable source, and verified generator training evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '147e511d211919afefb8decac6333e66d24919db'
MODELS = ('icfm', 'dt', 'd', 'mbgan')
KEYS = ('train', 'test', *MODELS)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_file(root, name):
    path = Path(name)
    require(not path.is_absolute() and '..' not in path.parts, f'Unsafe evidence path: {name}')
    full = root / path
    require(full.is_file() and full.resolve().is_relative_to(root.resolve()),
            f'Missing or external evidence file: {name}')
    return full


def validate_generation(path, inputs, source_root):
    """Check the training-to-samples chain and two complete independent reruns."""
    path = Path(path)
    require(path.is_file(), 'Missing generation_manifest.json: verified from-training evidence is required')
    record = json.loads(path.read_text())
    require(record['schema_version'] == 1 and record['mode'] == 'from_training',
            'Expected generation manifest schema 1 and mode from_training')
    require(record['model_order'] == list(MODELS) and set(record['models']) == set(MODELS),
            'Generation model order or model set differs')
    train_hash = inputs['sample_train.csv']['sha256']
    require(record['training_input'] == {'sha256': train_hash, 'shape': [1166, 123]},
            'Generator training data differ from the preserved training matrix')
    require(record['test_input_sha256'] == inputs['sample_test.csv']['sha256'],
            'Generator provenance has a different real test matrix')
    require(record['test_used_for_training'] is False and record['test_used_for_selection'] is False,
            'Test observations were used for generator fitting or selection')
    require(bool(record['source_hashes']), 'Generation source hashes are required')
    for name, digest in record['source_hashes'].items():
        require(sha(relative_file(source_root, name)) == digest,
                f'Generation source hash changed: {name}')
    artifacts = record['validation_artifacts']
    require(bool(artifacts), 'Training and reproduction evidence files are required')
    for name, digest in artifacts.items():
        require(sha(relative_file(path.parent, name)) == digest,
                f'Generation evidence hash changed: {name}')
    reproduction_name = record['reproduction_evidence']
    require(reproduction_name in artifacts, 'Reproduction evidence must be hash-bound')
    reproduction = json.loads(relative_file(path.parent, reproduction_name).read_text())
    require(reproduction['method'] == 'independent_rerun'
            and set(reproduction['models']) == set(MODELS),
            'Expected two complete independent from-training reproductions')
    common = {'threads': 2}
    recipes = {
        'icfm': dict(common, device='cpu', training_seed=20261003, sampling_seed=2, steps=20000,
                     batch_size=256, optimizer='Adam', learning_rate=.001,
                     hidden_layers=3, width=64, sigma=.0001),
        'dt': dict(training_seed=None, sampling_seed=1),
        'd': dict(training_seed=None, sampling_seed=1),
        'mbgan': dict(common, device='gpu', training_seed=1, sampling_seed=256, iterations=500000,
                      batch_size=32, optimizer='RMSprop', learning_rate=.00005,
                      critic_updates=5, gradient_penalty_connected=True),
    }
    for key in MODELS:
        model = record['models'][key]
        require(model['trained_from_scratch'] is True
                and model['training_input_sha256'] == train_hash,
                f'{key}: training was not verified from the preserved training rows')
        expected_initialization = 'random' if key in ('icfm', 'mbgan') else 'fit_from_training'
        require(model['initialization'] == expected_initialization,
                f'{key}: unexpected initialization')
        config = model['configuration']
        for field, value in recipes[key].items():
            require(config[field] == value, f'{key}: frozen recipe differs for {field}')
        expected_steps = {'icfm': 20000, 'mbgan': 500000}.get(key)
        require(model['completion']['completed'] is True
                and model['completion']['completed_steps'] == expected_steps,
                f'{key}: training budget is incomplete')
        expected_output = inputs[f'sample_{key}.csv']
        require(model['output'] == expected_output, f'{key}: generated data differ from manifest')
        verification = model['verification']
        require(verification['passed'] is True and verification['method'] == 'independent_rerun'
                and verification['output_sha256'] == expected_output['sha256'],
                f'{key}: independent retraining reproduction was not verified')
        paired = reproduction['models'][key]
        require(paired['matched'] is True
                and paired['first_output_sha256'] == paired['second_output_sha256']
                == expected_output['sha256'], f'{key}: independent output hashes disagree')
        metadata_name = model['training_metadata']
        require(metadata_name in artifacts, f'{key}: training metadata must be hash-bound')
        metadata = json.loads(relative_file(path.parent, metadata_name).read_text())
        require(metadata['model'] == key and bool(metadata['trainer_metadata']),
                f'{key}: native training metadata are required')
        for field in ('training_input_sha256', 'trained_from_scratch', 'initialization',
                      'configuration', 'completion'):
            require(metadata[field] == model[field], f'{key}: training metadata differ for {field}')
        require(metadata['test_used_for_training'] is False
                and metadata['test_used_for_selection'] is False,
                f'{key}: training metadata indicate generator test use')
    return record


def verify_frozen_generation(run, manifest=None):
    manifest = manifest or json.loads((run / 'input_manifest.json').read_text())
    for name, item in manifest['inputs'].items():
        require(sha(run / 'inputs' / name) == item['sha256'], f'Frozen input hash changed: {name}')
    generation = manifest['generation']
    path = relative_file(run, generation['manifest'])
    require(sha(path) == generation['sha256'], 'Frozen generator manifest hash changed')
    return validate_generation(path, manifest['inputs'], run / 'generation' / 'source')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=ROOT / 'data')
    parser.add_argument('--generation-manifest', type=Path)
    parser.add_argument('--verify-generation-only', action='store_true')
    args = parser.parse_args()
    run, data = args.run_root.resolve(), args.data_dir.resolve()
    if args.verify_generation_only:
        verify_frozen_generation(run)
        print('From-training generation evidence verified for all four models.')
        return
    if run == ROOT or run == ROOT / 'data' or run == ROOT / 'results':
        parser.error('Use a dedicated output directory.')
    inputs = {}
    for key in KEYS:
        name = f'sample_{key}.csv'
        path = data / name
        with path.open(newline='') as stream:
            rows = [[float(v) for v in row] for row in csv.reader(stream)]
        expected = {'train': 1166, 'test': 292}.get(key, 1000)
        require(len(rows) == expected and all(len(row) == 123 for row in rows),
                f'{name}: expected ({expected},123)')
        require(all(math.isfinite(v) and 0 <= v <= 1 for row in rows for v in row),
                f'{name}: invalid composition values')
        require(max(abs(sum(row)-1) for row in rows) <= 1e-6, f'{name}: invalid row sums')
        inputs[name] = {'sha256': sha(path), 'shape': [expected, 123]}
    generation_path = (args.generation_manifest or data / 'generation_manifest.json').resolve()
    generation = validate_generation(generation_path, inputs, ROOT)
    scripts = sorted(p for p in (ROOT / 'scripts').rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    source_hashes = {'source/' + str(p.relative_to(ROOT)): sha(p) for p in scripts}
    manifest = {'BATTS_commit': COMMIT, 'BATTS_version': '0.3.0',
                'model_order': list(MODELS), 'inputs': inputs,
                'source_scripts': source_hashes,
                'generation': {'manifest': 'generation/generation_manifest.json',
                               'sha256': sha(generation_path)},
                'preprocessing': 'Original pooled 1166/292 by 123 matrices and published split; collection-wide filtering, retained-taxon normalization and float32 precision.',
                'generator_scope': 'All four generators fitted from 1166 preserved training rows. Neural models start from seeded random initialization. Frozen recipes and final planned updates; no test use for generator fitting or selection.'}
    run.mkdir(parents=True, exist_ok=True)
    target = run / 'input_manifest.json'
    if target.exists():
        prior = json.loads(target.read_text())
        require(prior == manifest, 'Existing run has different inputs, source, or training evidence. Choose a new output directory.')
        for name, item in inputs.items():
            require(sha(run / 'inputs' / name) == item['sha256'], f'Frozen input changed: {name}')
        for name, digest in source_hashes.items():
            require(sha(run / name) == digest, f'Frozen source changed: {name}')
        verify_frozen_generation(run, manifest)
        print('Existing frozen inputs, source, and generator evidence verified.')
        return
    require(not any(run.glob('fits/*/*/*.rds')),
            'Existing fit files without a manifest; choose a new output directory.')
    (run / 'inputs').mkdir(exist_ok=True)
    for name in inputs:
        dest = run / 'inputs' / name
        if (data / name).resolve() != dest.resolve():
            shutil.copy2(data / name, dest)
    for src in scripts:
        dest = run / 'source' / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    generation_dir = run / 'generation'
    generation_dir.mkdir(exist_ok=True)
    shutil.copy2(generation_path, generation_dir / 'generation_manifest.json')
    for name in generation['validation_artifacts']:
        dest = generation_dir / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(generation_path.parent / name, dest)
    for name in generation['source_hashes']:
        dest = generation_dir / 'source' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, dest)
    verify_frozen_generation(run, manifest)
    target.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Frozen six matrices, {len(scripts)} analysis source files and verified generator evidence in {run}')


if __name__ == '__main__':
    main()
