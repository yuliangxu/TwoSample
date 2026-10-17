#!/usr/bin/env python3
"""Freeze analysis inputs, executable source, and verified generator training evidence."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from generators.cohort import cohort_binding, expected_rows, runtime_data, external_path
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


def validate_generation(path, inputs, source_root, data_dir=None):
    """Check the training-to-samples chain and two complete independent reruns."""
    path = Path(path)
    require(path.is_file(), 'Missing generation_manifest.json: verified from-training evidence is required')
    record = json.loads(path.read_text())
    data = runtime_data(data_dir or path.parent)
    config, cohort = cohort_binding(source_root, data_dir=data)
    require(record.get('cohort_id') == config['cohort_id'],
            'Generator evidence belongs to a different cohort')
    cohort_name = config['cohort_manifest']
    require(record.get('cohort_manifest_sha256') == sha(data / cohort_name),
            'Generator evidence has a different corrected-cohort manifest')
    require(record['schema_version'] == 2 and record['mode'] == 'from_training',
            'Expected runtime generation manifest schema 2 and mode from_training')
    require(record['model_order'] == list(MODELS) and set(record['models']) == set(MODELS),
            'Generation model order or model set differs')
    train_hash = inputs['sample_train.csv']['sha256']
    require(record['training_input'] == {'sha256': train_hash,
                                        'shape': [config['training_rows'], config['taxa']]},
            'Generator training data differ from the corrected training matrix')
    require(record['test_input_sha256'] == inputs['sample_test.csv']['sha256'],
            'Generator provenance has a different real test matrix')
    for split in ('train', 'test'):
        name = f'sample_{split}.csv'
        require(inputs[name]['sha256'] == cohort['files'][name],
                f'{split}: input does not belong to the verified corrected cohort')
    require(record['test_used_for_training'] is False and record['test_used_for_seed_or_checkpoint_selection'] is False,
            'Test observations were used for generator fitting or seed/checkpoint selection')
    require(record['mbgan_variant_choice'] == 'retrospective_restoration_after_reviewing_prior_results',
            'Retrospective MBGAN variant choice must be disclosed')
    require(bool(record['source_hashes']), 'Generation source hashes are required')
    require({'generators/training_config.json', 'generators/cohort.py'}.issubset(record['source_hashes']),
            'Generation evidence must bind executable source and the recipe')
    require(record.get('cohort_files') == cohort['files'], 'Runtime input/asset manifest differs')
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
    require(reproduction['status'] == 'complete' and reproduction['method'] == 'independent_rerun'
            and set(reproduction['models']) == set(MODELS),
            'Expected two complete independent from-training reproductions')
    common = {'threads': 2}
    recipes = {
        'icfm': dict(common, device='cpu', training_seed=20261003, sampling_seed=2, steps=20000,
                     batch_size=256, optimizer='Adam', learning_rate=.001,
                     hidden_layers=3, width=64, sigma=.0001),
        'dt': dict(training_seed=None, sampling_seed=1),
        'd': dict(training_seed=None, sampling_seed=1),
        'mbgan': dict(common, device='cpu', training_seed=1, sampling_seed=256, iterations=500000,
                      batch_size=32, optimizer='RMSprop', learning_rate=.00005,
                      critic_updates=5, gradient_penalty_connected=False,
                      variant='historical_local_no_penalty',
                      callback_rng_policy='isolated_seeded_diagnostics', callback_interval=10000),
    }
    for key in MODELS:
        model = record['models'][key]
        require(model['trained_from_scratch'] is True
                and model['training_input_sha256'] == train_hash,
                f'{key}: training was not verified from the preserved training rows')
        expected_initialization = 'random' if key in ('icfm', 'mbgan') else 'fit_from_training'
        require(model['initialization'] == expected_initialization,
                f'{key}: unexpected initialization')
        model_config = model['configuration']
        require(model_config == config['models'][key],
                f'{key}: configuration differs from the corrected-cohort training protocol')
        for field, value in recipes[key].items():
            require(model_config[field] == value, f'{key}: frozen recipe differs for {field}')
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
                and metadata['test_used_for_seed_or_checkpoint_selection'] is False,
                f'{key}: training metadata indicate generator test use')
        parameter_hash = metadata['final_parameter_sha256']
        require(isinstance(parameter_hash, str) and len(parameter_hash) == 64
                and paired['first_parameter_sha256'] == paired['second_parameter_sha256']
                == parameter_hash and paired['parameter_arrays_identical'] is True
                and paired['exact_numeric_match'] is True,
                f'{key}: independent parameter reproduction was not verified')
    return record


def verify_frozen_generation(run, manifest=None):
    manifest = manifest or json.loads((run / 'input_manifest.json').read_text())
    for name, item in manifest['inputs'].items():
        require(sha(run / 'inputs' / name) == item['sha256'], f'Frozen input hash changed: {name}')
    for name, digest in manifest['cohort_provenance'].items():
        require(sha(relative_file(run / 'provenance', name)) == digest,
                f'Frozen cohort provenance changed: {name}')
    generation = manifest['generation']
    path = relative_file(run, generation['manifest'])
    require(sha(path) == generation['sha256'], 'Frozen generator manifest hash changed')
    return validate_generation(path, manifest['inputs'], run / 'generation' / 'source', data_dir=run / 'inputs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, help='External prepared generation/data directory')
    parser.add_argument('--generation-manifest', type=Path)
    parser.add_argument('--verify-generation-only', action='store_true')
    args = parser.parse_args()
    run = external_path(args.run_root, "Analysis output")
    if args.verify_generation_only:
        verify_frozen_generation(run)
        print('From-training generation evidence verified for all four models.')
        return
    data = runtime_data(args.data_dir)
    config, cohort = cohort_binding(ROOT, data_dir=data)
    counts = expected_rows(config)
    inputs = {}
    for key in KEYS:
        name = f'sample_{key}.csv'
        path = data / name
        with path.open(newline='') as stream:
            rows = [[float(v) for v in row] for row in csv.reader(stream)]
        expected, taxa = counts[key], config['taxa']
        require(len(rows) == expected and all(len(row) == taxa for row in rows),
                f'{name}: expected ({expected},{taxa})')
        require(all(math.isfinite(v) and 0 <= v <= 1 for row in rows for v in row),
                f'{name}: invalid composition values')
        require(max(abs(sum(row)-1) for row in rows) <= 1e-6, f'{name}: invalid row sums')
        inputs[name] = {'sha256': sha(path), 'shape': [expected, taxa]}
    generation_path = (args.generation_manifest or data / 'generation_manifest.json').resolve()
    generation = validate_generation(generation_path, inputs, ROOT, data_dir=data)
    scripts = sorted(p for p in (ROOT / 'scripts').rglob('*')
                     if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    scripts += [ROOT / 'generators/training_config.json', ROOT / 'generators/cohort.py',
                ROOT / 'generators/__init__.py']
    source_hashes = {'source/' + str(p.relative_to(ROOT)): sha(p) for p in scripts}
    provenance_root = data / 'provenance'
    provenance = {str(p.relative_to(provenance_root)): sha(p)
                  for p in sorted(provenance_root.rglob('*')) if p.is_file()}
    manifest = {'BATTS_commit': COMMIT, 'BATTS_version': '0.3.0',
                'cohort_id': config['cohort_id'], 'cohort_provenance': provenance, 'configuration': config,
                'model_order': list(MODELS), 'inputs': inputs,
                'source_scripts': source_hashes,
                'generation': {'manifest': 'generation/generation_manifest.json',
                               'sha256': sha(generation_path)},
                'preprocessing': f"Corrected {cohort['observed_rows']}-observation four-study cohort; collection-wide completeness, taxon and retained-abundance filters, no zero-count rule, sample-ID-matched study membership, retained-taxon normalization, observation split with seed 1 and float32 precision. Dimensions and provenance are bound to the cohort manifest.",
                'generator_scope': f"All four generators fitted from {config['training_rows']} corrected training rows; {config['test_rows']} observations reserved for generator assessment. Neural models start from seeded random initialization. Final planned updates and no test use for generator fitting or a new seed/checkpoint search. The inherited historical local MBGAN variant was selected retrospectively in the earlier cohort analysis; all recipes were frozen before fitting this corrected cohort."}
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
    for name in [*cohort['files'], config['cohort_manifest']]:
        dest = run / 'inputs' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if (data / name).resolve() != dest.resolve():
            shutil.copy2(data / name, dest)
    for src in scripts:
        dest = run / 'source' / src.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    for name in provenance:
        dest = run / 'provenance' / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(provenance_root / name, dest)
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
