#!/usr/bin/env python3
"""HPC runner: study-only preprocessing, fresh generators, BATTS, seven figures."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BATTS_COMMIT = '77c217297910a5ba50b71313e8289d9024b669c9'
MODELS = ('d', 'dt', 'icfm', 'mbgan')
RESOURCE = '2021-10-14.HMP_2019_ibdmdb.relative_abundance'


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def validate_matrix(path, columns=None, rows=None):
    n = 0
    width = None
    max_sum_error = 0.0
    with path.open(newline='') as handle:
        for row in csv.reader(handle):
            values = [float(v) for v in row]
            if width is None:
                width = len(values)
            if len(values) != width or not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise ValueError(f'Invalid composition in {path}, row {n+1}')
            error = abs(sum(values)-1)
            if error > 1e-5:
                raise ValueError(f'Row {n+1} of {path} does not sum to one')
            max_sum_error = max(max_sum_error, error)
            n += 1
    if n < 2 or width is None or width < 2 or (columns is not None and width != columns) or (rows is not None and n != rows):
        raise ValueError(f'Unexpected dimensions for {path}: {n} x {width}')
    return {'rows': n, 'columns': width, 'max_row_sum_error': max_sum_error, 'sha256': sha256(path)}


class Experiment:
    def __init__(self, args):
        self.args = args
        self.run = args.output.resolve()
        if self.run == (ROOT/'data').resolve() or (ROOT/'data').resolve() in self.run.parents:
            raise ValueError('Output cannot be the bundled data directory')
        self.run.mkdir(parents=True, exist_ok=True)
        self.inputs = self.run/'inputs'
        self.tree = self.run/'tree'
        self.logs = self.run/'logs'
        self.logs.mkdir(exist_ok=True)
        self.env = os.environ.copy()
        libraries = [str(args.r_lib.resolve())]
        if self.env.get('R_LIBS'):
            libraries.append(self.env['R_LIBS'])
        self.env.update(R_LIBS=os.pathsep.join(libraries), TWO_SAMPLE_R_LIB=str(args.r_lib.resolve()),
                        TWO_SAMPLE_RUN_ROOT=str(self.run), TWO_SAMPLE_DATA_DIR=str(self.inputs),
                        TWO_SAMPLE_BATTS_COMMIT=BATTS_COMMIT,
                        TWO_SAMPLE_RATIO_ROOT=str(self.run/'revision'),
                        TWO_SAMPLE_FIGURE_DIR=str(self.run/'figures'),
                        TWO_SAMPLE_NULL_OUTPUT=str(self.run/'null'),
                        TWO_SAMPLE_NULL_INTERVALS=str(self.run/'null/test_pointwise_credible_intervals.csv'))

    def command(self, label, argv):
        argv = [str(a) for a in argv]
        write_json(self.logs/(label+'.command.json'), {'argv': argv, 'cwd': str(ROOT),
                   'started_utc': datetime.now(timezone.utc).isoformat()})
        print('Running', label, '(log:', self.logs/(label+'.log'), ')', flush=True)
        with (self.logs/(label+'.log')).open('w') as log:
            subprocess.run(argv, cwd=ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT, check=True)

    def r(self, script, *args):
        return [self.args.rscript, ROOT/'scripts'/script, *args]

    def real_files(self):
        return [self.inputs/'sample_train.csv', self.inputs/'sample_test.csv', self.inputs/'taxa.txt',
                self.inputs/'samples.csv', *[self.tree/f for f in ('childNode.csv', 'treeTaxa.csv', 'outTaxa.csv')]]

    def prepared(self):
        taxa = self.inputs/'taxa.txt'
        columns = len(taxa.read_text().splitlines())
        real = {s: validate_matrix(self.inputs/f'sample_{s}.csv', columns) for s in ('train', 'test')}
        with (self.inputs/'samples.csv').open(newline='') as handle:
            records = list(csv.DictReader(handle))
        if not records or any(r['study_name'] != 'HMP_2019_ibdmdb' for r in records):
            raise ValueError('Sample provenance contains a different study')
        train_ids = {r['sample_id'] for r in records if r['split'] == 'train'}
        test_ids = {r['sample_id'] for r in records if r['split'] == 'test'}
        if train_ids & test_ids or len(train_ids) != real['train']['rows'] or len(test_ids) != real['test']['rows']:
            raise ValueError('Invalid split membership')
        if len(records) != real['train']['rows'] + real['test']['rows']:
            raise ValueError('Invalid split membership: duplicate or unassigned observations')
        return real

    def require_preparation(self):
        fingerprint = {'resource': RESOURCE, 'seed': self.args.split_seed,
                       'source_sha256': sha256(ROOT/'scripts/prepare_ibdmdb.R')}
        if not self.cached(self.run/'preprocessing_manifest.json', fingerprint):
            raise ValueError('Run --stage prepare first to establish verified preprocessing provenance')
        return self.prepared()

    def cached(self, marker, fingerprint):
        if not marker.exists():
            return False
        saved = json.loads(marker.read_text())
        if saved['fingerprint'] != fingerprint:
            raise ValueError(f'Run settings changed for {marker.name}; choose a new --output directory')
        for name, checksum in saved['outputs'].items():
            path = self.run/name
            if not path.exists() or sha256(path) != checksum:
                raise ValueError(f'Completed output changed or missing: {path}')
        print('Validated completed stage:', marker.stem, flush=True)
        return True

    def complete(self, marker, fingerprint, paths):
        write_json(marker, {'fingerprint': fingerprint,
                   'outputs': {str(p.relative_to(self.run)): sha256(p) for p in paths}})

    def prepare(self):
        marker = self.run/'preprocessing_manifest.json'
        fingerprint = {'resource': RESOURCE, 'seed': self.args.split_seed,
                       'source_sha256': sha256(ROOT/'scripts/prepare_ibdmdb.R')}
        if self.cached(marker, fingerprint):
            self.prepared()
            return
        if any((self.inputs/f'sample_{key}.csv').exists() for key in MODELS):
            raise ValueError('Synthetic data already exist without a validated preparation; use a new output directory')
        argv = self.r('prepare_ibdmdb.R', f'--output={self.run}', f'--seed={self.args.split_seed}')
        if self.args.cache_dir:
            argv.append(f'--cache-dir={self.args.cache_dir.resolve()}')
        if self.args.offline:
            argv.append('--offline')
        self.command('prepare', argv)
        self.prepared()
        self.complete(marker, fingerprint, self.real_files() + [self.run/'preprocessing_settings.R',
                      self.run/'sample_filter_audit.csv', self.run/'taxon_filter_audit.csv'])

    def model_fingerprint(self, key):
        source = ROOT/'generators'/f'{dict(d="dirichlet", dt="dirichlet_tree", icfm="icfm", mbgan="mbgan")[key]}.py'
        fingerprint = {'model': key, 'train_sha256': sha256(self.inputs/'sample_train.csv'),
                       'taxonomy_sha256': sha256(self.inputs/'taxa.txt'),
                       'tree_sha256': {p.name: sha256(p) for p in self.real_files()[4:]},
                       'n_synthetic': self.args.n_synthetic, 'source_sha256': sha256(source),
                       'tree_source_sha256': sha256(ROOT/'generators/tree.py')}
        if key == 'icfm':
            fingerprint.update(steps=self.args.icfm_steps, seed=self.args.icfm_seed,
                               device=self.args.icfm_device, threads=self.args.threads)
        if key == 'mbgan':
            fingerprint.update(iterations=self.args.mbgan_iterations, seed=self.args.mbgan_seed,
                               device=self.args.mbgan_device, threads=self.args.threads)
        return fingerprint

    def model(self, key):
        real = self.require_preparation()
        py = self.args.python_mbgan if key == 'mbgan' else self.args.python_generators
        if not py.is_file():
            raise FileNotFoundError(f'Missing Python environment: {py}')
        fingerprint = self.model_fingerprint(key)
        marker = self.run/f'generator_{key}_complete.json'
        if self.cached(marker, fingerprint):
            validate_matrix(self.inputs/f'sample_{key}.csv', real['train']['columns'], self.args.n_synthetic)
            return
        model_dir = self.run/'generators'/key
        if model_dir.exists():
            raise ValueError(f'Incomplete generator directory exists: {model_dir}. Preserve it and use a new output directory; neural training does not resume mid-run.')
        model_dir.parent.mkdir(exist_ok=True)
        output = self.inputs/f'sample_{key}.csv'
        train = self.inputs/'sample_train.csv'
        if key in ('d', 'dt'):
            module = 'generators.dirichlet' if key == 'd' else 'generators.dirichlet_tree'
            argv = [py, '-m', module, 'fit-sample', '--train', train,
                    '--checkpoint', model_dir/'model.npz', '--output', output,
                    '--seed', '1', '--n-samples', str(self.args.n_synthetic), '--report', model_dir/'fit.json']
            if key == 'dt':
                argv.extend(['--tree', self.tree])
            self.command('generator_'+key, argv)
        elif key == 'icfm':
            self.command('icfm_train', [py, '-m', 'generators.icfm', 'train', '--train', train,
                 '--tree', self.tree, '--output', model_dir/'model.pt', '--steps', str(self.args.icfm_steps),
                 '--seed', str(self.args.icfm_seed), '--threads', str(self.args.threads), '--device', self.args.icfm_device])
            self.command('icfm_sample', [py, '-m', 'generators.icfm', 'sample', '--checkpoint', model_dir/'model.pt',
                 '--tree', self.tree, '--output', output, '--n', str(self.args.n_synthetic), '--seed', '2',
                 '--threads', str(self.args.threads), '--device', self.args.icfm_device])
        else:
            self.command('mbgan_train', [py, ROOT/'generators/mbgan.py', 'train', '--train', train,
                 '--taxonomy', self.inputs/'taxa.txt', '--output-dir', model_dir,
                 '--iterations', str(self.args.mbgan_iterations), '--seed', str(self.args.mbgan_seed),
                 '--threads', str(self.args.threads), '--device', self.args.mbgan_device])
            self.command('mbgan_sample', [py, ROOT/'generators/mbgan.py', 'sample', '--checkpoint', model_dir/'generator.npz',
                 '--output', output, '--n', str(self.args.n_synthetic), '--seed', '256',
                 '--threads', str(self.args.threads), '--device', self.args.mbgan_device])
        validate_matrix(output, real['train']['columns'], self.args.n_synthetic)
        self.complete(marker, fingerprint, [output, *[p for p in sorted(model_dir.rglob('*')) if p.is_file()]])

    def require_models(self):
        real = self.require_preparation()
        for key in MODELS:
            marker = self.run/f'generator_{key}_complete.json'
            if not marker.exists():
                raise ValueError(f'Generator {key} has not completed')
            saved = json.loads(marker.read_text())
            if saved['fingerprint']['train_sha256'] != real['train']['sha256']:
                raise ValueError(f'Generator {key} was trained on different data')
            self.cached(marker, self.model_fingerprint(key))
            validate_matrix(self.inputs/f'sample_{key}.csv', real['train']['columns'], self.args.n_synthetic)

    def batts(self):
        self.require_models()
        self.command('batts', self.r('refit_case_study.R', str(self.args.workers)))
        self.command('null', self.r('run_figure_S12_null_experiment.R'))

    def figures(self):
        self.require_models()
        self.command('figures', self.r('recreate_all_case_study_figures.R'))

    def report(self):
        self.require_models()
        self.command('report', self.r('summarize_ibdmdb.R'))
        figures = [self.run/f'figures/figure{k}_ggplot2.pdf' for k in ('6','7','S8','S9','S10','S11','S12')]
        for path in figures:
            with path.open('rb') as handle:
                if handle.read(5) != b'%PDF-':
                    raise ValueError(f'Invalid figure PDF: {path}')
        write_json(self.run/'experiment_provenance.json', {'scope': 'Full-study IBDMDB-only filtering, fresh generators, BATTS comparisons',
                   'BATTS_commit': BATTS_COMMIT, 'resource': RESOURCE,
                   'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(self.args).items()},
                   'real': self.prepared(), 'generator_checkpoints': 'All four newly fitted on this experiment training CSV',
                   'figure_sha256': {str(path.relative_to(self.run)): sha256(path) for path in figures},
                   'figure_outputs': [f'figures/figure{k}_ggplot2.pdf' for k in ('6','7','S8','S9','S10','S11','S12')]})
        (self.run/'COMPLETE').write_text('All four generators, eight BATTS comparisons, null and seven figures completed.\n')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('prepare', *MODELS, 'batts', 'figures', 'report', 'all'), default='all')
    p.add_argument('--output', type=Path, default=ROOT/'output/ibdmdb_study')
    p.add_argument('--python-generators', type=Path, default=ROOT/'.venv-generators/bin/python')
    p.add_argument('--python-mbgan', type=Path, default=ROOT/'.venv-mbgan/bin/python')
    p.add_argument('--rscript', default='Rscript')
    p.add_argument('--r-lib', type=Path, default=ROOT/'output/ibdmdb_study/r_lib')
    p.add_argument('--cache-dir', type=Path)
    p.add_argument('--offline', action='store_true')
    p.add_argument('--split-seed', type=int, default=20261003)
    p.add_argument('--icfm-seed', type=int, default=20261003)
    p.add_argument('--mbgan-seed', type=int, default=1)
    p.add_argument('--icfm-steps', type=int, default=20000)
    p.add_argument('--mbgan-iterations', type=int, default=500000)
    p.add_argument('--n-synthetic', type=int, default=1000)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--icfm-device', default='cpu')
    p.add_argument('--mbgan-device', choices=('cpu', 'gpu', 'auto'), default='cpu')
    args = p.parse_args()
    if args.n_synthetic < 2:
        p.error('--n-synthetic must be at least two')
    for name in ('icfm_steps', 'mbgan_iterations', 'n_synthetic', 'threads', 'workers'):
        if getattr(args, name) < 1:
            p.error(f'--{name.replace("_", "-")} must be positive')
    # Keep venv interpreter symlinks: resolving them selects the global Python
    # and loses the environment's site-packages.
    args.python_generators = args.python_generators.absolute()
    args.python_mbgan = args.python_mbgan.absolute()
    experiment = Experiment(args)
    stages = ['prepare', *MODELS, 'batts', 'figures', 'report'] if args.stage == 'all' else [args.stage]
    for stage in stages:
        if stage in MODELS:
            experiment.model(stage)
        else:
            getattr(experiment, stage)()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, subprocess.CalledProcessError) as exc:
        print(f'Experiment failed: {exc}', file=sys.stderr)
        sys.exit(1)
