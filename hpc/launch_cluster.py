#!/usr/bin/env python3
"""Stage immutable sources and submit two independent CPU training runs."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--stage-only', action='store_true',
                        help='Write the immutable run and launch command without submitting')
    args = parser.parse_args()
    run = args.run_root.resolve()
    if run.exists():
        parser.error('Run directory must be new')
    if run == ROOT or ROOT in run.parents:
        parser.error('The immutable run directory must be outside the source checkout')
    if any(p.suffix in {'.pt', '.h5', '.npz'} for p in (ROOT / 'generators').rglob('*')):
        raise RuntimeError('Remove fitted model inputs before staging the code-only workflow')
    run.mkdir(parents=True)
    repo = run / 'repository'
    shutil.copytree(ROOT, repo, ignore=shutil.ignore_patterns('.git', 'output', '__pycache__', '*.pyc'))
    logs = run / 'logs'
    logs.mkdir()
    manifest = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(repo.rglob('*')) if p.is_file()}
    (run / 'staged_source_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    exports = {
        'TWO_SAMPLE_SOURCE_ROOT': str(repo),
        'TWO_SAMPLE_EXECUTION_ROOT': str(run),
        'TWO_SAMPLE_RUNTIME_PROVENANCE': str(run / 'runtime_environment.json'),
        'TWO_SAMPLE_RUN_ROOT': str(run / 'analysis'),
        'TWO_SAMPLE_GENERATION_ROOT': str(run / 'generators'),
        'GENERATOR_PYTHON': '/usr/bin/python3',
        'MBGAN_PYTHON': '/cwork/yx306/TwoSample/ibdmdb_20261003_f713917/historical_environment/venv/bin/python',
        'CASE_STUDY_RSCRIPT': '/opt/apps/rhel9/R-4.6.0/bin/Rscript',
        'REPORT_PYTHON': '/cwork/yx306/TwoSample/pooled_updated_generators_20261004/report_environment/bin/python',
        'R_LIBS_USER': '/cwork/yx306/TwoSample-final-case-study-20261013/r_lib:/cwork/yx306/TwoSample/ibdmdb_20261003_f713917/r_lib',
        'CUDA_VISIBLE_DEVICES': '',
        'TF_ENABLE_ONEDNN_OPTS': '1',
        'TF_DETERMINISTIC_OPS': '1',
    }
    # Preserve the validated installed Python interpreter selected by the caller.
    import sys
    exports['GENERATOR_PYTHON'] = sys.executable
    script = '\n'.join(['#!/usr/bin/env bash', 'set -euo pipefail'] +
                       ['export ' + key + '=' + shlex.quote(value) for key, value in exports.items()] +
                       ['bash hpc/from_training.slurm']) + '\n'
    job = run / 'run.sh'
    job.write_text(script)
    command = ['sbatch', '--parsable', '--job-name=twosample-historical-mbgan', '--account=mastatlab',
               '--partition=common', '--nodes=1', '--ntasks=1', '--cpus-per-task=8',
               '--mem=32G', '--time=1-00:00:00', '--chdir=' + str(repo),
               '--output=' + str(logs / 'slurm-%j.log'), str(job)]
    receipt = {'status': 'staged', 'job_id': None, 'run_root': str(run), 'source_files': len(manifest),
               'pretrained_model_inputs': [], 'independent_training_runs': 2,
               'replica_workers': 2, 'device': 'cpu', 'threads_per_replica': 2,
               'runtime_evidence': 'runtime_environment.json',
               'command': command, 'environment': exports}
    (run / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    if not args.stage_only:
        result = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=True)
        receipt.update(status='submitted', job_id=result.stdout.strip())
    (run / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
