#!/usr/bin/env python3
"""Stage immutable sources and submit two independent A5000 training runs."""
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
    args = parser.parse_args()
    run = args.run_root.resolve()
    if run.exists():
        parser.error('Run directory must be new')
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
        'TWO_SAMPLE_EXECUTION_ROOT': str(run),
        'TWO_SAMPLE_RUNTIME_PROVENANCE': str(run / 'runtime_environment.json'),
        'TWO_SAMPLE_RUN_ROOT': str(run / 'analysis'),
        'TWO_SAMPLE_GENERATION_ROOT': str(run / 'generators'),
        'GENERATOR_PYTHON': '/usr/bin/python3',
        'MBGAN_PYTHON': '/hpc/home/yx306/venvs/rdr-tf/bin/python',
        'CASE_STUDY_RSCRIPT': '/opt/apps/rhel9/R-4.6.0/bin/Rscript',
        'REPORT_PYTHON': '/cwork/yx306/TwoSample/pooled_updated_generators_20261004/report_environment/bin/python',
        'R_LIBS_USER': '/cwork/yx306/TwoSample-final-case-study-20261013/r_lib:/cwork/yx306/TwoSample/ibdmdb_20261003_f713917/r_lib',
        'TWO_SAMPLE_GPU_LIBRARY_ROOT': '/hpc/home/yx306/.local/lib/python3.9/site-packages/nvidia',
        'TWO_SAMPLE_CUDA_FALLBACK': '/opt/apps/rhel9/cuda-12.8/lib64',
        'CUBLAS_WORKSPACE_CONFIG': ':4096:8',
        'TF_FORCE_GPU_ALLOW_GROWTH': 'true',
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
    command = ['sbatch', '--parsable', '--job-name=twosample-from-training', '--account=mastatlab',
               '--partition=mastatlab', '--nodes=1', '--ntasks=1', '--cpus-per-task=8',
               '--gres=gpu:a5000:2', '--mem=32G', '--time=08:00:00',
               '--output=' + str(logs / 'slurm-%j.log'), str(job)]
    result = subprocess.run(command, cwd=repo, text=True, capture_output=True, check=True)
    receipt = {'job_id': result.stdout.strip(), 'run_root': str(run), 'source_files': len(manifest),
               'pretrained_model_inputs': [], 'independent_training_runs': 2,
               'gpu_runtime_evidence': 'runtime_environment.json',
               'command': command, 'environment': exports}
    (run / 'submission.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
