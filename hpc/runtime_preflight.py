#!/usr/bin/env python3
"""Verify the historical CPU environment and record its execution provenance."""
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {
    'tensorflow-cpu': '2.13.1', 'keras': '2.13.1', 'numpy': '1.24.3',
    'scipy': '1.10.1', 'h5py': '3.9.0', 'pandas': '1.5.3',
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    run = Path(os.environ['TWO_SAMPLE_EXECUTION_ROOT'])
    target = Path(os.environ['TWO_SAMPLE_RUNTIME_PROVENANCE'])
    if target.exists():
        raise FileExistsError('Execution requires new runtime evidence: ' + str(target))
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('Historical MBGAN must run with CUDA_VISIBLE_DEVICES empty')
    cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', '0'))
    if cpus < 4:
        raise RuntimeError('Two simultaneous replicas need at least four allocated CPU threads')
    for name, expected in {
        'OMP_NUM_THREADS': '2', 'OPENBLAS_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
        'TF_NUM_INTRAOP_THREADS': '2', 'TF_NUM_INTEROP_THREADS': '1',
        'TF_ENABLE_ONEDNN_OPTS': '1', 'TF_DETERMINISTIC_OPS': '1',
    }.items():
        if os.environ.get(name) != expected:
            raise RuntimeError('Incorrect numerical thread setting: ' + name)
    requirements = ROOT / 'generators/requirements-mbgan.txt'
    required = dict(REQUIRED)
    for line in requirements.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if line.count('==') != 1:
            raise RuntimeError('Expected exact dependency pin: ' + line)
        name, version = line.split('==')
        if name in required and required[name] != version:
            raise RuntimeError('Requirements differ from the historical CPU environment: ' + name)
        required[name] = version
    versions = {name: metadata.version(name) for name in required}
    if versions != required:
        raise RuntimeError('Historical MBGAN dependency mismatch: ' + repr(versions))
    config = json.loads((ROOT / 'generators/training_config.json').read_text())
    cfg = config['models']['mbgan']
    if cfg['device'] != 'cpu' or cfg['threads'] != 2 or cfg['iterations'] != 500000:
        raise RuntimeError('Expected the full 500000-iteration, two-thread CPU recipe')
    staged = run / 'staged_source_manifest.json'
    source_record = None
    if staged.exists():
        manifest = json.loads(staged.read_text())
        for name, expected in manifest.items():
            path = ROOT / name
            if not path.is_file() or sha256(path) != expected:
                raise RuntimeError('Staged source changed before execution: ' + name)
        source_record = dict(files_verified=len(manifest), manifest_sha256=sha256(staged))
    import tensorflow as tf
    if tf.__version__ != REQUIRED['tensorflow-cpu']:
        raise RuntimeError('Loaded TensorFlow differs from the installed pinned distribution')
    if tf.config.list_physical_devices('GPU') or tf.test.is_built_with_cuda():
        raise RuntimeError('A CPU-only TensorFlow build is required')
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.experimental.enable_op_determinism()
    with tf.device('/CPU:0'):
        product = tf.linalg.matmul(tf.ones((4, 4)), tf.ones((4, 4)))
        if 'CPU:0' not in product.device or float(product[0, 0].numpy()) != 4.:
            raise RuntimeError('CPU matrix multiplication preflight failed')
    record = dict(
        status='passed', device='cpu', independent_training_runs=2,
        replica_workers=2, threads_per_replica=2, allocated_cpus=cpus,
        slurm_job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
        python=sys.version, executable=sys.executable, platform=platform.platform(),
        machine=platform.machine(), cpu_affinity=sorted(os.sched_getaffinity(0)),
        lscpu=subprocess.check_output(['lscpu'], text=True),
        tensorflow_version=tf.__version__, tensorflow_build=tf.sysconfig.get_build_info(),
        cpu_checks=dict(matmul_device=product.device, matmul_value=float(product[0, 0].numpy())),
        package_versions=versions, required_package_versions=required,
        mbgan_configuration=cfg,
        training_configuration_sha256=sha256(ROOT / 'generators/training_config.json'),
        requirements_sha256=sha256(requirements),
        pip_freeze=subprocess.check_output([sys.executable, '-m', 'pip', 'freeze', '--all'], text=True),
        staged_sources=source_record,
        environment={name: os.environ.get(name) for name in (
            'CUDA_VISIBLE_DEVICES', 'PYTHONHASHSEED', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
            'MKL_NUM_THREADS', 'TF_NUM_INTRAOP_THREADS', 'TF_NUM_INTEROP_THREADS',
            'TF_ENABLE_ONEDNN_OPTS', 'TF_DETERMINISTIC_OPS', 'LD_LIBRARY_PATH')},
        note='Two independent full CPU training sequences; no pretrained model inputs. '
             'Determinism is established by comparing the complete repeated fits and samples.',
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open('x') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print('Historical CPU preflight passed; runtime evidence: ' + str(target), flush=True)


if __name__ == '__main__':
    main()
