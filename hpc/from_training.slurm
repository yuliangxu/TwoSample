#!/usr/bin/env bash
#SBATCH --job-name=twosample-training
#SBATCH --account=mastatlab
#SBATCH --partition=mastatlab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a5000:2
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=slurm-%x-%j.log
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
export PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TWO_SAMPLE_RUN_ROOT="${TWO_SAMPLE_RUN_ROOT:-$PWD/output/from_training_analysis}"
export TWO_SAMPLE_GENERATION_ROOT="${TWO_SAMPLE_GENERATION_ROOT:-$PWD/output/from_training_generators}"
export TWO_SAMPLE_EXECUTION_ROOT="${TWO_SAMPLE_EXECUTION_ROOT:-$PWD/output/from_training_execution}"
export TWO_SAMPLE_RUNTIME_PROVENANCE="${TWO_SAMPLE_RUNTIME_PROVENANCE:-$TWO_SAMPLE_EXECUTION_ROOT/runtime_environment.json}"
export MBGAN_PYTHON="${MBGAN_PYTHON:-$PWD/.venv-mbgan/bin/python}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TF_FORCE_GPU_ALLOW_GROWTH=true TF_DETERMINISTIC_OPS=1
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
export MPLCONFIGDIR="$TWO_SAMPLE_EXECUTION_ROOT/matplotlib"
# The cluster launcher supplies the exact existing CUDA library root validated
# in the A5000 benchmark. A portable install can discover it in the TF venv.
if [[ -z "${TWO_SAMPLE_GPU_LIBRARY_ROOT:-}" ]]; then
  TWO_SAMPLE_GPU_LIBRARY_ROOT="$($MBGAN_PYTHON - <<'PY'
import importlib.util
spec = importlib.util.find_spec('nvidia')
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError('Install generators/requirements-mbgan-cuda.txt or set TWO_SAMPLE_GPU_LIBRARY_ROOT')
print(next(iter(spec.submodule_search_locations)))
PY
)"
  export TWO_SAMPLE_GPU_LIBRARY_ROOT
fi
TWO_SAMPLE_GPU_LIBRARY_PATH="$($MBGAN_PYTHON - <<'PY'
import os
from pathlib import Path
root = Path(os.environ['TWO_SAMPLE_GPU_LIBRARY_ROOT'])
libraries = sorted(str(p) for p in root.glob('*/lib'))
if not libraries:
    raise RuntimeError('No CUDA shared-library directories found at ' + str(root))
fallback = os.environ.get('TWO_SAMPLE_CUDA_FALLBACK', '')
if fallback:
    if not Path(fallback).is_dir():
        raise RuntimeError('CUDA fallback directory is missing: ' + fallback)
    libraries.append(fallback)
print(':'.join(libraries))
PY
)"
export LD_LIBRARY_PATH="$TWO_SAMPLE_GPU_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
"$MBGAN_PYTHON" - <<'PY'
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import tensorflow as tf

root = Path(os.environ['TWO_SAMPLE_EXECUTION_ROOT'])
root.mkdir(parents=True, exist_ok=True)
target = Path(os.environ['TWO_SAMPLE_RUNTIME_PROVENANCE'])
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists():
    raise FileExistsError('GPU execution requires a new evidence output: ' + str(target))
visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
if len(visible) != 2 or len(set(visible)) != 2 or not all(visible):
    raise RuntimeError('Allocate exactly two GPUs; CUDA_VISIBLE_DEVICES=' + repr(visible))
if int(os.environ.get('SLURM_CPUS_PER_TASK', '0')) < 4:
    raise RuntimeError('At least four allocated CPU threads are required for two replicas')
devices = tf.config.list_physical_devices('GPU')
if len(devices) != 2:
    raise RuntimeError('TensorFlow must see the two allocated GPUs: ' + repr(devices))
tf.keras.utils.set_random_seed(1)
tf.config.experimental.enable_op_determinism()
checks = []
for index, device in enumerate(devices):
    tf.config.experimental.set_memory_growth(device, True)
for index, device in enumerate(devices):
    with tf.device('/GPU:' + str(index)):
        product = tf.linalg.matmul(tf.ones((4, 4)), tf.ones((4, 4)))
        if 'GPU:' + str(index) not in product.device or float(product[0, 0].numpy()) != 4.:
            raise RuntimeError('GPU matrix multiplication preflight failed')
        checks.append(dict(device=device.name, details=tf.config.experimental.get_device_details(device),
                           matmul_device=product.device, matmul_value=float(product[0, 0].numpy())))
site = Path(os.environ['TWO_SAMPLE_GPU_LIBRARY_ROOT']).parent
versions = {d.metadata['Name']: d.version for d in metadata.distributions(path=[str(site)])
            if d.metadata['Name'].lower().startswith('nvidia-')}
required = {}
for line in Path('generators/requirements-mbgan-cuda.txt').read_text().splitlines():
    if line.startswith('nvidia-'):
        name, version = line.split('==')
        required[name] = version
        if versions.get(name) != version:
            raise RuntimeError('GPU dependency mismatch: ' + name + ': ' + repr(versions.get(name)))
loaded = sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
                 if '/' in line and any(name in line for name in
                    ('libcuda', 'libcublas', 'libcudnn', 'libcurand', 'libcusolver', 'libcufft', 'libcusparse'))})
record = dict(status='passed', cuda_visible_devices=visible,
              slurm_job_id=os.environ.get('SLURM_JOB_ID'), node=os.environ.get('SLURMD_NODENAME'),
              allocated_cpus=int(os.environ['SLURM_CPUS_PER_TASK']), gpu_checks=checks,
              tensorflow_version=tf.__version__, tensorflow_build=tf.sysconfig.get_build_info(),
              nvidia_package_versions=versions, required_nvidia_package_versions=required,
              loaded_cuda_libraries=loaded, ld_library_path=os.environ['LD_LIBRARY_PATH'],
              cublas_workspace_config=os.environ['CUBLAS_WORKSPACE_CONFIG'],
              tf_deterministic_ops=os.environ['TF_DETERMINISTIC_OPS'],
              nvidia_smi=subprocess.check_output(['nvidia-smi'], text=True),
              note='Each independent replica receives one allocated GPU; no pretrained model inputs')
with target.open('x') as stream:
    json.dump(record, stream, indent=2)
    stream.write('\n')
print('Two-GPU preflight passed; runtime evidence: ' + str(target), flush=True)
PY
"${GENERATOR_PYTHON:-python3}" -m generators.reproduce \
  --tensorflow-python "$MBGAN_PYTHON" \
  --output "$TWO_SAMPLE_GENERATION_ROOT"
export TWO_SAMPLE_DATA_DIR="$TWO_SAMPLE_GENERATION_ROOT/data"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"${CASE_STUDY_RSCRIPT:-Rscript}" scripts/reproduce_case_study_batts.R "${SLURM_CPUS_PER_TASK:-1}"
"${REPORT_PYTHON:-$PWD/.venv-report/bin/python}" scripts/build_report.py "$TWO_SAMPLE_RUN_ROOT"
