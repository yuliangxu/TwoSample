# MBGAN trained from random initialization

The released case-study route fits a new MBGAN using all 1,166 rows in
`data/sample_train.csv` and the 123-taxon order in `data/provenance/taxa.txt`.
It never reads the testing CSV, a recovered neural checkpoint, or a prior
synthetic matrix. Sampling requires the explicitly supplied NPZ produced by a
completed training run. The selected checkpoint is the final prescribed
iteration; no validation or test result selects an epoch, seed, or architecture.

## Commands and environment

Use the separate TensorFlow environment in `requirements-mbgan-cuda.txt`
(Python 3.9.25, TensorFlow 2.20.0, Keras 3.10.0, NumPy 2.0.2, h5py 3.14.0).
Apply the [GPU library setup](#gpu-libraries-and-hardware-evidence) below
before running these commands from the repository root:

```sh
.venv-mbgan/bin/python generators/mbgan.py train \
  --train data/sample_train.csv --taxonomy data/provenance/taxa.txt \
  --output-dir output/mbgan --iterations 500000 --batch-size 32 \
  --n-critic 5 --seed 1 --threads 2 --device gpu
.venv-mbgan/bin/python generators/mbgan.py sample \
  --checkpoint output/mbgan/generator.npz \
  --output output/sample_mbgan.csv --n 1000 --seed 256 \
  --batch-size 32 --threads 2 --device gpu
```

Output paths must be new. A short implementation check uses the same training
command with `--iterations 120`; it does not replace the final 500,000-update
fit. Independent full repeats are compared by the enclosing reproduction
workflow. One NVIDIA RTX A5000 and two CPU threads per replica are the reference
execution settings. Numerical
identity is validated within this pinned environment; fixed seeds alone do
not promise identical output across hardware, versions, devices, or thread
settings. oneDNN and deterministic TensorFlow operations are explicitly enabled.

## Architecture and objective

The network follows the recovered MBGAN architecture and taxonomy-aware critic,
adapted from the public [MB-GAN implementation](https://github.com/zhanxw/MB-GAN).

* Generator: Gaussian latent dimension 100; three width-512 Dense blocks,
  batch normalization (momentum 0.8, epsilon 0.001), ReLU, and a
  123-dimensional softmax. The first Dense applies ReLU before its batch
  normalization, preserving the recovered architecture.
* Critic: aggregation to taxonomy prefixes, `log(1+1000*x)/log(1001)`, three
  width-256 Dense layers, LeakyReLU slope 0.2, dropout 0.25, scalar output.
* Critic loss: mean(fake score) minus mean(real score), plus ten times the
  mean squared deviation of interpolated-input gradient norms from one.
  Each observation has one uniform interpolation coefficient. A `1e-12`
  square-root epsilon avoids a zero-norm derivative singularity.
* Generator loss: minus mean(critic(fake)). Generator batch-normalization
  state is frozen during critic updates and updated during generator updates.
  Critic dropout remains active during both updates.
* RMSprop: learning rate `5e-5`, rho `0.9`, epsilon `1e-7`, zero momentum,
  uncentered. Five critic updates precede each generator update, with batches
  of 32 training observations drawn with replacement. The fixed budget is
  500,000 generator updates (2,500,000 critic updates).

The recovered local legacy graph did not connect its gradient-penalty output
correctly. This implementation differentiates the explicit connected WGAN-GP
objective. Consequently this is documented new training, not reconstruction of
the historical weights or a promise of matching historical figures.

## Randomness and compiled execution

`tf.keras.utils.set_random_seed(1)` fixes Dense Glorot initialization and other
initialization randomness. One explicitly seeded TensorFlow Philox generator
supplies every training-batch index, latent Gaussian, interpolation coefficient,
and critic dropout mask. The dropout mask is Bernoulli with keep probability
0.75 and retained activations are divided by 0.75. There are no implicit
unrecorded dropout RNG streams. Sampling uses NumPy `RandomState(256)` Gaussian
latent draws and fixed batches of 32, with batch normalization in inference mode.
No thresholding or post-sampling renormalization is applied.

Training updates run inside `tf.function`; each critic update differentiates
through the input-gradient penalty, and every gradient and loss is checked for
finiteness. The `--eager` option executes the same objective code for a small
implementation comparison. Compilation may change floating-point arithmetic;
bitwise equality between eager and compiled execution is not the reproduction
criterion. The new explicit TensorFlow RNG stream replaces the earlier mixed
NumPy/global-TensorFlow training RNG; it preserves the declared distributions
while defining a new fixed training trajectory. Compiled GPU runs with the
same settings are the reference route.

## GPU libraries and hardware evidence

The A5000 reference uses the CUDA shared-library versions pinned in
`requirements-mbgan-cuda.txt`, including cuBLAS 12.4.5.8 and cuDNN 9.1.0.70.
Install the companion file with `python -m pip install -r
generators/requirements-mbgan-cuda.txt`; it includes the base TensorFlow
requirements and uses the ordinary dependency resolver. Do not also request
TensorFlow's `[and-cuda]` extras, which specify a different CUDA library set.
The tested TensorFlow 2.20.0 binary reports a CUDA 12.5.1 build; the exact pinned
runtime libraries were separately validated by two independent A5000 training
benchmarks. This records an observed compatible stack, without asserting
compatibility of arbitrary other CUDA versions.

For direct Python commands, run the following setup from the repository root
after installing `requirements-mbgan-cuda.txt`. This discovers the library
root in the TensorFlow environment; an existing `TWO_SAMPLE_GPU_LIBRARY_ROOT`
can instead select an explicitly recorded installation.

```sh
export MBGAN_PYTHON="$PWD/.venv-mbgan/bin/python"
if [ -z "${TWO_SAMPLE_GPU_LIBRARY_ROOT:-}" ]; then
  TWO_SAMPLE_GPU_LIBRARY_ROOT="$("$MBGAN_PYTHON" - <<'PYGPUROOT'
import importlib.util
spec = importlib.util.find_spec("nvidia")
if spec is None or not spec.submodule_search_locations:
    raise RuntimeError("Install generators/requirements-mbgan-cuda.txt first")
print(next(iter(spec.submodule_search_locations)))
PYGPUROOT
)"
fi
export TWO_SAMPLE_GPU_LIBRARY_ROOT
TWO_SAMPLE_GPU_LIBRARY_PATH="$("$MBGAN_PYTHON" - <<'PYGPULIBS'
import os
from pathlib import Path
root = Path(os.environ["TWO_SAMPLE_GPU_LIBRARY_ROOT"])
libraries = sorted(str(path) for path in root.glob("*/lib"))
if not libraries:
    raise RuntimeError("No NVIDIA library directories found at " + str(root))
print(":".join(libraries))
PYGPULIBS
)"
export LD_LIBRARY_PATH="$TWO_SAMPLE_GPU_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TF_FORCE_GPU_ALLOW_GROWTH=true TF_DETERMINISTIC_OPS=1
export TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1
```

Preserve `CUDA_VISIBLE_DEVICES` supplied by the scheduler. The full
`generators.reproduce` command runs its two independent replicas concurrently
when two GPUs are available and sequentially when only one is selected. It
defaults to the first two visible GPU identifiers, or GPU `0` if visibility is
unset. An explicit `--gpu-devices` accepts one or two identifiers within the
ambient allocation. The reference Slurm launcher requires exactly two GPUs.

`hpc/from_training.slurm` exposes each installed `nvidia/*/lib` directory in
`LD_LIBRARY_PATH`, sets `CUBLAS_WORKSPACE_CONFIG=:4096:8`, enables deterministic
TensorFlow operations, and checks both allocated GPUs before launching training.
Each replica receives one distinct allocated GPU. The cluster launcher uses
the already installed NVIDIA libraries under
`/hpc/home/yx306/.local/lib/python3.9/site-packages/nvidia`, with
`/opt/apps/rhel9/cuda-12.8/lib64` as a secondary search directory; no package
installation is performed by the job. A portable installation discovers the
NVIDIA package root in the TensorFlow environment or accepts an explicit
`TWO_SAMPLE_GPU_LIBRARY_ROOT`.

The short benchmark used an NVIDIA RTX A5000 (compute capability 8.6) with
driver 610.43.02. The full workflow records its actual driver output, GPU
identities, TensorFlow build, installed NVIDIA package versions, loaded CUDA
library paths, numerical environment, and per-device matrix-multiplication
checks at `TWO_SAMPLE_RUNTIME_PROVENANCE`, defaulting to
`runtime_environment.json` under `TWO_SAMPLE_EXECUTION_ROOT`.

## Provenance and outputs

Training writes its input, taxonomy, and executable-source SHA-256 hashes;
normalized training-tensor hash; initial generator/critic tensor hashes;
all optimizer settings, random-stream description, dependency versions;
and the loss/gradient history at every iteration. Progress includes elapsed
and estimated remaining time. Final records contain generator, critic and
optimizer tensor hashes, finite-gradient confirmation, parameter changes,
and an exact save/reload sampling check. Input/source changes during execution
cause failure rather than a completion record.

`generator.npz` contains the generator weights and immutable training
configuration; `critic.npz` contains the fitted critic. `training_state*`
preserves both networks, optimizer states, and Philox RNG state for auditing;
`rng_state.npy` independently records the final RNG counter. The CLI starts
fresh runs and does not resume intermediate training. `completion.json` is
written only after validation. The sampler rejects historical HDF5 files,
missing/inconsistent completion records, and tensor/hash mismatches.

## Attribution and licenses

The adapted MBGAN module is GPL-3.0-only; the credited Keras-contrib code retains
its MIT notice. See [licenses/MBGAN-NOTICE.md](licenses/MBGAN-NOTICE.md).
These module-specific notices do not relicense unrelated repository files.
