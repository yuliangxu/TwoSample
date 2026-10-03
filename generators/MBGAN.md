# MBGAN: saved-checkpoint replay and new training

`mbgan.py` provides two different operations:

* `sample` replays the recovered 500,000-iteration MBGAN generator. Seed 256 and 1,000 samples match the published synthetic matrix to tight floating-point tolerance.
* `train` fits a new MBGAN to the supplied training CSV. This is an explicit, portable training recipe; it does not claim to recreate the historical checkpoint's training run or establish which observations were used in that run.

The default training input is `data/sample_train.csv` (1,166 rows, 123 taxa). Its column order is given by `data/provenance/taxa.txt`. Neither training nor checkpoint selection reads `data/sample_test.csv`.

## Environment and commands

The MBGAN package uses TensorFlow. The validated environment is Python 3.9, TensorFlow 2.20.0, Keras 3.10.0, NumPy 2.0.2, and h5py 3.14.0. It can be installed in a separate environment with:

```sh
python -m pip install -r generators/requirements-mbgan.txt
```

Run from the repository root. Output files/directories must be new; existing files and bundled checkpoints are preserved.

```sh
# Replay the historical checkpoint, without training.
python generators/mbgan.py sample --output output/mbgan_replayed.csv --n 1000 --seed 256

# Exercise actual training, finite gradients, parameter updates, save/reload,
# deterministic sampling, and historical-checkpoint numerical agreement.
python generators/mbgan.py smoke-test --output-dir output/mbgan_smoke

# Full new training using the original iteration budget; this can be expensive.
python generators/mbgan.py train --output-dir output/mbgan_new_train --iterations 500000 --seed 1

# Sample the newly fitted generator.
python generators/mbgan.py sample --checkpoint output/mbgan_new_train/generator.npz --output output/mbgan_new_samples.csv
```

CPU with two threads is the default. `--device auto` enables TensorFlow's available accelerator selection; `--device gpu` requires a visible GPU and fails if none is available. `--threads` controls CPU parallelism. Training records these settings and enables deterministic TensorFlow operations. A fixed seed does not promise bitwise equality across TensorFlow versions, hardware, or devices.

## Checkpoint replay

`checkpoints/mbgan_generator.h5` has SHA-256
`457c13dbba3afefd1c0a7979eb6854118d25806143ac5306bd1543740eaf840f`.
Its saved metadata records Keras 2.13.1. The legacy Functional wrapper uses a graph-node index that Keras 3 cannot deserialize. The loader reads its saved Sequential layer configuration directly and assigns all 20 HDF5 weight arrays, verifying each exactly. The generator's Dense activations, batch-normalization parameters, and softmax are retained.

Sampling uses the original NumPy `RandomState` normal draws with latent dimension 100, seed 256, and Keras prediction batches of 32. No thresholding, smoothing, renormalization, or taxon reordering is applied to generated output. The prior validation found maximum absolute difference `1.7881393432617188e-7` and RMSE `3.987488790791777e-9` against `data/sample_mbgan.csv`; `allclose(rtol=1e-5, atol=1e-8)` passes. The result is not bitwise identical to the historical array.

## New-training recipe and changes

The network and training structure are adapted from the recovered MBGAN implementation and the public [zhanxw/MB-GAN source](https://github.com/zhanxw/MB-GAN/blob/master/model.py), with settings from its [training example](https://github.com/zhanxw/MB-GAN/blob/master/mbgan_train_demo.py) and the recovered research notebook:

* Generator: latent dimension 100; three width-512 Dense blocks with batch normalization (momentum 0.8, epsilon 0.001), ReLU, and final 123-dimensional softmax. The first Dense already applies ReLU before its batch normalization, as in the recovered source and checkpoint.
* Critic: taxonomy-prefix aggregation, `log(1 + 1000*x)/log(1001)`, three width-256 Dense layers, LeakyReLU slope 0.2, dropout 0.25, and scalar output.
* Losses: critic mean(fake) minus mean(real), plus 10 times the gradient penalty; generator minus mean(critic(fake)). Interpolations have one uniform mixing coefficient per observation.
* Optimization: RMSprop learning rate `5e-5`, rho 0.9, epsilon `1e-7`; five critic steps followed by one generator step; batches of 32 sampled with replacement. Default budget is 500,000 iterations.

The recovered modernized local source had a gradient-penalty layer whose output was not connected to the critic training graph, along with inconsistent loss-weight settings. This portable implementation explicitly includes the WGAN-GP penalty in the differentiated critic loss, following upstream's documented weight 10. It adds `1e-12` under the gradient-norm square root to avoid a zero-norm derivative singularity. These are documented corrections, not a claim about the objective used to produce the historical weights.

Generator batch-normalization state is frozen during critic updates and updated during generator updates. Critic dropout remains active during both updates, as in the original training graph. Training records input hashes, taxonomy, RNG seed, settings, dependencies, loss history, finite-gradient checks, and parameter changes. It saves a portable generator NPZ, critic weights, taxonomy matrix, and TensorFlow optimizer/model state. The CLI starts new training; it does not implement training resumption.

The short smoke test establishes that the training path executes and updates both models. It is not evidence that a newly trained model has converged or matches the paper's generative distribution.

## Attribution and licenses

The adapted MBGAN module is distributed under GPL-3.0; the credited Keras-contrib code retains its MIT notice. See [licenses/MBGAN-NOTICE.md](licenses/MBGAN-NOTICE.md). These module-specific notices do not relicense unrelated repository files.
