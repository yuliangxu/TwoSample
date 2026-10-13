# Historical local MBGAN trained from random initialization

This route restores the local MBGAN implementation used for the earlier
IBDMDB-only case study. It fits all 1,166 pooled training rows with 123 taxa,
starting from random initialization, without reading the testing CSV, a
historical checkpoint, or a prior synthetic matrix. The final prescribed
500,000th iteration supplies the sampling weights. This historical local
variant is the **selected final MBGAN implementation**. Two independent
120-iteration implementation checks passed, including exact parity with the
original training loop's parameters and optimizer states.
**Full 500,000-iteration reproduction and updated analysis are pending** in
job 58514810. This release contains code and real inputs; the preceding
implementation's results are preserved at
[commit b009075](https://github.com/yuliangxu/TwoSample/tree/b00907514c3bbb7ab17c4eeacc5091ab4f441d8a).

## Which earlier report used which MBGAN?

| Earlier report | Real training / test rows | Taxa | MBGAN source |
| --- | ---: | ---: | --- |
| `ibdmdb_case_study_report.pdf` | 945 / 236 | 77 | Fresh 500,000-iteration fit of the historical local Keras 2.13.1 implementation |
| `updated_generators_case_study_report.pdf` | 1,166 / 292 | 123 | Replay of an older 500,000-iteration generator checkpoint; no new neural training |

The verified IBDMDB training source was
`ibdmdb_20261003_f713917/historical_code/source/model.py`, with SHA-256
`9efc755e7fdcc18310c5fe4405319134f5d6e25ea50afb0a7b776b6971968e18`.
The byte-identical source is retained as
[`vendor/historical_mbgan_model.py`](vendor/historical_mbgan_model.py). Its runner
executed the historical model and notebook callback from a new initialization. That callback called `predict(seed=None)`, which invoked
`np.random.seed(None)` every 10,000 iterations. Thus its initial seed did not
make the whole trajectory reproducible. The restored callback uses an isolated
seeded stream without changing the training RNG.

The pooled report's checkpoint was
`curated_comp_marginal_500000_generator.h5`, with SHA-256
`457c13dbba3afefd1c0a7979eb6854118d25806143ac5306bd1543740eaf840f`.
Its original training membership and complete trajectory are unknown. Neither
that checkpoint nor the IBDMDB fitted weights are inputs to this workflow.

Returning to the historical implementation after viewing earlier results is a
retrospective method choice. The new run retains the fixed seed and final
iteration without a new test-based seed or checkpoint search. Favorable results
on the 77-taxon IBDMDB data do not establish stability or guarantee favorable
results on the pooled 123-taxon data.

## Commands and environment

Use the separate environment pinned in `requirements-mbgan.txt`: Python 3.9.25,
TensorFlow CPU 2.13.1, Keras 2.13.1, NumPy 1.24.3, and h5py 3.9.0. Training uses
two intra-operation numerical threads and one inter-operation thread, enables
oneDNN and deterministic TensorFlow operations, and disables GPU visibility.
Run from the repository root:

```sh
.venv-mbgan/bin/python -m generators.mbgan train \
  --train data/sample_train.csv --taxonomy data/provenance/taxa.txt \
  --output-dir output/mbgan --iterations 500000 --batch-size 32 \
  --n-critic 5 --seed 1 --threads 2 --device cpu
.venv-mbgan/bin/python -m generators.mbgan sample \
  --checkpoint output/mbgan/generator.npz \
  --output output/sample_mbgan.csv --n 1000 --seed 256 \
  --batch-size 32 --threads 2 --device cpu
```

Output paths must be new. A short implementation check may reduce the iteration
count, but it cannot replace the final 500,000-iteration fit. The enclosing
`generators.reproduce` workflow performs two full independent CPU fits,
concurrently by default or sequentially with `--replica-workers 1`. Exact
agreement is required within the recorded environment; fixed seeds do not
promise identical results across library versions, processors, or thread counts.

## Architecture and effective objective

The restored model preserves the historical local source's construction order
(critic before generator), compiled `train_on_batch` graphs, trainable-layer
switches, built-in dropout, batch normalization, and RMSprop optimizers.

- Generator: latent dimension 100; three width-512 Dense blocks with batch
  normalization (momentum 0.8, epsilon 0.001) and ReLU; 123-dimensional softmax.
  The first Dense also applies ReLU before batch normalization.
- Critic: taxonomy-prefix aggregation, `log(1+1000*x)/log(1001)`, three width-256
  Dense layers with LeakyReLU slope 0.2 and dropout 0.25, then a scalar output.
- Effective critic objective: mean(fake score) minus mean(real score).
  The historical source constructs a gradient-penalty branch but discards it;
  the compiled critic has only real and fake outputs and no penalty losses.
  There is **no effective gradient penalty and no weight clipping**.
- Generator objective: minus mean(critic(fake)). Historical Keras layer modes
  and state updates are retained through the original compiled graphs.
- RMSprop: learning rate `5e-5`, rho `0.9`, epsilon `1e-7`, zero momentum,
  uncentered. Five critic updates precede each generator update, using batches
  of 32 observations sampled with replacement. The budget is 500,000 generator
  updates and 2,500,000 critic updates.

This is a **historical local MBGAN variant**, not a faithful implementation of
the published WGAN-GP objective. The preceding TensorFlow 2.20 implementation
connected the gradient penalty and used different training random streams;
restoring this variant changes that procedure. Removing entropy reseeding also
means the old IBDMDB trajectory is not recovered byte for byte.

## Randomness and validation

The training seed fixes Python, NumPy, and TensorFlow initialization. Training
batch indices and latent draws follow the historical NumPy sequence; built-in
Keras dropout remains in the compiled graphs. At each 10,000-iteration diagnostic interval, sampling uses a private NumPy
`RandomState((training_seed + iteration) % 2**32)`. It preserves the Python and
NumPy training states and does not reseed either stream from system entropy.
Final sampling uses NumPy `RandomState(256)` Gaussian latent vectors in batches
of 32, with the generator in inference mode. No thresholding or post-sampling
renormalization is applied.

The full reproduction gate compares two fresh fits, including every recorded
iteration's losses, generator and critic arrays, optimizer arrays, explicitly
saved Python and NumPy random states, and the final sample CSV. It also checks
finite training losses and final arrays, changed parameters, training/source
hashes, complete update counts, and exact save/reload sampling. Timings are
excluded from equality. The records document TensorFlow graph seeds and update
counts; they do not claim to preserve its internal random-operation counters
or support mid-training resume. Finite losses and parameters are not a claim
that every gradient was separately inspected.

When both complete fits pass, these checks establish reproducibility in the
recorded environment. The passed short checks alone do not validate the full
500,000-iteration trajectory. The checks do
not establish stability across seeds, convergence of the adversarial objective,
or agreement between generated and real distributions. BATTS and descriptive
figures must be interpreted from the completed new run itself.

## Provenance and outputs

Training records bind the training data, taxonomy, source, normalized input,
initial and final model arrays, optimizers, dependencies, thread settings,
randomness policy, per-iteration losses, and parameter changes. A completion
record requires successful training and save/reload validation.

`generator.npz` stores sampling weights with the immutable training
configuration; `critic.npz` stores critic weights and `optimizer.npz` stores
optimizer arrays. `training_history.jsonl` records all five critic-loss vectors
and the generator loss per iteration; `callback_history.jsonl` records diagnostic
sample hashes and sparsity. `training_state.*` preserves model and optimizer
state. `rng_state.json` records Python/NumPy states and TensorFlow graph seeds
and completed update counts; `rng_state.npy` separately records NumPy MT19937
keys. These records support the independent-run comparison. The CLI starts
fresh fits; sampling requires the NPZ and matching completion evidence from a
completed fit. Historical HDF5 checkpoints cannot substitute for a training run.

See [the generator workflow](README.md) for the full independent reproduction
gate and [the verification records](../reproducibility/README.md) for completion
status. [License and attribution notices](licenses/MBGAN-NOTICE.md) credit the
public MB-GAN implementation and its Keras-contrib sources.
