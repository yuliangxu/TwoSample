# Selected regenerated synthetic data

These four **1,000 × 123** matrices are the selected generator outputs from
the October 2026 reproducibility audit. They use the same column order as
[`../provenance/taxa.txt`](../provenance/taxa.txt). CSVs have no headers.

| File | Construction | Sampling seed |
| --- | --- | ---: |
| `sample_d.csv` | Recovered Dirichlet Newton MLE fitted to the 1,166 public training rows | 1 |
| `sample_dt.csv` | Stabilized Dirichlet tree: positive log-parameter Beta MLE at all 122 nodes | 1 |
| `sample_icfm.csv` | Recovered original MLP checkpoint; restored torchdyn Dopri5, CPU, float32, tolerances `1e-4` | 2 |
| `sample_mbgan.csv` | Recovered MBGAN generator checkpoint; TensorFlow/Keras replay | 256 |

The Dirichlet and MBGAN outputs reproduce the published samples numerically
within the documented precision. The stabilized DT and selected ICFM outputs
are updated samples and differ from the paper's matrices. See
[`manifest.json`](manifest.json) for hashes and numerical differences, and
[`../../generators/README.md`](../../generators/README.md) for fitting and
generation commands.

Real observations remain in `../sample_train.csv` (1,166 rows) and
`../sample_test.csv` (292 rows). Parametric fitting and the new neural training
commands use training observations only. The historical neural checkpoints'
original fitting membership remains unverified.

The original `../sample_*.csv` matrices and their BATTS posterior summaries are
preserved for paper reproduction. **Those posterior summaries do not belong
to this updated dataset.** No BATTS fits or paper figures using these updated
samples are included in this release. Use a new analysis output directory when
fitting downstream models to this dataset.
