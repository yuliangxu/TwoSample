# Case-study input data

All CSVs are numeric and have no header. Rows retain the ordering used for fitting.

| File | Rows | Columns | Meaning |
| --- | ---: | ---: | --- |
| `sample_train.csv` | 1166 | 123 | Preprocessed real training compositions |
| `sample_test.csv` | 292 | 123 | Preprocessed real test compositions |
| `sample_d.csv` | 1000 | 123 | Dirichlet generated samples |
| `sample_dt.csv` | 1000 | 123 | Dirichlet-tree generated samples |
| `sample_icfm.csv` | 1000 | 123 | ICFM generated samples |
| `sample_mbgan.csv` | 1000 | 123 | MB-GAN generated samples |

These top-level matrices are the original paper inputs. The selected updated
synthetic data are versioned in [`regenerated_20261003/`](regenerated_20261003/),
with checksums and [fitting/sampling commands](../generators/README.md).
The posterior summaries below correspond to the original matrices; updated
samples require new downstream fits.

For each generator key (`d`, `dt`, `icfm`, `mbgan`),
`revision/train/log_w_<key>.csv` and `revision/test/log_w_<key>.csv`
contain posterior summaries. Real observations precede that generator's 1000
observations, giving 2166 training rows or 1292 test rows. The plotting scripts
use columns 3, 4, and 10 (R indexing) for the posterior mean log density ratio
and lower/upper 95% pointwise posterior interval endpoints.

The real compositions originate from `curatedMetagenomicData`; the other
sample matrices and fitted summaries were produced for this analysis.
Run `Rscript scripts/reprocess_microbiome.R` from the repository root to
reproduce the real training/testing matrices from the package's dated
resources. Outputs and numerical verification are described in
[provenance/README.md](provenance/README.md).

Please cite the package and all four contributing studies listed in
[CITATIONS.md](CITATIONS.md). See [DATA_TERMS.md](DATA_TERMS.md) for the
study-specific release-terms assessment. The numerical matrices have no
identifiers; their recovered sample and taxon ordering is provided separately
in `provenance/`.

## Figure S12 null experiment

`figure_S12/test_pointwise_credible_intervals.csv` contains the original
292-row null-experiment posterior summary. See [figure_S12/README.md](figure_S12/README.md)
for provenance and the plotting/refitting commands.
