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

For each generator key (`d`, `dt`, `icfm`, `mbgan`),
`revision/train/log_w_<key>.csv` and `revision/test/log_w_<key>.csv`
contain posterior summaries. Real observations precede that generator's 1000
observations, giving 2166 training rows or 1292 test rows. The plotting scripts
use columns 3, 4, and 10 (R indexing) for the posterior mean log density ratio
and lower/upper 95% pointwise posterior interval endpoints.

The real compositions originate from `curatedMetagenomicData`; the other
sample matrices and fitted summaries were produced for this analysis.
The CSVs do not contain study names, sample IDs, or taxon names. Absence of
identifiers alone does not establish redistribution permission.
See [DATA_TERMS.md](DATA_TERMS.md) for the outstanding provenance questions.
