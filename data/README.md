# Case-study data

All composition CSVs have no header and use the preserved order of 123 taxa.
The original full pooled real split contains 1,166 training observations and
292 test observations. These two real matrices remain unchanged.

The four validated 1,000-row synthetic matrices are in report order
**D, DT, ICFM, MBGAN** and were generated using only the training matrix.
Two complete independent fitting sequences produced identical parameters and
synthetic CSVs. [generation_manifest.json](generation_manifest.json) binds the
inputs, settings, source and outputs; [reproduction evidence](evidence/reproduction.json)
records the exact comparisons.

Run the [generator reproduction workflow](../generators/README.md) to create
`output/generated/data/`. That directory contains all six matrices,
`generation_manifest.json`, and `evidence/` recording two complete independent
training runs, each using CPU ICFM and GPU MBGAN. Exact agreement is required
between replicas in the recorded environment; CPU/GPU equality is not assumed.
Pass that exact directory as `TWO_SAMPLE_DATA_DIR` when running
the analysis. No historical neural checkpoint is required. Each neural model
uses its final prescribed training update, without test-based selection.

[Preprocessing provenance](provenance/README.md) fixes the dated source
resources, final sample selection, row order, taxa, and float32 normalization.
Run `Rscript scripts/reprocess_microbiome.R` to reconstruct and numerically
verify the real matrices. Cite the package and four source studies in
[CITATIONS.md](CITATIONS.md), retaining [DATA_TERMS.md](DATA_TERMS.md).
Preprocessing is collection-wide, and the preserved observation split does not
establish participant-level separation.

The analysis writes estimator summaries to
`TWO_SAMPLE_RUN_ROOT/revision/{train,test}/`. Real observations precede
generated observations. Columns are gradient boosting, Hellinger boosting,
Bayesian posterior mean, then posterior quantiles at 0.025, 0.05, 0.1, 0.5,
0.9, 0.95, and 0.975, all on the `log(p_real/q_generated)` scale. Each
train/test-versus-generator comparison has its own fitted density-ratio model.
