# Case-study data

All composition CSVs have no header and use the preserved order of 123 taxa.
The original full pooled real split contains 1,166 training observations and
292 test observations. These two real matrices remain unchanged.

This release includes all six matrices and preprocessing provenance. Two
independent full fitting sequences passed exact reproduction checks in job
58514810, including both 500,000-iteration fits of the selected historical
local MBGAN implementation. The released synthetic matrices are the outputs
of those completed fits.

The completed workflow produced four 1,000-row synthetic matrices using only
the training matrix. Figures display them in the order **D, DT, ICFM, MBGAN**.
Its [generation_manifest.json](generation_manifest.json) binds inputs, settings,
source, and outputs; [evidence/reproduction.json](evidence/reproduction.json)
records exact comparisons between the two complete fitting sequences. The
[MBGAN validation](evidence/mbgan_independent_validation.json) additionally
compares the full recorded trajectories, optimizer arrays, and explicit random
states. These data supplied the [final report](../results/final/case_study_report.pdf)
and [figures](../results/final/figures/).

Run the [generator reproduction workflow](../generators/README.md) to create
`output/generated/data/`. That directory contains all six matrices,
`generation_manifest.json`, and `evidence/` recording two complete independent
training runs, each using CPU ICFM and the historical local MBGAN variant
with TensorFlow CPU 2.13.1. Exact agreement is required between replicas in
the recorded environment; equality across hardware is not assumed.
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
