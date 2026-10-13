# Final case-study data

All six CSVs have no header and contain compositions in the preserved 123-taxon order.
The original pooled real split is 1,166 training and 292 test observations.
The four selected synthetic matrices each have 1,000 rows, in report order
ICFM, DT, D, MBGAN. These are the inputs used in results/final/.

ICFM uses the fixed restored torchdyn sampler; DT is the stabilized fit.
See [generator reproduction](../generators/README.md) for seeds, checkpoints,
reproduction commands and the historical neural-training provenance limits.
The generator integrity manifest pins all six matrix hashes.

[Preprocessing provenance](provenance/README.md) fixes the dated source resources,
final sample selection, row order, taxa and float32 normalization. Run
`Rscript scripts/reprocess_microbiome.R` to reconstruct and numerically verify
the real matrices. Cite the package and four source studies in
[CITATIONS.md](CITATIONS.md), retaining [DATA_TERMS.md](DATA_TERMS.md).

Final estimator summaries are in `results/final/revision/{train,test}/`.
Real observations precede generated observations. Columns are gradient boosting,
Hellinger boosting, Bayesian posterior mean, then posterior quantiles at
0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975, all on the log(p_real/q_generated) scale.
