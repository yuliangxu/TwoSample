# Final microbiome case study

BATTS 0.3.0, source commit `147e511d211919afefb8decac6333e66d24919db`.

Original pooled preprocessing and split: 1,166 training observations, 292 test observations, 123 taxa. Each generator contributes 1,000 compositions. Model order: ICFM, DT, D, MBGAN.

All four generators are fitted from the original 1,166 training rows using frozen recipes. ICFM and MBGAN start from seeded random initialization and use their final planned update; no test rows enter generator training or selection. Generation provenance and reproduction evidence must pass validation before completion.

All eight train/test comparisons are independently fitted. Test density-ratio analyses use the real test observations themselves; they are not predictions from a density-ratio estimator fitted on the training comparison.

The previous protocol is explicit: 200 trees, 2,000 burn-in and 1,000 retained iterations, thinning 1, fixed lambda 5, seed 1. Boosting uses five-fold CV, up to 1,000 trees, depth 4, learning rate 0.01, 32 bins, and subsample_fraction = 1. See settings.json for all arguments.

## Real-test descriptive summaries

| Generator | Intervals excluding zero | Mean absolute posterior-mean log ratio |
| --- | ---: | ---: |
| ICFM | 291/292 | 1.699123 |
| DT | 174/292 | 0.604964 |
| D | 210/292 | 1.234929 |
| MBGAN | 292/292 | 5.676278 |

These pointwise posterior intervals are descriptive. Inclusion of zero does not establish distributional equality; one null experiment does not establish a repeated-sampling coverage guarantee.

Null intervals containing zero: 292/292. Boosting fits at the 1,000-tree cap: 8/16.

Preprocessing is collection-wide. The original split is by observation and does not establish participant-level separation. Figures 6, 7, S8 and S9 use corrected Bray-Curtis PCoA; S10/S11 retain classical uncorrected PCoA. Posterior intervals are pointwise.

All eight comparison tables, 24 estimator fingerprints, the null diagnostic, and seven PDF signatures passed validation. analysis_validation.json records the checks.
