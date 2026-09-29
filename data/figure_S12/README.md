# Figure S12 reference posterior summary

`test_pointwise_credible_intervals.csv` is the original null-experiment result
used for Figure S12: 292 held-out observations and 500 posterior draws per
observation. Rows correspond to `../sample_test.csv` before ranking for display.

Columns record the test row, random test-fold label, posterior mean log density
ratio, 2.5% and 97.5% posterior quantiles, coverage of zero, and interval class.
All 292 original intervals contain zero. This table was verified against the
original saved posterior draws. It is a derived model summary, not a new
source cohort or raw clinical-data release.

Use `scripts/recreate_figure_S12.R` to redraw it, or
`scripts/run_figure_S12_null_experiment.R` to refit from the real sample CSVs.
The latter writes to `output/figure_S12_refit/`. See the root README for the
BATTS source commit, fitting settings, and platform notes.
