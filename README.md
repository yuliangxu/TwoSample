# TwoSample case study

Reproducible code for the microbiome case study in *Two-sample Comparison
through Additive Tree Models for Density Ratios*. Script names and PDF names
follow the paper: Figures 6, 7, and S8–S12.

For the additional full-study `HMP_2019_ibdmdb` experiment, see the
[HPC workflow](hpc/README.md): study-only preprocessing, an 80/20 split,
fresh training of all four generators, and seven figures with current BATTS.

The [generative-model release](generators/README.md) adds fitting and sampling
code for Dirichlet, stabilized Dirichlet tree, ICFM, and MBGAN, together with
[selected regenerated samples](data/regenerated_20261003/) and a
[validation report](results/generators_20261003/REPORT.md). Parametric fits are
reproducible from the public training data. Neural checkpoint replay and new
neural training recipes are documented separately because historical neural
training provenance is incomplete. The figure commands below continue to use
the original data and matching BATTS results.

The [updated BATTS rerun report](results/batts_909ea357/REPORT.md) contains
new results, comparisons with the paper, and all seven regenerated PDFs for
commit `909ea357`. The commands below distinguish redrawing the original
results from performing the new refit.

## Reproduce all seven figures

Install the plotting dependencies in R:

```r
install.packages(c("ggplot2", "vegan", "patchwork", "scales"))
```

Then run from the repository root:

```sh
Rscript scripts/recreate_all_case_study_figures.R
```

All PDFs, plotting tables, cached coordinates, and `sessionInfo.txt` are written
to `output/case_study/` (ignored by Git). The scripts also work when invoked by
absolute path from another directory. Plotting uses the bundled compositions
and posterior summaries; no microbiome download or model refitting is needed.

| Paper figure | Individual command | PDF in `output/case_study/` |
| --- | --- | --- |
| 6: test PCoA comparison | `Rscript scripts/recreate_figure_6.R` | `figure6_ggplot2.pdf` |
| 7: test density-ratio results | `Rscript scripts/recreate_figure_7.R` | `figure7_ggplot2.pdf` |
| S8: training PCoA comparison | `Rscript scripts/recreate_figure_S8.R` | `figureS8_ggplot2.pdf` |
| S9: training density-ratio results | `Rscript scripts/recreate_figure_S9.R` | `figureS9_ggplot2.pdf` |
| S10: training mean/quantile grids | `Rscript scripts/recreate_figure_S10.R` | `figureS10_ggplot2.pdf` |
| S11: test mean/quantile grids | `Rscript scripts/recreate_figure_S11.R` | `figureS11_ggplot2.pdf` |
| S12: null-experiment intervals | `Rscript scripts/recreate_figure_S12.R` | `figureS12_ggplot2.pdf` |

The grouped commands `recreate_figures_6_7.R` and
`recreate_figures_S10_S11.R` are also available in `scripts/`.
Shared implementations live in `scripts/lib/` so individual entry points use
identical analysis and styling.

## Refit with the updated BATTS package

The complete rerun uses BATTS commit
[`909ea357bdb3f97018613552ef13430d1bb60c6e`](https://github.com/nawaya040/BATTS/tree/909ea357bdb3f97018613552ef13430d1bb60c6e).
Install that exact commit and the plotting dependencies with a working R/C++
build toolchain:

```r
install.packages(c("remotes", "ggplot2", "vegan", "patchwork", "scales"))
remotes::install_github("nawaya040/BATTS@909ea357bdb3f97018613552ef13430d1bb60c6e", upgrade = "never")
```

```sh
Rscript scripts/reproduce_case_study_batts.R 2
```

The optional integer controls independent comparison workers on macOS/Linux;
Windows runs sequentially. This command refits density-ratio comparisons for
all four generators against each real split, reruns the held-out null experiment, exports comparison
tables, and renders Figures 6, 7, and S8–S12. Outputs are under
`output/batts_909ea357/`; the bundled paper results are preserved.

Each of the eight comparisons uses seed 1, 200 Bayesian trees, 2,000 burn-in
iterations, 1,000 retained draws, thinning 1, and fixed lambda 5. Both boosting
estimators use five-fold CV over up to 1,000 trees, depth 4, learning rate 0.01,
and 32 bins. The comparisons retain the original `margin_scale = 0.1`;
all 123 columns vary in each combined real/generated dataset. The null fit
retains its separate seed 2026, 500 burn-in and 500 retained iterations, and
`margin_scale = -1` for compositions in [0,1].

Train and test comparisons are **separately fitted two-sample analyses**, as
in the original case-study workflow. The null experiment alone fits on a
random split of the training data and evaluates on held-out test data.
Generated compositions are fixed inputs; BATTS does not train their generators.

The refit checks the installed commit and input checksums. Completed individual
fits are cached with their inputs/settings so interrupted runs can resume.
Posterior draws, fitted objects, CV summaries, interval classifications,
comparisons with the paper, and session information are retained. New package
fixes and floating-point differences can change estimates; numerical equality
with the historical paper results is not assumed.

Individual stages are also available:

```sh
Rscript scripts/refit_case_study.R 2
Rscript scripts/summarize_case_study_refit.R
```

The summary command requires the null output from the complete rerun. To plot
alternative fits, set `TWO_SAMPLE_RATIO_ROOT`, `TWO_SAMPLE_FIGURE_DIR`, and
`TWO_SAMPLE_NULL_INTERVALS` before invoking the existing figure scripts.

## Figure conventions

Figures 6, 7, S8 and S9 use Bray–Curtis PCoA with an additive correction, as in
the revised ggplot2 figures. These coordinates are cached per comparison and
recomputed if either input CSV or the algorithm identifier changes. A first
run can take several minutes.

Figures S10/S11 reproduce the older manuscript grids using classical
Bray–Curtis PCoA **without** the additive correction, matching the original
Python eigendecomposition. They distinguish generated samples (crosses) from
real samples (dots), and show separate posterior-mean, lower-2.5%, and
upper-97.5% rows. The color scale uses Matplotlib-style symmetric-log
normalization and the seismic palette, with limits [-30, 30] for training
and [-12, 12] for testing. PCoA eigenvector signs are anchored to match the manuscript orientation;
these axis reflections preserve the ordination distances.

In Figures 7/S9, the 20 displayed rows are selected by the across-method
average absolute posterior mean. Matching row numbers across generators are
display indices, not matched biological subjects. The interval endpoints are
pointwise posterior quantiles. Plot-data CSVs are exported alongside PDFs.

## Refit the Figure S12 null experiment

The exact paper interval table is bundled in
`data/figure_S12/test_pointwise_credible_intervals.csv`. Its 292 intervals all
cover zero. The default S12 command redraws that table, and the annotation is
computed from the intervals rather than hard-coded.

To rerun the model, install BATTS and a suitable C++ toolchain. The original
run used BATTS 0.0.0.9000 from commit
`fccebbf3909dbcdcc5cb37d7a05ca06d92f69694`:

```r
install.packages("remotes")
remotes::install_github("MaStatLab/BATTS@fccebbf3909dbcdcc5cb37d7a05ca06d92f69694")
```

That historical source uses `bits/stdc++.h`, which is unavailable with the
standard macOS compiler. The original macOS run replaced those includes with
equivalent standard C++ headers; model logic was unchanged. A compatible GCC
build environment can build the historical source directly.

```sh
Rscript scripts/run_figure_S12_null_experiment.R
Rscript scripts/recreate_figure_S12.R output/figure_S12_refit/test_pointwise_credible_intervals.csv
```

The fit uses seed 2026, a balanced random split of the 1,166 training samples,
200 trees, 500 burn-in iterations, 500 retained iterations, thinning 1,
fixed lambda 5, and `margin_scale = -1` for compositions already in [0,1].
It evaluates the log density ratio on the 292 held-out observations. Refitted
summaries, posterior draws/model, and session details go to
`output/figure_S12_refit/`; the bundled paper table is never overwritten.
Different compiler/package environments can affect MCMC output, so use the
bundled summary for the published numerical result.

## Inputs and provenance

The real training and testing compositions can be reproduced from the
`curatedMetagenomicData` R package using our preprocessing script and the
bundled sample/taxon manifests:

```r
install.packages("BiocManager")
BiocManager::install("curatedMetagenomicData")
```

```sh
Rscript scripts/reprocess_microbiome.R
```

The command downloads four dated resources, restores the published sample and
taxon ordering, normalizes over the retained taxa, converts to float32, and
checks the numerical contents against the published matrices. It writes
`sample_train.csv` and `sample_test.csv` to `output/reprocessed_data/`.
Validation used curatedMetagenomicData 3.20.0. See
[data/provenance/README.md](data/provenance/README.md) for resource versions,
selection details, and duplicate-profile ambiguity.

The original `scripts/microbiome_subset.R` records the upstream filtering;
`scripts/reprocess_microbiome.R` completes reconstruction of the final real
matrices using the recovered selection. For synthetic-sample regeneration and
generator fitting, use the separate [generative-model workflow](generators/README.md).

See [data/README.md](data/README.md) for matrix conventions,
[data/CITATIONS.md](data/CITATIONS.md) for the package and all four contributing
studies, and [data/DATA_TERMS.md](data/DATA_TERMS.md) for source terms.
`plot_revision1.py` and `utils/` are retained as the legacy Python workflow.

## Citation

Please cite **curatedMetagenomicData and all four source studies** when using
the real microbiome data. Full references and DOI links are in
[data/CITATIONS.md](data/CITATIONS.md). The source mixture comprises IBDMDB
(931 rows), T2D (277), Hall (209), and Hannigan (41), rather than IBDMDB alone.
