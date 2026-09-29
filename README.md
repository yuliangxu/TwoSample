# TwoSample case study

Reproducible code for the microbiome case study in *Two-sample Comparison
through Additive Tree Models for Density Ratios*. Script names and PDF names
follow the paper: Figures 6, 7, and S8–S12.

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

See [data/README.md](data/README.md) for matrix conventions and
[data/DATA_TERMS.md](data/DATA_TERMS.md) for the initial release-terms review.
The real and generated sample matrices and existing fitted density ratios
are unchanged. `scripts/microbiome_subset.R` is upstream filtering context;
it does not reconstruct the final sample selection, train/test split, or
synthetic generation. The plotting workflow is reproducible from the bundled
inputs, not a complete regeneration of the original generative-model fits.

The sample-matching audit identifies four source studies in the real CSVs:
HMP_2019_ibdmdb (931), HMP_2019_t2d (277), HallAB_2017 (209), and
HanniganGD_2017 (41). These counts supersede the initial terms note's unresolved
HMP-only hypothesis; no blanket dataset license is asserted here.

`plot_revision1.py` and `utils/` are retained as the legacy Python workflow.

## Citation

Pasolli E, et al. *Accessible, curated metagenomic data through ExperimentHub.*
Nature Methods. 2017;14:1023–1024. <https://doi.org/10.1038/nmeth.4468>.
Also cite the original contributing studies.
