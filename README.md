# TwoSample case study

Reproducible plotting code for the microbiome generative-model analysis in
*Two-sample Comparison through Additive Tree Models for Density Ratios*.
The current workflow uses ggplot2 and the round-2 figure layouts.

## Reproduce the figures

Install R and the plotting dependencies once:

```r
install.packages(c("ggplot2", "vegan", "patchwork", "scales"))
```

From the repository root, run these commands in order:

```sh
Rscript scripts/recreate_figures_6_7.R
Rscript scripts/recreate_figures_13_14.R
```

Validated with ggplot2 4.0.3, vegan 2.7.5, patchwork 1.3.2, and scales 1.4.0.

The scripts resolve inputs relative to their own location, so they also work
when invoked by absolute path from another directory. No model fitting,
Python installation, or new microbiome download is required.

The first script computes Bray–Curtis PCoA with an additive correction and
writes `output/figures_6_7/`:

- `figure6_ggplot2.pdf`: test versus generated samples in PCoA space (Figure 6).
- `figure7_ggplot2.pdf`: test-sample posterior means, interval classifications,
  and intervals for 20 selected generated-sample rows (Figure 7).
- `figureS8_ggplot2.pdf`: training versus generated samples in PCoA space (Figure S8).
- `figureS9_ggplot2.pdf`: training-sample density-ratio results (Figure S9).
- `pcoa_coordinates.rds`, `figure7_plot_data.csv`, and `figureS9_plot_data.csv`.

The second script uses those coordinates and writes `output/figures_13_14/`:

- `figure13_ggplot2.pdf` and `figure14_ggplot2.pdf`: train/test grids of posterior
  means and 2.5%/97.5% quantiles, with symmetric pseudo-log color limits of
  [-30, 30] and [-12, 12], respectively.
- Corresponding `figure13_plot_data.csv` and `figure14_plot_data.csv`.

PCoA is recomputed on each first-script run to avoid stale coordinates. This can
take several minutes. Generated outputs are ignored by Git.
In Figures 7 and S9, the 20 rows are selected by the across-method average absolute
posterior mean. Matching row numbers across generators are display indices,
not matched biological subjects.

## Inputs and provenance

See [data/README.md](data/README.md) for the input layout and
[data/DATA_TERMS.md](data/DATA_TERMS.md) for the data-release review.
The bundled numerical inputs are unchanged from the preceding workflow.
The scripts plot existing posterior summaries; they do not retrain the models.

`scripts/microbiome_subset.R` records upstream filtering context. It downloads
from `curatedMetagenomicData`, but does not record the historical resource
versions, export the final matrices, or reproduce the train/test split and
synthetic generation. It is not an end-to-end reconstruction of the bundled CSVs.

`plot_revision1.py` and `utils/` are retained as the legacy Python workflow;
the R commands above are the current case-study entry points.

## Citation

Pasolli E, et al. *Accessible, curated metagenomic data through ExperimentHub.*
Nature Methods. 2017;14:1023–1024. <https://doi.org/10.1038/nmeth.4468>.
Original contributing studies should also be cited once the historical sample
provenance is recovered.
