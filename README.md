# TwoSample Public

This repository contains a de-identified public copy of the minimal files needed to reproduce the revision plotting workflow.

This is the reproducible workflow for the microbiome generative-model analysis in the paper *Two-sample Comparison through Additive Tree Models for Density Ratios* (arXiv:2508.03059v3, March 11, 2026).

## Reproducing `plot_revision1.py`

The main plotting script for the revision figures is `plot_revision1.py`.

Required code:

- `plot_revision1.py`
- `utils/DRE_baloss.py`
- `utils/microbiome_help.py`
- `utils/plot_ci_helpers.py`

Required data files:

- `data/sample_train.csv`
- `data/sample_test.csv`
- `data/sample_d.csv`
- `data/sample_dt.csv`
- `data/sample_icfm.csv`
- `data/sample_mbgan.csv`
- `data/revision/train/log_w_d.csv`
- `data/revision/train/log_w_dt.csv`
- `data/revision/train/log_w_icfm.csv`
- `data/revision/train/log_w_mbgan.csv`
- `data/revision/test/log_w_d.csv`
- `data/revision/test/log_w_dt.csv`
- `data/revision/test/log_w_icfm.csv`
- `data/revision/test/log_w_mbgan.csv`


## Preprocessing

The upstream preprocessing entry point is `scripts/microbiome_subset.R`.

That script currently downloads and subsets `curatedMetagenomicData`. It is therefore preprocessing context. The real and synthetic datasets used to reproduce the plotting workflow are included under `data/`.

## Data Source

The microbiome preprocessing step uses the Bioconductor package `curatedMetagenomicData`.

Recommended citation:

Pasolli E, Schiffer L, Manghi P, Renson A, Obenchain V, Truong DT, Beghini F, Malik F, Ramos M, Dowd JB, Huttenhower C, Morgan M, Segata N, Waldron L. *Accessible, curated metagenomic data through ExperimentHub.* Nature Methods. 2017;14(11):1023-1024. doi:10.1038/nmeth.4468.
