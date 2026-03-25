# TwoSample Public

This repository contains a de-identified public copy of the minimal files needed to reproduce the revision plotting workflow.

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

That script currently downloads and subsets `curatedMetagenomicData`. It is therefore preprocessing context. The real data and synthetic data used for reproducing the plots in the data analysis section is available in ./data/*

