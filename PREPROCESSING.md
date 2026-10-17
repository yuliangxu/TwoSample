# Corrected preprocessing protocol

The case study uses 2,052 observations and 123 taxa, split into 1,641 training
and 411 test observations. Apply these operations in order, on the original
**percentage scale**: `99` means 99%, and `0.1` means 0.1%.

1. Start with all 93 dated source resources and 22,584 metadata-supported
   observations. Restrict taxa to the frozen phylogenetic-tree species universe
   and fill taxa absent from an individual resource with zero.
2. Mark an observation complete when its total abundance is at least `99`.
   There are 22,001 complete observations.
3. Retain taxa with mean abundance at least `0.1` across complete observations.
   This selects 123 taxa. Let `B` be these taxa over **all original observations**.
4. Compute `m = colMeans(B)` and the cutoff `mean(m)` over all 22,584 observations,
   including those that failed completeness. Retain observations satisfying
   both completeness and `m >= mean(m)`. The recorded cutoff is
   `0.717983668249196` percentage units; 16,824 observations pass.
5. Select all eligible observations from `HMP_2019_ibdmdb`, `HMP_2019_t2d`,
   `HallAB_2017`, and `HanniganGD_2017`, matching by `(study, sample_id)`.
   Their counts are 1,491, 277, 209, and 75. Do not apply a metadata mask
   positionally to abundance columns. **There is no zero-count or sparsity filter.**
6. Preserve the frozen observation and taxon ordering. Normalize each
   observation over the 123 taxa in float64, then round to float32. Apply
   `train_test_split(test_size=0.2, random_state=1)` and preserve the returned
   order within each bank. Write 17 significant digits to retain float32 values.

The selection arithmetic is implemented in
[microbiome_subset.R](scripts/microbiome_subset.R). The full reconstruction
recomputes the thresholds from source matrices and checks the selected keys;
it does not merely reuse an old sample-selection mask.

## External prerequisites

Data and tree assets are intentionally external to this code-only repository.
For exact reproduction, provide the following original provenance inputs in
one directory. These files contain data or identifiers and must not be committed.

| File | Required contents |
| --- | --- |
| `collection_resources.csv` | One row per dated source: `study`, `title`, `path`, `sha256`; 93 resources. |
| `collection_metadata_ids.csv` | The original 22,584 observations as `study`, `sample_id`, in the recorded collection order. |
| `collection_species.txt` | One species label per line in the frozen full-tree universe. |
| `taxa.txt` | The 123 retained labels in their original model-input order. |
| `cohort_order.csv` | The corrected 2,052 observations with `cohort_row`, `sample_id`, `source_studies`, and `historical_filtered_column`, in the historical abundance-column order. |

Also provide participant metadata with `study_name`, `sample_id`, and
`subject_id`, and the exact original tree assets `childNode.csv`, `treeTaxa.csv`,
and `outTaxa.csv`. Tree tables use the original row-name column; the taxa tables
have labels in their second column, and `childNode.csv` has the internal node
label followed by two child labels. The scripts validate a rooted binary tree
with 123 tips, 122 internal nodes, and taxon order matching the real matrices.

The abundance sources are dated relative-abundance resources from
curatedMetagenomicData/ExperimentHub. The four selected study resources are
`2021-10-14.HallAB_2017.relative_abundance`,
`2021-03-31.HanniganGD_2017.relative_abundance`,
`2021-10-14.HMP_2019_ibdmdb.relative_abundance`, and
`2021-03-31.HMP_2019_t2d.relative_abundance`. The full selection requires the
other collection resources as well; using just these four studies changes the
threshold calculation. Source paths may be relocated with `--cache-root`,
but every file must match its recorded SHA-256.

Obtaining the current public abundance resources alone does not recover the
frozen species universe, exact tree topology, or historical observation order.
These original external inputs are required to reproduce the same seeded
split and generator fits. The code fails when they are missing or inconsistent;
it does not substitute an arbitrary tree or ordering.

## Reconstruct from the full collection

Install [requirements-preprocessing.txt](requirements-preprocessing.txt), then
run from the repository root. All destinations below must be new directories:

```sh
Rscript scripts/prepare_corrected_cohort.R \
  --source-provenance /absolute/source-provenance \
  --output /absolute/new-preparation

python3 -B scripts/finalize_corrected_cohort.py /absolute/new-preparation \
  --output /absolute/prepared-data \
  --tree-dir /absolute/original-tree \
  --participant-metadata /absolute/participant-metadata.csv

python3 -B scripts/verify_corrected_cohort.py --data-dir /absolute/prepared-data
```

If the source-cache directory has moved, add `--cache-root /absolute/cache` to
the preparation command. An optional `--legacy-reference /absolute/raw_ra.csv`
on finalization performs an additional value comparison against the original
normalized taxa-by-observation table. That optional check is separate from the
mandatory source reconstruction and keyed selection.

The runtime `prepared-data` directory contains `sample_train.csv`,
`sample_test.csv`, `provenance/`, and `tree/`. The cohort manifest binds the
matrices, identities, preprocessing evidence, and tree assets by SHA-256.
Pass this directory to the training workflow in [README.md](README.md).

The observation split is not participant-disjoint. In the original corrected
cohort, 138 participants occur in both banks, and 392 of the 411 test
observations belong to a participant represented in training. Participant
overlap is recalculated from the external metadata during finalization.
