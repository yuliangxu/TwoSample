# Reproduce the processed real data

From the repository root, after installing the Bioconductor package
`curatedMetagenomicData`, run:

```sh
Rscript scripts/reprocess_microbiome.R
```

An optional first argument selects the output directory. By default, outputs
go to `output/reprocessed_data/`, leaving the bundled inputs unchanged.
The script retrieves exactly the four dated resource titles in `resources.txt`;
it does not choose the latest resource on each run. Internet access is needed
unless the ExperimentHub resources and metadata are already cached. Validation
used R 4.6.1 and curatedMetagenomicData 3.20.0. Other package versions must pass
the same numerical checks; dated resources and public sample metadata must
remain available upstream.

## What is reproduced

- `samples.csv` fixes training/testing membership and row order. `sample_id`
  is a reconstruction choice from `source_samples`, the complete set of
  numerically matching source candidates found in the audit.
- `taxa.txt` fixes the 123 columns in their published order, using long taxon
  names. Taxa absent from a source resource are filled with zero, matching
  the original `mergeData()` behavior.
- Each sample is divided by its sum over the retained taxa, then rounded to
  IEEE-754 float32. CSV output uses 17 significant digits, preserving those
  values when read as doubles.
- `numeric_checksums.csv` contains MD5 checksums of row-major, little-endian
  float32 values from the published inputs. The script verifies these before
  exporting each split: 1,166 training rows and 292 testing rows, each with
  123 columns. CSV text formatting need not be byte-for-byte identical.

The original `scripts/microbiome_subset.R` filters the full merged collection:
samples first require total abundance at least 99; taxa require mean abundance
at least 0.1 among those samples. Samples are then retained if over 10% of
retained taxa have zero abundance and their retained-taxon mean is at least
the across-sample mean. The reconstruction script uses the audited final
taxon/sample selection, so it need not download unrelated studies or rerun
collection-wide thresholds that could change with future package releases.

All 1,458 published rows were matched numerically. Of these, 1,426 have unique
source matches and 32 have multiple identical source profiles. All candidates
for each ambiguous row belong to the same study. Distinct candidate IDs are
chosen deterministically without reusing a source sample. This reproduces
the numerical matrices, but does not establish which identical-profile ID
was selected historically. The original random split seed was not recovered;
the manifest preserves the published split directly. It must not be treated
as proof of participant-level separation between training and testing.

This workflow reproduces the **real processed compositions**, not the trained
generators, generated samples, or fitted density-ratio summaries. Citations
and source terms are in [CITATIONS.md](../CITATIONS.md) and
[DATA_TERMS.md](../DATA_TERMS.md).
