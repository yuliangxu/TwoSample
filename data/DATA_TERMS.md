# Data-release terms review

Reviewed 2026-09-29. This is a provenance/terms assessment, not a new data license
or legal clearance for the bundled data.

## Verified sources

- The package used by `scripts/microbiome_subset.R` is
  [`curatedMetagenomicData`](https://github.com/waldronlab/curatedMetagenomicData),
  whose [DESCRIPTION](https://github.com/waldronlab/curatedMetagenomicData/blob/661de6271741b3dcb436e950ae9b7cb65e918a25/DESCRIPTION)
  declares Artistic-2.0. This declaration does not by itself establish all
  rights in each underlying study's data.
- The upstream [ExperimentHub resource metadata](https://github.com/waldronlab/curatedMetagenomicData/blob/661de6271741b3dcb436e950ae9b7cb65e918a25/inst/extdata/metadata.csv)
  inspected at commit `661de6271741b3dcb436e950ae9b7cb65e918a25` contains 732
  resource records and no explicit License column. It supplies source/provider
  information, but is not proof of a blanket data redistribution license.
- [NCBI's molecular data policy](https://www.ncbi.nlm.nih.gov/home/about/policies/#data)
  places no NCBI restrictions on use or distribution and says NCBI does not accept
  submitter restrictions on reuse/redistribution. It also explains that third-party
  rights may exist and NCBI cannot grant unrestricted permission on their behalf.
  This policy is relevant to NCBI-sourced data, not proof that every retained
  sample came from such a source.

## Assessment for this subset

Public redistribution of a processed subset can be permissible when the source
terms permit redistribution and adaptation and their conditions are satisfied.
Subsampling, normalization, and omission of identifiers do not independently
remove source conditions. An article's open-access license must not be assumed
to cover separately deposited data.

The current script queries all relative-abundance resources rather than a fixed
study list. It filters samples/taxa but does not preserve the historical package
version, resource dates/IDs, final sample mapping, normalization/export steps,
or train/test split. Metadata is subset by position, which should be checked
against abundance sample IDs before relying on it as provenance.
The anonymous numerical CSVs cannot establish the exact contributing studies.
Therefore this review does not certify the existing subset for public release;
it also found no evidence that the subset is prohibited from redistribution.

## HMP-specific follow-up

The data owner recalls that the subset is HMP data; this remains unverified
against the exported matrices. The upstream metadata includes
`2021-03-31.HMP_2012.relative_abundance`, sourced from NCBI SRA, but this is a
candidate resource rather than a confirmed historical download.

The [HMP FAQ](https://www.hmpdacc.org/overview/faq/) says permission is not
required for scientific use and requests citation of the consortium papers.
[NIH HMP release guidelines](https://commonfund.nih.gov/hmp/hmp-data-release-and-resource-sharing-guidelines-human-microbiome-project-data-production)
endorse unrestricted sharing of metagenomic data while distinguishing
potentially identifying data that belong in controlled-access dbGaP.
The [HMP data model](https://hmpdacc.org/hmp/overview/data-model.php)
likewise distinguishes public microbial sequence data from controlled human
sequences and clinical metadata.

Taken together with NCBI's molecular-data policy, these sources support the
assessment that redistributing a processed subset of public HMP microbial
abundances is generally permissible, with source attribution and documentation
of modifications. This assessment does not extend to controlled-access data.
The unresolved issue here is confirming that the matrices contain only those
public HMP resources, since the saved preprocessing script is not HMP-specific.

If HMP1 provenance is confirmed, cite the applicable source study plus:

- Human Microbiome Project Consortium. *A framework for human microbiome
  research.* Nature 486, 215–221 (2012).
- Human Microbiome Project Consortium. *Structure, function and diversity of
  the healthy human microbiome.* Nature 486, 207–214 (2012).

Retain the curatedMetagenomicData citation as well. If the data instead derive
from iHMP, use the relevant iHMP study and consortium attribution.

## Information needed to finish the review

Recover the original package/Bioconductor versions, ExperimentHub IDs and resource
dates, and retained sample-to-study mapping. For each contributing study, record
its accession, publication, applicable data terms, and any required attribution.
Document filtering, transformations, random seeds, and final split/export steps.
Keep this provenance alongside the data, and retain applicable source notices
without claiming to relicense third-party data under a code license.

If applicable terms remain unclear, obtain clarification from the data provider
or package maintainers before treating the subset as cleared. A pinned download
and preprocessing workflow is an alternative to redistributing uncertain source
matrices once the missing reconstruction steps have been recovered.
