# Source data terms

Reviewed 2026-09-29 for the public microbial-abundance data used in this
repository. The numerical provenance audit identifies four source cohorts;
this supersedes the earlier unresolved HMP-only hypothesis. Full study and
package references are in [CITATIONS.md](CITATIONS.md), and reconstruction
instructions are in [provenance/README.md](provenance/README.md).

| Source | Assessment and supporting policy |
| --- | --- |
| HMP_2019_ibdmdb | Public microbial data are available for unrestricted use, distinct from controlled-access human data. Historical publication moratoria have expired. See the [HMPDACC resource paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC7778886/) and [iHMP policy](https://hmpdacc.org/ihmp/overview/datapolicy.php). |
| HMP_2019_t2d | The [study's availability statements](https://www.nature.com/articles/s41586-019-1236-x) explicitly identify CC0 for available data and describe public HMP2 data as unrestricted. This assessment excludes separate controlled-access exome data. |
| HallAB_2017 | The [paper](https://link.springer.com/article/10.1186/s13073-017-0490-5) deposits cohort sequences in SRA PRJNA385949 and applies CC0 to data made available with the article unless otherwise stated. Together with NCBI policy, this supports redistribution of derived abundances; the article notice is not an independently verified blanket CC0 license for every separately hosted resource. |
| HanniganGD_2017 | The [paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC6247079/) deposits sequences in public SRA PRJNA389927. No redistribution prohibition was found. A separate dataset license is less explicit than for T2D; the article's CC-BY license and the authors' software license should not alone be treated as licenses for separately deposited data. |

[NCBI's molecular-data policy](https://www.ncbi.nlm.nih.gov/home/about/policies/)
places no NCBI restrictions on use or distribution and states that NCBI does
not accept submissions conditioned on reuse or redistribution restrictions.
It also notes that third-party rights may exist and that NCBI cannot grant
permission on their behalf.

The reviewed sources support redistribution of these derived public microbial
abundances; no reviewed policy requires their removal. This is a source-based
assessment, not a new license or a separate permission grant from the data
owners. Subsampling and preprocessing do not erase source restrictions.
The curatedMetagenomicData software license and this repository's code license
do not automatically license the underlying third-party data.

Retain the source citations and applicable notices when redistributing the
matrices, identify the preprocessing modifications, and do not describe the
mixture as IBDMDB-only or assign it a blanket software license. This assessment
does not cover controlled-access human sequences or additional clinical data.
The reprocessing workflow also allows users to retrieve public inputs directly
from curatedMetagenomicData and regenerate the processed real matrices.
