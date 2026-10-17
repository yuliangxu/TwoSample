# Final preprocessing: collection-wide selection, before selecting the four
# approved studies by matched sample IDs and retained-taxon normalization.
# Input is a nonnegative taxa-by-observation relative-abundance matrix on the
# original percentage scale. Select taxa using the full collection, not only
# the four studies and not separately within training/testing. There is no
# zero-count filter. The final 2052 by 123 cohort is fixed by externally supplied provenance;
# scripts/reprocess_microbiome.R reconstructs those sample IDs and the 1641/411
# split into a runtime directory. Never apply a metadata mask to matrix positions without matching IDs.
original_microbiome_selection <- function(relative_abundance) {
  stopifnot(is.matrix(relative_abundance), all(is.finite(relative_abundance)),
            all(relative_abundance >= 0))
  complete <- colSums(relative_abundance) >= 99
  stopifnot(any(complete))
  taxa <- rowMeans(relative_abundance[, complete, drop = FALSE]) >= 0.1
  retained <- relative_abundance[taxa, , drop = FALSE]
  stopifnot(nrow(retained) > 0L)
  selected <- complete & colMeans(retained) >= mean(colMeans(retained))
  list(taxa = taxa, samples = selected, abundance = retained[, selected, drop = FALSE])
}
