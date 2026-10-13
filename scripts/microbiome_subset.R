# Original collection-wide selection, before final retained-taxon normalization.
# Input is a nonnegative taxa-by-observation relative-abundance matrix on the
# original percentage scale. The final case study uses the audited sample and
# taxon manifests; scripts/reprocess_microbiome.R reproduces those exact rows.
original_microbiome_selection <- function(relative_abundance) {
  stopifnot(is.matrix(relative_abundance), all(is.finite(relative_abundance)),
            all(relative_abundance >= 0))
  complete <- colSums(relative_abundance) >= 99
  stopifnot(any(complete))
  taxa <- rowMeans(relative_abundance[, complete, drop = FALSE]) >= 0.1
  retained <- relative_abundance[taxa, , drop = FALSE]
  stopifnot(nrow(retained) > 0L)
  selected <- complete & colSums(retained == 0) > nrow(retained) * 0.1 &
              colMeans(retained) >= mean(colMeans(retained))
  list(taxa = taxa, samples = selected, abundance = retained[, selected, drop = FALSE])
}
