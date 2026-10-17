#!/usr/bin/env Rscript
# Full cache-based preparation requires only base R and sha256sum.
# Install this optional stack only for replaying dated resources from ExperimentHub.
args <- commandArgs(TRUE)
if (length(args) == 1L && args[[1]] %in% c('--help', '-h')) {
  cat('Usage: Rscript scripts/install_preprocessing.R [--package-source /external/curatedMetagenomicData_3.20.0.tar.gz]\n')
  quit(status = 0)
}
if (length(args) && (length(args) != 2L || args[[1]] != '--package-source'))
  stop('Unknown arguments; use --help.')
options(repos = c(CRAN = 'https://cloud.r-project.org'), timeout = 1200)
if (!requireNamespace('BiocManager', quietly = TRUE)) install.packages('BiocManager')
# The Bioconductor release compatible with the user's R is selected by BiocManager.
# If it supplies another data-package version, use an externally obtained source
# archive of the validated version in a compatible R/Bioconductor environment.
BiocManager::install(c('curatedMetagenomicData', 'ExperimentHub'), ask = FALSE, update = FALSE)
if (length(args)) {
  source <- normalizePath(args[[2]], mustWork = TRUE)
  install.packages(source, repos = NULL, type = 'source')
}
actual <- as.character(utils::packageVersion('curatedMetagenomicData'))
if (actual != '3.20.0') stop('Validated package is curatedMetagenomicData 3.20.0; installed ', actual,
  '. Use --package-source with its external source archive and compatible dependencies, or replay pinned raw caches.')
message('Installed curatedMetagenomicData 3.20.0; all reconstructed values must still pass numerical checksums.')
