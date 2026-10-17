#!/usr/bin/env Rscript
options(repos = c(CRAN = 'https://cloud.r-project.org'), timeout = 1200)
args <- commandArgs(TRUE)
if (length(args) && !identical(args, '--check')) stop('Usage: Rscript scripts/install_dependencies.R [--check]')
check_only <- identical(args, '--check')
# Direct analysis/plotting dependencies used by the completed case study.
# Install in a dedicated R library; transitive/system dependencies must also
# remain compatible. The validated R version was 4.6.0.
versions <- c(ggplot2 = '4.0.3', vegan = '2.7-5', patchwork = '1.3.2',
              scales = '1.4.0', jsonlite = '2.0.0', digest = '0.6.39')
installed_version <- function(name) {
  tryCatch(as.character(utils::packageDescription(name)$Version),
           error = function(e) NA_character_)
}
if (!check_only && !requireNamespace('remotes', quietly = TRUE)) install.packages('remotes')
for (name in names(versions)) {
  if (!identical(installed_version(name), unname(versions[[name]]))) {
    if (check_only) stop('Install ', name, ' ', versions[[name]], ' before reproducing the analysis.')
    remotes::install_version(name, version = versions[[name]], upgrade = 'never')
  }
}
commit <- '147e511d211919afefb8decac6333e66d24919db'
batts_commit <- function() {
  value <- tryCatch(utils::packageDescription('BATTS')$RemoteSha,
                    error = function(e) NULL)
  stamp <- system.file('source_commit.txt', package = 'BATTS')
  if (is.null(value) && nzchar(stamp)) value <- readLines(stamp, warn = FALSE)
  value
}
installed_commit <- batts_commit()
if (!identical(installed_version('BATTS'), '0.3.0') || !identical(installed_commit, commit)) {
  if (check_only) stop('Install BATTS 0.3.0 at ', commit)
  remotes::install_github(paste0('nawaya040/BATTS@',commit), upgrade='never')
}
stopifnot(packageVersion('BATTS') == '0.3.0', identical(batts_commit(), commit))
for (name in names(versions)) stopifnot(identical(installed_version(name), unname(versions[[name]])))
message('Pinned BATTS and direct analysis/plotting dependencies verified.')
