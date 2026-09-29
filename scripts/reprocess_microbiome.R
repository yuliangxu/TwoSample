#!/usr/bin/env Rscript
# Reconstruct the real train/test matrices from curatedMetagenomicData.
# See data/provenance/README.md for the recovered selection and precision.
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- gsub("~+~", " ", sub("^--file=", "", script_arg), fixed = TRUE)
root <- dirname(dirname(normalizePath(script_file)))
args <- commandArgs(TRUE)
out <- if (length(args)) args[[1]] else file.path(root, "output", "reprocessed_data")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
out <- normalizePath(out)
if (out == normalizePath(file.path(root, "data")))
  stop("Choose an output directory other than the bundled data directory.")
suppressPackageStartupMessages(library(curatedMetagenomicData))
if (as.character(packageVersion("curatedMetagenomicData")) != "3.20.0")
  warning("Validated with curatedMetagenomicData 3.20.0; numeric checksums will detect changed inputs.")
options(timeout = max(1200, getOption("timeout")))
provenance <- file.path(root, "data", "provenance")
taxa <- readLines(file.path(provenance, "taxa.txt"))
manifest <- read.csv(file.path(provenance, "samples.csv"), stringsAsFactors = FALSE)
resources <- readLines(file.path(provenance, "resources.txt"))
expected <- read.csv(file.path(provenance, "numeric_checksums.csv"))
stopifnot(length(taxa) == 123L, !anyDuplicated(taxa),
          nrow(manifest) == 1458L, !anyDuplicated(manifest$sample_id),
          all(manifest$split %in% c("train", "test")))
x <- matrix(NA_real_, nrow(manifest), length(taxa))
for (resource in resources) {
  message("Loading ", resource)
  pattern <- paste0("^", gsub(".", "\\.", resource, fixed = TRUE), "$")
  loaded <- curatedMetagenomicData(pattern, dryrun = FALSE, counts = FALSE,
                                  rownames = "long")
  stopifnot(identical(names(loaded), resource))
  se <- loaded[[1]]
  study <- unique(as.character(SummarizedExperiment::colData(se)$study_name))
  stopifnot(length(study) == 1L)
  i <- which(manifest$source_studies == study)
  j <- match(manifest$sample_id[i], colnames(se))
  if (anyNA(j)) stop("Missing manifest samples in ", resource)
  abundance <- as.matrix(SummarizedExperiment::assay(se))
  # mergeData() in the original filtering script fills absent taxa with zero.
  retained <- matrix(0, length(taxa), length(j))
  k <- match(taxa, rownames(abundance))
  retained[!is.na(k), ] <- abundance[k[!is.na(k)], j, drop = FALSE]
  x[i, ] <- t(retained)
}
stopifnot(all(is.finite(x)), all(x >= 0), all(rowSums(x) > 0))
x <- x / rowSums(x)
# Match the historical export's IEEE-754 single-precision values.
x[] <- readBin(writeBin(as.double(x), raw(), size = 4, endian = "little"),
               "double", n = length(x), size = 4, endian = "little")
for (split in c("train", "test")) {
  i <- which(manifest$split == split)
  stopifnot(identical(manifest$csv_row[i], seq_along(i)))
  values <- x[i, , drop = FALSE]
  check <- expected[expected$split == split, ]
  stopifnot(nrow(values) == check$rows, ncol(values) == check$columns)
  binary <- tempfile()
  writeBin(as.double(t(values)), binary, size = 4, endian = "little")
  actual <- unname(tools::md5sum(binary))
  unlink(binary)
  if (!identical(actual, check$md5_float32_le_row_major))
    stop("Numeric checksum mismatch for ", split, "; no CSV exported for this split.")
  lines <- apply(values, 1, function(row) paste(sprintf("%.17g", row), collapse = ","))
  writeLines(lines, file.path(out, paste0("sample_", split, ".csv")))
  message(split, ": verified ", nrow(values), " x ", ncol(values), " values")
}
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
message("Reprocessed data written to ", out)
