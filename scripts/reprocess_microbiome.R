#!/usr/bin/env Rscript
# Reconstruct the corrected 2052-row cohort from its four pinned raw resources.
# Joins always use study and sample ID; the split seed and order are recorded.
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- gsub("~+~", " ", sub("^--file=", "", script_arg), fixed = TRUE)
root <- dirname(dirname(normalizePath(script_file)))
# Resolve the longest existing prefix so relative paths and symlinks cannot
# redirect runtime data into this source checkout, including a new leaf path.
external_path <- function(path, label, must_exist = FALSE) {
  path <- path.expand(path)
  if (!grepl("^/", path)) path <- file.path(getwd(), path)
  current <- path
  tail <- character()
  while (!file.exists(current) && !dir.exists(current)) {
    link <- Sys.readlink(current)
    if (!is.na(link) && nzchar(link)) stop(label, " contains an unresolved symlink: ", current)
    parent <- dirname(current)
    if (identical(parent, current)) stop("Cannot resolve ", label, ": ", path)
    tail <- c(basename(current), tail)
    current <- parent
  }
  resolved <- normalizePath(current, mustWork = TRUE)
  for (part in tail) {
    if (part == "..") resolved <- dirname(resolved)
    else if (part != ".") resolved <- file.path(resolved, part)
  }
  if (resolved == root || startsWith(resolved, paste0(root, "/")))
    stop(label, " must be outside the code repository: ", resolved)
  if (must_exist && !file.exists(resolved) && !dir.exists(resolved))
    stop(label, " does not exist: ", resolved)
  resolved
}
args <- commandArgs(TRUE)
if (length(args) == 1L && args[[1]] %in% c("--help", "-h")) {
  cat("Usage: Rscript scripts/reprocess_microbiome.R --data-dir RUNTIME_DATA --output NEW_DIR\n")
  quit(status = 0)
}
if (!length(args) || length(args) %% 2L) stop("Use --data-dir DIR --output NEW_DIR.")
opts <- setNames(as.list(args[seq.int(2L, length(args), 2L)]), args[seq.int(1L, length(args), 2L)])
if (anyDuplicated(names(opts)) || any(!names(opts) %in% c("--data-dir", "--output")) ||
    is.null(opts[["--data-dir"]]) || is.null(opts[["--output"]]))
  stop("Use --data-dir DIR --output NEW_DIR.")
data_dir <- external_path(opts[["--data-dir"]], "Runtime data", must_exist = TRUE)
out <- external_path(opts[["--output"]], "Replay output")
if (file.exists(out) || dir.exists(out)) stop("Choose a new replay output directory.")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
out <- normalizePath(out)
options(timeout = max(1200, getOption("timeout")))
provenance <- file.path(data_dir, "provenance")
taxa <- readLines(file.path(provenance, "taxa.txt"))
manifest <- read.csv(file.path(provenance, "samples.csv"), stringsAsFactors = FALSE)
inventory <- read.csv(file.path(provenance, "collection_resources.csv"), stringsAsFactors = FALSE)
resources <- inventory$title[inventory$study %in% unique(manifest$source_studies)]
stopifnot(length(resources) == 4L)
expected <- read.csv(file.path(provenance, "numeric_checksums.csv"))
stopifnot(length(taxa) == 123L, !anyDuplicated(taxa),
          nrow(manifest) == 2052L, !anyDuplicated(paste(manifest$source_studies, manifest$sample_id)),
          all(manifest$split %in% c("train", "test")))
x <- matrix(NA_real_, nrow(manifest), length(taxa))
for (resource in resources) {
  message("Loading ", resource)
  source <- inventory[inventory$title == resource, , drop = FALSE]
  stopifnot(nrow(source) == 1L)
  study <- source$study
  if (file.exists(source$path)) {
    source$path <- external_path(source$path, "Raw resource", must_exist = TRUE)
    actual <- strsplit(system2('sha256sum', shQuote(source$path), stdout = TRUE), ' ')[[1]][1]
    stopifnot(identical(actual, source$sha256))
    env <- new.env()
    loaded <- load(source$path, envir = env)
    stopifnot(identical(loaded, resource))
    abundance <- env[[loaded]]
  } else {
    if (!requireNamespace("curatedMetagenomicData", quietly = TRUE))
      stop("Install curatedMetagenomicData or restore the pinned source cache.")
    if (as.character(utils::packageVersion("curatedMetagenomicData")) != "3.20.0")
      stop("Online replay requires validated curatedMetagenomicData 3.20.0 or the pinned raw cache.")
    pattern <- paste0("^", gsub(".", "\\.", resource, fixed = TRUE), "$")
    loaded <- curatedMetagenomicData::curatedMetagenomicData(pattern, dryrun = FALSE,
                                  counts = FALSE, rownames = "long")
    stopifnot(identical(names(loaded), resource))
    se <- loaded[[1]]
    stopifnot(identical(unique(as.character(SummarizedExperiment::colData(se)$study_name)), study))
    abundance <- as.matrix(SummarizedExperiment::assay(se))
  }
  i <- which(manifest$source_studies == study)
  j <- match(manifest$sample_id[i], colnames(abundance))
  if (anyNA(j)) stop("Missing manifest samples in ", resource)
  # mergeData() in the original filtering script fills absent taxa with zero.
  retained <- matrix(0, length(taxa), length(j))
  k <- match(taxa, rownames(abundance))
  retained[!is.na(k), ] <- abundance[k[!is.na(k)], j, drop = FALSE]
  x[i, ] <- t(retained)
}
stopifnot(all(is.finite(x)), all(x >= 0), all(rowSums(x) > 0))
x <- x / rowSums(x)
# The corrected protocol records normalization followed by float32 rounding.
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
