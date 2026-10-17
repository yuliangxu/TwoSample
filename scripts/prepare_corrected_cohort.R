#!/usr/bin/env Rscript
# Recompute collection-wide eligibility from all pinned raw source matrices.
# Metadata is joined by (study, sample_id); row-position masks are never used.
# Required external inputs are never distributed with this code-only repository.
script_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- gsub("~+~", " ", sub("^--file=", "", script_arg), fixed = TRUE)
root <- dirname(dirname(normalizePath(script_file, mustWork = TRUE)))
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
  cat("Usage: Rscript scripts/prepare_corrected_cohort.R --source-provenance DIR --output NEW_DIR [--cache-root DIR]\n",
      "External DIR requires collection_resources.csv, collection_metadata_ids.csv,\n",
      "collection_species.txt, taxa.txt and cohort_order.csv. Cache-root relocates\n",
      "resource paths by basename; every source file must match its recorded SHA-256.\n")
  quit(status = 0)
}
if (!length(args) || length(args) %% 2L != 0L) stop("Arguments must be --option value pairs; use --help.")
opts <- setNames(as.list(args[seq.int(2L, length(args), 2L)]), args[seq.int(1L, length(args), 2L)])
if (anyDuplicated(names(opts)) || any(!names(opts) %in% c("--source-provenance", "--output", "--cache-root")))
  stop("Unknown or duplicated argument; use --help.")
if (is.null(opts[["--source-provenance"]]) || is.null(opts[["--output"]]))
  stop("Both --source-provenance and --output are required.")
prov <- external_path(opts[["--source-provenance"]], "Source provenance", must_exist = TRUE)
out <- external_path(opts[["--output"]], "Preparation output")
if (file.exists(out) || dir.exists(out)) stop("Use a new preparation output directory.")
required <- c("collection_resources.csv", "collection_metadata_ids.csv", "collection_species.txt", "taxa.txt", "cohort_order.csv")
missing <- required[!file.exists(file.path(prov, required))]
if (length(missing)) stop("Missing external preprocessing inputs: ", paste(missing, collapse = ", "))
resources <- read.csv(file.path(prov, 'collection_resources.csv'), stringsAsFactors = FALSE)
metadata <- read.csv(file.path(prov, 'collection_metadata_ids.csv'), stringsAsFactors = FALSE)
universe <- readLines(file.path(prov, 'collection_species.txt'))
reference_taxa <- readLines(file.path(prov, 'taxa.txt'))
order <- read.csv(file.path(prov, 'cohort_order.csv'), stringsAsFactors = FALSE)
require_columns <- function(x, columns, name) {
  missing <- setdiff(columns, names(x))
  if (length(missing)) stop(name, " is missing columns: ", paste(missing, collapse = ", "))
}
require_columns(resources, c("study", "title", "path", "sha256"), "collection_resources.csv")
require_columns(metadata, c("study", "sample_id"), "collection_metadata_ids.csv")
require_columns(order, c("cohort_row", "source_studies", "sample_id", "historical_filtered_column"), "cohort_order.csv")
if (!is.null(opts[["--cache-root"]])) {
  cache <- external_path(opts[["--cache-root"]], "Source cache", must_exist = TRUE)
  if (anyDuplicated(basename(resources$path))) stop("Cache filenames are not unique.")
  resources$path <- file.path(cache, basename(resources$path))
}
# Relative cache paths are resolved against the external provenance directory.
relative <- !grepl("^/", resources$path)
resources$path[relative] <- file.path(prov, resources$path[relative])
missing <- resources$path[!file.exists(resources$path)]
if (length(missing)) stop("Missing pinned raw cache files (use --cache-root to relocate): ", paste(missing, collapse = ", "))
resources$path <- vapply(resources$path, external_path, character(1), label = "Raw resource", must_exist = TRUE)
dir.create(out, recursive = TRUE)
out <- normalizePath(out, mustWork = TRUE)
for (name in setdiff(required, "collection_resources.csv"))
  stopifnot(file.copy(file.path(prov, name), file.path(out, name), overwrite = FALSE))
write.csv(resources, file.path(out, "collection_resources.csv"), row.names = FALSE)
source_hash <- function(path) strsplit(system2("sha256sum", shQuote(path), stdout = TRUE), " ")[[1]][1]
write.csv(data.frame(name = required, source_path = file.path(prov, required),
                    sha256 = vapply(file.path(prov, required), source_hash, character(1))),
          file.path(out, "preparation_source_inputs.csv"), row.names = FALSE)
key <- function(study, sample) paste(study, sample, sep = '::')
stopifnot(nrow(resources) == 93L, nrow(metadata) == 22584L,
          !anyDuplicated(key(metadata$study, metadata$sample_id)),
          !anyDuplicated(resources$study), !anyDuplicated(universe),
          length(reference_taxa) == 123L,
          nrow(order) == 2052L, !anyDuplicated(order$sample_id),
          identical(order$cohort_row, seq_len(nrow(order))))
load_species <- function(i) {
  actual <- strsplit(system2('sha256sum', shQuote(resources$path[i]), stdout = TRUE), ' ')[[1]][1]
  stopifnot(identical(actual, resources$sha256[i]))
  env <- new.env()
  loaded <- load(resources$path[i], envir = env)
  stopifnot(identical(loaded, resources$title[i]))
  raw <- env[[loaded]]
  ids <- metadata$sample_id[metadata$study == resources$study[i]]
  stopifnot(all(ids %in% colnames(raw)), !anyDuplicated(ids))
  x <- raw[intersect(rownames(raw), universe), ids, drop = FALSE]
  stopifnot(is.matrix(x), all(is.finite(x)), all(x >= 0),
            !anyDuplicated(rownames(x)), all(grepl('\\|s__', rownames(x))))
  x
}
taxon_sums <- setNames(numeric(length(universe)), universe)
complete_count <- 0L
resource_counts <- list()
for (i in seq_len(nrow(resources))) {
  x <- load_species(i)
  complete <- colSums(x) >= 99
  taxon_sums[rownames(x)] <- taxon_sums[rownames(x)] + rowSums(x[, complete, drop = FALSE])
  complete_count <- complete_count + sum(complete)
  resource_counts[[i]] <- data.frame(study = resources$study[i], species = nrow(x),
                                    observations = ncol(x), complete = sum(complete))
}
taxon_means <- taxon_sums / complete_count
selected_taxa <- names(taxon_means)[taxon_means >= 0.1]
stopifnot(setequal(selected_taxa, reference_taxa), complete_count == 22001L)
selected_taxa <- reference_taxa
statistics <- list()
cohort <- matrix(NA_real_, nrow(order), length(selected_taxa),
                  dimnames = list(order$sample_id, selected_taxa))
source_title <- character(nrow(order))
for (i in seq_len(nrow(resources))) {
  x <- load_species(i)
  retained <- matrix(0, length(selected_taxa), ncol(x),
                     dimnames = list(selected_taxa, colnames(x)))
  present <- intersect(selected_taxa, rownames(x))
  retained[present, ] <- x[present, , drop = FALSE]
  statistics[[i]] <- data.frame(study = resources$study[i], sample_id = colnames(x),
    total_abundance_percent = colSums(x), retained_mean_percent = colMeans(retained),
    zero_taxa = colSums(retained == 0))
  wanted <- which(order$source_studies == resources$study[i])
  if (length(wanted)) {
    matched <- match(key(order$source_studies[wanted], order$sample_id[wanted]),
                     key(resources$study[i], colnames(x)))
    stopifnot(!anyNA(matched))
    cohort[wanted, ] <- t(retained[, matched, drop = FALSE])
    source_title[wanted] <- resources$title[i]
  }
}
stats <- do.call(rbind, statistics)
mean_cutoff <- mean(stats$retained_mean_percent)
stats$passes_total <- stats$total_abundance_percent >= 99
stats$passes_retained_mean <- stats$retained_mean_percent >= mean_cutoff
stats$eligible <- stats$passes_total & stats$passes_retained_mean
studies <- c('HMP_2019_ibdmdb', 'HMP_2019_t2d', 'HallAB_2017', 'HanniganGD_2017')
stats$selected_four_study_cohort <- stats$eligible & stats$study %in% studies
selected_keys <- key(stats$study[stats$selected_four_study_cohort],
                     stats$sample_id[stats$selected_four_study_cohort])
stopifnot(sum(stats$eligible) == 16824L, sum(stats$selected_four_study_cohort) == 2052L,
          setequal(selected_keys, key(order$source_studies, order$sample_id)),
          all(is.finite(cohort)), all(cohort >= 0), all(rowSums(cohort) > 0))
expected_counts <- c(HMP_2019_ibdmdb = 1491L, HMP_2019_t2d = 277L,
                     HallAB_2017 = 209L, HanniganGD_2017 = 75L)
stopifnot(all(table(order$source_studies)[names(expected_counts)] == expected_counts))
cohort <- cohort / rowSums(cohort)
order$source_resource <- source_title
write.csv(order, file.path(out, 'cohort_order.csv'), row.names = FALSE)
writeBin(as.double(t(cohort)), file.path(out, 'cohort_normalized_float64_le.bin'),
         size = 8, endian = 'little')
write.csv(stats, file.path(out, 'collection_sample_filter.csv'), row.names = FALSE)
write.csv(data.frame(taxon = names(taxon_means), mean_abundance_percent = taxon_means,
                     selected = taxon_means >= 0.1),
          file.path(out, 'collection_taxon_filter.csv'), row.names = FALSE)
write.csv(do.call(rbind, resource_counts), file.path(out, 'collection_resource_counts.csv'), row.names = FALSE)
write.csv(data.frame(source_studies = 93L, source_observations = nrow(stats),
                    complete_observations = complete_count, selected_taxa = length(selected_taxa),
                    retained_mean_percent_cutoff = mean_cutoff,
                    eligible_collection_observations = sum(stats$eligible), selected_observations = nrow(cohort)),
          file.path(out, 'preprocessing_summary.csv'), row.names = FALSE)
writeLines(capture.output(sessionInfo()), file.path(out, 'sessionInfo.txt'))
writeLines('Source-based collection preprocessing and corrected 2052-ID selection verified.',
           file.path(out, 'PREPARATION_COMPLETE'))
cat('Verified', nrow(cohort), 'observations across', ncol(cohort), 'taxa from raw sources.\n')
