#!/usr/bin/env Rscript
# Validate the final analysis and export descriptive summaries in report order.
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
root <- dirname(dirname(normalizePath(script_file)))
source(file.path(root, "scripts", "lib", "case_study_config.R"))
run <- Sys.getenv("TWO_SAMPLE_RUN_ROOT", file.path(root, "output", "final"))
data_dir <- Sys.getenv("TWO_SAMPLE_DATA_DIR", file.path(root, "data"))
figure_dir <- Sys.getenv("TWO_SAMPLE_FIGURE_DIR", file.path(run, "figures"))
stopifnot(identical(readLines(file.path(run, "BATTS_commit.txt")), case_study$batts_commit))
intervals <- diagnostics <- list()
validated_fits <- 0L
for (split in c("train", "test")) for (key in case_study$keys) {
  paths <- file.path(data_dir, paste0("sample_", c(split, key), ".csv"))
  real <- read_case_study_matrix(paths[[1]], case_study$dimensions[[split]])
  gen <- read_case_study_matrix(paths[[2]], case_study$dimensions[["generated"]])
  z <- as.matrix(read.csv(file.path(run, "revision", split, paste0("log_w_", key, ".csv")), header = FALSE))
  stopifnot(nrow(z) == nrow(real) + nrow(gen), ncol(z) == 10L,
            all(is.finite(z)), all(z[, 4:9, drop = FALSE] <= z[, 5:10, drop = FALSE]))
  expected <- list(commit = case_study$batts_commit, settings = case_study,
                   source_hashes = case_study_sources(root, "refit_case_study.R"),
                   inputs = unname(tools::md5sum(paths)))
  for (est in c("batts", "gradient", "hellinger")) {
    saved <- readRDS(file.path(run, "fits", split, key, paste0(est, ".rds")))
    stopifnot(identical(saved$fingerprint, expected))
    validated_fits <- validated_fits + 1L
    if (est == "batts") {
      draws <- 2 * log(saved$fit$balance_weight_BART_data)
      stopifnot(identical(dim(draws), c(nrow(z), case_study$bayes$size_backfitting)),
                all(is.finite(draws)), max(abs(rowMeans(draws) - z[, 3])) < 1e-10)
      quantiles <- t(apply(draws, 1, quantile, probs = c(.025,.05,.1,.5,.9,.95,.975)))
      stopifnot(max(abs(quantiles - z[,4:10])) < 1e-10)
    } else {
      column <- if (est == "gradient") 1L else 2L
      stopifnot(max(abs(2 * log(saved$fit$balance_weight_boosting_data) - z[,column])) < 1e-10)
      curve <- colMeans(saved$fit$loss_CV_store)
      stopifnot(length(curve) == case_study$boosting$num_trees_max, all(is.finite(curve)))
      diagnostics[[length(diagnostics) + 1L]] <- data.frame(split = split, generator = key,
        estimator = est, selected_trees = which.min(curve), minimum_cv_loss = min(curve),
        selected_at_search_limit = which.min(curve) == length(curve))
    }
  }
  for (group in c("Real", "Generated")) {
    idx <- if (group == "Real") seq_len(nrow(real)) else seq.int(nrow(real) + 1L, nrow(z))
    intervals[[length(intervals) + 1L]] <- data.frame(split = split, generator = key,
      source = group, n = length(idx), contain_zero = sum(z[idx, 4] <= 0 & z[idx, 10] >= 0),
      mean_abs_log_ratio = mean(abs(z[idx, 3])), mean_interval_width = mean(z[idx, 10] - z[idx, 4]))
  }
}
ci <- do.call(rbind, intervals)
cv <- do.call(rbind, diagnostics)
stopifnot(nrow(ci) == 16L, nrow(cv) == 16L, validated_fits == 24L)
null_dir <- Sys.getenv("TWO_SAMPLE_NULL_OUTPUT", file.path(run, "null"))
null <- read.csv(file.path(null_dir, "test_pointwise_credible_intervals.csv"))
null_fit <- readRDS(file.path(null_dir, "null_experiment_fit.rds"))
null_expected <- list(commit = case_study$batts_commit, settings = case_study$null,
  seed = case_study$null_seed,
  source_hashes = case_study_sources(root, "run_figure_S12_null_experiment.R"),
  inputs = unname(tools::md5sum(file.path(data_dir, paste0("sample_", c("train", "test"), ".csv")))))
stopifnot(identical(null_fit$fingerprint, null_expected), nrow(null) == case_study$dimensions[["test"]],
          identical(null$test_row, seq_len(nrow(null))),
          all(is.finite(as.matrix(null[, c("posterior_mean", "ci_lower_025", "ci_upper_975")]))),
          all(null$ci_lower_025 <= null$ci_upper_975),
          max(abs(rowMeans(null_fit$log_ratio_draws) - null$posterior_mean)) < 1e-10)
null_quantiles <- t(apply(null_fit$log_ratio_draws, 1, quantile, probs=c(.025,.975)))
stopifnot(max(abs(null_quantiles - as.matrix(null[,c("ci_lower_025","ci_upper_975")]))) < 1e-10,
          identical(null$covers_zero, null$ci_lower_025 <= 0 & null$ci_upper_975 >= 0))
figures <- file.path(figure_dir, paste0("figure", case_study$figures, "_ggplot2.pdf"))
stopifnot(all(file.exists(figures)), all(file.info(figures)$size > 1000))
for (path in figures) {
  connection <- file(path, "rb")
  signature <- readBin(connection, "raw", n = 5L)
  close(connection)
  stopifnot(identical(rawToChar(signature), "%PDF-"))
}
write.csv(ci, file.path(run, "interval_summary.csv"), row.names = FALSE)
write.csv(cv, file.path(run, "boosting_cv_summary.csv"), row.names = FALSE)
validation <- list(BATTS_commit = case_study$batts_commit, BATTS_version = case_study$batts_version,
  model_order = case_study$keys, real_train = case_study$dimensions[["train"]],
  real_test = case_study$dimensions[["test"]], taxa = case_study$dimensions[["taxa"]],
  generated_per_model = case_study$dimensions[["generated"]],
  validated_comparisons = 8L, validated_estimator_fits = validated_fits,
  validated_figures = length(figures), null_intervals = nrow(null),
  null_intervals_containing_zero = sum(null$ci_lower_025 <= 0 & null$ci_upper_975 >= 0),
  boosting_at_search_limit = sum(cv$selected_at_search_limit),
  figure_md5 = as.list(setNames(unname(tools::md5sum(figures)), basename(figures))))
jsonlite::write_json(validation, file.path(run, "analysis_validation.json"),
                     pretty = TRUE, auto_unbox = TRUE, digits = NA)
lines <- c("# Final microbiome case study", "",
  sprintf("BATTS %s, source commit `%s`.", case_study$batts_version, case_study$batts_commit), "",
  "Original pooled preprocessing and split: 1,166 training observations, 292 test observations, 123 taxa. Each generator contributes 1,000 compositions. Model order: ICFM, DT, D, MBGAN.", "",
  "D and stabilized DT are fitted on the original training rows. ICFM and MBGAN replay recovered checkpoints; historical neural fitting membership and complete training settings remain unverified.", "",
  "All eight train/test comparisons are independently fitted. Test density-ratio analyses use the real test observations themselves; they are not predictions from a density-ratio estimator fitted on the training comparison.", "",
  "The previous protocol is explicit: 200 trees, 2,000 burn-in and 1,000 retained iterations, thinning 1, fixed lambda 5, seed 1. Boosting uses five-fold CV, up to 1,000 trees, depth 4, learning rate 0.01, 32 bins, and subsample_fraction = 1. See settings.json for all arguments.", "",
  "## Real-test descriptive summaries", "",
  "| Generator | Intervals excluding zero | Mean absolute posterior-mean log ratio |",
  "| --- | ---: | ---: |")
for (key in case_study$keys) {
  row <- ci[ci$split == "test" & ci$generator == key & ci$source == "Real", ]
  lines <- c(lines, sprintf("| %s | %d/%d | %.6f |", case_study$labels[[key]],
    row$n - row$contain_zero, row$n, row$mean_abs_log_ratio))
}
lines <- c(lines, "",
  "These pointwise posterior intervals are descriptive. Inclusion of zero does not establish distributional equality; one null experiment does not establish a repeated-sampling coverage guarantee.", "",
  sprintf("Null intervals containing zero: %d/%d. Boosting fits at the 1,000-tree cap: %d/16.",
    validation$null_intervals_containing_zero, nrow(null), validation$boosting_at_search_limit), "",
  "The original split is by observation and does not establish participant-level separation. Figures 6, 7, S8 and S9 use corrected Bray-Curtis PCoA; S10/S11 retain classical uncorrected PCoA. Posterior intervals are pointwise.", "",
  "All eight comparison tables, 24 estimator fingerprints, the null diagnostic, and seven PDF signatures passed validation. analysis_validation.json records the checks.")
writeLines(lines, file.path(run, "REPORT.md"))
writeLines(format(Sys.time(), tz = "UTC", usetz = TRUE), file.path(run, "ANALYSIS_COMPLETE"))
print(ci, row.names = FALSE)
cat("All eight comparison tables, null intervals, and seven figures passed checks.\n")
