#!/usr/bin/env Rscript
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
root <- dirname(dirname(normalizePath(script_file)))
run <- file.path(root, "output", "batts_909ea357")
read_matrix <- function(path) as.matrix(read.csv(path, header = FALSE))
differences <- list(); intervals <- list(); diagnostics <- list()
classify <- function(z) ifelse(z[, 4] > 0, "above", ifelse(z[, 10] < 0, "below", "covers"))
for (split in c("train", "test")) for (key in c("d", "dt", "icfm", "mbgan")) {
  old <- read_matrix(file.path(root, "data", "revision", split, paste0("log_w_", key, ".csv")))
  new <- read_matrix(file.path(run, "revision", split, paste0("log_w_", key, ".csv")))
  stopifnot(identical(dim(old), dim(new)), all(is.finite(new)), all(new[, 4] <= new[, 10]))
  for (j in seq_len(10)) {
    delta <- new[, j] - old[, j]
    differences[[length(differences) + 1L]] <- data.frame(
      split = split, generator = key,
      statistic = c("gradient", "hellinger", "posterior_mean", "q025", "q05", "q10", "q50", "q90", "q95", "q975")[[j]],
      max_abs_difference = max(abs(delta)), rmse = sqrt(mean(delta^2)),
      mean_difference = mean(delta), correlation = cor(new[, j], old[, j]))
  }
  n_real <- if (split == "train") 1166L else 292L
  for (group in c("Real", "Generated")) {
    idx <- if (group == "Real") seq_len(n_real) else seq.int(n_real + 1L, nrow(new))
    intervals[[length(intervals) + 1L]] <- data.frame(
      split = split, generator = key, source = group, n = length(idx),
      old_cover_zero = sum(classify(old[idx, ]) == "covers"),
      new_cover_zero = sum(classify(new[idx, ]) == "covers"),
      changed_interval_class = sum(classify(new[idx, ]) != classify(old[idx, ])),
      new_mean_abs_log_ratio = mean(abs(new[idx, 3])),
      new_mean_interval_width = mean(new[idx, 10] - new[idx, 4]))
  }
  for (estimator in c("gradient", "hellinger")) {
    fit <- readRDS(file.path(run, "fits", split, key, paste0(estimator, ".rds")))$fit
    curve <- colMeans(fit$loss_CV_store)
    diagnostics[[length(diagnostics) + 1L]] <- data.frame(
      split = split, generator = key, estimator = estimator,
      selected_trees = which.min(curve), minimum_cv_loss = min(curve),
      selected_at_search_limit = which.min(curve) == length(curve))
  }
}
write.csv(do.call(rbind, differences), file.path(run, "comparison_to_paper.csv"), row.names = FALSE)
write.csv(do.call(rbind, intervals), file.path(run, "interval_summary.csv"), row.names = FALSE)
write.csv(do.call(rbind, diagnostics), file.path(run, "boosting_cv_summary.csv"), row.names = FALSE)
old <- read.csv(file.path(root, "data", "figure_S12", "test_pointwise_credible_intervals.csv"))
new <- read.csv(file.path(run, "null", "test_pointwise_credible_intervals.csv"))
stopifnot(identical(old$test_row, new$test_row))
fields <- c("posterior_mean", "ci_lower_025", "ci_upper_975")
null_diff <- do.call(rbind, lapply(fields, function(field) data.frame(
  statistic = field, max_abs_difference = max(abs(new[[field]] - old[[field]])),
  rmse = sqrt(mean((new[[field]] - old[[field]])^2)))))
write.csv(null_diff, file.path(run, "null_comparison_to_paper.csv"), row.names = FALSE)
print(do.call(rbind, intervals), row.names = FALSE)
print(null_diff, row.names = FALSE)
coverage <- read.csv(file.path(run, "null", "coverage_summary.csv"))
means <- do.call(rbind, differences)
means <- means[means$statistic == "posterior_mean", ]
ci <- do.call(rbind, intervals)
cv <- do.call(rbind, diagnostics)
lines <- c(
  "# Case-study rerun with BATTS 909ea357", "",
  "BATTS source commit: `909ea357bdb3f97018613552ef13430d1bb60c6e`.", "",
  "Eight real-versus-generated comparisons were refitted with gradient boosting,",
  "Hellinger boosting, and Bayesian additive trees. The null experiment was",
  "refitted separately, and Figures 6, 7, S8, S9, S10, S11, and S12 were redrawn.",
  "Real and generated input compositions are unchanged. Figures 6 and S8 depend",
  "only on these compositions, so their ordinations are unaffected by BATTS.", "",
  "## Posterior means compared with the saved paper results", "",
  "Generator keys: d = Dirichlet; dt = Dirichlet Tree; icfm = ICFM; mbgan = MB-GAN.", "",
  "Differences below are in log-density-ratio units and include real and",
  "generated observations. They describe the rerun versus the bundled results;",
  "they do not isolate individual package changes from numerical/environment effects.", "",
  "| Split | Generator | Maximum absolute difference | RMSE | Correlation |",
  "| --- | --- | ---: | ---: | ---: |",
  apply(means, 1, function(z) sprintf("| %s | %s | %.6g | %.6g | %.6f |",
    z[["split"]], z[["generator"]], as.numeric(z[["max_abs_difference"]]),
    as.numeric(z[["rmse"]]), as.numeric(z[["correlation"]]))), "",
  "## Pointwise intervals containing zero", "",
  "These are descriptive counts for the fitted comparisons, not coverage",
  "estimates under a known null. Only the separate null experiment has that design.", "",
  "| Split | Generator | Observations | n | Paper | Rerun | Changed interval class |",
  "| --- | --- | --- | ---: | ---: | ---: | ---: |",
  apply(ci, 1, function(z) paste0("| ", paste(z[c("split", "generator", "source", "n",
    "old_cover_zero", "new_cover_zero", "changed_interval_class")], collapse = " | "), " |")), "",
  "## Held-out null experiment", "",
  sprintf("%d of %d test intervals contain zero (%.1f%% observed coverage).",
    coverage$n_cover[1], coverage$n[1], 100 * coverage$proportion_cover[1]), "",
  sprintf("The maximum change in the posterior mean from the paper is %.6g.",
    null_diff$max_abs_difference[null_diff$statistic == "posterior_mean"]), "",
  "## Settings and scope", "",
  "Comparisons: seed 1 for each estimator; 200 Bayesian trees; 2,000 burn-in",
  "iterations; 1,000 retained iterations; thinning 1; fixed lambda 5; margin 0.1.",
  "Boosting: five-fold CV; up to 1,000 trees; depth 4; learning rate 0.01; 32 bins.",
  "Training and testing comparisons are fitted separately, following the original",
  "case-study scripts; these are not predictions from a model trained on the other split.", "",
  "Null: seed 2026; balanced random labels for the real training samples; 200 trees;",
  "500 burn-in and 500 retained iterations; fixed lambda 5; margin -1;",
  "evaluation on all 292 held-out real test observations.", "",
  sprintf("%d of %d boosting fits selected the 1,000-tree search limit; the original search range was retained.",
    sum(cv$selected_at_search_limit), nrow(cv)), "",
  "This reproduces the specified single-chain workflow; no new multi-chain",
  "convergence claim is made. The four generative models were not retrained.", "",
  "## Files", "",
  "- `revision/train/` and `revision/test/`: all ten-column estimator/quantile tables.",
  "- `comparison_to_paper.csv`: differences for all estimators and posterior summaries.",
  "- `interval_summary.csv`: interval classifications by split, generator and source.",
  "- `boosting_cv_summary.csv`: CV-selected tree counts and losses.",
  "- `null/`: held-out intervals and coverage; `null_comparison_to_paper.csv`: numerical changes.",
  "- `input_checksums.csv`, `BATTS_commit.txt`, `settings.R`, `sessionInfo.txt`: run provenance.",
  "- `fits/` in the local output directory: cached fitted objects and posterior draws.", "",
  "## Figures", "",
  vapply(c("6", "7", "S8", "S9", "S10", "S11", "S12"), function(label)
    sprintf("- [Figure %s](figures/figure%s_ggplot2.pdf)", label, label), character(1)), "",
  "Use `Rscript scripts/reproduce_case_study_batts.R 2` from the repository root",
  "after installing the exact BATTS commit specified in the main README.")
writeLines(lines, file.path(run, "REPORT.md"))
