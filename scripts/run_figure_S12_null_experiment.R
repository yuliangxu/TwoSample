args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
# Rscript may encode spaces as ~+~ in its --file argument.
script_path <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
script_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
project_root <- normalizePath(file.path(script_dir, ".."))

suppressPackageStartupMessages(library(BATTS))

seed <- 2026L
train_path <- file.path(project_root, "data", "sample_train.csv")
test_path <- file.path(project_root, "data", "sample_test.csv")
output_dir <- Sys.getenv("TWO_SAMPLE_NULL_OUTPUT", file.path(project_root, "output", "figure_S12_refit"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- as.matrix(read.csv(train_path, header = FALSE, check.names = FALSE))
test <- as.matrix(read.csv(test_path, header = FALSE, check.names = FALSE))
stopifnot(ncol(train) == ncol(test), all(is.finite(train)), all(is.finite(test)))
stopifnot(all(train >= 0 & train <= 1), all(test >= 0 & test <= 1))

make_balanced_folds <- function(n) {
  sample(rep(0:1, length.out = n), size = n, replace = FALSE)
}

set.seed(seed)
train_fold <- make_balanced_folds(nrow(train))
test_fold <- make_balanced_folds(nrow(test))

# margin_scale < 0 tells BATTS the data already occupy the [0,1]^d domain.
# This preserves the compositional data exactly and avoids division by zero for
# training-constant columns such as V104.
set.seed(seed)
fit <- batts(
  data = train,
  group_labels = train_fold,
  num_trees = 200,
  margin_scale = -1,
  size_burnin = 500,
  size_backfitting = 500,
  thin = 1,
  lambda_0 = 5,
  update_lambda = FALSE,
  output_BART_ensembles = TRUE,
  quiet = FALSE
)

evaluation <- eval_balance_weight(fit, test, is_Bayes = TRUE)
log_ratio_draws <- 2 * log(evaluation$balancing_weight_BART)
stopifnot(all(is.finite(log_ratio_draws)))
# Check the package's prediction normalization against its stored draws.
training_evaluation <- eval_balance_weight(fit, train, is_Bayes = TRUE)
prediction_difference <- max(abs(2 * log(training_evaluation$balancing_weight_BART) -
                                 2 * log(fit$balance_weight_BART_data)))
stopifnot(is.finite(prediction_difference), prediction_difference < 1e-10)
writeLines(paste("Training prediction/draw agreement; max abs log-ratio difference",
                 prediction_difference), file.path(output_dir, "prediction_check.txt"))
rm(training_evaluation)
posterior_mean <- rowMeans(log_ratio_draws)
ci <- t(apply(log_ratio_draws, 1, quantile, probs = c(0.025, 0.975), names = FALSE))
covers_zero <- ci[, 1] <= 0 & ci[, 2] >= 0
interval_class <- ifelse(ci[, 2] < 0, "below_zero", ifelse(ci[, 1] > 0, "above_zero", "covers_zero"))

results <- data.frame(
  test_row = seq_len(nrow(test)),
  test_fold = test_fold,
  posterior_mean = posterior_mean,
  ci_lower_025 = ci[, 1],
  ci_upper_975 = ci[, 2],
  covers_zero = covers_zero,
  interval_class = interval_class
)

coverage <- rbind(
  data.frame(group = "overall", n = nrow(results), n_cover = sum(results$covers_zero),
             proportion_cover = mean(results$covers_zero)),
  do.call(rbind, lapply(0:1, function(k) {
    z <- results$test_fold == k
    data.frame(group = paste0("test_fold_", k), n = sum(z),
               n_cover = sum(results$covers_zero[z]),
               proportion_cover = mean(results$covers_zero[z]))
  }))
)

write.csv(results, file.path(output_dir, "test_pointwise_credible_intervals.csv"), row.names = FALSE)
write.csv(coverage, file.path(output_dir, "coverage_summary.csv"), row.names = FALSE)
write.csv(results[!results$covers_zero, ], file.path(output_dir, "intervals_excluding_zero.csv"), row.names = FALSE)
saveRDS(list(seed = seed, train_fold = train_fold, test_fold = test_fold,
             settings = list(num_trees = 200, size_burnin = 500,
                             size_backfitting = 500, thin = 1,
                             lambda_0 = 5, margin_scale = -1),
             fit = fit, log_ratio_draws = log_ratio_draws),
        file.path(output_dir, "null_experiment_fit.rds"))

print(coverage, row.names = FALSE)
cat("Intervals excluding zero:", sum(!covers_zero), "\n")

writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
