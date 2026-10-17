args <- commandArgs(trailingOnly = TRUE)
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
# Rscript may encode spaces as ~+~ in its --file argument.
script_path <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
script_dir <- dirname(normalizePath(script_path, mustWork = TRUE))
project_root <- normalizePath(file.path(script_dir, ".."))

source(file.path(script_dir, "lib", "case_study_config.R"))
check_case_study_batts()

seed <- case_study$null_seed
data_dir <- runtime_path("TWO_SAMPLE_DATA_DIR")
train_path <- file.path(data_dir, "sample_train.csv")
test_path <- file.path(data_dir, "sample_test.csv")
output_dir <- runtime_path("TWO_SAMPLE_NULL_OUTPUT", file.path(runtime_path("TWO_SAMPLE_RUN_ROOT"), "null"))
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

train <- read_case_study_matrix(train_path, case_study$dimensions[["train"]])
test <- read_case_study_matrix(test_path, case_study$dimensions[["test"]])
stopifnot(ncol(train) == ncol(test), all(is.finite(train)), all(is.finite(test)))
stopifnot(all(train >= 0 & train <= 1), all(test >= 0 & test <= 1))

make_balanced_folds <- function(n) {
  sample(rep(0:1, length.out = n), size = n, replace = FALSE)
}

set.seed(seed)
train_fold <- make_balanced_folds(nrow(train))
test_fold <- make_balanced_folds(nrow(test))
train_sizes <- tabulate(train_fold + 1L, nbins = 2L)
test_sizes <- tabulate(test_fold + 1L, nbins = 2L)
stopifnot(diff(range(train_sizes)) <= 1L, diff(range(test_sizes)) <= 1L)
jsonlite::write_json(list(cohort_id = case_study$cohort_id, seed = seed,
  training_group_sizes = train_sizes, test_group_sizes = test_sizes,
  split_unit = "observation; participants may occur in both splits"),
  file.path(output_dir, "null_design.json"), pretty = TRUE, auto_unbox = TRUE)

# margin_scale < 0 tells BATTS the data already occupy the [0,1]^d domain.
# This preserves the compositional data exactly and avoids division by zero for
# training-constant columns such as V104.
set.seed(seed)
fit <- do.call(BATTS::batts, c(list(data = train, group_labels = train_fold), case_study$null))


evaluation <- BATTS::eval_balance_weight(fit, test, is_Bayes = TRUE)
log_ratio_draws <- 2 * log(evaluation$balancing_weight_BART)
stopifnot(all(is.finite(log_ratio_draws)))
# Check the package's prediction normalization against its stored draws.
training_evaluation <- BATTS::eval_balance_weight(fit, train, is_Bayes = TRUE)
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
fingerprint <- list(commit = case_study$batts_commit, settings = case_study$null,
                    cohort_id = case_study$cohort_id,
                    training_config_sha256 = case_study$training_config_sha256,
                    seed = seed,
                    source_hashes = case_study_sources(project_root, "run_figure_S12_null_experiment.R"),
                    inputs = unname(tools::md5sum(c(train_path, test_path))))
saveRDS(list(seed = seed, train_fold = train_fold, test_fold = test_fold,
             settings = case_study$null, fingerprint = fingerprint,
             fit = fit, log_ratio_draws = log_ratio_draws),
        file.path(output_dir, "null_experiment_fit.rds"))
writeLines(case_study$batts_commit, file.path(output_dir, "BATTS_commit.txt"))


print(coverage, row.names = FALSE)
cat("Intervals excluding zero:", sum(!covers_zero), "\n")

writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt"))
