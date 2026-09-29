#!/usr/bin/env Rscript
# Full case-study refit. Arguments: workers, optional --posterior-only.
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
project_root <- dirname(dirname(normalizePath(script_file)))
script_dir <- file.path(project_root, "scripts")
args <- commandArgs(TRUE)
workers <- if (length(args)) as.integer(args[[1]]) else 1L
posterior_only <- "--posterior-only" %in% args
stopifnot(length(workers) == 1L, !is.na(workers), workers >= 1L)
suppressPackageStartupMessages(library(BATTS))
commit <- "909ea357bdb3f97018613552ef13430d1bb60c6e"
installed_commit <- packageDescription("BATTS")$RemoteSha
stamp <- system.file("source_commit.txt", package = "BATTS")
if (is.null(installed_commit) && nzchar(stamp)) installed_commit <- readLines(stamp)
if (!identical(installed_commit, commit))
  stop("Install nawaya040/BATTS@", commit, " with remotes::install_github first.")
run_root <- file.path(project_root, "output", "batts_909ea357")
dir.create(run_root, recursive = TRUE, showWarnings = FALSE)
writeLines(commit, file.path(run_root, "BATTS_commit.txt"))
writeLines(capture.output(sessionInfo()), file.path(run_root, "sessionInfo.txt"))
data_dir <- file.path(project_root, "data")
inputs <- file.path(data_dir, paste0("sample_", c("train", "test", "d", "dt", "icfm", "mbgan"), ".csv"))
write.csv(data.frame(file = basename(inputs), md5 = unname(tools::md5sum(inputs))),
          file.path(run_root, "input_checksums.csv"), row.names = FALSE)
settings <- list(seed = 1L, num_trees = 200L, size_burnin = 2000L,
                 size_backfitting = 1000L, thin = 1L, lambda_0 = 5,
                 update_lambda = FALSE, margin_scale = 0.1,
                 num_trees_max = 1000L, K_CV = 5L, max_resol_boosting = 4L,
                 learn_rate = 0.01, n_bins_boosting = 32L)
dput(settings, file.path(run_root, "settings.R"))
read_matrix <- function(path) as.matrix(read.csv(path, header = FALSE))
fit_one <- function(job) {
  split <- job[[1]]; key <- job[[2]]
  message(format(Sys.time()), " Starting ", split, "/", key)
  reference_path <- file.path(data_dir, paste0("sample_", split, ".csv"))
  generated_path <- file.path(data_dir, paste0("sample_", key, ".csv"))
  reference <- read_matrix(reference_path)
  generated <- read_matrix(generated_path)
  x <- rbind(reference, generated)
  labels <- c(rep(0L, nrow(reference)), rep(1L, nrow(generated)))
  stopifnot(all(is.finite(x)), all(x >= 0 & x <= 1),
            all(apply(x, 2, function(v) diff(range(v))) > 0))
  destination <- file.path(run_root, "revision", split)
  dir.create(destination, recursive = TRUE, showWarnings = FALSE)
  checkpoint_dir <- file.path(run_root, "fits", split, key)
  dir.create(checkpoint_dir, recursive = TRUE, showWarnings = FALSE)
  fingerprint <- list(commit = commit, settings = settings,
                      inputs = unname(tools::md5sum(c(reference_path, generated_path))))
  component <- function(name, fn) {
    path <- file.path(checkpoint_dir, paste0(name, ".rds"))
    if (file.exists(path)) {
      saved <- readRDS(path)
      if (identical(saved$fingerprint, fingerprint)) return(saved)
    }
    message(format(Sys.time()), " Fitting ", split, "/", key, "/", name)
    set.seed(settings$seed)
    start <- proc.time()[["elapsed"]]
    fit <- fn()
    saved <- list(fingerprint = fingerprint, fit = fit,
                  elapsed_seconds = proc.time()[["elapsed"]] - start)
    saveRDS(saved, path)
    message(format(Sys.time()), " Completed ", split, "/", key, "/", name,
            " in ", round(saved$elapsed_seconds, 1), " seconds")
    saved
  }
  # Posterior fitting is independent of the two boosting fits.
  bayes <- component("batts", function() BATTS::batts(
    data = x, group_labels = labels, num_trees = settings$num_trees,
    size_burnin = settings$size_burnin, size_backfitting = settings$size_backfitting,
    thin = settings$thin, lambda_0 = settings$lambda_0, update_lambda = FALSE,
    margin_scale = settings$margin_scale, output_BART_ensembles = FALSE, quiet = TRUE))
  if (posterior_only) return(data.frame(split = split, generator = key,
                                      estimator = "batts", seconds = bayes$elapsed_seconds))
  boost <- function(gradient) BATTS::boots(
    data = x, group_labels = labels, num_trees_max = settings$num_trees_max,
    K_CV = settings$K_CV, max_resol = settings$max_resol_boosting,
    learn_rate = settings$learn_rate, n_bins = settings$n_bins_boosting,
    margin_scale = settings$margin_scale, use_gradient = gradient, quiet = TRUE)
  grad <- component("gradient", function() boost(TRUE))
  hell <- component("hellinger", function() boost(FALSE))
  draws <- 2 * log(bayes$fit$balance_weight_BART_data)
  stopifnot(identical(dim(draws), c(nrow(x), settings$size_backfitting)), all(is.finite(draws)))
  quantiles <- t(apply(draws, 1, quantile, probs = c(.025, .05, .1, .5, .9, .95, .975)))
  result <- cbind(2 * log(grad$fit$balance_weight_boosting_data),
                  2 * log(hell$fit$balance_weight_boosting_data), rowMeans(draws), quantiles)
  stopifnot(all(is.finite(result)), all(result[, 4] <= result[, 10]))
  write.table(result, file.path(destination, paste0("log_w_", key, ".csv")),
              sep = ",", row.names = FALSE, col.names = FALSE)
  timing <- data.frame(split = split, generator = key,
                      estimator = c("batts", "gradient", "hellinger"),
                      seconds = c(bayes$elapsed_seconds, grad$elapsed_seconds, hell$elapsed_seconds))
  write.csv(timing, file.path(checkpoint_dir, "timing.csv"), row.names = FALSE)
  timing
}
grid <- expand.grid(key = c("d", "dt", "icfm", "mbgan"), split = c("test", "train"),
                    stringsAsFactors = FALSE)
jobs <- lapply(seq_len(nrow(grid)), function(i) c(grid$split[i], grid$key[i]))
results <- if (.Platform$OS.type != "windows" && workers > 1L)
  parallel::mclapply(jobs, fit_one, mc.cores = min(workers, length(jobs)), mc.preschedule = FALSE) else
  lapply(jobs, fit_one)
if (any(vapply(results, inherits, logical(1), "try-error"))) stop("One or more fits failed; see log.")
write.csv(do.call(rbind, results), file.path(run_root,
  if (posterior_only) "posterior_timing.csv" else "timing.csv"), row.names = FALSE)
message(if (posterior_only) "All eight posterior fits completed. Results: " else
        "All eight comparisons completed. Results: ", run_root)
