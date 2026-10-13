# Frozen final-case-study protocol. All model ordering and fitting settings live here.
case_study <- list(
  batts_repository = "nawaya040/BATTS",
  batts_commit = "147e511d211919afefb8decac6333e66d24919db",
  batts_version = "0.3.0",
  keys = c("icfm", "dt", "d", "mbgan"),
  labels = c(icfm = "ICFM", dt = "DT", d = "D", mbgan = "MBGAN"),
  dimensions = c(train = 1166L, test = 292L, generated = 1000L, taxa = 123L),
  seed = 1L,
  bayes = list(num_trees = 200L, max_resol = 0L, learn_rate = 0.01,
               n_bins = 100L, n_min_obs_per_node = 1L, n_ratio_per_node = 1e-100,
               margin_scale = 0.1, use_gradient = FALSE, size_burnin = 2000L,
               size_backfitting = 1000L, thin = 1L, prob_moves = c(1/3, 1/3, 1/3),
               lambda_0 = 5, omega_prior_parameters = c(1, 1),
               tree_priors = c(0.95, 2), output_BART_ensembles = FALSE, quiet = TRUE),
  boosting = list(num_trees_max = 1000L, K_CV = 5L, max_resol = 4L,
                  learn_rate = 0.01, n_bins = 32L, n_min_obs_per_node = 1L,
                  n_ratio_per_node = 1e-100, margin_scale = 0.1, quiet = TRUE,
                  subsample_fraction = 1),
  null_seed = 2026L,
  figures = c("6", "7", "S8", "S9", "S10", "S11", "S12"))
# BATTS 0.3.0 changed defaults only. Fraction 1 preserves the old case-study
# boosting protocol and RNG use; the package's new default is fraction 0.5.
case_study$null <- modifyList(case_study$bayes,
  list(size_burnin = 500L, size_backfitting = 500L, margin_scale = -1,
       output_BART_ensembles = TRUE))

check_case_study_batts <- function() {
  suppressPackageStartupMessages(requireNamespace("BATTS", quietly = TRUE))
  installed <- utils::packageDescription("BATTS")$RemoteSha
  stamp <- system.file("source_commit.txt", package = "BATTS")
  if (is.null(installed) && nzchar(stamp)) installed <- readLines(stamp, warn = FALSE)
  if (!identical(installed, case_study$batts_commit) ||
      !identical(as.character(utils::packageVersion("BATTS")), case_study$batts_version))
    stop("Install ", case_study$batts_repository, "@", case_study$batts_commit,
         " (BATTS ", case_study$batts_version, ") before fitting.")
  invisible(installed)
}

case_study_sources <- function(project_root, script) {
  paths <- file.path(project_root, "scripts", c("lib/case_study_config.R", script))
  setNames(unname(tools::md5sum(paths)), c("lib/case_study_config.R", script))
}

read_case_study_matrix <- function(path, expected_rows) {
  value <- as.matrix(read.csv(path, header = FALSE))
  stopifnot(identical(dim(value), c(as.integer(expected_rows), case_study$dimensions[["taxa"]])),
            all(is.finite(value)), all(value >= 0 & value <= 1),
            max(abs(rowSums(value) - 1)) < 1e-6)
  value
}
