#!/usr/bin/env Rscript
# Install the pinned BATTS version first; see README.md.
# Optional first argument: independent comparison workers (default 1).
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
root <- dirname(dirname(normalizePath(script_file)))
args <- commandArgs(TRUE)
workers <- if (length(args)) args[[1]] else "1"
run <- file.path(root, "output", "batts_909ea357")
run_script <- function(name, args = character()) {
  status <- system2(file.path(R.home("bin"), "Rscript"),
                    c(shQuote(file.path(root, "scripts", name)), shQuote(args)))
  if (status != 0) stop(name, " failed with status ", status)
}
run_script("refit_case_study.R", workers)
Sys.setenv(TWO_SAMPLE_NULL_OUTPUT = file.path(run, "null"))
run_script("run_figure_S12_null_experiment.R")
Sys.setenv(TWO_SAMPLE_RATIO_ROOT = file.path(run, "revision"),
           TWO_SAMPLE_FIGURE_DIR = file.path(run, "figures"),
           TWO_SAMPLE_NULL_INTERVALS = file.path(run, "null", "test_pointwise_credible_intervals.csv"))
run_script("recreate_all_case_study_figures.R")
run_script("summarize_case_study_refit.R")
message("Completed the BATTS refit, comparison tables, and seven figures in ", run)
