#!/usr/bin/env Rscript
# Full final analysis from six verified composition matrices. Optional workers.
file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
script_file <- gsub("~+~", " ", sub("^--file=", "", file_arg[[1]]), fixed = TRUE)
root <- dirname(dirname(normalizePath(script_file)))
args <- commandArgs(TRUE)
workers <- if (length(args)) as.integer(args[[1]]) else 1L
stopifnot(length(workers) == 1L, !is.na(workers), workers >= 1L)
run <- Sys.getenv("TWO_SAMPLE_RUN_ROOT", file.path(root, "output", "final"))
dir.create(run, recursive = TRUE, showWarnings = FALSE)
run <- normalizePath(run)
Sys.setenv(TWO_SAMPLE_RUN_ROOT = run,
           TWO_SAMPLE_NULL_OUTPUT = file.path(run, "null"),
           TWO_SAMPLE_RATIO_ROOT = file.path(run, "revision"),
           TWO_SAMPLE_FIGURE_DIR = file.path(run, "figures"),
           TWO_SAMPLE_NULL_INTERVALS = file.path(run, "null", "test_pointwise_credible_intervals.csv"))
run_script <- function(name, args = character()) {
  status <- system2(file.path(R.home("bin"), "Rscript"),
                    c(shQuote(file.path(root, "scripts", name)), shQuote(as.character(args))))
  if (status != 0) stop(name, " failed with status ", status)
}
input_dir <- Sys.getenv("TWO_SAMPLE_DATA_DIR", file.path(root, "data"))
python <- Sys.getenv("TWO_SAMPLE_PYTHON", "python3")
status <- system2(python, c(shQuote(file.path(root, "scripts", "prepare_run.py")),
                           "--run-root", shQuote(run), "--data-dir", shQuote(input_dir)))
if (status != 0) stop("Input/source staging failed; no fits started.")
Sys.setenv(TWO_SAMPLE_DATA_DIR = file.path(run, "inputs"))
# A failed rerun cannot leave an old completion marker claiming success.
for (marker in c("ANALYSIS_COMPLETE", "COMPLETE")) {
  path <- file.path(run, marker)
  if (file.exists(path)) unlink(path)
}
run_script("refit_case_study.R", workers)
run_script("run_figure_S12_null_experiment.R")
run_script("precompute_pcoa.R", workers)
run_script("recreate_all_case_study_figures.R")
run_script("summarize_case_study_refit.R")
message("Completed eight final comparisons, the null diagnostic, and seven figures in ", run)
